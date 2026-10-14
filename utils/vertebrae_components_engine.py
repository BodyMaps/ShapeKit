"""Reviewed component refinement, extracted from Chandan Yalavarthy's warmup.

Internal labels: 1=L5 through 17=T1, 18=C7 through 24=C1.
All distances are physical millimetres. No file IO or mutable module settings.
Reviewed seeds and boundary splits are inputs, not automatic anatomical truth.
"""
import numpy as np
import cc3d
from scipy import ndimage
from skimage.graph import MCP_Geometric

LABEL_NAMES = ('BG', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11',
               'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2',
               'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1')
THIN_CONNECTION_PENALTY_MM2 = 4.0
MINIMUM_VOLUME_MM3 = 100.0
MINIMUM_CONTACT_MM2 = 20.0

def component_bounds(box, offset):
    lower = np.array([axis.start for axis in box]) + offset
    upper = np.array([axis.stop - 1 for axis in box]) + offset
    return lower, upper

def separation_mm(first, second, spacing):
    first_lower, first_upper = first
    second_lower, second_upper = second

    gaps = np.maximum(
        0,
        np.maximum(
            first_lower - second_upper,
            second_lower - first_upper,
        ),
    )

    return float(np.linalg.norm(gaps * spacing))

def get_foreground_box(labels):
    foreground = labels > 0
    box = []

    for axis in range(3):
        other_axes = tuple(
            value for value in range(3) if value != axis
        )
        occupied = np.flatnonzero(
            np.any(foreground, axis=other_axes)
        )

        if not len(occupied):
            raise ValueError("Empty foreground.")

        box.append(
            slice(int(occupied[0]), int(occupied[-1]) + 1)
        )

    return tuple(box)

def conservative_cleanup(baseline, spacing):
    voxel_volume = float(np.prod(spacing))
    refined = baseline.copy()

    boxes = ndimage.find_objects(baseline, max_label=24)
    changes = []

    for label_id, label_name in enumerate(LABEL_NAMES[1:], 1):
        label_box = boxes[label_id - 1]

        if label_box is None:
            continue

        binary = np.ascontiguousarray(
            baseline[label_box] == label_id
        )

        components = cc3d.connected_components(
            binary,
            connectivity=26,
        )
        statistics = cc3d.statistics(components)
        sizes = statistics["voxel_counts"]

        component_ids = sorted(
            range(1, len(sizes)),
            key=lambda component_id: (
                -int(sizes[component_id]),
                component_id,
            ),
        )

        total = int(sizes[1:].sum())

        if total == 0:
            continue

        primary_id = component_ids[0]
        primary_fraction = int(sizes[primary_id]) / total
        offset = np.array([axis.start for axis in label_box])

        primary_bounds = component_bounds(
            statistics["bounding_boxes"][primary_id],
            offset,
        )

        for rank, component_id in enumerate(component_ids[1:], start=2):
            bounds = component_bounds(
                statistics["bounding_boxes"][component_id],
                offset,
            )

            lower, upper = bounds
            distance = separation_mm(
                primary_bounds,
                bounds,
                spacing,
            )

            voxel_count = int(sizes[component_id])
            fraction = voxel_count / total

            touches_boundary = any(
                lower[axis] == 0
                or upper[axis] == baseline.shape[axis] - 1
                for axis in range(3)
            )

            remove = (
                fraction <= 0.05
                and primary_fraction >= 0.80
                and distance >= 100.0
                and not touches_boundary
            )

            if not remove:
                continue

            refined_view = refined[label_box]
            refined_view[components == component_id] = 0

            changes.append({

                "label_id": label_id,
                "label_name": label_name,
                "component_id": component_id,
                "size_rank": rank,
                "removed_voxels": voxel_count,
                "removed_volume_mm3": voxel_count * voxel_volume,
                "label_fraction": fraction,
                "primary_fraction": primary_fraction,
                "separation_lower_bound_mm": distance,
            })

        del binary, components, statistics

    changed_voxels = int(np.count_nonzero(baseline != refined))
    logged_voxels = sum(row["removed_voxels"] for row in changes)

    if changed_voxels != logged_voxels:
        raise RuntimeError(f"modification count mismatch.")

    if np.any((baseline != refined) & (refined != 0)):
        raise RuntimeError(f"unexpected label reassignment.")

    return refined, changes

def geodesic_assignment(image, markers, mask, spacing):
    # The existing runner supplies negative physical foreground distance.
    distance = np.maximum(-image, 0)

    costs = np.full(image.shape, np.inf, dtype=np.float64)
    costs[mask] = (
        1.0
        + THIN_CONNECTION_PENALTY_MM2
        / (distance[mask] + 0.5) ** 2
    )

    starts = np.argwhere(markers > 0)
    if len(starts) == 0:
        raise ValueError("No seed markers were supplied.")
    if np.any(~mask[tuple(starts.T)]):
        raise ValueError("A marker lies outside the foreground.")

    print(
        "Computing physical minimum-cost paths...",
        flush=True,
    )

    solver = MCP_Geometric(
        costs,
        fully_connected=True,
        sampling=tuple(spacing),
    )
    cumulative_costs, traceback = solver.find_costs(starts)

    del costs, starts, distance

    # Traceback offsets describe the step from predecessor to current voxel.
    # Convert these steps into offsets in a C-order flattened array.
    shape = markers.shape
    strides = np.array(
        [shape[1] * shape[2], shape[2], 1],
        dtype=np.int64,
    )
    offsets = np.asarray(solver.offsets, dtype=np.int64)
    flat_steps = offsets @ strides

    flat_costs = cumulative_costs.ravel()
    flat_traceback = traceback.ravel()
    result = markers.astype(np.int32, copy=True)
    flat_result = result.ravel()

    reachable = np.flatnonzero(np.isfinite(flat_costs))
    order = reachable[
        np.argsort(flat_costs[reachable], kind="stable")
    ]
    del reachable

    print(
        "Propagating seed names along the minimum-cost paths...",
        flush=True,
    )

    for number, index in enumerate(order, start=1):
        if flat_result[index] == 0:
            offset_id = int(flat_traceback[index])
            if offset_id < 0:
                raise RuntimeError("A reachable voxel has no predecessor.")

            predecessor = int(index - flat_steps[offset_id])

            if not 0 <= predecessor < flat_result.size:
                raise RuntimeError("Invalid traceback predecessor.")
            if flat_result[predecessor] == 0:
                raise RuntimeError("Predecessor was not assigned first.")
            if flat_costs[predecessor] >= flat_costs[index]:
                raise RuntimeError("Path costs did not increase.")

            flat_result[index] = flat_result[predecessor]

        if number % 250000 == 0:
            print(
                f"Assigned {number:,}/{len(order):,} reachable voxels",
                flush=True,
            )

    if not np.array_equal(
        result[markers > 0],
        markers[markers > 0],
    ):
        raise RuntimeError("Seed preservation failed.")

    # Regions without any reachable seed remain zero here.
    # The existing runner retains their original prediction labels.
    return result

def split_component(mask, markers, spacing):
    # Distances and paths are restricted to this original component.
    padded = np.pad(mask, 1, constant_values=False)

    padded_distance = ndimage.distance_transform_edt(
        padded,
        sampling=spacing,
    )

    distance = padded_distance[1:-1, 1:-1, 1:-1].copy()
    del padded, padded_distance

    assigned = geodesic_assignment(
        -distance,
        markers=markers,
        mask=mask,
        spacing=spacing,
    )

    if np.any(assigned[~mask] != 0):
        raise RuntimeError("Assignment escaped the component.")

    if np.any(assigned[mask] == 0):
        raise RuntimeError("Some component voxels were not assigned.")

    allowed = set(np.unique(markers)) - {0}
    produced = set(np.unique(assigned[mask]))

    if not produced.issubset(allowed):
        raise RuntimeError("An unsupported label was assigned.")

    return assigned

def reconstruct_seeds(labels, spacing, voxel_volume, rows, erosion_mm):
    seeds = np.zeros(labels.shape, dtype=np.uint8)
    centers = []
    if not np.isfinite(erosion_mm) or erosion_mm <= 0:
        raise ValueError('Positive finite erosion radius required')
    ids = [int(row['component_id']) for row in rows]
    if len(ids) != len(set(ids)) or any(i <= 0 for i in ids):
        raise ValueError('Seed components must be distinct positive IDs')
    expected_names = list(LABEL_NAMES[1:18])

    if [row["proposed_label"] for row in rows] != expected_names:
        raise ValueError("Unexpected proposed seed sequence.")

    print("Reconstructing proposed seed supports...", flush=True)

    padded = np.pad(labels > 0, 1, constant_values=False)

    distance = ndimage.distance_transform_edt(
        padded,
        sampling=spacing,
    )

    components = cc3d.connected_components(
        distance[1:-1, 1:-1, 1:-1] > erosion_mm,
        connectivity=26,
    )

    del padded, distance

    for label_id, row in enumerate(rows, start=1):
        mask = components == int(row["component_id"])
        size = int(np.count_nonzero(mask))

        if size == 0:
            raise ValueError("A proposed seed is missing.")

        if not np.isclose(
            size * voxel_volume,
            float(row["core_volume_mm3"]),
            rtol=0,
            atol=1e-3,
        ):
            raise ValueError("A proposed seed's volume has changed.")

        counts = np.bincount(labels[mask], minlength=25)
        dominant = int(np.argmax(counts[1:]) + 1)

        if (
            LABEL_NAMES[dominant]
            != row["dominant_original_label"]
        ):
            raise ValueError("A seed's original composition changed.")

        seeds[mask] = label_id

        coordinates = np.argwhere(mask)
        centers.append(coordinates.mean(axis=0))

    return seeds


def correct_components(before, seeds, spacing):
    after = before.copy()
    changes = []
    figures = []

    for old_id in range(1, 18):
        components = cc3d.connected_components(
            before == old_id,
            connectivity=26,
        )

        sizes = np.bincount(components.ravel())
        boxes = ndimage.find_objects(components)

        component_ids = sorted(
            range(1, len(sizes)),
            key=lambda value: int(sizes[value]),
            reverse=True,
        )

        for rank, component_id in enumerate(component_ids, start=1):
            box = boxes[component_id - 1]
            mask = components[box] == component_id

            local_seeds = seeds[box]
            overlap = np.bincount(
                local_seeds[mask],
                minlength=25,
            )

            touched = [
                value
                for value in range(1, 18)
                if overlap[value] > 0
            ]

            if not touched:
                continue

            if len(touched) == 1 and touched[0] == old_id:
                continue

            old_name = LABEL_NAMES[old_id]
            target_view = after[box]

            if len(touched) == 1:
                target_view[mask] = touched[0]
                method = "intact_relabeling"

            else:
                names = [
                    LABEL_NAMES[value]
                    for value in touched
                ]

                print(
                    f"Splitting {old_name} rank {rank} "
                    f"between {names}...",
                    flush=True,
                )

                markers = np.where(
                    mask,
                    local_seeds,
                    0,
                ).astype(np.int32)

                assigned = split_component(
                    mask,
                    markers,
                    spacing,
                )

                target_view[mask] = assigned[mask]
                method = "component_local_geodesic"

            modified = int(np.count_nonzero(
                target_view[mask] != old_id
            ))

            if modified == 0:
                continue

            changes.append({
                "original_label": old_name,
                "component_rank": rank,
                "component_voxels": int(sizes[component_id]),
                "method": method,
                "allowed_names": [
                    LABEL_NAMES[value]
                    for value in touched
                ],
                "modified_voxels": modified,
            })

        del components

    if not np.array_equal(before > 0, after > 0):
        raise RuntimeError("Foreground preservation failed.")

    cervical = before >= 18

    if not np.array_equal(before[cervical], after[cervical]):
        raise RuntimeError("Cervical preservation failed.")

    if not np.array_equal(after[seeds > 0], seeds[seeds > 0]):
        raise RuntimeError("Proposed seed preservation failed.")

    if not np.array_equal(np.unique(before), np.unique(after)):
        raise RuntimeError("The set of labels changed.")

    modified = int(np.count_nonzero(before != after))

    if modified != sum(row["modified_voxels"] for row in changes):
        raise RuntimeError("Modification accounting failed.")

    return after, changes

def correct_contacts(original, before, seeds, spacing):
    after = before.copy()
    voxel_volume = float(np.prod(spacing))
    seed_mask = seeds > 0

    if not np.array_equal(before[seed_mask], seeds[seed_mask]):
        raise RuntimeError("Input candidate disagrees with proposed seeds.")

    # Support contains current labels only inside original components
    # that overlap at least one seed.
    support = np.zeros(original.shape, dtype=np.uint8)

    # Each unseeded component receives a separate identifier.
    fragment_ids = np.zeros(original.shape, dtype=np.int32)
    fragments = []

    print("Identifying seed-supported and unseeded components...", flush=True)

    for label in range(1, 18):
        components = cc3d.connected_components(
            original == label,
            connectivity=26,
        )

        counts = np.bincount(components.ravel())

        ordered_ids = sorted(
            range(1, len(counts)),
            key=lambda component_id: (
                -int(counts[component_id]),
                component_id,
            ),
        )

        seeded_ids = np.unique(components[seed_mask])
        seeded_ids = seeded_ids[seeded_ids != 0]

        known_lookup = np.zeros(len(counts), dtype=bool)
        known_lookup[seeded_ids] = True

        known_mask = known_lookup[components]
        support[known_mask] = before[known_mask]

        fragment_lookup = np.zeros(len(counts), dtype=np.int32)

        for rank, component_id in enumerate(ordered_ids, start=1):
            if known_lookup[component_id]:
                continue

            fragment_id = len(fragments) + 1
            fragment_lookup[component_id] = fragment_id

            fragments.append({
                "fragment_id": fragment_id,
                "original_label_id": label,
                "original_label": LABEL_NAMES[label],
                "component_rank": rank,
                "voxels": int(counts[component_id]),
                "volume_mm3": (
                    float(counts[component_id]) * voxel_volume
                ),
            })

        fragment_ids += fragment_lookup[components]

    del components, known_mask, seeds, seed_mask

    print("Measuring physical voxel-face contacts...", flush=True)

    contacts = np.zeros(
        (len(fragments) + 1, 25),
        dtype=np.float64,
    )

    # Only shared faces count. Edge and corner contacts do not.
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3

        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)

        lower = tuple(lower)
        upper = tuple(upper)

        face_area = float(np.prod([
            spacing[other_axis]
            for other_axis in range(3)
            if other_axis != axis
        ]))

        for fragment_slice, support_slice in (
            (lower, upper),
            (upper, lower),
        ):
            local_fragments = fragment_ids[fragment_slice]
            local_support = support[support_slice]

            valid = (
                (local_fragments > 0)
                & (local_support > 0)
            )

            codes = (
                local_fragments[valid].astype(np.int64) * 25
                + local_support[valid]
            )

            face_counts = np.bincount(
                codes,
                minlength=contacts.size,
            ).reshape(contacts.shape)

            contacts += face_counts * face_area

    # Contacts are measured once against fixed support.
    # Newly changed fragments never become support for another change.
    boxes = ndimage.find_objects(fragment_ids)
    changes = []
    review_jobs = []

    for fragment in fragments:
        fragment_id = fragment["fragment_id"]
        old_label = fragment["original_label_id"]

        areas = contacts[fragment_id]
        touching_labels = np.flatnonzero(areas > 0)

        fragment["contact_area_mm2"] = {
            LABEL_NAMES[int(label)]: float(areas[label])
            for label in touching_labels
        }

        if fragment["volume_mm3"] < MINIMUM_VOLUME_MM3:
            fragment["decision"] = "preserved_small_fragment"
            continue

        if len(touching_labels) != 1:
            fragment["decision"] = "preserved_ambiguous_or_no_contact"
            continue

        new_label = int(touching_labels[0])

        if new_label == old_label:
            fragment["decision"] = "preserved_same_name"
            continue

        if abs(new_label - old_label) != 1:
            fragment["decision"] = "preserved_nonadjacent_name"
            continue

        if areas[new_label] < MINIMUM_CONTACT_MM2:
            fragment["decision"] = "preserved_insufficient_contact"
            continue

        box = boxes[fragment_id - 1]
        mask = fragment_ids[box] == fragment_id

        if not np.all(before[box][mask] == old_label):
            raise RuntimeError(
                "An unseeded fragment was already modified."
            )

        after[box][mask] = new_label

        fragment["decision"] = "experimental_relabel"
        fragment["new_label"] = LABEL_NAMES[new_label]
        changes.append(fragment.copy())

    if not changes:
        print("No fragments met the experimental rules.")
        return after, changes

    if not np.array_equal(before > 0, after > 0):
        raise RuntimeError("Foreground changed.")

    if not np.array_equal(
        before[original >= 18],
        after[original >= 18],
    ):
        raise RuntimeError("Cervical labels changed.")

    if not np.array_equal(
        before[support > 0],
        after[support > 0],
    ):
        raise RuntimeError("Seed-supported regions changed.")

    changed_voxels = int(np.count_nonzero(before != after))

    if changed_voxels != sum(item["voxels"] for item in changes):
        raise RuntimeError("Change-count verification failed.")

    return after, changes

def split_reviewed_fragment(original, before, spacing, selection):
    """Split only a specifically reviewed component; do not guess its identity."""
    label = int(selection['original_label'])
    rank = int(selection['rank'])
    targets = [int(value) for value in selection['targets']]
    if not 1 <= label <= 17 or len(targets) != 2 or len(set(targets)) != 2:
        raise ValueError('One thoracolumbar component and two distinct targets required')
    if any(not 1 <= target <= 17 for target in targets) or abs(targets[0] - targets[1]) != 1:
        raise ValueError('Targets must be adjacent thoracolumbar levels')
    components = cc3d.connected_components(original == label, connectivity=26)
    counts = np.bincount(components.ravel())
    ordered = sorted(range(1, len(counts)), key=lambda i: (-int(counts[i]), i))
    if not 1 <= rank <= len(ordered):
        raise ValueError('Reviewed fragment rank is missing')
    component_id = ordered[rank - 1]
    if int(counts[component_id]) != int(selection['voxels']):
        raise ValueError('Reviewed fragment size changed')
    box = ndimage.find_objects(components)[component_id - 1]
    margin = np.ceil(30.0 / spacing).astype(int)
    crop = tuple(slice(max(0, box[a].start - int(margin[a])),
                       min(original.shape[a], box[a].stop + int(margin[a]))) for a in range(3))
    fragment = components[crop] == component_id
    local = before[crop]
    if not np.all(local[fragment] == label):
        raise ValueError('Reviewed fragment was already relabeled')
    faces = ndimage.generate_binary_structure(3, 1)
    contacts = [fragment & ndimage.binary_dilation((local == target) & ~fragment, structure=faces)
                for target in targets]
    conflict = contacts[0] & contacts[1]
    markers = np.zeros(fragment.shape, dtype=np.int32)
    for target, contact in zip(targets, contacts):
        markers[contact & ~conflict] = target
        if not np.any(markers == target):
            raise ValueError('Both targets require unambiguous contact markers')
    assigned = split_component(fragment, markers, spacing)
    after = before.copy()
    after[crop][fragment] = assigned[fragment]
    return after, {'original_label': label, 'rank': rank,
                   'modified_voxels': int(np.count_nonzero(local[fragment] != assigned[fragment])),
                   'assignments': {LABEL_NAMES[t]: int(np.count_nonzero(assigned[fragment] == t)) for t in targets}}


def refine(baseline, spacing, voxel_volume, plan):
    """Run cleanup, then only the corrections explicitly enabled by a review plan."""
    original, removed = conservative_cleanup(baseline, spacing)
    report = {'removed_components': removed, 'component_changes': [],
              'contact_changes': [], 'boundary_splits': [], 'mode': plan['mode']}
    if plan['mode'] == 'cleanup_only':
        report['modified_voxels'] = int(np.count_nonzero(original != baseline))
        return original, report
    if plan['mode'] != 'reviewed':
        raise ValueError('Unknown review mode')
    box = get_foreground_box(original)
    cropped = original[box]
    seeds = reconstruct_seeds(cropped, spacing, voxel_volume,
                              plan['seeds'], float(plan['erosion_mm']))
    candidate, report['component_changes'] = correct_components(cropped, seeds, spacing)
    candidate, report['contact_changes'] = correct_contacts(cropped, candidate, seeds, spacing)
    after = original.copy()
    after[box] = candidate
    for selection in plan.get('boundary_splits', []):
        after, change = split_reviewed_fragment(original, after, spacing, selection)
        report['boundary_splits'].append(change)
    if not np.array_equal(after > 0, original > 0):
        raise RuntimeError('Correction changed cleaned foreground')
    if not np.array_equal(after[original >= 18], original[original >= 18]):
        raise RuntimeError('Correction changed cervical labels')
    if not np.array_equal(after[box][seeds > 0], seeds[seeds > 0]):
        raise RuntimeError('Correction changed reviewed seeds')
    if not np.array_equal(np.unique(after), np.unique(original)):
        raise RuntimeError('Correction removed or added a label')
    report['modified_voxels'] = int(np.count_nonzero(after != baseline))
    return after, report
