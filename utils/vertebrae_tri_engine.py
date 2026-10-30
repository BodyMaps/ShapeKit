"""Mask-only vertebral post-processing (internal labels: 1=L5, ..., 24=C1).

Three stages: global anatomical numbering, connected-component cleanup,
and local plane-based separation. The CT is not used by this algorithm.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

NUM_VERTEBRAE = 24
MIN_DISTANCE_MM = 10.0
MAX_DISTANCE_MM = 60.0
DEFAULT_DISTANCE_MM = 25.0
SMALL_COMPONENT_THRESHOLD = 50
SPACING_ERROR_WEIGHT = 2.0
LATERAL_ERROR_WEIGHT = 0.75
KEEP_FACTOR = 1.5
REMOVE_FACTOR = 4.0
FAR_FROM_ALL_FACTOR = 3.0
PLANE_MARGIN_MM = 5.0
CENTROIDS_MAX_DISTANCE_ERROR_FACTOR = 0.45
PLANE_MIN_OVERFLOW_RATIO = 0.10
PLANE_MAX_OVERFLOW_RATIO = 0.85
PLANE_MIN_OVERFLOW_VOXELS = 300
NUMBERING_BIN_MM = 2.0
NUMBERING_MIN_ANCHOR_PURITY = 0.70
NUMBERING_MAX_SHIFT = 2
POSTERIOR_START_MM = 8.0
POSTERIOR_BLEND_MM = 18.0
POSTERIOR_SHIFT_MM = 10.0

def voxel_to_mm(voxel_coords, affine):
    coords = np.asarray(voxel_coords)
    return coords @ affine[:3, :3].T + affine[:3, 3]

def find_connected_components(mask, affine):
    """Connected components for each label, computed on cropped ROIs.

    The component identifiers follow scipy.ndimage.label scan order,
    matching the original full-volume implementation.
    """
    components_by_label = {}
    structure = ndi.generate_binary_structure(3, 3)
    rois = ndi.find_objects(mask, max_label=NUM_VERTEBRAE)
    for label_id in range(1, NUM_VERTEBRAE + 1):
        roi = rois[label_id - 1]
        if roi is None:
            continue
        binary = mask[roi] == label_id
        cc_map, n = ndi.label(binary, structure=structure)
        # Compute centroids from foreground coordinates only. Calling
        # ndi.center_of_mass on a large sparse 3D bounding box can be slow.
        fg = np.nonzero(cc_map)
        ids = cc_map[fg]
        sizes = np.bincount(ids, minlength=n + 1)
        sums = [np.bincount(ids, weights=axis, minlength=n + 1)
                for axis in fg]
        boxes = ndi.find_objects(cc_map)
        offset = np.asarray([s.start for s in roi])
        candidates = []
        for cid in range(1, n + 1):
            center_voxel = np.asarray([axis_sum[cid] / sizes[cid] for axis_sum in sums]) + offset
            local_box = boxes[cid - 1]
            box = tuple(slice(s.start + off, s.stop + off) for s, off in zip(local_box, offset))
            candidates.append({
                'id': cid,
                'label_id': label_id,
                'num_voxels': int(sizes[cid]),
                'centroid_voxel': center_voxel,
                'centroid_mm': voxel_to_mm(center_voxel, affine),
                'bounding_box': box,
            })
        candidates.sort(key=lambda item: item['num_voxels'], reverse=True)
        components_by_label[label_id] = candidates
    return components_by_label

def anatomical_cost(vertebra1, vertebra2, expected_spacing):
    # cost function that penalizes structures that are not anatomically consistent
    levels = vertebra2["label_id"] - vertebra1["label_id"]
    center1 = vertebra1["centroid_mm"]
    center2 = vertebra2["centroid_mm"]
    z_distance = center2[2] - center1[2]
    # Vertebrae must respect the anatomical order
    if z_distance <= levels:
        return np.inf

    expected_distance = expected_spacing * levels
    ratio = z_distance / expected_distance
    spacing_error = max(0, 0.6 - ratio, ratio - 1.8)
    spacing_cost = SPACING_ERROR_WEIGHT * spacing_error ** 2
    lateral_distance = np.linalg.norm(
        np.asarray(center2[:2]) - np.asarray(center1[:2])
    )
    lateral_error = max(0, lateral_distance / levels / 20.0 - 1.0)
    lateral_cost = LATERAL_ERROR_WEIGHT * lateral_error ** 2
    return spacing_cost + lateral_cost

def find_vertebrae(connected_components, max_candidates=5, skip_penalty=5.0):
    labels = sorted(connected_components.keys())
    # Biggest connected component for each vertebra label
    initial_vertebrae = {}
    for label_id in labels:
        components = connected_components[label_id]
        if len(components) == 0:
            continue
        initial_vertebrae[label_id] = max(
            components,
            key=lambda component: component["num_voxels"]
        )
    labels = sorted(initial_vertebrae.keys())

    # Estimate expected spacing between vertebrae
    distances = []
    for label1, label2 in zip(labels[:-1], labels[1:]):
        if label2 != label1 + 1:
            continue
        z1 = initial_vertebrae[label1]["centroid_mm"][2]
        z2 = initial_vertebrae[label2]["centroid_mm"][2]
        distance = z2 - z1
        if MIN_DISTANCE_MM <= distance <= MAX_DISTANCE_MM:
            distances.append(distance)
    expected_spacing = np.median(distances) if distances else DEFAULT_DISTANCE_MM

    # Optimization of a cost function to identify the best components for each class
    # The cost function: Lvolume + Lanatomy + Lskip
    # Lvolume prioritizes larger components, Lskip penalizes skipping vertebrae and Lanatomy comes from the anatomical_cost function
    states = {None: (0.0, [])}
    for label_id in labels:
        components = sorted(
            connected_components[label_id],
            key=lambda component: component["num_voxels"],
            reverse=True
        )[:max_candidates]
        largest_volume = components[0]["num_voxels"]
        new_states = {}
        for last_key, (cost, selected) in states.items():
            # Skip penalty
            new_cost = cost + skip_penalty
            if last_key not in new_states or new_cost < new_states[last_key][0]:
                new_states[last_key] = (new_cost, selected)

            for component in components:
                volume = component["num_voxels"]
                if component["id"] != components[0]["id"]:
                    if volume < SMALL_COMPONENT_THRESHOLD or volume < largest_volume * 0.02:
                        continue

                # Larger components
                volume_cost = np.log(largest_volume / volume)
                new_cost = cost + volume_cost
                # Anatomical consistency
                if len(selected) > 0:
                    new_cost += anatomical_cost(
                        selected[-1],
                        component,
                        expected_spacing
                    )
                if not np.isfinite(new_cost):
                    continue
                key = (label_id, component["id"])
                if key not in new_states or new_cost < new_states[key][0]:
                    new_states[key] = (
                        new_cost,
                        selected + [component]
                    )
        states = new_states

    _, vertebrae = min(
        states.values(),
        key=lambda state: state[0]
    )
    return vertebrae

def components_distance(components1, components2, norm_factor=1.0):
    return np.linalg.norm(np.asarray(components1["centroid_mm"]) - np.asarray(components2["centroid_mm"])) / norm_factor

def get_component_extent(component, affine):
    bbox = component["bounding_box"]
    corners = np.array([
        [i, j, k]
        for i in (bbox[0].start, bbox[0].stop - 1)
        for j in (bbox[1].start, bbox[1].stop - 1)
        for k in (bbox[2].start, bbox[2].stop - 1)
    ])
    z_corners = voxel_to_mm(corners, affine)[:, 2]
    return np.ptp(z_corners)

def correct_global_numbering(mask, affine):
    # Correct wrong ordered vertebrae labels using physical centers and anatomical order.
    binary = mask > 0
    roi = ndi.find_objects(binary.astype(np.uint8))[0] if binary.any() else None
    if roi is None:
        return mask
    cropped = mask[roi]
    core = ndi.binary_erosion(cropped > 0, iterations=2)
    points = np.argwhere(core)
    if len(points) < 100:
        return mask
    offset = np.array([s.start for s in roi])
    z_mm = (points + offset) @ affine[2, :3] + affine[2, 3]
    z_min = float(z_mm.min())
    # construction of a function that for each z, tells us how much segmented bone structure is present
    bins = np.floor((z_mm - z_min) / NUMBERING_BIN_MM).astype(int)
    profile = np.bincount(bins).astype(float)
    smooth = ndi.gaussian_filter1d(profile, sigma=1.5)
    positive = smooth[smooth > 1]
    if len(positive) == 0:
        return mask
    reference_height = float(np.median(positive))
    # max of the bone density structure function
    peaks, _ = find_peaks(smooth, distance=int(14 / NUMBERING_BIN_MM), prominence=max(3.0, 0.20 * reference_height), height=max(4.0, 0.55 * reference_height))
    peak_z = z_min + (peaks + 0.5) * NUMBERING_BIN_MM
    if len(peaks) < 2:
        return mask
    # estimate distance between vertebrae
    gaps = np.diff(peak_z)
    valid_gaps = gaps[(gaps >= 12) & (gaps <= 55)]
    spacing = float(np.median(valid_gaps)) if len(valid_gaps) else DEFAULT_DISTANCE_MM
    local_spacing = []
    for i, gap in enumerate(gaps):
        neighbors = np.concatenate((gaps[max(0, i - 2):i], gaps[i + 1:i + 3]))
        closest = np.sort(neighbors)[:2]
        local_spacing.append(float(np.median(closest)) if len(closest) else float(gap))

    # predicted-label distribution around each physical center
    labels_at_points = cropped[tuple(points.T)]
    evidence = []
    dominant = []
    purity = []
    center_xy = []
    for z in peak_z:
        nearby = np.abs(z_mm - z) <= min(8.0, 0.30 * spacing)
        if not np.any(nearby):
            return mask
        counts = np.bincount(labels_at_points[nearby], minlength=NUM_VERTEBRAE + 1)
        counts[0] = 0
        fraction = counts / max(int(counts.sum()), 1)
        evidence.append(fraction)
        dominant.append(int(np.argmax(counts)))
        purity.append(float(fraction[dominant[-1]]))
        pts = (points[nearby] + offset) @ affine[:2, :3].T + affine[:2, 3]
        center_xy.append(np.median(pts, axis=0))

    # optimize function L = Lmi + Lgeo + Lskip where:
    # 1. Lmi is less when the label is higly rappresented around a center
    # 2. Lgeo is high when the observed physical distance is not compatible with the number of anatomical levels between two centers
    # 3. Lskip penalize the solution when there are missing levels
    n = len(peaks)
    costs = np.full((n, NUM_VERTEBRAE + 1), np.inf)
    previous = np.zeros((n, NUM_VERTEBRAE + 1), dtype=np.int16)
    mismatch_weight = 2.0
    geometry_weight = 8.0
    missing_level_penalty = 0.20
    for label_id in range(1, NUM_VERTEBRAE + 1):
        costs[0, label_id] = mismatch_weight * (1.0 - evidence[0][label_id])
    for i in range(1, n):
        gap_mm = float(peak_z[i] - peak_z[i - 1])
        expected_gap = local_spacing[i - 1]
        for label_id in range(1, NUM_VERTEBRAE + 1):
            mismatch = mismatch_weight * (1.0 - evidence[i][label_id])
            for old_label in range(1, label_id):
                levels = label_id - old_label
                per_level = gap_mm / levels
                ratio = per_level / expected_gap
                if not 0.55 <= ratio <= 1.80:
                    continue
                geometry = geometry_weight * np.log(ratio) ** 2
                skipping = missing_level_penalty * (levels - 1)
                new_cost = costs[i - 1, old_label] + mismatch + geometry + skipping
                if new_cost < costs[i, label_id]:
                    costs[i, label_id] = new_cost
                    previous[i, label_id] = old_label
    if not np.isfinite(costs[-1, 1:]).any():
        return mask
    label_id = int(np.argmin(costs[-1, 1:]) + 1)
    sequence = []
    for i in range(n - 1, -1, -1):
        sequence.append((i, label_id))
        if i > 0:
            label_id = int(previous[i, label_id])
    sequence.reverse()

    reliable = [pos for pos, (i, label_id) in enumerate(sequence) if dominant[i] == label_id and purity[i] >= NUMBERING_MIN_ANCHOR_PURITY]
    if len(reliable) < 2:
        return mask
    corrected = mask.copy()
    active_points = np.argwhere(cropped > 0)
    source_labels = cropped[tuple(active_points.T)].astype(np.int16)
    world = (active_points + offset) @ affine[:3, :3].T + affine[:3, 3]
    corrected_z = world[:, 2].copy()
    nearest = np.searchsorted(peak_z, world[:, 2])
    nearest = np.clip(nearest, 0, len(peak_z) - 1)
    previous_peak = np.clip(nearest - 1, 0, len(peak_z) - 1)
    use_previous = np.abs(world[:, 2] - peak_z[previous_peak]) < np.abs(world[:, 2] - peak_z[nearest])
    nearest[use_previous] = previous_peak[use_previous]
    spine_y = np.asarray(center_xy)[nearest, 1]
    posterior_depth = spine_y - world[:, 1]
    posterior_weight = np.clip((posterior_depth - POSTERIOR_START_MM) / POSTERIOR_BLEND_MM, 0.0, 1.0)
    peak_spacing = np.full(len(peak_z), spacing)
    for i in range(len(peak_z)):
        adjacent_gaps = []
        if i > 0:
            adjacent_gaps.append(gaps[i - 1])
        if i < len(gaps):
            adjacent_gaps.append(gaps[i])
        if adjacent_gaps:
            peak_spacing[i] = min(adjacent_gaps)
    shift = np.minimum(POSTERIOR_SHIFT_MM, 0.35 * peak_spacing[nearest])
    corrected_z += shift * posterior_weight
    def boundary_between(a, b):
        center = (peak_z[a] + peak_z[b]) / 2
        left = int((peak_z[a] + 0.30 * (peak_z[b] - peak_z[a]) - z_min) / NUMBERING_BIN_MM)
        right = int((peak_z[a] + 0.70 * (peak_z[b] - peak_z[a]) - z_min) / NUMBERING_BIN_MM)
        left = max(left, 0)
        right = min(right, len(smooth) - 1)
        if right <= left:
            return center, False
        idx = left + int(np.argmin(smooth[left:right + 1]))
        valley_ratio = smooth[idx] / max(min(smooth[peaks[a]], smooth[peaks[b]]), 1.0)
        return z_min + (idx + 0.5) * NUMBERING_BIN_MM, bool(valley_ratio < 0.85)

    # search for wrong labels between consecutive reliable landmarks and reassign labels
    for start, stop in zip(reliable[:-1], reliable[1:]):
        interior = sequence[start + 1:stop]
        if not interior:
            continue
        if not any(dominant[i] != level and purity[i] >= 0.45 for i, level in interior):
            continue
        block = sequence[start:stop + 1]
        corrected_levels = []
        for k in range(1, len(block) - 1):
            peak_index, target_label = block[k]
            prev_index, prev_level = block[k - 1]
            next_index, next_level = block[k + 1]
            if prev_level != target_label - 1 or next_level != target_label + 1:
                continue
            lower_gap = peak_z[peak_index] - peak_z[prev_index]
            upper_gap = peak_z[next_index] - peak_z[peak_index]
            if not (0.5 * local_spacing[prev_index] <= lower_gap <= 1.8 * local_spacing[prev_index] and 0.5 * local_spacing[peak_index] <= upper_gap <= 1.8 * local_spacing[peak_index]):
                continue
            if dominant[peak_index] == target_label and purity[peak_index] >= 0.90:
                continue
            z_low, lower_valley = boundary_between(prev_index, peak_index)
            z_high, upper_valley = boundary_between(peak_index, next_index)
            if not (lower_valley and upper_valley):
                continue
            lateral = np.linalg.norm(world[:, :2] - center_xy[peak_index], axis=1)
            indices = np.where((corrected_z >= z_low) & (corrected_z < z_high) & (lateral <= 85.0) & (np.abs(source_labels - target_label) <= NUMBERING_MAX_SHIFT) & (source_labels != target_label))[0]
            if len(indices) < 20:
                continue
            corrected_levels.append(target_label)
            local = active_points[indices]
            corrected_roi = corrected[roi]
            corrected_roi[tuple(local.T)] = target_label
    return corrected

def refine_vertebrae(mask, affine, return_stages=False):
    """Refine one labeled vertebra volume, preserving its voxel grid.

    Parameters
    ----------
    mask : 3D integer ndarray, values 0..24 (1=L5, 24=C1).
    affine : 4x4 NIfTI voxel-to-RAS matrix.
    return_stages : bool
        Return separate numbering and component intermediate masks for QA.
    """
    mask = np.asarray(mask)
    affine = np.asarray(affine, dtype=float)
    if mask.ndim != 3 or any(s == 0 for s in mask.shape):
        raise ValueError('Expected a nonempty 3D segmentation')
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Segmentation must contain integer labels')
    if np.any(mask > NUM_VERTEBRAE) or np.any(mask < 0):
        raise ValueError('Expected internal labels between 0 and 24')
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Expected a finite 4x4 affine')
    if not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError('Invalid NIfTI affine geometry')
    # The anatomical world directions assumed are RAS: +Z is superior, -Y posterior.
    # The algorithm uses world coordinates, not a fixed voxel axis.
    original = mask.astype(np.uint8, copy=False)
    if not np.any(original):
        empty = original.copy()
        return (empty, {'numbering': empty.copy(), 'components': empty.copy()}) if return_stages else empty

    # Stage 1: global numbering, unchanged from the standalone algorithm.
    mask = correct_global_numbering(original, affine)
    numbering_mask = mask.copy() if return_stages else None

    # Stage 2: connected-component cleanup.
    connected_components = find_connected_components(mask, affine)
    vertebrae = find_vertebrae(connected_components)
    if len(vertebrae) < 2:
        # The standalone code requires at least two selected anchors to estimate
        # spacing; abstain rather than fail on partial-field-of-view scans.
        result = mask.copy()
        return (result, {'numbering': numbering_mask, 'components': result.copy()}) if return_stages else result

    distances = []
    for i in range(1, len(vertebrae)):
        v1 = vertebrae[i - 1]
        v2 = vertebrae[i]
        levels = v2["label_id"] - v1["label_id"]
        distance = components_distance(v1, v2)
        distances.append(distance / levels)
    if not distances:
        result = mask.copy()
        return (result, {'numbering': numbering_mask, 'components': result.copy()}) if return_stages else result
    median_vertebrae_distance = np.median(distances)

    # -------------------------------------
    # Modify components:
    # 1. Remove components with high distance from all other vertebrae
    # 2. Keep vertebrae and other components with same label and small distance
    # 3. Reassign components with high distance from their vertebrae to the closest anatomically adjacent vertebrae
    vertebrae_dict = {v["label_id"]: v for v in vertebrae}
    refined_mask = mask.copy()
    structure = ndi.generate_binary_structure(3, 3)
    label_slices = ndi.find_objects(mask, max_label=NUM_VERTEBRAE)
    for label_id, components in connected_components.items():
        if label_id not in vertebrae_dict:
            continue
        my_vertebra = vertebrae_dict[label_id]
        roi = label_slices[label_id - 1]
        connected_map, _ = ndi.label(
            mask[roi] == label_id,
            structure=structure
        )
        refined_roi = refined_mask[roi]
        for component in components:
            if component["id"] == my_vertebra["id"]:
                continue
            my_distance = components_distance(component, my_vertebra)
            if my_distance <= median_vertebrae_distance * KEEP_FACTOR:
                continue

            new_label = label_id
            # neighboring vertebrae labels
            neighbors = [n for n in (label_id - 1, label_id + 1) if n in vertebrae_dict]
            if neighbors:
                closest_neighbor = min(neighbors, key=lambda n: components_distance(component, vertebrae_dict[n]))
                neighbor_distance = components_distance(component, vertebrae_dict[closest_neighbor])
                if neighbor_distance <= median_vertebrae_distance * (KEEP_FACTOR+0.5):
                    new_label = closest_neighbor

            if new_label == label_id:
                closest_vertebra = min(vertebrae, key=lambda v: components_distance(component, v))
                closest_distance = components_distance(component, closest_vertebra)
                if closest_distance <= median_vertebrae_distance * (KEEP_FACTOR+0.5):
                    new_label = closest_vertebra["label_id"]
                else:
                    new_label = 0

            if new_label != label_id:
                refined_roi[connected_map == component["id"]] = new_label

    component_mask = refined_mask.copy() if return_stages else None

    # Stage 3: construct local anatomical separation planes.
    clean_mask = refined_mask.copy()
    updated_components = find_connected_components(clean_mask, affine)
    updated_vertebrae = find_vertebrae(updated_components)
    centers = {v["label_id"]: np.asarray(v["centroid_mm"], dtype=float) for v in updated_vertebrae}
    original_centers = {label_id: center.copy() for label_id, center in centers.items()}
    distances = []
    for label_id in range(1, NUM_VERTEBRAE):
        if label_id in centers and label_id + 1 in centers:
            distance = centers[label_id + 1][2] - centers[label_id][2]
            if MIN_DISTANCE_MM <= distance <= MAX_DISTANCE_MM:
                distances.append(distance)
    median_distance = float(np.median(distances)) if distances else DEFAULT_DISTANCE_MM
    vertebrae_extents = {}
    for vertebra in updated_vertebrae:
        extent = get_component_extent(vertebra, affine)
        vertebrae_extents[vertebra["label_id"]] = extent
    median_extent = np.median(list(vertebrae_extents.values())) if vertebrae_extents else 0.0

    overflowing_vertebrae = set()
    for label_id, extent in vertebrae_extents.items():
        position_error = 0.0
        if label_id - 1 in centers and label_id + 1 in centers:
            expected = (centers[label_id - 1] + centers[label_id + 1]) / 2
            position_error = np.linalg.norm(centers[label_id] - expected)
        # too big labels or labels with high center position error
        if extent > 1.5 * median_extent or (position_error > CENTROIDS_MAX_DISTANCE_ERROR_FACTOR * median_distance and extent > 1.15 * median_extent):
            overflowing_vertebrae.add(label_id)

    correct_vertebrae = sorted(label_id for label_id in centers if label_id not in overflowing_vertebrae)
    for label_id in range(1, NUM_VERTEBRAE + 1):
        if label_id in centers and label_id not in overflowing_vertebrae:
            continue
        lower = [v for v in correct_vertebrae if v < label_id]
        upper = [v for v in correct_vertebrae if v > label_id]
        if not lower or not upper:
            # Not reliable coordinate for plane computation
            centers.pop(label_id, None)
            continue
        lower = lower[-1]
        upper = upper[0]
        levels = upper - lower
        z_distance = original_centers[upper][2] - original_centers[lower][2]
        if 0.5 * median_distance * levels <= z_distance <= 2.0 * median_distance * levels:
            # Interpolate the center between the two reliable vertebrae
            fraction = (label_id - lower) / levels
            centers[label_id] = original_centers[lower] * (1 - fraction) + original_centers[upper] * fraction
        else:
            centers.pop(label_id, None)

    label_slices = ndi.find_objects(clean_mask, max_label=NUM_VERTEBRAE)
    plane_changes = 0
    accepted_pairs = 0
    for label_id in range(1, NUM_VERTEBRAE + 1):
        if label_id not in centers:
            continue
        roi = label_slices[label_id - 1]
        if roi is None:
            continue
        voxels_local = np.argwhere(clean_mask[roi] == label_id)
        if len(voxels_local) == 0:
            continue
        offset = np.array([s.start for s in roi])
        voxels_mm = voxel_to_mm(voxels_local + offset, affine)
        new_labels = np.full(len(voxels_local), label_id, dtype=np.uint8)
        best_score = np.zeros(len(voxels_local), dtype=float)
        for neighbor in (label_id - 1, label_id + 1):
            if neighbor not in centers:
                continue
            direction = centers[neighbor] - centers[label_id]
            distance = np.linalg.norm(direction)
            z_distance = abs(direction[2])
            if distance == 0 or not (0.4 * median_distance <= z_distance <= 2.5 * median_distance):
                continue
            midpoint = (centers[label_id] + centers[neighbor]) / 2
            normal = direction / distance
            # distance from voxel to plane is computed as (voxel position - midpoint of the plane) * normal of the plane
            signed_distance = (voxels_mm - midpoint) @ normal
            # take in account also the spinous processes behind the vertebrae bone
            posterior_axis = np.array([0.0, -1.0, 0.0])
            posterior_axis -= np.dot(posterior_axis, normal) * normal
            posterior_axis /= max(np.linalg.norm(posterior_axis), 1e-8)
            posterior_depth = (voxels_mm - midpoint) @ posterior_axis
            posterior_weight = np.clip((posterior_depth - POSTERIOR_START_MM) / POSTERIOR_BLEND_MM, 0.0, 1.0)
            shift = min(POSTERIOR_SHIFT_MM, 0.35 * distance)
            signed_distance += np.sign(neighbor - label_id) * shift * posterior_weight
            overflow = signed_distance > PLANE_MARGIN_MM
            overflow_size = int(np.count_nonzero(overflow))
            overflow_ratio = overflow_size / len(voxels_local)
            min_ratio = PLANE_MIN_OVERFLOW_RATIO if label_id in overflowing_vertebrae else 0.25
            if overflow_size < PLANE_MIN_OVERFLOW_VOXELS or overflow_ratio < min_ratio:
                continue
            if overflow_ratio > PLANE_MAX_OVERFLOW_RATIO:
                continue
            crossing = signed_distance > 0
            update = crossing & (signed_distance > best_score)
            new_labels[update] = neighbor
            best_score[update] = signed_distance[update]
            accepted_pairs += 1
        changed = new_labels != label_id
        if np.any(changed):
            refined_roi = refined_mask[roi]
            indices = voxels_local[changed]
            refined_roi[tuple(indices.T)] = new_labels[changed]
            plane_changes += int(np.count_nonzero(changed))

    if return_stages:
        return refined_mask, {'numbering': numbering_mask, 'components': component_mask}
    return refined_mask
