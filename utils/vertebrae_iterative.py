"""
Iterative vertebrae post-processing module for ShapeKit.

Implements VerSe-inspired anatomic consistency cycle (Meng et al., 2022)
on top of ShapeKit's vertebrae functions, adapted for the 26-based label
scheme (26=L5 ... 49=C1).

Key additions over the default vertebrae_postprocessing.py:
  - Residual connected component reassignment
  - Gap detection and filling
  - Fishing for boundary vertebrae
  - Duplicate vertebrae removal (IoU-based)
  - Anatomical size consistency validation
  - Iterative refinement until convergence

Usage:
  Set `vertebrae_engine: shapekit_iterative` in config.yaml to use this module.
"""

import numpy as np
import cc3d
from copy import deepcopy
from scipy.ndimage import binary_erosion

from .vertebrae_postprocessing import all_labels, fill, wise_split_vertebra
from .utils import remove_small_components


# ─── Improved existing functions (adapted from SuPreM) ───────────────────────

def supress_non_largest_components(segmentation, default_val=0):
    """Keep only the 2 largest connected components per vertebra label."""
    result = deepcopy(segmentation)
    new_background = np.zeros(segmentation.shape, dtype=bool)
    for label_id in all_labels:
        cc = cc3d.connected_components(segmentation == label_id, connectivity=6)
        uv, uc = np.unique(cc, return_counts=True)
        if len(uv) < 2:
            continue
        dominant_vals = uv[np.argsort(uc)[::-1][:2]]
        if len(dominant_vals) >= 2:
            keep_mask = (cc == dominant_vals[0]) | (cc == dominant_vals[1])
            new_background |= ~keep_mask & (segmentation == label_id)
    result[new_background] = default_val
    return result


def _merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold):
    """Merge connected components of adjacent vertebrae that are misplaced."""
    nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
    relevant_cc = [(idx, cnt) for idx, cnt in nof_voxels_cc if cnt > voxel_supression_threshold]
    relevant_cc = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[1:]

    nof_voxels_above = [(x, np.sum(cc_above == x)) for x in np.unique(cc_above)]
    relevant_cc_above = [(idx, cnt) for idx, cnt in nof_voxels_above if cnt > voxel_supression_threshold]
    relevant_cc_above = sorted(relevant_cc_above, key=lambda x: x[1], reverse=True)[2:]

    if len(relevant_cc_above) > 0:
        pool = np.zeros(cc_cur.shape, dtype=bool)
        for idx, _ in relevant_cc_above:
            pool |= cc_above == idx
        for idx, _ in relevant_cc:
            pool |= cc_cur == idx
        cc_pool = cc3d.connected_components(pool)
        rel_pool = sorted([(x, np.sum(cc_pool == x)) for x in np.unique(cc_pool)], key=lambda x: x[1], reverse=True)[1:]
        if len(rel_pool) > 0:
            return cc_pool == rel_pool[0][0]
    return None


def spine_adjacent_pairs(segmentation, voxel_supression_threshold=1000, default_val=0):
    """
    Check alternating connected components to identify fragments assigned to the wrong vertebra.
    For each vertebra, examine its neighbors above and below, and reassign misplaced fragments.
    """
    labels = list(all_labels.keys())  # 26..49
    mod_img = deepcopy(segmentation)

    for idx, current in enumerate(labels):
        above = labels[idx - 1] if idx > 0 else None
        below = labels[idx + 1] if idx < len(labels) - 1 else None

        msk_cur = mod_img == current
        cc_cur = cc3d.connected_components(msk_cur, connectivity=6)

        nof_voxels = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
        for cc_id, cnt in nof_voxels:
            if cnt <= voxel_supression_threshold and cc_id != 0:
                mod_img[cc_cur == cc_id] = default_val

        if above is not None:
            msk_above = mod_img == above
            cc_above = cc3d.connected_components(msk_above, connectivity=6)
            consolidated = _merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold)
            if consolidated is not None:
                mod_img[consolidated] = current

        if below is not None:
            msk_below = mod_img == below
            cc_below = cc3d.connected_components(msk_below, connectivity=6)
            consolidated = _merge_cc_of_adjacent(cc_cur, cc_below, voxel_supression_threshold)
            if consolidated is not None:
                mod_img[consolidated] = current

    return mod_img


def relabel_by_z_order(segmentation, label_z_centers, start_label=26):
    """
    Relabel vertebrae based on Z-axis center ordering (bottom to top).
    The lowest vertebra gets label 26 (L5), the highest gets the largest label.
    """
    sorted_labels = sorted(label_z_centers.items(), key=lambda x: x[1], reverse=False)
    new_seg = segmentation.copy()
    new_z_centers = {}
    for new_id, (old_id, z_center) in enumerate(sorted_labels, start=start_label):
        new_seg[segmentation == old_id] = new_id
        new_z_centers[new_id] = z_center
    return new_seg, new_z_centers


def split_overmerged_triplets(merged_seg, size_dict, label_z_centers, counter, size_threshold_ratio=1.5):
    """
    Split over-merged vertebrae: if label i is much larger than min(label i-1, i-2), split by Z-axis.
    """
    sorted_labels = sorted(size_dict.keys(), reverse=True)
    next_new_label = np.max(merged_seg) + 1

    for i in range(2, len(sorted_labels)):
        i2, i1, i0 = sorted_labels[i - 2], sorted_labels[i - 1], sorted_labels[i]
        if i0 not in size_dict or i1 not in size_dict or i2 not in size_dict:
            continue
        threshold = size_threshold_ratio * min(size_dict[i1], size_dict[i2])
        if size_dict[i0] > threshold and counter > 0:
            mask = merged_seg == i0
            coords = np.argwhere(mask)
            if coords.shape[0] == 0:
                continue
            sorted_coords = coords[np.argsort(coords[:, 2])[::-1]]
            half = len(sorted_coords) // 2
            coords_upper = sorted_coords[:half]
            coords_lower = sorted_coords[half:]
            for voxel in coords_lower:
                merged_seg[tuple(voxel)] = next_new_label
            size_dict[i0] = len(coords_upper)
            size_dict[next_new_label] = len(coords_lower)
            label_z_centers[i0] = np.median(coords_upper[:, 2])
            label_z_centers[next_new_label] = np.median(coords_lower[:, 2])
            next_new_label += 1
            counter -= 1
    return merged_seg, label_z_centers


def balance_protrusion(segmentation, label_z_centers, min_cc_voxel=1000):
    """
    For each adjacent pair (A=lower label, B=higher label):
    - If a sub-region of A protrudes above B's center, reassign it to B.
    - If a sub-region of B drops below A's center, reassign it to A.
    """
    corrected = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())
    for i in range(len(sorted_labels) - 1):
        A = sorted_labels[i]
        B = sorted_labels[i + 1]
        z_A = label_z_centers[A]
        z_B = label_z_centers[B]

        cc_A = cc3d.connected_components(corrected == A, connectivity=6)
        for cc_id in np.unique(cc_A):
            if cc_id == 0:
                continue
            coords = np.argwhere(cc_A == cc_id)
            if coords.shape[0] < min_cc_voxel:
                continue
            if np.median(coords[:, 2]) > z_B:
                corrected[cc_A == cc_id] = B

        cc_B = cc3d.connected_components(corrected == B, connectivity=6)
        for cc_id in np.unique(cc_B):
            if cc_id == 0:
                continue
            coords = np.argwhere(cc_B == cc_id)
            if coords.shape[0] < min_cc_voxel:
                continue
            if np.median(coords[:, 2]) < z_A:
                corrected[cc_B == cc_id] = A
    return corrected


def reallocate_based_on_size(segmentation):
    """
    Handle extra-small (merge into nearest neighbor) and extra-large (split) vertebrae.
    Then relabel by Z-order and balance protrusion.
    """
    size_dict = {}
    label_z_centers = {}
    for label_id in np.unique(segmentation):
        if label_id == 0:
            continue
        mask = segmentation == label_id
        mask = remove_small_components(mask, threshold=max(int(np.sum(mask) / 10), 100))
        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            continue
        label_z_centers[label_id] = np.median(coords[:, 2])
        size_dict[label_id] = np.sum(mask)

    size_threshold_ratio = 2 / 3
    merged_seg = segmentation.copy()
    to_merge = []
    for label_id in label_z_centers:
        neighbors = [label_id - 1, label_id + 1]
        neighbor_sizes = [size_dict.get(n, 0) for n in neighbors if n in size_dict]
        if len(neighbor_sizes) < 2:
            continue
        if size_dict[label_id] < size_threshold_ratio * np.mean(neighbor_sizes):
            to_merge.append(label_id)

    split_counter = len(to_merge)
    for label_id in to_merge:
        min_dist = np.inf
        nearest = None
        z = label_z_centers[label_id]
        for other_id, other_z in label_z_centers.items():
            if other_id == label_id:
                continue
            dist = abs(z - other_z)
            if dist < min_dist:
                min_dist = dist
                nearest = other_id
        if nearest is not None:
            merged_seg[merged_seg == label_id] = nearest
            size_dict[nearest] = size_dict.get(nearest, 0) + size_dict[label_id]
            del size_dict[label_id]
            del label_z_centers[label_id]

    split_seg, label_z_centers = split_overmerged_triplets(
        merged_seg, size_dict, label_z_centers, counter=split_counter
    )

    new_seg, label_z_centers = relabel_by_z_order(split_seg, label_z_centers)
    new_seg = balance_protrusion(new_seg, label_z_centers)

    return new_seg


# ─── VerSe-inspired functions (Meng et al., 2022) ────────────────────────────
# Adapted from https://gitlab.inria.fr/spine/vertebrae_segmentation

def find_residual_components(binary_spine, segmentation, min_size=500):
    """
    Find connected components in the residual between the binary spine mask
    and the current individual vertebrae labels.
    """
    individual_union = (segmentation > 0).astype(np.uint8)
    residual = binary_spine.astype(np.uint8) - individual_union
    residual[residual != 1] = 0

    if not np.any(residual):
        return []

    if np.sum(residual) > min_size * 5:
        residual = binary_erosion(residual).astype(np.uint8)

    cc = cc3d.connected_components(residual, connectivity=6)
    components = []
    labels_cc, counts = np.unique(cc, return_counts=True)

    for lbl, cnt in zip(labels_cc, counts):
        if lbl == 0:
            continue
        if cnt >= min_size:
            components.append((cc == lbl, cnt))

    return components


def reassign_residual_to_nearest(segmentation, residual_components, label_z_centers):
    """
    Reassign residual connected components to the nearest vertebra
    based on 3D centroid distance.
    """
    if not residual_components or not label_z_centers:
        return segmentation

    result = segmentation.copy()

    centroids = {}
    for label_id in label_z_centers:
        mask = segmentation == label_id
        if not np.any(mask):
            continue
        coords = np.argwhere(mask)
        centroids[label_id] = (np.median(coords[:, 0]),
                               np.median(coords[:, 1]),
                               np.median(coords[:, 2]))

    for component, cnt in residual_components:
        coords = np.argwhere(component)
        if coords.shape[0] == 0:
            continue
        cx, cy, cz = np.median(coords[:, 0]), np.median(coords[:, 1]), np.median(coords[:, 2])

        min_dist = np.inf
        nearest_label = 0
        for label_id, (mx, my, mz) in centroids.items():
            dist = np.sqrt((cx - mx)**2 + (cy - my)**2 + (cz - mz)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_label = label_id

        if nearest_label > 0:
            result[component] = nearest_label

    return result


def detect_and_fill_gaps(segmentation, label_z_centers, binary_spine, min_size=500, gap_threshold=1.8):
    """
    Detect anomalously large gaps between consecutive vertebrae Z-centers.
    Try to assign residual components in the gap region to a new vertebra.
    """
    if len(label_z_centers) < 3:
        return segmentation

    result = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())
    z_centers = [label_z_centers[l] for l in sorted_labels]

    gaps = [abs(z_centers[i+1] - z_centers[i]) for i in range(len(z_centers)-1)]
    median_gap = np.median(gaps)
    if median_gap == 0:
        return result

    for i, gap in enumerate(gaps):
        if gap > gap_threshold * median_gap:
            z_low = min(z_centers[i], z_centers[i+1])
            z_high = max(z_centers[i], z_centers[i+1])

            gap_mask = binary_spine.copy()
            gap_mask[result > 0] = 0
            gap_mask[:, :, :int(z_low)] = 0
            gap_mask[:, :, int(z_high)+1:] = 0

            if np.any(gap_mask):
                cc = cc3d.connected_components(gap_mask.astype(np.uint8), connectivity=6)
                labels_cc, counts = np.unique(cc, return_counts=True)
                for lbl, cnt in zip(labels_cc, counts):
                    if lbl == 0 or cnt < min_size:
                        continue
                    expected_label = min(sorted_labels[i], sorted_labels[i+1]) + 1
                    if expected_label not in label_z_centers:
                        result[cc == lbl] = expected_label
                        print(f"    Gap filled: assigned {cnt} voxels to label {expected_label}")

    return result


def fishing_for_boundary_vertebrae(segmentation, label_z_centers, binary_spine, min_size=500):
    """
    Check for missing vertebrae at the inferior and superior boundaries.
    If the lowest detected vertebra is not L5 (label 26) or highest is not C1 (label 49),
    look for residual components beyond the boundaries.
    """
    if not label_z_centers:
        return segmentation

    result = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())
    z_centers = [label_z_centers[l] for l in sorted_labels]

    # Inferior boundary: should reach label 26 (L5)
    lowest_label = sorted_labels[0]
    lowest_z = z_centers[0]

    if lowest_label > 26:
        below_mask = binary_spine.copy()
        below_mask[result > 0] = 0
        below_mask[:, :, int(lowest_z + 5):] = 0

        if np.any(below_mask):
            cc = cc3d.connected_components(below_mask.astype(np.uint8), connectivity=6)
            labels_cc, counts = np.unique(cc, return_counts=True)
            for lbl, cnt in zip(labels_cc, counts):
                if lbl == 0 or cnt < min_size:
                    continue
                new_label = lowest_label - 1
                result[cc == lbl] = new_label
                print(f"    Fished inferior: assigned {cnt} voxels to label {new_label}")

    # Superior boundary: should reach label 49 (C1)
    highest_label = sorted_labels[-1]
    highest_z = z_centers[-1]

    if highest_label < 49:
        above_mask = binary_spine.copy()
        above_mask[result > 0] = 0
        above_mask[:, :, :int(highest_z - 5)] = 0

        if np.any(above_mask):
            cc = cc3d.connected_components(above_mask.astype(np.uint8), connectivity=6)
            labels_cc, counts = np.unique(cc, return_counts=True)
            for lbl, cnt in zip(labels_cc, counts):
                if lbl == 0 or cnt < min_size:
                    continue
                new_label = highest_label + 1
                result[cc == lbl] = new_label
                print(f"    Fished superior: assigned {cnt} voxels to label {new_label}")

    return result


def remove_duplicate_vertebrae(segmentation, iou_threshold=0.5):
    """
    Check for overlapping vertebrae masks (duplicated detections).
    If two adjacent labels have IoU > threshold, merge the smaller into the larger.
    """
    labels_present = [l for l in np.unique(segmentation) if l > 0]
    if len(labels_present) < 2:
        return segmentation

    result = segmentation.copy()

    for i in range(len(labels_present) - 1):
        l1 = labels_present[i]
        l2 = labels_present[i + 1]
        m1 = result == l1
        m2 = result == l2

        intersection = np.logical_and(m1, m2).sum()
        if intersection == 0:
            continue

        union = np.logical_or(m1, m2).sum()
        iou = intersection / union if union > 0 else 0

        if iou > iou_threshold:
            if m1.sum() < m2.sum():
                result[m1] = l2
            else:
                result[m2] = l1
            print(f"    Merged duplicate labels {l1} and {l2} (IoU={iou:.3f})")

    return result


def check_anatomical_size_consistency(segmentation, label_z_centers, size_dict):
    """
    Validate vertebrae sizes against anatomical expectations.
    Lumbar (L1-L5, labels 26-30) should be larger than thoracic (T1-T12, labels 31-42),
    which should be larger than cervical (C1-C7, labels 43-49).
    Flag outliers that deviate significantly from their group median.
    """
    if not size_dict or len(size_dict) < 3:
        return segmentation, label_z_centers, size_dict

    result = segmentation.copy()
    new_z_centers = dict(label_z_centers)
    new_size = dict(size_dict)

    sorted_labels = sorted(new_z_centers.keys())

    # Group by region: 26-30 = lumbar, 31-42 = thoracic, 43-49 = cervical
    lumbar = [l for l in sorted_labels if 26 <= l <= 30]
    thoracic = [l for l in sorted_labels if 31 <= l <= 42]
    cervical = [l for l in sorted_labels if 43 <= l <= 49]

    lumbar_median = np.median([new_size[l] for l in lumbar]) if lumbar else 0
    cervical_median = np.median([new_size[l] for l in cervical]) if cervical else 0

    # Remove cervical vertebrae that are abnormally small (< 20% of cervical median)
    for label_id in cervical:
        if cervical_median > 0 and new_size.get(label_id, 0) < 0.2 * cervical_median:
            print(f"    Anatomical check: label {label_id} too small for cervical, removing")
            result[result == label_id] = 0
            if label_id in new_z_centers:
                del new_z_centers[label_id]
            if label_id in new_size:
                del new_size[label_id]

    # Remove lumbar vertebrae that are abnormally small (< 20% of lumbar median)
    for label_id in lumbar:
        if lumbar_median > 0 and new_size.get(label_id, 0) < 0.2 * lumbar_median:
            print(f"    Anatomical check: label {label_id} too small for lumbar, removing")
            result[result == label_id] = 0
            if label_id in new_z_centers:
                del new_z_centers[label_id]
            if label_id in new_size:
                del new_size[label_id]

    return result, new_z_centers, new_size


# ─── Iterative refinement pipeline ───────────────────────────────────────────

def iterative_refinement(segmentation, max_iter=3, voxel_supression_threshold=1000):
    """
    Run the postprocessing pipeline iteratively until convergence.
    Each iteration: clean → residual reassignment → gap fill → fishing → reallocate.
    """
    prev_seg = None

    for iteration in range(max_iter):
        print(f"  ─ Iteration {iteration+1}/{max_iter} ─")

        # Save binary spine mask before processing (union of all labels)
        binary_spine = (segmentation > 0).astype(np.uint8)

        # Step 1: Remove small noise components per vertebra
        for label_id in all_labels:
            mask = (segmentation == label_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            cleaned = remove_small_components(mask, threshold=500)
            segmentation[~cleaned.astype(bool) & (segmentation == label_id)] = 0

        # Step 2: Suppress non-largest connected components (keep top 2)
        segmentation = supress_non_largest_components(segmentation)

        # Step 3: Fill holes
        segmentation = fill(segmentation)

        # Step 4: Spine adjacent pairs correction
        segmentation = spine_adjacent_pairs(segmentation, voxel_supression_threshold=voxel_supression_threshold)

        # Step 5: Compute label centers and sizes
        label_z_centers = {}
        size_dict = {}
        for label_id in np.unique(segmentation):
            if label_id == 0:
                continue
            mask = segmentation == label_id
            coords = np.argwhere(mask)
            if coords.shape[0] == 0:
                continue
            label_z_centers[label_id] = np.median(coords[:, 2])
            size_dict[label_id] = np.sum(mask)

        # Step 6: Anatomical size consistency check
        segmentation, label_z_centers, size_dict = check_anatomical_size_consistency(
            segmentation, label_z_centers, size_dict
        )

        # Step 7: Reassign residual components to nearest vertebrae
        residual_components = find_residual_components(binary_spine, segmentation, min_size=500)
        if residual_components:
            print(f"    Found {len(residual_components)} residual components, reassigning...")
            segmentation = reassign_residual_to_nearest(segmentation, residual_components, label_z_centers)

        # Step 8: Detect and fill gaps
        segmentation = detect_and_fill_gaps(segmentation, label_z_centers, binary_spine, min_size=500)

        # Step 9: Fishing for boundary vertebrae
        segmentation = fishing_for_boundary_vertebrae(segmentation, label_z_centers, binary_spine, min_size=500)

        # Step 10: Remove duplicates
        segmentation = remove_duplicate_vertebrae(segmentation, iou_threshold=0.5)

        # Step 11: Reallocate based on size (merge small, split large, relabel, balance)
        segmentation = reallocate_based_on_size(segmentation)

        # Step 12: Fill holes again
        segmentation = fill(segmentation)

        # Check convergence
        if prev_seg is not None:
            changed = np.sum(prev_seg != segmentation)
            total = np.sum((prev_seg > 0) | (segmentation > 0))
            if total > 0:
                change_pct = changed / total
                print(f"    Change rate: {change_pct:.4f}")
                if change_pct < 0.01:
                    print("  Convergence reached.")
                    break

        prev_seg = segmentation.copy()

    # Final cleanup: remove any labels outside the valid range 26-49
    valid_min = min(all_labels.keys())
    valid_max = max(all_labels.keys())
    segmentation[(segmentation > valid_max) | ((segmentation > 0) & (segmentation < valid_min))] = 0

    return segmentation


# ─── Entry point (same signature as vertebrae_postprocessing.postprocessing_vertebrae) ─

def postprocessing_vertebrae(patiend_id: str, segmentation_dict: dict, logger):
    """
    Post-processing for vertebrae labels using iterative VerSe-inspired refinement.

    Steps:
        1. Build combined segmentation from individual vertebrae masks
        2. Run iterative refinement pipeline (VerSe-inspired consistency loop)
        3. Split back into individual vertebrae masks
    """
    vertebrae_segmentations = np.zeros_like(next(iter(segmentation_dict.values())), dtype=np.uint8)

    # Assign fixed anatomical label IDs
    for label_id, vertebra_name in all_labels.items():
        if vertebra_name not in segmentation_dict:
            logger.info(f"[INFO] {patiend_id}, Missing: {vertebra_name}, skipping...")
            continue
        mask = segmentation_dict[vertebra_name]
        vertebrae_segmentations[mask > 0] = label_id

    # Run iterative refinement pipeline
    segmentation = iterative_refinement(vertebrae_segmentations, max_iter=3, voxel_supression_threshold=1000)

    # Split back into vertebrae masks
    processed_dict = segmentation_dict.copy()
    for label_id, vertebra_name in all_labels.items():
        if np.any(segmentation == label_id):
            processed_dict[vertebra_name] = (segmentation == label_id).astype(np.uint8)

    return processed_dict
