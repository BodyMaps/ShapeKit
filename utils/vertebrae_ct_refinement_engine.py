from __future__ import annotations
import argparse
import json
from pathlib import Path
import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
LEVELS = ['BG', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']

def largest_component_centres(raw: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    centres = []
    objects = ndimage.find_objects(raw, max_label=24)
    for label in range(1, 25):
        box = objects[label - 1]
        if box is None:
            continue
        local = raw[box] == label
        cc = cc3d.connected_components(local, connectivity=26)
        counts = np.bincount(cc.ravel())
        if len(counts) <= 1:
            continue
        component = int(np.argmax(counts[1:]) + 1)
        offset = np.asarray([s.start for s in box])
        points = np.argwhere(cc == component) + offset
        centres.append((points * spacing).mean(axis=0))
    return np.asarray(centres)

def cluster_close_cores(cores: list[dict], maximum_gap_mm: float=8.0) -> list[dict]:
    cores = sorted(cores, key=lambda item: item['z_mm'])
    clusters: list[list[dict]] = []
    for core in cores:
        if not clusters or core['z_mm'] - clusters[-1][-1]['z_mm'] > maximum_gap_mm:
            clusters.append([core])
        else:
            clusters[-1].append(core)
    merged = []
    for cluster in clusters:
        points = np.concatenate([item['points'] for item in cluster], axis=0)
        weights = np.asarray([len(item['points']) for item in cluster], dtype=float)
        merged.append({'points': points, 'z_mm': float(np.average([item['z_mm'] for item in cluster], weights=weights)), 'volume_mm3': float(sum((item['volume_mm3'] for item in cluster))), 'islands': len(cluster)})
    return merged

def extract_full_body_cores(raw: np.ndarray, image: nib.Nifti1Image) -> tuple[list[dict], dict]:
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    centres = largest_component_centres(raw, spacing)
    centre_x = float(np.median(centres[:, 0]))
    centre_y = float(np.median(centres[:, 1]))
    shape = np.asarray(raw.shape)
    voxel_axes = [np.arange(shape[axis]) * spacing[axis] for axis in range(3)]
    x_indices = np.where(np.abs(voxel_axes[0] - centre_x) <= 45.0)[0]
    y_indices = np.where((voxel_axes[1] >= centre_y - 130.0) & (voxel_axes[1] <= centre_y + 130.0))[0]
    x0, x1 = (int(x_indices[0]), int(x_indices[-1] + 1))
    y0, y1 = (int(y_indices[0]), int(y_indices[-1] + 1))
    foreground = raw[x0:x1, y0:y1, :] > 0
    local_y = voxel_axes[1][y0:y1]
    anterior = np.zeros_like(foreground)
    for z in range(foreground.shape[2]):
        points = np.argwhere(foreground[:, :, z])
        if len(points) == 0:
            continue
        y_threshold = float(np.percentile(local_y[points[:, 1]], 45.0))
        anterior[:, :, z] = foreground[:, :, z] & (local_y[None, :] >= y_threshold)
    distance = ndimage.distance_transform_edt(anterior, sampling=spacing)
    attempts = []
    selected = None
    for erosion_mm in (3.5, 4.0, 3.0, 4.5, 2.5, 5.0, 5.5, 6.0):
        cc = cc3d.connected_components(distance >= erosion_mm, connectivity=26)
        counts = np.bincount(cc.ravel())
        cores = []
        for component in range(1, len(counts)):
            n = int(counts[component])
            volume = n * voxel_volume
            if volume < 300.0:
                continue
            points = np.argwhere(cc == component)
            points[:, 0] += x0
            points[:, 1] += y0
            world_z = points[:, 2] * spacing[2]
            cores.append({'points': points, 'z_mm': float(world_z.mean()), 'volume_mm3': float(volume), 'islands': 1})
        merged = cluster_close_cores(cores)
        centres_z = np.asarray([item['z_mm'] for item in merged])
        gaps = np.diff(centres_z)
        plausible = len(merged) == 24 and len(gaps) == 23 and (float(np.min(gaps)) >= 8.0) and (float(np.max(gaps)) <= 50.0)
        attempts.append({'erosion_mm': erosion_mm, 'core_count': len(merged), 'minimum_gap_mm': None if not len(gaps) else float(np.min(gaps)), 'maximum_gap_mm': None if not len(gaps) else float(np.max(gaps)), 'plausible': plausible})
        if plausible:
            selected = merged
            break
    diagnostics = {'centre_x_mm': centre_x, 'centre_y_mm': centre_y, 'attempts': attempts, 'status': 'IDENTITY_READY' if selected is not None else 'ABSTAIN'}
    return ([] if selected is None else selected, diagnostics)

def extract_partial_body_cores(raw: np.ndarray, image: nib.Nifti1Image) -> tuple[list[dict], dict]:
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
    centres = largest_component_centres(raw, spacing)
    centre_x = float(np.median(centres[:, 0]))
    centre_y = float(np.median(centres[:, 1]))
    shape = np.asarray(raw.shape)
    voxel_axes = [np.arange(shape[axis]) * spacing[axis] for axis in range(3)]
    x_indices = np.where(np.abs(voxel_axes[0] - centre_x) <= 45.0)[0]
    y_indices = np.where((voxel_axes[1] >= centre_y - 130.0) & (voxel_axes[1] <= centre_y + 130.0))[0]
    x0, x1 = (int(x_indices[0]), int(x_indices[-1] + 1))
    y0, y1 = (int(y_indices[0]), int(y_indices[-1] + 1))
    foreground = raw[x0:x1, y0:y1, :] > 0
    local_y = voxel_axes[1][y0:y1]
    anterior = np.zeros_like(foreground)
    for z in range(foreground.shape[2]):
        points = np.argwhere(foreground[:, :, z])
        if len(points) == 0:
            continue
        y_threshold = float(np.percentile(local_y[points[:, 1]], 45.0))
        anterior[:, :, z] = foreground[:, :, z] & (local_y[None, :] >= y_threshold)
    distance = ndimage.distance_transform_edt(anterior, sampling=spacing)
    attempts = []
    selected = None
    accepted = []
    for erosion_mm in (3.5, 4.0, 3.0, 4.5, 2.5, 5.0, 5.5, 6.0):
        cc = cc3d.connected_components(distance >= erosion_mm, connectivity=26)
        counts = np.bincount(cc.ravel())
        cores = []
        for component in range(1, len(counts)):
            n = int(counts[component])
            volume = n * voxel_volume
            if volume < 300.0:
                continue
            points = np.argwhere(cc == component)
            points[:, 0] += x0
            points[:, 1] += y0
            world_z = points[:, 2] * spacing[2]
            cores.append({'points': points, 'z_mm': float(world_z.mean()), 'volume_mm3': float(volume), 'islands': 1})
        merged = cluster_close_cores(cores)
        centres_z = np.asarray([item['z_mm'] for item in merged])
        gaps = np.diff(centres_z)
        plausible = 6 <= len(merged) <= 24 and len(gaps) == len(merged) - 1 and (float(np.min(gaps)) >= 8.0) and (float(np.max(gaps)) <= 50.0)
        attempts.append({'erosion_mm': erosion_mm, 'core_count': len(merged), 'minimum_gap_mm': None if not len(gaps) else float(np.min(gaps)), 'maximum_gap_mm': None if not len(gaps) else float(np.max(gaps)), 'plausible': plausible})
        if plausible:
            fits = []
            for start in range(1, 26 - len(merged)):
                fractions = [float(np.mean(raw[tuple(c['points'].T)] == start + i)) for i, c in enumerate(merged)]
                if min(fractions[:2] + fractions[-2:]) >= 0.9 and np.mean(fractions) >= 0.65:
                    fits.append((start, fractions))
            if len(fits) == 1:
                accepted.append((merged, fits[0][0], erosion_mm, fits[0][1]))
    chosen = None
    for a in accepted:
        for b in accepted:
            if a is b or a[1] != b[1] or len(a[0]) != len(b[0]):
                continue
            if max((abs(x['z_mm'] - y['z_mm']) for x, y in zip(a[0], b[0]))) <= 3.0:
                selected = a[0]
                chosen = a
                break
        if chosen is not None:
            break
    diagnostics = {'centre_x_mm': centre_x, 'centre_y_mm': centre_y, 'attempts': attempts, 'status': 'IDENTITY_READY' if selected is not None else 'ABSTAIN'}
    if chosen is not None:
        diagnostics.update(start_label=chosen[1], selected_radius=chosen[2], core_label_fractions=chosen[3], status='PARTIAL_IDENTITY_READY', accepted_scales=[{'radius': a[2], 'start_label': a[1], 'count': len(a[0])} for a in accepted])
    return ([] if selected is None else selected, diagnostics)

def extract_body_cores(raw, image):
    cores, diagnostics = extract_full_body_cores(raw, image)
    if cores:
        return (cores, diagnostics)
    cores, partial = extract_partial_body_cores(raw, image)
    partial['full_chain_status'] = diagnostics['status']
    return (cores, partial)

def query_nearest_core(points: np.ndarray, tree: cKDTree, seed_labels: np.ndarray, spacing: np.ndarray, chunk_size: int=250000) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(len(points), dtype=np.uint8)
    distances = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        d, nearest = tree.query(points[start:stop] * spacing, k=1, workers=1)
        labels[start:stop] = seed_labels[nearest]
        distances[start:stop] = d
    return (labels, distances)

def ct_guided_boundary_growth(candidate: np.ndarray, ct: np.ndarray, spacing: np.ndarray, radius_mm: float, minimum_hu: float) -> int:
    if radius_mm <= 0:
        return 0
    points = np.argwhere(candidate > 0)
    if not len(points):
        return 0
    margins = np.ceil(radius_mm / spacing).astype(int) + 2
    lower = np.maximum(points.min(axis=0) - margins, 0)
    upper = np.minimum(points.max(axis=0) + margins + 1, candidate.shape)
    box = tuple((slice(int(a), int(b)) for a, b in zip(lower, upper)))
    local = candidate[box]
    background = local == 0
    distance, nearest_indices = ndimage.distance_transform_edt(background, sampling=spacing, return_indices=True)
    nearest_label = local[tuple(nearest_indices)]
    add = background & (distance <= radius_mm) & (ct[box] >= minimum_hu) & (nearest_label > 0)
    candidate[box][add] = nearest_label[add]
    return int(np.sum(add))

def strict_tiny_remote_cleanup(raw: np.ndarray, candidate: np.ndarray, spacing: np.ndarray, tree: cKDTree | None, seed_labels: np.ndarray | None) -> list[dict]:
    operations = []
    voxel_volume = float(np.prod(spacing))
    objects = ndimage.find_objects(raw, max_label=24)
    for source in range(1, 25):
        box = objects[source - 1]
        if box is None:
            continue
        local = raw[box] == source
        cc = cc3d.connected_components(local, connectivity=26)
        counts = np.bincount(cc.ravel())
        if len(counts) <= 2:
            continue
        largest = int(np.argmax(counts[1:]) + 1)
        offset = np.asarray([s.start for s in box])
        total = int(local.sum())
        for component in range(1, len(counts)):
            if component == largest:
                continue
            n = int(counts[component])
            volume = n * voxel_volume
            fraction = n / total
            if volume > 500.0 or fraction > 0.02:
                continue
            points = np.argwhere(cc == component) + offset
            if tree is not None and np.any(seed_labels == source):
                nearest_labels, _ = query_nearest_core(points, tree, seed_labels, spacing)
                longitudinal_mismatch = abs(int(np.median(nearest_labels)) - source) >= 3
                source_tree = cKDTree(tree.data[seed_labels == source])
                distances, _ = source_tree.query(points * spacing, k=1, workers=1)
                remote = float(np.min(distances)) >= 90.0
            else:
                main_points = np.argwhere(cc == largest) + offset
                main_tree = cKDTree(main_points * spacing)
                distances, _ = main_tree.query(points * spacing, k=1, workers=1)
                longitudinal_mismatch = True
                remote = float(np.min(distances)) >= 120.0
            if remote and longitudinal_mismatch:
                candidate[tuple(points.T)] = 0
                operations.append({'action': 'REMOVE_TINY_REMOTE', 'source': LEVELS[source], 'voxels': n, 'volume_mm3': volume, 'fraction': fraction, 'minimum_distance_mm': float(np.min(distances))})
    return operations

def radial_core_support(raw, ct, spacing, cores, diagnostics):
    if 'start_label' not in diagnostics or len({r['radius'] for r in diagnostics.get('accepted_scales', []) if r['start_label'] == diagnostics['start_label'] and r['count'] == len(cores)}) < 2:
        return (False, {'status': 'NO_MULTISCALE_PARTIAL_SCAFFOLD'})
    angles = np.arange(24) * 2 * np.pi / 24
    radii = np.arange(4.0, 30.01, 1.0)
    rows = []
    for label, core in enumerate(cores, start=diagnostics['start_label']):
        center = np.median(core['points'], axis=0)
        margin = np.ceil(33 / spacing).astype(int)
        lo = np.maximum(np.floor(center).astype(int) - margin, 0)
        hi = np.minimum(np.ceil(center).astype(int) + margin + 1, raw.shape)
        box = tuple((slice(int(a), int(b)) for a, b in zip(lo, hi)))
        image = ct[box]
        foreground = (raw[box] > 0).astype(np.float32)
        offsets = np.zeros((3, 24, len(radii), 3))
        offsets[:, :, :, 0] = np.cos(angles)[None, :, None] * radii[None, None, :]
        offsets[:, :, :, 1] = np.sin(angles)[None, :, None] * radii[None, None, :]
        offsets[:, :, :, 2] = np.array([-1.5, 0, 1.5])[:, None, None]
        points = center - lo + offsets / spacing
        hu = ndimage.map_coordinates(image, points.reshape(-1, 3).T, order=1, mode='constant', cval=-1000).reshape(3, 24, -1)
        fg = ndimage.map_coordinates(foreground, points.reshape(-1, 3).T, order=1, mode='constant', cval=0).reshape(3, 24, -1)
        supported = (hu >= 150) & (fg >= 0.75)
        rays = np.any(supported[:, :, :-1] & supported[:, :, 1:], axis=2)
        fractions = rays.mean(1)
        quadrants = rays.reshape(3, 4, 6).mean(2)
        plane_ok = (fractions >= 0.75) & (quadrants.min(1) >= 0.5)
        core_hu = ct[tuple(core['points'].T)]
        median = float(np.median(core_hu))
        accepted = bool(plane_ok.sum() >= 2 and median >= 0)
        rows.append({'label': int(label), 'core_median_hu': median, 'radial_support_fraction': fractions.tolist(), 'quadrant_support': quadrants.tolist(), 'accepted': accepted})
    return (all((r['accepted'] for r in rows)), {'status': 'RADIAL_CT_CHECK', 'rows': rows, 'meaning': 'Surrounding bone signal, not independent label truth or an expert cortex annotation.'})

def build_case(raw_path: Path, ct_path: Path, out_dir: Path, boundary_radius_mm: float, boundary_minimum_hu: float) -> dict:
    image = nib.load(raw_path)
    ct_image = nib.load(ct_path)
    if image.shape != ct_image.shape or not np.allclose(image.affine, ct_image.affine, atol=1e-05):
        raise ValueError(f'CT/prediction geometry mismatch: {ct_path}')
    raw = np.asarray(image.dataobj, dtype=np.uint8)
    ct = np.asarray(ct_image.dataobj, dtype=np.float32)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if not np.any(raw):
        out_dir.mkdir(parents=True, exist_ok=True)
        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(raw, image.affine, header), out_dir / 'combined_labels.nii.gz')
        seg_dir = out_dir / 'segmentations'
        seg_dir.mkdir(exist_ok=True)
        for label in range(1, 25):
            nib.save(nib.Nifti1Image(raw, image.affine, header), seg_dir / f'vertebrae_{LEVELS[label]}.nii.gz')
        return {'raw': str(raw_path), 'ct': str(ct_path), 'identity': {'status': 'RAW_EMPTY'}, 'changed_voxels': 0, 'removed_voxels': 0, 'relabelled_voxels': 0, 'added_voxels': 0, 'operations': []}
    cores, diagnostics = extract_body_cores(raw, image)
    if cores:
        ct_rows = []
        for target, core in enumerate(cores, start=diagnostics.get('start_label', 1)):
            hu = ct[tuple(core['points'].T)]
            ct_rows.append({'target': LEVELS[target], 'median_hu': float(np.median(hu)), 'fraction_hu_gt_100': float(np.mean(hu > 100.0))})
        diagnostics['ct_core_evidence'] = ct_rows
        if any((row['median_hu'] < 120.0 or row['fraction_hu_gt_100'] < 0.9 for row in ct_rows)):
            supported, radial = radial_core_support(raw, ct, spacing, cores, diagnostics)
            diagnostics['radial_ct_evidence'] = radial
            if supported:
                diagnostics['status'] = 'PARTIAL_IDENTITY_RADIAL_CT_SUPPORTED'
            else:
                cores = []
                diagnostics['status'] = 'ABSTAIN_CT_CORE_NOT_BONE_SUPPORTED'
    if cores:
        endpoint_fraction = {}
        for target in (1, 2, 3, 22, 23, 24) if 'start_label' not in diagnostics else sorted(set([diagnostics['start_label'], diagnostics['start_label'] + 1, diagnostics['start_label'] + len(cores) - 2, diagnostics['start_label'] + len(cores) - 1])):
            values = raw[tuple(cores[target - diagnostics.get('start_label', 1)]['points'].T)]
            endpoint_fraction[LEVELS[target]] = float(np.mean(values == target))
        diagnostics['endpoint_anchor_fraction'] = endpoint_fraction
        if min(endpoint_fraction.values()) < 0.9:
            cores = []
            diagnostics['status'] = 'ABSTAIN_ENDPOINT_ANCHORS_UNCERTAIN'
    candidate = raw.copy()
    operations = []
    tree = None
    seed_labels = None
    if cores:
        seed_points = []
        labels = []
        core_rows = []
        for target, core in enumerate(cores, start=diagnostics.get('start_label', 1)):
            points = core['points']
            seed_points.append(points)
            labels.append(np.full(len(points), target, dtype=np.uint8))
            raw_values, counts = np.unique(raw[tuple(points.T)], return_counts=True)
            order = np.argsort(counts)[::-1]
            composition = [{'raw': LEVELS[int(raw_values[i])], 'fraction': float(counts[i] / len(points))} for i in order]
            core_rows.append({'target': LEVELS[target], 'z_mm': core['z_mm'], 'volume_mm3': core['volume_mm3'], 'islands': core['islands'], 'raw_composition': composition, 'expected_raw_fraction': float(np.mean(raw[tuple(points.T)] == target))})
        seed_points_array = np.concatenate(seed_points, axis=0)
        seed_labels = np.concatenate(labels, axis=0)
        tree = cKDTree(seed_points_array * spacing)
        core_y_mm = np.zeros(25, dtype=float)
        core_x_mm = np.zeros(25, dtype=float)
        for target, core in enumerate(cores, start=diagnostics.get('start_label', 1)):
            core_y_mm[target] = float(np.median(core['points'][:, 1] * spacing[1]))
            core_x_mm[target] = float(np.median(core['points'][:, 0] * spacing[0]))
        diagnostics['cores'] = core_rows
        source_total = np.zeros(25, dtype=np.int64)
        source_correct = np.zeros(25, dtype=np.int64)
        for target, core in enumerate(cores, start=diagnostics.get('start_label', 1)):
            values = raw[tuple(core['points'].T)]
            for source in np.unique(values):
                if source == 0:
                    continue
                n = int(np.sum(values == source))
                source_total[source] += n
                if source == target:
                    source_correct[source] += n
        affected = {source for source in range(1, 25) if source_total[source] > 0 and source_correct[source] / source_total[source] < 0.8}
        diagnostics['source_core_identity_fraction'] = {LEVELS[source]: float(source_correct[source] / source_total[source]) for source in range(1, 25) if source_total[source] > 0}
        diagnostics['affected_sources'] = [LEVELS[x] for x in sorted(affected)]
        for source in sorted(affected):
            points = np.argwhere(raw == source)
            nearest, distances = query_nearest_core(points, tree, seed_labels, spacing)
            posterior = (points[:, 1] * spacing[1] < core_y_mm[nearest] - 30.0) & (np.abs(points[:, 0] * spacing[0] - core_x_mm[nearest]) <= 22.0)
            if np.any(posterior):
                adjusted = points[posterior].astype(float)
                adjusted[:, 2] += 8.0 / spacing[2]
                compensated, compensated_distance = query_nearest_core(adjusted, tree, seed_labels, spacing)
                nearest[posterior] = compensated
                distances[posterior] = compensated_distance
            changed = nearest != source
            candidate[tuple(points.T)] = nearest
            operations.append({'action': 'BODY_ANCHOR_REPARTITION', 'source': LEVELS[source], 'voxels_examined': int(len(points)), 'voxels_relabelled': int(np.sum(changed)), 'posterior_compensated_voxels': int(np.sum(posterior)), 'median_core_distance_mm': float(np.median(distances)), 'targets': {LEVELS[int(label)]: int(count) for label, count in zip(*np.unique(nearest, return_counts=True))}})
        if affected and max(affected) < 24:
            source = max(affected) + 1
            target = source - 1
            points = np.argwhere(raw == source)
            nearest, distances = query_nearest_core(points, tree, seed_labels, spacing)
            posterior = (points[:, 1] * spacing[1] < core_y_mm[nearest] - 30.0) & (np.abs(points[:, 0] * spacing[0] - core_x_mm[nearest]) <= 22.0)
            if np.any(posterior):
                adjusted = points[posterior].astype(float)
                adjusted[:, 2] += 8.0 / spacing[2]
                compensated, compensated_distance = query_nearest_core(adjusted, tree, seed_labels, spacing)
                nearest[posterior] = compensated
                distances[posterior] = compensated_distance
            move = posterior & (nearest == target)
            if np.any(move):
                candidate[tuple(points[move].T)] = target
                operations.append({'action': 'ADJACENT_POSTERIOR_TAIL_REASSIGN', 'source': LEVELS[source], 'target': LEVELS[target], 'voxels': int(np.sum(move)), 'posterior_candidate_voxels': int(np.sum(posterior)), 'median_core_distance_mm': float(np.median(distances[move]))})
        objects = ndimage.find_objects(raw, max_label=24)
        main_trees = {}
        main_centres = {}
        for source in range(1, 25):
            box = objects[source - 1]
            if box is None:
                continue
            local = raw[box] == source
            cc = cc3d.connected_components(local, connectivity=26)
            counts = np.bincount(cc.ravel())
            largest = int(np.argmax(counts[1:]) + 1)
            offset = np.asarray([s.start for s in box])
            main_points = np.argwhere(cc == largest) + offset
            physical = main_points * spacing
            main_trees[source] = cKDTree(physical)
            main_centres[source] = physical.mean(axis=0)
        anchor_interval = float(np.median([np.linalg.norm(main_centres[source + 1] - main_centres[source]) for source in range(1, 24) if source in main_centres and source + 1 in main_centres]))
        adjacency_distance = float(np.linalg.norm(spacing))
        for source in range(1, 25):
            if source in affected:
                continue
            box = objects[source - 1]
            if box is None:
                continue
            local = raw[box] == source
            cc = cc3d.connected_components(local, connectivity=26)
            counts = np.bincount(cc.ravel())
            if len(counts) <= 2:
                continue
            largest = int(np.argmax(counts[1:]) + 1)
            offset = np.asarray([s.start for s in box])
            total = int(np.sum(local))
            for component in range(1, len(counts)):
                if component == largest or counts[component] < 100:
                    continue
                points = np.argwhere(cc == component) + offset
                physical = points * spacing
                centre = physical.mean(axis=0)
                source_distance = float(main_trees[source].query(physical, k=1, workers=1)[0].min())
                source_z_error = float(abs(centre[2] - main_centres[source][2]) / anchor_interval)
                best = None
                for target in (source - 1, source + 1):
                    if not 1 <= target <= 24 or target not in main_trees:
                        continue
                    target_distance = float(main_trees[target].query(physical, k=1, workers=1)[0].min())
                    target_z_error = float(abs(centre[2] - main_centres[target][2]) / anchor_interval)
                    evidence = {'target': target, 'target_distance_mm': target_distance, 'target_z_error_intervals': target_z_error, 'distance_advantage_mm': source_distance - target_distance, 'z_advantage_intervals': source_z_error - target_z_error}
                    if best is None or (target_z_error, target_distance) < (best['target_z_error_intervals'], best['target_distance_mm']):
                        best = evidence
                volume_mm3 = float(len(points) * np.prod(spacing))
                fraction = float(len(points) / total)
                bone_fraction = float(np.mean(ct[tuple(points.T)] > 100.0))
                accept = bool(best and volume_mm3 >= 50.0 and (0.002 <= fraction <= 0.2) and (bone_fraction >= 0.6) and (best['target_z_error_intervals'] <= 0.55) and (best['z_advantage_intervals'] >= 0.5) and (best['target_distance_mm'] <= adjacency_distance) and (best['distance_advantage_mm'] >= 0.5))
                if accept:
                    target = int(best['target'])
                    candidate[tuple(points.T)] = target
                    operations.append({'action': 'EVIDENCE_GATED_COMPONENT_REASSIGN', 'source': LEVELS[source], 'target': LEVELS[target], 'voxels': int(len(points)), 'volume_mm3': volume_mm3, 'source_fraction': fraction, 'bone_fraction_hu_gt_100': bone_fraction, **{key: value for key, value in best.items() if key != 'target'}})
    operations.extend(strict_tiny_remote_cleanup(raw, candidate, spacing, tree, seed_labels))
    added_by_growth = ct_guided_boundary_growth(candidate, ct, spacing, boundary_radius_mm, boundary_minimum_hu)
    operations.append({'action': 'CT_GUIDED_BOUNDARY_GROWTH', 'radius_mm': boundary_radius_mm, 'minimum_hu': boundary_minimum_hu, 'voxels_added': added_by_growth})
    out_dir.mkdir(parents=True, exist_ok=True)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(candidate, image.affine, header), out_dir / 'combined_labels.nii.gz')
    seg_dir = out_dir / 'segmentations'
    seg_dir.mkdir(exist_ok=True)
    for label in range(1, 25):
        label_header = image.header.copy()
        label_header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image((candidate == label).astype(np.uint8), image.affine, label_header), seg_dir / f'vertebrae_{LEVELS[label]}.nii.gz')
    changed = raw != candidate
    return {'raw': str(raw_path), 'ct': str(ct_path), 'identity': diagnostics, 'changed_voxels': int(np.sum(changed)), 'removed_voxels': int(np.sum((raw > 0) & (candidate == 0))), 'relabelled_voxels': int(np.sum((raw > 0) & (candidate > 0) & changed)), 'added_voxels': int(np.sum((raw == 0) & (candidate > 0))), 'operations': operations}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--ct-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary-radius-mm', type=float, default=0.8)
    parser.add_argument('--boundary-minimum-hu', type=float, default=250.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'method': 'Phase19: core identity, supplemental radial CT evidence, conservative abstention, and CT boundary recovery', 'boundary_radius_mm': args.boundary_radius_mm, 'boundary_minimum_hu': args.boundary_minimum_hu, 'cases': {}}
    for case_dir in sorted((path for path in args.raw_root.iterdir() if path.is_dir())):
        case = case_dir.name
        report['cases'][case] = build_case(case_dir / 'combined_labels.nii.gz', args.ct_root / case / 'ct.nii.gz', args.output / case, args.boundary_radius_mm, args.boundary_minimum_hu)
        summary = report['cases'][case]
        print(case, summary['identity']['status'], {'changed': summary['changed_voxels'], 'removed': summary['removed_voxels'], 'relabelled': summary['relabelled_voxels'], 'added': summary['added_voxels']}, flush=True)
    (args.output / 'METHOD_MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
if __name__ == '__main__':
    main()
