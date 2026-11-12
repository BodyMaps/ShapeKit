"""BodyMaps vertebrae warm-up: reproducible automatic postprocessing.

Prepared for Keyu Kuang, China Medical University. AI-assisted implementation.
components: conservative removal of small remote same-identity components.
rooted: ordered body cores, CT-rooted optimization and posterior-unit consistency.
cache: optional official-model probability retention, only with --allow-inference.

No manual landmarks or case-ID branches are used. Outputs never replace inputs.
No clinical accuracy, Dice improvement or independent generalization is asserted.
"""
import argparse
import copy
import gc
import gzip
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np
from scipy import ndimage, sparse
from scipy.special import expit, logsumexp
from scipy.sparse.csgraph import dijkstra, maximum_flow, breadth_first_order
CONNECTED=np.ones((3,3,3),dtype=np.uint8)
NAMES=['background']+[f'vertebrae_L{i}' for i in range(5,0,-1)]+[f'vertebrae_T{i}' for i in range(12,0,-1)]+[f'vertebrae_C{i}' for i in range(7,0,-1)]
CT_OVERRIDE=None

def resolve_ct(cache):
    return CT_OVERRIDE if CT_OVERRIDE is not None else Path(cache['batch']['image_meta_dict']['filename_or_obj'][0])


def sequence_labels(histograms):
    """Monotone DP, optional skips; endpoint model anchors are explicitly provisional."""
    h = np.asarray(histograms, dtype=float)
    n = len(h)
    if n < 3 or n > 24:
        raise ValueError('Need 3--24 plausible body cores; count is not forced.')
    prob = (h + 0.25) / (h.sum(axis=1, keepdims=True) + 6.0)
    emission = -2 * np.log(prob)
    anchors = []
    for i in (0, n - 1):
        dominant = int(h[i].argmax())
        purity = float(h[i, dominant] / max(h[i].sum(), 1))
        if purity >= 0.85:
            emission[i, np.arange(24) != dominant] = np.inf
            anchors.append({'candidate': i + 1, 'model_label': dominant + 1, 'purity': purity})
    cost = np.full((n, 24), np.inf)
    prev = np.full((n, 24), -1, dtype=int)
    cost[0] = emission[0]
    for i in range(1, n):
        for j in range(i, 24):
            transitions = cost[i - 1, :j] + 2.5 * (j - np.arange(j) - 1)
            k = int(np.argmin(transitions))
            cost[i, j] = transitions[k] + emission[i, j]
            prev[i, j] = k
    end = int(np.argmin(cost[-1]))
    if not np.isfinite(cost[-1, end]):
        raise ValueError('Body-core count/order conflicts with provisional endpoint labels; preserve input for review.')
    labels = [end + 1]
    for i in range(n - 1, 0, -1):
        end = int(prev[i, end])
        labels.append(end + 1)
    labels = labels[::-1]
    return (labels, {'method': 'strict superior ordering with skip penalty; not independent anatomical numbering', 'endpoint_anchors_from_model': anchors, 'cost': float(np.min(cost[-1])), 'skipped_internal_labels': [i for i in range(labels[0], labels[-1] + 1) if i not in labels], 'confidence': 'No calibrated confidence or verified vertebral count is available.'})

def extract_cores(mask, prediction, spacing, radius_mm=5.0):
    distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    (parts, n) = ndimage.label(distance >= radius_mm, structure=CONNECTED)
    sizes = np.bincount(parts.ravel())
    boxes = ndimage.find_objects(parts)
    rows = []
    rejected = []
    for cid in range(1, n + 1):
        volume = float(sizes[cid] * np.prod(spacing))
        if volume < 1000:
            continue
        box = boxes[cid - 1]
        inside = parts[box] == cid
        peak = np.array(np.unravel_index(np.argmax(np.where(inside, distance[box], -1)), inside.shape))
        peak += np.array([s.start for s in box])
        extent = np.array([s.stop - s.start for s in box]) * spacing
        hist = np.bincount(prediction[box][inside], minlength=25)[1:25]
        body_fraction = float(hist[:17].sum() / max(hist.sum(), 1))
        reasons = []
        if not 8 <= extent[2] <= 45:
            reasons.append('unusual_superior_extent')
        if body_fraction < 0.8:
            reasons.append('mainly_cervical_or_other_label')
        if extent[1] > 65 or extent[0] > 65:
            reasons.append('possibly_merged_core')
        item = {'component': cid, 'peak': peak.tolist(), 'volume_mm3': volume, 'extent_mm': extent.tolist(), 'histogram': hist.tolist(), 'body_fraction': body_fraction}
        if reasons:
            rejected.append(dict(item, reasons=reasons))
        else:
            rows.append(item)
    rows.sort(key=lambda r: r['peak'][2])
    for (a, b) in zip(rows, rows[1:]):
        gap = (b['peak'][2] - a['peak'][2]) * spacing[2]
        if gap < 12 or gap > 65:
            raise ValueError(f'Implausible body-core spacing ({gap:.1f} mm); no forced relabeling.')
    return (rows, parts, rejected)

def select_automatic_cores(mask, prediction, spacing):
    """Compare erosion scales using spacing and model/order consistency only."""
    trials = []
    best = None
    for radius in (3.0, 4.0, 5.0, 6.0):
        try:
            (rows, parts, rejected) = extract_cores(mask, prediction, spacing, radius)
            (assigned, numbering) = sequence_labels([r['histogram'] for r in rows])
            if any((label > 17 for label in assigned)):
                raise ValueError('Core ordering crosses cervical identities.')
            impurity = sum((1.0 - r['histogram'][label - 1] / max(sum(r['histogram']), 1) for (r, label) in zip(rows, assigned)))
            skips = len(numbering['skipped_internal_labels'])
            score = 3.0 * len(rows) - 4.0 * skips - 2.0 * impurity
            trials.append({'radius_mm': radius, 'cores': len(rows), 'score': score, 'assigned_labels': assigned, 'status': 'eligible'})
            if best is None or score > best[0]:
                best = (score, rows, parts, rejected, assigned, numbering, radius)
        except ValueError as exc:
            trials.append({'radius_mm': radius, 'status': 'rejected', 'reason': str(exc)})
        print(f'Automatic core scale {radius:g} mm assessed.', flush=True)
    if best is None:
        raise ValueError('All automatic core scales rejected: ' + json.dumps(trials))
    (_, rows, parts, rejected, assigned, numbering, radius) = best
    numbering['automatic_scale_selection'] = {'selected_radius_mm': radius, 'trials': trials, 'criterion': 'coverage and model/order consistency; not calibrated accuracy'}
    return (rows, parts, rejected, assigned, numbering)

def rooted_valid(labels, parents, markers):
    for label in np.unique(labels):
        if label == 0:
            continue
        selected = (labels == label) & (parents[label] >= 0)
        if np.any(labels[parents[label, selected]] != label):
            return False
    return bool(np.all(labels[markers > 0] == markers[markers > 0]))

def rooted_move(labels, unary, u, v, w, parents, markers, eligible, alpha):
    """Binary alpha expansion preserving predecessor closure of every label."""
    active = np.asarray(eligible, bool) & (markers == 0) & (labels != alpha)
    count = int(active.sum())
    if not count:
        return (labels.copy(), {'alpha': alpha, 'changed': 0})
    ids = np.full(len(labels), -1, dtype=np.int32)
    ids[active] = np.arange(count, dtype=np.int32)
    pos = np.flatnonzero(active)
    old = labels[active]
    d0 = unary[old, pos].astype(np.int64).copy()
    d1 = unary[alpha, pos].astype(np.int64).copy()
    (au, av) = (active[u], active[v])
    inside = au & av
    (iu, iv, iw) = (ids[u[inside]], ids[v[inside]], w[inside])
    for (a, b, chosen) in ((u, v, au & ~av), (v, u, av & ~au)):
        node = ids[a[chosen]]
        outside = labels[b[chosen]]
        ew = w[chosen]
        np.add.at(d0, node, ew * (labels[a[chosen]] != outside))
        np.add.at(d1, node, ew * (alpha != outside))
    (data0, data1) = (d0.copy(), d1.copy())
    e00 = iw * (old[iu] != old[iv])
    e01 = iw * (old[iu] != alpha)
    e10 = iw * (alpha != old[iv])
    cap = e01 + e10 - e00
    if np.any(cap < 0):
        raise RuntimeError('Invalid expansion energy.')
    np.add.at(d1, iu, e10 - e00)
    np.add.at(d1, iv, -e10)
    shift = np.minimum(d0, d1)
    d0 -= shift
    d1 -= shift
    hard = int(d0.sum() + d1.sum() + cap.sum() + 1)
    if hard >= np.iinfo(np.int32).max:
        raise ValueError('Root-constraint capacity overflow.')
    cu = []
    cv = []
    p = parents[alpha, pos]
    valid = p >= 0
    child = pos[valid]
    parent = p[valid]
    pair = active[parent]
    cu.append(ids[parent[pair]])
    cv.append(ids[child[pair]])
    deny = child[~pair & (labels[parent] != alpha)]
    d1[ids[deny]] += hard
    d1[ids[pos[~valid]]] += hard
    for label in np.unique(labels):
        if label in (0, alpha):
            continue
        child = np.flatnonzero((labels == label) & (parents[label] >= 0))
        parent = parents[label, child]
        if np.any(labels[parent] != label):
            raise RuntimeError('Initialization violates rooted closure.')
        pair = active[child] & active[parent]
        cu.append(ids[child[pair]])
        cv.append(ids[parent[pair]])
        deny_parent = parent[~active[child] & active[parent]]
        d1[np.unique(ids[deny_parent])] += hard
    ru = np.concatenate(cu)
    rv = np.concatenate(cv)
    (source, sink) = (count, count + 1)
    nodes = np.arange(count, dtype=np.int32)
    graph = sparse.coo_matrix((np.concatenate((cap, np.full(len(ru), hard, dtype=np.int64), d1, d0)), (np.concatenate((iu, ru, np.full(count, source, dtype=np.int32), nodes)), np.concatenate((iv, rv, nodes, np.full(count, sink, dtype=np.int32))))), shape=(count + 2, count + 2)).tocsr()
    if graph.data.max(initial=0) >= np.iinfo(np.int32).max:
        raise ValueError('Aggregated capacity overflow.')
    graph.eliminate_zeros()
    flow = maximum_flow(graph, source, sink, method='dinic')
    residual = graph - flow.flow
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    reach = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    keep = np.zeros(count + 2, bool)
    keep[reach] = True
    keep = keep[:count]
    proposed = np.where(keep, old, alpha).astype(labels.dtype)
    before = int(data0.sum() + e00.sum())
    after = int(np.where(keep, data0, data1).sum() + iw[proposed[iu] != proposed[iv]].sum())
    result = labels.copy()
    if after < before:
        result[active] = proposed
    if not rooted_valid(result, parents, markers):
        raise RuntimeError('Rooted closure violation.')
    if after > before:
        raise RuntimeError('Energy increased despite feasible initial assignment.')
    return (result, {'alpha': int(alpha), 'nodes': count, 'changed': int(np.count_nonzero(result != labels)), 'before': before, 'after': after})

def remote_label_fragments(baseline, spacing, typical_gap_mm, maximum_label):
    """Detect small remote identity islands; retain nearby detached anatomy.

    This is a spatial identity check, not a bone removal operation. Only islands
    below 10% of their same-label principal component and more than two typical
    vertebral spacings away are eligible for re-identification.
    """
    remote = np.zeros(baseline.shape, dtype=bool)
    events = []
    for label in range(maximum_label + 1, int(baseline.max()) + 1):
        (parts, n) = ndimage.label(baseline == label, structure=np.ones((3, 3, 3)))
        if n < 2:
            continue
        sizes = np.bincount(parts.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        distance = ndimage.distance_transform_edt(parts != largest, sampling=spacing)
        for component in range(1, n + 1):
            if component == largest or sizes[component] > 0.1 * sizes[largest]:
                continue
            selected = parts == component
            separation = float(distance[selected].min())
            if separation <= 2 * typical_gap_mm:
                continue
            remote[selected] = True
            events.append({'original_label': label, 'component': component, 'voxels': int(sizes[component]), 'distance_to_same_label_main_component_mm': separation})
    return (remote, events)

def prepare(cache_path, output, repair_remote_labels=False, neighbors=6, ct_strength=4):
    import torch
    d = torch.load(str(cache_path), map_location='cpu')
    bounds = np.array(d['bounds'])
    box = tuple((slice(a, b) for (a, b) in bounds))
    baseline = d['baseline_full'].numpy()[box].copy()
    spacing = np.array(d['spacing'])
    (parts, n) = ndimage.label(baseline > 0, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(parts.ravel())
    sizes[0] = 0
    main = parts == sizes.argmax()
    del parts, sizes
    (rows, cores, rejected, assigned, numbering) = select_automatic_cores(main, baseline, spacing)
    markers = np.zeros_like(baseline, dtype=np.uint8)
    for (row, label) in zip(rows, assigned):
        markers[cores == row['component']] = label
    del cores
    (z0, z1) = (rows[0]['peak'][2], rows[-1]['peak'][2])
    lo = max(0, int(z0 - (rows[1]['peak'][2] - z0) / 2))
    hi = min(main.shape[2], int(z1 + (z1 - rows[-2]['peak'][2]) / 2 + 1))
    remote = np.zeros_like(main)
    remote_events = []
    if repair_remote_labels:
        typical_gap = float(np.median(np.diff([r['peak'][2] for r in rows])) * spacing[2])
        (remote, remote_events) = remote_label_fragments(baseline, spacing, typical_gap, max(assigned))
        print(f'Remote identity islands eligible for reassignment: {int(remote.sum())} voxels.', flush=True)
    domain = main & ((baseline <= max(assigned)) | remote)
    domain[:, :, :lo] = False
    domain[:, :, hi:] = False
    if np.any((markers > 0) & ~domain):
        raise ValueError('Automatic body markers outside domain.')
    meta = d['batch']['image_meta_dict']
    native_aff = meta['original_affine'][0].numpy()
    grid_aff = meta['affine'][0].numpy().copy()
    start = d['batch']['foreground_start_coord'][0].numpy() + bounds[:, 0]
    grid_aff[:3, 3] += grid_aff[:3, :3] @ start
    case = d['batch']['name_img'][0]
    ct_path = resolve_ct(d)
    ct_image = nib.load(str(ct_path))
    mapping = np.linalg.inv(native_aff) @ grid_aff
    corners = np.array(list(__import__('itertools').product(*[(0, s - 1) for s in baseline.shape])))
    native_corners = corners @ mapping[:3, :3].T + mapping[:3, 3]
    origin = np.maximum(0, np.floor(native_corners.min(axis=0)).astype(int) - 2)
    end = np.minimum(ct_image.shape, np.ceil(native_corners.max(axis=0)).astype(int) + 3)
    crop = tuple((slice(int(a), int(b)) for (a, b) in zip(origin, end)))
    native_ct = np.asarray(ct_image.dataobj[crop], dtype=np.float32)
    ct = ndimage.affine_transform(native_ct, mapping[:3, :3], offset=mapping[:3, 3] - origin, output_shape=baseline.shape, order=1, mode='constant', cval=-1000, prefilter=False)
    del native_ct
    ids = np.full(domain.shape, -1, dtype=np.int32)
    ids[domain] = np.arange(domain.sum(), dtype=np.int32)
    cost = 1 + ct_strength * expit((200 - ct) / 80)
    us = []
    vs = []
    lengths = []
    surfaces = []
    for axis in range(3):
        a = [slice(None)] * 3
        b = a.copy()
        a[axis] = slice(None, -1)
        b[axis] = slice(1, None)
        (a, b) = (tuple(a), tuple(b))
        valid = domain[a] & domain[b]
        us.append(ids[a][valid])
        vs.append(ids[b][valid])
        lengths.append(0.5 * (cost[a][valid] + cost[b][valid]) * spacing[axis])
        area = float(np.prod(spacing) / spacing[axis])
        (hu_a, hu_b) = (ct[a][valid], ct[b][valid])
        surfaces.append(area * (0.05 + 0.95 * expit((np.minimum(hu_a, hu_b) - 150) / 80)) * np.exp(-np.abs(hu_a - hu_b) / 400))
    (u, v, length, surface) = map(np.concatenate, (us, vs, lengths, surfaces))
    del ids, cost
    n = int(domain.sum())
    if neighbors == 6:
        graph = sparse.coo_matrix((length, (u, v)), shape=(n, n)).tocsr()
    elif neighbors == 26:
        import itertools
        ids = np.full(domain.shape, -1, dtype=np.int32)
        ids[domain] = np.arange(n, dtype=np.int32)
        pu = []
        pv = []
        pw = []
        for delta in itertools.product((-1, 0, 1), repeat=3):
            if delta <= (0, 0, 0):
                continue
            a = []
            b = []
            for (shift, size) in zip(delta, domain.shape):
                a.append(slice(max(0, -shift), min(size, size - shift)))
                b.append(slice(max(0, shift), min(size, size + shift)))
            (a, b) = (tuple(a), tuple(b))
            valid = domain[a] & domain[b]
            pu.append(ids[a][valid])
            pv.append(ids[b][valid])
            hu = np.minimum(ct[a][valid], ct[b][valid])
            pw.append(np.linalg.norm(np.array(delta) * spacing) * (1 + ct_strength * expit((200 - hu) / 80)))
        graph = sparse.coo_matrix((np.concatenate(pw), (np.concatenate(pu), np.concatenate(pv))), shape=(n, n)).tocsr()
        del pu, pv, pw, ids
    else:
        raise ValueError('Path neighbors must be 6 or 26.')
    node_markers = markers[domain]
    seeds = np.flatnonzero(node_markers)
    (dist, global_parent, source) = dijkstra(graph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    if not np.isfinite(dist).all():
        raise ValueError('Foreground contains unseeded disconnected regions.')
    initial = node_markers[source]
    k = max(assigned)
    parents = np.full((k + 1, n), -9999, dtype=np.int32)
    distances = np.full((k + 1, n), np.inf, dtype=np.float32)
    confusion = np.zeros((k + 1, 25), dtype=np.float32)
    probabilities = d['logits'].numpy().astype(np.float32)
    probabilities = np.exp(probabilities - logsumexp(probabilities, axis=0, keepdims=True))
    for label in assigned:
        print(f'Automatic body-root paths: {label}/{k}', flush=True)
        (dd, pp, _) = dijkstra(graph, directed=False, indices=np.flatnonzero(node_markers == label), min_only=True, return_predecessors=True)
        parents[label] = pp
        distances[label] = dd
        owned = initial == label
        parents[label, owned] = global_parent[owned]
        confusion[label] = probabilities[:, markers == label].mean(axis=1)
    if not rooted_valid(initial, parents, node_markers):
        raise RuntimeError('Rooted initialization invalid.')
    scores = confusion @ probabilities[:, domain]
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'prepared.npz', u=u, v=v, surface=surface, distances=distances, parents=parents, markers=node_markers, initial=initial, model_scores=scores, domain=domain, baseline=baseline, ct=ct, grid_affine=grid_aff)
    report = {'cache': str(cache_path), 'case': case, 'numbering': numbering, 'body_rows': [dict(r, assigned_label=int(l)) for (r, l) in zip(rows, assigned)], 'manual_points_used': False, 'anatomical_numbering_validated': False, 'path_neighbors': neighbors, 'ct_path_strength': ct_strength, 'remote_identity_repair': repair_remote_labels, 'remote_identity_events': remote_events}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Prepared automatic rooted problem: {output}', flush=True)

def run(prepared, output, smoothness=10, model_weight=0.25, shape_weight=0.25, passes=3):
    if output.exists():
        raise ValueError('Output exists; use a new directory.')
    d = np.load(prepared / 'prepared.npz')
    labels = d['initial'].copy()
    markers = d['markers']
    parents = d['parents']
    distances = d['distances']
    nearest = distances[1:].min(axis=0)
    scores = d['model_scores']
    (u, v) = (d['u'], d['v'])
    unary = -model_weight * np.log(np.maximum(scores, 1e-07))
    unary += shape_weight * np.minimum((distances - nearest[None, :]) / 40, 20)
    unary[0] = 100
    unary = np.rint(np.clip(unary, 0, 100) * 10).astype(np.int32)
    w = np.maximum(1, np.rint(smoothness * d['surface'] * 10)).astype(np.int64)
    history = []
    for iteration in range(1, passes + 1):
        changed = 0
        for alpha in range(1, len(parents)):
            eligible = (distances[alpha] - nearest <= 100) & np.isfinite(distances[alpha])
            print(f'Root-constrained pass {iteration}, label {alpha}', flush=True)
            (labels, entry) = rooted_move(labels, unary, u, v, w, parents, markers, eligible, alpha)
            entry['pass'] = iteration
            history.append(entry)
            changed += entry['changed']
            print(entry, flush=True)
        if changed == 0:
            break
    result = d['baseline'].copy()
    result[d['domain']] = labels
    output.mkdir(parents=True)
    ref = nib.Nifti1Image(result, d['grid_affine'])
    ref.header.set_xyzt_units('mm')
    nib.save(ref, str(output / 'candidate_model_grid.nii.gz'))
    np.save(output / 'labels.npy', result)
    report = {'status': 'experimental_requires_anatomical_review', 'smoothness': smoothness, 'model_weight': model_weight, 'shape_weight': shape_weight, 'passes': passes, 'manual_points_used': False, 'rooted_closure_valid': rooted_valid(labels, parents, markers), 'foreground_changed': False, 'history': history}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved rooted candidate: {output}', flush=True)

def extract_posterior_units(d, candidate, halfwidth_mm=4.5, posterior_depth_mm=20, threshold_hu=120):
    markers = np.zeros(candidate.shape, dtype=np.uint8)
    markers[d['domain']] = d['markers']
    centers = []
    for label in np.unique(markers):
        if label == 0:
            continue
        xyz = np.argwhere(markers == label).mean(axis=0)
        centers.append([*xyz, label])
    centers = np.array(centers)
    centers = centers[np.argsort(centers[:, 2])]
    (xx, yy, zz) = np.ogrid[:candidate.shape[0], :candidate.shape[1], :candidate.shape[2]]
    cx = np.interp(np.arange(candidate.shape[2]), centers[:, 2], centers[:, 0])
    cy = np.interp(np.arange(candidate.shape[2]), centers[:, 2], centers[:, 1])
    spacing = np.linalg.norm(d['grid_affine'][:3, :3], axis=0)
    depth = (cy[None, None, :] - yy) * spacing[1]
    mask = (candidate > 0) & (candidate <= 17) & (np.abs(xx - cx[None, None, :]) * spacing[0] <= halfwidth_mm)
    mask &= (depth >= posterior_depth_mm) & (d['ct'] >= threshold_hu)
    (units, n) = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(units.ravel())
    rows = []
    for component in range(1, n + 1):
        if sizes[component] < 10:
            continue
        pos = np.argwhere(units == component)
        labels = candidate[tuple(pos.T)]
        relative_depth = (cy[pos[:, 2]] - pos[:, 1]) * spacing[1]
        anterior = relative_depth <= np.percentile(relative_depth, 20)
        superior = pos[:, 2] >= np.percentile(pos[:, 2], 80)
        proximal = anterior & superior
        if not proximal.any():
            proximal = superior

        def hist(chosen):
            return {int(k): int(v) for (k, v) in zip(*np.unique(labels[chosen], return_counts=True))}
        rows.append({'component': component, 'voxels': int(sizes[component]), 'min': pos.min(axis=0).tolist(), 'max': pos.max(axis=0).tolist(), 'all_labels': hist(np.ones(len(labels), bool)), 'anterior_labels': hist(anterior), 'superior_labels': hist(superior), 'proximal_labels': hist(proximal)})
    return (units, rows)

def dominant(hist):
    if not hist:
        return (0, 0.0)
    label = max(hist, key=hist.get)
    return (int(label), float(hist[label] / sum(hist.values())))

def eligible(row, typical_gap_mm, spacing_z):
    (label, purity) = dominant(row['proximal_labels'])
    (anterior, front_purity) = dominant(row['anterior_labels'])
    extent = (row['max'][2] - row['min'][2] + 1) * spacing_z
    values = list(row['all_labels'].values())
    mixed = len(values) > 1 and min(values) >= 10 and (min(values) / sum(values) >= 0.05)
    adjacent = max(row['all_labels']) - min(row['all_labels']) <= 1
    return (mixed and adjacent and (purity >= 0.9) and (anterior == label) and (front_purity >= 0.65) and (sum(row['proximal_labels'].values()) >= 8) and (row['voxels'] >= 30) and (extent <= 1.7 * typical_gap_mm), label)

def augment(prepared, candidate, output):
    if output.exists():
        raise FileExistsError(output)
    with np.load(prepared / 'prepared.npz') as loaded:
        d = {k: loaded[k] for k in loaded.files}
    original_markers = d['markers'].copy()
    markers = original_markers.copy()
    labels = np.load(candidate / 'labels.npy')
    spacing = np.linalg.norm(d['grid_affine'][:3, :3], axis=0)
    report = json.loads((prepared / 'preparation.json').read_text())
    typical_gap = float(np.median(np.diff([r['peak'][2] for r in report['body_rows']])) * spacing[2])
    (broad, broad_rows) = extract_posterior_units(d, labels, 2.25, 25, 180)
    (narrow, narrow_rows) = extract_posterior_units(d, labels, 1.5, 25, 200)
    narrow_index = {r['component']: r for r in narrow_rows}
    ids = np.full(labels.shape, -1, dtype=np.int32)
    ids[d['domain']] = np.arange(len(markers))
    anchor_map = np.zeros_like(labels)
    events = []
    for row in broad_rows:
        (accept, target) = eligible(row, typical_gap, spacing[2])
        if not accept:
            continue
        selected = broad == row['component']
        (components, counts) = np.unique(narrow[selected], return_counts=True)
        chosen = [(c, n) for (c, n) in zip(components, counts) if c > 0 and c in narrow_index]
        agreement = False
        for (component, count) in chosen:
            small = narrow_index[component]
            (ok, small_target) = eligible(small, typical_gap, spacing[2])
            if ok and small_target == target and (count >= 0.5 * small['voxels']):
                agreement = True
        event = dict(row, automatic_target=target, narrow_slab_agreement=agreement)
        if not agreement:
            events.append(dict(event, status='rejected_width_or_HU_instability'))
            continue
        nodes = ids[selected]
        nodes = nodes[nodes >= 0]
        if len(nodes) != int(selected.sum()):
            events.append(dict(event, status='rejected_outside_body_domain'))
            continue
        chain = np.zeros(len(markers), dtype=bool)
        front = np.unique(nodes)
        for _ in range(len(markers)):
            front = front[(front >= 0) & ~chain[np.maximum(front, 0)]]
            if not len(front):
                break
            chain[front] = True
            front = np.unique(d['parents'][target, front])
        else:
            raise RuntimeError('Cycle in body-root predecessor field.')
        conflict = chain & (markers > 0) & (markers != target)
        if conflict.any():
            events.append(dict(event, status='rejected_conflicting_body_or_unit_root'))
            continue
        if not np.any(chain & (original_markers == target)):
            events.append(dict(event, status='rejected_no_original_body_connection'))
            continue
        markers[chain] = target
        anchor_map[selected] = target
        events.append(dict(event, status='accepted_automatic_posterior_unit', unit_seed_voxels=int(selected.sum()), root_path_voxels=int(chain.sum())))
    accepted = [e for e in events if e['status'] == 'accepted_automatic_posterior_unit']
    print(json.dumps(events, indent=2), flush=True)
    if not accepted:
        print('No stable mixed posterior units found; preserving automatic body-root problem.', flush=True)
        output.mkdir(parents=True)
        np.savez(output / 'prepared.npz', **d)
        report['automatic_posterior_unit_events'] = events
        (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return events
    positions = np.argwhere(d['domain'])
    ct = d['ct'][d['domain']]
    cost = 1 + report.get('ct_path_strength', 4) / (1 + np.exp(np.clip((ct - 200) / 80, -80, 80)))
    (u, v) = (d['u'], d['v'])
    step = np.linalg.norm((positions[u] - positions[v]) * spacing, axis=1)
    graph = sparse.coo_matrix((step * 0.5 * (cost[u] + cost[v]), (u, v)), shape=(len(markers), len(markers))).tocsr()
    (dist, global_parent, source) = dijkstra(graph, directed=False, indices=np.flatnonzero(markers), min_only=True, return_predecessors=True)
    if not np.isfinite(dist).all():
        raise ValueError('New initialization contains unreachable regions.')
    initial = markers[source]
    parents = np.full_like(d['parents'], -9999)
    distances = np.full_like(d['distances'], np.inf)
    for label in range(1, len(parents)):
        print(f'Posterior-unit body-root paths: {label}/{len(parents) - 1}', flush=True)
        (dd, pp, _) = dijkstra(graph, directed=False, indices=np.flatnonzero(markers == label), min_only=True, return_predecessors=True)
        parents[label] = pp
        distances[label] = dd
        owned = initial == label
        parents[label, owned] = global_parent[owned]
    if not rooted_valid(initial, parents, markers):
        raise RuntimeError('Posterior-unit initialization invalid.')
    d.update(markers=markers, parents=parents, distances=distances, initial=initial)
    output.mkdir(parents=True)
    np.savez(output / 'prepared.npz', **d)
    nib.save(nib.Nifti1Image(anchor_map, d['grid_affine']), str(output / 'automatic_posterior_units.nii.gz'))
    report['automatic_posterior_unit_events'] = events
    report['manual_points_used'] = False
    (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return events

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def official_modules(repo):
    sys.path.insert(0, str(repo))
    from dataset.dataloader_test import get_loader
    from model.SwinUNETR_target import SwinUNETR
    from utils.utils import invert_transform
    return (get_loader, SwinUNETR, invert_transform)

def loader_options(data_root):
    return SimpleNamespace(data_root_path=str(data_root), original_label=False, space_x=1.5, space_y=1.5, space_z=1.5, a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, roi_x=96, roi_y=96, roi_z=96, num_samples=1, phase='test', cache_dataset=False, cache_rate=0.0, batch_size=1, num_workers=0, dist=False)

def cache_inference(args, case, get_loader, Model):
    import torch
    from monai.inferers import sliding_window_inference
    cache = args.cache_root / f'{case}.pt'
    manifest_path = args.cache_root / f'{case}.json'
    ct_path = args.data_root / case / 'ct.nii.gz'
    identity = {'ct_sha256': digest(ct_path), 'checkpoint_sha256': digest(args.checkpoint), 'model_spacing_mm': [1.5] * 3, 'overlap': 0.75, 'logits_dtype': 'float32', 'schema': 2}
    if cache.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != identity:
            raise ValueError('Probability cache identity mismatch; choose a new cache directory.')
        print(f'Reusing probability cache: {cache}', flush=True)
        return cache
    if args.cache_only:
        raise ValueError('No compatible cache exists; inference is required first.')
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; activate the suprem environment.')
    (test_loader, _) = get_loader(loader_options(args.data_root))
    test_loader.dataset.data = [d for d in test_loader.dataset.data if d['name_img'] == case]
    if len(test_loader.dataset.data) != 1:
        raise ValueError('Case was not found exactly once.')
    model = Model(img_size=(96, 96, 96), in_channels=1, out_channels=25, feature_size=48, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False)
    state = torch.load(str(args.checkpoint), map_location='cpu')['net']
    expected = model.state_dict()
    mapped = {}
    for (key, value) in state.items():
        candidates = (key, '.'.join(key.split('.')[1:]))
        for candidate in candidates:
            if candidate in expected:
                if candidate in mapped:
                    raise ValueError('Duplicate checkpoint key mapping.')
                mapped[candidate] = value
                break
    if set(mapped) != set(expected):
        raise ValueError('Checkpoint did not match all model parameters.')
    model.load_state_dict(mapped, strict=True)
    del state, mapped, expected
    model = model.cuda().eval()
    for batch in test_loader:
        print(f'Official model inference with probability retention: {case}', flush=True)
        with torch.no_grad():
            logits = sliding_window_inference(batch['image'].cuda(), (96, 96, 96), 1, model, overlap=0.75, mode='gaussian', sw_device='cuda', device='cpu')
        raw = logits[0].argmax(dim=0).numpy().astype(np.uint8)
        (parts, count) = ndimage.label(raw > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        sizes = np.bincount(parts.ravel())
        sizes[0] = 0
        if count == 0:
            raise ValueError('The model produced no vertebral foreground.')
        main = int(sizes.argmax())
        main_box = ndimage.find_objects(parts)[main - 1]
        del parts, sizes
        box = tuple((slice(max(0, s.start - 8), min(n, s.stop + 8)) for (s, n) in zip(main_box, raw.shape)))
        if np.prod([s.stop - s.start for s in box]) > 12000000:
            raise ValueError('Probability crop exceeds memory guard.')
        crop_logits = logits[(0, slice(None)) + box].clone()
        del logits
        data = {'batch': batch, 'shape': list(raw.shape), 'baseline_full': torch.from_numpy(raw), 'bounds': [(s.start, s.stop) for s in box], 'logits': crop_logits, 'spacing': [1.5] * 3, 'identity': identity}
        args.cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.partial')
        torch.save(data, str(temporary))
        temporary.replace(cache)
        manifest_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return cache

def native_labels(grid, batch, transforms, invert):
    import torch
    working = copy.deepcopy(batch)
    working['refined'] = torch.from_numpy(np.asarray(grid, dtype=np.uint8))[None, None]
    native = invert('refined', working, transforms)[0]['refined'].cpu().numpy()[0]
    del working
    return np.asarray(native, dtype=np.uint8)

def write_native(data, reference, destination):
    if data.shape != reference.shape:
        raise ValueError('Inverse transform returned incorrect native shape.')
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(data, reference.affine, header)
    output.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    output.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(output, str(destination))

def export(prepared, candidate, repo):
    import torch
    report = json.loads((prepared / 'preparation.json').read_text())
    cache = torch.load(report['cache'], map_location='cpu')
    ct_path = resolve_ct(cache)
    ct_hash = digest(ct_path)
    if ct_hash != cache['identity']['ct_sha256']:
        raise ValueError('CT differs from cached model input.')
    destination = candidate / 'combined_labels.nii.gz'
    if destination.exists():
        raise FileExistsError(destination)
    roi = np.load(candidate / 'labels.npy')
    full = cache['baseline_full'].numpy().copy()
    box = tuple((slice(a, b) for (a, b) in cache['bounds']))
    if roi.shape != full[box].shape:
        raise ValueError('Candidate grid shape mismatch.')
    full[box] = roi
    sys.path.insert(0, str(repo))
    from dataset.dataloader_test import get_loader
    from utils.utils import invert_transform
    (_, transforms) = get_loader(loader_options(ct_path.parent.parent))
    print('Restoring native CT grid; no model inference is performed.', flush=True)
    native = native_labels(full, cache['batch'], transforms, invert_transform)
    reference = nib.load(str(ct_path))
    write_native(native, reference, destination)
    restored = nib.load(str(destination))
    if restored.shape != reference.shape or not np.allclose(restored.affine, reference.affine):
        raise ValueError('Native geometry validation failed.')
    if not np.array_equal(restored.get_qform(), reference.get_qform()) or not np.array_equal(restored.get_sform(), reference.get_sform()):
        raise ValueError('Native qform/sform validation failed.')
    verification = {'ct': str(ct_path), 'ct_sha256': ct_hash, 'cache': report['cache'], 'file': str(destination), 'output_sha256': digest(destination), 'shape': list(restored.shape), 'affine_equals_ct': True, 'qform_sform_equal_ct': True, 'manual_points_used_by_algorithm': False, 'anatomical_accuracy_established': False, 'foreground_geometry_corrected': False}
    (candidate / 'native_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2), flush=True)
    return verification


def write_binary_masks(combined_path):
    """Write all 24 masks in native coordinates using bounded-memory slabs."""
    ref=nib.load(str(combined_path));labels=np.asarray(ref.dataobj,dtype=np.uint8)
    destination=combined_path.parent/'segmentations'
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir()
    for label,name in enumerate(NAMES[1:],1):
        header=ref.header.copy();header.set_data_dtype(np.uint8)
        header['vox_offset']=352;header.set_slope_inter(1,0)
        header['cal_min']=0;header['cal_max']=1
        path=destination/(name+'.nii.gz')
        with path.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as stream:
            stream.write(header.binaryblock);stream.write(bytes(4))
            for z in range(0,labels.shape[2],16):
                stream.write((labels[:,:,z:z+16]==label).astype(np.uint8).tobytes(order='F'))
        print('Exported binary mask: '+name,flush=True)


def component_filter(input_path,output_root,selected_labels,max_ratio=.1,min_gap=50):
    ref=nib.load(str(input_path));data=np.asarray(ref.dataobj)
    if data.ndim!=3 or not np.issubdtype(data.dtype,np.integer) or data.min()<0 or data.max()>24:
        raise ValueError('Expected a native integer 0..24 segmentation.')
    axes=ref.affine[:3,:3];spacing=np.linalg.norm(axes,axis=0)
    if not np.allclose((axes/spacing).T@(axes/spacing),np.eye(3),atol=1e-4):
        raise ValueError('Sheared grids require explicit physical resampling.')
    if not 0<=max_ratio<=1 or min_gap<0:raise ValueError('Invalid component thresholds.')
    result=data.astype(np.uint8,copy=True);events=[]
    for label in selected_labels:
        locations=np.where(data==label)
        if not len(locations[0]):continue
        box=tuple(slice(int(v.min()),int(v.max())+1) for v in locations)
        parts,n=ndimage.label(data[box]==label,structure=CONNECTED)
        sizes=np.bincount(parts.ravel());sizes[0]=0;main=int(sizes.argmax())
        distance=ndimage.distance_transform_edt(parts!=main,sampling=spacing)
        for component in range(1,n+1):
            if component==main:continue
            chosen=parts==component;gap=float(distance[chosen].min());ratio=float(sizes[component]/sizes[main])
            remove=ratio<=max_ratio and gap>=min_gap
            events.append(dict(label=label,voxels=int(sizes[component]),ratio=ratio,gap_mm=gap,removed=remove))
            if remove:result[box][chosen]=0
    destination=output_root/input_path.parent.name
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    write_native(result,ref,destination/'combined_labels.nii.gz')
    (destination/'component_report.json').write_text(json.dumps(dict(max_volume_ratio=max_ratio,min_gap_mm=min_gap,
        labels=selected_labels,removed_voxels=int(np.count_nonzero(data!=result)),events=events),indent=2),encoding='utf-8')
    write_binary_masks(destination/'combined_labels.nii.gz')


def cli():
    global CT_OVERRIDE
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('components',help='Automatic filtering of remote component outliers')
    c.add_argument('--input',type=Path,required=True);c.add_argument('--output-root',type=Path,required=True)
    c.add_argument('--labels',nargs='+',type=int,default=list(range(1,25)),choices=range(1,25))
    c.add_argument('--max-volume-ratio',type=float,default=.1);c.add_argument('--min-gap-mm',type=float,default=50)
    r=sub.add_parser('rooted',help='Reproduce accepted body/posterior refinement from an existing trusted probability cache')
    r.add_argument('--cache',type=Path,required=True);r.add_argument('--output-root',type=Path,required=True)
    r.add_argument('--suprem-repo',type=Path,required=True);r.add_argument('--ct',type=Path)
    k=sub.add_parser('cache',help='Optional official-model inference with probability retention')
    k.add_argument('--data-root',type=Path,required=True);k.add_argument('--case-id',required=True)
    k.add_argument('--checkpoint',type=Path,required=True);k.add_argument('--cache-root',type=Path,required=True)
    k.add_argument('--suprem-repo',type=Path,required=True);k.add_argument('--allow-inference',action='store_true')
    args=p.parse_args()
    if args.mode=='components':
        component_filter(args.input,args.output_root,args.labels,args.max_volume_ratio,args.min_gap_mm);return
    if args.mode=='cache':
        if not args.allow_inference:raise ValueError('New inference requires --allow-inference explicitly.')
        args.cache_only=False
        get_loader,Model,_=official_modules(args.suprem_repo)
        path=cache_inference(args,args.case_id,get_loader,Model)
        print(str(path));return
    import torch
    if args.output_root.exists():raise FileExistsError('Use a new output root.')
    cache=torch.load(str(args.cache),map_location='cpu');case=cache['batch']['name_img'][0]
    if Path(case).name!=case or case in ('.','..'):raise ValueError('Invalid case name.')
    CT_OVERRIDE=args.ct
    if digest(resolve_ct(cache))!=cache['identity']['ct_sha256']:raise ValueError('CT/cache identity mismatch.')
    del cache
    args.output_root.mkdir(parents=True)
    body=args.output_root/'automatic_body_references';initial=args.output_root/'initial_candidate'
    posterior=args.output_root/'automatic_posterior_references';destination=args.output_root/case
    prepare(args.cache,body,repair_remote_labels=True)
    run(body,initial);events=augment(body,initial,posterior);run(posterior,destination)
    verification=export(posterior,destination,args.suprem_repo)
    write_binary_masks(destination/'combined_labels.nii.gz')
    summary=dict(status='automatic_processing_complete_anatomical_review_required',case=case,
        native_segmentation=verification['file'],manual_inputs=False,new_inference=False,
        automatic_posterior_units=sum(e['status']=='accepted_automatic_posterior_unit' for e in events),
        foreground_geometry='preserved; identities reassigned',
        independent_accuracy_metrics=None)
    (args.output_root/'workflow_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Completed: '+verification['file'],flush=True)


if __name__=='__main__':cli()
