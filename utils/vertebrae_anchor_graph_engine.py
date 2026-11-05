"""
All corrections are made in a superior-up (RAS) view of the label map:
  1. a vertebra whose main body is out of anatomical order is demoted to a fragment
  2. fragments touching the spine take the label they share most surface with
  3. isolated fragments are kept only if they lie next to a vertebra, otherwise removed
  4. one bone split across two labels  is merged into one label; nothing is renumbered, the gap is reported instead
  5. enclosed holes are filled 
"""

import os
import csv
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
import cc3d
import fastremap
from scipy import ndimage
from tqdm import tqdm

VERTEBRAE = {
    1: 'vertebrae_L5', 2: 'vertebrae_L4', 3: 'vertebrae_L3', 4: 'vertebrae_L2', 5: 'vertebrae_L1',
    6: 'vertebrae_T12', 7: 'vertebrae_T11', 8: 'vertebrae_T10', 9: 'vertebrae_T9', 10: 'vertebrae_T8',
    11: 'vertebrae_T7', 12: 'vertebrae_T6', 13: 'vertebrae_T5', 14: 'vertebrae_T4', 15: 'vertebrae_T3',
    16: 'vertebrae_T2', 17: 'vertebrae_T1', 18: 'vertebrae_C7', 19: 'vertebrae_C6', 20: 'vertebrae_C5',
    21: 'vertebrae_C4', 22: 'vertebrae_C3', 23: 'vertebrae_C2', 24: 'vertebrae_C1',
}
REGION = {k: 'L' if k <= 5 else 'T' if k <= 17 else 'C' for k in VERTEBRAE}
PITCH_PRIOR_MM = {'L': 33.0, 'T': 25.0, 'C': 17.0} 

PARAMS = {
    'merge': True,
    'fill': True,
    'near_mm': 3.0,
    'min_anchor_ml': 1.0,
    'order_tol_mm': 5.0,
    'spine_cluster_frac': 0.2,
    'merge_contact_ratio': 1.5,
    'undersized_ratio': 0.6,
    'partial_ratio': 0.35,
    'merged_volume_ratio': 1.3,
    'half_pitch_ratio': 0.6,
    'max_merges': 2,
    'large_change_ml': 1.0,
    'flag_pitch_low': 0.6,
    'flag_pitch_high': 1.6,
    'flag_volume_low': 0.6,
    'flag_volume_high': 1.4,
    'disc_score': 0.7,
    'disc_min_gap_mm': 6.0,
    'ct_height_low': 0.6,
    'ct_height_high': 1.45,
    'ct_pure_share': 0.8,
    'ct_vote_margin': 2,
    'ct_span_share': 0.15,
    'ct_split_ratio': 1.3,
    'ct_volume_low': 0.6,
    'ct_volume_high': 1.4,
    'ct_one_piece_share': 0.98,
    'max_ct_relabels': 12,
    'trim_hu': 0.0,
    'trim_max_share': 0.02,
    'notch_hu': 200.0,
    'rim_hu': 250.0,
    'rim_mm': 3.0,
    'rim_hu_max': 2000.0,
}


def _name(k):
    return VERTEBRAE[int(k)].replace('vertebrae_', '')


class Geometry:
    """Maps coordinates of the superior-up crop back to the original voxel grid and world mm."""

    def __init__(self, affine, orig_shape, ornt, offset):
        self.affine = affine
        self.orig_shape = orig_shape
        self.ornt = ornt
        self.offset = np.asarray(offset, float)

    def locate(self, crop_xyz):
        canon = self.offset + np.asarray(crop_xyz, float)
        ijk = np.empty(3)
        for i, (axis, flip) in enumerate(self.ornt):
            v = canon[int(axis)]
            ijk[i] = self.orig_shape[i] - 1 - v if flip < 0 else v
        mm = self.affine[:3, :3] @ ijk + self.affine[:3, 3]
        return {'ijk': [round(float(v), 1) for v in ijk], 'mm': [round(float(v), 1) for v in mm]}


def _touches_fov(bbox, offset, full_shape):
    return any(offset[a] + bbox[a].start <= 0 or offset[a] + bbox[a].stop >= full_shape[a] for a in range(3))


def _components(crop, sp, offset, full_shape):
    """Multi-label 26-connected components with label, size, centroid, bbox and field-of-view contact."""
    cc = cc3d.connected_components(crop, connectivity=26)
    stats = cc3d.statistics(cc)
    label_of = np.zeros(len(stats['voxel_counts']), np.int64)
    for cid, lab in fastremap.component_map(cc, crop).items():
        label_of[int(cid)] = int(lab)
    cen_vox = np.nan_to_num(np.asarray(stats['centroids'], float))
    bbox = stats['bounding_boxes']
    return {
        'cc': cc,
        'label_of': label_of,
        'vox': np.asarray(stats['voxel_counts'], np.int64),
        'cen_vox': cen_vox,
        'cen_mm': cen_vox * sp,
        'bbox': bbox,
        'fov': np.array([_touches_fov(b, offset, full_shape) for b in bbox]),
    }


def _contact_graph(cc, sp):
    """Shared surface in mm^2 between touching components; edge/corner-only touches get a tiny weight."""
    weight = {}
    for pair, area in cc3d.contacts(cc, connectivity=6, surface_area=True, anisotropy=tuple(sp)).items():
        weight[pair] = weight.get(pair, 0.0) + float(area)
    for pair, count in cc3d.contacts(cc, connectivity=26, surface_area=False).items():
        weight[pair] = weight.get(pair, 0.0) + 1e-3 * float(count)
    graph = {}
    for (a, b), w in weight.items():
        graph.setdefault(int(a), {})[int(b)] = w
        graph.setdefault(int(b), {})[int(a)] = w
    return graph


def _groups(nodes, graph):
    """Connected groups of `nodes` in the contact graph (union-find)."""
    nodes = list(nodes)
    node_set = set(nodes)
    parent = {c: c for c in nodes}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a in nodes:
        for b in graph.get(a, {}):
            if b in node_set:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for c in nodes:
        groups.setdefault(find(c), []).append(c)
    return sorted(groups.values(), key=min)


def _increasing_chain(first, cen_mm, vox, tol_mm):
    """Volume-weighted longest chain of labels whose main bodies move up the body as the label id goes up;
    a label may sit up to tol_mm below the one beneath it."""
    ks = sorted(first)
    z = [cen_mm[first[k]][2] for k in ks]
    w = [float(vox[first[k]]) for k in ks]
    best, prev = list(w), [-1] * len(ks)
    for i in range(len(ks)):
        for j in range(i):
            if z[j] < z[i] + tol_mm and best[j] + w[i] > best[i]:
                best[i], prev[i] = best[j] + w[i], j
    keep, i = set(), int(np.argmax(best))
    while i >= 0:
        keep.add(ks[i])
        i = prev[i]
    return keep


def _choose_anchors(comp, spine, voxel_ml, params, geo):
    """Main body of each label: its largest spine component, kept only if it is in anatomical order."""
    label_of, vox, cen_mm, fov = comp['label_of'], comp['vox'], comp['cen_mm'], comp['fov']
    candidates = {}
    for c in spine:
        if vox[c] * voxel_ml >= params['min_anchor_ml'] or fov[c]:
            candidates.setdefault(int(label_of[c]), []).append(c)
    for cs in candidates.values():
        cs.sort(key=lambda c: -vox[c])
    first = {k: cs[0] for k, cs in candidates.items()}
    if not first:
        return {}, []
    anchors = {k: first[k] for k in _increasing_chain(first, cen_mm, vox, params['order_tol_mm'])}

    order_log = []
    for k in sorted(set(first) - set(anchors)):
        below = [anchors[j] for j in anchors if j < k]
        above = [anchors[j] for j in anchors if j > k]
        lo = max((cen_mm[c][2] for c in below), default=-np.inf)
        hi = min((cen_mm[c][2] for c in above), default=np.inf)
        alt = next((c for c in candidates[k][1:] if lo < cen_mm[c][2] < hi), None)
        order_log.append({
            'label': _name(k),
            'demoted_volume_ml': round(float(vox[first[k]] * voxel_ml), 2),
            'demoted_location': geo.locate(comp['cen_vox'][first[k]]),
            'replacement': None if alt is None else geo.locate(comp['cen_vox'][alt]),
        })
        if alt is not None:
            anchors[k] = alt
    return anchors, order_log


def _propagate(anchors, label_of, graph, members):
    """Spread labels from the anchors over touching fragments, one ring per round."""
    resolved = {c: k for k, c in anchors.items()}
    scores = {}
    while True:
        ring = {}
        for c in members:
            if c in resolved:
                continue
            score = {}
            for d, w in graph.get(c, {}).items():
                if d in resolved:
                    score[resolved[d]] = score.get(resolved[d], 0.0) + w
            if score:
                own = label_of[c]
                ring[c] = (max(score, key=lambda k: (score[k], -abs(k - own), -k)), score)
        if not ring:
            return resolved, scores
        for c, (k, score) in ring.items():
            resolved[c] = k
            scores[c] = score


def _nearest_label(group, comp, res_lut, sp, keep_mm):
    """Distance (mm) from an isolated group to the closest resolved vertebra; label 0 means delete."""
    cc, bbox, vox, label_of = comp['cc'], comp['bbox'], comp['vox'], comp['label_of']
    pad = [int(np.ceil(keep_mm / sp[a])) + 1 for a in range(3)]
    win = tuple(slice(max(0, min(bbox[c][a].start for c in group) - pad[a]),
                      min(cc.shape[a], max(bbox[c][a].stop for c in group) + pad[a])) for a in range(3))
    sub = cc[win]
    near = res_lut[sub]
    present = np.unique(near)
    present = present[present > 0]
    if not present.size:
        return 0, None
    dist = ndimage.distance_transform_edt(~np.isin(sub, group), sampling=sp)
    dmin = np.atleast_1d(ndimage.minimum(dist, labels=near, index=present))
    own = label_of[max(group, key=lambda c: vox[c])]
    d, k = min(zip(dmin, present), key=lambda t: (round(float(t[0]), 6), int(t[1]) != own, int(t[1])))
    return (int(k) if d <= keep_mm else 0), float(d)


def _label_stats(crop, sp, offset, full_shape, voxel_ml):
    stats = cc3d.statistics(crop)
    tab = {}
    for k in VERTEBRAE:
        if k < len(stats['voxel_counts']) and stats['voxel_counts'][k] > 0:
            cen = np.asarray(stats['centroids'][k], float)
            n = int(stats['voxel_counts'][k])
            tab[k] = {'vox': n, 'ml': n * voxel_ml, 'mm3': n * voxel_ml * 1000.0, 'cen_vox': cen,
                      'cen_mm': cen * sp, 'bbox': stats['bounding_boxes'][k],
                      'fov': _touches_fov(stats['bounding_boxes'][k], offset, full_shape)}
    return tab


def _contact(con, a, b):
    return float(con.get((min(a, b), max(a, b)), 0.0))


def _dist(tab, a, b):
    return float(np.linalg.norm(tab[a]['cen_mm'] - tab[b]['cen_mm']))


def _local_volume(tab, k, exclude):
    vals = [tab[j]['ml'] for j in tab if j not in exclude and 0 < abs(j - k) <= 2]
    return float(np.median(vals)) if len(vals) >= 2 else None


def _local_pitch(tab, a, b):
    lo, hi = min(a, b) - 3, max(a, b) + 3
    vals = [_dist(tab, j, j + 1) for j in tab
            if j + 1 in tab and lo <= j and j + 1 <= hi and not {j, j + 1} & {a, b}]
    return float(np.median(vals)) if len(vals) >= 2 else PITCH_PRIOR_MM[REGION[min(a, b)]]


def _merge_candidates(tab, con, params):
    """Undersized labels that, together with the neighbour they touch most, look like one split bone."""
    present = sorted(tab)
    out = []
    for k in present:
        vref = _local_volume(tab, k, {k})
        if vref is None or tab[k]['ml'] / vref > params['undersized_ratio']:
            continue
        neighbours = [m for m in (max([j for j in present if j < k], default=None),
                                  min([j for j in present if j > k], default=None))
                      if m is not None and _contact(con, k, m) > 0]
        if not neighbours:
            continue
        m = max(neighbours, key=lambda j: _contact(con, k, j))
        area = _contact(con, k, m)
        cr = area / min(tab[k]['mm3'], tab[m]['mm3']) ** (2.0 / 3.0)
        vref_pair = _local_volume(tab, k, {k, m})
        merged = (tab[k]['ml'] + tab[m]['ml']) / vref_pair if vref_pair else None
        d, pitch = _dist(tab, k, m), _local_pitch(tab, k, m)
        partial = tab[k]['ml'] / vref <= params['partial_ratio']
        halves = d <= params['half_pitch_ratio'] * pitch
        ok = (cr >= params['merge_contact_ratio'] and merged is not None
              and merged <= params['merged_volume_ratio'] and (partial or halves)
              and not tab[k]['fov'] and not tab[m]['fov'])
        out.append({
            'label': k, 'partner': m, 'merge': bool(ok),
            'contact_mm2': round(area, 1), 'contact_ratio': round(cr, 2),
            'volume_ratio': round(tab[k]['ml'] / vref, 2),
            'merged_volume_ratio': None if merged is None else round(merged, 2),
            'centroid_distance_mm': round(d, 1), 'local_pitch_mm': round(pitch, 1),
            'other_neighbour': [{'label': _name(j), 'contact_mm2': round(_contact(con, k, j), 1)}
                                for j in neighbours if j != m],
        })
    return out


def _flags(tab, con, params, geo, changes, systematic):
    flags = []

    def add(kind, labels, values, where, message):
        flags.append({'type': kind, 'labels': [_name(k) for k in labels], 'values': values,
                      'location': geo.locate(where), 'message': message})

    present = sorted(tab)
    # labels missing inside the predicted range (including labels emptied by a merge)
    run = []
    for g in range(present[0] + 1, present[-1]) if present else []:
        if g not in tab:
            run.append(g)
        if run and (g + 1 in tab):
            a, b = run[0] - 1, run[-1] + 1
            pitches = _dist(tab, a, b) / _local_pitch(tab, a, b)
            if pitches < len(run) + 0.5:
                msg = ('spacing across the gap is %.1f local pitches, so no vertebra is physically missing: '
                       'labels from %s upwards are probably shifted by %d' % (pitches, _name(b), len(run)))
            else:
                msg = 'spacing across the gap fits %d missing vertebra(e)' % len(run)
            add('missing_label', run, {'pitches_across_gap': round(pitches, 2)},
                (tab[a]['cen_vox'] + tab[b]['cen_vox']) / 2, msg)
            run = []
    for k in present:
        if k + 1 in tab:
            ratio = _dist(tab, k, k + 1) / _local_pitch(tab, k, k + 1)
            if ratio > params['flag_pitch_high'] or ratio < params['flag_pitch_low']:
                kind = 'pitch_too_large' if ratio > 1 else 'pitch_too_small'
                add(kind, [k, k + 1], {'pitch_ratio': round(ratio, 2)},
                    (tab[k]['cen_vox'] + tab[k + 1]['cen_vox']) / 2,
                    'possible missing vertebra between them' if ratio > 1 else 'possible duplicate labels')
    volume_flagged = set()
    for k in present:
        vref = _local_volume(tab, k, {k})
        if vref is None or k in (23, 24):  # C2 and C1 are naturally different in size
            continue
        ratio = tab[k]['ml'] / vref
        if ratio < params['flag_volume_low'] or ratio > params['flag_volume_high']:
            volume_flagged.add(k)
            add('volume_too_small' if ratio < 1 else 'volume_too_large', [k],
                {'volume_ml': round(tab[k]['ml'], 1), 'volume_ratio': round(ratio, 2)}, tab[k]['cen_vox'],
                'smaller than its neighbours: possible compression fracture, partial vertebra or segmentation error'
                if ratio < 1 else 'larger than its neighbours: possible merged vertebrae or segmentation error')
    for k in present:
        if k + 1 in tab:
            cr = _contact(con, k, k + 1) / min(tab[k]['mm3'], tab[k + 1]['mm3']) ** (2.0 / 3.0)
            if cr >= params['merge_contact_ratio'] and {k, k + 1} & volume_flagged:
                add('label_boundary_through_bone', [k, k + 1], {'contact_ratio': round(cr, 2)},
                    (tab[k]['cen_vox'] + tab[k + 1]['cen_vox']) / 2,
                    'large shared surface: the boundary between these labels probably runs through bone, '
                    'not through the disc')
    for k in present:
        if tab[k]['fov']:
            add('truncated_by_field_of_view', [k], {}, tab[k]['cen_vox'], 'vertebra is cut by the scan border')
    for c in changes:
        if c['step'] in ('reassign', 'keep_isolated') and c['volume_ml'] >= params['large_change_ml']:
            flags.append({'type': 'large_reassignment', 'labels': [c['from'], c['to']],
                          'values': {'volume_ml': c['volume_ml']}, 'location': c['location'],
                          'message': 'large fragment relabelled, please review'})
    if systematic:
        flags.append({'type': 'systematic_mislabel', 'labels': [], 'values': {}, 'location': None,
                      'message': 'more split-vertebra candidates than max_merges; labelling of this case is unreliable'})
    return flags


def _ct_crop(ct, box, ornt, orig_shape):
    """Read only the CT block under the superior-up crop `box`, returned superior-up (HU)."""
    sl = [None] * 3
    for i, (axis, flip) in enumerate(ornt):
        s = box[int(axis)]
        sl[i] = slice(orig_shape[i] - s.stop, orig_shape[i] - s.start) if flip < 0 else s
    block = np.asarray(ct[tuple(sl)])
    if not np.issubdtype(block.dtype, np.integer):
        block = block.astype(np.float32)
    return nib.orientations.apply_orientation(block, ornt)


def _body_profile(crop, hu):
    """Per axial slice: vertebral-body voxels, their label counts, mean HU and centre."""
    nz = crop.shape[2]
    body = np.zeros(crop.shape, bool)
    counts = np.zeros((nz, 25), np.int64)
    mean_hu = np.full(nz, np.nan)
    centre = np.zeros((nz, 2))
    for z in range(nz):
        xs, ys = np.nonzero(crop[:, :, z])
        if not xs.size:
            continue
        front = ys >= ys.max() - 0.4 * (ys.max() - ys.min())  # canonical axis 1 points anterior
        xs, ys = xs[front], ys[front]
        body[xs, ys, z] = True
        counts[z] = np.bincount(crop[xs, ys, z], minlength=25)
        mean_hu[z] = hu[xs, ys, z].mean()
        centre[z] = xs.mean(), ys.mean()
    return body, counts, mean_hu, centre


def _find_discs(area, mean_hu, sp, params):
    """Discs: the labelled body area narrows (labels leave out disc tissue) and the voxels left there are dense endplate rims."""
    has = area > 0
    zs = np.nonzero(has)[0]
    idx = np.arange(len(area))
    sigma = 1.0 / sp[2]
    a = ndimage.gaussian_filter1d(area.astype(float), sigma)
    h = ndimage.gaussian_filter1d(np.interp(idx, zs, mean_hu[zs]), sigma)
    win = 2 * int(round(15.0 / sp[2])) + 1
    a_plateau = ndimage.maximum_filter1d(a, win)
    h_plateau = ndimage.median_filter(h, size=win)
    score = (1.0 - a / np.maximum(a_plateau, 1e-6)) + (h - h_plateau) / 200.0
    lo, hi = int(zs[0]), int(zs[-1])
    candidates = [z for z in range(lo + 1, hi)
                  if score[z] >= params['disc_score'] and score[z] >= score[z - 1] and score[z] >= score[z + 1]]
    discs = []
    for z in sorted(candidates, key=lambda z: (-score[z], z)):
        if all(abs(z - d) * sp[2] >= params['disc_min_gap_mm'] for d in discs):
            discs.append(z)
    return sorted(discs), score, lo, hi


def _split_piece(crop, s, k, spans, bones, body, hu, sp):
    """Split label k  between the bones it spans: its vertebral-body voxels are cut at the disc centres; the rest follows a CT-weighted watershed inside the piece only, so it goes with the body it is attached to."""
    region = crop[s] == k
    zz = np.arange(s[2].start, s[2].stop)
    markers = np.zeros(region.shape, np.int16)
    used = []
    for j in spans:
        inside = (zz >= bones[j]['start']) & (zz < bones[j]['stop'])
        seed = region & body[s] & inside[None, None, :]
        if seed.any():
            markers[seed] = len(used) + 1
            used.append(j)
    if len(used) < 2:
        return None
    smooth = ndimage.gaussian_filter(hu[s].astype(np.float32), sigma=1.0 / sp)
    cost = np.full(region.shape, 65535, np.uint16)  # leaving the piece is never cheaper than staying inside
    cost[region] = np.clip(1500.0 - smooth[region], 0, 3000).astype(np.uint16)
    ws = ndimage.watershed_ift(cost, markers)  # every voxel is reached from one of the positive seeds
    return [(region & (ws == t), j) for t, j in enumerate(used, 1)]


def _ct_step(crop, hu, sp, offset, full_shape, voxel_ml, params, geo, changes):
    """count bones from the CT and renumber a shifted run that lies between two agreeing blocks."""
    out = {'discs': [], 'bones': [], 'stretches': [], 'gaps': []}
    flags = []

    def flag(kind, labels, values, where, message):
        flags.append({'type': kind, 'labels': [_name(k) for k in labels], 'values': values,
                      'location': None if where is None else geo.locate(where), 'message': message})

    body, counts, mean_hu, centre = _body_profile(crop, hu)
    area = counts[:, 1:].sum(1) * sp[0] * sp[1]
    if np.count_nonzero(area) < 3:
        return out, flags
    discs, score, lo, hi = _find_discs(area, mean_hu, sp, params)
    at = lambda z: [centre[z][0], centre[z][1], z]
    out['discs'] = [{'location': geo.locate(at(z)), 'score': round(float(score[z]), 2)} for z in discs]

    edges = [lo] + discs + [hi + 1]
    bones = []
    for i in range(len(edges) - 1):
        c = counts[edges[i]:edges[i + 1]].sum(0)
        c[0] = 0
        total = int(c.sum())
        d = int(np.argmax(c)) if total else 0
        bones.append({'start': edges[i], 'stop': edges[i + 1], 'height_mm': (edges[i + 1] - edges[i]) * sp[2],
                      'counts': c, 'dom': d, 'share': c[d] / total if total else 0.0})
    n = len(bones)
    doms = [b['dom'] for b in bones]
    for i, b in enumerate(bones):
        ref = [bones[j]['height_mm'] for j in range(max(1, i - 3), min(n - 1, i + 4)) if j != i]
        b['reliable'] = (0 < i < n - 1 and len(ref) >= 2
                         and params['ct_height_low'] <= b['height_mm'] / float(np.median(ref)) <= params['ct_height_high'])
        b['pure'] = b['share'] >= params['ct_pure_share']
        b['unique'] = b['dom'] > 0 and doms.count(b['dom']) == 1
    mid = lambda b: at((b['start'] + b['stop'] - 1) // 2)

    unreliable = [i for i in range(1, n - 1) if not bones[i]['reliable']]
    if unreliable:
        flag('ct_unreliable', [], {'bones': len(unreliable)}, mid(bones[unreliable[0]]),
             'disc detection is unreliable for %d bone(s) (thick slices, tilted discs or a missed disc); '
             'their labels are not changed' % len(unreliable))

    stretches, current = [], []
    for i in range(n):
        if bones[i]['reliable']:
            current.append(i)
        elif current:
            stretches.append(current)
            current = []
    if current:
        stretches.append(current)

    gaps = []
    for stretch in stretches:
        voters = [i for i in stretch if bones[i]['pure'] and bones[i]['unique']]
        tally = {}
        for i in voters:
            tally[bones[i]['dom'] - i] = tally.get(bones[i]['dom'] - i, 0) + 1
        ranked = sorted(tally.items(), key=lambda t: -t[1])
        info = {'bones': [_name(bones[stretch[0]]['dom']) if bones[stretch[0]]['dom'] else None,
                          _name(bones[stretch[-1]]['dom']) if bones[stretch[-1]]['dom'] else None],
                'offset': None, 'anchor_runs': []}
        out['stretches'].append(info)
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < params['ct_vote_margin']:
            flag('count_mismatch', [], {'offsets': {str(o): c for o, c in ranked}}, mid(bones[stretch[0]]),
                 'two blocks of the spine disagree on the numbering with similar support; nothing renumbered')
            continue
        o = ranked[0][0]
        info['offset'] = o
        runs = []
        for i in voters:
            if bones[i]['dom'] - i == o:
                if runs and runs[-1][-1] == i - 1:
                    runs[-1].append(i)
                else:
                    runs.append([i])
        runs = [r for r in runs if len(r) >= 2]
        info['anchor_runs'] = [[_name(bones[r[0]]['dom']), _name(bones[r[-1]]['dom'])] for r in runs]
        found = [list(range(a[-1] + 1, b[0])) for a, b in zip(runs[:-1], runs[1:]) if b[0] > a[-1] + 1]
        gaps += [(g, o) for g in found]
        inside = {j for g in found for j in g}
        outside = [i for i in voters if bones[i]['dom'] - i != o and i not in inside]
        if outside:
            flag('count_mismatch', [bones[i]['dom'] for i in outside], {'bones': len(outside)}, mid(bones[outside[0]]),
                 'these vertebrae disagree with the rest of the spine on the numbering, but are not enclosed by two '
                 'agreeing blocks (possible miscount, variant spine or scan-edge block); not renumbered')

    for i, b in enumerate(bones):
        out['bones'].append({'index': i, 'location': geo.locate(mid(b)), 'height_mm': round(b['height_mm'], 1),
                             'dominant': _name(b['dom']) if b['dom'] else None, 'share': round(float(b['share']), 2),
                             'pure': bool(b['pure']), 'unique': bool(b['unique']), 'reliable': bool(b['reliable'])})

    M = np.array([b['counts'] for b in bones])
    for gap, o in gaps:
        entry = {'bones': gap, 'labels_before': [_name(bones[j]['dom']) if bones[j]['dom'] else None for j in gap],
                 'labels_after': [], 'assignments': [], 'accepted': False, 'problems': []}
        out['gaps'].append(entry)
        target = {j: j + o for j in gap}
        entry['labels_after'] = [_name(target[j]) for j in gap]
        labels_for = {i: b['dom'] for i, b in enumerate(bones)}
        labels_for.update(target)

        tab = _label_stats(crop, sp, offset, full_shape, voxel_ml)
        work = crop.copy()
        gap_changes, moved = [], set()
        for k in range(1, max(VERTEBRAE) + 1):
            cl = M[:, k]
            total = cl.sum()
            if not total or k not in tab:
                continue
            majority = int(np.argmax(cl))
            in_gap = majority in gap
            if not in_gap and cl[gap].sum() < params['ct_span_share'] * total:
                continue
            if tab[k]['fov']:
                entry['problems'].append('%s is cut by the scan edge' % _name(k))
                break
            spans = [j for j in range(n) if cl[j] >= params['ct_span_share'] * total]
            vref = _local_volume(tab, k, {k})
            s = tuple(slice(max(0, b.start - 1), min(crop.shape[a], b.stop + 1)) for a, b in enumerate(tab[k]['bbox']))
            parts = None
            if len(spans) >= 2 and vref and tab[k]['ml'] >= params['ct_split_ratio'] * vref:
                parts = _split_piece(crop, s, k, spans, bones, body, hu, sp)
            step = 'ct_split' if parts else 'ct_relabel'
            if not parts:
                if not in_gap:
                    continue
                parts = [(crop[s] == k, majority)]
            entry['assignments'].append({'label': _name(k), 'split': step == 'ct_split',
                                         'to': [_name(labels_for[j]) for _, j in parts]})
            for mask, j in parts:
                new = labels_for[j]
                nvox = int(mask.sum())
                if new == k or not nvox:
                    continue
                work[s][mask] = new
                moved.add(k)
                cen = np.array(ndimage.center_of_mass(mask)) + [x.start for x in s]
                gap_changes.append({'step': step, 'from': _name(k), 'to': _name(new), 'n_voxels': nvox,
                                    'volume_ml': round(nvox * voxel_ml, 3), 'location': geo.locate(cen),
                                    'reason': 'bones counted from the CT discs: this bone is %s; the blocks below '
                                              'and above agree on the numbering' % _name(new)})
        if entry['problems']:
            continue

        lo_lab, hi_lab = labels_for[gap[0] - 1], labels_for[gap[-1] + 1]
        wtab = _label_stats(work, sp, offset, full_shape, voxel_ml)
        for k in range(lo_lab, hi_lab + 1):
            if k not in wtab:
                continue
            s = tuple(slice(max(0, b.start - 1), min(work.shape[a], b.stop + 1)) for a, b in enumerate(wtab[k]['bbox']))
            sub = work[s]
            cc = cc3d.connected_components((sub == k).astype(np.uint8), connectivity=26)
            vc = cc3d.statistics(cc)['voxel_counts']
            if len(vc) <= 2:
                continue
            main = int(np.argmax(vc[1:])) + 1
            for piece_id in range(1, len(vc)):
                if piece_id == main:
                    continue
                piece = cc == piece_id
                touch = np.bincount(sub[ndimage.binary_dilation(piece, np.ones((3, 3, 3), bool)) & ~piece], minlength=25)
                touch[0] = touch[k] = 0
                if not touch.any():
                    continue
                new = int(np.argmax(touch))
                sub[piece] = new
                cen = np.array(ndimage.center_of_mass(piece)) + [x.start for x in s]
                gap_changes.append({'step': 'ct_relabel', 'from': _name(k), 'to': _name(new), 'n_voxels': int(vc[piece_id]),
                                    'volume_ml': round(float(vc[piece_id]) * voxel_ml, 3), 'location': geo.locate(cen),
                                    'reason': 'detached bit left by the split; lies in %s' % _name(new)})

        wtab = _label_stats(work, sp, offset, full_shape, voxel_ml)
        for k in range(lo_lab, hi_lab + 1):
            if k not in wtab:
                entry['problems'].append('%s missing after renumbering' % _name(k))
                continue
            m = (work[wtab[k]['bbox']] == k).astype(np.uint8)
            vc = cc3d.statistics(cc3d.connected_components(m, connectivity=26))['voxel_counts'][1:]
            if vc.max() / vc.sum() < params['ct_one_piece_share']:
                entry['problems'].append('%s is not one piece (largest %.0f %%)' % (_name(k), 100.0 * vc.max() / vc.sum()))
            vref = _local_volume(wtab, k, {k})
            if vref and not wtab[k]['fov'] and not (
                    params['ct_volume_low'] <= wtab[k]['ml'] / vref <= params['ct_volume_high']):
                entry['problems'].append('%s volume %.2fx its neighbours' % (_name(k), wtab[k]['ml'] / vref))
        if len(moved) > params['max_ct_relabels']:
            entry['problems'].append('%d labels would change (max %d)' % (len(moved), params['max_ct_relabels']))
        if entry['problems']:
            continue
        crop[...] = work
        changes.extend(gap_changes)
        entry['accepted'] = True
        entry['volumes_ml_after'] = {_name(k): round(wtab[k]['ml'], 1) for k in range(lo_lab, hi_lab + 1)}

    for entry in out['gaps']:
        if not entry['accepted']:
            flag('ct_fix_rejected', [], {'problems': entry['problems']}, mid(bones[entry['bones'][0]]),
                 'a renumbering between agreeing blocks was found but not applied: ' + '; '.join(entry['problems']))
    return out, flags


def _refine_surface(crop, hu, sp, params, voxel_ml, geo, changes):
    """trim labelled fat/air reached from outside, then add the missed cortical rim and fill bone-dense staircase notches. Returns flags."""
    six = ndimage.generate_binary_structure(3, 1)
    flags = []

    low = (crop > 0) & (hu < params['trim_hu'])
    if low.any():
        bg = cc3d.connected_components((crop == 0).view(np.uint8), connectivity=6)
        faces = [bg[0], bg[-1], bg[:, 0], bg[:, -1], bg[:, :, 0], bg[:, :, -1]]
        border_ids = np.unique(np.concatenate([f.ravel() for f in faces]))
        outside = np.isin(bg, border_ids[border_ids > 0])
        del bg
        cc = cc3d.connected_components(low.view(np.uint8), connectivity=6)
        open_ids = np.unique(cc[low & ndimage.binary_dilation(outside, six)])
        del outside
        trim = np.isin(cc, open_ids[open_ids > 0])
        del cc
        counts = np.bincount(crop[trim], minlength=max(VERTEBRAE) + 1)
        volume = np.bincount(crop.ravel(), minlength=max(VERTEBRAE) + 1)
        for k in np.nonzero(counts[1:] > params['trim_max_share'] * volume[1:])[0] + 1:
            region = trim & (crop == k)
            flags.append({'type': 'trim_skipped', 'labels': [_name(k)],
                          'values': {'volume_ml': round(float(counts[k]) * voxel_ml, 2),
                                     'share': round(float(counts[k]) / volume[k], 3)},
                          'location': geo.locate(ndimage.center_of_mass(region)),
                          'message': 'large low-density region reached from outside (possible fatty marrow or a '
                                     'broken cortex at a fracture); not trimmed'})
            trim &= ~region
            counts[k] = 0
        for k in np.nonzero(counts[1:])[0] + 1:
            cen = np.array(ndimage.center_of_mass(trim & (crop == k)))
            changes.append({'step': 'trim', 'from': _name(k), 'to': 'background', 'n_voxels': int(counts[k]),
                            'volume_ml': round(float(counts[k]) * voxel_ml, 3), 'location': geo.locate(cen),
                            'reason': 'labelled voxels below %.0f HU (fat/air) reached from outside' % params['trim_hu']})
        crop[trim] = 0

    reasons = {'grow_rim': 'cortical rim missed by the 1.5 mm resampling: attached bone (>= %.0f HU) staying '
                           'within %.1f mm of the vertebra' % (params['rim_hu'], params['rim_mm']),
               'fill_notch': 'staircase notch on bone (>= %.0f HU)' % params['notch_hu']}
    while True:
        added = [('grow_rim', a) for a in _grow_rim(crop, hu, sp, params)]
        added += [('fill_notch', a) for a in _fill_notches(crop, hu, params)]
        for step, (k, n, cen) in added:
            changes.append({'step': step, 'from': 'background', 'to': _name(k), 'n_voxels': n,
                            'volume_ml': round(n * voxel_ml, 3), 'location': geo.locate(cen), 'reason': reasons[step]})
        if not added:
            return flags


def _grow_rim(crop, hu, sp, params):
    """Unlabelled bone-dense structures that touch a vertebra and never extend more than rim_mm from the vertebrae; structures running further out are skipped. A structure touching several vertebrae is split by flooding through the structure from each vertebra, so every part stays attached to its vertebra; existing labels never change."""
    labelled = crop > 0
    dense = ~labelled & (hu >= params['rim_hu'])
    if not dense.any():
        return []
    comp = cc3d.connected_components(dense.view(np.uint8), connectivity=26)
    touching = np.unique(comp[dense & ndimage.binary_dilation(labelled, np.ones((3, 3, 3), bool))])
    faces = [comp[0], comp[-1], comp[:, 0], comp[:, -1], comp[:, :, 0], comp[:, :, -1]]
    touching = np.setdiff1d(touching, np.concatenate([f.ravel() for f in faces]))
    if not touching.size:
        return []
    dist = ndimage.distance_transform_edt(~labelled, sampling=sp)
    reach = np.atleast_1d(ndimage.maximum(dist, labels=comp, index=touching))
    del dist
    peak = np.atleast_1d(ndimage.maximum(hu, labels=comp, index=touching))  # cement / metal is never rim
    keep = touching[(reach <= params['rim_mm']) & (peak < params['rim_hu_max'])]
    if not keep.size:
        return []
    boxes = cc3d.statistics(comp)['bounding_boxes']
    ref = crop.copy()
    cube = np.ones((3, 3, 3), bool)
    total = {}
    for c in keep:
        s = tuple(slice(max(0, b.start - 1), min(crop.shape[a], b.stop + 1)) for a, b in enumerate(boxes[c]))
        piece = comp[s] == c
        seeds = ref[s].astype(np.int16)
        cost = np.full(piece.shape, 65535, np.uint16)
        cost[piece] = 1
        cost[seeds > 0] = 0
        nearest = ndimage.watershed_ift(cost, seeds, structure=cube)[piece]
        crop[s][piece] = nearest
        pts = np.argwhere(piece) + [x.start for x in s]
        for k in np.unique(nearest):
            sel = pts[nearest == k]
            n, tot = total.get(int(k), (0, np.zeros(3)))
            total[int(k)] = (n + len(sel), tot + sel.sum(0))
    return [(k, n, tot / n) for k, (n, tot) in sorted(total.items())]


def _fill_notches(crop, hu, params):
    """What a 1-voxel closing of each vertebra adds, only into background that is bone-dense and does not touch another vertebra."""
    cube = np.ones((3, 3, 3), bool)
    out = []
    for k in sorted(np.unique(crop)[1:]):
        found = ndimage.find_objects((crop == k).astype(np.uint8))
        if not found:
            continue
        s = tuple(slice(max(0, b.start - 2), min(crop.shape[a], b.stop + 2)) for a, b in enumerate(found[0]))
        sub = crop[s]
        other = (sub > 0) & (sub != k)
        gate = (sub == 0) & (hu[s] >= params['notch_hu']) & ~ndimage.binary_dilation(other, cube)
        mask = sub == k
        added = np.zeros(mask.shape, bool)
        while True:
            new = ndimage.binary_closing(mask, cube) & gate & ~mask
            if not new.any():
                break
            mask |= new
            added |= new
        n = int(added.sum())
        if n:
            sub[added] = k
            out.append((int(k), n, np.array(ndimage.center_of_mass(added)) + [x.start for x in s]))
    return out


def _label_metrics(comp, voxel_ml, geo):
    label_of, vox = comp['label_of'], comp['vox']
    out = {}
    for k in VERTEBRAE:
        ids = np.nonzero(label_of == k)[0]
        ids = ids[ids > 0]
        if not ids.size:
            continue
        v = vox[ids]
        total = int(v.sum())
        cen = (comp['cen_vox'][ids] * v[:, None]).sum(0) / total
        out[_name(k)] = {'volume_ml': round(total * voxel_ml, 2), 'n_components': int(ids.size),
                         'main_share': round(float(v.max()) / total, 3), 'centroid': geo.locate(cen),
                         'touches_fov': bool(comp['fov'][ids].any())}
    return out


def postprocess_array(arr, affine, zooms, params=None, ct=None):
    """Correct a vertebrae label map in place and return it with a report. `ct` is the CT on the same voxel grid: an array or a nibabel dataobj, read only under the crop."""
    params = dict(PARAMS, **(params or {}))
    ornt = nib.io_orientation(affine)
    view = nib.orientations.apply_orientation(arr, ornt)  # superior-up view sharing memory with arr
    sp = np.empty(3)
    sp[ornt[:, 0].astype(int)] = np.asarray(zooms, float)[:3]
    voxel_ml = float(np.prod(sp)) / 1000.0
    keep_mm = max(params['near_mm'], 1.05 * float(sp.max()))
    report = {'shape': list(arr.shape), 'axcodes': ''.join(nib.aff2axcodes(affine)),
              'spacing_mm': [round(float(s), 3) for s in zooms[:3]], 'params': dict(params, keep_mm=keep_mm),
              'before': {}, 'after': {}, 'order_check': [], 'isolated': [], 'merge_check': [],
              'changes': [], 'flags': []}

    nonzero = [np.nonzero(view.max(axis=tuple(b for b in range(3) if b != a)))[0] for a in range(3)]
    if not nonzero[0].size:
        report['note'] = 'empty prediction, nothing to do'
        return arr, report
    box = tuple(slice(max(0, int(nz[0]) - 2), min(view.shape[a], int(nz[-1]) + 3)) for a, nz in enumerate(nonzero))
    offset = [s.start for s in box]
    crop = np.ascontiguousarray(view[box])
    crop_before = crop.copy()
    geo = Geometry(affine, arr.shape, ornt, offset)
    report['crop_bbox_superior_up'] = [[s.start, s.stop] for s in box]
    changes = report['changes']

    comp = _components(crop, sp, offset, view.shape)
    report['before'] = _label_metrics(comp, voxel_ml, geo)
    report['components_before'] = int(len(comp['vox']) - 1)

    label_of, vox = comp['label_of'], comp['vox']
    members = [c for c in range(1, len(label_of)) if label_of[c] > 0]
    graph = _contact_graph(comp['cc'], sp)
    groups = _groups(members, graph)
    biggest = max(sum(vox[c] for c in g) for g in groups)
    spine = [c for g in groups if sum(vox[c] for c in g) >= params['spine_cluster_frac'] * biggest for c in g]
    anchors, report['order_check'] = _choose_anchors(comp, spine, voxel_ml, params, geo)
    demoted = {o['label'] for o in report['order_check']}
    resolved, scores = _propagate(anchors, label_of, graph, members)
    final = dict(resolved)

    res_lut = np.zeros(len(label_of), np.uint8)
    for c, k in resolved.items():
        res_lut[c] = k
    for group in _groups([c for c in members if c not in resolved], graph):
        k, d = _nearest_label(group, comp, res_lut, sp, keep_mm)
        ml = float(sum(vox[c] for c in group)) * voxel_ml
        report['isolated'].append({'labels': sorted({_name(label_of[c]) for c in group}), 'volume_ml': round(ml, 3),
                                   'distance_mm': None if d is None else round(d, 2),
                                   'decision': 'keep as %s' % _name(k) if k else 'delete',
                                   'location': geo.locate(comp['cen_vox'][max(group, key=lambda c: vox[c])])})
        for c in group:
            final[c] = k
            if k:
                scores[c] = {'distance_mm': d}

    for c in members:
        new, old = final[c], int(label_of[c])
        if new == old:
            continue
        s = comp['bbox'][c]
        sub = crop[s]
        sub[comp['cc'][s] == c] = new
        if new == 0:
            step, reason = 'delete_isolated', 'no vertebra within %.1f mm' % keep_mm
        elif c in resolved:
            step = 'reassign'
            top = sorted(scores[c].items(), key=lambda t: -t[1])[:3]
            reason = 'touches ' + ', '.join('%s (%.0f mm2)' % (_name(k), w) for k, w in top)
            if _name(old) in demoted and anchors.get(old) != c and vox[c] * voxel_ml >= params['min_anchor_ml']:
                reason += '; main body demoted by the order check'
        else:
            step, reason = 'keep_isolated', 'nearest vertebra %.1f mm away' % scores[c]['distance_mm']
        changes.append({'step': step, 'from': _name(old), 'to': _name(new) if new else 'background',
                        'n_voxels': int(vox[c]), 'volume_ml': round(float(vox[c]) * voxel_ml, 3),
                        'location': geo.locate(comp['cen_vox'][c]), 'reason': reason})
    del comp

    systematic = False
    if params['merge']:
        merges = 0
        while True:
            tab = _label_stats(crop, sp, offset, view.shape, voxel_ml)
            con = cc3d.contacts(crop, connectivity=6, surface_area=True, anisotropy=tuple(sp))
            candidates = _merge_candidates(tab, con, params)
            report['merge_check'].append([dict(c, label=_name(c['label']), partner=_name(c['partner'])) for c in candidates])
            ok = [c for c in candidates if c['merge']]
            if not ok:
                break
            if merges >= params['max_merges']:
                systematic = True
                break
            best = max(ok, key=lambda c: c['contact_ratio'])
            k, m = best['label'], best['partner']
            keep, gone = (k, m) if (tab[k]['vox'], -k) > (tab[m]['vox'], -m) else (m, k)
            sub = crop[tab[gone]['bbox']]
            sub[sub == gone] = keep
            changes.append({'step': 'merge', 'from': _name(gone), 'to': _name(keep), 'n_voxels': tab[gone]['vox'],
                            'volume_ml': round(tab[gone]['ml'], 3), 'location': geo.locate(tab[gone]['cen_vox']),
                            'reason': 'one bone split across two labels: contact ratio %.2f, volume ratio %.2f, '
                                      'merged volume ratio %.2f, centroid distance %.1f mm (local pitch %.1f mm); '
                                      '%s is now missing and nothing was renumbered'
                                      % (best['contact_ratio'], best['volume_ratio'], best['merged_volume_ratio'],
                                         best['centroid_distance_mm'], best['local_pitch_mm'], _name(gone))})
            merges += 1

    ct_flags = []
    report['ct_used'] = ct is not None
    if ct is not None:
        if tuple(ct.shape) != tuple(arr.shape):
            raise ValueError('CT shape %s does not match the label map %s' % (tuple(ct.shape), tuple(arr.shape)))
        hu = _ct_crop(ct, box, ornt, arr.shape)
        report['ct_check'], ct_flags = _ct_step(crop, hu, sp, offset, view.shape, voxel_ml, params, geo, changes)
        # step 7 (with a CT): trim fat/air, add the missed cortical rim, fill bone-dense staircase notches; before
        # step 5, so a hole that the added bone encloses is filled in the same run
        ct_flags += _refine_surface(crop, hu, sp, params, voxel_ml, geo, changes)
        del hu

    if params['fill']:
        for k, t in _label_stats(crop, sp, offset, view.shape, voxel_ml).items():
            s = tuple(slice(max(0, b.start - 1), min(crop.shape[a], b.stop + 1)) for a, b in enumerate(t['bbox']))
            sub = crop[s]
            add = ndimage.binary_fill_holes(sub == k) & (sub == 0)
            n = int(add.sum())
            if n:
                sub[add] = k
                changes.append({'step': 'fill', 'from': 'background', 'to': _name(k), 'n_voxels': n,
                                'volume_ml': round(n * voxel_ml, 3), 'location': geo.locate(t['cen_vox']),
                                'reason': 'enclosed hole'})

    tab = _label_stats(crop, sp, offset, view.shape, voxel_ml)
    con = cc3d.contacts(crop, connectivity=6, surface_area=True, anisotropy=tuple(sp))
    report['flags'] = _flags(tab, con, params, geo, changes, systematic) + ct_flags
    comp = _components(crop, sp, offset, view.shape)
    report['after'] = _label_metrics(comp, voxel_ml, geo)
    report['components_after'] = int(len(comp['vox']) - 1)
    del comp

    changed = int(np.count_nonzero(crop != crop_before))
    added = int(np.count_nonzero((crop_before == 0) & (crop != 0)))
    logged = sum(c['n_voxels'] for c in changes)
    filled = sum(c['n_voxels'] for c in changes if c['step'] in ('fill', 'fill_notch', 'grow_rim'))
    assert changed <= logged, 'unlogged voxel changes'
    assert added <= filled, 'voxels were added outside hole/notch filling and rim growth'
    report['voxels_changed'] = changed
    report['voxels_changed_by_steps'] = logged

    view[box] = crop
    return arr, report


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def process_case(case, args):
    t0 = time.time()
    src = os.path.join(args.pred_root, case, 'combined_labels.nii.gz')
    img = nib.load(src)
    arr = np.asanyarray(img.dataobj)
    if arr.dtype != np.uint8:
        if arr.min() < 0 or arr.max() > max(VERTEBRAE):
            raise ValueError('%s: label values outside 0..%d' % (src, max(VERTEBRAE)))
        arr = np.rint(arr).astype(np.uint8)
    elif not arr.flags.writeable:
        arr = arr.copy()

    params = dict(PARAMS, near_mm=args.near_mm, min_anchor_ml=args.min_anchor_ml,
                  merge_contact_ratio=args.merge_contact_ratio, max_merges=args.max_merges,
                  merge=not args.no_merge, fill=not args.no_fill,
                  disc_score=args.disc_score, max_ct_relabels=args.max_ct_relabels)
    ct = None
    if args.ct_root:
        ct_img = nib.load(os.path.join(args.ct_root, case, 'ct.nii.gz'))
        if not np.allclose(ct_img.affine, img.affine, atol=1e-3):  # the shape is checked during processing
            raise ValueError('%s: CT and prediction are not on the same voxel grid' % case)
        ct = ct_img.dataobj
    arr, report = postprocess_array(arr, img.affine, img.header.get_zooms()[:3], params, ct=ct)

    case_dir = os.path.join(args.save_root, case)
    seg_dir = os.path.join(case_dir, 'segmentations')
    os.makedirs(seg_dir, exist_ok=True)
    out = nib.Nifti1Image(arr, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, os.path.join(case_dir, 'combined_labels.nii.gz'))
    buf = np.empty(arr.shape, bool)
    for k, name in VERTEBRAE.items():
        np.equal(arr, k, out=buf)
        seg = nib.Nifti1Image(buf.view(np.uint8), img.affine, img.header)
        seg.set_data_dtype(np.uint8)
        nib.save(seg, os.path.join(seg_dir, name + '.nii.gz'))

    report = dict({'case': case, 'input': os.path.abspath(src), 'output': os.path.abspath(case_dir),
                   'runtime_s': round(time.time() - t0, 1)}, **report)
    with open(os.path.join(case_dir, 'postprocessing_report.json'), 'w') as f:
        json.dump(report, f, indent=1, default=_jsonable)

    changes = report['changes']
    pick = lambda step: [c for c in changes if c['step'] == step]
    return {
        'case': case, 'status': 'ok', 'error': '', 'runtime_s': report['runtime_s'],
        'comps_before': report.get('components_before', 0), 'comps_after': report.get('components_after', 0),
        'labels_before': len(report['before']), 'labels_after': len(report['after']),
        'n_reassigned': len(pick('reassign')), 'ml_reassigned': round(sum(c['volume_ml'] for c in pick('reassign')), 2),
        'n_deleted': len(pick('delete_isolated')),
        'ml_deleted': round(sum(c['volume_ml'] for c in pick('delete_isolated')), 2),
        'n_kept_isolated': sum(1 for i in report['isolated'] if i['decision'] != 'delete'),
        'merges': ' '.join('%s->%s' % (c['from'], c['to']) for c in pick('merge')),
        'ct_renumbered': ' '.join(sorted({'%s->%s' % (c['from'], c['to'])
                                          for c in changes if c['step'] in ('ct_relabel', 'ct_split')})),
        'missing_labels': ' '.join(l for f in report['flags'] if f['type'] == 'missing_label' for l in f['labels']),
        'filled_voxels': sum(c['n_voxels'] for c in pick('fill')),
        'n_flags': len(report['flags']),
        'flag_types': ' '.join(sorted({f['type'] for f in report['flags']})),
    }


def _run_case(case, args):
    try:
        return process_case(case, args)
    except Exception as e:  # keep going with the other cases
        return {'case': case, 'status': 'failed', 'error': '%s: %s' % (type(e).__name__, e)}


def main(args):
    pred_root = os.path.normcase(os.path.realpath(args.pred_root))
    save_root = os.path.normcase(os.path.realpath(args.save_root))
    assert os.path.isdir(pred_root), 'prediction folder not found: %s' % args.pred_root
    same_drive = os.path.splitdrive(pred_root)[0] == os.path.splitdrive(save_root)[0]
    if same_drive and os.path.commonpath([pred_root, save_root]) == pred_root:
        raise SystemExit('--save_root must not be the prediction folder or inside it (predictions are never overwritten)')
    cases = sorted(d for d in os.listdir(args.pred_root)
                   if os.path.isfile(os.path.join(args.pred_root, d, 'combined_labels.nii.gz')))
    if args.cases:
        cases = [c for c in cases if c in set(args.cases)]
    os.makedirs(args.save_root, exist_ok=True)

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_case, c, args) for c in cases]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                rows.append(fut.result())
    else:
        for c in tqdm(cases):
            rows.append(_run_case(c, args))
    rows.sort(key=lambda r: r['case'])

    fields = ['case', 'status', 'error', 'runtime_s', 'comps_before', 'comps_after', 'labels_before',
              'labels_after', 'n_reassigned', 'ml_reassigned', 'n_deleted', 'ml_deleted', 'n_kept_isolated',
              'merges', 'ct_renumbered', 'missing_labels', 'filled_voxels', 'n_flags', 'flag_types']
    with open(os.path.join(args.save_root, 'postprocessing_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r['status'] == 'ok':
            print('%s: components %d -> %d, reassigned %d (%.1f ml), deleted %d (%.1f ml), merges [%s], '
                  'CT renumbered [%s], missing [%s], %d flags'
                  % (r['case'], r['comps_before'], r['comps_after'], r['n_reassigned'], r['ml_reassigned'],
                     r['n_deleted'], r['ml_deleted'], r['merges'], r['ct_renumbered'], r['missing_labels'],
                     r['n_flags']))
        else:
            print('%s: FAILED %s' % (r['case'], r['error']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process AI-predicted vertebrae masks')
    parser.add_argument('--pred_root', default='./AbdomenAtlasDemoPredict', help='output folder of inference.py')
    parser.add_argument('--save_root', default='./AbdomenAtlasDemoPostprocessed', help='where corrected masks go')
    parser.add_argument('--cases', nargs='*', help='only these case folders')
    parser.add_argument('--workers', default=1, type=int,
                        help='parallel cases ')
    parser.add_argument('--near_mm', default=PARAMS['near_mm'], type=float, help='keep isolated fragments this close to a vertebra')
    parser.add_argument('--min_anchor_ml', default=PARAMS['min_anchor_ml'], type=float, help='smallest main body of a vertebra')
    parser.add_argument('--merge_contact_ratio', default=PARAMS['merge_contact_ratio'], type=float,
                        help='minimum shared surface / min(volume)^(2/3) for merging a split vertebra')
    parser.add_argument('--max_merges', default=PARAMS['max_merges'], type=int, help='merges per case before giving up')
    parser.add_argument('--ct_root', default=None,
                        help='enables the CT-based bone count and renumbering')
    parser.add_argument('--disc_score', default=PARAMS['disc_score'], type=float, help='disc detection threshold')
    parser.add_argument('--max_ct_relabels', default=PARAMS['max_ct_relabels'], type=int,
                        help='largest number of vertebrae one CT repair may relabel')
    parser.add_argument('--no_merge', action='store_true', default=False, help='will not merge split vertebrae')
    parser.add_argument('--no_fill', action='store_true', default=False, help='will not fill holes')
    main(parser.parse_args())
