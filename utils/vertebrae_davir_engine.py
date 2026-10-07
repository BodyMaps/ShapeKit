#!/usr/bin/env python
"""
DAVIR: Disc-Aware Vertebral Instance Recovery and Re-identification.

Fixes wrongly numbered, split and fragmented vertebrae in AI-predicted CT masks (C1-L5). The
vertebral bodies are found from the predicted bone and the CT, named jointly along the spine, and
only labels that disagree with this naming are changed.

In ShapeKit this file is used through utils/vertebrae_davir.py (vertebrae_engine: shapekit_davir).
Standalone version and documentation: https://github.com/Nikhil-Rao20/ShapeKit-DAVIR
"""
import argparse
import json
import os
import time

import cc3d
import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

# SuPreM vertebrae label map (id increases cranially)
NAMES = {1: "L5", 2: "L4", 3: "L3", 4: "L2", 5: "L1",
         6: "T12", 7: "T11", 8: "T10", 9: "T9", 10: "T8", 11: "T7", 12: "T6",
         13: "T5", 14: "T4", 15: "T3", 16: "T2", 17: "T1",
         18: "C7", 19: "C6", 20: "C5", 21: "C4", 22: "C3", 23: "C2", 24: "C1"}
NUM_LABELS = 24
INV_NAMES = {v: k for k, v in NAMES.items()}

# Mean / SD centroid distance (mm) from level k to k+1, VerSe statistics (Payer et al. 2020)
GAP_MU = {1: 31.7, 2: 33.8, 3: 34.1, 4: 33.4, 5: 32.0, 6: 29.8, 7: 27.6, 8: 25.7, 9: 24.9, 10: 24.3,
          11: 24.0, 12: 23.5, 13: 22.8, 14: 22.3, 15: 22.2, 16: 20.9, 17: 18.8, 18: 17.7, 19: 16.1,
          20: 16.1, 21: 15.9, 22: 18.5, 23: 17.1}
GAP_SD = {k: 2.0 for k in GAP_MU}
GAP_SD.update({1: 2.7, 5: 2.3, 17: 1.2, 23: 2.0})

PARAMS = dict(
    crop_margin_mm=15.0,
    far_component_mm=10.0,       # farther from the spine than this = false positive
    min_fragment_mm3=30.0,
    keep_large_mm3=2000.0,       # large components on the spine axis are kept
    axis_tolerance_mm=25.0,
    centreline_smooth_mm=8.0,
    tube_radius_factor=0.7,      # body-core tube (x body radius)
    body_tube_factor=1.25,       # body-territory tube (x body radius)
    w_transition=1.0,
    w_pitch=1.0,
    pitch_sigma=0.15,
    local_scale_window_mm=80.0,
    vote_eps=1e-3,
    vote_sigma=0.8,
    w_gap=0.5,
    skip_cost=4.0,
    split_cost=4.0,
    c1_core_penalty=5.0,
    clean_ratio=0.8,             # share needed to keep a model label as-is (renamed only)
    attach_dominance=0.75,       # share of body contact needed for an arch piece to follow one body
    surface_refine=True,
    surface_add_hu=350,
    surface_remove_hu=50,
)


def to_ras(data, affine):
    """Reorient to RAS; also return the transform that undoes it."""
    src = io_orientation(affine)
    ras = axcodes2ornt(("R", "A", "S"))
    fwd = ornt_transform(src, ras)
    inv = ornt_transform(ras, src)
    return apply_orientation(data, fwd), inv


def load_prediction(case_dir):
    """Load combined_labels.nii.gz, or rebuild it from the per-level masks."""
    path = os.path.join(case_dir, "combined_labels.nii.gz")
    if os.path.exists(path):
        img = nib.load(path)
        return img, np.asarray(img.dataobj).astype(np.uint8)
    ref, lab = None, None
    for k, n in NAMES.items():
        p = os.path.join(case_dir, "segmentations", f"vertebrae_{n}.nii.gz")
        if not os.path.exists(p):
            continue
        img = nib.load(p)
        m = np.asarray(img.dataobj) > 0
        if lab is None:
            ref, lab = img, np.zeros(m.shape, np.uint8)
        lab[m] = k
    if lab is None:
        raise FileNotFoundError(f"no vertebrae prediction in {case_dir}")
    return ref, lab


def save_case(out_case_dir, ref_img, lab_native):
    """Write combined_labels.nii.gz and all 24 per-level masks on the original grid."""
    os.makedirs(os.path.join(out_case_dir, "segmentations"), exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(lab_native.astype(np.uint8), ref_img.affine, hdr),
             os.path.join(out_case_dir, "combined_labels.nii.gz"))
    for k, n in NAMES.items():
        nib.save(nib.Nifti1Image((lab_native == k).astype(np.uint8), ref_img.affine, hdr),
                 os.path.join(out_case_dir, "segmentations", f"vertebrae_{n}.nii.gz"))


def keep_spinal_column(lab, zooms, p, log):
    """Remove false-positive components far from the spinal column."""
    u = lab > 0
    cc, n = cc3d.connected_components(u, connectivity=26, return_N=True)
    if n <= 1:
        return lab
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    main = int(sizes.argmax())
    vmm3 = float(np.prod(zooms))
    idx = np.argwhere(cc == main)
    pad = np.ceil((p["far_component_mm"] + 5) / zooms).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, u.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    dist = ndi.distance_transform_edt(cc[sl] != main, sampling=zooms)
    sub = cc[sl]
    mind = np.full(n + 1, np.inf)
    present = np.unique(sub)
    md = ndi.minimum(dist, sub, index=present)
    mind[present] = md
    # a large component on the spine axis is spine beyond a missed vertebra, not a false positive
    stats = cc3d.statistics(cc)
    cen = stats["centroids"]
    zz, xs, ys = idx[:, 2], idx[:, 0], idx[:, 1]
    out = lab.copy()
    removed = []
    for i in range(1, n + 1):
        if i == main:
            continue
        vol = sizes[i] * vmm3
        if vol >= p["keep_large_mm3"]:
            zc = cen[i][2]
            near_z = np.abs(zz - zc) <= np.abs(zz - zc).min() + 20.0 / zooms[2]
            axis_xy = np.array([np.median(xs[near_z]), np.median(ys[near_z])])
            if np.linalg.norm((cen[i][:2] - axis_xy) * zooms[:2]) <= p["axis_tolerance_mm"]:
                continue
        if mind[i] > p["far_component_mm"] or vol < p["min_fragment_mm3"]:
            m = cc == i
            labs = np.bincount(lab[m], minlength=NUM_LABELS + 1)
            out[m] = 0
            removed.append({"volume_mm3": round(vol, 1), "dist_mm": None if np.isinf(mind[i]) else round(float(mind[i]), 1),
                            "labels": [NAMES[k] for k in np.nonzero(labs)[0] if k > 0]})
    log["removed_components"] = removed
    return out


def body_centreline(u_filled, zooms, p):
    """Centreline through the vertebral bodies: the in-plane distance-map maximum of each slice."""
    zs = np.nonzero(u_filled.any(axis=(0, 1)))[0]
    pts = []
    for z in range(zs.min(), zs.max() + 1):
        s = u_filled[:, :, z]
        if s.sum() < 10:
            continue
        e = ndi.distance_transform_edt(s, sampling=zooms[:2])
        i = np.unravel_index(int(e.argmax()), e.shape)
        pts.append((z, i[0], i[1], e.max()))
    pts = np.asarray(pts, float)
    win = max(3, int(round(40 / zooms[2])) | 1)
    med = ndi.median_filter(pts[:, 3], size=win, mode="nearest")
    good = pts[:, 3] > 0.6 * med  # skip slices through discs
    zz = np.arange(zs.min(), zs.max() + 1)
    k = max(3, int(round(10 / zooms[2])) | 1)
    cx = np.interp(zz, pts[good, 0], ndi.median_filter(pts[good, 1], k, mode="nearest"))
    cy = np.interp(zz, pts[good, 0], ndi.median_filter(pts[good, 2], k, mode="nearest"))
    rr = np.interp(zz, pts[good, 0], ndi.median_filter(pts[good, 3], k, mode="nearest"))
    sig = p["centreline_smooth_mm"] / zooms[2]
    cx = ndi.gaussian_filter1d(cx, sig, mode="nearest")
    cy = ndi.gaussian_filter1d(cy, sig, mode="nearest")
    rr = ndi.gaussian_filter1d(rr, 3 * sig, mode="nearest")
    C = np.stack([cx * zooms[0], cy * zooms[1], zz * zooms[2]], 1)
    s_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    S = np.arange(0, s_arc[-1] + 1e-6, 0.5)
    Cs = np.stack([np.interp(S, s_arc, C[:, j]) for j in range(3)], 1)
    Rs = np.interp(S, s_arc, rr)
    return dict(S=S, C=Cs, R=Rs, z_of_slice=zz, cx=cx, cy=cy, rr=rr)


def tube_voxels(shape, zooms, cl, radius_factor):
    """Voxels within radius_factor x body radius of the centreline, with their arc-length position."""
    gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cand = []
    for j, z in enumerate(cl["z_of_slice"]):
        d = np.hypot((gx - cl["cx"][j]) * zooms[0], (gy - cl["cy"][j]) * zooms[1])
        ii = np.argwhere(d < 1.3 * cl["rr"][j])
        cand.append(np.c_[ii, np.full(len(ii), z)])
    vox = np.concatenate(cand)
    dist, nn = cKDTree(cl["C"]).query(vox * zooms)
    keep = dist < radius_factor * cl["R"][nn]
    return vox[keep], cl["S"][nn[keep]], dist[keep]


def disc_profiles(lab, ct, vox, s_pos):
    """Bone fraction, mean HU and model label votes along the centreline (1 mm bins)."""
    nb = int(np.ceil(s_pos.max())) + 2
    b = np.floor(s_pos).astype(int)
    u = lab[vox[:, 0], vox[:, 1], vox[:, 2]] > 0
    cnt = np.maximum(np.bincount(b, minlength=nb), 1)
    A = np.bincount(b, u, minlength=nb) / cnt
    H = None
    if ct is not None:
        h = ct[vox[:, 0], vox[:, 1], vox[:, 2]].astype(float)
        H = np.bincount(b, np.clip(h, -200, 1200), minlength=nb) / cnt
    ll = lab[vox[:, 0], vox[:, 1], vox[:, 2]].astype(int)
    votes = np.zeros((nb, NUM_LABELS + 1))
    np.add.at(votes, (b, ll), 1)
    votes[:, 0] = 0
    return dict(A=ndi.gaussian_filter1d(A, 1.0), H=None if H is None else ndi.gaussian_filter1d(H, 1.0),
                votes=votes, nb=nb)


def detect_discs(prof, p, log):
    """Choose disc positions by dynamic programming.

    Candidates are gaps in the predicted bone and model label changes. Each is scored by gap depth,
    HU dip and label change; each segment between two discs is penalised if its length is not a
    plausible vertebral pitch. This rejects label changes in the middle of a body."""
    A, H, votes = prof["A"], prof["H"], prof["votes"]
    nb = len(A)
    present = np.nonzero(A > 0.5)[0]
    s0, s1 = int(present.min()), int(present.max()) + 1
    dom = votes.argmax(1)
    valid = votes.sum(1) > 0
    lab_pos = np.nonzero(valid)[0]
    dom_f = np.interp(np.arange(nb), lab_pos, dom[lab_pos]).round().astype(int)
    mu = np.array([GAP_MU[min(max(d, 1), 23)] for d in dom_f])
    if H is not None:
        Hs = ndi.median_filter(H, size=41, mode="nearest")
        dip = np.clip((Hs - ndi.minimum_filter1d(H, 9)) / 100.0, 0, 1.5)
    else:
        dip = np.zeros(nb)
    Amin = ndi.minimum_filter1d(A, 9)
    cand = {}
    for i in range(s0 + 1, s1 - 1):
        if A[i] <= A[i - 1] and A[i] <= A[i + 1] and A[i] < 0.9:
            cand[i] = 0.0
    trans = [i for i in range(s0 + 1, s1) if valid[i] and valid[i - 1] and dom[i] != dom[i - 1]]
    for t in trans:
        near = [c for c in cand if abs(c - t) <= 3]
        if near:
            c = min(near, key=lambda c: A[c])
            cand[c] = 1.0
        else:
            cand[t] = 1.0
    pos = sorted(cand)
    ev = np.array([2.0 * (1.0 - Amin[c]) + dip[c] + p["w_transition"] * cand[c] for c in pos])
    # patient scale from the deep, unambiguous disc gaps
    deep = [c for c in pos if A[c] < 0.3]
    rat = [(b - a) / mu[(a + b) // 2] for a, b in zip(deep[:-1], deep[1:]) if 0.6 < (b - a) / mu[(a + b) // 2] < 1.6]
    scale = float(np.clip(np.median(rat), 0.75, 1.35)) if rat else 1.0

    def run_dp(local_scale):
        def seg_cost(a, b, end_seg):
            m = (a + b) // 2
            r = np.log(max(b - a, 1) / (local_scale[m] * mu[m]))
            if end_seg and r < 0:  # end vertebrae may be cut by the scan border
                return 0.0
            return p["w_pitch"] * (r / p["pitch_sigma"]) ** 2

        nodes = [s0] + pos + [s1]
        evn = np.r_[0.0, ev, 0.0]
        best = np.full(len(nodes), -np.inf)
        prev = np.full(len(nodes), -1)
        best[0] = 0.0
        for j in range(1, len(nodes)):
            for i in range(j - 1, -1, -1):
                m = (nodes[i] + nodes[j]) // 2
                if nodes[j] - nodes[i] > 2.6 * local_scale[m] * mu[m] and np.isfinite(best[j]):
                    break
                if not np.isfinite(best[i]):
                    continue
                v = best[i] + evn[j] - seg_cost(nodes[i], nodes[j], i == 0 or j == len(nodes) - 1)
                if v > best[j]:
                    best[j], prev[j] = v, i
        path = [len(nodes) - 1]
        while path[-1] > 0:
            path.append(prev[path[-1]])
        assert path[-1] == 0, "disc DP failed to reach the start node"
        return [nodes[k] for k in reversed(path)]

    # second pass: pitch scale re-estimated locally from neighbouring segments
    bounds = run_dp(np.full(nb, scale))
    for _ in range(2):
        segs = [(a, b) for a, b in zip(bounds[1:-2], bounds[2:-1])]
        if len(segs) < 3:
            break
        cen = np.array([(a + b) / 2 for a, b in segs])
        rat = np.array([(b - a) / mu[(a + b) // 2] for a, b in segs])
        loc = np.full(nb, scale)
        for x in range(nb):
            w = np.abs(cen - x) <= p["local_scale_window_mm"]
            if w.sum() >= 3:
                loc[x] = np.median(rat[w])
        loc = np.clip(ndi.gaussian_filter1d(loc, 10.0), 0.7, 1.4)
        new_bounds = run_dp(loc)
        if new_bounds == bounds:
            break
        bounds = new_bounds
    rejected = [t for t in trans if all(abs(t - b) > 3 for b in bounds)]
    log["disc_detection"] = dict(scale=round(scale, 3), discs_mm=[int(b) for b in bounds[1:-1]],
                                 rejected_label_transitions_mm=[int(t) for t in rejected])
    return bounds


def body_cores(lab, zooms, vox, s_pos, bounds, log):
    """One body core (vertebra instance) between each pair of consecutive discs."""
    u = lab > 0
    vmm3 = float(np.prod(zooms))
    cores = []
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        m = (s_pos >= a + 1) & (s_pos < b - 1)
        v = vox[m]
        v = v[u[v[:, 0], v[:, 1], v[:, 2]]]
        if len(v) * vmm3 < 100:
            continue
        cores.append(dict(vox=v, centroid_mm=v.mean(0) * zooms, core_mm3=len(v) * vmm3, s_range=(int(a), int(b))))
    log["n_instances"] = len(cores)
    return cores


def body_territory(lab, zooms, cl, bounds, cores, labels, p):
    """Vertebral-body voxels, each tagged with the name of the segment it falls in."""
    vox, s_pos, _ = tube_voxels(lab.shape, zooms, cl, p["body_tube_factor"])
    keep = lab[vox[:, 0], vox[:, 1], vox[:, 2]] > 0
    vox, s_pos = vox[keep], s_pos[keep]
    seg_name = np.zeros(len(bounds), int)
    for j, c in enumerate(cores):
        seg_name[int(np.searchsorted(bounds, c["s_range"][0], side="right")) - 1] = labels[j]
    seg = np.clip(np.searchsorted(bounds, s_pos, side="right") - 1, 0, len(bounds) - 1)
    return vox, seg_name[seg]


def name_instances(lab, cores, p, log):
    """Name the bodies jointly (Viterbi) from the model's votes and the spacing prior.

    Allowed steps between consecutive bodies: next level, skip one level, or same level (split body)."""
    n = len(cores)
    K = NUM_LABELS
    votes = np.zeros((n, K))
    for i, c in enumerate(cores):
        v = c["vox"]
        votes[i] = np.bincount(lab[v[:, 0], v[:, 1], v[:, 2]], minlength=K + 1)[1:]
    raw_frac = votes / np.maximum(votes.sum(1, keepdims=True), 1)
    sm = ndi.gaussian_filter1d(raw_frac, p["vote_sigma"], axis=1, mode="constant")
    sm = sm / np.maximum(sm.sum(1, keepdims=True), 1e-9)
    U = -np.log(p["vote_eps"] + sm)
    # C1 has no body: only allow C1 where the model itself votes C1
    U[raw_frac[:, 23] < 0.5, 23] += p["c1_core_penalty"]
    cen = np.array([c["centroid_mm"] for c in cores])
    d = np.linalg.norm(np.diff(cen, axis=0), axis=1)
    dom = raw_frac.argmax(1) + 1
    ratios = [d[i] / GAP_MU[dom[i]] for i in range(n - 1) if dom[i] in GAP_MU and dom[i + 1] == dom[i] + 1]
    scale = float(np.clip(np.median(ratios), 0.75, 1.35)) if ratios else 1.0

    def trans(a, b, gap):
        if b == a + 1 and a in GAP_MU:
            return p["w_gap"] * 0.5 * ((gap - scale * GAP_MU[a]) / (scale * GAP_SD[a])) ** 2
        if b == a + 2 and a in GAP_MU and (a + 1) in GAP_MU:
            mu = GAP_MU[a] + GAP_MU[a + 1]
            sd = np.hypot(GAP_SD[a], GAP_SD[a + 1])
            return p["skip_cost"] + p["w_gap"] * 0.5 * ((gap - scale * mu) / (scale * sd)) ** 2
        if b == a and a in GAP_MU:
            return p["split_cost"] + p["w_gap"] * 0.5 * ((gap - 0.5 * scale * GAP_MU[a]) / (scale * GAP_SD[a])) ** 2
        return np.inf

    D = np.full((n, K), np.inf)
    back = np.zeros((n, K), int)
    D[0] = U[0]
    for i in range(1, n):
        for b in range(1, K + 1):
            best, arg = np.inf, -1
            for a in (b - 1, b - 2, b):
                if 1 <= a <= K and np.isfinite(D[i - 1, a - 1]):
                    c = D[i - 1, a - 1] + trans(a, b, d[i - 1])
                    if c < best:
                        best, arg = c, a
            D[i, b - 1] = best + U[i, b - 1]
            back[i, b - 1] = arg
    labels = np.zeros(n, int)
    labels[-1] = int(np.argmin(D[-1])) + 1
    for i in range(n - 1, 0, -1):
        labels[i - 1] = back[i, labels[i] - 1]
    total = float(D[-1].min())

    def path_cost(lbl):
        if lbl.min() < 1 or lbl.max() > K:
            return np.inf
        c = U[0, lbl[0] - 1]
        for i in range(1, n):
            t = trans(lbl[i - 1], lbl[i], d[i - 1])
            if not np.isfinite(t):
                return np.inf
            c += t + U[i, lbl[i] - 1]
        return c
    alt = min(path_cost(labels + 1), path_cost(labels - 1))
    log["naming"] = dict(
        patient_scale=round(scale, 3), total_cost=round(total, 2),
        shift_margin=None if not np.isfinite(alt) else round(float(alt - total), 2),
        instances=[dict(name=NAMES[int(l)], raw_majority=NAMES[int(dom[i])],
                        raw_purity_of_assigned=round(float(raw_frac[i, l - 1]), 3),
                        core_cm3=round(cores[i]["core_mm3"] / 1000, 2),
                        gap_to_next_mm=None if i == n - 1 else round(float(d[i]), 1))
                   for i, l in enumerate(labels)])
    return labels, votes


def compose(lab, cores, labels, votes, body_vox, body_name, u_filled, zooms, cl, p, log):
    """Keep model labels that map onto one vertebra (renamed only); re-assign the rest."""
    K = NUM_LABELS
    M = np.zeros((K, K + 1))  # model label x final name, counted on body voxels
    for j, c in enumerate(cores):
        M[:, labels[j]] += votes[j]
    rowsum = M.sum(1, keepdims=True)
    colsum = M.sum(0, keepdims=True)
    out = np.zeros_like(lab)
    mapping = {}
    for k in range(1, K + 1):
        if rowsum[k - 1, 0] == 0:
            continue
        t = int(M[k - 1].argmax())
        r = M[k - 1, t] / rowsum[k - 1, 0]
        c = M[k - 1, t] / colsum[0, t]
        clean = r >= p["clean_ratio"] and c >= p["clean_ratio"]
        mapping[NAMES[k]] = dict(to=NAMES[t], row=round(float(r), 3), col=round(float(c), 3), clean=bool(clean))
        if clean:
            out[lab == k] = t
    # labels with no body (C1 ring, vertebra cut by the scan border) are kept if they extend the sequence
    named = set(int(x) for x in labels)
    dominant = set(int(v.argmax()) + 1 for v in votes if v.sum() > 0)
    clean_to = {INV_NAMES[a]: INV_NAMES[m["to"]] for a, m in mapping.items() if m.get("clean")}
    for k in range(1, K + 1):
        if not (lab == k).any() or k in dominant:
            continue
        nb = [q for q in (k - 1, k + 1, k - 2, k + 2) if q in clean_to]
        delta = clean_to[nb[0]] - nb[0] if nb else 0
        t = k + delta
        if 1 <= t <= K and t not in named and (t == max(named) + 1 or t == min(named) - 1):
            out[lab == k] = t
            mapping[NAMES[k]] = dict(to=NAMES[t], clean=True, bodyless=True)
    todo = (lab > 0) & (out == 0)
    if todo.any():
        out = assign_by_anatomy(lab, todo, out, body_vox, body_name, u_filled, zooms, cl, cores, labels, p, log)
    log["label_mapping"] = mapping
    log["repartitioned_frac"] = round(float(todo.sum() / max((lab > 0).sum(), 1)), 4)
    log["voxels_relabelled_frac"] = round(float((out[lab > 0] != lab[lab > 0]).mean()), 4)
    return out


def _pieces(mask):
    """26-connected components with padded bounding boxes."""
    cc, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
    if n == 0:
        return cc, []
    bbs = cc3d.statistics(cc)["bounding_boxes"]
    res = []
    for i in range(1, n + 1):
        sl = tuple(slice(max(q.start - 2, 0), min(q.stop + 2, dim)) for q, dim in zip(bbs[i], mask.shape))
        res.append((i, sl))
    return cc, res


def _split_piece(o, piece, contact, keep_vals, edt_box, zooms):
    """Split a piece between the labels it touches (watershed on the inverted distance map)."""
    seed = contact & np.isin(o, keep_vals)
    mk = np.where(seed, o, 0).astype(np.int32)
    ws = watershed(-edt_box, mk, mask=piece | seed, connectivity=3)
    miss = piece & (ws == 0)
    if miss.any():
        _, ind = ndi.distance_transform_edt(ws == 0, sampling=zooms, return_indices=True)
        ws[miss] = ws[tuple(i[miss] for i in ind)]
    o[piece] = ws[piece]


def assign_by_anatomy(lab, todo, out, body_vox, body_name, u_filled, zooms, cl, cores, labels, p, log):
    """Re-assign labels that straddle vertebrae.

    Bodies are cut at the discs; each arch piece follows the body it is attached to, and a piece
    touching two vertebrae is split between them."""
    vmm3 = float(np.prod(zooms))
    t = todo[body_vox[:, 0], body_vox[:, 1], body_vox[:, 2]] & (body_name > 0)
    out[body_vox[t, 0], body_vox[t, 1], body_vox[t, 2]] = body_name[t]
    is_body = np.zeros(out.shape, bool)
    is_body[body_vox[:, 0], body_vox[:, 1], body_vox[:, 2]] = body_name > 0
    rest = todo & (out == 0)
    st = np.ones((3, 3, 3), bool)
    pieces = []
    for k in np.unique(lab[rest]):
        cc, items = _pieces(rest & (lab == k))
        for i, sl in items:
            pieces.append((cc[sl] == i, sl))
    n_whole, n_split, n_contact = 0, 0, 0
    pending = []
    edt = None
    for piece, sl in pieces:
        ring = ndi.binary_dilation(piece, st) & ~piece
        o = out[sl]
        contact = ring & is_body[sl]
        bl = o[contact]
        bl = bl[bl > 0]
        if bl.size == 0:
            pending.append((piece, sl))
            continue
        vals, cnt = np.unique(bl, return_counts=True)
        if cnt.max() >= p["attach_dominance"] * cnt.sum():
            o[piece] = vals[cnt.argmax()]
            n_whole += 1
            continue
        if edt is None:
            edt = ndi.distance_transform_edt(u_filled, sampling=zooms)
        _split_piece(o, piece, contact, vals[cnt >= 0.15 * cnt.sum()], edt[sl], zooms)
        n_split += 1
    # pieces without body contact: join or split by contact with assigned arches (repeat, as each
    # assignment creates new contacts); pieces touching nothing are placed by position along the spine
    for _ in range(10):
        left = []
        for piece, sl in pending:
            ring = ndi.binary_dilation(piece, st) & ~piece
            o = out[sl]
            contact = ring & (o > 0)
            nb = o[contact]
            if nb.size == 0:
                left.append((piece, sl))
                continue
            vals, cnt = np.unique(nb, return_counts=True)
            if cnt.max() >= p["attach_dominance"] * cnt.sum():
                o[piece] = vals[cnt.argmax()]
            else:
                if edt is None:
                    edt = ndi.distance_transform_edt(u_filled, sampling=zooms)
                _split_piece(o, piece, contact, vals[cnt >= 0.15 * cnt.sum()], edt[sl], zooms)
                n_split += 1
            n_contact += 1
        if not left or len(left) == len(pending):
            pending = left
            break
        pending = left
    if pending:
        tree = cKDTree(cl["C"])

        def s_of(v):
            return cl["S"][tree.query(v * zooms)[1]]

        rng_ = {}
        for j, c in enumerate(cores):
            a, b = c["s_range"]
            lo_, hi_ = rng_.get(labels[j], (a, b))
            rng_[labels[j]] = (min(lo_, a), max(hi_, b))
        centre = {k: 0.5 * (a + b) for k, (a, b) in rng_.items()}
        # arch-to-body offset, measured on the arches already assigned in this scan
        arch_off = {}
        for k in centre:
            av = np.argwhere((out == k) & ~is_body)
            if len(av) > 50:
                arch_off[k] = float(np.median(s_of(av[:: max(1, len(av) // 4000)])) - centre[k])
        ks = np.array(sorted(arch_off))
        for piece, sl in pending:
            if piece.sum() * vmm3 < p["min_fragment_mm3"] or len(ks) == 0:
                continue
            pv_ = np.argwhere(piece) + np.array([q.start for q in sl])
            sp = float(np.median(s_of(pv_[:: max(1, len(pv_) // 2000)])))
            best_k, best_d = None, np.inf
            for k, c0 in centre.items():
                near = ks[np.abs(ks - k) <= 2]
                off = float(np.median([arch_off[q] for q in near])) if len(near) else 0.0
                d = abs(sp - (c0 + off)) / GAP_MU.get(min(max(k, 1), 23), 25.0)
                if d < best_d:
                    best_k, best_d = k, d
            out[sl][piece] = best_k
            n_contact += 1
        pending = []
    log["anatomy_assignment"] = dict(pieces=len(pieces), whole=n_whole, split=n_split, by_contact=n_contact,
                                     isolated=len(pending))
    return out


def clean_components(out, zooms, p, log):
    """Keep the main component of each label; give stray pieces to the label they touch."""
    vmm3 = float(np.prod(zooms))
    st = np.ones((3, 3, 3), bool)
    moved, dropped = 0, 0
    for k in range(1, NUM_LABELS + 1):
        m = out == k
        if not m.any():
            continue
        cc, items = _pieces(m)
        if len(items) <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        main = int(sizes.argmax())
        for i, sl in items:
            if i == main:
                continue
            piece = cc[sl] == i
            ring = ndi.binary_dilation(piece, st) & ~piece
            o = out[sl]
            nb = o[ring]
            nb = nb[(nb > 0) & (nb != k)]
            if nb.size:
                o[piece] = np.bincount(nb).argmax()
                moved += 1
            elif sizes[i] * vmm3 < p["min_fragment_mm3"]:
                o[piece] = 0
                dropped += 1
    log["fragments_reassigned"] = moved
    log["fragments_dropped"] = dropped
    return out


def ct_surface_refine(out, ct, p, log):
    """One-voxel surface correction from the CT: add clear bone just outside, remove soft tissue on the surface."""
    if ct is None:
        return out
    st = ndi.generate_binary_structure(3, 1)
    u = out > 0
    ring = ndi.binary_dilation(u, st) & ~u
    add = ring & (ct >= p["surface_add_hu"])
    nbl = ndi.grey_dilation(out, footprint=st)
    out[add] = nbl[add]
    inner = u & ~ndi.binary_erosion(u, st)
    rm = inner & (ct < p["surface_remove_hu"])
    out[rm] = 0
    log["surface_refine"] = dict(added=int(add.sum()), removed=int(rm.sum()))
    return out


def fill_holes(out):
    """Fill holes in each label."""
    for k in range(1, NUM_LABELS + 1):
        m = out == k
        if not m.any():
            continue
        f = ndi.binary_fill_holes(m)
        out[f & (out == 0)] = k
    return out


def refine(lab_native, affine, ct_native=None, p=PARAMS, log=None):
    """Refine a vertebrae label map (1=L5 ... 24=C1) given its affine.

    ct_native, if given, must be on the same grid as lab_native. Returns the refined label map
    (same shape and orientation); details of every decision are written into log."""
    log = {} if log is None else log
    lab_native = np.where(lab_native <= NUM_LABELS, lab_native, 0).astype(np.uint8)
    lab_ras, inv = to_ras(lab_native, affine)
    zooms = nib.affines.voxel_sizes(affine)[np.argsort(io_orientation(affine)[:, 0])].astype(float)
    ct_ras = None if ct_native is None else to_ras(ct_native.astype(np.int16), affine)[0]
    log["ct_used"] = ct_ras is not None
    out_ras = np.zeros_like(lab_ras)
    if lab_ras.max() == 0:
        log["mode"] = "empty"
        return lab_native
    lab_ras = keep_spinal_column(lab_ras, zooms, p, log)
    nzi = np.argwhere(lab_ras > 0)
    pad = np.ceil(p["crop_margin_mm"] / zooms).astype(int)
    lo = np.maximum(nzi.min(0) - pad, 0)
    hi = np.minimum(nzi.max(0) + pad + 1, lab_ras.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    lab = lab_ras[sl].copy()
    ct = None if ct_ras is None else ct_ras[sl]
    u_filled = ndi.binary_fill_holes(lab > 0)

    try:
        cl = body_centreline(u_filled, zooms, p)
        vox, s_pos, _ = tube_voxels(lab.shape, zooms, cl, p["tube_radius_factor"])
        prof = disc_profiles(lab, ct, vox, s_pos)
        bounds = detect_discs(prof, p, log)
        cores = body_cores(lab, zooms, vox, s_pos, bounds, log)
        if len(cores) < 3:
            raise RuntimeError(f"only {len(cores)} vertebral bodies found")
        labels, votes = name_instances(lab, cores, p, log)
        body_vox, body_name = body_territory(lab, zooms, cl, bounds, cores, labels, p)
        out = compose(lab, cores, labels, votes, body_vox, body_name, u_filled, zooms, cl, p, log)
        log["mode"] = "full"
    except Exception as e:  # fallback: keep the model's names, clean up only
        log["mode"] = f"cleanup_only ({type(e).__name__}: {e})"
        for key in ("disc_detection", "naming", "label_mapping", "anatomy_assignment", "n_instances",
                    "repartitioned_frac", "voxels_relabelled_frac"):
            log.pop(key, None)
        out = lab.copy()
    out = clean_components(out, zooms, p, log)
    out = fill_holes(out)
    if p["surface_refine"]:
        out = ct_surface_refine(out, ct, p, log)

    out_ras[sl] = out
    return apply_orientation(out_ras, inv).astype(np.uint8)


def process_case(pred_case_dir, ct_path, out_case_dir, p=PARAMS, qc_path=None):
    """Post-process one case folder and save the result."""
    t0 = time.time()
    log = {"case": os.path.basename(pred_case_dir.rstrip("/"))}
    ref, lab_native = load_prediction(pred_case_dir)
    ct_native = None
    if ct_path and os.path.exists(ct_path):
        cimg = nib.load(ct_path)
        if cimg.shape[:3] == ref.shape[:3] and np.allclose(cimg.affine, ref.affine, atol=1e-3):
            ct_native = np.asarray(cimg.dataobj).astype(np.int16)
        else:
            log["warning"] = "CT grid differs from the prediction grid; CT cues disabled"
    out_native = refine(lab_native, ref.affine, ct_native, p, log)
    save_case(out_case_dir, ref, out_native)
    log["runtime_s"] = round(time.time() - t0, 1)
    _write_qc(qc_path, log)
    return log


def _write_qc(qc_path, log):
    if qc_path:
        os.makedirs(os.path.dirname(os.path.abspath(qc_path)), exist_ok=True)
        with open(qc_path, "w") as f:
            json.dump(log, f, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True, help="folder with <case>/combined_labels.nii.gz")
    ap.add_argument("--ct_dir", default=None, help="folder with <case>/ct.nii.gz (optional but recommended)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--qc_dir", default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override parameters, e.g. --set surface_refine=false")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=", 1)
        old = PARAMS[k]
        PARAMS[k] = (v.lower() in ("1", "true", "yes")) if isinstance(old, bool) else type(old)(v)
    cases = args.cases or sorted(c for c in os.listdir(args.pred_dir) if os.path.isdir(os.path.join(args.pred_dir, c)))
    for c in cases:
        ct = os.path.join(args.ct_dir, c, "ct.nii.gz") if args.ct_dir else None
        qc = os.path.join(args.qc_dir, f"{c}.json") if args.qc_dir else None
        log = process_case(os.path.join(args.pred_dir, c), ct, os.path.join(args.out_dir, c), p=PARAMS, qc_path=qc)
        nm = log.get("naming", {})
        print(f"[{c}] mode={log.get('mode')} instances={log.get('n_instances')} "
              f"relabelled={log.get('voxels_relabelled_frac')} time={log.get('runtime_s')}s")
        for it in nm.get("instances", []):
            flag = "" if it["name"] == it["raw_majority"] else "   <-- renamed"
            print(f"    {it['name']:>4} (raw {it['raw_majority']:>4}, purity {it['raw_purity_of_assigned']:.2f}, "
                  f"gap {it['gap_to_next_mm']}){flag}")


if __name__ == "__main__":
    main()
