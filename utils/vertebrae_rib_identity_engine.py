"""Rib-anchored vertebral identity correction - scientific core.

PROVENANCE
----------
This module is the scientific core of the accepted BodyMaps vertebrae warm-up
solution, vendored unchanged. The body below (from ``from __future__`` to the
end of ``process``) is a byte-for-byte copy of lines 59-422 of the canonical
source file ``postprocessing_vertebrae.py``:

    SHA-256  c98e3b7233c29e860c7afddc51ce50db0518b732a81e156edae13e19971b5e01

It is kept as a literal slice, rather than rewritten to fit ShapeKit's style,
so that a single ``diff`` can demonstrate the algorithm has not changed.

For that reason the import block is also left exactly as it was, which means
``argparse``, ``json``, ``os``, ``copy`` and ``nibabel`` are imported but
unused here: they belonged to the original file's command-line wrapper, which
this module deliberately omits. They are kept so the slice stays verifiable
byte-for-byte; they can be trimmed once equivalence has been established. All ShapeKit integration - configuration, input discovery,
label-space conversion, logging and fallback handling - lives in the adapter
``vertebrae_rib_identity.py`` and never in this file.

PURPOSE
-------
Vertebra segmentation models can produce a column whose bone is well delineated
but whose level names are displaced, so that a run of vertebrae carries the
identity of a neighbouring level. This module estimates each vertebral level's
superior position from costovertebral rib attachment geometry and relabels only
those levels whose position disagrees with that estimate by more than the
canonical threshold.

The distinguishing property of this approach is the cue it uses: it reads an
external anatomical structure - the ribs - rather than deriving level identity
from the vertebral predictions themselves, their ordering, spacing, connected
components or internal consistency.

EXTERNAL INPUTS
---------------
Beyond the vertebra prediction it requires:

  * the case CT, on the same voxel grid as the prediction; and
  * a precomputed TotalSegmentator ``total`` multilabel volume, on the same
    grid, from which only the 24 rib masks (ids 92-115) are read.

Only the *geometry* of the rib masks is consumed. TotalSegmentator's vertebral
labels are deliberately not read and play no part in any decision: the
prediction model's vertebra head was fine-tuned on those labels, so they cannot
serve as an independent check on it. Nothing in this module runs
TotalSegmentator; the rib volume is supplied as a precomputed input.

Requires numpy, scipy, nibabel - all existing ShapeKit dependencies.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict

import nibabel as nib
import numpy as np
from nibabel.orientations import (aff2axcodes, apply_orientation, axcodes2ornt,
                                  ornt_transform)
from scipy import ndimage, signal

# --------------------------------------------------------------------------------------
# Label convention: index 1 = L5, rising superiorly to 24 = C1
# (SuPreM dataset/dataloader_test.py, class_map_part_vertebrae)
# --------------------------------------------------------------------------------------

CLASS_MAP: Dict[int, str] = {
    1: "vertebrae_L5", 2: "vertebrae_L4", 3: "vertebrae_L3", 4: "vertebrae_L2",
    5: "vertebrae_L1", 6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10",
    9: "vertebrae_T9", 10: "vertebrae_T8", 11: "vertebrae_T7", 12: "vertebrae_T6",
    13: "vertebrae_T5", 14: "vertebrae_T4", 15: "vertebrae_T3", 16: "vertebrae_T2",
    17: "vertebrae_T1", 18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5",
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2", 24: "vertebrae_C1",
}
N_CLASSES = len(CLASS_MAP)
SHORT = {k: v.replace("vertebrae_", "") for k, v in CLASS_MAP.items()}
CONN26 = np.ones((3, 3, 3), dtype=bool)

# ---- parameters (all physical units) --------------------------------------------------
RIB_LR_TOL_MM = 15.0      # max left/right disagreement for a rib to be trusted
MIN_DELTA_MM = 12.0       # a level is corrected only if it must move further than this
SNAP_MM = 7.0             # snap a boundary onto a CT disc trough within this distance
DEBRIS_REL = 0.02         # foreground components below this fraction are far-field debris
BONE_HU = 175.0
K_LO, K_HI = 4, 17        # solve over L2 .. T1; L2 is the pinned caudal anchor
W_RIB, W_SMOOTH, W_TIE, W_PIN = 1.0, 0.9, 0.12, 50.0
TROUGH_PROMINENCE = 0.03
TROUGH_MIN_SEP_MM = 8.0

# TotalSegmentator `total` label ids for the rib structures (v2 class map).
TS_RIB_IDS = {
    "rib_left_1": 92, "rib_left_2": 93, "rib_left_3": 94, "rib_left_4": 95,
    "rib_left_5": 96, "rib_left_6": 97, "rib_left_7": 98, "rib_left_8": 99,
    "rib_left_9": 100, "rib_left_10": 101, "rib_left_11": 102, "rib_left_12": 103,
    "rib_right_1": 104, "rib_right_2": 105, "rib_right_3": 106, "rib_right_4": 107,
    "rib_right_5": 108, "rib_right_6": 109, "rib_right_7": 110, "rib_right_8": 111,
    "rib_right_9": 112, "rib_right_10": 113, "rib_right_11": 114, "rib_right_12": 115,
}


def k_of_T(n: int) -> int:
    """Our label index for thoracic vertebra TN (T12 -> 6, ... T1 -> 17)."""
    return 6 + (12 - n)


# --------------------------------------------------------------------------------------
# Orientation (lossless axis permutation, no resampling)
# --------------------------------------------------------------------------------------

def to_ras(arr, affine):
    src = axcodes2ornt(aff2axcodes(affine))
    dst = axcodes2ornt(("R", "A", "S"))
    xf = ornt_transform(src, dst)
    return apply_orientation(arr, xf), xf


def from_ras(arr, affine):
    src = axcodes2ornt(aff2axcodes(affine))
    dst = axcodes2ornt(("R", "A", "S"))
    return apply_orientation(arr, ornt_transform(dst, src))


def ras_zooms(zooms, xf):
    out = np.zeros(3, dtype=float)
    for i in range(3):
        out[int(xf[i, 0])] = float(zooms[i])
    return out


# --------------------------------------------------------------------------------------
# Spine geometry
# --------------------------------------------------------------------------------------

def spine_centreline(fg, z, smooth_mm=25.0):
    """Per-slice robust centre of the spine, interpolated and smoothed along S."""
    ns = fg.shape[2]
    cen = np.full((ns, 2), np.nan)
    for s in range(ns):
        sl = fg[:, :, s]
        if sl.any():
            r, c = np.nonzero(sl)
            cen[s] = (np.median(r), np.median(c))
    ok = ~np.isnan(cen[:, 0])
    idx = np.flatnonzero(ok)
    if idx.size < 3:
        return None, None, None
    for j in range(2):
        cen[:, j] = np.interp(np.arange(ns), idx, cen[ok, j])
        cen[:, j] = ndimage.gaussian_filter1d(cen[:, j],
                                              sigma=max(smooth_mm / z[2], 1.0),
                                              mode="nearest")
    return cen, int(idx[0]), int(idx[-1])


def body_radius_profile(fg, cen, z, lo, hi, frac=0.55, floor_mm=8.0, cap_mm=26.0):
    """Column radius tracking the vertebral body; cervical bodies are far smaller."""
    ns = fg.shape[2]
    rad = np.full(ns, np.nan)
    rr, cc = np.meshgrid(np.arange(fg.shape[0]), np.arange(fg.shape[1]), indexing="ij")
    for s in range(lo, hi + 1):
        sl = fg[:, :, s]
        if not sl.any():
            continue
        d = np.hypot((rr[sl] - cen[s, 0]) * z[0], (cc[sl] - cen[s, 1]) * z[1])
        rad[s] = np.clip(np.percentile(d, 70) * frac, floor_mm, cap_mm)
    ok = ~np.isnan(rad)
    rad = np.interp(np.arange(ns), np.flatnonzero(ok), rad[ok])
    return ndimage.gaussian_filter1d(rad, sigma=max(30.0 / z[2], 1.0), mode="nearest")


def bone_profile(ct, cen, rad, z, lo, hi, bone_hu=BONE_HU):
    """Fraction of the tracking column that is bone, per superior slice."""
    ns = ct.shape[2]
    prof = np.zeros(ns)
    rr, cc = np.meshgrid(np.arange(ct.shape[0]), np.arange(ct.shape[1]), indexing="ij")
    for s in range(lo, hi + 1):
        d2 = ((rr - cen[s, 0]) * z[0]) ** 2 + ((cc - cen[s, 1]) * z[1]) ** 2
        col = d2 < rad[s] ** 2
        n = col.sum()
        if n:
            prof[s] = np.count_nonzero(col & (ct[:, :, s] > bone_hu)) / n
    return prof


def debris_filtered(labels):
    """Foreground with far-field specks removed - used only to locate the centreline."""
    fg = labels > 0
    if not fg.any():
        return fg
    cc, n = ndimage.label(fg, structure=CONN26)
    sz = np.bincount(cc.ravel())
    sz[0] = 0
    return np.isin(cc, [i for i in range(1, n + 1) if sz[i] > DEBRIS_REL * sz.max()])


def level_stats(labels, z):
    """Centroid of each level's dominant component, and that component's share."""
    objs = ndimage.find_objects(labels.astype(np.int32), max_label=N_CLASSES)
    centroid, frac = {}, {}
    for k in range(1, N_CLASSES + 1):
        sl = objs[k - 1]
        if sl is None:
            continue
        sub = labels[sl] == k
        if not sub.any():
            continue
        ccl, _ = ndimage.label(sub, structure=CONN26)
        s2 = np.bincount(ccl.ravel())
        s2[0] = 0
        main = ccl == int(s2.argmax())
        w = main.sum(axis=(0, 1)).astype(float)
        nz = np.flatnonzero(w)
        cum = np.cumsum(w[nz]) / w[nz].sum()
        centroid[k] = float(np.interp(0.5, cum, nz + sl[2].start)) * z[2]
        frac[k] = float(s2.max()) / float(sub.sum())
    return centroid, frac


# --------------------------------------------------------------------------------------
# Rib evidence
# --------------------------------------------------------------------------------------

def rib_attachments(ts, cen, z, log):
    """Superior coordinate where each rib pair meets the spine, left/right cross-checked."""
    def one(mask):
        idx = np.nonzero(mask)
        if idx[0].size < 50:
            return None
        rv, av, sv = idx
        d = np.hypot((rv - cen[sv, 0]) * z[0], (av - cen[sv, 1]) * z[1])
        return float(np.median(sv[d <= np.percentile(d, 10)])) * z[2]

    ribs, excluded = {}, []
    for num in range(1, 13):
        la = one(ts == TS_RIB_IDS[f"rib_left_{num}"])
        ra = one(ts == TS_RIB_IDS[f"rib_right_{num}"])
        if la is None or ra is None:
            continue
        if abs(la - ra) > RIB_LR_TOL_MM:
            excluded.append({"rib": num, "lr_mm": round(abs(la - ra), 1)})
            continue
        ribs[num] = float(np.mean([la, ra]))
    log["ribs_used"] = sorted(ribs)
    log["ribs_excluded_lr"] = excluded
    return ribs


def solve_centroids(ribs, centroid, frac, log):
    """Least squares: rib equations + smoothness + model tie + pinned lumbar anchor."""
    ks = list(range(K_LO, K_HI + 1))
    idx = {k: i for i, k in enumerate(ks)}
    rows, rhs, wts = [], [], []

    def add(row, val, w):
        rows.append(row); rhs.append(val); wts.append(w)

    for num, meas in ribs.items():
        r = np.zeros(len(ks))
        if num == 1 or num >= 10:
            k = k_of_T(num)
            if k not in idx:
                continue
            r[idx[k]] = 1.0
        else:
            ka, kb = k_of_T(num - 1), k_of_T(num)
            if ka not in idx or kb not in idx:
                continue
            r[idx[ka]] = 0.5
            r[idx[kb]] = 0.5
        add(r, meas, W_RIB)

    for i in range(1, len(ks) - 1):
        r = np.zeros(len(ks))
        r[i - 1], r[i], r[i + 1] = 1.0, -2.0, 1.0
        add(r, 0.0, W_SMOOTH)

    for k in ks:
        if k not in centroid:
            continue
        r = np.zeros(len(ks))
        r[idx[k]] = 1.0
        add(r, centroid[k], W_TIE * max(frac.get(k, 0.0), 0.05))

    if K_LO in centroid:
        r = np.zeros(len(ks))
        r[idx[K_LO]] = 1.0
        add(r, centroid[K_LO], W_PIN)

    A = np.array(rows) * np.array(wts)[:, None]
    b = np.array(rhs) * np.array(wts)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    solved = {k: float(sol[idx[k]]) for k in ks}
    delta = {k: solved[k] - centroid[k] for k in ks if k in centroid}
    log["solved_mm"] = {SHORT[k]: round(v, 1) for k, v in solved.items()}
    log["delta_mm"] = {SHORT[k]: round(v, 1) for k, v in delta.items()}
    return solved, delta


def disc_troughs(labels, ct, z):
    """Candidate disc planes, in millimetres, from CT attenuation along the column."""
    fgc = debris_filtered(labels)
    cen, lo, hi = spine_centreline(fgc, z)
    if cen is None:
        return np.array([])
    rad = body_radius_profile(fgc, cen, z, lo, hi)
    prof = bone_profile(ct, cen, rad, z, lo, hi)
    seg = ndimage.gaussian_filter1d(prof[lo:hi + 1], sigma=max(1.8 / z[2], 0.8),
                                    mode="nearest")
    win = int(max(60.0 / z[2], 5))
    env = ndimage.gaussian_filter1d(
        ndimage.maximum_filter1d(seg, size=win, mode="nearest"),
        sigma=max(win / 4, 1.0), mode="nearest")
    norm = seg / np.maximum(env, 1e-6)
    tr, _ = signal.find_peaks(-norm, distance=max(TROUGH_MIN_SEP_MM / z[2], 2),
                             prominence=TROUGH_PROMINENCE)
    return (tr + lo) * z[2]


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def process(labels_native, affine, zooms_native, ct_native, ts_native):
    log: dict = {}
    lab, xf = to_ras(labels_native, affine)
    z = ras_zooms(zooms_native, xf)
    ct = to_ras(ct_native, affine)[0] if ct_native is not None else None
    before = lab.copy()

    if ts_native is None or ct is None:
        log["status"] = "no rib volume or CT supplied - no correction applied"
        return labels_native, log

    ts = to_ras(ts_native, affine)[0]
    fgc = debris_filtered(lab)
    cen, _, _ = spine_centreline(fgc, z)
    if cen is None:
        log["status"] = "spine centreline unavailable"
        return labels_native, log

    centroid, frac = level_stats(lab, z)
    ribs = rib_attachments(ts, cen, z, log)
    if len(ribs) < 4:
        log["status"] = "too few usable ribs - no correction applied"
        return labels_native, log

    solved, delta = solve_centroids(ribs, centroid, frac, log)
    corr = sorted(k for k, v in delta.items() if abs(v) > MIN_DELTA_MM)
    log["levels_corrected"] = [SHORT[k] for k in corr]

    if corr:
        lo_k = max(min(corr) - 1, 1)
        hi_k = min(max(corr) + 1, N_CLASSES)
        seq = [k for k in range(lo_k, hi_k + 1) if k in solved]
        troughs = disc_troughs(lab, ct, z)

        bounds = []
        for i in range(len(seq) - 1):
            mid = (solved[seq[i]] + solved[seq[i + 1]]) / 2.0
            if len(troughs):
                j = int(np.argmin(np.abs(troughs - mid)))
                if abs(troughs[j] - mid) <= SNAP_MM:
                    mid = float(troughs[j])
            bounds.append(mid)
        log["boundaries_mm"] = [round(b, 1) for b in bounds]

        edges = [-1e9] + bounds + [1e9]
        slab = np.zeros(lab.shape[2], dtype=np.int16)
        for i, k in enumerate(seq):
            a = int(np.ceil(max(edges[i], 0) / z[2]))
            bb = int(np.floor(min(edges[i + 1], (lab.shape[2] - 1) * z[2]) / z[2]))
            if bb >= a:
                slab[a:bb + 1] = k

        in_band = np.isin(lab, list(set(seq)))
        moved = 0
        for s in range(lab.shape[2]):
            if slab[s] == 0:
                continue
            m = in_band[:, :, s]
            if m.any():
                prev = lab[:, :, s][m]
                lab[:, :, s][m] = slab[s]
                moved += int((prev != slab[s]).sum())
        log["voxels_moved"] = moved
    else:
        log["voxels_moved"] = 0

    # cavities inside a level, claiming only voxels nothing else owns
    objs = ndimage.find_objects(lab.astype(np.int32), max_label=N_CLASSES)
    added = 0
    for k in range(1, N_CLASSES + 1):
        sl = objs[k - 1]
        if sl is None:
            continue
        sub = lab[sl] == k
        if not sub.any():
            continue
        gain = ndimage.binary_fill_holes(sub) & ~sub & (lab[sl] == 0)
        if gain.any():
            blk = lab[sl]
            blk[gain] = k
            added += int(gain.sum())
    log["hole_fill"] = added

    fgb, fga = before > 0, lab > 0
    kept = int((fgb & fga & (before == lab)).sum())
    log["churn"] = {"kept": kept,
                    "relabelled": int((fgb & fga & (before != lab)).sum()),
                    "fraction_untouched": round(kept / max(int(fgb.sum()), 1), 4)}
    return from_ras(lab, affine).astype(np.uint8), log
