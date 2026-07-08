# -*- coding: utf-8 -*-
"""
vertebrae_reidentification.py
=============================

Anatomic-consistency **re-identification** of vertebra labels for ShapeKit.

This module provides a stable replacement for the (currently unfinished)
size-based `reallocate_based_on_size` step in `vertebrae_postprocessing.py`.
The dominant error of TotalSegmentator / SuPreM-style vertebra predictions is
mis-**identification** in the mid / thoraco-lumbar spine: the network segments
the *bone* well but assigns the wrong *level* (e.g. it under-segments one
vertebra and shifts every level above it by one).  A per-level Dice therefore
collapses in the middle while the anchored ends stay high.

Because Dice punishes deleted true-positive bone, the method is **non-
destructive**: it re-labels rather than deletes, and — crucially — it keeps the
model's own (Dice-optimal) mask on every vertebra that is already correct,
rebuilding only the mis-identified span.

Method (see the accompanying report for the derivation):
  1. Treat the union of all vertebra labels as spine bone; isolate vertebral
     **body cores** with a physical distance transform and grow them back over
     all bone by a nearest-core watershed -> ordered vertebra instances.
  2. Anchor the sequence: each confident instance votes an offset
     (model_label - rank); a strong agreement means the count is reliable
     (the two anchored ends must agree -- the anatomic-consistency cycle),
     otherwise the case is left as the model.
  3. Copy the model mask through on every correctly-labelled vertebra; rebuild
     only the shifted bone (plus a small superior buffer whose model boundaries
     the compression corrupts) with a watershed of the shifted cores.
  4. Keep the largest connected component per vertebra, fill interior holes, and
     accept the result only if every level is strictly ordered along the spine.

Works in world space via the affine, so it is orientation- and spacing-aware.

Author: Rao Farhat Masood  (contributed to ShapeKit)
"""

import numpy as np
from collections import defaultdict
from scipy import ndimage

try:
    import cc3d
    _HAS_CC3D = True
except Exception:                                    # pragma: no cover
    _HAS_CC3D = False

# Vertebra order used internally: 1 = L5 (most inferior) ... 24 = C1 (most superior)
ORDER = ["vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2",
         "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10",
         "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
         "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2",
         "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
         "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1"]
NAME_TO_ID = {n: i + 1 for i, n in enumerate(ORDER)}
ID_TO_NAME = {i + 1: n for i, n in enumerate(ORDER)}
NUM_CLASSES = 24


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _connected_components(mask, connectivity=26):
    if _HAS_CC3D:
        return cc3d.connected_components(mask.astype(np.uint8),
                                         connectivity=connectivity, return_N=True)
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    return lab, n


def _bbox(mask):
    if not mask.any():
        return None
    c = np.array(np.nonzero(mask))
    lo, hi = c.min(1), c.max(1) + 1
    return tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))


def _voxel_spacing(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


# --------------------------------------------------------------------------- #
#  core re-identification
# --------------------------------------------------------------------------- #
def reidentify_core(label_vol, affine, r_mm=5.0, min_core_ml=0.3,
                    conf=0.6, agree_thr=0.8, buffer_sup=2):
    """Return the re-identified 1..24 label volume, or ``None`` if unreliable."""
    spacing = _voxel_spacing(affine)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    rot, trans = affine[:3, :3], affine[:3, 3]

    bone = ndimage.binary_fill_holes(label_vol > 0)
    if not bone.any():
        return None
    bb = _bbox(bone)
    lo = np.array([bb[d].start for d in range(3)])
    B, Vs = bone[bb], label_vol[bb]

    edt = ndimage.distance_transform_edt(B, sampling=spacing)
    clab, nc = _connected_components(edt > r_mm)
    csz = np.bincount(clab.ravel()); csz[0] = 0
    keep = [c for c in range(1, nc + 1) if csz[c] >= min_core_ml / voxel_ml]
    if len(keep) < 6:
        return None

    def core_z(c):
        return float(rot[2] @ (np.array(np.nonzero(clab == c)).mean(1) + lo) + trans[2])
    order = sorted(keep, key=core_z)
    remap = np.zeros(nc + 1, int)
    for rank, c in enumerate(order, 1):
        remap[c] = rank
    clab = remap[clab]
    K = len(order)

    idx = ndimage.distance_transform_edt(clab == 0, return_indices=True, sampling=spacing)[1]
    inst = np.where(B, clab[tuple(idx)], 0)

    maj, votes = {}, []
    for i in range(1, K + 1):
        labs = Vs[inst == i]; labs = labs[labs > 0]
        if labs.size == 0:
            maj[i] = 0; continue
        vals, cnts = np.unique(labs, return_counts=True)
        j = int(np.argmax(cnts)); maj[i] = int(vals[j])
        if cnts[j] / cnts.sum() >= conf:
            votes.append(int(vals[j]) - i)
    if not votes:
        return None
    vv, vc = np.unique(votes, return_counts=True)
    offset = int(vv[np.argmax(vc)])
    if vc.max() / len(votes) < agree_thr:
        return None

    corr = {i: i + offset for i in range(1, K + 1)}
    label_insts = defaultdict(list)
    for i in range(1, K + 1):
        if maj[i] > 0:
            label_insts[maj[i]].append(i)

    wrong = {i for i in range(1, K + 1)
             if maj[i] > 0 and 1 <= corr[i] <= NUM_CLASSES
             and corr[i] != maj[i] and abs(corr[i] - maj[i]) <= 1}
    watershed_insts = []
    if wrong:
        wrong_majs = {maj[i] for i in wrong}
        span = set(wrong) | {i for i in range(1, K + 1) if maj[i] in wrong_majs}
        buf, j = [], max(span) + 1
        while j <= K and len(buf) < buffer_sup:
            if 1 <= corr[j] <= NUM_CLASSES and abs(corr[j] - maj[j]) <= 1:
                buf.append(j); j += 1
            else:
                break
        watershed_insts = sorted(span | set(buf))

    ws_majs = {maj[i] for i in watershed_insts}
    correct_M = set()
    for M, insts_M in label_insts.items():
        if M in ws_majs:
            continue
        if (len(insts_M) == 1 and corr[insts_M[0]] == M) or \
           any(abs(corr[i] - M) > 1 for i in insts_M):
            correct_M.add(M)

    out = np.zeros_like(Vs)
    for M in correct_M:
        out[Vs == M] = M
    covered = np.isin(Vs, list(correct_M)) if correct_M else np.zeros(Vs.shape, bool)
    shifted = B & ~covered
    if watershed_insts and shifted.any():
        seed = np.isin(clab, watershed_insts)
        sidx = ndimage.distance_transform_edt(~seed, return_indices=True, sampling=spacing)[1]
        near = clab[tuple(sidx)]
        lut = np.zeros(K + 1, int)
        for i in watershed_insts:
            lut[i] = corr[i]
        out[shifted] = lut[near[shifted]]

    for v in range(1, NUM_CLASSES + 1):                  # never let a level vanish
        if (Vs == v).any() and not (out == v).any():
            out[Vs == v] = v

    full = np.zeros_like(label_vol)
    full[bb] = out
    return full


def keep_largest_component(label_vol):
    """One connected component per vertebra (removes leak false-positives)."""
    out = label_vol.copy()
    for k in range(1, NUM_CLASSES + 1):
        bb = _bbox(out == k)
        if bb is None:
            continue
        cc, n = _connected_components(out[bb] == k)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        big = int(sizes.argmax())
        drop = (cc > 0) & (cc != big)
        if drop.any():
            region = out[bb]; region[drop] = 0; out[bb] = region
    return out


def fill_interior_holes(label_vol):
    out = label_vol.copy()
    for k in range(1, NUM_CLASSES + 1):
        bb = _bbox(out == k)
        if bb is None:
            continue
        region = out[bb]
        holes = ndimage.binary_fill_holes(region == k) & (region == 0)
        if holes.any():
            region[holes] = k; out[bb] = region
    return out


def _is_monotonic(label_vol, affine):
    rot, trans = affine[:3, :3], affine[:3, 3]
    zs = [float(rot[2] @ np.array(np.nonzero(label_vol == k)).mean(1) + trans[2])
          for k in range(1, NUM_CLASSES + 1) if (label_vol == k).any()]
    return all(zs[i] < zs[i + 1] for i in range(len(zs) - 1))


# --------------------------------------------------------------------------- #
#  ShapeKit entry point
# --------------------------------------------------------------------------- #
def reidentify_vertebrae_dict(segmentation_dict, reference_img, logger=None,
                              patient_id=""):
    """
    ShapeKit-facing wrapper.

    Parameters
    ----------
    segmentation_dict : dict {vertebra_name: np.ndarray(bool/uint8 mask)}
    reference_img     : nibabel image whose affine gives voxel spacing / axes
    logger            : optional logging.Logger

    Returns the segmentation_dict with corrected vertebra masks. Falls back to
    the input labels (unchanged) whenever re-identification is judged unreliable
    or the result is not anatomically ordered.
    """
    def _log(msg):
        if logger is not None:
            logger.info(msg)

    present = [n for n in ORDER if n in segmentation_dict
               and np.any(segmentation_dict[n])]
    if len(present) < 6:
        _log(f"[Vertebrae] {patient_id}: <6 levels present, skipping re-identification.")
        return segmentation_dict

    shape = np.asarray(segmentation_dict[present[0]]).shape
    combined = np.zeros(shape, dtype=np.uint8)
    for name in ORDER:
        m = segmentation_dict.get(name)
        if m is not None and np.any(m):
            combined[np.asarray(m) > 0] = NAME_TO_ID[name]

    affine = reference_img.affine
    reid = reidentify_core(combined, affine)
    if reid is None:
        _log(f"[Vertebrae] {patient_id}: re-identification unreliable -> kept model labels.")
        result = combined
    else:
        result = reid

    result = keep_largest_component(result)
    result = fill_interior_holes(result)

    if reid is not None and not _is_monotonic(result, affine):
        _log(f"[Vertebrae] {patient_id}: non-monotonic result -> reverted to model labels.")
        result = fill_interior_holes(keep_largest_component(combined))
    elif reid is not None:
        _log(f"[Vertebrae] {patient_id}: re-identification applied.")

    out = dict(segmentation_dict)
    for k, name in ID_TO_NAME.items():
        mask = (result == k).astype(np.uint8)
        if mask.any() or name in segmentation_dict:
            out[name] = mask
    return out
