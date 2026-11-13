#!/usr/bin/env python
"""Clean up SuPreM Swin-UNETR vertebra masks (C1-L5) on abdominal CT, and flag what still looks wrong.

Usage:
    python postprocessing_vertebrae.py --input PRED_ROOT --ct_root CT_ROOT --output OUT_ROOT
        [--cases CASE ...] [--log_dir DIR] [--verbose] [--timing] [--no_relabel]

    PRED_ROOT/<case>/combined_labels.nii.gz  and  segmentations/vertebrae_{C1..L5}.nii.gz   (SuPreM output)
    CT_ROOT/<case>/ct.nii.gz                                      (optional; the CT checks are skipped without it)
    --no_relabel skips the two repairs that can change which level a voxel belongs to: label drift (stage 3)
    and the lost-level repair (in stage 4).
Output per case: the same files under OUT_ROOT (same names, shape, affine, header, uint8) plus
postprocessing_log.json, which lists every change and every flag.

Label map (SuPreM): 0 background, 1 = L5, 2 = L4, ..., 5 = L1, 6 = T12, ..., 17 = T1, 18 = C7, ..., 24 = C1.
A bigger value is a more superior vertebra.  No array axis is assumed to be "up": positions go through the
image affine to world millimetres (RAS+, so world coordinate 2 = S grows towards the head), and every
threshold is in mm, mm^3 or HU.

The pipeline, in order (everything runs on a bounding-box crop of the labels):
    1. Load + crop      read the labels (and the CT), cut away the empty space around the spine.
    2. Clean-up         delete tiny specks, decide the fate of detached fragments and far debris, and drop
                        label blobs that lie far off the spinal chain (field-of-view outliers).
    3. Label drift      if label boundaries cut through vertebral bodies instead of the discs, split the
                        bone at the discs and give every vertebra one label (counted between pure anchors).
    4. Sacrum + level   find the sacrum below the lowest vertebra; if the sacrum confirms L5 and the
                        chain is one level too low above a duplicated label, restore the lost level.
    5. Order check      look at the vertebra centroids along S and FLAG inverted or oversized steps.
                        It never changes a voxel.
    Then the files and the JSON log are written (plumbing at the end of this file).

What we deliberately do NOT do, and why:
    - No global relabelling from priors (no Viterbi over spacings).  SuPreM's raw level identity is already
      0.95 on typical spines, and a wrong whole-chain shift is catastrophic (Dice 0.97 -> 0.00 on a real case),
      while a missed one just leaves the network's labels.  We count only where the CT supports it (stage 4), else flag.
    - No HU gating of mask interiors and no dilation into bone: no strong published system does it, gating
      touched voxels in only 1 of 41 real cases (no metric change), and dilation cost Dice on correct masks.
    - No cavity filling and no majority vote over mixed blocks: on real masks they moved Dice by 0.0004 or less.
    - No "unlabelled vertebra below the lowest label" shift: that image cue fired wrongly on 2 of 2 real cases.
    - No L6 / T13: SuPreM has only 24 classes, so a transitional vertebra cannot be written.

Known weak spot: the count in stage 4 assumes the lowest vertebra really is L5.  On one spine with an L6 in the
held-out test that assumption moved 14 vertebrae the wrong way.  If your data has many transitional spines, run
with --no_relabel.

Only numpy, scipy, nibabel and the standard library are used.  The script is deterministic (no randomness,
stable sorts, fixed tie-breaks).  Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import io
import itertools
import json
import math
import os
import platform
import shutil
import struct
import sys
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import scipy
from nibabel.affines import apply_affine
from scipy import ndimage as ndi

# =============================================================================================================
# Constants.  Format:  value  # why [source].
# [report] = reports/Vertebra mask postprocessing methods.md   [tune] = the 9 tuning cases s1411..s1424
# [BDMAP31] = demo case BDMAP_00000031   [inference] = my engineering judgement   [TS] = TotalSegmentator
# [Mayur] = the vertebra engine by MayurGP25 merged into BodyMaps/ShapeKit   [Meng] = Meng et al. 2023
# [Payer] = Payer et al., VerSe winner   [inspect] = the same constant in inspect_predictions.py
# =============================================================================================================
STRUCT_26 = np.ones((3, 3, 3), dtype=bool)  # thin pedicle/process bridges are often diagonal, so 26-connectivity [evaluate CONNECTIVITY=26]
CROP_PAD_VOX = 3  # slack around the cropped bone so neighbour look-ups never fall off the array [inference]
AFFINE_ATOL = 1e-3  # CT and mask share a grid when their affines agree to a micrometre [inspect AFFINE_ATOL]

# --- clean-up: specks, fragments, debris
MIN_SPECK_MM3 = 100.0  # an isolated island under 0.1 mL is noise, about 1/400 of a lumbar vertebra [report; TS min_size_mm3=100]
NONBONE_MEDIAN_HU = 100.0  # a piece with median HU < 100 is soft tissue (muscle is 40-60 HU), not bone [inspect; inference]
REATTACH_GAP_MM = 5.0  # bone within 5 mm of a vertebra is a detached process or a partial-volume split of it [report; ShapeKit <=5 mm]
MIN_ISLAND_MM3 = 500.0  # an isolated piece under 0.5 mL farther than 5 mm is not worth keeping [report; Mayur min_island_ml=0.5]
ATTACH_GAP_MM = 20.0  # nothing farther than 20 mm from every vertebra belongs to the spinal column [report; Mayur max_gap_mm=20]
MIN_VERTEBRA_MM3 = 3910.0  # half the smallest training vertebra; smaller isolated pieces are not a whole vertebra [report; Meng 2023]
NEGLIGIBLE_INSTANCE_MM3 = MIN_VERTEBRA_MM3  # an unclipped instance below this is a fragment, not a level: it is ignored by the sacrum, lost-level and order checks [report; Meng 2023; s1413]
INSTANCE_MIN_FRACTION = 0.5  # a separate piece of at least half the median vertebra is a vertebra of its own (a duplicated level) [report RESIDUAL_FRAC; Meng]
FOV_EDGE_MM = 10.0  # a vertebra within 10 mm of the top/bottom image edge is clipped, so its shape is unreliable [report; Payer]
SPACING_GATE_MM = (12.5, 50.0)  # one level step is 12.5-50 mm between centroids; the upper end also bounds the order check and the isolation test [report; Payer, VerSe'19]
FOV_ISOLATION_MM = SPACING_GATE_MM[1]  # an instance farther than one legal level step from all others is off the chain [report; Payer]

# --- label drift: split the bone into vertebrae at the discs, one label per vertebra
DRIFT_COARSE_MM = 2.0  # the split works on a ~2 mm grid: disc gaps (3-10 mm) survive and the distance transform stays fast [inference; BDMAP31]
DRIFT_EDT_STEP_MM = 1.0  # threshold step of the body search on the distance transform, about half a coarse voxel [inference]
DRIFT_BODY_PEAK_MM = 5.0  # a body seed is at least 5 mm from the background (10 mm wide); pedicles and processes are thinner [inference; BDMAP31]
DRIFT_BODY_PROMINENCE_MM = 3.0  # a body peak stands 3 mm above the saddle joining it to the next body (bridges 2-4 mm, bodies 8-12 mm) [inference; BDMAP31]
DRIFT_MINORITY_MIN = 0.10  # a body is "drifted" when an adjacent level's label covers >= 10 % of its core (clean cases <= 0.09, BDMAP31 0.18-0.46) [tune; BDMAP31]
DRIFT_MIN_DRIFTED = 2  # a drift run needs two drifted bodies; one capped body is a merge or a splitter slip, left alone [tune]
DRIFT_CLUSTER_GAP = 1  # drifted bodies with at most one other body between them form one run [BDMAP31]
DRIFT_REGION_VOLUME_RANGE = (0.4, 2.0)  # a region is one whole vertebra when its volume is 0.4-2.0 x the median of its +-2 neighbours [inference]
DRIFT_ANCHOR_PURITY = 0.90  # a body whose core is >= 90 % one label (no adjacent cap) is a pure anchor to count from [tune; BDMAP31]
DRIFT_MIN_SEGMENT = 4  # fewer than 4 consecutive trustworthy vertebrae say nothing about order [inference]
DRIFT_MAX_SKIP_LEVELS = 1  # between two anchors the run may have at most one more level than bodies (one undetected level); more means abstain [inference]
DRIFT_MAX_REGIONS = 250  # seed ids live in a uint8 map [engineering]
VOTE_PSEUDOCOUNT = 1.0  # Laplace smoothing of the per-body label votes [report alpha]
TIE_EPS = 1e-6  # deterministic tie-break between equally good level assignments [report tie rule]
VOLUME_NEIGHBOURS = 2  # neighbours on each side used for the "typical vertebra volume" [report]
SPACING_MM = {  # (mean, sd) of consecutive centroid distances in adults [report; VerSe'20 via Meng 2023 Supp. B]
    "cervical": (16.77, 2.18),
    "thoracic": (23.32, 3.55),
    "lumbar": (32.68, 2.84),
}

# --- lost level: the network lost one thoracolumbar vertebra, so every label above it is one level too low
# (This is the stage I would ship switched off if I did it again; --no_relabel does exactly that.)
LOST_MIN_BELOW = 3  # counting up from the confirmed L5, 3 vertebrae (L5, L4, L3) must carry exactly their counted label before the count is trusted [inference; s1423]
LOST_MIN_SHIFTED = 3  # at least 3 vertebrae in a row above the duplicate carry the label of the level below; one or two is a local mislabel [inference; s1423]
LOST_OWN_LABEL_MIN = 0.5  # in a shifted vertebra the network's label is still the majority of the body; 50/50 is drift, not a lost level [inference; s1423]
LOST_GAP_SIGMA = 3.0  # the two vertebrae sharing a label must be a whole level apart along S (>= mean - 3 sd); a body cut in two has halves ~half a level apart [report; Meng 2023; s1422]
LOST_OVERSIZE = 2.0  # a region more than 2x the median of its neighbours is probably two bodies seen as one [inference]

# --- sacrum anchor (heuristic)
HU_BONE = 200.0  # smoothed CT >= 200 HU is bone [report; Lessmann 2019]
SACRUM_SLAB_MM = 1.5 * 32.68  # look 1.5 lumbar spacings (~49 mm) below the lowest vertebra [report anchors]
SACRUM_MIDLINE_MM = 25.0  # the sacral body lies within 25 mm of the spinal axis [report anchors]
SACRUM_SEARCH_RADIUS_MM = 100.0  # the pelvic ring reaches about 100 mm from the midline [inference]
SACRUM_MIN_BONE_MM3 = 20000.0  # >= 20 mL of unlabelled bone below means sacrum/pelvis, not a speck [report anchors]
SACRUM_LATERAL_RATIO = 1.5  # pelvic bone spans >= 1.5 x the lowest vertebra's half-width [inference]
SACRUM_SMOOTH_VOX = 3  # 3x3x3 mean before the HU threshold, to suppress noise [report anchors]

# Everything above, as written to the log of every case (name -> value).
SETTINGS = {k: v for k, v in list(globals().items()) if k.isupper() and isinstance(v, (int, float, tuple, dict))}

FLOAT_DECIMALS = 3
INF = math.inf

# =============================================================================================================
# Label map and small helpers
# =============================================================================================================
LABEL_NAMES = {}
for _i, _n in enumerate(
    [f"L{k}" for k in range(5, 0, -1)] + [f"T{k}" for k in range(12, 0, -1)] + [f"C{k}" for k in range(7, 0, -1)]
):
    LABEL_NAMES[_i + 1] = _n
NUM_LABELS = 24
N_VALUES = NUM_LABELS + 1  # 0 (background) plus the 24 levels
VALUE_L5 = 1


def label_name(value):
    """Vertebra name of a label value (1 -> 'L5')."""
    return LABEL_NAMES.get(int(value), f"value_{int(value)}")


def region_of(value):
    """'lumbar', 'thoracic' or 'cervical' for a label value 1..24."""
    return "lumbar" if value <= 5 else "thoracic" if value <= 17 else "cervical"


def to_jsonable(obj):
    """Numpy scalars/arrays, tuples and non-finite floats -> JSON-safe values (floats rounded)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return round(f, FLOAT_DECIMALS)
    return obj


def pad_box(box, pad, shape):
    """Grow a tuple of slices by ``pad`` voxels per side (int or per-axis), clipped to ``shape``."""
    pads = np.broadcast_to(np.asarray(pad, dtype=int), (len(shape),))
    return tuple(slice(max(s.start - int(p), 0), min(s.stop + int(p), n)) for s, p, n in zip(box, pads, shape))


def nonzero_bbox(arr):
    """Bounding box (tuple of slices) of the non-zero voxels of ``arr``, or None when it is empty."""
    plane = arr.max(axis=0) if arr.size else np.zeros((0, 0))
    profiles = [arr.reshape(arr.shape[0], -1).max(axis=1) if arr.size else np.zeros(0), plane.max(axis=1), plane.max(axis=0)]
    box = []
    for p in profiles:
        idx = np.flatnonzero(p)
        if idx.size == 0:
            return None
        box.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def sub_box(outer, inner):
    """Express slices ``inner`` (same frame as ``outer``) relative to the start of ``outer``."""
    return tuple(slice(i.start - o.start, i.stop - o.start) for o, i in zip(outer, inner))


class Geometry:
    """Voxel size, voxel volume and axis codes derived from the affine.

    Ordering along the spine always uses world S from ``apply_affine``; ``si_axis`` (the voxel axis that
    runs most nearly head-foot) is only used to ask "is this piece cut by the top/bottom image face?".
    """

    def __init__(self, affine, shape):
        self.affine = np.asarray(affine, dtype=np.float64)
        self.shape = tuple(int(s) for s in shape)
        self.spacing = np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0))
        self.vox_mm3 = float(abs(np.linalg.det(self.affine[:3, :3])))
        self.axcodes = "".join(nib.aff2axcodes(self.affine))
        self.si_axis = int(np.argmax(np.abs(self.affine[2, :3])))
        self.inv_affine = np.linalg.inv(self.affine)

    def world(self, ijk):
        """World RAS mm of voxel coordinates (N x 3 or 3,)."""
        return apply_affine(self.affine, ijk)

    def as_dict(self):
        return {"shape": list(self.shape), "axcodes": self.axcodes, "spacing_mm": self.spacing,
                "voxel_volume_mm3": self.vox_mm3, "superior_voxel_axis": self.si_axis,
                "world_S_mm_per_index_step": self.affine[2, :3], "affine": self.affine}


class CaseLog:
    """Collects every change or flag (step, action, label, voxels, reason) and prints it with --verbose."""

    def __init__(self, case, vox_mm3=1.0, verbose=False):
        self.case, self.vox_mm3, self.verbose, self.changes = case, vox_mm3, verbose, []

    def add(self, step, action, label=None, voxels=0, reason="", **extra):
        entry = {"step": step, "action": action}
        if label is not None:
            entry["label"] = label_name(label) if isinstance(label, (int, np.integer)) else label
        entry.update(voxels=int(voxels), mm3=float(voxels) * self.vox_mm3, reason=reason)
        for k, v in extra.items():
            entry[k] = label_name(v) if k in ("to", "from") and isinstance(v, (int, np.integer)) else v
        self.changes.append(to_jsonable(entry))
        if self.verbose:
            e = self.changes[-1]
            print(f"[{self.case}] {e['step']:<20s} {e['action']:<10s} {e.get('label', '')} "
                  + (f"-> {e['to']} " if "to" in e else "") + (f"{e['voxels']} vox " if e["voxels"] else "")
                  + (f": {e['reason']}" if e["reason"] else ""))

    def summary(self):
        """Per-step counts of entries, and voxels / mm3 by action."""
        out = {}
        for e in self.changes:
            s = out.setdefault(e["step"], {"entries": 0})
            a = e["action"]
            s["entries"] += 1
            s[f"n_{a}"] = s.get(f"n_{a}", 0) + 1
            s[f"voxels_{a}"] = s.get(f"voxels_{a}", 0) + e["voxels"]
            s[f"mm3_{a}"] = round(s.get(f"mm3_{a}", 0.0) + e["mm3"], FLOAT_DECIMALS)
        return out


# =============================================================================================================
# Stage 1: load + crop
# =============================================================================================================
def read_nifti_array(path):
    """Return (img, array) for a NIfTI file, scaled as nibabel would scale it.

    For an unscaled single-member ``.nii.gz`` the file is inflated with one zlib call (fast, releases the
    GIL) and viewed in place (F-ordered, read-only).  Anything else falls back to ``img.dataobj``.
    """
    img = nib.load(path)
    proxy = img.dataobj
    ok = all(hasattr(proxy, a) for a in ("slope", "inter", "offset", "dtype", "order", "shape"))
    if ok and path.endswith(".gz") and float(proxy.slope) == 1.0 and float(proxy.inter) == 0.0 and proxy.order == "F":
        with open(path, "rb") as fh:
            raw = fh.read()
        inflater = zlib.decompressobj(zlib.MAX_WBITS | 16)  # 16 = gzip framing
        buf = inflater.decompress(raw)
        del raw
        n = int(np.prod(proxy.shape))
        offset = int(proxy.offset)
        if inflater.eof and not inflater.unused_data and len(buf) >= offset + n * proxy.dtype.itemsize:
            data = np.frombuffer(buf, dtype=proxy.dtype, count=n, offset=offset)
            return img, data.reshape(proxy.shape, order="F")
    return img, np.asanyarray(proxy)


def _as_3d(arr, path):
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D volume, got shape {arr.shape}")
    return arr


def load_label_volume(path):
    """Load a label NIfTI as uint8 without a float64 copy; return (img, array)."""
    img, data = read_nifti_array(path)
    data = _as_3d(data, path)
    if data.dtype != np.uint8:
        if np.issubdtype(data.dtype, np.floating):
            data = np.rint(data)
        if data.size and (data.min() < 0 or data.max() > 255):
            raise ValueError(f"{path}: label values outside the uint8 range")
        data = data.astype(np.uint8)
    return img, data


def load_ct(path):
    """Load a CT as int16 HU (rounded, clipped); return (img, array)."""
    img, data = read_nifti_array(path)
    data = _as_3d(data, path)
    if data.dtype != np.int16:
        if np.issubdtype(data.dtype, np.floating):
            data = np.rint(data)
        data = np.clip(data, -32768, 32767).astype(np.int16)
    return img, data


# =============================================================================================================
# Stage 2: clean-up (specks, fragments, debris, field-of-view outliers)
# =============================================================================================================
def touches_si_faces(box, offset, geo):
    """True when a bounding box (slices in the crop; ``offset`` = crop origin in the full grid) reaches the first or
    last slice of the image along the head-foot axis, i.e. the scan edge cuts it."""
    a = geo.si_axis
    return bool(box[a].start + int(offset[a]) <= 0 or box[a].stop + int(offset[a]) >= geo.shape[a])


# Sweep the floor first: dust the network left behind is not a vertebra, however confident its label sounds.
def remove_isolated_specks(work, geo, log, hu, offset):
    """Delete 26-connected islands of the label union smaller than MIN_SPECK_MM3.

    A piece of a vertebra either touches the spinal column or is large, so a tiny island floating in soft tissue
    is a false positive.  We look at the union of all labels so scattered speckle cannot inflate later boxes.
    Exception: an island on the first/last image slice that is bone and within ATTACH_GAP_MM of the labelled
    chain is the clipped end of a vertebra whose rest lies outside the scan.  ``hu`` is the CT crop (or None).
    """
    cc, n = ndi.label(work > 0, structure=STRUCT_26, output=np.int32)
    if n == 0:
        return
    fg = np.flatnonzero(work)  # count on foreground voxels only: a bincount over the whole crop would be huge
    cid = cc.reshape(-1)[fg]
    small = np.bincount(cid, minlength=n + 1) * geo.vox_mm3 < MIN_SPECK_MM3
    small[0] = False
    a = geo.si_axis
    if small.any():
        faces = []
        if int(offset[a]) <= 0:
            faces.append(np.take(cc, 0, axis=a))
        if int(offset[a]) + work.shape[a] >= geo.shape[a]:
            faces.append(np.take(cc, work.shape[a] - 1, axis=a))
        edge_ids = np.unique(np.concatenate([f.ravel() for f in faces])) if faces else []
        objs = ndi.find_objects(cc) if len(edge_ids) else []
        for i in edge_ids:
            if i == 0 or not small[i]:
                continue
            box = objs[i - 1]
            m = cc[box] == i
            if hu is not None and float(np.median(hu[box][m])) < NONBONE_MEDIAN_HU:
                continue
            wbox = pad_box(box, np.ceil(ATTACH_GAP_MM / geo.spacing).astype(int) + 1, work.shape)
            others = (work[wbox] > 0) & (cc[wbox] != i)
            if not others.any():
                continue
            gap = float(ndi.distance_transform_edt(~others, sampling=geo.spacing)[cc[wbox] == i].min())
            if gap > ATTACH_GAP_MM:
                continue
            small[i] = False
            labs = np.unique(work[box][m])
            log.add("S1_specks", "kept", int(labs[0]) if len(labs) == 1 else None, 0,
                    f"{int(m.sum())} voxel(s) on the first/last slice, bone, {gap:.1f} mm from the chain: "
                    "clipped end of a vertebra", labels=[label_name(int(x)) for x in labs])
    del cc
    if not small.any():
        return
    kill = small[cid]
    idx = fg[kill]
    flat = work.reshape(-1)
    labs, comps = flat[idx], cid[kill].astype(np.int64)
    voxels = np.bincount(labs, minlength=N_VALUES)
    islands = np.bincount(np.unique(comps * N_VALUES + labs) % N_VALUES, minlength=N_VALUES)
    flat[idx] = 0
    for v in range(1, N_VALUES):
        if voxels[v]:
            log.add("S1_specks", "removed", v, voxels[v],
                    f"{int(islands[v])} isolated island(s), each < {MIN_SPECK_MM3:g} mm3", islands=int(islands[v]))


def label_components(sub):
    """26-connected components of a boolean crop: (component map, sizes with sizes[0]=0, ids largest-first)."""
    cc, n = ndi.label(sub, structure=STRUCT_26, output=np.int32)
    sizes = np.bincount(cc[sub], minlength=n + 1)
    sizes[0] = 0
    return cc, sizes, sorted(range(1, n + 1), key=lambda c: (-int(sizes[c]), c))  # ties by id


def touching_labels(lab, fbox, fmask, own):
    """Contact counts (26-adjacent voxels) between a fragment and every other label value."""
    pbox = pad_box(fbox, 1, lab.shape)
    fm = np.zeros(tuple(s.stop - s.start for s in pbox), dtype=bool)
    fm[sub_box(pbox, fbox)] = fmask
    shell = ndi.binary_dilation(fm, structure=STRUCT_26) & ~fm
    counts = np.bincount(lab[pbox][shell], minlength=N_VALUES)
    counts[0] = counts[own] = 0
    return counts


def nearest_label_gap(lab, fbox, fmask, spacing, window_mm):
    """Smallest distance (mm) from a fragment to any other labelled voxel, and that voxel's label.
    Searched in a window around the fragment; beyond ``window_mm`` the answer is (inf, 0)."""
    wbox = pad_box(fbox, np.ceil(window_mm / spacing).astype(int) + 1, lab.shape)
    inner = sub_box(wbox, fbox)
    sub = lab[wbox].copy()
    sub[inner][fmask] = 0
    others = sub > 0
    if not others.any():
        return INF, 0
    dist, idx = ndi.distance_transform_edt(~others, sampling=spacing, return_indices=True)
    fm = np.zeros(sub.shape, dtype=bool)
    fm[inner] = fmask
    d = dist[fm]
    j = int(np.argmin(d))
    gap = float(d[j])
    if gap > window_mm:
        return INF, 0
    return gap, int(sub[tuple(int(i[fm][j]) for i in idx)])


def resolve_fragments(lab, hu, geo, offset, log):
    """Keep each label's largest piece and decide the fate of every other 26-connected piece of that label.

    A vertebra is one bone, so a detached piece is one of: part of the vertebra it touches (a process the
    network labelled wrongly), a partial-volume chip of a nearby vertebra, a second vertebra that carries a
    duplicated label, or a false positive in soft tissue.  Rules, in order:
      soft tissue touching nothing -> removed;  vertebra-sized -> kept as its own instance;  on the first/last
      slice -> kept (clipped vertebra);  debris far from every body -> removed;  touching another label ->
      moved there when small, else kept;  within REATTACH_GAP_MM of a label -> joined when small, else kept;
      otherwise removed when small or far, else kept and logged as uncertain.

    Why pieces of 0.5 mL or more are never moved on contact alone: the spinous and transverse processes of one
    level are separate pieces that lie against the next level, and a mislabelled piece looks exactly the same,
    so geometry cannot tell them apart and moving them costs Dice on correct input.
    """
    objs = ndi.find_objects(lab, max_label=NUM_LABELS)
    fragments, main_sizes = [], []
    body = np.zeros(lab.shape, dtype=bool)  # each label's largest component: the vertebral bodies
    for v in range(1, N_VALUES):
        s = objs[v - 1]
        if s is None:
            continue
        cc, sizes, order = label_components(lab[s] == v)
        main_sizes.append(int(sizes[order[0]]))
        body[s] |= cc == order[0]
        cobjs = ndi.find_objects(cc)
        for c in order[1:]:
            f = cobjs[c - 1]
            fbox = tuple(slice(a.start + b.start, a.start + b.stop) for a, b in zip(s, f))
            fragments.append((v, fbox, cc[f] == c, int(sizes[c])))
    if not fragments:
        return
    median_main_mm3 = float(np.median(main_sizes)) * geo.vox_mm3
    label_mm3 = np.bincount(lab.reshape(-1), minlength=N_VALUES) * geo.vox_mm3
    # Bone clusters across labels.  A fragment whose cluster holds no vertebral body touches only other stray
    # pieces, and that contact says nothing about which vertebra (if any) it belongs to.  Two pieces of debris
    # vouching for each other is not evidence: that is how a bone in the arm once got called T10 (BDMAP_00000031).
    joint, _ = ndi.label(lab > 0, structure=STRUCT_26)
    cluster_mm3 = np.bincount(joint.reshape(-1)) * geo.vox_mm3
    body_clusters = set(np.unique(joint[body]).tolist())
    body_lab = np.where(body, lab, 0).astype(lab.dtype)
    del body
    for v, fbox, fmask, size in fragments:
        vol = size * geo.vox_mm3
        idx = np.nonzero(fmask)
        cent_vox = np.array([i.mean() + b.start for i, b in zip(idx, fbox)]) + offset
        info = {"centroid_world_mm": geo.world(cent_vox)}
        med_hu = float(np.median(hu[fbox][fmask])) if hu is not None else None
        if med_hu is not None:
            info["median_hu"] = med_hu
        contacts = touching_labels(lab, fbox, fmask, v)
        touching = int(contacts.sum()) > 0
        view = lab[fbox]

        def keep(why):
            log.add("S3_fragments", "kept", v, 0, why, **info)

        def remove(why):
            view[fmask] = 0
            log.add("S3_fragments", "removed", v, size, why, **info)

        def move(to, why):
            view[fmask] = to
            log.add("S3_fragments", "relabelled", v, size, why, to=to, **info)

        if not touching and med_hu is not None and med_hu < NONBONE_MEDIAN_HU:
            remove(f"isolated non-bone fragment (median HU {med_hu:.0f} < {NONBONE_MEDIAN_HU:g})")
        elif vol >= INSTANCE_MIN_FRACTION * median_main_mm3:
            keep(f"vertebra-sized separate piece ({vol:.0f} mm3): becomes its own instance (possible duplicate level)")
        elif touches_si_faces(fbox, offset, geo):
            keep("piece on the first/last slice: clipped end of a vertebra whose processes are separate pieces")
        else:
            jid = int(joint[fbox][fmask][0])
            if jid not in body_clusters and cluster_mm3[jid] < MIN_VERTEBRA_MM3:
                # Debris: stray pieces of different labels can touch each other far from the spine (an arm
                # labelled T10 is a fun idea, but not anatomy), so only the distance to a real body decides.
                body_gap, _ = nearest_label_gap(body_lab, fbox, fmask, geo.spacing, ATTACH_GAP_MM)
                if body_gap > ATTACH_GAP_MM:
                    remove(f"debris: its bone cluster ({cluster_mm3[jid]:.0f} mm3) holds no vertebral body and lies "
                           f"> {ATTACH_GAP_MM:g} mm from every body")
                    continue
            if touching:
                target = int(np.argmax(contacts))
                if vol >= MIN_ISLAND_MM3:
                    keep(f"piece of {vol:.0f} mm3 touches {label_name(target)}: process of a real vertebra or a "
                         "mislabel, not decidable from geometry")
                elif label_mm3[target] < NEGLIGIBLE_INSTANCE_MM3:
                    keep(f"fragment touches {label_name(target)}, itself a sub-minimum piece: a fragment is not "
                         "moved onto a fragment")
                else:
                    move(target, f"fragment touches {label_name(target)} ({int(contacts[target])} contact voxels)")
                continue
            gap, near = nearest_label_gap(lab, fbox, fmask, geo.spacing, ATTACH_GAP_MM)
            info["gap_mm"] = gap
            if gap <= REATTACH_GAP_MM:
                if near == v:
                    keep(f"detached piece {gap:.1f} mm from its own vertebra")
                elif vol >= MIN_ISLAND_MM3:
                    keep(f"piece of {vol:.0f} mm3 is {gap:.1f} mm from {label_name(near)} but too large to move on proximity")
                else:
                    move(near, f"bone fragment {gap:.1f} mm from {label_name(near)} (<= {REATTACH_GAP_MM:g} mm)")
            elif vol < MIN_ISLAND_MM3:
                remove(f"isolated fragment {gap:.1f} mm from the nearest vertebra and < {MIN_ISLAND_MM3:g} mm3")
            elif gap > ATTACH_GAP_MM and vol < MIN_VERTEBRA_MM3:
                remove(f"fragment > {ATTACH_GAP_MM:g} mm from every vertebra and < {MIN_VERTEBRA_MM3:g} mm3")
            else:
                keep("isolated bone fragment, uncertain: left in place")


def find_instances(lab, hu, geo, offset, log):
    """Vertebra instances: the pieces that stage 3, the sacrum check and the order check reason about.

    After the fragment clean-up each label is normally one connected piece.  The largest piece of a label is
    its instance; another piece of at least INSTANCE_MIN_FRACTION x the median vertebra is an instance of its own
    (a duplicated level); smaller ones ride along with the main piece.  Two things are then dropped from the
    instance list (never from the labels, except the first): an instance far off the chain that is small or soft
    tissue (a "C3" blob in an abdominal scan) is deleted, and an unclipped fragment under NEGLIGIBLE_INSTANCE_MM3
    is not a level and is only ignored.  Returns (instances sorted inferior -> superior, instance id map).
    """
    objs = ndi.find_objects(lab, max_label=NUM_LABELS)
    comp = {v: label_components(lab[objs[v - 1]] == v) for v in range(1, N_VALUES) if objs[v - 1] is not None}
    inst_map = np.zeros(lab.shape, dtype=np.int32)
    if not comp:
        return [], inst_map
    median_main = float(np.median([c[1][c[2][0]] for c in comp.values()]))
    meta, next_id = [], 1
    for v, (cc, sizes, order) in comp.items():
        lut = np.zeros(sizes.size, dtype=np.int32)
        main_id = lut[order[0]] = next_id
        next_id += 1
        meta.append((main_id, v, False))
        for c in order[1:]:
            if sizes[c] >= INSTANCE_MIN_FRACTION * median_main:
                lut[c] = next_id
                meta.append((next_id, v, True))
                log.add("S5_instances", "flagged", v, 0,
                        f"second vertebra-sized component of {label_name(v)}: separate instance", instance=next_id)
                next_id += 1
            else:
                lut[c] = main_id
        m = lut[cc]
        sel = m > 0
        inst_map[objs[v - 1]][sel] = m[sel]
    iobjs = ndi.find_objects(inst_map)
    a = geo.si_axis
    instances = []
    for iid, v, extra in meta:
        s = iobjs[iid - 1]
        m = inst_map[s] == iid
        idx = np.nonzero(m)
        cent = np.array([i.mean() + b.start for i, b in zip(idx, s)]) + offset
        edge = min((s[a].start + offset[a]) * geo.spacing[a], (geo.shape[a] - (s[a].stop + offset[a])) * geo.spacing[a])
        instances.append({
            "id": iid, "label": v, "duplicate_component": extra, "voxels": int(idx[0].size),
            "mm3": idx[0].size * geo.vox_mm3, "centroid_world_mm": geo.world(cent),
            "fov_edge_distance_mm": float(edge), "truncated": bool(edge < FOV_EDGE_MM), "bbox": s,
            "median_hu": float(np.median(hu[s][m])) if hu is not None else None,
        })
    instances = remove_fov_outliers(instances, lab, inst_map, log)
    kept = []
    for inst in instances:
        if inst["mm3"] < NEGLIGIBLE_INSTANCE_MM3 and not inst["truncated"]:
            s = inst["bbox"]
            inst_map[s][inst_map[s] == inst["id"]] = 0
            log.add("S6_field_of_view", "flagged", inst["label"], inst["voxels"],
                    f"fragment of {inst['mm3']:.0f} mm3 is not a level: ignored by the later checks, labels kept",
                    centroid_world_mm=inst["centroid_world_mm"])
        else:
            kept.append(inst)
    kept.sort(key=lambda i: (float(i["centroid_world_mm"][2]), i["label"], i["id"]))
    return kept, inst_map


def remove_fov_outliers(instances, lab, inst_map, log):
    """Delete an instance that lies far off the chain of vertebrae and is clearly not a vertebra.

    Evidence needed: its centroid is more than FOV_ISOLATION_MM (one legal level step) from every other
    instance, AND it is smaller than MIN_VERTEBRA_MM3 or soft tissue by HU.  With fewer than two instances there
    is no chain, so nothing is removed.  Returns the remaining instances.
    """
    if len(instances) < 2:
        return instances
    cents = np.array([i["centroid_world_mm"] for i in instances])
    d = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    keep = []
    for k, inst in enumerate(instances):
        nearest = float(d[k].min())
        small = inst["mm3"] < MIN_VERTEBRA_MM3
        nonbone = inst["median_hu"] is not None and inst["median_hu"] < NONBONE_MEDIAN_HU
        if nearest > FOV_ISOLATION_MM and (small or nonbone):
            s = inst["bbox"]
            m = inst_map[s] == inst["id"]
            lab[s][m] = 0
            inst_map[s][m] = 0
            log.add("S6_field_of_view", "removed", inst["label"], inst["voxels"],
                    f"{region_of(inst['label'])} label {nearest:.0f} mm from the nearest vertebra "
                    f"(> {FOV_ISOLATION_MM:g} mm) and {'small' if small else 'non-bone'}",
                    centroid_world_mm=inst["centroid_world_mm"], median_hu=inst["median_hu"])
        else:
            keep.append(inst)
    return keep


# =============================================================================================================
# Stage 3: label drift (split the bone at the discs, one label per vertebra)
# =============================================================================================================
def pool_blocks(mask, f):
    """Count the foreground voxels of a bool volume in blocks of ``f`` voxels per axis (uint16 counts)."""
    pad = [(0, (-s) % k) for s, k in zip(mask.shape, f)]
    if any(p[1] for p in pad):
        mask = np.pad(mask, pad)
    sh = mask.shape
    blocks = mask.reshape(sh[0] // f[0], f[0], sh[1] // f[1], f[1], sh[2] // f[2], f[2])
    return blocks.sum(axis=(1, 3, 5), dtype=np.uint16)


def body_seeds(edt):
    """Vertebral-body seeds: the peaks of the distance transform that stand out from the saddles joining them.

    A vertebral body is the thickest part of the bone (half-width 8-15 mm); the arch, pedicles, processes and
    ribs that tie neighbouring vertebrae together are thinner.  Raising a distance threshold therefore
    separates the bodies while the connections vanish.  A region is split at the first threshold where at
    least two parts keep a peak >= DRIFT_BODY_PEAK_MM that stands >= DRIFT_BODY_PROMINENCE_MM above that
    threshold (the h-maxima rule); smaller parts are dropped and later absorbed by growth.  Returns
    (seed map with ids 1..n, 0 = none; n), in a deterministic order (scan order of the splits).
    """
    seeds = np.zeros(edt.shape, dtype=np.uint8)
    n_seeds = 0
    stack = [(tuple(slice(0, n) for n in edt.shape), edt > 0.0, 0.0)]
    while stack:
        bb, mask, r0 = stack.pop()
        ee = np.where(mask, edt[bb], 0.0)
        peak = float(ee.max()) if ee.size else 0.0
        if peak < DRIFT_BODY_PEAK_MM:
            continue
        r1 = r0 + DRIFT_EDT_STEP_MM
        children = None
        while r1 < peak:
            cc, n = ndi.label(ee > r1, structure=STRUCT_26)
            if n >= 2:
                inside = cc > 0  # maxima over the bone voxels only (sorting the empty box would dominate)
                mx = ndi.maximum(ee[inside], cc[inside], index=np.arange(1, n + 1))
                prom = [i + 1 for i in range(n)
                        if mx[i] - r1 >= DRIFT_BODY_PROMINENCE_MM and mx[i] >= DRIFT_BODY_PEAK_MM]
                if len(prom) >= 2:
                    objs = ndi.find_objects(cc)
                    children = [(tuple(slice(b.start + s.start, b.start + s.stop) for b, s in zip(bb, objs[i - 1])),
                                 cc[objs[i - 1]] == i, r1) for i in prom]
                    break
            r1 += DRIFT_EDT_STEP_MM
        if children is not None:
            stack.extend(reversed(children))  # popped in scan order
        else:
            n_seeds += 1
            if n_seeds > DRIFT_MAX_REGIONS:
                return seeds, n_seeds
            seeds[bb][mask] = n_seeds
    return seeds, n_seeds


def grow_seeds(seeds, allowed):
    """Grow the seed ids through ``allowed`` voxels one 26-neighbour shell at a time (ties: higher id wins).
    Works on flat indices, so the cost follows the bone, not the box."""
    g = np.pad(np.where(allowed, seeds, 0).astype(np.uint8), 1)
    ok = np.pad(allowed, 1)
    flat, okf = g.reshape(-1), ok.reshape(-1)
    strides = np.array([g.shape[1] * g.shape[2], g.shape[2], 1])
    offs = np.array([(a * strides[0] + b * strides[1] + c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                     if (a, b, c) != (0, 0, 0)])
    front = np.flatnonzero(flat)
    while front.size:
        nb = (front[:, None] + offs[None, :]).reshape(-1)
        lbl = np.repeat(flat[front], offs.size)
        keep = okf[nb] & (flat[nb] == 0)
        nb, lbl = nb[keep], lbl[keep]
        if nb.size == 0:
            break
        order = np.argsort(lbl, kind="stable")  # a repeated index takes the last (highest) id
        flat[nb[order]] = lbl[order]
        front = np.unique(nb)
    return g[1:-1, 1:-1, 1:-1]


def labels_between(lo_label, hi_label, votes):
    """Labels for the m mixed vertebrae lying between a pure vertebra labelled ``lo_label`` (below) and one labelled
    ``hi_label`` (above): the levels in between, in order, one per vertebra.  Counting decides; only when the
    levels outnumber the vertebrae (by up to DRIFT_MAX_SKIP_LEVELS) the network's votes pick the undetected level.
    None when the vertebrae do not fit (more vertebrae than levels, or too many levels left over)."""
    m = len(votes)
    levels = list(range(lo_label + 1, hi_label))
    if m > len(levels) or len(levels) - m > DRIFT_MAX_SKIP_LEVELS:
        return None
    v = np.asarray(votes, dtype=np.float64)[:, 1:]
    u = -np.log((v + VOTE_PSEUDOCOUNT) / (v.sum(axis=1, keepdims=True) + NUM_LABELS * VOTE_PSEUDOCOUNT))
    best, best_cost = None, INF
    for omit in itertools.combinations(range(len(levels)), len(levels) - m):
        labs = [lv for k, lv in enumerate(levels) if k not in omit]
        cost = sum(u[i, lv - 1] for i, lv in enumerate(labs))
        if cost < best_cost - TIE_EPS:
            best, best_cost = labs, cost
    return best


def split_into_vertebrae(lab, geo):
    """Split the labelled bone into single vertebrae, ignoring the labels (they are what may be wrong).

    Returns (regions, info).  ``regions`` = (index arrays of the bone voxels, region id of each: 1..n from
    inferior to superior by the world S of the centroid); info holds the voxel count of every region, the label
    votes of the whole region and of its body core (the seed), and the centroids (voxel coordinates).
    """
    fg = lab > 0
    f = np.maximum(1, np.rint(DRIFT_COARSE_MM / geo.spacing)).astype(int)
    count = pool_blocks(fg, f)
    body = count * 2 >= int(np.prod(f))  # majority-pooled: bodies keep their discs
    reach = count > 0  # any-pooled: growth must reach every bone voxel
    seeds, n_seeds = body_seeds(ndi.distance_transform_edt(body, sampling=geo.spacing * f))
    if n_seeds == 0 or n_seeds > DRIFT_MAX_REGIONS:
        return None, {"n_regions": int(n_seeds)}
    grown = grow_seeds(seeds, reach)
    ci = np.nonzero(grown)
    cid = grown[ci]
    cent = np.stack([np.bincount(cid, weights=c, minlength=n_seeds + 1)[1:] for c in ci], axis=1)
    cent = cent / np.maximum(np.bincount(cid, minlength=n_seeds + 1)[1:], 1)[:, None] * f + (f - 1) / 2.0
    order = np.argsort(cent @ geo.affine[2, :3], kind="stable")
    rank = np.zeros(n_seeds + 1, dtype=np.uint8)
    rank[order + 1] = np.arange(1, n_seeds + 1)
    idx = np.nonzero(fg)  # per bone voxel: its block, hence its region and whether it is in a body core
    block = tuple(i // k for i, k in zip(idx, f))
    region, core = rank[grown[block]], rank[seeds[block]]
    labels = lab[idx].astype(np.int64)

    def vote_table(ids):
        return np.bincount(ids.astype(np.int64) * N_VALUES + labels,
                           minlength=(n_seeds + 1) * N_VALUES).reshape(n_seeds + 1, N_VALUES)[1:]
    votes, core_votes = vote_table(region), vote_table(core)
    return (idx, region), {"n_regions": int(n_seeds), "votes": votes, "core_votes": core_votes,
                           "centroid_vox": cent[order], "voxels": votes.sum(axis=1)}


def region_trust(mm3, world):
    """Which split regions look like one whole vertebra, and which consecutive regions have a legal centroid gap.

    A region is one whole vertebra when its volume is within DRIFT_REGION_VOLUME_RANGE x the median of its
    +-VOLUME_NEIGHBOURS neighbours (two bodies seeded as one are too big, a fragment too small), and two
    consecutive vertebrae are one level step apart (SPACING_GATE_MM).  Returns (trusted bool array, gap_ok list).
    """
    n = len(mm3)
    lo_v, hi_v = DRIFT_REGION_VOLUME_RANGE
    trusted = np.ones(n, dtype=bool)
    for i in range(n):
        nb = [mm3[j] for j in range(max(i - VOLUME_NEIGHBOURS, 0), min(i + VOLUME_NEIGHBOURS, n - 1) + 1) if j != i]
        if nb:
            med = float(np.median(nb))
            trusted[i] = lo_v * med <= mm3[i] <= hi_v * med
    gap_ok = [SPACING_GATE_MM[0] <= float(np.linalg.norm(world[i + 1] - world[i])) <= SPACING_GATE_MM[1]
              for i in range(n - 1)]
    return trusted, gap_ok


# The network drew the border through the middle of the bone; we move it back to the disc, where the textbook keeps it.
def resolve_label_drift(lab, geo, log, keep=None):
    """When label boundaries cut through vertebral bodies, give every vertebra one label.

    Neighbouring vertebral bodies are separated by an intervertebral disc, so the bone splits into one piece
    per vertebra at the discs whatever the labels say.  SuPreM sometimes drifts: the labels change in the
    middle of a body, so a body carries a cap of its neighbour's label (while each label is still vertebra-
    sized, so the fragment rules cannot see it).  Procedure: (1) split the bone into vertebrae (body seeds from
    the distance transform, grown through the bone); (2) keep only vertebrae of plausible size and spacing (a
    trusted segment); (3) drift is a trusted vertebra in which a second label of an adjacent level covers
    >= DRIFT_MINORITY_MIN, in >= DRIFT_MIN_DRIFTED nearby vertebrae (a run); (4) a run is anchored by the
    nearest pair of consistent pure vertebrae on each side, and the vertebrae between them get the levels
    between the anchors' labels in order (counting says which level a half-and-half body is; the votes only
    pick an undetected level); a run that does not fit, or that reaches the end of the trusted chain, abstains;
    (5) overwrite the vertebrae of the run that are drifted or whose label changes.  Returns an info dict.
    ``keep`` receives the split, which the lost-level step reuses instead of repeating it.
    """
    info = {"status": "no_bone"}
    if not (lab > 0).any():
        return info
    regions, rinfo = split_into_vertebrae(lab, geo)
    if keep is not None:
        keep["regions"], keep["rinfo"] = regions, rinfo
    n = rinfo["n_regions"]
    info = {"status": "no_drift", "n_regions": n}
    if regions is None:
        info.update(status="no_split", reason=f"{n} body seeds" if n else "no vertebral body found")
        return info
    votes, cent, vox = rinfo["core_votes"], rinfo["centroid_vox"], rinfo["voxels"]  # labels of the body core
    mm3 = vox * geo.vox_mm3
    cnt = np.maximum(votes.sum(axis=1), 1)
    maj = np.argmax(votes[:, 1:], axis=1) + 1
    frac = votes[np.arange(n), maj] / cnt
    second = votes.copy()  # drift evidence per vertebra: the strongest other label, if it is an adjacent level
    second[np.arange(n), maj] = 0
    second[:, 0] = 0
    sec_lab = np.argmax(second, axis=1)
    sec_frac = second[np.arange(n), sec_lab] / cnt
    drifted = (sec_frac >= DRIFT_MINORITY_MIN) & (np.abs(sec_lab - maj) == 1)
    trusted, gap_ok = region_trust(mm3, cent @ geo.affine[:3, :3].T)
    segments, cur = [], []  # runs of consecutive trusted vertebrae with legal gaps
    for i in range(n):
        if trusted[i] and (not cur or gap_ok[i - 1]):
            cur.append(i)
        else:
            if len(cur) >= DRIFT_MIN_SEGMENT:
                segments.append(cur)
            cur = [i] if trusted[i] else []
    if len(cur) >= DRIFT_MIN_SEGMENT:
        segments.append(cur)
    info.update(regions=[{"index": i + 1, "mm3": mm3[i], "majority": label_name(maj[i]),
                          "majority_fraction": frac[i], "second": label_name(sec_lab[i]) if sec_lab[i] else None,
                          "second_fraction": sec_frac[i], "trusted": bool(trusted[i]), "drifted": bool(drifted[i])}
                         for i in range(n)], n_segments=len(segments))
    if int((drifted & trusted).sum()) < DRIFT_MIN_DRIFTED:
        return info
    todo = [seg for seg in segments if drifted[seg].sum() >= DRIFT_MIN_DRIFTED]
    if not todo:
        info.update(status="abstained", flagged=True,
                    reason=f"drift seen in {int((drifted & trusted).sum())} vertebrae but not {DRIFT_MIN_DRIFTED} of them "
                           f"inside {DRIFT_MIN_SEGMENT}+ consecutive trustworthy vertebrae: no order evidence")
        log.add("S4b_label_drift", "flagged", None, 0, info["reason"])
        return info

    def pure(i):
        return frac[i] >= DRIFT_ANCHOR_PURITY and not drifted[i]

    def anchor(seg, k, step):
        """A pure vertebra whose outer neighbour is pure too and carries the next level's label.  Two consistent
        pure vertebrae in a row are not a body that the drift itself made look pure."""
        j = k + step
        return pure(seg[k]) and 0 <= j < len(seg) and pure(seg[j]) and int(maj[seg[j]]) - int(maj[seg[k]]) == step

    changed_regions, skipped = [], []
    for seg in todo:
        runs = []  # drifted bodies separated by at most DRIFT_CLUSTER_GAP other bodies form one run
        for k in [k for k, i in enumerate(seg) if drifted[i]]:
            if runs and k - runs[-1][1] <= DRIFT_CLUSTER_GAP + 1:
                runs[-1][1] = k
                runs[-1][2] += 1
            else:
                runs.append([k, k, 1])
        for first, last, n_drifted in runs:
            if n_drifted < DRIFT_MIN_DRIFTED:
                continue  # an isolated capped body is not drift evidence
            lo, hi = first - 1, last + 1  # extend to the nearest anchor on each side
            while lo >= 0 and not anchor(seg, lo, -1):
                lo -= 1
            while hi < len(seg) and not anchor(seg, hi, +1):
                hi += 1
            if lo < 0 or hi >= len(seg):
                skipped.append(f"drift run {label_name(maj[seg[first]])}..{label_name(maj[seg[last]])} reaches the end "
                               "of the trusted chain: no pure vertebra beyond it to count from")
                continue
            run = seg[lo + 1:hi]
            labs = labels_between(int(maj[seg[lo]]), int(maj[seg[hi]]), votes[run])
            if labs is None:
                skipped.append(f"{len(run)} vertebrae between pure {label_name(maj[seg[lo]])} and "
                               f"{label_name(maj[seg[hi]])}: the vertebra count does not match the levels")
                continue
            changed_regions += [(i, int(v)) for i, v in zip(run, labs) if v != maj[i] or drifted[i]]
    if skipped:
        info.update(flagged=True, reason="; ".join(skipped))
        log.add("S4b_label_drift", "flagged", None, 0, "label drift suspected, split ambiguous, kept: " + info["reason"])
    if not changed_regions:
        info["status"] = "abstained"
        return info
    n_vox = 0
    idx, region = regions
    for i, v in changed_regions:
        m = (region == i + 1) & (lab[idx] != v)
        k = int(np.count_nonzero(m))
        lab[tuple(a[m] for a in idx)] = v
        n_vox += k
        log.add("S4b_label_drift", "relabelled", int(maj[i]), k,
                f"vertebra {i + 1} of {n} (from the bottom): label {label_name(maj[i])} "
                f"{frac[i]:.0%} / {label_name(sec_lab[i]) if sec_lab[i] else '-'} {sec_frac[i]:.0%} -> one label", to=v)
    info.update(status="applied", n_vertebrae_relabelled=len(changed_regions), voxels=n_vox)
    return info


# =============================================================================================================
# Stage 4: sacrum anchor, then the lost level
# =============================================================================================================
# A blunt test on purpose: lots of wide, unlabelled bone just under the last vertebra is the sacrum or the pelvis.
# We only use the answer as found / not found, because it can't tell an L6 from an L5.
def detect_sacrum(lowest, lab, inst_map, offset, ct_full, geo):
    """Is there a wide unlabelled bone mass directly below the lowest vertebra (a sacrum / pelvis)?

    The CT must be available and the lowest instance must not be clipped by the scan edge.  In a slab from its
    inferior edge down SACRUM_SLAB_MM, within SACRUM_SEARCH_RADIUS_MM of its axis, smoothed HU >= HU_BONE marks
    bone.  Unlabelled bone that reaches within SACRUM_MIDLINE_MM of the axis must total >= SACRUM_MIN_BONE_MM3 and
    spread sideways >= SACRUM_LATERAL_RATIO x the lowest vertebra's half-width.  SuPreM has no sacrum class,
    so the network leaves it unlabelled, and the pelvic ring is far wider than any vertebra.  Known weak spot: an
    unlabelled lumbar vertebra next to the iliac crests could pass, so the measurements are logged.
    """
    res = {"found": False}
    if ct_full is None:
        res["reason"] = "no CT"
        return res
    if lowest["truncated"]:
        res["reason"] = "lowest vertebra truncated by the field of view"
        return res
    s = lowest["bbox"]
    idx = np.nonzero(inst_map[s] == lowest["id"])
    w = geo.world(np.stack([i + b.start + o for i, b, o in zip(idx, s, offset)], axis=1))
    cx, cy = lowest["centroid_world_mm"][0], lowest["centroid_world_mm"][1]
    s_min = float(w[:, 2].min())
    half_width = float(np.abs(w[:, 0] - cx).max())
    s_lo = s_min - SACRUM_SLAB_MM
    r = SACRUM_SEARCH_RADIUS_MM
    corners = np.array([[x, y, z] for x in (cx - r, cx + r) for y in (cy - r, cy + r) for z in (s_lo, s_min)])
    ijk = apply_affine(geo.inv_affine, corners)
    lo = np.maximum(np.floor(ijk.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + 2, np.array(geo.shape))
    res.update(lowest_inferior_S_mm=s_min, lowest_half_width_mm=half_width)
    if np.any(hi <= lo):
        res["reason"] = "slab below the lowest vertebra is outside the image"
        return res
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    ct = ct_full[box].astype(np.float32)
    labs = np.zeros(ct.shape, dtype=np.uint8)  # the labels, placed on the slab's grid so labelled bone is excluded
    tbox = tuple(slice(int(o), int(o) + n) for o, n in zip(offset, lab.shape))
    ov = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(box, tbox))
    if all(o.stop > o.start for o in ov):
        labs[sub_box(box, ov)] = lab[sub_box(tbox, ov)]
    g = np.ogrid[box]
    A = geo.affine
    x, y, z = (A[k, 0] * g[0] + A[k, 1] * g[1] + A[k, 2] * g[2] + A[k, 3] for k in range(3))  # world coordinates
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    slab = (z < s_min) & (z >= s_lo) & (r2 <= r * r)
    bone = (ndi.uniform_filter(ct, size=SACRUM_SMOOTH_VOX, mode="nearest") >= HU_BONE) & (labs == 0) & slab
    if not bone.any():
        res.update(reason="no unlabelled bone below the lowest vertebra", bone_mm3=0.0)
        return res
    cc, _ = ndi.label(bone, structure=STRUCT_26, output=np.int32)
    near_axis = np.unique(cc[bone & (r2 <= SACRUM_MIDLINE_MM ** 2)])  # bone pieces that reach the midline
    sel = np.isin(cc, near_axis[near_axis > 0])
    vol = float(np.count_nonzero(sel)) * geo.vox_mm3
    xs = np.broadcast_to(x, sel.shape)[sel]
    span = float(np.abs(xs - cx).max()) if xs.size else 0.0
    res.update(bone_mm3=vol, lateral_half_span_mm=span)
    if vol < SACRUM_MIN_BONE_MM3:
        res["reason"] = f"midline-connected bone {vol:.0f} mm3 < {SACRUM_MIN_BONE_MM3:g} mm3"
    elif span < SACRUM_LATERAL_RATIO * half_width:
        res["reason"] = f"bone half-span {span:.0f} mm < {SACRUM_LATERAL_RATIO:g} x vertebra half-width {half_width:.0f} mm"
    else:
        res.update(found=True, reason="wide unlabelled bone mass directly below the lowest vertebra (heuristic sacrum/pelvis)")
    return res


def spacing_prior(k):
    """(mean, sd) in mm of the centroid distance between level k and level k+1 (k+1 is superior).
    At a junction (L1-T12, T1-C7) the mean of the two regions' means and the larger sd are used."""
    r1, r2 = region_of(k), region_of(k + 1)
    if r1 == r2:
        return SPACING_MM[r1]
    (m1, s1), (m2, s2) = SPACING_MM[r1], SPACING_MM[r2]
    return (m1 + m2) / 2.0, max(s1, s2)


def _oversize(mm3, i):
    """Region i is more than LOST_OVERSIZE x the median of its +-VOLUME_NEIGHBOURS neighbours."""
    nb = [mm3[j] for j in range(max(i - VOLUME_NEIGHBOURS, 0), min(i + VOLUME_NEIGHBOURS, len(mm3) - 1) + 1) if j != i]
    return bool(nb) and mm3[i] > LOST_OVERSIZE * float(np.median(nb))


# Counting up from L5 works until a patient has a sixth lumbar vertebra. Then it is off-by-one, the oldest bug there is, but with bones.
def plan_lost_level(votes, mm3, world, anchor_ok):
    """Decide whether a thoracolumbar level was lost (a pure function of the split regions, inferior -> superior).

    SuPreM's commonest remaining real error is to lose one thoracolumbar vertebra and stretch its neighbours:
    the two vertebrae around the loss carry one label, and every vertebra above it is labelled one level too
    low.  The chain still looks legal, so no order check can notice.  The CT shows the truth: counting vertebral
    bodies up from the sacrum, region k must be level k.  The labels agree with that count up to a duplicated
    label and are exactly one too low from there to the top.

    A relabelling needs ALL of: the sacrum confirms L5 as the lowest vertebra and the lowest region is labelled
    L5; every region of the chain has a plausible volume and a legal gap (so the count of bodies is believable);
    exactly one pair of neighbouring regions shares a label; the regions below the pair carry exactly their
    count (>= LOST_MIN_BELOW); every region above it is exactly one too low, with its label still the majority
    (>= LOST_MIN_SHIFTED of them); the two regions that share the label are a whole level apart along S; the
    shifted top label exists.  A trailing sliver cut by the scan edge is not counted.  If the shifted signature
    is present but a condition fails, or a region is more than twice its neighbours (two bodies seen as one),
    nothing is changed and the case is flagged.  Returns a dict: status 'none' | 'flagged' | 'relabel'.
    """
    n = len(mm3)
    out = {"status": "none"}
    if n == 0:
        return out
    cnt = np.maximum(votes.sum(axis=1), 1)
    maj = np.argmax(votes[:, 1:], axis=1) + 1
    frac = votes[np.arange(n), maj] / cnt
    trusted, gap_ok = region_trust(mm3, world)
    lo_v = DRIFT_REGION_VOLUME_RANGE[0]
    med = float(np.median(mm3[trusted])) if trusted.any() else 0.0
    end = n  # a small untrusted region at the top is the clipped end of the last vertebra: not counted
    while end > 0 and not trusted[end - 1] and mm3[end - 1] < lo_v * med:
        end -= 1
    out["n_regions"], out["n_counted"] = n, end
    if end < 2:
        return out
    shift = np.arange(1, end + 1) - maj[:end]  # 0 = label equals the count from L5, 1 = one level too low
    dups = [k for k in range(end - 1) if maj[k] == maj[k + 1]]
    over = [i for i in range(end) if _oversize(mm3, i)]
    sig = bool(dups) and bool((shift[:dups[0] + 1] == 0).all()) and bool(shift[dups[0] + 1] == 1) \
        and (dups[0] + 2 >= end or bool(shift[dups[0] + 2] == 1))
    if not sig:
        if over:
            i = over[0]
            out.update(status="flagged", kind="oversize_region", region=i + 1,
                       reason=f"vertebra {i + 1} of {n} (from the bottom, labelled {label_name(maj[i])}) is more than "
                              f"{LOST_OVERSIZE:g}x its neighbours ({mm3[i] / 1000:.0f} mL): two bodies may be seen as one, "
                              "so a level could be missing here")
        return out
    k = dups[0]
    fails = []
    if not (anchor_ok and maj[0] == VALUE_L5):
        fails.append("L5 is not confirmed as the lowest vertebra (no sacrum anchor, or the lowest region is not L5)")
    if not (trusted[:end].all() and all(gap_ok[:end - 1])):
        fails.append("a vertebra of the chain has an implausible volume or gap, so the count of bodies is not reliable")
    if len(dups) != 1:
        fails.append(f"{len(dups)} neighbouring pairs share a label")
    if k + 1 < LOST_MIN_BELOW:
        fails.append(f"only {k + 1} vertebrae below the duplicate match their count (< {LOST_MIN_BELOW})")
    above = list(range(k + 1, end))
    if len(above) < LOST_MIN_SHIFTED or not (shift[above] == 1).all():
        fails.append(f"fewer than {LOST_MIN_SHIFTED} consecutive vertebrae above the duplicate are exactly one level too low")
    elif (frac[above] < LOST_OWN_LABEL_MIN).any():
        fails.append("a shifted vertebra's own label is not the majority of its body (label drift, not a lost level)")
    mu, sd = spacing_prior(int(maj[k]))
    dz = abs(float(world[k + 1][2] - world[k][2]))
    if dz < mu - LOST_GAP_SIGMA * sd:
        fails.append(f"the two vertebrae sharing {label_name(maj[k])} are only {dz:.0f} mm apart along S "
                     f"(< {mu - LOST_GAP_SIGMA * sd:.0f} mm): one body cut in two, not two vertebrae")
    if int(maj[end - 1]) + 1 > NUM_LABELS:
        fails.append("the shifted top label would be above C1")
    out.update(region=k + 2, shared_label=label_name(maj[k]), gap_along_S_mm=dz, kind="duplicate_label_then_shift")
    if fails:
        out.update(status="flagged", reason="; ".join(fails))
        return out
    out.update(status="relabel", new_labels={i: i + 1 for i in above},
               reason=f"counting {end} vertebral bodies up from the confirmed L5, {label_name(maj[k])} appears on "
                      f"vertebrae {k + 1} and {k + 2} and all {len(above)} above are one level too low")
    return out


# The bravest function in this file: when it fires it moves many labels at once, so it demands a lot of proof first.
# It was right on s1423 and s1985, and wrong on an L6 spine (sub-verse517).  --no_relabel turns it off.
def resolve_lost_level(lab, geo, offset, keep, drift, sacrum, instances, log):
    """Restore a lost thoracolumbar level, or flag it ('possible_missed_level').

    Uses the vertebra split of stage 3 (``keep``), the label votes of its body cores and the sacrum anchor; the
    rule is in plan_lost_level.  A relabelling overwrites whole regions (every bone voxel of the vertebra) with
    the label the count gives; nothing else changes.  Skipped when stage 3 already changed labels, because its
    split then no longer describes them.  Returns an info dict; ``info['relabelled']`` says whether voxels changed.
    """
    info = {"status": "none", "relabelled": False, "possible_missed_level": None}
    regions, rinfo = keep.get("regions"), keep.get("rinfo")
    if regions is None or drift.get("status") == "applied":
        info["status"] = "skipped"
        return info
    world = geo.world(rinfo["centroid_vox"] + np.asarray(offset, dtype=np.float64))
    mm3 = rinfo["voxels"] * geo.vox_mm3
    low = instances[0] if instances else None
    anchor_ok = bool(sacrum.get("found") and low is not None and low["label"] == VALUE_L5 and not low["truncated"])
    plan = plan_lost_level(rinfo["core_votes"], mm3, world, anchor_ok)
    info["status"] = plan["status"]
    info.update({k: v for k, v in plan.items() if k not in ("status", "new_labels")})
    if plan["status"] == "none":
        return info
    i = plan["region"] - 1
    loc = {"region_from_bottom": plan["region"], "world_S_mm": float(world[i][2]), "kind": plan["kind"],
           "reason": plan["reason"]}
    if plan["status"] == "flagged":
        info["possible_missed_level"] = loc
        log.add("S7b_lost_level", "flagged", None, 0, "possible missed level, labels kept: " + plan["reason"],
                world_S_mm=loc["world_S_mm"])
        return info
    idx, region = regions
    cur = lab[idx]
    n_vox = 0
    for r, v in sorted(plan["new_labels"].items()):
        in_r = region == r + 1
        old = int(np.bincount(cur[in_r].astype(np.int64), minlength=N_VALUES).argmax())
        m = in_r & (cur > 0) & (cur != v)
        k = int(np.count_nonzero(m))
        lab[tuple(a[m] for a in idx)] = v
        n_vox += k
        log.add("S7b_lost_level", "relabelled", old, k,
                f"vertebra {r + 1} of {len(mm3)} (from the bottom): level from the count of bodies up from the confirmed L5", to=v)
    info.update(relabelled=True, voxels=n_vox, possible_missed_level=dict(loc, action="relabelled"))
    return info


# =============================================================================================================
# Stage 5: order check (flags only, no voxel changes)
# =============================================================================================================
# Flags only, no voxels touched, so it cannot break anything. It can only be annoyingly right.
def check_level_order(instances, log, sacrum=None):
    """Flag vertebra levels whose centroids are in the wrong order along S, or too far apart.

    Levels run L5 (value 1) up to C1 (value 24), so each level should lie higher (larger world S) than the one
    before it, by one vertebra: roughly 17-33 mm.  For every pair of consecutive PRESENT levels we flag
    (a) an inversion: the higher level sits lower than the lower one (swapped or mislabelled neighbours), and
    (b) a gap over SPACING_GATE_MM[1] = 50 mm, which is too much for one step and usually means a vertebra went
    missing in between.  Nothing is relabelled: the reason is that a centroid can only say "something is off
    here", not which of two neighbours is wrong.  A level with two pieces is represented by its larger piece.
    (c) an anchor mismatch: the sacrum sits right under the lowest vertebra, but that vertebra is not labelled
    L5.  Either the whole chain is shifted by a level (order and gaps can't see that: everything moved
    together, politely) or this spine has a transitional vertebra.  Both deserve a human look.
    Returns the list of flags (each also logged).
    """
    main = {}
    for inst in instances:
        if inst["label"] not in main or inst["voxels"] > main[inst["label"]]["voxels"]:
            main[inst["label"]] = inst
    flags = []
    levels = sorted(main)
    for lo, hi in zip(levels, levels[1:]):
        step = float(main[hi]["centroid_world_mm"][2] - main[lo]["centroid_world_mm"][2])  # > 0 means correct order
        pair = f"{label_name(lo)} -> {label_name(hi)}"
        for kind, why in (("inverted", step < 0 and f"{label_name(hi)} lies {-step:.0f} mm below {label_name(lo)} along S"),
                          ("gap", abs(step) > SPACING_GATE_MM[1] and
                           f"{abs(step):.0f} mm along S from {label_name(lo)} to {label_name(hi)} (> {SPACING_GATE_MM[1]:g} mm)")):
            if why:
                flags.append({"kind": kind, "lower_level": label_name(lo), "upper_level": label_name(hi),
                              "delta_S_mm": step, "reason": why})
                log.add("S8_order_check", "flagged", None, 0, f"{kind} {pair}: {why}; labels not changed")
    if sacrum and sacrum.get("found") and instances:
        lowest = min(instances, key=lambda i: float(i["centroid_world_mm"][2]))["label"]
        if lowest != VALUE_L5:
            why = (f"the sacrum lies directly below the lowest vertebra, which is labelled {label_name(lowest)}, "
                   "not L5: a whole-chain shift or a transitional vertebra")
            flags.append({"kind": "anchor", "lower_level": label_name(lowest), "upper_level": None,
                          "delta_S_mm": None, "reason": why})
            log.add("S8_order_check", "flagged", lowest, 0, f"anchor: {why}; labels not changed")
    return flags


def label_qa(arr, offset, geo):
    """Present labels, missing levels in their range, and centroid order violations (world S) of a crop."""
    idx = np.nonzero(arr)
    labs = arr[idx].astype(np.int64)
    ok = labs <= NUM_LABELS  # ignore values outside the label map
    idx, labs = tuple(i[ok] for i in idx), labs[ok]
    cnt = np.bincount(labs, minlength=N_VALUES)
    present = [int(v) for v in np.nonzero(cnt[1:])[0] + 1]
    out = {"labels": [label_name(v) for v in present], "volume_ml": {}}
    if not present:
        out.update(missing_levels=[], order_violations=0)
        return out
    cent = np.stack([np.bincount(labs, weights=idx[a] + offset[a], minlength=N_VALUES) for a in range(3)], axis=1)
    s = geo.world(cent[present] / cnt[present][:, None])[:, 2]
    p = np.array(present)
    out["order_violations"] = int(np.count_nonzero((p[:, None] > p[None, :]) & (s[:, None] < s[None, :])))
    out["missing_levels"] = [label_name(v) for v in range(min(present), max(present) + 1) if v not in present]
    out["volume_ml"] = {label_name(v): cnt[v] * geo.vox_mm3 / 1000.0 for v in present}
    return out


# =============================================================================================================
# One case, start to finish
# =============================================================================================================
# Hash the CT so the log says exactly which scan this ran on. Future me will thank present me.
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_case(case, input_root, ct_root, output_root, verbose=False, log_dir=None, timing=False, relabel=True):
    """Run the five stages on one case, write the outputs and the JSON log, and return the log dictionary."""
    times = {}
    t0 = time.perf_counter()
    clock = [t0]

    def lap(name):
        now = time.perf_counter()
        times[name] = now - clock[0]
        clock[0] = now

    in_dir = os.path.join(input_root, case)
    mask_path = os.path.join(in_dir, "combined_labels.nii.gz")
    ct_path = os.path.join(ct_root, case, "ct.nii.gz") if ct_root else None
    out_dir = os.path.join(output_root, case)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------------------ 1. load + crop
    with ThreadPoolExecutor(max_workers=2) as ex:  # the two files inflate in parallel (zlib releases the GIL)
        f_mask = ex.submit(load_label_volume, mask_path)
        f_ct = ex.submit(load_ct, ct_path) if ct_path and os.path.exists(ct_path) else None
        mask_img, combined = f_mask.result()
        ct_img, ct_full = f_ct.result() if f_ct is not None else (None, None)
    file_geo = Geometry(mask_img.affine, combined.shape)
    log = CaseLog(case, file_geo.vox_mm3, verbose)
    ct_note = "used"
    if ct_img is None:
        ct_note = "missing: CT checks (HU tests, sacrum anchor) skipped"
    elif tuple(ct_full.shape) != file_geo.shape or not np.allclose(ct_img.affine, file_geo.affine, atol=AFFINE_ATOL):
        ct_note = "grid differs from the mask: CT checks skipped"
        ct_full = None
    # NIfTI data are stored F-ordered, but scipy.ndimage is much faster on C-ordered memory.  So we work on the
    # transposed views (no copy) with the affine's voxel columns permuted to match.  All geometry is in world mm,
    # so the results are identical; the outputs are transposed back before writing.
    if combined.flags.f_contiguous and not combined.flags.c_contiguous:
        combined = combined.T
        ct_full = ct_full.T if ct_full is not None else None
        geo = Geometry(file_geo.affine[:, [2, 1, 0, 3]], combined.shape)
        to_file = np.transpose
    else:
        geo, to_file = file_geo, (lambda a: a)
    if ct_full is None and ct_img is not None:
        log.add("S0_input", "flagged", None, 0, f"CT {ct_note}")
    lap("load")

    result = {
        "case": case,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "nibabel": nib.__version__},
        "inputs": {"mask": mask_path, "mask_sha256": file_sha256(mask_path), "ct": ct_path, "ct_status": ct_note},
        "geometry": file_geo.as_dict(),
        "constants": SETTINGS,
        "options": {"relabel": relabel},
    }
    wbox = nonzero_bbox(combined)
    out_full = np.array(combined, order="C", copy=True)
    drift = {"status": "no_bone"}
    lost = {"status": "disabled", "relabelled": False, "possible_missed_level": None}
    instances, sacrum, order_flags = [], {}, []
    if wbox is None:
        log.add("S0_input", "flagged", None, 0, "no labelled voxels: output equals input")
        work = np.zeros((0, 0, 0), np.uint8)
        wbox = tuple(slice(0, 0) for _ in range(3))
        result.update(qa_before={}, qa_after={})
    else:
        wbox = pad_box(wbox, CROP_PAD_VOX, geo.shape)
        work = np.ascontiguousarray(combined[wbox])
        before = work.copy()
        woff = np.array([s.start for s in wbox])
        unknown = work > NUM_LABELS
        if unknown.any():
            log.add("S0_input", "removed", None, int(np.count_nonzero(unknown)),
                    f"label values outside 1..24: {np.unique(work[unknown]).tolist()}")
            work[unknown] = 0
        result["qa_before"] = label_qa(before, woff, geo)
        lap("crop")

        # -------------------------------------------------------------------------- 2. clean-up
        remove_isolated_specks(work, geo, log, ct_full[wbox] if ct_full is not None else None, woff)
        lap("specks")
        tb = nonzero_bbox(work)
        if tb is not None:
            tl = pad_box(tb, CROP_PAD_VOX, work.shape)  # a tighter crop now that the specks are gone
            lab = np.ascontiguousarray(work[tl])
            offset = woff + np.array([s.start for s in tl])
            hu = np.ascontiguousarray(ct_full[tuple(slice(int(o), int(o) + n) for o, n in zip(offset, lab.shape))]) \
                if ct_full is not None else None
            resolve_fragments(lab, hu, geo, offset, log)
            lap("fragments")

            # ---------------------------------------------------------------------- 3. label drift
            split_keep = {}
            drift = resolve_label_drift(lab, geo, log, keep=split_keep) if relabel else {"status": "disabled"}
            lap("label_drift")
            instances, inst_map = find_instances(lab, hu, geo, offset, log)
            lap("instances")

            # ---------------------------------------------------------------------- 4. sacrum + lost level
            if instances:
                sacrum = detect_sacrum(instances[0], lab, inst_map, offset, ct_full, geo)
                log.add("S7_sacrum_anchor", "flagged", instances[0]["label"], 0,
                        ("anchor set: " if sacrum["found"] else "no anchor: ") + sacrum.get("reason", ""),
                        **{k: v for k, v in sacrum.items() if k not in ("found", "reason")})
            lap("sacrum")
            if relabel and instances and split_keep:
                lost = resolve_lost_level(lab, geo, offset, split_keep, drift, sacrum, instances, log)
                if lost["relabelled"]:  # the order check should judge the corrected chain
                    instances, inst_map = find_instances(lab, hu, geo, offset, CaseLog(case, geo.vox_mm3, False))
            del split_keep
            lap("lost_level")

            # ---------------------------------------------------------------------- 5. order check
            order_flags = check_level_order(instances, log, sacrum)
            work[tl] = lab
            lap("order_check")
        out_full[wbox] = work
        result["qa_after"] = label_qa(work, woff, geo)
        result["voxels_changed"] = int(np.count_nonzero(work != before))
        result["instances"] = [
            {"id": i["id"], "label_after": label_name(i["label"]), "voxels": i["voxels"], "mm3": i["mm3"],
             "centroid_world_mm": i["centroid_world_mm"], "truncated": i["truncated"],
             "fov_edge_distance_mm": i["fov_edge_distance_mm"], "median_hu": i["median_hu"],
             "duplicate_component": i["duplicate_component"]} for i in instances]
    result.update(label_drift=drift, lost_level=lost, possible_missed_level=lost["possible_missed_level"],
                  sacrum=sacrum, order_flags=order_flags)

    # ------------------------------------------------------------------------------ write + log
    save_outputs(out_dir, out_full, work, wbox, mask_img, os.path.join(in_dir, "segmentations"), to_file)
    lap("write")
    times["total"] = time.perf_counter() - t0
    result["summary"] = {
        "steps": log.summary(),
        "n_changes": len(log.changes),
        "labels_before": result.get("qa_before", {}).get("labels", []),
        "labels_after": result.get("qa_after", {}).get("labels", []),
        "voxels_changed": result.get("voxels_changed", 0),
        "n_order_flags": len(order_flags),
    }
    result["changes"] = log.changes
    result["timing_s"] = times
    result = to_jsonable(result)
    text = json.dumps(result, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "postprocessing_log.json"), "w", encoding="utf-8") as fh:
        fh.write(text)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{case}_postprocessing_log.json"), "w", encoding="utf-8") as fh:
            fh.write(text)
    if verbose:
        s = result["summary"]
        print(f"[{case}] summary: {s['n_changes']} log entries, {s['voxels_changed']} voxels changed, "
              f"{s['n_order_flags']} order flag(s)")
        for step, d in sorted(s["steps"].items()):
            print(f"[{case}]   {step}: " + ", ".join(f"{k}={v}" for k, v in sorted(d.items())))
    if timing:
        print(f"[{case}] timing (s): " + ", ".join(f"{k}={v:.2f}" for k, v in times.items()))
    return result


def find_cases(input_root, subset=None):
    """Sorted case folders of ``input_root`` that contain combined_labels.nii.gz (optionally a subset)."""
    cases = sorted(d for d in os.listdir(input_root)
                   if os.path.isfile(os.path.join(input_root, d, "combined_labels.nii.gz")))
    if subset:
        missing = [c for c in subset if c not in cases]
        if missing:
            raise SystemExit(f"cases not found in {input_root}: {missing}")
        cases = [c for c in cases if c in set(subset)]
    return cases


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="prediction root (<input>/<case>/combined_labels.nii.gz)")
    ap.add_argument("--ct_root", default=None, help="CT root (<ct_root>/<case>/ct.nii.gz)")
    ap.add_argument("--output", required=True, help="output root (same layout as --input)")
    ap.add_argument("--cases", nargs="+", default=None, help="process only these case folders")
    ap.add_argument("--log_dir", default=None, help="also copy each case log to this folder")
    ap.add_argument("--verbose", action="store_true", help="print every logged change and the summary")
    ap.add_argument("--timing", action="store_true", help="print per-stage timings")
    ap.add_argument("--no_relabel", action="store_true", help="skip the label-drift and lost-level stages (the only ones that can change a level)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cases = find_cases(args.input, args.cases)
    if not cases:
        print(f"no cases with combined_labels.nii.gz under {args.input}", file=sys.stderr)
        return 1
    failed = []
    for case in cases:
        t = time.perf_counter()
        try:
            res = process_case(case, args.input, args.ct_root, args.output, verbose=args.verbose,
                               log_dir=args.log_dir, timing=args.timing, relabel=not args.no_relabel)
            s = res["summary"]
            print(f"{case}: {s['voxels_changed']} voxels changed, {s['n_order_flags']} order flag(s), "
                  f"{time.perf_counter() - t:.1f} s")
        except Exception:  # keep going with the other cases, report at the end
            traceback.print_exc()
            failed.append(case)
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


# =============================================================================================================
# File-writing plumbing  (nothing below decides anything about anatomy; safe to skip when reading)
#
# Why this exists: nib.save would write exactly the same files (same header, same voxels), just about 20 s
# slower on a big case.  We write 25 files per case (one combined map + 24 single-level maps) and almost all
# of every single-level file is zeros, so we stream each file straight into a gzip stream, compress an
# all-zero slab only once, and compress several files in parallel.  Each header is still produced by nibabel.
# (Compressing the same block of zeros 24 times per case felt like a poor use of everyone's afternoon.)
# =============================================================================================================
WRITE_WORKERS = 6  # parallel gzip writers (zlib releases the GIL); each holds a few slabs, not a volume
WRITE_SLAB_SLICES = 16  # slices along the last file axis streamed to the compressor at a time
WRITE_GZIP_LEVEL = 1  # same as nibabel's default for .nii.gz
GZIP_HEADER = bytes([0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 4, 0xFF])  # [RFC 1952] deflate, no name/mtime, fastest, OS unknown
CRC32_POLY = 0xEDB88320  # [reflected CRC-32 polynomial, RFC 1952]
NIFTI1_HEADER_BYTES = 348  # [NIfTI-1 standard]


class _HeaderCaptured(Exception):
    """Raised by ``_HeaderSink`` when nibabel starts writing voxel data."""


class _HeaderSink(io.BytesIO):
    """File object that keeps what nibabel writes before the voxel data, then aborts the save.
    (nibabel seeks to ``vox_offset`` after the header and extensions, then streams the voxels.)"""

    data_offset = None

    def seek(self, pos, whence=0):
        if whence == 0 and pos >= NIFTI1_HEADER_BYTES:
            self.data_offset = pos
        return super().seek(pos, whence)

    def write(self, b):
        if len(b) and self.data_offset is not None:
            raise _HeaderCaptured
        return super().write(b)


def nifti_prefix(img):
    """Bytes nibabel writes before the voxel data of ``img`` (header, extensions, padding to vox_offset)."""
    sink = _HeaderSink()
    try:
        img.to_file_map({"image": nib.fileholders.FileHolder(fileobj=sink)})
    except _HeaderCaptured:
        pass
    assert sink.data_offset is not None
    return sink.getvalue().ljust(sink.data_offset, bytes(1))


# Yes, this is matrix maths over GF(2) just to write gzip files faster.  It saves about 20 s on the big case and
# the tests check the output against nibabel, so it lives down here where nobody has to read it.
def _gf2_times(mat, vec):
    out, i = 0, 0
    while vec:
        if vec & 1:
            out ^= mat[i]
        vec >>= 1
        i += 1
    return out


def _gf2_square(mat):
    return [_gf2_times(mat, m) for m in mat]


@functools.lru_cache(maxsize=None)
def crc32_zeros_operator(n_bytes):
    """GF(2) matrix that advances a CRC-32 register over ``n_bytes`` zero bytes (as zlib's crc32_combine)."""
    one_bit = [CRC32_POLY] + [1 << i for i in range(31)]
    op = _gf2_square(_gf2_square(_gf2_square(one_bit)))  # one zero byte
    result = [1 << i for i in range(32)]
    while n_bytes:
        if n_bytes & 1:
            result = [_gf2_times(op, col) for col in result]
        op = _gf2_square(op)
        n_bytes >>= 1
    return result


def write_gzip_nifti(path, prefix, slabs, zero_slab):
    """Write ``prefix`` + the byte strings of ``slabs`` as one gzip member.

    A slab that is ``None`` stands for ``zero_slab`` (all zeros).  Compressing zeros is the slowest part of
    writing a sparse mask, so the deflate bytes of ``zero_slab`` are made once and copied: deflate output
    after a full flush is byte-aligned and independent of what came before, so the pieces concatenate into
    a valid stream that decodes to exactly the same voxels.
    """
    zero_deflated = zlib.compressobj(WRITE_GZIP_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
    zero_deflated = zero_deflated.compress(zero_slab) + zero_deflated.flush(zlib.Z_FULL_FLUSH)
    comp = zlib.compressobj(WRITE_GZIP_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)  # raw deflate; gzip framing below
    zero_crc = zlib.crc32(zero_slab)
    zero_shift = crc32_zeros_operator(len(zero_slab))
    crc, size, pending_zero = 0, 0, False
    with open(path, "wb") as fh:
        fh.write(GZIP_HEADER)
        for chunk in itertools.chain([prefix], slabs):
            if chunk is None:
                if not pending_zero:
                    fh.write(comp.flush(zlib.Z_FULL_FLUSH))
                    pending_zero = True
                fh.write(zero_deflated)
                crc = _gf2_times(zero_shift, crc) ^ zero_crc  # crc of the data so far + zero_slab
                size += len(zero_slab)
                continue
            pending_zero = False
            fh.write(comp.compress(chunk))
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
        fh.write(comp.flush(zlib.Z_FINISH))
        fh.write(struct.pack("<II", crc & 0xFFFFFFFF, size & 0xFFFFFFFF))


def save_outputs(out_dir, out_full, work, wbox, mask_img, seg_in_dir, to_file=lambda a: a):
    """Write combined_labels.nii.gz and the 24 per-level files (combined == k) with the input headers.

    ``out_full``/``work`` are in the internal (C-ordered) axis order; ``to_file`` maps an internal array to
    the file's axis order (a transpose view).  Each output reuses the header and affine of its input file
    (dtype set to uint8).  Voxels are streamed slab by slab straight into a gzip member, so a per-level
    file never needs a full-size array.  The written headers are reloaded and checked.
    """
    seg_dir = os.path.join(out_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    shape = to_file(out_full).shape  # file axis order
    # Slabs run along the last file axis, which is internal axis 0 when the file is F-ordered (a transpose
    # view), so each slab is a contiguous block of the internal arrays.
    ax = 0 if to_file(np.empty((1, 2, 3))).shape == (3, 2, 1) else 2
    objs = ndi.find_objects(work, max_label=NUM_LABELS) if work.size else [None] * NUM_LABELS
    woff = [s.start for s in wbox]
    n_slices = out_full.shape[ax]
    zeros = np.zeros(shape[:2] + (WRITE_SLAB_SLICES,), np.uint8).tobytes(order="F")
    stub = np.broadcast_to(np.uint8(0), shape)  # header only; never read
    prefixes = {}

    def make(ref):
        hdr = ref.header.copy()
        hdr.set_data_dtype(np.uint8)
        img = ref.__class__(stub, ref.affine, hdr)
        img.set_data_dtype(np.uint8)
        assert img.shape == shape and np.array_equal(img.affine, ref.affine)
        assert img.get_data_dtype() == np.uint8
        return img

    jobs = [(os.path.join(out_dir, "combined_labels.nii.gz"), None, mask_img)]
    for v, name in LABEL_NAMES.items():
        ref = mask_img
        p_in = os.path.join(seg_in_dir, f"vertebrae_{name}.nii.gz")
        if os.path.exists(p_in):
            r = nib.load(p_in)
            if tuple(r.shape[:3]) == tuple(shape) and np.allclose(r.affine, mask_img.affine, atol=AFFINE_ATOL):
                ref = r
        jobs.append((os.path.join(seg_dir, f"vertebrae_{name}.nii.gz"), v, ref))

    def slabs(v):
        s = None if v is None else objs[v - 1]
        for z0 in range(0, n_slices, WRITE_SLAB_SLICES):
            z1 = min(z0 + WRITE_SLAB_SLICES, n_slices)
            window = tuple(slice(z0, z1) if i == ax else slice(None) for i in range(3))
            if v is None:
                slab = out_full[window]
            else:
                lo, hi = (max(z0, s[ax].start + woff[ax]), min(z1, s[ax].stop + woff[ax])) if s else (0, 0)
                if lo >= hi:  # no voxels of this label in the slab
                    yield None if z1 - z0 == WRITE_SLAB_SLICES else zeros[: shape[0] * shape[1] * (z1 - z0)]
                    continue
                slab = np.zeros(out_full[window].shape, np.uint8)
                dst = tuple(slice(lo - z0, hi - z0) if i == ax else slice(s[i].start + woff[i], s[i].stop + woff[i])
                            for i in range(3))
                src = tuple(slice(lo - woff[ax], hi - woff[ax]) if i == ax else s[i] for i in range(3))
                slab[dst] = work[src] == v
            yield to_file(slab).tobytes(order="F")

    def write(job):
        path, v, ref = job
        key = (ref.header.binaryblock, ref.affine.tobytes())
        if key not in prefixes:
            prefixes[key] = nifti_prefix(make(ref))
        write_gzip_nifti(path, prefixes[key], slabs(v), zeros)
        chk = nib.load(path)
        assert tuple(chk.shape) == tuple(shape), path
        assert np.allclose(chk.affine, ref.affine, atol=AFFINE_ATOL), path
        assert chk.get_data_dtype() == np.uint8, path
        return path

    # Empty per-level files with identical header and affine are byte-identical: compress one, copy the rest.
    first_empty, copies = {}, []
    for j in jobs:
        if j[1] is not None and objs[j[1] - 1] is None:
            key = (j[2].header.binaryblock, j[2].affine.tobytes())
            if key in first_empty:
                copies.append((first_empty[key], j))
            else:
                first_empty[key] = j
    copied = {id(j) for _, j in copies}
    with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as ex:
        written = list(ex.map(write, [j for j in jobs if id(j) not in copied]))
    for src, j in copies:
        shutil.copyfile(src[0], j[0])
        written.append(j[0])
    return written


# If you read this far: the tests pass, the coffee is cold, and L6 is still a known limitation (see REPORT.md).
if __name__ == "__main__":
    sys.exit(main())
