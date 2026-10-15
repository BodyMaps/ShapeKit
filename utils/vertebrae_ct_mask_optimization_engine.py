#!/usr/bin/env python3
"""Post-processing of SuPreM vertebrae segmentations (C1-L5).

What this script does
---------------------
SuPreM's vertebrae model labels each of the 24 vertebrae C1-L5 in a CT scan. Its raw masks can break basic
anatomy: one vertebra split between two labels, labels shifted by one level, a spinous process labelled as the
neighbouring vertebra, or small stray pieces far from the spine. This script corrects those errors so that every
vertebra is one labelled structure, the levels run C1 -> L5 in order, and the files keep exactly the format the
model wrote.

Requirements
------------
- Python 3.9 or newer with: numpy, scipy, nibabel, connected-components-3d (imported as cc3d), scikit-image
      pip install numpy scipy nibabel connected-components-3d scikit-image
- About 3 GB of memory for a 512 x 512 x 1400 CT. A case takes roughly 1-3 minutes on one CPU core.

Input
-----
Two folders, one sub-folder per case (e.g. BDMAP_00000031):
    <input>/<case>/combined_labels.nii.gz                 the model's label map (see LABEL VALUES)
    <input>/<case>/segmentations/vertebrae_<level>.nii.gz one binary mask per vertebra (C1 ... L5)
    <ct_root>/<case>/ct.nii.gz                             the CT the labels were predicted from (required:
                                                           several steps read the bone density and the thin
                                                           gaps between neighbouring bones; it must be on the
                                                           same voxel grid as the labels). A case whose CT is
                                                           missing or on another grid fails with a message;
                                                           the other cases are still processed.

Output
------
    <output>/<case>/...  the same files with the same names, shapes, affines and headers as the input files.
                         Every input mask gets an output mask (an empty vertebra stays an all-zero mask), and
                         combined_labels.nii.gz is rebuilt from the corrected per-vertebra masks.
    a log file           one line per correction (default: postprocessing_vertebrae.log next to the output folder)

Usage
-----
    python postprocessing_vertebrae.py --input AbdomenAtlasDemoPredict_raw --output AbdomenAtlasDemoPredict
    python postprocessing_vertebrae.py --input RAW --output OUT --ct_root CTS --cases BDMAP_00000006

By default the CTs are looked for in a folder named AbdomenAtlasDemo next to the input folder, or else next to the
output folder. The script never
modifies the input (it checks this with sha256 checksums), refuses to write into the input folder, and does not
overwrite existing case folders in the output unless --overwrite is given. It exits with a non-zero code if any
case fails. Two runs on the same input give identical files.

LABEL VALUES
------------
    1 = L5, 2 = L4, 3 = L3, 4 = L2, 5 = L1, 6 = T12, 7 = T11, ..., 17 = T1, 18 = C7, ..., 24 = C1, 0 = background

How it works (overview)
-----------------------
All work is done on the volume reoriented to RAS (x = to the patient's left, y = to the front, z = up) and is
converted back before saving. The steps below are applied to the range L5-T4, where the corrections are needed;
C1-T3 keep the model's labels except for the final clean-up, which covers all 24 levels.

1. Vertebral bodies: the bodies are found from the shape of the spine mask alone (a deep erosion separates
   them at the discs) and named as one consecutive sequence that best matches the model's labels. This fixes
   bodies that the model split or shifted.
2. Arches: every vertebral arch is attached to its own body through the pedicles. Where the bone next to a body
   (its pedicles) carries a different label and the model also mislabelled a body nearby, the model's arch
   label is moved to the body its pedicles belong to.
3. Thoracic spinous processes: each process is isolated from its neighbours using the thin gaps in the CT and
   given the label of its root (where it leaves the spinal canal).
4. Lumbar spinous processes: each takes the level of the slab it lies in along the curve of the spine.
5. Clean-up: pieces of a label far from that vertebra are removed; small detached pieces are given to the
   vertebra they belong to; enclosed holes of bone density are filled (dark joint spaces stay open).

Safety checks: steps 1-4 renumber the spine from the bodies found in step 1, so before they run the bodies are
checked (enough bodies, no body more than one level away from the model's label, no missing or split body, as many
bodies as the model has complete levels, and a numbering that fits the vertebra above the range). If a check fails,
or the scan does not contain L5-T4 at all, steps 1-4 are skipped for that case and the model's labels are kept; only
step 5 is applied, without moving detached pieces between labels (FALLBACK_REASSIGN_PIECES), since that relies on
the model's labels. The reason is written to the log and the case's status line says "steps 1-4 skipped". A vertebra
that the scan cuts off, and for which no complete body is found, keeps the model's labels in steps 1-4.

Numbering the bodies: when the scan holds the vertebra just above the range (T3, never changed by steps 1-4), the
bodies are numbered down from it (ANCHOR_TO_LEVEL_ABOVE); otherwise the numbering that most body voxels agree with
is used.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed

# =====================================================================================================
# Labels and file names
# =====================================================================================================
LEVELS = ["L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4",
          "T3", "T2", "T1", "C7", "C6", "C5", "C4", "C3", "C2", "C1"]
LABEL_NAME = {value: name for value, name in enumerate(LEVELS, start=1)}   # 1 -> "L5", ..., 24 -> "C1"
LABEL_VALUE = {name: value for value, name in LABEL_NAME.items()}          # "L5" -> 1, ..., "C1" -> 24
COMBINED_FILE = "combined_labels.nii.gz"
MASK_DIR = "segmentations"
TOP_LEVEL = "T4"               # steps 1-4 process L5 up to this level

# =====================================================================================================
# Settings. Distances are in mm and volumes in cm3, so they do not depend on the voxel size. CT values are
# in Hounsfield units (HU).
# =====================================================================================================
BONE_HU = 150.0                # below this a voxel is unlikely to be bone

# Step 1: vertebral bodies
BODY_CORE_DEPTH_MM = 6.0       # erosion depth that separates neighbouring bodies at the discs
BODY_CORE_MIN_CM3 = 0.3        # smaller cores are not bodies
BODY_FRONT_FRACTION = 0.5      # body cores lie in the front half of the spine's front-to-back extent
BODY_GROW_MM = 14.0            # a body is its core grown back by about the erosion depth plus the outer shell
SOFT_TISSUE_COST = 10.0        # path cost per mm through non-bone (bone costs 1), so regions meet at joints
SMOOTHING_MM = 1.0             # Gaussian smoothing of the CT before the bone test in steps 1, 2 and 4

# Safety checks before steps 1-4 (if one fails, steps 1-4 are skipped and the model's labels are kept)
MIN_BODIES = 3                 # fewer bodies than this: too little spine to number reliably
MAX_NAME_SHIFT = 1             # a body's new name may differ from the model's label by at most this many levels
BODY_GAP_RATIO = (0.6, 1.6)    # spacing to the next body vs the neighbouring spacings: outside = a body missing or split
LEVEL_MIN_CM3 = 2.0            # a level counts as present in the scan when the model labels at least this much of it
GRID_TOLERANCE_MM = 1e-3       # CT and labels must share the voxel grid (affines equal up to this, in mm)
EDGE_SLICES = 2                # levels touching the first or last slices are cut off by the scan and are not counted
ANCHOR_TO_LEVEL_ABOVE = True   # when the scan holds the vertebra above the range, number the bodies down from it
FALLBACK_REASSIGN_PIECES = False  # when steps 1-4 are skipped, also move detached pieces between labels (step 5)

# Step 2: pedicles and arches
PEDICLE_DEPTH_THORACIC_MM = 15.0   # pedicle zone behind the body, in the vertebra's own (tilted) frame
PEDICLE_DEPTH_LUMBAR_MM = 10.0     # pedicle zone behind the body, horizontal and shallow
PEDICLE_ZONE_SIDE_MM = (4.0, 25.0) # pedicle zone: this far to each side of the body's centre
PEDICLE_MIN_SHARE = 0.6            # a pedicle zone "carries" a label when the label covers this share of it

# Step 3: thoracic spinous processes
SPINOUS_BAND_MM = 4.0          # half-width of the midline band in which the processes are isolated
SPINOUS_ROOT_MM = 5.0          # the root is the front part of a process (nearest the spinal canal), this deep
ROOT_MIN_SHARE = 0.8           # a root's label must cover at least this share of the root
SPINOUS_MIN_MOVE_CM3 = 0.2     # smaller label errors on a process are left alone (too small to be reliable)
SPINOUS_TIP_WIDTH_MM = 12.0    # a process tip may reach this far from the midline
SPINOUS_KEEPOUT_MM = 4.0       # a relabelled tip never extends into bone this close to another process
BAR_THRESHOLDS_HU = (600, 500, 450, 400, 350, 300)   # bone thresholds, high to low, for isolating processes
BAR_FLOOR_HU = 150.0           # softer bone inside a process's shell, down to this value, belongs to it
BAR_MIN_SEED_CM3 = 0.15        # smallest dense bone piece that starts a process of its own

# Step 4: lumbar spinous processes
LUMBAR_BAND_MM = 8.0           # half-width of the midline band holding the lumbar processes
LUMBAR_PROCESS_OFFSET_MM = 12.0  # a lumbar process sits about this far below its body centre along the spine
CANAL_MIN_MM = 8.0             # processes start at least this far behind the bodies' back wall
CUT_SNAP_MM = 8.0              # each slab boundary may move this far to the nearest gap between processes
CURVE_FRAME_MARGIN_MM = 10.0   # length of spine around a slab boundary used for the curvature check
MODEL_TRUST_SHARE = 0.85       # a slab where the model's labels already agree this much is left as it is
PROCESS_MAX_LENGTH_MM = 40.0   # longer bone pieces may be two touching processes and are cut voxel by voxel

# Step 5: clean-up
FAR_PIECE_MM = 50.0            # a piece this far from the rest of its vertebra cannot belong to it
PIECE_MIDLINE_MM = 8.0         # a piece is "on the midline" when its centre is this close to the midline
PIECE_BACK_FRACTION = 0.5      # "in the back of the spine" = in the back half of its front-to-back extent
TRANSVERSE_BAND_MM = (12.0, 55.0)  # where transverse processes lie: this far to the side of the midline
TRANSVERSE_ROOT_MM = 5.0       # the root of a transverse process is its inner end, this long
PIECE_MAX_ROUNDS = 5           # detached-piece passes, repeated until nothing changes
PIECE_SKIP_LEVELS = ("C1", "C2")   # C1 (a ring) and C2 often have separate pieces that are correct


# =====================================================================================================
# Reading, orientation and small helpers
# =====================================================================================================
def load_as_ras(path: str):
    """Load a NIfTI file and reorient its voxel array to RAS (x = left, y = front, z = up).

    Args:
        path: path of a .nii or .nii.gz file.

    Returns:
        (image, orientation, data, spacing): the nibabel image as stored on disk, the orientation transform
        (needed by `restore_orientation` to undo the reorientation), the reoriented voxel array, and the voxel
        size in mm along the reoriented axes.
    """
    image = nib.load(path)
    orientation = nib.orientations.io_orientation(image.affine)
    data = nib.orientations.apply_orientation(np.asarray(image.dataobj), orientation)
    spacing = np.array(image.header.get_zooms()[:3])[orientation[:, 0].astype(int)]
    return image, orientation, data, spacing


def restore_orientation(data: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Undo `load_as_ras`: bring a RAS voxel array back to the voxel order of the original file."""
    back = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")), orientation)
    return nib.orientations.apply_orientation(data, back)


def midline_x(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Left-right position of the spine's midline in every axial slice.

    Args:
        mask: boolean RAS volume of the spine (or part of it).
        spacing: voxel size in mm.

    Returns:
        A float array with one x index per slice (z), the median x of the mask in that slice, smoothed over
        15 mm along the spine and interpolated over slices without mask.
    """
    points = np.argwhere(mask)
    slices = np.unique(points[:, 2])
    x_per_slice = np.array([np.median(points[points[:, 2] == z, 0]) for z in slices])
    x_per_slice = ndimage.median_filter(x_per_slice, size=max(3, int(round(15 / spacing[2])) | 1), mode="nearest")
    return np.interp(np.arange(mask.shape[2]), slices, x_per_slice)


def midline_back_band(mask: np.ndarray, spacing: np.ndarray, half_width_mm: float) -> np.ndarray:
    """Voxels of `mask` within `half_width_mm` of the midline and in the back 45% of the spine.

    This is where the spinous processes and the laminae that join them lie.
    """
    xs = np.arange(mask.shape[0])[:, None, None]
    ys = np.arange(mask.shape[1])[None, :, None]
    x_mid = midline_x(mask, spacing)[None, None, :]
    y_back = np.where(mask, ys, 10 ** 6).min(axis=(0, 1)).astype(float)
    y_front = np.where(mask, ys, -1).max(axis=(0, 1)).astype(float)
    in_back = (ys - y_back[None, None, :]) <= 0.45 * (y_front - y_back)[None, None, :]
    return (np.abs(xs - x_mid) * spacing[0] <= half_width_mm) & in_back & mask


def split_bone_by_ct_gaps(region: np.ndarray, ct: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Divide the bone inside `region` into separate pieces ("bars") along the thin gaps visible in the CT.

    Neighbouring spinous processes are separated by 1-2 mm of soft tissue (ligaments), which shows up as a darker
    line between two bright bone surfaces. The bone is examined at several thresholds from high to low: at the
    highest threshold the densest bone forms seeds; at each lower threshold a connected piece that contains one
    seed joins it, a piece that contains several seeds is divided between them along the darkest line, and a
    piece without a seed becomes a new seed if it is large enough. Last, each bar grows into the softer bone
    inside its shell. Bones that touch at every threshold stay one bar.

    Args:
        region: boolean volume of where to look (e.g. a midline band).
        ct: CT volume (HU), same shape as `region`, not smoothed.
        spacing: voxel size in mm.

    Returns:
        An int32 volume with one id per bar (1, 2, ...) and 0 elsewhere.
    """
    voxel_mm3 = float(np.prod(spacing))
    points = np.argwhere(region)
    lo, hi = points.min(0), points.max(0) + 1
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    region_box, ct_box = region[box], ct[box]
    bars = np.zeros(region_box.shape, np.int32)
    next_id = 1
    for threshold in BAR_THRESHOLDS_HU:
        pieces, _ = cc3d.connected_components(region_box & (ct_box >= threshold), connectivity=6, return_N=True)
        for piece_id, piece_slice in enumerate(ndimage.find_objects(pieces), start=1):
            if piece_slice is None:
                continue
            piece = pieces[piece_slice] == piece_id
            seeds = np.unique(bars[piece_slice][piece])
            seeds = seeds[seeds > 0]
            if len(seeds) == 0:
                if piece.sum() * voxel_mm3 >= BAR_MIN_SEED_CM3 * 1000:
                    bars[piece_slice][piece] = next_id
                    next_id += 1
            elif len(seeds) == 1:
                bars[piece_slice][piece] = seeds[0]
            else:
                markers = np.where(piece, bars[piece_slice], 0)
                bars[piece_slice][piece] = watershed(-ct_box[piece_slice], markers, mask=piece)[piece]
    soft_bone = region_box & (ct_box >= BAR_FLOOR_HU)
    bars = np.where(soft_bone, watershed(-ct_box, bars, mask=soft_bone), 0)
    result = np.zeros(region.shape, np.int32)
    result[box] = bars
    return result


def describe_pairs(volumes_cm3: dict, limit: int = 6) -> str:
    """Summarise label changes {(old, new): cm3}, largest first, e.g. "2.91 cm3 L1 -> T12, 0.43 cm3 T12 -> T11";
    beyond `limit` entries the rest is summed up as "and N smaller changes"."""
    name = lambda v: LABEL_NAME.get(v, "background")
    items = sorted(volumes_cm3.items(), key=lambda item: -item[1])
    shown = ", ".join(f"{v:.2f} cm3 {name(a)} -> {name(b)}" for (a, b), v in items[:limit])
    rest = items[limit:]
    if rest:
        shown += f", and {len(rest)} smaller changes ({sum(v for _, v in rest):.2f} cm3)"
    return shown or "nothing"


def describe_changes(before: np.ndarray, after: np.ndarray, spacing: np.ndarray, limit: int = 6) -> str:
    """Summarise how labels changed between two label volumes (see `describe_pairs`)."""
    changed = before != after
    voxel_cm3 = float(np.prod(spacing)) / 1000
    pairs = {}
    for old, new in zip(before[changed].tolist(), after[changed].tolist()):
        pairs[(old, new)] = pairs.get((old, new), 0) + voxel_cm3
    return describe_pairs(pairs, limit)


# =====================================================================================================
# Step 1: vertebral bodies
# =====================================================================================================
def find_vertebral_bodies(spine: np.ndarray, spacing: np.ndarray) -> list:
    """Find one core per vertebral body from the shape of the spine mask, without using its labels.

    Eroding the mask by BODY_CORE_DEPTH_MM removes the thin discs, the arches and the processes, so what is left
    falls apart into one thick core per body. Cores in the back half of the spine (thick parts of the arches)
    are discarded.

    Args:
        spine: boolean RAS volume of the processed range of the spine.
        spacing: voxel size in mm.

    Returns:
        A list of (core_mask, volume_cm3, front_fraction) tuples, ordered from the lowest body upwards.
    """
    depth = ndimage.distance_transform_edt(spine, sampling=spacing)
    cores, n_cores = cc3d.connected_components(depth > BODY_CORE_DEPTH_MM, connectivity=6, return_N=True)
    voxel_mm3 = float(np.prod(spacing))
    ys = np.arange(spine.shape[1])[None, :, None]
    y_back = np.where(spine, ys, 10 ** 6).min(axis=(0, 1)).astype(float)
    y_front = np.where(spine, ys, -1).max(axis=(0, 1)).astype(float)
    bodies = []
    for core_id in range(1, n_cores + 1):
        core = cores == core_id
        volume_cm3 = core.sum() * voxel_mm3 / 1000
        if volume_cm3 < BODY_CORE_MIN_CM3:
            continue
        points = np.argwhere(core)
        z = points[:, 2]
        front_fraction = float(np.median((points[:, 1] - y_back[z]) / np.maximum(y_front[z] - y_back[z], 1)))
        if front_fraction >= BODY_FRONT_FRACTION:
            bodies.append((core, volume_cm3, front_fraction))
    bodies.sort(key=lambda body: np.argwhere(body[0])[:, 2].mean())
    return bodies


def assign_regions_to_bodies(spine: np.ndarray, ct: np.ndarray, bodies: list, spacing: np.ndarray) -> np.ndarray:
    """Give every spine voxel to the body it reaches by the cheapest path inside the mask.

    A path costs 1 per mm through bone and SOFT_TISSUE_COST per mm through non-bone, so the boundary between two
    vertebrae falls in the disc or joint between them.

    Returns:
        An int16 volume with the body's index (1 = lowest body) per spine voxel, 0 outside the spine.
    """
    smoothed = ndimage.gaussian_filter(ct, sigma=SMOOTHING_MM / spacing)
    cost = np.where(spine, np.where(smoothed >= BONE_HU, 1.0, SOFT_TISSUE_COST), np.inf)
    best = np.full(spine.shape, np.inf, np.float32)
    region = np.zeros(spine.shape, np.int16)
    for index, (core, _, _) in enumerate(bodies, start=1):
        seeds = np.argwhere(core)
        distance, _ = MCP_Geometric(cost, sampling=tuple(spacing)).find_costs(
            [tuple(p) for p in seeds[:: max(1, len(seeds) // 2000)]])
        closer = distance < best
        best[closer] = distance[closer]
        region[closer] = index
    region[~spine] = 0
    return region


def body_masks(spine: np.ndarray, region: np.ndarray, bodies: list, names: dict, spacing: np.ndarray) -> np.ndarray:
    """Full body masks: each core grown back by BODY_GROW_MM, inside the spine and inside its own region.

    Returns:
        An int16 volume holding each body's final label value (from `names`), 0 elsewhere.
    """
    body = np.zeros(spine.shape, np.int16)
    for index, (core, _, _) in enumerate(bodies, start=1):
        near = ndimage.distance_transform_edt(~core, sampling=spacing) <= BODY_GROW_MM
        body[near & spine & (region == index)] = names[index]
    return body


def name_bodies(labels: np.ndarray, bodies: list, top_value=None):
    """Name the bodies as one consecutive sequence of levels.

    With `top_value` given (the scan holds the vertebra just above the range, which steps 1-4 leave unchanged), the
    top body is named `top_value` and the others are numbered down from it. Otherwise every possible starting level
    is tried and the one under which most body voxels keep the model's label wins. (That vote can be wrong on a
    short scan where the model is one level off on the larger bodies; the vertebra above is the more reliable
    reference when it is in the scan.)

    Returns:
        (names, agreement, model_label_per_body): a dict body index -> label value, the share of body voxels
        whose model label equals the chosen name, and a dict label value -> the model's majority label inside
        that body.
    """
    n_bodies = len(bodies)
    votes = np.array([np.bincount(labels[core], minlength=25)[:25] for core, _, _ in bodies], float)
    best_start, best_score = 1, -1.0
    starts = [top_value - n_bodies + 1] if top_value is not None and top_value >= n_bodies else range(1, 24 - n_bodies + 2)
    for start in starts:
        score = sum(votes[i, start + i] for i in range(n_bodies))
        if score > best_score:
            best_start, best_score = start, score
    names = {i + 1: best_start + i for i in range(n_bodies)}
    model_label = {names[i + 1]: int(votes[i, 1:].argmax()) + 1 for i in range(n_bodies)}
    return names, best_score / max(votes[:, 1:].sum(), 1), model_label


def levels_cut_off(labels: np.ndarray, range_mask: np.ndarray, top_value: int, scan_z_edges: tuple) -> set:
    """Levels in the range whose labelled voxels reach the scan's bottom or top edge (within EDGE_SLICES slices),
    i.e. vertebrae the scan shows only in part."""
    cut = set()
    for value in range(1, top_value + 1):
        level_mask = range_mask & (labels == value)
        if not level_mask.any():
            continue
        z = np.nonzero(level_mask.any(axis=(0, 1)))[0]
        if ((scan_z_edges[0] is not None and z.min() <= scan_z_edges[0] + EDGE_SLICES)
                or (scan_z_edges[1] is not None and z.max() >= scan_z_edges[1] - EDGE_SLICES)):
            cut.add(value)
    return cut


def check_bodies(bodies: list, names: dict, model_label: dict, labels: np.ndarray, spacing: np.ndarray,
                 range_mask: np.ndarray, top_value: int, scan_z_edges: tuple, level_above_in_scan: bool):
    """Decide whether the bodies found in step 1 can be trusted to renumber the spine.

    Steps 1-4 change labels based on these bodies, so a mistake here would spread to the whole spine. The bodies
    are rejected when:
      - fewer than MIN_BODIES were found;
      - a body's new name differs from the model's label for it by more than MAX_NAME_SHIFT levels (correcting a
        one-level shift is the purpose of step 1; a larger difference means bodies were missed or merged);
      - the spacing between two neighbouring bodies is far from the neighbouring spacings (BODY_GAP_RATIO), which
        happens when a body is missed (a double gap) or one body is split in two (a half gap);
      - fewer bodies were found than the model has levels in the range, not counting levels cut off by the scan;
      - the numbering does not fit the vertebra above the range, which steps 1-4 leave unchanged: no body may be
        named above the range's top level, and when the scan holds the vertebra above, the top body must be the
        top level. (The numbering is chosen by majority, so when the model is one level off over most of a short
        scan, the majority can be wrong; this check catches that whenever the vertebra above is in the scan.)

    Args:
        bodies, names, model_label: from `find_vertebral_bodies` and `name_bodies`.
        labels: the model's labels in the working box; range_mask: the processed range in that box.
        top_value: the top level of the range; scan_z_edges: (first, last) slice index of the box that lie on the
            scan's bottom or top edge, or None for a side that does not.
        level_above_in_scan: whether the model labels the vertebra just above the range (at least LEVEL_MIN_CM3).

    Returns:
        None when the bodies are trusted, else a sentence explaining why not.
    """
    if len(bodies) < MIN_BODIES:
        return f"only {len(bodies)} vertebral bodies were found (at least {MIN_BODIES} are needed)"
    top_name = max(names.values())
    if top_name > top_value or (level_above_in_scan and top_name != top_value):
        return (f"the top body would be numbered {LABEL_NAME[top_name]}, which does not fit the vertebra above "
                f"({LABEL_NAME[top_value + 1]}, left unchanged), so the numbering is probably off by a level")
    for index in range(1, len(bodies) + 1):
        level = names[index]
        if abs(level - model_label[level]) > MAX_NAME_SHIFT:
            return (f"the body numbered {LABEL_NAME[level]} is labelled {LABEL_NAME[model_label[level]]} by the model, "
                    f"{abs(level - model_label[level])} levels apart, so bodies were probably missed or merged")
    centres = np.array([np.argwhere(core).mean(0) * spacing for core, _, _ in bodies])
    gaps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    for i, gap in enumerate(gaps):
        neighbours = [gaps[j] for j in (i - 1, i + 1) if 0 <= j < len(gaps)]
        if not neighbours:
            continue
        ratio = gap / np.mean(neighbours)
        if not BODY_GAP_RATIO[0] <= ratio <= BODY_GAP_RATIO[1]:
            return (f"the spacing between bodies {LABEL_NAME[names[i + 1]]} and {LABEL_NAME[names[i + 2]]} is "
                    f"{ratio:.1f} times the neighbouring spacings, so a body was probably missed or split")
    voxel_cm3 = float(np.prod(spacing)) / 1000
    cut = levels_cut_off(labels, range_mask, top_value, scan_z_edges)
    expected = sum(1 for value in range(1, top_value + 1) if value not in cut
                   and (range_mask & (labels == value)).sum() * voxel_cm3 >= LEVEL_MIN_CM3)
    if len(bodies) < expected:
        return (f"{len(bodies)} vertebral bodies were found but the model labels {expected} complete levels in this "
                "range, so bodies were probably missed")
    return None


# =====================================================================================================
# Step 2: pedicles and arches
# =====================================================================================================
def pedicle_zones_horizontal(spine, ct, body, bodies, names, spacing, depth_mm):
    """Pedicle zone of each body measured in axial slices: bone just behind the body's back wall.

    The zone spans the body's height, lies 0..`depth_mm` behind its back wall, and PEDICLE_ZONE_SIDE_MM to each
    side of its centre. It is used for the lumbar spine, whose vertebrae sit nearly level.

    Returns:
        A dict label value -> boolean zone volume.
    """
    smoothed = ndimage.gaussian_filter(ct, sigma=SMOOTHING_MM / spacing)
    xs = np.arange(spine.shape[0])[:, None, None]
    ys = np.arange(spine.shape[1])[None, :, None]
    zs = np.arange(spine.shape[2])[None, None, :]
    zones = {}
    for index, (core, _, _) in enumerate(bodies, start=1):
        level = names[index]
        points = np.argwhere(body == level)
        if not len(points):
            continue
        core_z = np.argwhere(core)[:, 2]
        x_centre = points[:, 0].mean()
        back_wall = np.percentile(points[:, 1], 3)
        side = np.abs(xs - x_centre) * spacing[0]
        zones[level] = (spine & (body == 0) & (smoothed >= BONE_HU)
                        & (zs >= core_z.min()) & (zs <= core_z.max())
                        & (ys <= back_wall) & (ys >= back_wall - depth_mm / spacing[1])
                        & (side >= PEDICLE_ZONE_SIDE_MM[0]) & (side <= PEDICLE_ZONE_SIDE_MM[1]))
    return zones


def pedicle_zones_tilted(spine, ct, body, bodies, names, spacing, depth_mm):
    """Pedicle zone of each body measured in the vertebra's own frame (used for the thoracic spine).

    Thoracic vertebrae are tilted by the spine's curvature, so a horizontal zone would drift up or down into the
    neighbouring vertebra's arch. Here the spine's direction at each body is taken from the neighbouring body
    centres; the zone spans the body's extent along that direction and lies 0..`depth_mm` behind the back wall,
    measured perpendicular to it.

    Returns:
        A dict label value -> boolean zone volume.
    """
    smoothed = ndimage.gaussian_filter(ct, sigma=SMOOTHING_MM / spacing)
    xs = np.arange(spine.shape[0])[:, None, None]
    ys_mm = np.arange(spine.shape[1])[None, :, None] * spacing[1]
    zs_mm = np.arange(spine.shape[2])[None, None, :] * spacing[2]
    centres = {}
    for index, (core, _, _) in enumerate(bodies, start=1):
        points = np.argwhere(body == names[index])
        if len(points):
            centres[names[index]] = (points * spacing).mean(0)
    levels_bottom_up = sorted(centres, key=lambda level: centres[level][2])
    zones = {}
    for index, (core, _, _) in enumerate(bodies, start=1):
        level = names[index]
        if level not in centres:
            continue
        position = levels_bottom_up.index(level)
        below = centres[levels_bottom_up[max(position - 1, 0)]]
        above = centres[levels_bottom_up[min(position + 1, len(levels_bottom_up) - 1)]]
        up = np.array([above[1] - below[1], above[2] - below[2]])            # spine direction (y, z)
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else np.array([0.0, 1.0])
        back = np.array([up[1], -up[0]])                                      # perpendicular to it ...
        if back[0] > 0:
            back = -back                                                      # ... pointing backwards
        points = np.argwhere(body == level) * spacing
        core_mm = np.argwhere(core) * spacing
        y0, z0 = centres[level][1], centres[level][2]
        back_wall = np.percentile((points[:, 1] - y0) * back[0] + (points[:, 2] - z0) * back[1], 97)
        core_along = (core_mm[:, 1] - y0) * up[0] + (core_mm[:, 2] - z0) * up[1]
        along = (ys_mm - y0) * up[0] + (zs_mm - z0) * up[1]
        behind = (ys_mm - y0) * back[0] + (zs_mm - z0) * back[1] - back_wall
        side = np.abs(xs - points[:, 0].mean() / spacing[0]) * spacing[0]
        zones[level] = (spine & (body == 0) & (smoothed >= BONE_HU)
                        & (along >= core_along.min()) & (along <= core_along.max())
                        & (behind >= 0) & (behind <= depth_mm)
                        & (side >= PEDICLE_ZONE_SIDE_MM[0]) & (side <= PEDICLE_ZONE_SIDE_MM[1]))
    return zones


def correct_arches(spine, ct, labels, body, bodies, names, spacing, model_label, log):
    """Move each mislabelled arch to the body its pedicles belong to.

    A pedicle joins a body to its own arch, so the label most of a body's pedicle zone carries should be the
    body's own. Where it is not, and the model also mislabelled at least one body in that run of levels (the
    evidence that the model's numbering is off there), every arch voxel carrying the pedicles' label is given
    the body's label. If two bodies claim the same arch label, the arch is divided between their pedicles,
    growing from each pedicle through the softer inside of the bone so that the bright outer shell of each bone
    forms the boundary.

    Args:
        spine, ct, labels: RAS volumes of the processed range (mask, HU, current label values).
        body: body masks from `body_masks`; bodies, names: from `find_vertebral_bodies` / `name_bodies`.
        model_label: the model's majority label inside each body (from `name_bodies`).
        log: list that receives one line per decision.

    Returns:
        (labels, corrected_levels): the corrected label volume and the set of body labels whose arches were
        corrected (used by step 3).
    """
    voxel_mm3 = float(np.prod(spacing))
    lumbar_zones = pedicle_zones_horizontal(spine, ct, body, bodies, names, spacing, PEDICLE_DEPTH_LUMBAR_MM)
    tilted_zones = pedicle_zones_tilted(spine, ct, body, bodies, names, spacing, PEDICLE_DEPTH_THORACIC_MM)
    zones = {level: (tilted_zones[level] if LABEL_VALUE["T12"] <= level <= LABEL_VALUE["T1"] else lumbar_zones[level])
             for level in lumbar_zones if level in tilted_zones}
    carried = {}                       # body label -> (label its pedicles carry, share)
    for level, zone in zones.items():
        counts = np.bincount(labels[zone], minlength=25)[1:]
        if counts.sum():
            top = int(counts.argmax()) + 1
            carried[level] = (top, counts[top - 1] / counts.sum())
    mismatched = {level: top for level, (top, share) in carried.items() if top != level and share >= PEDICLE_MIN_SHARE}
    if mismatched:
        runs, run = [], []
        for level in sorted(mismatched):
            if run and level != run[-1] + 1:
                runs.append(run)
                run = []
            run.append(level)
        runs.append(run)
        confirmed = {}
        for run in runs:
            if any(model_label.get(level, level) != level for level in run):
                confirmed.update({level: mismatched[level] for level in run})
            else:
                log.append(f"Step 2 (arches): at {LABEL_NAME[run[-1]]}-{LABEL_NAME[run[0]]} the bone joining arch and body "
                           f"(the pedicles) carries a neighbouring label, but the model labelled the bodies there "
                           f"correctly, so this is normal overlap between neighbours; nothing changed")
        mismatched = confirmed
    if not mismatched:
        log.append("Step 2 (arches): every arch carries the same label as the body it is attached to; nothing changed")
        return labels, set()
    log.append("Step 2 (arches): these bodies have an arch labelled as a different vertebra, as the label of their "
               "pedicles shows: " + "; ".join(
                   f"body {LABEL_NAME[level]}, whose pedicles are {carried[level][1]:.0%} labelled {LABEL_NAME[top]}"
                   for level, top in sorted(mismatched.items(), reverse=True)))
    corrected = labels.copy()
    arch = spine & (body == 0)
    for arch_label in sorted(set(mismatched.values()), reverse=True):
        claimants = [level for level, (top, share) in carried.items() if top == arch_label and share >= PEDICLE_MIN_SHARE]
        selected = arch & (labels == arch_label)
        if len(claimants) == 1:
            corrected[selected] = claimants[0]
            log.append(f"    the arch bone labelled {LABEL_NAME[arch_label]} ({selected.sum() * voxel_mm3 / 1000:.2f} cm3): "
                       f"{LABEL_NAME[arch_label]} -> {LABEL_NAME[claimants[0]]}, the body it is attached to")
            continue
        points = np.argwhere(selected)
        box = tuple(slice(a, b) for a, b in zip(points.min(0), points.max(0) + 1))
        markers = np.zeros(selected[box].shape, np.int32)
        for level in claimants:
            markers[zones[level][box] & selected[box]] = level
        owner = np.where(selected[box], watershed(ct[box], markers, mask=selected[box]), 0).astype(np.uint8)
        target = corrected[box]
        reached = selected[box] & (owner > 0)
        target[reached] = owner[reached]
        log.append(f"    the arch bone labelled {LABEL_NAME[arch_label]} belongs to two vertebrae; it is divided between "
                   "them along the dense outer layer of bone where the two arches meet: " + ", ".join(
                       f"{(reached & (owner == level)).sum() * voxel_mm3 / 1000:.2f} cm3 to {LABEL_NAME[level]}"
                       for level in claimants))
    return corrected, set(mismatched)


# =====================================================================================================
# Step 3: thoracic spinous processes
# =====================================================================================================
def correct_thoracic_spinous(region_mask, ct, labels, spacing, corrected_levels, log):
    """Give each thoracic spinous process the label of its root.

    The processes are isolated with `split_bone_by_ct_gaps` in a narrow midline band. A process's root is its
    front SPINOUS_ROOT_MM, where it leaves the spinal canal; its label must be clear (ROOT_MIN_SHARE). Two
    corrections are made on the part of the process below and behind the root (a thoracic process points down
    and back):
      - bone labelled as the vertebra one level up becomes the root's label (a common split of the model's);
      - at levels whose arches step 2 corrected, bone labelled as either neighbouring vertebra becomes the root's
        label.
    The relabelled tip may spread up to SPINOUS_TIP_WIDTH_MM from the midline, but never into another process.

    Args:
        region_mask: boolean RAS volume of the processed range (including the level above it, whose label the
            top process's tip may carry).
        ct, labels: CT (HU) and current labels, same shape.
        corrected_levels: label values whose arches were corrected in step 2.
        log: list that receives one line per correction.

    Returns:
        The corrected label volume.
    """
    voxel_mm3 = float(np.prod(spacing))
    xs = np.arange(region_mask.shape[0])[:, None, None]
    ys = np.arange(region_mask.shape[1])[None, :, None]
    zs = np.arange(region_mask.shape[2])[None, None, :]
    x_mid = midline_x(region_mask, spacing)[None, None, :]
    y_back = np.where(region_mask, ys, 10 ** 6).min(axis=(0, 1)).astype(float)
    y_front = np.where(region_mask, ys, -1).max(axis=(0, 1)).astype(float)
    in_back = (ys - y_back[None, None, :]) <= 0.45 * (y_front - y_back)[None, None, :]
    # the spine's local direction (front-back slope per slice), taken from the bodies at the front of the spine
    front_part = region_mask & ((ys - y_back[None, None, :]) >= 0.55 * (y_front - y_back)[None, None, :])
    front_z = np.nonzero(front_part.any(axis=(0, 1)))[0]
    front_y = np.array([np.argwhere(front_part[:, :, z])[:, 1].mean() for z in front_z]) * spacing[1]
    front_y = ndimage.gaussian_filter1d(
        ndimage.median_filter(front_y, size=max(3, int(round(15 / spacing[2])) | 1), mode="nearest"),
        sigma=10 / spacing[2], mode="nearest")
    slope = np.interp(np.arange(region_mask.shape[2]), front_z, np.gradient(front_y, front_z * spacing[2]))
    band = (np.abs(xs - x_mid) * spacing[0] <= SPINOUS_BAND_MM) & in_back & region_mask
    bars = split_bone_by_ct_gaps(band, ct, spacing)
    distance_to_bar, nearest_index = ndimage.distance_transform_edt(bars == 0, sampling=spacing, return_indices=True)
    nearest_bar = bars[tuple(nearest_index)]
    roots, root_top = {}, {}
    for bar_id in np.unique(bars[bars > 0]):
        bar = bars == bar_id
        points = np.argwhere(bar)
        front = points[points[:, 1] >= points[:, 1].max() - 5.0 / spacing[1]]
        root_top[bar_id] = front[np.argmax(front[:, 2])]          # where the process leaves the canal
        root = bar & (ys >= points[:, 1].max() - SPINOUS_ROOT_MM / spacing[1])
        counts = np.bincount(labels[root], minlength=25)[1:]
        if counts.sum():
            level = int(counts.argmax()) + 1
            roots[bar_id] = (level, counts[level - 1] / counts.sum(), root)
    corrected = labels.copy()
    n_corrected = 0
    tip_band = (np.abs(xs - x_mid) * spacing[0] <= SPINOUS_TIP_WIDTH_MM) & in_back & region_mask
    for bar_id, (level, _, root) in sorted(roots.items(), key=lambda item: -item[1][0]):
        if not LABEL_VALUE["T12"] <= level <= LABEL_VALUE["T1"] or level + 1 > 24:
            continue
        counts = np.bincount(labels[root], minlength=25)[1:]
        counts[level] = 0                         # the level above may reach into the root: not counted
        if counts[level - 1] / max(counts.sum(), 1) < ROOT_MIN_SHARE:
            continue
        bar = bars == bar_id
        top = root_top[bar_id]
        up = np.array([slope[top[2]], 1.0])
        up /= np.linalg.norm(up)                  # up along the spine (y, z)
        back = np.array([-up[1], up[0]]) if -up[1] < 0 else np.array([up[1], -up[0]])
        dy_mm, dz_mm = (ys - top[1]) * spacing[1], (zs - top[2]) * spacing[2]
        below_and_behind = (dy_mm * up[0] + dz_mm * up[1] < 0) & (dy_mm * back[0] + dz_mm * back[1] > 0)
        whole_process = level in corrected_levels
        wrong_labels = [q for q in (level - 1, level + 1) if 1 <= q <= 24] if whole_process else [level + 1]
        if whole_process:
            wrong = bar & np.isin(labels, wrong_labels)
        else:
            clear_upper = [b for b, (lv, share, _) in roots.items() if lv == level + 1 and share >= ROOT_MIN_SHARE]
            near_bar = ndimage.distance_transform_edt(~bar, sampling=spacing) <= 3.0
            nearby = near_bar & (bars > 0) & (bars != bar_id) & ~np.isin(bars, clear_upper)
            wrong = (bar | nearby) & (labels == level + 1)
        move = wrong & below_and_behind
        if not move.any() or move.sum() * voxel_mm3 < SPINOUS_MIN_MOVE_CM3 * 1000:
            continue
        corrected[move] = level
        keep_out = (nearest_bar > 0) & (nearest_bar != bar_id) & (distance_to_bar <= SPINOUS_KEEPOUT_MM)
        candidates = (tip_band & np.isin(labels, wrong_labels) & (corrected == labels) & below_and_behind
                      & ~((bars > 0) & (bars != bar_id)) & ~keep_out)
        groups = cc3d.connected_components(candidates | move, connectivity=6)
        touched = np.unique(groups[move])
        spread = np.isin(groups, touched[touched > 0]) & candidates
        corrected[spread] = level
        n_corrected += 1
        reason = ("this level's arch was corrected in step 2, so the whole process follows its root" if whole_process
                  else "the shaft and tip carried the label of the vertebra above")
        log.append(f"    spinous process of {LABEL_NAME[level]}: {(move.sum() + spread.sum()) * voxel_mm3 / 1000:.2f} cm3 "
                   f"{'/'.join(LABEL_NAME[q] for q in wrong_labels)} -> {LABEL_NAME[level]} ({reason})")
    head = ("Step 3 (thoracic spinous processes): each process was given the label of its root, where it leaves the "
            "spinal canal; " + (f"{n_corrected} processes corrected:" if n_corrected else "none needed a correction"))
    log.insert(len(log) - n_corrected, head)
    return corrected


# =====================================================================================================
# Step 4: lumbar spinous processes
# =====================================================================================================
def spine_curve(centres_mm: np.ndarray):
    """Smooth curve through the body centres (mm, ordered bottom to top), extended 80 mm past both ends.

    Returns:
        (centre_positions, position_of, samples): the arc length (mm) of each body centre along the curve, a
        function mapping points (N x 3, mm) to the arc length of their nearest curve point, and the curve as
        points sampled every 0.25 mm.
    """
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(centres_mm, axis=0), axis=1))])
    spline = CubicSpline(t, centres_mm, bc_type="natural")
    middle = spline(np.arange(0, t[-1] + 1e-9, 0.25))
    start_dir = spline(0, 1) / np.linalg.norm(spline(0, 1))
    end_dir = spline(t[-1], 1) / np.linalg.norm(spline(t[-1], 1))
    extension = np.arange(0.25, 80.01, 0.25)[:, None]
    samples = np.vstack([(centres_mm[0] - extension * start_dir)[::-1], middle, centres_mm[-1] + extension * end_dir])
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))])
    tree = cKDTree(samples)
    centre_positions = arc[len(extension) + np.round(t / 0.25).astype(int)]
    return centre_positions, (lambda points: arc[tree.query(points)[1]]), samples


def curve_bend_radius(samples: np.ndarray) -> np.ndarray:
    """Radius of curvature (mm) of the sampled curve at every sample."""
    d1 = np.gradient(samples, axis=0)
    d2 = np.gradient(d1, axis=0)
    curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
    return np.where(curvature > 0, 1 / np.maximum(curvature, 1e-12), np.inf)


def correct_lumbar_spinous(spine, ct, labels, body, bodies, names, spacing, log):
    """Give each lumbar spinous process the level of the slab it lies in along the spine.

    Lumbar processes point straight back and sit a little below their own body. Behind the spinal canal, the
    midline band is cut into slabs along the spine's curve, one per lumbar level, with each boundary halfway
    between two body centres moved down by LUMBAR_PROCESS_OFFSET_MM and then snapped to the nearby position that
    crosses the least bone (the gap between two processes). Each connected piece of bone takes the level of its
    centre. Slabs are left as the model labelled them where the model already agrees (MODEL_TRUST_SHARE), and
    where the curve bends so sharply that positions along it become unreliable.

    Returns:
        The corrected label volume.
    """
    nx, ny, nz = spine.shape
    order = sorted(range(len(bodies)), key=lambda i: names[i + 1])           # lowest level (L5) first
    levels = [names[i + 1] for i in order]
    centre_z = np.array([np.argwhere(bodies[i][0])[:, 2].mean() for i in order]) * spacing[2]
    centre_x = np.array([np.argwhere(bodies[i][0])[:, 0].mean() for i in order])
    x_mid = np.interp(np.arange(nz), centre_z / spacing[2], centre_x)
    xs = np.arange(nx)[:, None, None]
    ys = np.arange(ny)[None, :, None]
    band = np.abs(xs - x_mid[None, None, :]) * spacing[0] <= LUMBAR_BAND_MM
    back_wall = np.where((body > 0) & band, ys, 10 ** 6).min(axis=(0, 1)).astype(float)
    has_wall = back_wall < 10 ** 6
    back_wall = np.interp(np.arange(nz), np.nonzero(has_wall)[0], back_wall[has_wall])
    behind_canal = ys < (back_wall[None, None, :] - CANAL_MIN_MM / spacing[1])
    lumbar = [i for i, level in enumerate(levels) if level <= LABEL_VALUE["L1"]]
    if not lumbar:
        log.append("Step 4 (lumbar spinous processes): no lumbar vertebrae in this scan; nothing to do")
        return labels
    centres_mm = np.array([np.argwhere(bodies[i][0]).mean(0) * spacing for i in order])
    centre_positions, position_of, curve = spine_curve(centres_mm)
    points = np.argwhere(spine & (body == 0) & band & behind_canal)
    positions = position_of(points * spacing)
    boundaries = [(centre_positions[i] + centre_positions[i + 1]) / 2 - LUMBAR_PROCESS_OFFSET_MM
                  for i in range(len(centre_positions) - 1)]
    smoothed = ndimage.gaussian_filter(ct, sigma=SMOOTHING_MM / spacing)
    is_bone = smoothed[tuple(points.T)] >= BONE_HU
    bone_positions = positions[is_bone]
    snapped = []
    for boundary in boundaries:
        grid = np.arange(boundary - CUT_SNAP_MM, boundary + CUT_SNAP_MM + 0.01, 0.5)
        bone_crossed = np.array([np.sum(np.abs(bone_positions - g) <= 0.75) for g in grid])
        least = grid[bone_crossed == bone_crossed.min()]
        snapped.append(float(least[np.argmin(np.abs(least - boundary))]))
    lower_edge, upper_edge = [-np.inf] + snapped, snapped + [np.inf]
    assigned = np.zeros(len(points), np.uint8)
    for i in lumbar:
        assigned[(positions >= lower_edge[i]) & (positions < upper_edge[i])] = levels[i]
    in_lumbar = assigned > 0
    bone_volume = np.zeros(spine.shape, bool)
    bone_volume[tuple(points.T)] = is_bone
    piece_of = cc3d.connected_components(bone_volume, connectivity=6)[tuple(points.T)]
    for piece in range(1, piece_of.max() + 1):
        selected = piece_of == piece
        if not (selected & in_lumbar).any() or np.ptp(positions[selected]) > PROCESS_MAX_LENGTH_MM:
            continue
        centre = np.median(positions[selected])
        level = 0
        for i in lumbar:
            if lower_edge[i] <= centre < upper_edge[i]:
                level = levels[i]
        if level:
            assigned[selected] = level
    model_here = labels[tuple(points.T)]
    trusted = []
    for i in lumbar:
        in_slab = assigned == levels[i]
        if in_slab.any() and (model_here[in_slab] == levels[i]).mean() >= MODEL_TRUST_SHARE:
            assigned[in_slab] = 0
            trusted.append(LABEL_NAME[levels[i]])
    radius = curve_bend_radius(curve)
    curve_arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    distance_to_curve = cKDTree(curve).query(points * spacing)[0]
    unreliable = set()
    cuts = [(i, upper_edge[i]) for i in lumbar if np.isfinite(upper_edge[i])]
    cuts += [(lumbar[0] - 1, lower_edge[lumbar[0]])] if np.isfinite(lower_edge[lumbar[0]]) else []
    for i, cut in cuts:
        near_cut = np.abs(curve_arc - cut) <= CURVE_FRAME_MARGIN_MM
        local_radius = float(radius[near_cut].min()) if near_cut.any() else np.inf
        nearby_bone = (np.abs(positions - cut) <= CURVE_FRAME_MARGIN_MM) & is_bone
        farthest = float(np.percentile(distance_to_curve[nearby_bone], 95)) if nearby_bone.any() else 0.0
        if not farthest < local_radius:          # bone lies beyond the bend's centre: positions fold over
            unreliable.update(q for q in (i, i + 1) if q in lumbar)
    for i in sorted(unreliable):
        assigned[assigned == levels[i]] = 0
    changed = assigned > 0
    corrected = labels.copy()
    corrected[tuple(points[changed].T)] = assigned[changed]
    # a slab already trusted above was kept for that reason; list only the slabs the bend check added
    curved = [LABEL_NAME[levels[i]] for i in sorted(unreliable) if LABEL_NAME[levels[i]] not in trusted]
    kept = []
    if trusted:
        kept.append(f"{', '.join(trusted)}, where the model's labels already match the position")
    if curved:
        kept.append(f"{', '.join(curved)}, where the spine curves too sharply to judge the position reliably")
    log.append("Step 4 (lumbar spinous processes): each process was given to a vertebra by its position along the spine "
               f"(a lumbar process sits just below its own body). Changed (old -> new label): "
               f"{describe_changes(labels, corrected, spacing)}. Processes left as the model labelled them: "
               f"{'; and '.join(kept) if kept else 'none'}")
    return corrected


# =====================================================================================================
# Step 5: clean-up (all 24 levels)
# =====================================================================================================
def label_boxes(labels: np.ndarray) -> dict:
    """Bounding box (a tuple of slices) of every label value present, found in one pass over the volume.

    The clean-up works on each label inside its own box: a label's pieces never leave it, and the voxel order
    inside the box is the same as in the whole volume, so the pieces are found and numbered exactly as they would
    be in the whole volume, at a fraction of the cost.
    """
    return {value: box for value, box in enumerate(ndimage.find_objects(labels), start=1)
            if box is not None and value <= 24}


def remove_far_pieces(labels: np.ndarray, spacing: np.ndarray, log: list) -> np.ndarray:
    """Remove pieces of a label that lie more than FAR_PIECE_MM from that label's largest piece.

    Such pieces are false detections elsewhere in the body (e.g. in a hip or shoulder) and are set to background.
    """
    cleaned = labels.copy()
    voxel_mm3 = float(np.prod(spacing))
    removed = []
    for value, box in label_boxes(labels).items():
        offset = np.array([s.start for s in box])
        pieces, n_pieces = cc3d.connected_components(labels[box] == value, connectivity=26, return_N=True)
        if n_pieces < 2:
            continue
        main = int(np.bincount(pieces.ravel())[1:].argmax()) + 1
        tree = cKDTree((np.argwhere(pieces == main) + offset) * spacing)
        for piece_id, piece_slice in enumerate(ndimage.find_objects(pieces), start=1):
            if piece_id == main or piece_slice is None:
                continue
            points = np.argwhere(pieces[piece_slice] == piece_id) + np.array([s.start for s in piece_slice]) + offset
            gap = float(tree.query(points * spacing)[0].min())
            if gap > FAR_PIECE_MM:
                cleaned[tuple(points.T)] = 0
                removed.append((len(points) * voxel_mm3, LABEL_NAME[value], gap))
    if removed:
        removed.sort(reverse=True)
        log.append(f"Step 5 (clean-up): removed {len(removed)} stray pieces ({sum(r[0] for r in removed):.0f} mm3 in all) "
                   f"lying more than {FAR_PIECE_MM:.0f} mm from the vertebra whose label they carry; they are false "
                   "detections elsewhere in the body: " + "; ".join(f"{name} {size:.0f} mm3, {gap:.0f} mm away"
                                                          for size, name, gap in removed))
    else:
        log.append("Step 5 (clean-up): no stray pieces far from their vertebra")
    return cleaned


def largest_pieces(labels: np.ndarray):
    """Split every label into its largest piece and its other (detached) pieces.

    Returns:
        (main, detached): a uint8 volume holding only each label's largest piece, and a list of
        (label value, N x 3 voxel indices) for every other piece. C1 and C2 are not listed (PIECE_SKIP_LEVELS).
    """
    main = np.zeros(labels.shape, np.uint8)
    detached = []
    for value, box in label_boxes(labels).items():
        offset = np.array([s.start for s in box])
        pieces, n_pieces = cc3d.connected_components(labels[box] == value, connectivity=26, return_N=True)
        if n_pieces == 0:
            continue
        largest = int(np.bincount(pieces.ravel())[1:].argmax()) + 1
        main[box][pieces == largest] = value
        if LABEL_NAME[value] in PIECE_SKIP_LEVELS:
            continue
        for piece_id, piece_slice in enumerate(ndimage.find_objects(pieces), start=1):
            if piece_id != largest and piece_slice is not None:
                detached.append((value, np.argwhere(pieces[piece_slice] == piece_id)
                                 + np.array([s.start for s in piece_slice]) + offset))
    return main, detached


def spinous_root_of_piece(labels, ct, spacing, points, own_value):
    """Label of the spinous process a midline piece belongs to, read at that process's root.

    The bone around the piece is split into processes with `split_bone_by_ct_gaps`; the process holding the piece
    is found, and its root (front SPINOUS_ROOT_MM) is read, ignoring the piece's own label.

    Returns:
        (label value, share) or (None, 0.0) when it cannot be decided.
    """
    centre = points.mean(0)
    half_x, half_z = int(np.ceil(30 / spacing[0])), int(np.ceil(45 / spacing[2]))
    crop = (slice(max(int(centre[0]) - half_x, 0), int(centre[0]) + half_x), slice(0, labels.shape[1]),
            slice(max(int(centre[2]) - half_z, 0), int(centre[2]) + half_z))
    labels_crop = labels[crop]
    band = midline_back_band(labels_crop > 0, spacing, PIECE_MIDLINE_MM)
    if not band.any():
        return None, 0.0
    bars = split_bone_by_ct_gaps(band, ct[crop].astype(np.float32), spacing)
    local = points - np.array([s.start for s in crop])
    local = local[np.all((local >= 0) & (local < np.array(bars.shape)), axis=1)]
    ids = bars[tuple(local.T)]
    ids = ids[ids > 0]
    if not len(ids):
        return None, 0.0
    bar_points = np.argwhere(bars == int(np.bincount(ids).argmax()))
    root = bar_points[bar_points[:, 1] >= bar_points[:, 1].max() - SPINOUS_ROOT_MM / spacing[1]]
    counts = np.bincount(labels_crop[tuple(root.T)], minlength=25)[1:]
    counts[own_value - 1] = 0
    if counts.sum() == 0:
        return None, 0.0
    value = int(counts.argmax()) + 1
    return value, counts[value - 1] / counts.sum()


def transverse_root_of_piece(main, ct, spacing, points, x_mid, y_range):
    """Label of the transverse process a side piece belongs to, read at that process's inner end.

    The bone to the side of the spine (TRANSVERSE_BAND_MM from the midline, back half of the spine) is split
    into processes with `split_bone_by_ct_gaps`; the process holding the piece is found, and its inner end
    (TRANSVERSE_ROOT_MM, next to the vertebra) is read, counting only the vertebrae's largest pieces.

    Returns:
        (label value, share) or (None, 0.0) when it cannot be decided.
    """
    centre = points.mean(0)
    half_x, half_z = int(np.ceil(50 / spacing[0])), int(np.ceil(40 / spacing[2]))
    crop = (slice(max(int(centre[0]) - half_x, 0), int(centre[0]) + half_x), slice(0, main.shape[1]),
            slice(max(int(centre[2]) - half_z, 0), int(centre[2]) + half_z))
    origin = np.array([s.start for s in crop])
    spine_crop = main[crop] > 0
    shape = spine_crop.shape
    xs = np.arange(shape[0])[:, None, None] + origin[0]
    zs = np.arange(shape[2]) + origin[2]
    offset_mm = (xs - x_mid[zs][None, None, :]) * spacing[0]
    side = 1 if centre[0] > x_mid[int(round(centre[2]))] else -1
    y_back, y_front = y_range[0][zs], y_range[1][zs]
    ys = np.arange(shape[1])[None, :, None]
    in_back = (ys - y_back[None, None, :]) <= PIECE_BACK_FRACTION * (y_front - y_back)[None, None, :]
    piece = np.zeros(shape, bool)
    local = points - origin
    local = local[np.all((local >= 0) & (local < np.array(shape)), axis=1)]
    piece[tuple(local.T)] = True
    band = ((spine_crop | piece) & in_back
            & (side * offset_mm >= TRANSVERSE_BAND_MM[0]) & (side * offset_mm <= TRANSVERSE_BAND_MM[1]))
    if not band[piece].any():
        return None, 0.0
    bars = split_bone_by_ct_gaps(band, ct[crop].astype(np.float32), spacing)
    ids = bars[piece & band]
    ids = ids[ids > 0]
    if not len(ids):
        return None, 0.0
    bar_points = np.argwhere(bars == int(np.bincount(ids).argmax()))
    distance_out = side * offset_mm[bar_points[:, 0], 0, bar_points[:, 2]]
    root = bar_points[distance_out <= distance_out.min() + TRANSVERSE_ROOT_MM]
    root = root[~piece[tuple(root.T)]]
    counts = np.bincount(main[crop][tuple(root.T)], minlength=25)[1:]
    if counts.sum() == 0:
        return None, 0.0
    value = int(counts.argmax()) + 1
    return value, counts[value - 1] / counts.sum()


def reassign_detached_pieces(labels: np.ndarray, spacing: np.ndarray, ct: np.ndarray, log: list) -> np.ndarray:
    """Give every detached piece of a label to the vertebra it belongs to.

    For each piece other than a label's largest one, in this order:
      1. if it touches the largest piece of exactly one other vertebra (and not its own), it takes that label;
      2. if it lies on the midline in the back of the spine (a spinous process), it takes the label of that
         process's root (`spinous_root_of_piece`);
      3. if it lies to the side in the back of the spine (a transverse process), it takes the label of that
         process's root (`transverse_root_of_piece`);
      4. otherwise it takes the label of the vertebra it shares the most surface with.
    Rules 2 and 3 apply only when the root's label is clear (ROOT_MIN_SHARE); otherwise rule 4 decides. A piece
    touching nothing stays as it is. The passes repeat against the updated labels until nothing changes, so a
    piece next to a piece that just moved is judged by its new label.
    """
    result = labels.copy()
    voxel_mm3 = float(np.prod(spacing))
    spine_points = np.argwhere(labels > 0)
    slices = np.unique(spine_points[:, 2])
    x_per_slice = np.array([np.median(spine_points[spine_points[:, 2] == z, 0]) for z in slices])
    x_mid = np.interp(np.arange(labels.shape[2]), slices,
                      ndimage.median_filter(x_per_slice, size=max(3, int(round(15 / spacing[2])) | 1)))
    ys = np.arange(labels.shape[1])[None, :, None]
    spine = labels > 0
    y_back = np.where(spine, ys, 10 ** 6).min(axis=(0, 1)).astype(float)
    y_front = np.where(spine, ys, -1).max(axis=(0, 1)).astype(float)
    moved = {}
    for _ in range(PIECE_MAX_ROUNDS):
        main, detached = largest_pieces(result)
        n_moved = 0
        for value, points in detached:
            box = tuple(slice(max(points[:, i].min() - 1, 0), min(points[:, i].max() + 2, labels.shape[i]))
                        for i in range(3))     # one voxel of margin, kept inside the scan
            piece = np.zeros(tuple(b.stop - b.start for b in box), bool)
            piece[tuple((points - np.array([b.start for b in box])).T)] = True
            surface = ndimage.binary_dilation(piece) & ~piece
            touching = np.bincount(main[box][surface], minlength=25)[1:]
            other_vertebrae = [q + 1 for q in np.nonzero(touching)[0] if q + 1 != value]
            centre = points.mean(0)
            z = int(round(centre[2]))
            off_midline_mm = abs(centre[0] - x_mid[z]) * spacing[0]
            in_back = (centre[1] - y_back[z]) <= PIECE_BACK_FRACTION * (y_front[z] - y_back[z])
            new_value = None
            if len(other_vertebrae) == 1 and touching[value - 1] == 0:
                new_value = other_vertebrae[0]
            elif off_midline_mm <= PIECE_MIDLINE_MM and in_back:
                root_value, share = spinous_root_of_piece(result, ct, spacing, points, value)
                if root_value is not None and share >= ROOT_MIN_SHARE:
                    new_value = root_value
            elif in_back:
                root_value, share = transverse_root_of_piece(main, ct, spacing, points, x_mid, (y_back, y_front))
                if root_value is not None and share >= ROOT_MIN_SHARE:
                    new_value = root_value
            if new_value is None:
                contact = {q + 1: int(touching[q]) for q in np.nonzero(touching)[0]}
                if not contact:
                    continue
                new_value = max(contact, key=contact.get)
            if new_value != value:
                result[tuple(points.T)] = new_value
                moved[(value, new_value)] = moved.get((value, new_value), 0) + len(points) * voxel_mm3
                n_moved += 1
        if n_moved == 0:
            break
    total = sum(moved.values()) / 1000
    log.append("Step 5 (clean-up): small pieces detached from their vertebra were given to the vertebra they belong to "
               "(the one they touch, or the root of the process they lie in): " + (
                   f"{total:.2f} cm3 in all (old -> new label): " + describe_pairs({k: v / 1000 for k, v in moved.items()})
                   if moved else "none needed"))
    return result


def fill_bone_holes(labels: np.ndarray, spacing: np.ndarray, ct: np.ndarray, log: list) -> np.ndarray:
    """Fill holes fully enclosed by one vertebra, when the hole has bone density.

    An enclosed group of background voxels with a mean CT value of at least BONE_HU becomes part of the
    vertebra around it. A darker enclosed gap is usually the joint space between two vertebrae, which the mask
    happens to wrap around, and is left open. Voxels of other labels are never changed.
    """
    filled_labels = labels.copy()
    voxel_mm3 = float(np.prod(spacing))
    filled, left_open = {}, 0
    for value in range(1, 25):
        mask = filled_labels == value
        if not mask.any():
            continue
        points = np.argwhere(mask)
        box = tuple(slice(a, b) for a, b in zip(points.min(0), points.max(0) + 1))
        sub = mask[box]
        holes = ndimage.binary_fill_holes(sub) & ~sub & (filled_labels[box] == 0)
        if not holes.any():
            continue
        groups, n_groups = cc3d.connected_components(holes, connectivity=6, return_N=True)
        ct_box = ct[box]
        for group_id in range(1, n_groups + 1):
            group = groups == group_id
            if float(ct_box[group].mean()) >= BONE_HU:
                filled_labels[box][group] = value
                filled[LABEL_NAME[value]] = filled.get(LABEL_NAME[value], 0) + group.sum() * voxel_mm3
            else:
                left_open += 1
    log.append(f"Step 5 (clean-up): holes enclosed by a vertebra's mask that have bone density ({BONE_HU:.0f} HU or "
               "more) were filled: " + (", ".join(f"{v:.0f} mm3 in {name}" for name, v in filled.items()) or "none")
               + (f"; {left_open} enclosed gap{'s' if left_open > 1 else ''} darker than bone "
                  f"{'were' if left_open > 1 else 'was'} left open (joint space{'s' if left_open > 1 else ''} between "
                  "vertebrae, not missing bone)" if left_open else ""))
    return filled_labels


# =====================================================================================================
# The whole correction for one case
# =====================================================================================================
def refine_labels(labels_path: str, ct_path: str):
    """Run steps 1-5 on one case.

    Args:
        labels_path: the case's combined_labels.nii.gz.
        ct_path: the case's ct.nii.gz (same grid as the labels).

    Returns:
        (refined, log, status): the corrected label array in the voxel order of the input file (uint8), the list
        of log lines describing each correction, and a short status: "refined", "empty", or "steps 1-4 skipped
        (...)" when the safety checks kept the model's labels (see `check_bodies`).
    """
    if not os.path.isfile(ct_path):
        raise FileNotFoundError(f"CT not found: {ct_path} (the correction needs the case's CT)")
    labels_image, orientation, labels, spacing = load_as_ras(labels_path)
    ct_image, _, ct, _ = load_as_ras(ct_path)
    if ct_image.shape[:3] != labels_image.shape[:3] or not np.allclose(ct_image.affine, labels_image.affine,
                                                                        atol=GRID_TOLERANCE_MM):
        what = (f"shape {ct_image.shape[:3]} vs {labels_image.shape[:3]}" if ct_image.shape[:3] != labels_image.shape[:3]
                else f"same shape, but the position or orientation differs by up to "
                     f"{np.abs(ct_image.affine - labels_image.affine).max():.2f} mm")
        raise ValueError(f"the CT {ct_path} is not on the same voxel grid as the labels {labels_path} ({what}); "
                         "the predictions must be on the CT's own grid")
    labels = labels.astype(np.uint8)
    log = []
    top_value = LABEL_VALUE[TOP_LEVEL]

    # the spine: the largest connected labelled structure; steps 1-4 work in a box around its L5..TOP part
    if not (labels > 0).any():
        log.append("No vertebra is labelled in this case; the empty masks are written unchanged")
        return restore_orientation(labels, orientation).astype(np.uint8), log, "empty"
    components = cc3d.connected_components(labels > 0, connectivity=26)
    spine_all = components == np.bincount(components.ravel())[1:].argmax() + 1
    in_range = spine_all & (labels >= 1) & (labels <= top_value)
    if not in_range.any():
        log.append(f"Steps 1-4 skipped: the scan contains none of the vertebrae L5-{TOP_LEVEL} that they correct")
        return clean_up_only(labels, orientation, spacing, ct, log, "steps 1-4 skipped (range not in scan)")
    points = np.argwhere(in_range)
    margin = np.ceil(10 / spacing).astype(int)
    lo = np.maximum(points.min(0) - margin, 0)
    hi = np.minimum(points.max(0) + margin + 1, labels.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    spine, box_labels, box_ct = in_range[box], labels[box], ct[box].astype(np.float32)

    # step 1
    bodies = find_vertebral_bodies(spine, spacing)
    if len(bodies) < MIN_BODIES:                # checked here too: naming and region growing need bodies
        log.append(f"Steps 1-4 skipped, the model's labels are kept: only {len(bodies)} vertebral bodies were found "
                   f"(at least {MIN_BODIES} are needed). Only the clean-up (step 5) is applied")
        return clean_up_only(labels, orientation, spacing, ct, log, "steps 1-4 skipped (bodies not reliable)",
                             labels_unreliable=True)
    region = assign_regions_to_bodies(spine, box_ct, bodies, spacing)
    above = spine_all & (labels == top_value + 1)
    level_above_in_scan = above.sum() * float(np.prod(spacing)) / 1000 >= LEVEL_MIN_CM3
    names, agreement, model_label = name_bodies(
        box_labels, bodies, top_value if ANCHOR_TO_LEVEL_ABOVE and level_above_in_scan else None)
    scan_z_edges = (0 if lo[2] == 0 else None, hi[2] - 1 - lo[2] if hi[2] == labels.shape[2] else None)
    problem = check_bodies(bodies, names, model_label, box_labels, spacing, spine, top_value, scan_z_edges,
                           level_above_in_scan)
    if problem is not None:
        log.append(f"Steps 1-4 skipped, the model's labels are kept: {problem}. Only the clean-up (step 5) is applied")
        return clean_up_only(labels, orientation, spacing, ct, log, "steps 1-4 skipped (bodies not reliable)",
                             labels_unreliable=True)
    # a vertebra the scan shows only in part, with no body found for it, keeps the model's labels in steps 1-4
    # (otherwise its bone would be given to the neighbouring body)
    partial = sorted(levels_cut_off(box_labels, spine, top_value, scan_z_edges) - set(names.values()))
    if partial:
        spine = spine & ~np.isin(box_labels, partial)
        log.append("Steps 1-4 leave " + ", ".join(LABEL_NAME[v] for v in partial) + " as the model labelled "
                   + ("it: the scan cuts it off and no complete body was found for it" if len(partial) == 1 else
                      "them: the scan cuts them off and no complete body was found for them"))
    body = body_masks(spine, region, bodies, names, spacing)
    working = box_labels.copy()
    working[body > 0] = body[body > 0]
    if agreement >= 0.98:
        what = (f"the model had numbered the bodies correctly ({agreement:.0%} of each body's centre already carried "
                "the right label); only the edges between neighbouring bodies, at the discs, were redrawn")
    else:
        what = (f"only {agreement:.0%} of the bodies' centres carried the right label, so bodies the model had split "
                "or numbered one level off were relabelled")
    how = (f"numbered them down from {LABEL_NAME[top_value + 1]}, the vertebra just above, which these steps leave "
           "unchanged" if ANCHOR_TO_LEVEL_ABOVE and level_above_in_scan
           else "numbered them as the consecutive sequence that most of the model's body labels agree with")
    log.append(f"Step 1 (vertebral bodies): found {len(bodies)} bodies for the {top_value} levels L5-{TOP_LEVEL} and "
               f"{how}; {what}. Changed (old -> new label): {describe_changes(box_labels, working, spacing)}")
    # step 2
    working, corrected_levels = correct_arches(spine, box_ct, working, body, bodies, names, spacing, model_label, log)
    # step 3 also sees the level above the range, whose label the top process's tip may carry
    spinous_region = spine | (spine_all[box] & (box_labels == top_value + 1))
    working = correct_thoracic_spinous(spinous_region, box_ct, working, spacing, corrected_levels, log)
    # step 4
    working = correct_lumbar_spinous(spine, box_ct, working, body, bodies, names, spacing, log)
    refined = labels.copy()
    refined[box] = np.where(spine | spinous_region, working, box_labels)
    # step 5
    refined = remove_far_pieces(refined, spacing, log)
    refined = reassign_detached_pieces(refined, spacing, ct, log)
    refined = fill_bone_holes(refined, spacing, ct, log)
    return restore_orientation(refined, orientation).astype(np.uint8), log, "refined"


def clean_up_only(labels, orientation, spacing, ct, log, status, labels_unreliable=False):
    """Apply only step 5 (clean-up) to the model's labels; used when steps 1-4 cannot be applied safely.

    When the safety checks rejected the case (`labels_unreliable`), the detached-piece reassignment is left out
    unless FALLBACK_REASSIGN_PIECES is set: it decides which piece is a vertebra's main part from the model's labels,
    which the failed checks have just shown cannot be trusted here.
    """
    refined = remove_far_pieces(labels.copy(), spacing, log)
    if not labels_unreliable or FALLBACK_REASSIGN_PIECES:
        refined = reassign_detached_pieces(refined, spacing, ct, log)
    else:
        log.append("Step 5 (clean-up): detached pieces are left with the model's labels, since the checks above "
                   "found the model's numbering unreliable")
    refined = fill_bone_holes(refined, spacing, ct, log)
    return restore_orientation(refined, orientation).astype(np.uint8), log, status


# =====================================================================================================
# Files: reading the input, writing the output in the input's format
# =====================================================================================================
LOG_GUIDE = """
How to read this log
- Each line starts with the case and says what one step checked and what it changed. Steps 1-4 cover the
  vertebrae L5-T4; step 5 (clean-up) covers all 24 vertebrae C1-L5.
- "X -> Y" means bone that the model labelled X is now labelled Y. Volumes are in cm3 (1 cm3 = 1000 mm3);
  for scale, a whole lumbar vertebra is about 60-70 cm3.
- Terms: the body is the round front part of a vertebra; the arch is the ring of bone behind it, around the
  spinal canal; the pedicles are the two short bars of bone joining the arch to the body; the spinous process
  points backwards from the arch, and its root is where it leaves the arch, at the back of the spinal canal.
- Each case ends with a summary line and a status: "refined" means all steps were applied; "steps 1-4 skipped"
  means the safety checks did not trust the bodies found in step 1, so the model's labels were kept and only
  the clean-up ran (the reason is given); "empty" means the model labelled no vertebra.
"""


def sha256_of(path: str) -> str:
    """sha256 checksum of a file (used to prove the input is not modified)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files(case_dir: str) -> list:
    """The case's combined label map followed by its per-vertebra masks (sorted)."""
    mask_dir = os.path.join(case_dir, MASK_DIR)
    masks = sorted(f for f in os.listdir(mask_dir) if f.startswith("vertebrae_") and f.endswith(".nii.gz"))
    return [os.path.join(case_dir, COMBINED_FILE)] + [os.path.join(mask_dir, f) for f in masks]


def process_case(case: str, args, log_file) -> str:
    """Correct one case and write its output folder.

    The output is first written to a hidden folder <output>/.<case>.partial and renamed to <output>/<case> only
    when every file is complete, so a failed run never leaves a half-written case. Each per-vertebra mask is
    saved with the affine and a copy of the header of the corresponding input mask; combined_labels.nii.gz is
    rebuilt from those masks and saved with the input combined map's affine and header.

    Returns:
        A one-line summary for the console.
    """
    files = input_files(os.path.join(args.input, case))
    combined_input = nib.load(files[0])
    refined, log, status = refine_labels(files[0], os.path.join(args.ct_root, case, "ct.nii.gz"))
    log_file.write("".join(f"{case}: {line}\n" for line in log))
    if refined.shape != combined_input.shape:
        raise RuntimeError(f"refined shape {refined.shape} differs from the input shape {combined_input.shape}")

    partial = os.path.join(args.output, f".{case}.partial")
    if os.path.exists(partial):
        shutil.rmtree(partial)
    os.makedirs(os.path.join(partial, MASK_DIR))
    combined = np.zeros(refined.shape, np.uint8)
    written = set()
    for path in files[1:]:
        level = os.path.basename(path)[len("vertebrae_"):-len(".nii.gz")]
        if level not in LABEL_VALUE:
            raise RuntimeError(f"unexpected mask file {path}")
        value = LABEL_VALUE[level]
        source = nib.load(path)
        if source.shape != refined.shape or not np.allclose(source.affine, combined_input.affine):
            raise RuntimeError(f"{path} does not have the same grid as {files[0]}")
        mask = (refined == value).astype(np.uint8)
        combined[mask > 0] = value
        written.add(value)
        nib.save(nib.Nifti1Image(mask, source.affine, source.header.copy()),
                 os.path.join(partial, MASK_DIR, os.path.basename(path)))
    missing = sorted(set(np.unique(refined)) - {0} - written)
    if missing:
        raise RuntimeError(f"levels {[LABEL_NAME[v] for v in missing]} have no mask file in the input")
    nib.save(nib.Nifti1Image(combined, combined_input.affine, combined_input.header.copy()),
             os.path.join(partial, COMBINED_FILE))
    final = os.path.join(args.output, case)
    if os.path.exists(final):
        shutil.rmtree(final)                # only reached with --overwrite (checked before the run)
    os.rename(partial, final)
    voxel_cm3 = float(np.prod(combined_input.header.get_zooms()[:3])) / 1000
    original = np.asarray(combined_input.dataobj)
    return (f"{case}: done ({status}); {len(files) - 1} vertebra masks and {COMBINED_FILE} written. "
            f"{(original != combined).sum() * voxel_cm3:.2f} cm3 of the {(original > 0).sum() * voxel_cm3:.2f} cm3 "
            f"labelled by the model changed label; total labelled volume now {(combined > 0).sum() * voxel_cm3:.2f} cm3")


def main():
    parser = argparse.ArgumentParser(
        description="Correct SuPreM vertebrae segmentations (C1-L5); see the top of this file for details.")
    parser.add_argument("--input", required=True, help="folder with the model's output, one sub-folder per case")
    parser.add_argument("--output", required=True, help="folder for the corrected output (same layout)")
    parser.add_argument("--ct_root", help="folder with <case>/ct.nii.gz (default: AbdomenAtlasDemo next to --input, else next to --output)")
    parser.add_argument("--cases", nargs="*", help="process only these cases (default: every case in --input)")
    parser.add_argument("--log", help="log file (default: postprocessing_vertebrae.log next to the output folder)")
    parser.add_argument("--overwrite", action="store_true", help="replace case folders that exist in --output")
    args = parser.parse_args()
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    if args.ct_root is None:   # default: a folder AbdomenAtlasDemo next to the input folder, else next to the output
        candidates = [os.path.join(os.path.dirname(folder), "AbdomenAtlasDemo") for folder in (args.input, args.output)]
        found = [c for c in candidates if os.path.isdir(c)]
        if not found:
            raise SystemExit("CT folder not found: looked for " + " and ".join(candidates) + " (use --ct_root)")
        args.ct_root = found[0]
    args.ct_root = os.path.abspath(args.ct_root)

    # safety checks before anything is written
    if os.path.realpath(args.input) == os.path.realpath(args.output):
        raise SystemExit("refusing to run: --output is the same folder as --input")
    if os.path.realpath(args.output).startswith(os.path.realpath(args.input) + os.sep):
        raise SystemExit("refusing to run: --output lies inside --input")
    cases = args.cases or sorted(c for c in os.listdir(args.input)
                                 if os.path.isfile(os.path.join(args.input, c, COMBINED_FILE)))
    if not cases:
        raise SystemExit(f"no case folders with {COMBINED_FILE} found in {args.input}")
    missing_ct = [case for case in cases if not os.path.isfile(os.path.join(args.ct_root, case, "ct.nii.gz"))]
    if len(missing_ct) == len(cases):
        raise SystemExit(f"no CT found for any case in {args.ct_root} (use --ct_root)")
    for case in missing_ct:     # these cases fail with a clear message; the others are still processed
        print(f"warning: CT not found for {case}: {os.path.join(args.ct_root, case, 'ct.nii.gz')}")
    for case in cases:
        if os.path.exists(os.path.join(args.output, case)) and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {os.path.join(args.output, case)} (use --overwrite)")
    os.makedirs(args.output, exist_ok=True)
    log_path = args.log or os.path.join(os.path.dirname(args.output), "postprocessing_vertebrae.log")

    checksums = {path: sha256_of(path) for case in cases for path in input_files(os.path.join(args.input, case))}
    failed = []
    with open(log_path, "w") as log_file:
        log_file.write(f"Vertebra post-processing log\ninput:  {args.input}\noutput: {args.output}\nCTs:    {args.ct_root}\n"
                       + LOG_GUIDE + "\n")
        for case in cases:
            started = time.time()
            try:
                summary = process_case(case, args, log_file) + f"  ({time.time() - started:.0f} s)"
            except Exception as error:      # report the failure, continue with the other cases
                failed.append(case)
                summary = f"{case}: FAILED - {type(error).__name__}: {error}"
            print(summary)
            log_file.write(summary + "\n")
        changed = [p for p, digest in checksums.items() if not os.path.exists(p) or sha256_of(p) != digest]
        status = (f"input files unchanged ({len(checksums)} files verified with sha256 checksums)" if not changed
                  else "WARNING: input files changed: " + ", ".join(changed))
        print(status)
        log_file.write(status + "\n")
    print(f"log: {log_path}")
    if failed or changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
