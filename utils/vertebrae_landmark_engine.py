"""
postprocessing_vertebrae.py
===========================

Post-processing for SuPreM vertebra predictions on abdominal CT
(BodyMaps warm-up, AbdomenAtlasDemo: BDMAP_00000006 and BDMAP_00000031).

The model (Swin UNETR trained on TotalSegmentator vertebrae) labels every voxel
as background or one of 24 vertebrae. This script corrects its anatomical
mistakes without retraining and writes the refined predictions in the same
layout as the input.

Label convention (combined_labels.nii.gz)
-----------------------------------------
    0          background
    1 .. 5     L5, L4, L3, L2, L1      (lumbar, bottom of the spine)
    6 .. 17    T12 ... T1              (thoracic)
    18 .. 24   C7 ... C1               (cervical, top of the spine)

Changes since the 22 October submission
---------------------------------------
Steps 1 and 2 are the submitted code, unchanged in behaviour. Added after
the first evaluation (average DSC 83.3%; L1-T5 at 49-84%), using ideas
from the related work in SuPreM direct_inference/vertebrae.md:
  * Step 2c  landmark_names(), rib_bearing(): number T1-L5 by counting
             bodies from L5 up to C7 and checking that the last rib is on
             T12 (TotalSpineSeg [1], Meng et al. [2]). Falls back to the
             submitted dynamic programming if any check fails.
  * Step 3   arches_follow_body(): each arch takes the name of the body
             its pedicles join (targets L1, T12, T11, T10, T9).
  * Step 3b  fragments_join_neighbour(): a detached piece lying against
             the vertebra above or below joins it (AtlasDataset [3];
             targets T5-T8).
  * Step 4   order_ok() now counts slice by slice (same result, far less
             memory on large scans).
BDMAP_00000006 is unchanged by all of these (identical to 22 October).

What was wrong with the raw predictions
---------------------------------------
The predictions were inspected visually (side/front overlays, per-piece
close-ups) and numerically (volume and number of connected pieces per
vertebra). The model outlines bone well; its errors are about WHICH
vertebra a piece of bone is assigned to.

  Issue 1  Bone far from the spine labelled as vertebra.
           BDMAP_00000006: pieces labelled C1, C2 and C3 (neck) sat in the
           pelvis, about 600 mm from the neck (CT 360-520 HU, i.e. pelvic bone).
           This alone made the spine "out of order" (C1 averaged below C2).

  Issue 2  Small islands of the wrong label inside another vertebra.
           BDMAP_00000031: e.g. "L3" and "L4" voxels embedded in T8/T9,
           "C7" bits embedded in T6/T7, more than 120 mm from where those
           names belong. This is why vertebrae came out in many pieces
           (up to 11 pieces for T11; 14 of 24 vertebrae fragmented).

  Issue 3  One vertebral body carrying two names (the largest error).
           BDMAP_00000031, T7 to L2: single bodies were split between
           neighbouring labels, e.g.
             - the body that should be T9 was labelled 64% T8, 36% T9,
               so "T8" measured 56 mL and "T9" only 35 mL;
             - two consecutive bodies were both mostly "T11";
             - the body under the last T11 body was labelled T12, and the
               L1 body was shared with L2, so L1 measured 22 mL instead of
               a full lumbar body (about 60 mL).
           Real vertebrae change size gradually from level to level, so
           these jumps in the size curve were the clearest evidence.

  Issue 4  Arches numbered one level too high (found from the first score).
           Each vertebra is its body PLUS its arch (pedicles, laminae,
           transverse and spinous processes), joined by the pedicles. In
           BDMAP_00000031, from T9 to L1, the body carried the right name but
           the arch behind it carried the name of the level above: e.g. the
           body named T10 sat in front of an arch labelled 80% "T9". Against
           the reference labels each of these vertebrae is wrong twice.
           Evidence: the lab's first evaluation (average DSC 83.3%) fell to
           49-58% exactly at T9-T12; ribs (T1-T12 only) confirmed that the
           body numbering is anatomically right; BDMAP_00000006 shows no
           such mismatch.

  Not an error (left untouched): the long spinous processes ("fins") at the
  back of T4-T6 slope down onto the vertebra below. Measured position
  (35-42 mm behind and 20-27 mm below their own body) shows they are normal
  anatomy carrying the correct name.

How each issue is fixed
-----------------------
  Step 1  clean_fragments()       -> Issues 1 and 2
          Each label is split into 26-connected pieces; the largest piece
          is the vertebra itself and is always kept. Every other piece is
          judged on the input labels (so processing order cannot matter):
            mostly not bone (mean CT < 150 HU)          -> removed
            small and embedded in another vertebra      -> given that name
            close to its own vertebra (<= 20 mm)        -> kept
            far from every vertebra                     -> removed
            large and pressed against a neighbour       -> kept for Step 2
          "Small" = under 1 mL AND under 5% of that vertebra.

  Step 2  one_name_per_body()     -> Issue 3 (T1 to L5)
          a) Trace the spine's centre line through the vertebral bodies.
          b) Find the discs: along the centre line the model leaves an
             unlabelled gap, or at least a clear dip in labelled coverage,
             at every intervertebral disc. These cut the column into bodies.
          c) Give every body ONE name by LANDMARK COUNTING (idea from the
             related work [1] TotalSpineSeg, landmark-based labelling, and
             [2] Meng et al., anatomical consistency): the lowest body is L5,
             the cervical spine (C7) sits directly above the top body, and
             every disc in between adds one level. It is used only when all
             17 bodies are found and the rib check agrees: ribs attach to
             T1-T12 only, so the 6th body from the bottom (T12) must carry
             ribs and the two below it (L1, L2) must not.
             If any landmark test fails, the names come instead from dynamic
             programming on the model's own labels: names rise strictly up
             the spine and agree as much as possible with the model (a name
             may be skipped at a small cost, allowing 4- or 6-lumbar spines).
          d) Only the body region is renamed (in front of the spinal canal);
             arches are handled by Step 3.

  Step 3  arches_follow_body()    -> Issue 4 (T1 to L5)
          a) For every arch voxel, find the body it is attached to: the
             cheapest path through the labelled bone from each body
             (skimage MCP); darker voxels on the CT (joint spaces) cost more,
             so the pedicles, solid bone, join an arch to its own body.
          b) On a correct spine only 15-30% of an arch is reached more cheaply
             from the body BELOW (the sloping spinous process). An arch label
             MOSTLY (> 50%) attached to the body below is numbered one level
             too high: each of its connected pieces takes the name of the
             body it is attached to. Renaming whole pieces keeps a spinous
             process with its own arch.
          c) At the top of a run of renamed arches, that vertebra's arch lay
             inside the label above; it is taken back voxel by voxel.
          d) The Step 1 rules run once more to absorb small leftover islands.

  Step 3b fragments_join_neighbour() -> leftover arch pieces (T1 to L5)
          A fragment moves only between two vertebrae listed in
          --frag_levels (default: all of T1-L5).
          Idea from [3] (AtlasDataset post-processing): a piece of label w
          that is cut off from w's main piece (6-connectivity) and lies
          against the main piece of the vertebra directly above or below is
          given that neighbour's name. Fixes the upper arch pieces of T4-T7
          in BDMAP_00000031 that carried the name of the vertebra above.

  Step 4  checks                  -> keeps every change safe
          a) Acceptance gates: Step 2 and Step 3 are each kept only if the
             vertebrae stay in head-to-feet order and the size curve from T1
             to L5 does not get rougher (Step 3 must also raise the share of
             arch bone named like its body). Otherwise that step is undone.
          b) Integrity checks on the written files: same shape and affine as
             the input, labels only 0-24, no background turned into bone,
             all 24 masks present and identical to combined_labels.

Results on the two demo cases
-----------------------------
  BDMAP_00000006  Step 1: 3 pelvic pieces removed, 14 islands renamed;
                  22/24 vertebrae now a single piece, order correct.
                  Step 2: rejected by the gate (it would have merged T2/T3),
                  so the already-correct Step 1 result is kept.
  BDMAP_00000031  Step 1: 8 junk pieces removed, 68 islands renamed.
                  Step 2: 17 bodies found (T1-L5) between L5 and C7; ribs on
                  the 6th body from the bottom (3.9 / 5.4 mL), none on the two
                  below it, so the landmark numbering was used. T8/T9,
                  T11/T12 and L1/L2 split bodies resolved; size-curve
                  roughness 6.6 -> 2.7.
                  Step 3: arches T9, T10, T11, T12, L1 were 73%, 73%, 86%,
                  89% and 62% attached to the body below and were renamed.
                  Share of arch bone named like the body in front of it,
                  T9-L2: 12-63% -> 81-98% (BDMAP_00000006 for comparison:
                  82-90%). Pieces per vertebra: T9 3 -> 1, T10 3 -> 1,
                  L1 5 -> 1, T8 4 -> 2, T7 5 -> 3. Size roughness 2.7 -> 1.9.
                  BDMAP_00000006: no arch renamed (every arch at most 29%
                  attached below), output unchanged.
                  Step 3b: 14 fragments (0.3-3.6 mL) joined the vertebra
                  they touch, mostly T4-T7 (against TotalSegmentator,
                  BDMAP_00000031: T5 +10.4, T6 +10.2, T7 +4.0, T4 +3.7).
                  BDMAP_00000006: no fragment qualified, output unchanged.
  Both cases pass every integrity check.

Known limitations
-----------------
  Ground-truth labels are not available here. As an independent check the
  CTs were also segmented with TotalSegmentator: it agrees with this output
  for every vertebra of BDMAP_00000006 and for L2-L5, T1-T7 and C1-C7 of
  BDMAP_00000031. For T8-L1 of BDMAP_00000031 TotalSegmentator itself is
  anatomically inconsistent (its L1 is 7 mL and its T8 spans two vertebral
  bodies), and SuPreM, trained on TotalSegmentator labels, partly inherits
  that error; there the numbering above rests on the landmark count and
  the rib check, not on a reference. Arch bone of T5-T7 that is still
  connected to the wrong vertebra (not a detached fragment) keeps its name.

References (SuPreM direct_inference/vertebrae.md, related work)
  [1] Warszawer et al. TotalSpineSeg: robust spine segmentation with
      landmark-based labeling in MRI. 2025.
  [2] Meng, Boyer, Pujades. Vertebrae localization, segmentation and
      identification using a graph optimization and an anatomic
      consistency cycle. CMIG 107 (2023) 102235.
  [3] Jaus et al. Towards unifying anatomy segmentation: automated
      generation of a full-body CT dataset. ICIP 2024 (AtlasDataset).

Output
------
    <output_dir>/<case>/combined_labels.nii.gz
    <output_dir>/<case>/segmentations/vertebrae_<name>.nii.gz   (all 24)
    <output_dir>/<case>/postprocessing_log.json   (every decision and why)

Usage (requires numpy, scipy, nibabel, scikit-image)
-----
    python postprocessing_vertebrae.py ^
        --input_dir  D:\\bodymaps\\AbdomenAtlasDemoPredict_raw ^
        --ct_dir     D:\\bodymaps\\AbdomenAtlasDemo ^
        --output_dir D:\\bodymaps\\AbdomenAtlasDemoPredict

    The raw input folder is never modified. --ct_dir is optional but enables
    the "not bone" test and the CT term of Step 3. --no_step2 / --no_step3 /
    --no_step3b
    skip those steps.
"""

import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage
from scipy.signal import find_peaks

NAMES = (["L5", "L4", "L3", "L2", "L1"]
         + ["T%d" % i for i in range(12, 0, -1)]
         + ["C%d" % i for i in range(7, 0, -1)])
N_LABELS = len(NAMES)                      # label v (1..24) is NAMES[v - 1]
THORACOLUMBAR = 17                         # labels 1..17 = L5 .. T1 (Step 2 range)
STRUCT = np.ones((3, 3, 3), bool)          # 26-connectivity for "pieces"


# =============================================================================
# Input / output
# =============================================================================
def load_case(case_dir, ct_path):
    """Load one case's combined labels (and its CT if available)."""
    img = nib.load(os.path.join(case_dir, "combined_labels.nii.gz"))
    lab = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    bad = set(np.unique(lab).tolist()) - set(range(N_LABELS + 1))
    if bad:
        raise ValueError("%s: unexpected label values %s" % (case_dir, sorted(bad)))
    ct = None
    if ct_path and os.path.exists(ct_path):
        ct = np.asarray(nib.load(ct_path).dataobj).astype(np.float32)
        if ct.shape != lab.shape:
            raise ValueError("%s: CT %s vs labels %s" % (case_dir, ct.shape, lab.shape))
    return img, lab, ct


def save_case(img, lab, out_case):
    """Write combined_labels plus one binary mask per vertebra, same geometry as input."""
    os.makedirs(os.path.join(out_case, "segmentations"), exist_ok=True)
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(lab.astype(np.uint8), img.affine, header),
             os.path.join(out_case, "combined_labels.nii.gz"))
    for v, name in enumerate(NAMES, start=1):
        nib.save(nib.Nifti1Image((lab == v).astype(np.uint8), img.affine, header),
                 os.path.join(out_case, "segmentations", "vertebrae_%s.nii.gz" % name))


# =============================================================================
# STEP 1 - clean fragments  (fixes Issue 1: far strays, Issue 2: wrong islands)
# =============================================================================
def padded_box(box, shape, pad):
    """Bounding box grown by `pad` voxels, clipped to the volume."""
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n))
                 for s, n in zip(box, shape))


def main_bodies(lab):
    """Volume holding only the LARGEST piece of each label.

    The largest piece of a label is taken to be the real vertebra. Every
    decision in Step 1 is made against these main bodies, computed once from
    the input, so the order in which labels are processed cannot change the
    result.
    """
    mains = np.zeros_like(lab)
    boxes = ndimage.find_objects(lab, max_label=N_LABELS)
    for v in range(1, N_LABELS + 1):
        if boxes[v - 1] is None:
            continue
        box = boxes[v - 1]
        comp, n = ndimage.label(lab[box] == v, structure=STRUCT)
        if n == 0:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        mains[box][comp == (np.argmax(sizes) + 1)] = v
    return mains


def clean_fragments(lab, ct, zooms, cfg):
    """Remove or rename the extra pieces of every vertebra label.

    For each label, every piece other than the main body is classified:

      * mean CT below --bone_hu (150 HU)  -> REMOVE
            Not bone at all (soft tissue mislabelled as vertebra).
      * small, touching another vertebra's main body -> RENAME to that vertebra
            Issue 2: an island of the wrong name sitting inside another bone
            (e.g. "L3" voxels inside T9). The bone it is embedded in owns it.
      * within --far_mm (20 mm) of its own main body -> KEEP
            A detached bit of the same vertebra (thin gap), or a large piece
            pressed against a neighbour, which Step 2 resolves.
      * far from its own body AND from every other vertebra -> REMOVE
            Issue 1: e.g. "C1" bone in the pelvis, ~600 mm from the neck.
      * far from its own body but touching a neighbour -> KEEP for Step 2

    Distances are in millimetres (voxel spacing), so the same thresholds work
    at any resolution. Voxels are only renamed or cleared, never added.
    Returns the cleaned labels and a list of logged decisions.
    """
    voxel_ml = float(np.prod(zooms)) / 1000.0
    mains = main_bodies(lab)
    out = lab.copy()
    actions = []
    boxes = ndimage.find_objects(lab, max_label=N_LABELS)
    far_vox = int(np.ceil(cfg.far_mm / min(zooms))) + 2

    for v in range(1, N_LABELS + 1):
        if boxes[v - 1] is None:
            continue
        box = padded_box(boxes[v - 1], lab.shape, 2)
        comp, n = ndimage.label(lab[box] == v, structure=STRUCT)
        if n <= 1:
            continue                                   # already a single piece
        sizes = np.bincount(comp.ravel())[1:]
        main_id = int(np.argmax(sizes)) + 1
        total = float(sizes.sum())
        # distance (mm) from every voxel to this vertebra's main body
        own_dist = ndimage.distance_transform_edt(comp != main_id, sampling=zooms)

        for cid in range(1, n + 1):
            if cid == main_id:
                continue
            piece = comp == cid
            nvox = int(sizes[cid - 1])
            size_ml = nvox * voxel_ml
            share = 100.0 * nvox / total
            d_own = float(own_dist[piece].min())

            # which OTHER vertebrae's main bodies this piece touches (1-voxel shell)
            shell = ndimage.binary_dilation(piece, structure=STRUCT) & ~piece
            touched = mains[box][shell]
            touched = touched[(touched > 0) & (touched != v)]
            contact = {}
            if touched.size:
                ids, counts = np.unique(touched, return_counts=True)
                contact = {int(i): int(c) for i, c in zip(ids, counts)}

            # is any other vertebra within --far_mm? (decides "far stray")
            near_other = bool(contact)
            if not near_other:
                wide = padded_box(boxes[v - 1], lab.shape, far_vox)
                other = (mains[wide] > 0) & (mains[wide] != v)
                if other.any():
                    sub_piece = np.zeros(other.shape, bool)
                    rel = tuple(slice(b.start - w.start, b.stop - w.start)
                                for b, w in zip(box, wide))
                    sub_piece[rel] = piece
                    d_other = ndimage.distance_transform_edt(~other, sampling=zooms)
                    near_other = float(d_other[sub_piece].min()) <= cfg.far_mm

            mean_hu = float(ct[box][piece].mean()) if ct is not None else None
            small = size_ml < cfg.small_ml and share < cfg.small_pct

            # ---- decision rules (see docstring) ----
            if mean_hu is not None and mean_hu < cfg.bone_hu:
                action, target, reason = "remove", 0, "mean %.0f HU: not bone" % mean_hu
            elif small and contact:
                target = max(contact, key=contact.get)
                action, reason = "relabel", "small piece embedded in %s" % NAMES[target - 1]
            elif d_own <= cfg.far_mm:
                action, target = "keep", v
                reason = ("large piece next to %s: left for step 2" % NAMES[max(contact, key=contact.get) - 1]
                          if contact and not small else "near its own vertebra")
            elif not near_other:
                action, target, reason = "remove", 0, "far from every vertebra"
            elif contact:
                action, target = "keep", v
                reason = "large piece against %s: left for step 2" % NAMES[max(contact, key=contact.get) - 1]
            else:
                action, target, reason = "remove", 0, "far from its vertebra, touches no other"

            if action == "remove":
                out[box][piece] = 0
            elif action == "relabel":
                out[box][piece] = target
            actions.append({
                "vertebra": NAMES[v - 1], "size_ml": round(size_ml, 3),
                "share_pct": round(share, 1), "dist_own_mm": round(d_own, 1),
                "mean_hu": None if mean_hu is None else round(mean_hu),
                "touches": {NAMES[k - 1]: c for k, c in contact.items()},
                "action": action,
                "to": NAMES[target - 1] if action == "relabel" else None,
                "reason": reason, "voxels": nvox,
            })
    return out, actions


# =============================================================================
# STEP 2 - one name per vertebral body, T1-L5  (fixes Issue 3)
# =============================================================================
def to_canonical(data, affine):
    """Reorient to RAS (x = right, y = front, z = up) and return the way back,
    so the geometry below works whatever orientation the file is stored in."""
    src = io_orientation(affine)
    ras = axcodes2ornt(("R", "A", "S"))
    fwd, back = ornt_transform(src, ras), ornt_transform(ras, src)
    return apply_orientation(data, fwd), (lambda d: apply_orientation(d, back)), fwd


def canonical_zooms(zooms, fwd):
    """Voxel spacing reordered to match the RAS array."""
    return tuple(float(zooms[int(fwd[i, 0])]) for i in range(3))


def centre_line(lab, zooms, zs, front_mm=25.0):
    """Centre (x, y) of the vertebral bodies on every axial slice.

    On each slice the labelled bone includes the body (front) and the arch,
    fins and rib joints (back). Taking only the front-most 25 mm keeps the
    line inside the bodies. The line is then gap-filled and smoothed.
    """
    sx, sy, sz = zooms
    cx = np.full(zs.size, np.nan)
    cy = np.full(zs.size, np.nan)
    for i, z in enumerate(zs):
        xs, ys = np.nonzero(lab[:, :, z] > 0)
        if xs.size < 20:
            continue
        front = ys >= np.percentile(ys, 98) - front_mm / sy
        cx[i], cy[i] = np.median(xs[front]), np.median(ys[front])
    ok = ~np.isnan(cx)
    if ok.sum() < 5:
        return None, None
    cx = np.interp(zs, zs[ok], cx[ok])
    cy = np.interp(zs, zs[ok], cy[ok])
    win = max(3, int(round(15.0 / sz)) | 1)
    cx = ndimage.gaussian_filter1d(ndimage.median_filter(cx, win), 4.0 / sz)
    cy = ndimage.gaussian_filter1d(ndimage.median_filter(cy, win), 4.0 / sz)
    return cx, cy


def body_zone(shape, zooms, zs, cx, cy, back_mm=15.0, half_width_mm=32.0):
    """Mask of the vertebral-BODY region only: in front of (centre - 15 mm) and
    within 32 mm left/right of the centre line. Arches, spinous processes and
    ribs lie outside it, so Step 2 never renames them (Step 3 handles arches)."""
    sx, sy, sz = zooms
    zone = np.zeros(shape, bool)
    xs = np.arange(shape[0])[:, None]
    ys = np.arange(shape[1])[None, :]
    for i, z in enumerate(zs):
        zone[:, :, z] = (ys >= cy[i] - back_mm / sy) & (np.abs(xs - cx[i]) * sx <= half_width_mm)
    return zone


def find_bodies(lab, zooms, zs, cx, cy, cfg):
    """Cut the spinal column into individual vertebral bodies at the discs.

    Measure, slice by slice, what share of a small disc (radius 8 mm) around
    the centre line is labelled ("occupancy"). At an intervertebral disc the
    model leaves either
      * an empty gap (occupancy below 0.25), or
      * where neighbouring labels touch across the disc, a clear dip in
        occupancy (drop of at least 0.15, bottoming out at 0.85 or less).
    Both give a cut. Cuts closer than --min_body_mm are merged (one per disc).

    CT brightness alone was tried first but proved unreliable: the spongy
    middle of a vertebral body is almost as dark as a disc. The model's own
    bone outline marks the discs much more consistently.
    """
    sx, sy, sz = zooms
    r = cfg.gap_radius_mm
    dx, dy = np.meshgrid(np.arange(-int(r / sx), int(r / sx) + 1),
                         np.arange(-int(r / sy), int(r / sy) + 1), indexing="ij")
    keep = (dx * sx) ** 2 + (dy * sy) ** 2 <= r * r
    dx, dy = dx[keep], dy[keep]
    xi = np.clip(np.rint(cx[:, None] + dx[None]).astype(int), 0, lab.shape[0] - 1)
    yi = np.clip(np.rint(cy[:, None] + dy[None]).astype(int), 0, lab.shape[1] - 1)
    occupancy = (lab[xi, yi, zs[:, None]] > 0).mean(axis=1)
    occ = ndimage.gaussian_filter1d(occupancy, 1.0 / sz)

    # (1) empty gaps: cut at the middle of each empty run
    cuts = []
    empty = occupancy < cfg.gap_occupancy
    i = 0
    while i < empty.size:
        if empty[i]:
            j = i
            while j + 1 < empty.size and empty[j + 1]:
                j += 1
            cuts.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    # (2) dips where neighbouring labels touch across a disc
    dips, _ = find_peaks(-occ, prominence=cfg.dip_prominence,
                         distance=max(1, int(cfg.min_body_mm / sz)))
    cuts += [int(d) for d in dips if occ[d] <= cfg.dip_max]
    cuts = sorted(set(cuts))
    merged = []
    for c in cuts:                                   # one cut per disc
        if merged and (c - merged[-1]) * sz < cfg.min_body_mm:
            merged[-1] = c if occupancy[c] < occupancy[merged[-1]] else merged[-1]
        else:
            merged.append(c)

    # bodies = stretches between cuts, with empty slices trimmed off the ends
    edges = [-1] + merged + [zs.size]
    runs = []
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = a + 1, b - 1
        while a <= b and empty[a]:
            a += 1
        while b >= a and empty[b]:
            b -= 1
        if a <= b:
            runs.append((a, b))

    # implausible lengths are not renamed: too short = partial slice at the
    # edge; too long = a disc was not found (two bodies merged)
    bodies = []
    for a, b in runs:
        length = (b - a + 1) * sz
        status = "ok"
        if length < cfg.min_body_mm:
            status = "too short (skipped)"
        elif length > cfg.max_body_mm:
            status = "too long, discs may be fused (skipped)"
        bodies.append({"i0": a, "i1": b, "z0": int(zs[a]), "z1": int(zs[b]),
                       "length_mm": round(length, 1), "status": status})
    return bodies


def assign_names(fracs, skip_cost):
    """Choose one name per body, bottom to top, by dynamic programming.

    fracs[i] = {label: share of body i's voxels currently carrying that label}.
    Constraint: names rise strictly up the spine (no repeats, no reversals).
    Objective: maximise the total agreement with the model's labels; skipping
    a name between two consecutive bodies costs `skip_cost` per skipped name
    (allows a genuinely missing level, e.g. a 4-lumbar spine).

    A simple majority vote per body is NOT enough: in BDMAP_00000031 the body
    that should be T9 was 64% "T8". Because the body above already takes T8,
    the ordering constraint forces this body to T9.
    """
    n, L = len(fracs), THORACOLUMBAR
    score = np.full((n, L + 1), -np.inf)
    prev = np.zeros((n, L + 1), int)
    for v in range(1, L + 1):
        score[0, v] = fracs[0].get(v, 0.0)
    for i in range(1, n):
        for v in range(1, L + 1):
            best, arg = -np.inf, 0
            for u in range(1, v):
                s = score[i - 1, u] - skip_cost * (v - u - 1)
                if s > best:
                    best, arg = s, u
            score[i, v] = best + fracs[i].get(v, 0.0)
            prev[i, v] = arg
    v = int(np.argmax(score[n - 1]))
    names = [v]
    for i in range(n - 1, 0, -1):
        v = int(prev[i, v])
        names.append(v)
    return names[::-1]


def rib_bearing(lab, ct, zooms, zs, cx, bodies, cfg):
    """Rib bone (mL, left and right) attached to each body.

    Ribs are bone the model did NOT label as vertebra. Unlabelled bone
    (CT > --rib_hu) is split into 3D pieces; a piece counts as rib when it
    touches the vertebra labels (within 4 mm) and reaches at least 60 mm
    sideways from the spine's centre line. It is credited to the body at the
    height where it meets the spine.
    """
    sx, sy, sz = zooms
    x0 = max(int(cx.min() - 130 / sx), 0); x1 = min(int(cx.max() + 130 / sx), lab.shape[0])
    z0, z1 = int(zs[0]), int(zs[-1]) + 1
    box = (slice(x0, x1), slice(None), slice(z0, z1))
    vert = lab[box] > 0
    bone = (ct[box] > cfg.rib_hu) & ~vert
    near = ndimage.distance_transform_edt(~vert, sampling=zooms) <= 4.0
    comp, _ = ndimage.label(bone, structure=STRUCT)
    del bone
    cxb = np.interp(np.arange(z0, z1), zs, cx) - x0
    lateral = np.abs(np.arange(vert.shape[0])[:, None] - cxb[None, :]) * sx     # (x, z)
    z_of_body = np.full(z1 - z0, -1)
    for i, b in enumerate(bodies):
        z_of_body[b["z0"] - z0:b["z1"] - z0 + 1] = i
    mids = np.array([(b["z0"] + b["z1"]) / 2 - z0 for b in bodies])
    vml = float(np.prod(zooms)) / 1000.0
    ribs = np.zeros((len(bodies), 2))
    for k, sl in enumerate(ndimage.find_objects(comp), start=1):
        if sl is None:
            continue
        piece = comp[sl] == k
        contact = np.argwhere(piece & near[sl])
        if contact.size == 0:
            continue
        pts = np.argwhere(piece) + [s.start for s in sl]
        if lateral[pts[:, 0], pts[:, 2]].max() < 60.0:
            continue
        contact += [s.start for s in sl]
        zc = int(np.median(contact[:, 2]))
        bi = z_of_body[zc] if z_of_body[zc] >= 0 else int(np.argmin(np.abs(mids - zc)))
        side = int(np.median(contact[:, 0] - cxb[contact[:, 2]]) < 0)
        ribs[bi, side] += piece.sum() * vml
    return ribs


def landmark_names(lab, bodies, usable, zooms, zs, cx, ct_native, affine, cfg):
    """Number the T1-L5 bodies by counting from anatomical landmarks.

    Idea from the related work (TotalSpineSeg [1]: landmark-based labelling;
    Meng et al. [2]: anatomical consistency) - do not trust the network's
    per-voxel names for the level; COUNT the bodies between landmarks:
      * bottom landmark: the lowest body is L5 (the model labels it L5,
        DSC 99% in the first evaluation);
      * top landmark: the cervical spine (C7) sits directly above the top body;
      * between them every disc adds one level, so with all 17 bodies found
        the names are fixed: L5, L4, ..., L1, T12, ..., T1;
      * rib check: ribs attach to T1-T12 only, so the last rib-bearing body
        must be the 6th from the bottom (T12).
    Used only if ALL of these hold; otherwise returns None (model votes used).
    """
    info = {"used": False}
    if any(b["status"] != "ok" for b in bodies):
        info["reason"] = "a body was too short or too long (a disc may be missing)"
        return None, info
    n = len(usable)
    info["bodies_found"] = n
    if n != THORACOLUMBAR:
        info["reason"] = "found %d bodies between the landmarks, expected 17" % n
        return None, info
    bottom = usable[0]["fractions"]
    if max(bottom, key=bottom.get) != 1:
        info["reason"] = "lowest body is not labelled L5 by the model"
        return None, info
    top = usable[-1]
    above = lab[:, :, top["z1"] + 1: top["z1"] + 1 + int(30.0 / zooms[2])]
    if not (above == THORACOLUMBAR + 1).any():
        info["reason"] = "no C7 directly above the top body"
        return None, info
    names = list(range(1, THORACOLUMBAR + 1))           # L5 .. T1, bottom to top
    if ct_native is not None:
        ct = to_canonical(ct_native, affine)[0]
        ribs = rib_bearing(lab, ct, zooms, zs, cx, usable, cfg)
        has = [bool(ribs[i].max() >= cfg.rib_min_ml) for i in range(n)]
        info["ribs_ml"] = {NAMES[names[i] - 1]: [round(float(r), 1) for r in ribs[i]] for i in range(n)}
        # the 6th body from the bottom (T12) must carry ribs, the two below it
        # (L1, L2) must not; L3-L5 are not tested because the pelvis touches them
        if not (has[5] and not has[4] and not has[3]):
            info["reason"] = ("rib check failed (rib-bearing: L2 %s, L1 %s, T12 %s)"
                              % (has[3], has[4], has[5]))
            return None, info
    info["used"] = True
    info["reason"] = "17 bodies between L5 and C7, last rib on T12: numbered by counting"
    return names, info


def one_name_per_body(lab_native, affine, zooms_native, cfg, ct_native=None):
    """Step 2: give every T1-L5 vertebral body a single name, in order.

    Works in RAS orientation and returns labels in the file's own orientation.
    The result passes through the Step 4 acceptance gate before it is kept.
    """
    lab, back, fwd = to_canonical(lab_native, affine)
    zooms = canonical_zooms(zooms_native, fwd)
    voxel_ml = float(np.prod(zooms)) / 1000.0
    tl = np.where(lab <= THORACOLUMBAR, lab, 0)          # thoracolumbar labels only
    zs = np.nonzero((tl > 0).any(axis=(0, 1)))[0]
    report = {"applied": False, "bodies": []}
    if zs.size < 10:
        report["reason"] = "too few thoracolumbar slices"
        return lab_native, report
    zs = np.arange(zs.min(), zs.max() + 1)
    cx, cy = centre_line(tl, zooms, zs)
    if cx is None:
        report["reason"] = "could not trace the spine"
        return lab_native, report

    # a) find the bodies and what the model currently calls each one
    zone = body_zone(lab.shape, zooms, zs, cx, cy)
    bodies = find_bodies(tl, zooms, zs, cx, cy, cfg)
    usable = []
    for b in bodies:
        sl = (slice(None), slice(None), slice(b["z0"], b["z1"] + 1))
        vals = tl[sl][zone[sl]]
        vals = vals[vals > 0]
        b["voxels"] = int(vals.size)
        if vals.size == 0:
            b["status"] = "no labelled bone (skipped)"
        counts = np.bincount(vals, minlength=THORACOLUMBAR + 1)
        b["fractions"] = {int(v): float(counts[v]) / max(vals.size, 1)
                          for v in range(1, THORACOLUMBAR + 1) if counts[v] > 0}
        b["now"] = ", ".join("%s %d%%" % (NAMES[v - 1], round(100 * f))
                             for v, f in sorted(b["fractions"].items(), key=lambda t: -t[1])[:3])
        if b["status"] == "ok":
            usable.append(b)
    if len(usable) < 3:
        report["reason"] = "fewer than 3 usable bodies"
        report["bodies"] = [{k: b[k] for k in ("length_mm", "status", "now")} for b in bodies]
        return lab_native, report

    # b) one name per body, in order up the spine.
    #    First choice: landmark numbering (count from both ends, check ribs);
    #    if the landmarks do not all agree, fall back to the model's own votes.
    names, landmarks = landmark_names(lab, bodies, usable, zooms, zs, cx, ct_native, affine, cfg)
    report["landmarks"] = landmarks
    if names is None:
        names = assign_names([b["fractions"] for b in usable], cfg.skip_cost)

    # c) rename the BODY region of each body to its single name
    out = lab.copy()
    changed = 0
    for b, v in zip(usable, names):
        b["name"] = NAMES[v - 1]
        sl = (slice(None), slice(None), slice(b["z0"], b["z1"] + 1))
        sub = out[sl]
        wrong = zone[sl] & (sub > 0) & (sub <= THORACOLUMBAR) & (sub != v)
        b["voxels_renamed"] = int(wrong.sum())
        changed += b["voxels_renamed"]
        sub[wrong] = v

    # d) a name that no body received is spurious; its leftover voxels (arch
    #    bits) at a named body's height join that body
    assigned = set(names)
    orphans = [v for v in range(1, THORACOLUMBAR + 1) if v not in assigned]
    if orphans:
        for b, v in zip(usable, names):
            sub = out[:, :, b["z0"]:b["z1"] + 1]
            stray = np.isin(sub, orphans)
            n = int(stray.sum())
            if n:
                sub[stray] = v
                b["voxels_renamed"] += n
                changed += n
    report["names_without_body"] = [NAMES[v - 1] for v in orphans
                                    if v < max(names) and v > min(names)]

    # e) Step 4a acceptance gate: keep only if order holds and sizes get smoother
    before = (volume_roughness(lab, voxel_ml), order_ok(lab))
    after = (volume_roughness(out, voxel_ml), order_ok(out))
    report["roughness_before"] = round(before[0], 3)
    report["roughness_after"] = round(after[0], 3)
    report["order_ok_after"] = after[1]
    report["voxels_renamed"] = changed
    report["bodies"] = [{k: b.get(k) for k in ("length_mm", "status", "now", "name", "voxels_renamed")}
                        for b in bodies]
    if changed == 0:
        report["reason"] = "every body already had a single consistent name"
        return lab_native, report
    if not after[1] or after[0] > before[0] * (1.0 + cfg.max_roughness_increase):
        report["reason"] = "reverted: checks got worse (order or size smoothness)"
        return lab_native, report
    report["applied"] = True
    report["reason"] = "applied: size curve smoother or equal, order kept"
    return back(out).astype(lab_native.dtype), report


# =============================================================================
# STEP 3 - arches follow their body  (fixes Issue 4)
# =============================================================================
def arch_owner(T, zone, ct, zooms, cfg):
    """For every thoracolumbar bone voxel, the vertebral body it is attached to.

    Shortest path through the labelled bone from each body (skimage MCP).
    Moving through bright bone costs its length in mm; crossing darker voxels
    (joint spaces, under --arch_gap_hu) costs up to (1 + --arch_gap_cost)
    times more. A posterior voxel is owned by the body it reaches most cheaply:
    the pedicles join an arch directly to its own body.
    """
    from skimage.graph import MCP_Geometric          # scikit-image
    M = T > 0
    cost = np.ones(T.shape, np.float32)
    if ct is not None:
        cost += cfg.arch_gap_cost * np.clip((cfg.arch_gap_hu - ct) / cfg.arch_gap_hu, 0, 1)
    cost[~M] = np.inf
    best = np.full(T.shape, np.inf, np.float32)
    owner = np.zeros(T.shape, np.uint8)
    win = int(cfg.arch_window_mm / zooms[2])
    for v in range(1, THORACOLUMBAR + 1):
        seeds = zone & (T == v)
        if not seeds.any():
            continue
        zz = np.nonzero(seeds.any(axis=(0, 1)))[0]
        z0, z1 = max(zz.min() - win, 0), min(zz.max() + win + 1, T.shape[2])
        pts = np.argwhere(seeds[:, :, z0:z1])
        pts = pts[:: max(1, len(pts) // 4000)]               # a few thousand seeds suffice
        d, _ = MCP_Geometric(cost[:, :, z0:z1], fully_connected=True,
                             sampling=zooms).find_costs([tuple(p) for p in pts])
        sub_b, sub_o = best[:, :, z0:z1], owner[:, :, z0:z1]
        closer = d < sub_b
        sub_b[closer] = d[closer]
        sub_o[closer] = v
    return owner


def arches_follow_body(lab_native, affine, zooms_native, ct_native, cfg):
    """Step 3: rename arches that the model numbered one level too high.

    Measured on the demo cases with arch_owner(): on a correctly labelled
    spine (BDMAP_00000006) only 15-30% of each arch is reached more cheaply
    from the body BELOW (the sloping spinous process lies beside the next
    body). In BDMAP_00000031 the T9-L1 arch labels were 62-89% attached to
    the body below: the model had numbered those arches one level too high.

    Rule: an arch label whose posterior voxels are MOSTLY (> --arch_shift_frac)
    attached to the body one level below is renamed, as a whole, to that
    level. Renaming the whole arch (not voxel by voxel) keeps the spinous
    process with its own arch. Where a run of shifted arches starts, the
    vertebra at the top of the run has lost its arch to the rename; its arch
    sits inside the label above, and is taken back voxel by voxel using
    arch_owner(). The result passes the Step 4 gate before it is kept.
    """
    lab, back, fwd = to_canonical(lab_native, affine)
    zooms = canonical_zooms(zooms_native, fwd)
    report = {"applied": False, "renamed": [], "taken_back_ml": 0.0}
    tlm = (lab > 0) & (lab <= THORACOLUMBAR)
    if not tlm.any():
        report["reason"] = "no thoracolumbar labels"
        return lab_native, report
    box = ndimage.find_objects(tlm.astype(np.uint8))[0]
    box = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, n)) for s, n in zip(box, lab.shape))
    L = lab[box].copy()
    T = np.where((L > 0) & (L <= THORACOLUMBAR), L, 0).astype(np.uint8)
    ct = None
    if ct_native is not None:
        ct = to_canonical(ct_native, affine)[0][box].astype(np.float32)
    zs = np.nonzero((T > 0).any(axis=(0, 1)))[0]
    zs = np.arange(zs.min(), zs.max() + 1)
    cx, cy = centre_line(T, zooms, zs)
    if cx is None:
        report["reason"] = "could not trace the spine"
        return lab_native, report
    zone = body_zone(T.shape, zooms, zs, cx, cy) & (T > 0)
    post = (T > 0) & ~zone
    owner = arch_owner(T, zone, ct, zooms, cfg)
    vml = float(np.prod(zooms)) / 1000.0

    # fraction of each arch label attached to the body one level below
    below = {}
    for v in range(2, THORACOLUMBAR + 1):
        sel = post & (T == v)
        if sel.sum() * vml < 1.0:
            continue
        below[v] = float((owner[sel] == v - 1).mean())
    report["share_attached_below"] = {NAMES[v - 1]: round(f, 2) for v, f in below.items()}
    shifted = sorted(v for v, f in below.items() if f > cfg.arch_shift_frac)
    if not shifted:
        report["reason"] = "every arch is mostly attached to its own body"
        return lab_native, report

    new = L.copy()
    st = np.ones((3, 3, 3), bool)
    for v in shifted:
        # each connected piece of the shifted arch label goes, as a whole, to the
        # body most of it is attached to (normally the level below); a piece that
        # is mostly attached to its own body keeps its name
        comp, n = ndimage.label(post & (T == v), structure=st)
        moved = 0
        for k, sl in enumerate(ndimage.find_objects(comp), start=1):
            if sl is None:
                continue
            piece = comp[sl] == k
            votes = np.bincount(owner[sl][piece], minlength=THORACOLUMBAR + 1)
            votes[0] = 0
            target = int(np.argmax(votes)) if votes.any() else v
            if target != v:
                new[sl][piece] = target
                moved += int(piece.sum())
        report["renamed"].append({"arch": NAMES[v - 1], "to": NAMES[v - 2],
                                  "ml": round(float(moved * vml), 2),
                                  "attached_below": round(below[v], 2)})
    # top of each run of shifted arches: take that vertebra's arch back from the label above
    for v in shifted:
        up = v + 1
        if up in shifted or up > THORACOLUMBAR:
            continue
        take = post & (T == up) & (owner == v)
        new[take] = v
        report["taken_back_ml"] += round(float(take.sum() * vml), 2)

    # Step 4a gate: order kept, size curve not rougher, arches agree with bodies more
    agree_before = float((T[post] == owner[post]).mean())
    agree_after = float((new[post] == owner[post]).mean())
    r0, r1 = volume_roughness(L, vml), volume_roughness(new, vml)
    report.update({"arch_agreement_before": round(agree_before, 3),
                   "arch_agreement_after": round(agree_after, 3),
                   "roughness_before": round(r0, 3), "roughness_after": round(r1, 3)})
    if not order_ok(new) or r1 > r0 * (1.0 + cfg.max_roughness_increase) or agree_after <= agree_before:
        report["reason"] = "reverted: checks got worse (order, size smoothness or arch agreement)"
        return lab_native, report
    out = lab.copy()
    out[box] = new
    report["applied"] = True
    report["reason"] = "applied: arches now follow their body"
    return back(out).astype(lab_native.dtype), report


def fragments_join_neighbour(lab, zooms, cfg):
    """Step 3b: a detached fragment joins the neighbouring vertebra it touches.

    Idea from the related work [3] (Jaus et al., AtlasDataset post-processing,
    'spine_adjacent_pairs'): bone that the model gave the name of vertebra w
    but that is NOT connected to w's main piece, and instead lies against the
    main piece of the vertebra directly above or below, belongs to that
    neighbour. Pieces are taken with 6-connectivity (face neighbours), so a
    fragment attached to its true vertebra only through a thin, edge-to-edge
    bridge is still recognised as a separate piece.

    Rule, for every T1-L5 label w and every piece of w other than its largest
    (and of at least --frag_min_ml): count the piece's face contacts with the
    main pieces of w-1, w and w+1; if a neighbour's main piece has the most
    contacts (strictly more than w's own), the piece takes that neighbour's
    name. All decisions are made on the input to this step.
    """
    s6 = ndimage.generate_binary_structure(3, 1)
    vml = float(np.prod(zooms)) / 1000.0
    mains = np.zeros(lab.shape, np.uint8)              # largest 6-connected piece of each label
    boxes = ndimage.find_objects(lab, max_label=N_LABELS)
    for w in range(1, THORACOLUMBAR + 2):              # T1-L5 plus C7 (neighbour of T1)
        if boxes[w - 1] is None:
            continue
        box = boxes[w - 1]
        cc, n = ndimage.label(lab[box] == w, structure=s6)
        if n:
            sub = mains[box]
            sub[cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)] = w
    out = lab.copy()
    moved = []
    allowed = {NAMES.index(n.strip()) + 1 for n in cfg.frag_levels.split(",") if n.strip()}
    for w in sorted(allowed):
        if boxes[w - 1] is None:
            continue
        box = padded_box(boxes[w - 1], lab.shape, 2)
        cc, n = ndimage.label(lab[box] == w, structure=s6)
        if n < 2:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        main = int(np.argmax(sizes)) + 1
        for k in range(1, n + 1):
            if k == main or sizes[k - 1] * vml < cfg.frag_min_ml:
                continue
            piece = cc == k
            shell = ndimage.binary_dilation(piece, structure=s6) & ~piece
            touch = mains[box][shell]
            votes = {nb: int((touch == nb).sum()) for nb in (w - 1, w, w + 1) if nb >= 1}
            target = max(votes, key=votes.get)
            if target != w and target in allowed and votes[target] > votes.get(w, 0):
                out[box][piece] = target
                moved.append({"from": NAMES[w - 1], "to": NAMES[target - 1],
                              "ml": round(float(sizes[k - 1] * vml), 2)})
    report = {"applied": False, "moved": moved}
    if not moved:
        report["reason"] = "no detached fragment lies against a neighbour"
        return lab, report
    if not order_ok(out):
        report["reason"] = "reverted: vertebra order would break"
        return lab, report
    report["applied"] = True
    report["reason"] = "applied: %d fragments joined the vertebra they touch" % len(moved)
    return out, report


# =============================================================================
# STEP 4 - checks  (acceptance gates for Steps 2-3 + integrity of the written files)
# =============================================================================
def volume_roughness(lab, voxel_ml):
    """How unevenly vertebra size changes from level to level, T1 to L5.

    Sum of absolute second differences of log volume. Real spines grow
    smoothly from top to bottom; a body split between two names shows up as a
    jump (e.g. T8 56 mL, T9 35 mL, T10 56 mL in BDMAP_00000031).
    """
    counts = np.bincount(lab.ravel(), minlength=N_LABELS + 1)[1:THORACOLUMBAR + 1]
    vols = [(v, c * voxel_ml) for v, c in enumerate(counts, start=1) if c > 0]
    logs = np.log([max(vml, 1e-3) for _, vml in vols])
    if logs.size < 3:
        return 0.0
    return float(np.abs(np.diff(logs, 2)).sum())


def order_ok(lab):
    """True if every T1-L5 vertebra sits higher than the one numbered below it."""
    # mean height (axis 2) of each label, accumulated slice by slice so that
    # large, fine-sliced scans never need a full-volume float array
    counts = np.zeros(N_LABELS + 1)
    zsum = np.zeros(N_LABELS + 1)
    for z in range(lab.shape[2]):
        c = np.bincount(lab[:, :, z].ravel(), minlength=N_LABELS + 1)[:N_LABELS + 1]
        counts += c
        zsum += c * z
    present = [v for v in range(1, THORACOLUMBAR + 1) if counts[v] > 0]
    if len(present) < 2:
        return True
    heights = [zsum[v] / counts[v] for v in present]
    return all(b > a for a, b in zip(heights, heights[1:]))


def verify(raw, refined, out_case, affine):
    """Re-read what was written and check it is a valid, consistent output."""
    problems = []
    img = nib.load(os.path.join(out_case, "combined_labels.nii.gz"))
    data = np.asarray(img.dataobj)
    if data.shape != raw.shape:
        problems.append("shape changed")
    if not np.allclose(img.affine, affine):
        problems.append("affine changed")
    if not set(np.unique(data).tolist()) <= set(range(N_LABELS + 1)):
        problems.append("invalid label values")
    if np.any((raw == 0) & (data > 0)):
        problems.append("background turned into bone")
    for v, name in enumerate(NAMES, start=1):
        p = os.path.join(out_case, "segmentations", "vertebrae_%s.nii.gz" % name)
        if not os.path.exists(p):
            problems.append("missing %s" % name)
        elif not np.array_equal(np.asarray(nib.load(p).dataobj) > 0, data == v):
            problems.append("%s mask disagrees with combined labels" % name)
    return problems


# =============================================================================
# Main
# =============================================================================
def build_parser(require_dirs=True):
    """Command-line options; every threshold has the default used for the evaluated run."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=require_dirs, help="raw predictions (never modified)")
    ap.add_argument("--output_dir", required=require_dirs, help="where refined predictions go")
    ap.add_argument("--ct_dir", default=None, help="CT folders (<case>/ct.nii.gz); enables the bone test")
    # Step 1 thresholds
    ap.add_argument("--small_ml", type=float, default=1.0)
    ap.add_argument("--small_pct", type=float, default=5.0)
    ap.add_argument("--far_mm", type=float, default=20.0)
    ap.add_argument("--bone_hu", type=float, default=150.0)
    # Step 2 settings
    ap.add_argument("--no_step2", action="store_true", help="run step 1 only")
    ap.add_argument("--gap_radius_mm", type=float, default=8.0)
    ap.add_argument("--gap_occupancy", type=float, default=0.25,
                    help="a slice is 'disc' when under this share of the centre disc is labelled")
    ap.add_argument("--dip_prominence", type=float, default=0.15,
                    help="how much labelled coverage must drop at a disc where labels touch")
    ap.add_argument("--dip_max", type=float, default=0.85,
                    help="a dip only counts as a disc if coverage falls to this or below")
    ap.add_argument("--min_body_mm", type=float, default=10.0)
    ap.add_argument("--max_body_mm", type=float, default=50.0)
    ap.add_argument("--skip_cost", type=float, default=0.3)
    ap.add_argument("--rib_hu", type=float, default=200.0, help="bone threshold for the rib check")
    ap.add_argument("--rib_min_ml", type=float, default=2.0,
                    help="a body is rib-bearing with at least this much rib bone on one side")
    # Step 3 settings
    ap.add_argument("--no_step3", action="store_true", help="skip the arch correction")
    ap.add_argument("--arch_shift_frac", type=float, default=0.5,
                    help="rename an arch when more than this share is attached to the body below")
    ap.add_argument("--arch_gap_hu", type=float, default=250.0)
    ap.add_argument("--arch_gap_cost", type=float, default=10.0)
    ap.add_argument("--arch_window_mm", type=float, default=70.0)
    ap.add_argument("--no_step3b", action="store_true", help="skip 'fragments join their neighbour'")
    ap.add_argument("--frag_levels", default=",".join(NAMES[:THORACOLUMBAR]),
                    help="Step 3b moves a fragment only between two of these vertebrae, so every "
                         "other vertebra stays exactly as before")
    ap.add_argument("--frag_min_ml", type=float, default=0.2,
                    help="smallest detached fragment Step 3b considers")
    # Step 4 gate tolerance
    ap.add_argument("--max_roughness_increase", type=float, default=0.05)
    return ap


def default_config():
    """The evaluated settings, for callers that use refine() directly (e.g. ShapeKit)."""
    return build_parser(require_dirs=False).parse_args([])


def refine(raw, ct, affine, zooms, cfg=None):
    """Run Steps 1-3b on one case held in memory.

    raw    integer label volume (0 = background, 1..24 = L5..C1)
    ct     CT in HU on the same grid, or None (disables the CT-based tests)
    Returns (refined labels, report dict). The input array is not modified.
    """
    cfg = cfg or default_config()
    # Step 1: clean fragments
    refined, actions = clean_fragments(raw, ct, zooms, cfg)
    # Step 2: one name per body (includes the Step 4a acceptance gate)
    step2 = {"applied": False, "reason": "disabled with --no_step2"}
    if not cfg.no_step2:
        refined, step2 = one_name_per_body(refined, affine, zooms, cfg, ct)
    # Step 3: arches follow their body (includes its own acceptance gate)
    step3 = {"applied": False, "reason": "disabled with --no_step3"}
    if not cfg.no_step3:
        refined, step3 = arches_follow_body(refined, affine, zooms, ct, cfg)
        if step3.get("applied"):
            # renaming arches can leave small islands of the old name behind;
            # the Step 1 rules absorb them into the vertebra they sit in
            refined, tidy = clean_fragments(refined, ct, zooms, cfg)
            step3["tidy_up"] = {"relabelled": sum(a["action"] == "relabel" for a in tidy),
                                "removed": sum(a["action"] == "remove" for a in tidy)}
    # Step 3b: detached fragments join the neighbouring vertebra they touch
    step3b = {"applied": False, "reason": "disabled with --no_step3b"}
    if not cfg.no_step3b:
        refined, step3b = fragments_join_neighbour(refined, zooms, cfg)
    step3["step3b"] = step3b
    return refined, {"actions": actions, "step2": step2, "step3": step3}


def main():
    cfg = build_parser().parse_args()

    if os.path.abspath(cfg.input_dir) == os.path.abspath(cfg.output_dir):
        raise SystemExit("output_dir must differ from input_dir: the raw baseline is never overwritten")
    cases = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(cfg.input_dir, "*", "combined_labels.nii.gz")))
    if not cases:
        raise SystemExit("no <case>/combined_labels.nii.gz under %s" % cfg.input_dir)
    if not cfg.ct_dir:
        print("note: no --ct_dir given, so the 'not bone' test is skipped")

    for case in cases:
        ct_path = os.path.join(cfg.ct_dir, case, "ct.nii.gz") if cfg.ct_dir else None
        img, raw, ct = load_case(os.path.join(cfg.input_dir, case), ct_path)
        zooms = img.header.get_zooms()[:3]

        refined, rep = refine(raw, ct, img.affine, zooms, cfg)
        actions, step2, step3 = rep["actions"], rep["step2"], rep["step3"]
        # write, then Step 4b: integrity checks on the files written
        out_case = os.path.join(cfg.output_dir, case)
        save_case(img, refined, out_case)
        problems = verify(raw, refined, out_case, img.affine)

        # ---- log and console summary ----
        counts = {}
        for a in actions:
            counts[a["action"]] = counts.get(a["action"], 0) + 1
        moved = {k: sum(a["voxels"] for a in actions if a["action"] == k)
                 for k in ("remove", "relabel")}
        flagged = [a for a in actions if "step 2" in a["reason"]]
        with open(os.path.join(out_case, "postprocessing_log.json"), "w") as fh:
            json.dump({"case": case, "settings": vars(cfg), "summary": counts,
                       "voxels_removed": moved["remove"], "voxels_relabelled": moved["relabel"],
                       "integrity_problems": problems, "step2": step2, "step3": step3,
                       "actions": actions}, fh, indent=1)

        print("%s: %d pieces removed (%d voxels), %d relabelled (%d voxels), %d kept, %d flagged for step 2"
              % (case, counts.get("remove", 0), moved["remove"], counts.get("relabel", 0),
                 moved["relabel"], counts.get("keep", 0), len(flagged)))
        for a in flagged:
            print("   flagged: %-4s %.2f mL (%.0f%%) %s" % (a["vertebra"], a["size_ml"],
                                                       a["share_pct"], a["reason"]))
        print("   step 2 (one name per body): %s" % step2.get("reason", ""))
        if step2.get("landmarks"):
            lm = step2["landmarks"]
            print("     landmark numbering: %s%s" % ("USED - " if lm.get("used") else "not used - ",
                                                   lm.get("reason", "")))
        if step2.get("bodies"):
            print("     %d bodies found; size roughness %s -> %s; %s voxels renamed"
                  % (len(step2["bodies"]), step2.get("roughness_before"),
                     step2.get("roughness_after"), step2.get("voxels_renamed", 0)))
            for b in reversed(step2["bodies"]):          # head first
                if b.get("voxels_renamed") or b.get("status") != "ok":
                    print("     %5.1f mm  now [%s] -> %s  %s"
                          % (b["length_mm"], b.get("now", ""), b.get("name") or "-",
                             "" if b.get("status") == "ok" else "(" + b["status"] + ")"))
        if step2.get("applied") and step2.get("names_without_body"):
            print("     note: no vertebral body was given the name(s) %s; check this level by eye"
                  % ", ".join(step2["names_without_body"]))
        print("   step 3 (arches follow their body): %s" % step3.get("reason", ""))
        for r in step3.get("renamed", []) if step3.get("applied") else []:
            print("     arch %-4s -> %-4s %6.2f mL (%.0f%% attached below)"
                  % (r["arch"], r["to"], r["ml"], 100 * r["attached_below"]))
        if "arch_agreement_before" in step3:
            print("     arch/body agreement %.1f%% -> %.1f%%; size roughness %s -> %s"
                  % (100 * step3["arch_agreement_before"], 100 * step3["arch_agreement_after"],
                     step3["roughness_before"], step3["roughness_after"]))
        s3b = step3.get("step3b", {})
        print("   step 3b (fragments join their neighbour): %s" % s3b.get("reason", ""))
        for m in s3b.get("moved", []) if s3b.get("applied") else []:
            print("     fragment %-4s -> %-4s %5.2f mL" % (m["from"], m["to"], m["ml"]))
        print("   step 4 (integrity): %s" % ("OK" if not problems else "; ".join(problems)))


if __name__ == "__main__":
    main()
