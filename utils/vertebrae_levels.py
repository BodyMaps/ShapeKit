"""ShapeKit level-numbering vertebrae engine (ShapeKit-Levels).

Adapter between the ShapeKit pipeline (per-organ binary masks) and a
vertebra-level renumbering step, for the failure mode where the model
outlines the bone correctly but names the levels wrong.

Why another engine: a whole-spine model can collapse one vertebra (commonly
L1 at the thoracolumbar junction) into a sliver, after which every level
above it carries its neighbour's name. Mask-shape reasoning cannot fix that,
because each mask on its own looks like a healthy vertebra; only the ordering
of the column is wrong. This engine restores the numbering:

  1. Trace the vertebral-body axis through the column.
  2. Read the mask along it: the prediction has a gap at every intervertebral
     disc, which marks where one vertebra ends and the next begins.
  3. Number the vertebrae between those gaps consecutively with a dynamic
     programme that balances disc evidence, plausible level lengths, and
     agreement with the model's own names.
  4. Touch only the labels that disagree with that numbering. Their body
     voxels take the level of the disc interval they fall in; their
     posterior-element voxels grow out from there through bone, with the
     case CT (when available) keeping boundaries in the dark joint gaps and
     the model's own piece boundaries kept expensive to cross.

Labels consistent with the numbering are returned exactly as predicted, so on
a correctly numbered case the engine is a no-op.

Measured on the BodyMaps warm-up evaluation (AbdomenAtlasDemo, SuPreM
swin_unetr_totalsegmentator_vertebrae predictions), average DSC over the 24
vertebrae rose from 77.3% to 92.8%. The gain is confined to the mid spine,
where one case was shifted by one level: L1 24.2 -> 83.6, T12 20.3 -> 89.3,
T11 35.0 -> 91.9, T10 43.7 -> 89.7, T9 28.5 -> 87.2, T8 52.0 -> 88.3. Levels
that were already correct were left untouched and kept their scores.

Deployment notes:
  - No new dependencies: numpy, scipy, nibabel and scikit-image are already
    ShapeKit requirements.
  - CPU only. Work is confined to the spinal column's bounding box (about a
    tenth of a whole-spine volume), and the growth step runs in overlapping
    z slabs, so peak memory stays near 1 GB even for a 0.7 mm whole-spine
    case. Runtime there is about 40 s on 2 cores, a few seconds at 2.5 mm.
  - The CT is optional. Without it the engine still renumbers; only the
    posterior-element growth loses the joint-gap cue, so it is worth
    pointing ct_file_name / ct_root at the scans when they exist.
"""

import os

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


CLASS_MAP = {
    1: "vertebrae_L5", 2: "vertebrae_L4", 3: "vertebrae_L3", 4: "vertebrae_L2",
    5: "vertebrae_L1", 6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10",
    9: "vertebrae_T9", 10: "vertebrae_T8", 11: "vertebrae_T7", 12: "vertebrae_T6",
    13: "vertebrae_T5", 14: "vertebrae_T4", 15: "vertebrae_T3", 16: "vertebrae_T2",
    17: "vertebrae_T1", 18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5",
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2", 24: "vertebrae_C1",
}
FULL = np.ones((3, 3, 3), dtype=bool)

NOMINAL_PITCH_MM = np.array([40, 40, 40, 38, 35, 32, 29, 27, 25, 24, 23, 23, 23, 23,
                             23, 22, 23, 20, 18, 18, 18, 18, 35, 15], dtype=float)
AXIS_DISK_MM = 4.0
AGREE_WEIGHT = 0.5
OWN_SHARE = 0.6
LEAK_SHARE = 0.15
BODY_FRONT_MM = -3.0
BODY_HALF_WIDTH_MM = 22.0
CUT_MARGIN_MM = 1.5
BOUNDARY_COST = 400.0


def name(k):
    return CLASS_MAP[int(k)].replace("vertebrae_", "")


class RAS:
    """Reorients a volume to RAS (z = superior) and back again."""

    def __init__(self, img):
        self.src = nib.orientations.io_orientation(img.affine)
        ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
        self.fwd = nib.orientations.ornt_transform(self.src, ras)
        self.bwd = nib.orientations.ornt_transform(ras, self.src)
        zooms = np.array(img.header.get_zooms()[:3], dtype=float)
        self.spacing = np.empty(3)
        for i in range(3):
            self.spacing[int(self.src[i, 0])] = zooms[i]
        self.shape = img.shape[:3]

    def to(self, arr):
        return np.ascontiguousarray(nib.orientations.apply_orientation(arr, self.fwd))

    def back(self, arr):
        out = np.ascontiguousarray(nib.orientations.apply_orientation(arr, self.bwd))
        assert out.shape == self.shape
        return out


def spine_box(full, spacing, margin_mm=30.0, step=4):
    """Bounding box of the spinal column plus a margin, ignoring stray labels.

    A raw prediction can carry specks far from the spine. Taking the box from
    every labelled voxel then returns nearly the whole volume, and the distance
    transforms below would need several GB for a fine-cut scan. Measuring it
    from the substantial pieces instead keeps the whole column (which may be
    split at the discs) and leaves out the specks, which are far too small
    and too far from the column to be vertebrae.
    """
    small = full[::step, ::step, ::step] > 0
    if not small.any():
        return None
    cc, n = ndi.label(small, structure=FULL)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    # keep every piece of a serious size: a column split at the discs is still
    # the spine, while stray specks are a fraction of a percent of the largest.
    keep = np.nonzero(sizes >= 0.05 * sizes.max())[0]
    idx = np.nonzero(np.isin(cc, keep))
    pad = np.ceil(margin_mm / spacing).astype(int)
    return tuple(slice(max(0, int(a.min()) * step - int(p)), min(int(s), (int(a.max()) + 1) * step + int(p)))
                 for a, p, s in zip(idx, pad, full.shape))


def body_axis(lab, spacing):
    """Centre line of the vertebral bodies in mm, one point per axial slice."""
    Z = lab.shape[2]
    pts = np.full((Z, 2), np.nan)
    for z in range(Z):
        m = lab[:, :, z] > 0
        if m.sum() < 30:
            continue
        xs, ys = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        dt = ndi.distance_transform_edt(np.pad(m[x0:x1, y0:y1], 1), sampling=spacing[:2])[1:-1, 1:-1]
        dt[:, : (y1 - y0) // 2] = 0            # bodies sit in the anterior half (y = anterior)
        ix, iy = np.unravel_index(np.argmax(dt), dt.shape)
        if dt[ix, iy] >= 4.0:                   # an 8 mm inscribed circle: a body, not a process
            pts[z] = (x0 + ix, y0 + iy)
    ok = ~np.isnan(pts[:, 0])
    if ok.sum() < 10:
        return None
    zi = np.arange(Z)
    first, last = zi[ok][0], zi[ok][-1]
    x = np.interp(zi, zi[ok], pts[ok, 0])
    y = np.interp(zi, zi[ok], pts[ok, 1])
    k = max(3, int(round(15.0 / spacing[2])) | 1)
    x = ndi.gaussian_filter1d(ndi.median_filter(x, k, mode="nearest"), 4.0 / spacing[2])
    y = ndi.gaussian_filter1d(ndi.median_filter(y, k, mode="nearest"), 4.0 / spacing[2])
    return np.stack([x * spacing[0], y * spacing[1], zi * spacing[2]], 1)[first:last + 1]


def sample_axis(lab, spacing, ds=1.0):
    """Per mm of axis: position, tangent, and each label's share of a small disk."""
    P = body_axis(lab, spacing)
    if P is None:
        return None
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    su = np.arange(0, s[-1], ds)
    Pu = np.stack([np.interp(su, s, P[:, i]) for i in range(3)], 1)
    T = np.gradient(ndi.gaussian_filter1d(Pu, 6.0 / ds, axis=0), axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    U = np.array([1.0, 0, 0]) - T[:, :1] * T
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    g = np.arange(-AXIS_DISK_MM, AXIS_DISK_MM + 0.01, 1.0)
    A, B = np.meshgrid(g, g)
    keep = A ** 2 + B ** 2 <= AXIS_DISK_MM ** 2
    pts = Pu[:, None, :] + A[keep][None, :, None] * U[:, None, :] + B[keep][None, :, None] * V[:, None, :]
    vox = np.rint(pts / spacing).astype(int)
    for i in range(3):
        np.clip(vox[..., i], 0, lab.shape[i] - 1, out=vox[..., i])
    lb = lab[vox[..., 0], vox[..., 1], vox[..., 2]]
    votes = np.stack([(lb == k).mean(1) for k in range(25)], 1)
    return {"s": su, "P": Pu, "T": T, "votes": votes}


def number_levels(votes, ds=1.0):
    """Dynamic programme: one cut per disc gap, consecutive level numbers.

    Score = disc evidence at each cut + AGREE_WEIGHT * share of each label's
    votes inside the interval given that label. Level lengths stay within a
    plausible range around the nominal pitch, scaled to the patient.
    Returns [(label, start_mm, end_mm)] from inferior to superior.
    """
    N = len(votes)
    occ = 1.0 - votes[:, 0]
    gap = np.clip(1.0 - ndi.uniform_filter1d(occ, 3), 0, 1)
    C = np.vstack([np.zeros((1, 25)), np.cumsum(votes, 0)])
    tot = C[-1] + 1e-6
    ratios = []
    for k in range(1, 25):
        idx = np.nonzero(votes[:, k] > 0.5)[0]
        if idx.size > 3:
            ratios.append((idx[-1] - idx[0] + 1) * ds / NOMINAL_PITCH_MM[k - 1])
    scale = float(np.clip(np.median(ratios), 0.6, 1.5)) if ratios else 1.0

    NEG = -1e18
    F = np.full((25, N + 1), NEG)
    back = np.full((25, N + 1), -1, dtype=np.int64)
    for k in range(1, 25):
        lo = max(1, int(0.55 * scale * NOMINAL_PITCH_MM[k - 1] / ds))
        hi = int(np.ceil(1.7 * scale * NOMINAL_PITCH_MM[k - 1] / ds))
        agree = AGREE_WEIGHT * (C[:, k] / tot[k])
        for b in range(1, N + 1):
            best, arg = NEG, -1
            a0, a1 = max(0, b - hi), b - lo
            if k > 1 and a1 >= a0:                     # continue from level k-1
                cand = F[k - 1, a0:a1 + 1] + (agree[b] - agree[a0:a1 + 1])
                j = int(np.argmax(cand))
                best, arg = cand[j], a0 + j
            a_start = max(0, b - hi)                   # or level k is the first in the scan
            cand = agree[b] - agree[a_start:b]
            j = int(np.argmax(cand))
            if cand[j] > best:
                best, arg = cand[j], -(a_start + j) - 2
            F[k, b] = best + (gap[b - 1] if b < N else 0.0)
            back[k, b] = arg
    k_end, b_end = np.unravel_index(np.argmax(F[1:, 1:]), (24, N))
    k, b = int(k_end) + 1, int(b_end) + 1
    intervals = []
    while True:
        a = int(back[k, b])
        if a <= -2:
            intervals.append((k, -a - 2, b))
            break
        intervals.append((k, a, b))
        k, b = k - 1, a
    intervals = intervals[::-1]
    cuts = []                                           # centre each cut in its disc gap
    for _, _, c in intervals[:-1]:
        i = min(max(c - 1, 0), N - 1)
        if gap[i] > 0.5:
            l = i
            while l > 0 and gap[l - 1] > 0.5:
                l -= 1
            r = i
            while r < N - 1 and gap[r + 1] > 0.5:
                r += 1
            cuts.append((l + r + 1) / 2.0)
        else:
            cuts.append(float(c))
    starts = [0.0] + cuts                               # end levels run to the ends of the axis
    ends = cuts + [float(N)]
    return [(k, a * ds, b * ds) for (k, _, _), a, b in zip(intervals, starts, ends)]


def inconsistent_labels(votes, levels):
    """Labels whose extent along the axis disagrees with the numbering."""
    atlas_axis = {23: 24, 24: 23}   # C1 has no body; its ring surrounds the dens of C2
    inside = np.zeros(25)
    bad = set()
    for k, a, b in levels:
        seg = votes[int(a):int(np.ceil(b))].sum(0)
        seg[0] = 0
        inside[k] += seg[k]
        if k in atlas_axis:
            inside[atlas_axis[k]] += seg[atlas_axis[k]]
            seg[atlas_axis[k]] = 0
        if seg.sum() > 0 and (seg.argmax() != k or seg[k] < OWN_SHARE * seg.sum()):
            bad.add(int(k))
    total = votes.sum(0)
    for j in range(1, 25):
        if total[j] > 0 and (total[j] - inside[j]) > LEAK_SHARE * total[j]:
            bad.add(j)
    return bad


def body_zone_levels(lab, spacing, axis, levels):
    """Level of each vertebral-body voxel from its disc interval (0 elsewhere)."""
    s, P, T = axis["s"], axis["P"], axis["T"]
    U = np.array([1.0, 0, 0]) - T[:, :1] * T
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)                                  # points anteriorly
    idx = np.nonzero(lab > 0)
    pts = np.stack(idx, 1) * spacing
    nn = cKDTree(P).query(pts)[1]
    rel = pts - P[nn]
    sproj = s[nn] + (rel * T[nn]).sum(1)
    u = (rel * U[nn]).sum(1)
    v = (rel * V[nn]).sum(1)
    ks = np.array([k for k, _, _ in levels])
    a_ = np.array([a for _, a, _ in levels], dtype=float)
    b_ = np.array([b for _, _, b in levels], dtype=float)
    a_[0], b_[-1] = -np.inf, np.inf
    j = np.clip(np.searchsorted(b_, sproj), 0, len(levels) - 1)
    zone = ((sproj >= a_[j]) & (sproj < b_[j]) & (v >= BODY_FRONT_MM) &
            (np.abs(u) <= BODY_HALF_WIDTH_MM) &
            (np.minimum(sproj - a_[j], b_[j] - sproj) >= CUT_MARGIN_MM))
    body = np.zeros(lab.shape, dtype=np.uint8)
    body[tuple(c[zone] for c in idx)] = ks[j[zone]]
    return body


def label_boundary(lab):
    """Voxels touching a different (nonzero) label along an axis."""
    b = np.zeros(lab.shape, dtype=bool)
    for ax in range(3):
        a = np.moveaxis(lab, ax, 0)
        diff = (a[1:] != a[:-1]) & (a[1:] > 0) & (a[:-1] > 0)
        bb = np.moveaxis(b, ax, 0)
        bb[1:] |= diff
        bb[:-1] |= diff
    return b


def grow_labels(markers, mask, ct, boundary, spacing, slab_mm=150.0, overlap_mm=50.0):
    """Marker watershed in overlapping z slabs, so memory stays bounded."""
    from skimage.segmentation import watershed
    out = np.zeros(mask.shape, dtype=np.uint8)
    Z = mask.shape[2]
    slab = max(1, int(slab_mm / spacing[2]))
    ov = int(overlap_mm / spacing[2])
    for z0 in range(0, Z, slab):
        a, b = max(0, z0 - ov), min(Z, z0 + slab + ov)
        if ct is not None:
            # flat inside bone so seeds compete by distance; cost only in dark gaps
            land = np.clip(250.0 - ndi.gaussian_filter(ct[:, :, a:b].astype(np.float32), 0.8), 0, 400)
        else:
            land = np.zeros((mask.shape[0], mask.shape[1], b - a), dtype=np.float32)
        land = land + BOUNDARY_COST * boundary[:, :, a:b]
        w = watershed(land, markers=markers[:, :, a:b].astype(np.int32),
                      mask=mask[:, :, a:b], connectivity=1)
        n = min(slab, Z - z0)
        out[:, :, z0:z0 + n] = w[:, :, z0 - a:z0 - a + n]
    miss = mask & (out == 0)
    if miss.any():
        _, ind = ndi.distance_transform_edt(out == 0, sampling=spacing, return_indices=True)
        out[miss] = out[tuple(i[miss] for i in ind)]
    return out


def renumber_levels(lab, ct, spacing, log):
    """Renumber vertebrae whose names disagree with the disc-gap numbering."""
    axis = sample_axis(lab, spacing)
    if axis is None:
        log.append({"step": "levels", "skipped": "no vertebral body axis found"})
        return lab
    levels = number_levels(axis["votes"])
    bad = inconsistent_labels(axis["votes"], levels)
    entry = {"step": "levels", "inconsistent": [name(k) for k in sorted(bad)]}
    if not bad:
        entry["voxels_relabelled"] = 0
        log.append(entry)
        return lab
    body = body_zone_levels(lab, spacing, axis, levels)
    in_bad = np.isin(lab, sorted(bad))
    fixed = np.where(in_bad, 0, lab).astype(np.uint8)
    markers = np.where(fixed > 0, fixed, np.where(in_bad, body, 0)).astype(np.uint8)
    grown = grow_labels(markers, lab > 0, ct, label_boundary(lab), spacing)
    out = np.where(fixed > 0, fixed, grown).astype(np.uint8)
    entry["voxels_relabelled"] = int(np.count_nonzero(out != lab))
    moved = []
    for k in sorted(bad):
        vals, counts = np.unique(out[lab == k], return_counts=True)
        share = counts / max(counts.sum(), 1)
        parts = [f"{name(v)} {100 * c:.0f}%" for v, c in zip(vals, share) if v and c >= 0.05]
        moved.append(f"{name(k)} -> {', '.join(parts)}")
    entry["relabelled"] = moved
    log.append(entry)
    return out


# ------------------------------------------------------------------ adapter

VERTEBRA_NAMES = [CLASS_MAP[i] for i in range(1, 25)]


def _load_ct(ct_path, reference_img, shape, patient_id, logger):
    """Takes: the case CT path, the reference image every mask was reoriented
        to, the mask shape, patient id and logger.
    Does: loads the CT and puts it on the mask grid, so a CT voxel and a mask
        voxel at the same index are the same place in the patient.
    Returns: the CT array, or None when it is missing, unreadable or a
        different grid (the engine then runs without it)."""
    if not ct_path or not os.path.exists(ct_path):
        logger.info(f"[ShapeKit-Levels] {patient_id}: no CT at {ct_path}; "
                    f"renumbering without the joint-gap cue")
        return None
    try:
        ct_img = nib.load(ct_path)
        transform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(ct_img.affine),
            nib.orientations.axcodes2ornt(nib.aff2axcodes(reference_img.affine)))
        ct = nib.orientations.apply_orientation(np.asanyarray(ct_img.dataobj), transform)
    except Exception as e:  # a batch run must not stop for one unreadable CT
        logger.warning(f"[ShapeKit-Levels] {patient_id}: CT unreadable ({e}); "
                       f"renumbering without the joint-gap cue")
        return None
    if ct.shape != shape:
        logger.warning(f"[ShapeKit-Levels] {patient_id}: CT grid {ct.shape} does not "
                       f"match the masks {shape}; renumbering without the joint-gap cue")
        return None
    return ct


def postprocessing_vertebrae_levels(patient_id, segmentation_dict,
                                    reference_img, ct_path, logger):
    """Takes: patient id, the ShapeKit segmentation dict (organ name -> binary
        mask, all on the reference image's axcodes), the reference image, the
        case CT path, and a logger.
    Does: assembles the vertebra masks into one labelled volume, renumbers the
        levels that disagree with the disc-gap numbering, and writes the
        result back into the dict. Levels that agree are returned unchanged.
    Returns: the segmentation dict with the vertebra masks replaced."""
    present = [n for n in VERTEBRA_NAMES
               if segmentation_dict.get(n) is not None and np.any(segmentation_dict[n])]
    if len(present) < 3:
        logger.info(f"[ShapeKit-Levels] {patient_id}: {len(present)} vertebra masks "
                    f"present, nothing to renumber")
        return segmentation_dict

    shape = segmentation_dict[present[0]].shape
    labels = np.zeros(shape, dtype=np.uint8)
    for k in range(1, 25):
        mask = segmentation_dict.get(CLASS_MAP[k])
        if mask is not None:
            labels[mask > 0] = k

    ct = _load_ct(ct_path, reference_img, shape, patient_id, logger)

    # The steps below read the anatomy off the axes, so work in RAS and put
    # the result back on the caller's axcodes afterwards.
    ras = RAS(reference_img)
    labels = ras.to(labels)
    spacing = ras.spacing
    box = spine_box(labels, spacing)
    if box is None:
        logger.info(f"[ShapeKit-Levels] {patient_id}: empty vertebrae mask")
        return segmentation_dict

    log = []
    crop = labels[box].copy()
    ct_crop = ras.to(ct)[box].astype(np.float32) if ct is not None else None
    renumbered = renumber_levels(crop, ct_crop, spacing, log)
    del ct_crop
    entry = log[-1]
    if entry.get("voxels_relabelled"):
        labels[box] = renumbered
        logger.info(f"[ShapeKit-Levels] {patient_id}: renumbered "
                    f"{entry['inconsistent']} ({entry['voxels_relabelled']} voxels); "
                    f"{'; '.join(entry.get('relabelled', []))}")
    else:
        logger.info(f"[ShapeKit-Levels] {patient_id}: numbering already consistent, "
                    f"masks unchanged")
        return segmentation_dict
    labels = ras.back(labels)

    for k in range(1, 25):
        name_k = CLASS_MAP[k]
        if segmentation_dict.get(name_k) is not None or np.any(labels == k):
            segmentation_dict[name_k] = (labels == k).astype(np.uint8)
    return segmentation_dict
