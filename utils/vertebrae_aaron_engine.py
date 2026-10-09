"""Automatic postprocessing of AI-predicted vertebrae masks (SuPreM warm-up).

The raw predictions (``combined_labels.nii.gz``, labels 1=L5 ... 24=C1) suffer from:
  1. false-positive fragments far away from the spine (e.g. "C1" blobs in the femur);
  2. mixed labels inside a single vertebra (body split between two labels, small
     islands of a neighboring label);
  3. identification errors: a run of vertebrae shifted by one level, a label
     duplicated on two vertebrae and another one skipped, so that the labels no
     longer follow the anatomical order L5 < L4 < ... < T1 < C7 < ... < C1;
  4. holes inside vertebrae.

Pipeline (uses only the label map, no CT intensities required):
  A. Keep the spine: starting from the largest component, keep every component of
     at least 2 ml within 10 mm of what is already kept; drop the rest (small
     and/or far away fragments).
  B. Vertebral-body cores: erode the binary spine mask by ~5 mm. This removes the
     thin posterior elements and the contact at the discs/facets, leaving one
     compact core per vertebral body. Tiny cores (fragments) are discarded.
     Cores still holding two bodies (thin discs, thick slices) are split by extra
     erosion; near-duplicate cores are merged.
  C. Anatomical identification: cores are sorted along the cranio-caudal axis and
     assigned strictly increasing labels by dynamic programming. The score rewards
     agreement with the AI votes (inside the body core and over the whole vertebra)
     and penalizes label jumps that disagree with the measured distance between
     cores (relative to the local inter-vertebral spacing) as well as ignoring a
     core. This enforces a consistent L5 -> C1 sequence and fixes shifted runs.
  D. Relabeling: each reliable core is grown back to its vertebral body (the
     morphological opening of the mask) and the body gets one single label. Every
     posterior-element voxel then takes the label of the body it connects to: the
     body reached by the cheapest path inside the mask, where paths are expensive
     through thin bone (facet joints, touching processes) and across AI label
     boundaries. Small cores (cervical spine) and implausibly large ones keep the
     AI labels, together with the arches connected to them.
  E. Per-label cleanup: keep the connected piece holding the identified body (else
     the largest), re-assign detached pieces to the touching neighbor, fill holes.

Usage:
    python postprocessing_vertebrae.py --input AbdomenAtlasDemoPredict --output OUT_DIR

For every case the script writes ``combined_labels.nii.gz``,
``segmentations/vertebrae_*.nii.gz`` and a short ``postprocessing_log.json``
describing what was changed to OUT_DIR/<case>; the input is never modified.
"""
import argparse
import json
import os
import shutil

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed

CLASS_MAP = {
    1: "vertebrae_L5", 2: "vertebrae_L4", 3: "vertebrae_L3", 4: "vertebrae_L2", 5: "vertebrae_L1",
    6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10", 9: "vertebrae_T9", 10: "vertebrae_T8",
    11: "vertebrae_T7", 12: "vertebrae_T6", 13: "vertebrae_T5", 14: "vertebrae_T4", 15: "vertebrae_T3",
    16: "vertebrae_T2", 17: "vertebrae_T1", 18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5",
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2", 24: "vertebrae_C1",
}
NUM_LABELS = 24
KEEP_AI = NUM_LABELS + 1   # seed meaning "leave the AI label"

# ---- parameters (mm / ml) ----
KEEP_MIN_ML = 2.0          # detached components smaller than this are removed ...
KEEP_MAX_DIST_MM = 10.0    # ... and larger ones are kept only if this close to the main spine
CORE_ERODE_MM = 5.0        # erosion radius that separates vertebral bodies
CORE_MIN_ML = 0.3          # absolute minimum core size
MARKER_MIN_ML = 1.0        # cores smaller than this do not seed a vertebra (AI labels are kept)
OVERSIZE_FACTOR = 1.8      # cores this much larger than their neighbors are not trusted
DUPLICATE_FRAC = 0.4       # cores closer than this fraction of the spacing are duplicates
SPLIT_EXTRA_MM = (1.0, 2.0, 3.0)  # extra erosion tried to split merged bodies
SPLIT_MIN_FRAC = 0.3       # each split piece must be >= this fraction of the largest one
CORE_REL_SIZE = 0.25       # core must be >= this fraction of the local median core size
VOTE_RADIUS_FACTOR = 1.0   # voxels within this many inter-vertebral spacings of a core vote
STEP_PENALTY = 1.0         # DP penalty per vertebra of mismatch between label step and geometry
DROP_PENALTY = 3.0         # DP penalty for ignoring a (full-size) core
ARCH_BOUNDARY_COST = 20.0  # arch path cost multiplier on AI label boundaries
ARCH_NECK_MM = 4.0         # arch path cost grows as (ARCH_NECK_MM / bone half-thickness)^2
MIN_PIECE_ML = 0.05        # detached pieces of a label smaller than this are deleted if isolated


def ball(radius_mm, zooms):
    rad = np.maximum(np.round(radius_mm / np.asarray(zooms)).astype(int), 1)
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in rad)]
    return sum((g * z / radius_mm) ** 2 for g, z in zip(grids, zooms)) <= 1.0 + 1e-6


# ----------------------------------------------------------------------------- A
def keep_spine(lab, zooms, log):
    """Remove false-positive fragments that are not part of the spinal column."""
    vox_ml = np.prod(zooms) / 1000.0
    cc, n = cc3d.connected_components(lab > 0, connectivity=26, return_N=True)
    if n <= 1:
        return lab
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    main = sizes.argmax()
    keep = np.zeros(n + 1, bool)
    keep[main] = True
    stats = cc3d.statistics(cc)
    margin = np.ceil(KEEP_MAX_DIST_MM / np.asarray(zooms)).astype(int)
    # grow the kept set as a chain, so a spine broken into several pieces (e.g. an
    # unlabeled disc) is kept whole, not only the pieces next to the largest one
    candidates = [i for i in range(1, n + 1)
                  if i != main and sizes[i] * vox_ml >= KEEP_MIN_ML]
    added = True
    while added:
        added = False
        for i in candidates:
            if keep[i]:
                continue
            sl = tuple(slice(max(s.start - m, 0), s.stop + m)
                       for s, m in zip(stats["bounding_boxes"][i], margin))
            near = keep[cc[sl]]
            if not near.any():
                continue
            d = ndi.distance_transform_edt(~near, sampling=zooms)[cc[sl] == i].min()
            if d <= KEEP_MAX_DIST_MM:
                keep[i] = added = True
    removed = int(np.count_nonzero(~keep[1:] & (sizes[1:] > 0)))
    out = np.where(keep[cc], lab, 0).astype(np.uint8)
    log["fragments_removed"] = removed
    log["fragment_voxels_removed"] = int(np.count_nonzero(lab) - np.count_nonzero(out))
    return out


# ----------------------------------------------------------------------------- B
def find_cores(lab, zooms):
    """Return a core instance map and per-core info (position, size, local spacing)."""
    vox_ml = np.prod(zooms) / 1000.0
    eroded = ndi.binary_erosion(lab > 0, ball(CORE_ERODE_MM, zooms))
    cc, n = cc3d.connected_components(eroded, connectivity=6, return_N=True)
    if n == 0:
        return cc, []
    stats = cc3d.statistics(cc)
    cores = []
    for i in range(1, n + 1):
        size_ml = stats["voxel_counts"][i] * vox_ml
        if size_ml < CORE_MIN_ML:
            continue
        cores.append(dict(id=i, z=float(stats["centroids"][i][2]), size=size_ml,
                          pos=np.asarray(stats["centroids"][i]) * zooms))
    cores.sort(key=lambda c: c["z"])
    # discard fragments much smaller than their neighbors (split bodies, osteophytes...)
    sizes = np.array([c["size"] for c in cores])
    kept = []
    for k, c in enumerate(cores):
        local = np.median(sizes[max(0, k - 3):k + 4])
        if c["size"] >= CORE_REL_SIZE * local:
            c["rel_size"] = min(1.0, c["size"] / local)
            kept.append(c)
    keep_ids = {c["id"] for c in kept}
    cc = np.where(np.isin(cc, list(keep_ids)), cc, 0)
    kept = split_merged_cores(cc, kept, zooms)
    kept.sort(key=lambda c: c["z"])
    kept = drop_duplicate_cores(cc, kept)
    # expected inter-vertebral spacing: local median distance between consecutive cores
    pos = np.array([c["pos"] for c in kept])
    gaps = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(kept) > 1 else np.array([30.0])
    for k, c in enumerate(kept):
        c["spacing"] = float(np.median(gaps[max(0, k - 3):k + 3]))
    return cc, kept


def drop_duplicate_cores(cc, cores):
    """Two cores much closer than the vertebral spacing belong to the same vertebra:
    keep the larger one (the other is a fragment, e.g. left/right halves of a body)."""
    changed = True
    while changed and len(cores) > 2:
        changed = False
        pos = np.array([c["pos"] for c in cores])
        gaps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        spacing = np.median(gaps)
        k = int(np.argmin(gaps))
        if gaps[k] < DUPLICATE_FRAC * spacing:
            small = k if cores[k]["size"] < cores[k + 1]["size"] else k + 1
            cc[cc == cores[small]["id"]] = 0
            del cores[small]
            changed = True
    return cores


def split_merged_cores(cc, cores, zooms):
    """Split cores that still contain two vertebral bodies (thin disc not opened).

    A core is eroded a little further; if it falls apart into two substantial pieces
    lying one above the other (centroids >= half an inter-vertebral spacing apart),
    it is replaced by these pieces. ``cc`` is modified in place.
    """
    vox_ml = np.prod(zooms) / 1000.0
    pos = np.array([c["pos"] for c in cores])
    spacing = np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)) if len(cores) > 1 else 0
    next_id = int(cc.max()) + 1
    out = []
    for c in cores:
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1)
                   for s in ndi.find_objects((cc == c["id"]).astype(np.uint8))[0])
        core = cc[sl] == c["id"]
        pieces = None
        for extra in SPLIT_EXTRA_MM:
            e = ndi.binary_erosion(core, ball(extra, zooms))
            pc, n = cc3d.connected_components(e, connectivity=6, return_N=True)
            if n < 2:
                continue
            sizes = np.bincount(pc.ravel())[1:]
            big = np.nonzero(sizes >= SPLIT_MIN_FRAC * sizes.max())[0] + 1
            if len(big) < 2:
                continue
            cents = np.array([np.argwhere(pc == b).mean(0) * zooms for b in big])
            if np.linalg.norm(cents.max(0) - cents.min(0)) >= 0.5 * spacing:
                pieces = (pc, big)
                break
        if pieces is None:
            out.append(c)
            continue
        pc, big = pieces
        # grow the pieces back inside the original core
        grown = watershed(-ndi.distance_transform_edt(core, sampling=zooms),
                          markers=np.where(np.isin(pc, big), pc, 0), mask=core)
        sub = cc[sl]
        for b in big:
            m = grown == b
            sub[m] = next_id
            idx = np.argwhere(m)
            cen = (idx.mean(0) + [s.start for s in sl])
            out.append(dict(id=next_id, z=float(cen[2]), size=m.sum() * vox_ml, pos=cen * zooms,
                            rel_size=c["rel_size"], split_from=c["id"]))
            next_id += 1
    return out


def vote(lab, zooms, core_cc, cores):
    """AI label votes for each core: average of the votes inside the body core and the
    votes over the whole vertebra (body + posterior elements).

    Each spine voxel is attached to a core by watershed on the distance map; voxels
    farther than VOTE_RADIUS_FACTOR * spacing from the cores (e.g. the neck above the
    top core) do not vote.
    """
    mask = lab > 0
    dist = ndi.distance_transform_edt(mask, sampling=zooms)
    inst = watershed(-dist, markers=core_cc.astype(np.int32), mask=mask, connectivity=1)
    d_core = ndi.distance_transform_edt(core_cc == 0, sampling=zooms)
    radius = np.zeros(int(core_cc.max()) + 1)
    for c in cores:
        radius[c["id"]] = VOTE_RADIUS_FACTOR * c["spacing"]
    near = mask & (d_core <= radius[inst])
    n_ids = int(core_cc.max()) + 1
    hist = np.zeros((n_ids, NUM_LABELS + 1))
    np.add.at(hist, (inst[near], lab[near]), 1)
    in_core = core_cc > 0
    hist_core = np.zeros((n_ids, NUM_LABELS + 1))
    np.add.at(hist_core, (core_cc[in_core], lab[in_core]), 1)
    for c in cores:
        v = hist[c["id"], 1:]
        vc = hist_core[c["id"], 1:]
        c["votes"] = 0.5 * v / max(v.sum(), 1) + 0.5 * vc / max(vc.sum(), 1)


# ----------------------------------------------------------------------------- C
def identify(cores):
    """Assign strictly increasing labels (inferior -> superior) to cores by DP.

    Score = sum of vote fractions of the assigned labels
            - STEP_PENALTY * |label step - geometric step| for consecutive kept cores,
              where the geometric step is the 3D distance between the two cores divided
              by the local inter-vertebral spacing (a gap of ~2 spacings means one
              vertebra is missing between them, ~1 spacing means direct neighbors);
            - DROP_PENALTY * relative size, for every core that is ignored.
    Unused labels before the first / after the last kept core are free (limited FOV).
    """
    n = len(cores)
    if n == 0:
        return []
    L = NUM_LABELS
    NEG = -1e9
    drop = np.array([DROP_PENALTY * c["rel_size"] for c in cores])
    cum_drop = np.concatenate([[0.0], np.cumsum(drop)])  # cost of dropping cores [a, b)
    # best[i][l]: best score of a solution whose last kept core is i with label l
    best = np.full((n, L + 1), NEG)
    back = {}
    for i in range(n):
        v = cores[i]["votes"]
        for l in range(1, L + 1):
            # start a new sequence at core i (all previous cores dropped)
            cand = v[l - 1] - cum_drop[i]
            arg = None
            for j in range(i):
                spacing = 0.5 * (cores[i]["spacing"] + cores[j]["spacing"])
                geo = np.linalg.norm(cores[i]["pos"] - cores[j]["pos"]) / max(spacing, 1e-3)
                for lp in range(1, l):
                    if best[j, lp] <= NEG / 2:
                        continue
                    s = (best[j, lp] + v[l - 1]
                         - STEP_PENALTY * abs((l - lp) - geo)
                         - (cum_drop[i] - cum_drop[j + 1]))
                    if s > cand:
                        cand, arg = s, (j, lp)
            best[i, l] = cand
            back[(i, l)] = arg
    # close: remaining cores after the last kept one are dropped
    final = best - (cum_drop[n] - cum_drop[1:])[:, None]
    i, l = np.unravel_index(np.argmax(final), final.shape)
    assign = [0] * n
    state = (int(i), int(l))
    while state is not None:
        assign[state[0]] = state[1]
        state = back[state]
    return assign


# ----------------------------------------------------------------------------- D
def relabel_vertebrae(lab, zooms, core_cc, cores, assign, log):
    """Give every vertebral body the label found by the identification, then give each
    posterior-element voxel the label of the body it is attached to.

    The body region is the morphological opening of the spine mask (the cores grown
    back by the erosion radius); it is split between cores by watershed.
    """
    mask = lab > 0
    markers = np.zeros(lab.shape, np.int32)
    changes, skipped = [], []
    sizes = np.array([c["size"] for c in cores])
    for k, (c, a) in enumerate(zip(cores, assign)):
        # tiny cores (cervical, thin slices) help the identification but are not
        # reliable enough to define a vertebra on their own
        if a == 0 or c["size"] < MARKER_MIN_ML:
            continue
        # a core much larger than its neighbors probably still spans two bodies
        neigh = np.concatenate([sizes[max(0, k - 2):k], sizes[k + 1:k + 3]])
        if neigh.size and c["size"] > OVERSIZE_FACTOR * np.median(neigh):
            skipped.append(f"{CLASS_MAP[a]} (core {c['size']:.1f} ml vs neighbors "
                           f"{np.median(neigh):.1f} ml)")
            continue
        markers[core_cc == c["id"]] = a
        raw = int(np.argmax(c["votes"])) + 1
        if raw != a:
            changes.append(f"{CLASS_MAP[raw]} -> {CLASS_MAP[a]} (body core at z={c['z_mm']:.0f} mm, "
                           f"AI agreement {c['votes'][raw - 1]:.2f})")
    log["relabeled_vertebrae"] = changes
    log["cores_not_used_oversized"] = skipped
    if not markers.any():
        return lab, markers
    # body region = cores grown back by the erosion radius (morphological opening)
    body = mask & (ndi.distance_transform_edt(markers == 0, sampling=zooms)
                   <= CORE_ERODE_MM + float(np.max(zooms)))
    dist = ndi.distance_transform_edt(mask, sampling=zooms)
    inst = watershed(-dist, markers=markers, mask=body, connectivity=1)
    out = lab.copy()
    sel = inst > 0
    out[sel] = inst[sel]
    log["body_voxels_relabeled"] = int(np.count_nonzero(out != lab))

    # Posterior elements go to the body they connect to through the pedicles: the body
    # reached by the cheapest path inside the mask. Height is no guide (thoracic spinous
    # processes slope down to the level of the vertebra below), and neither is the AI
    # label alone (it is shifted with the bodies). Paths are made expensive through thin
    # bone and across AI label boundaries, so they cannot slip into a neighboring arch
    # via a facet joint or a touching process. Bodies of unused cores are seeds too, so
    # the arches they reach keep their AI label. So are AI labels without an identified
    # body (e.g. a vertebra cut off at the edge of the scan has no usable core); without
    # this, the whole vertebra would flow to the nearest seeded body.
    seeds = inst.copy()
    unused = (ndi.distance_transform_edt((core_cc == 0) | sel, sampling=zooms)
              <= CORE_ERODE_MM + float(np.max(zooms)))
    seeds[mask & unused & ~sel] = KEEP_AI
    orphan = mask & ~sel & ~np.isin(lab, np.unique(inst[sel]))
    seeds[orphan] = KEEP_AI
    arch = mask & ~sel
    cost = np.where(mask, 1.0 + (ARCH_NECK_MM / np.maximum(dist, 1e-3)) ** 2, np.inf)
    cost[arch & label_boundary(np.where(arch, lab, 0))] *= ARCH_BOUNDARY_COST
    nearest = geodesic_nearest(seeds, cost, zooms)
    arch &= (nearest > 0) & (nearest != KEEP_AI)
    log["arch_voxels_relabeled"] = int(np.count_nonzero(nearest[arch] != lab[arch]))
    out[arch] = nearest[arch]
    return out, markers


def label_boundary(lab):
    """Labeled voxels with a 26-neighbor that carries a different non-zero label."""
    hi = ndi.maximum_filter(lab, size=3)
    lo = ndi.minimum_filter(np.where(lab > 0, lab, 255).astype(np.uint8), size=3)
    return (lab > 0) & ((hi != lab) | (lo != lab))


def geodesic_nearest(seeds, cost, zooms):
    """Label each voxel with the seed reached by the cheapest path (cost per mm traveled)."""
    mcp = MCP_Geometric(cost, sampling=tuple(zooms), fully_connected=True)
    _, tb = mcp.find_costs(np.argwhere(seeds > 0))
    # tb holds, for each reached voxel, the offset index of its predecessor on the
    # shortest path (-1 at seeds, -2 if unreached); follow the chains to their seed by
    # pointer jumping
    tb = tb.ravel()
    ids = np.flatnonzero(tb >= -1)
    pos = np.full(tb.size, -1, np.int64)
    pos[ids] = np.arange(ids.size)
    shape = cost.shape
    flat_off = np.asarray(mcp.offsets) @ np.array([shape[1] * shape[2], shape[2], 1])
    t = tb[ids]
    pred = np.arange(ids.size)
    m = t >= 0
    pred[m] = pos[ids[m] - flat_off[t[m]]]
    while True:
        nxt = pred[pred]
        if np.array_equal(nxt, pred):
            break
        pred = nxt
    out = np.zeros(shape, seeds.dtype)
    out.ravel()[ids] = seeds.ravel()[ids[pred]]
    return out


# ----------------------------------------------------------------------------- E
def cleanup_labels(lab, zooms, log, markers=None):
    """One connected piece per label; detached pieces go to the touching neighbor.

    The piece kept for a label is the one holding its identified body core (if any),
    otherwise the largest one.
    """
    vox_ml = np.prod(zooms) / 1000.0
    reassigned = 0
    for k in range(1, NUM_LABELS + 1):
        m = lab == k
        if not m.any():
            continue
        cc, n = cc3d.connected_components(m, connectivity=26, return_N=True)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        main = sizes.argmax()
        if markers is not None:
            on_core = np.bincount(cc[markers == k], minlength=n + 1)
            on_core[0] = 0
            if on_core.any():
                main = on_core.argmax()
        stats = cc3d.statistics(cc)
        for i in range(1, n + 1):
            if i == main or sizes[i] == 0:
                continue
            sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in stats["bounding_boxes"][i])
            piece = cc[sl] == i
            ring = ndi.binary_dilation(piece, np.ones((3, 3, 3), bool)) & ~piece
            neigh = lab[sl][ring]
            neigh = neigh[(neigh > 0) & (neigh != k)]
            sub = lab[sl]
            if neigh.size:
                sub[piece] = np.bincount(neigh).argmax()
                reassigned += 1
            elif sizes[i] * vox_ml < MIN_PIECE_ML:
                sub[piece] = 0
    # fill enclosed holes (background voxels fully surrounded by a single vertebra)
    filled = 0
    for k in range(1, NUM_LABELS + 1):
        m = lab == k
        if not m.any():
            continue
        sl = ndi.find_objects(m.astype(np.uint8))[0]
        sub = lab[sl]
        holes = ndi.binary_fill_holes(sub == k) & (sub == 0)
        filled += int(holes.sum())
        sub[holes] = k
    log["pieces_reassigned"] = reassigned
    log["hole_voxels_filled"] = filled
    return lab


# -----------------------------------------------------------------------------
def postprocess(lab, zooms, affine):
    """Full pipeline on a label volume in RAS-like orientation (axis 2 = superior)."""
    log = {"units": {"z_mm": "superior (S) scanner coordinate in mm, as shown in 3D Slicer",
                     "size_ml": "volume of the eroded body core in ml"}}
    lab = keep_spine(lab, zooms, log)
    if not lab.any():
        return lab, log
    # crop to the spine for speed
    bbox = ndi.find_objects((lab > 0).astype(np.uint8))[0]
    pad = 3
    bbox = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, dim)) for s, dim in zip(bbox, lab.shape))
    sub = lab[bbox].copy()
    core_cc, cores = find_cores(sub, zooms)
    if cores:
        vote(sub, zooms, core_cc, cores)
    assign = identify(cores)
    offset = np.array([s.start for s in bbox])
    for c in cores:
        c["z_mm"] = float(nib.affines.apply_affine(affine, c["pos"] / zooms + offset)[2])
    log["cores"] = [dict(z_mm=round(c["z_mm"], 1), size_ml=round(c["size"], 2),
                         ai_label=CLASS_MAP[int(np.argmax(c["votes"])) + 1],
                         assigned=CLASS_MAP.get(a, "dropped")) for c, a in zip(cores, assign)]
    sub, markers = relabel_vertebrae(sub, zooms, core_cc, cores, assign, log)
    sub = cleanup_labels(sub, zooms, log, markers)
    out = np.zeros_like(lab)
    out[bbox] = sub
    return out, log


def process_case(case_dir, out_dir):
    path = os.path.join(case_dir, "combined_labels.nii.gz")
    img = nib.load(path)
    # work in closest-canonical (RAS) orientation so that axis 2 is inferior->superior
    can = nib.as_closest_canonical(img)
    lab = np.asarray(can.dataobj).astype(np.uint8)
    zooms = np.asarray(can.header.get_zooms()[:3], float)
    before = np.bincount(lab.ravel(), minlength=NUM_LABELS + 1)

    new, log = postprocess(lab, zooms, can.affine)

    # back to the original orientation
    ornt = nib.orientations.ornt_transform(nib.orientations.io_orientation(can.affine),
                                           nib.orientations.io_orientation(img.affine))
    new = nib.orientations.apply_orientation(new, ornt).astype(np.uint8)
    after = np.bincount(new.ravel(), minlength=NUM_LABELS + 1)
    log["voxels_changed"] = int(np.count_nonzero(new != np.asarray(img.dataobj).astype(np.uint8)))
    log["labels_before"] = [CLASS_MAP[k] for k in range(1, NUM_LABELS + 1) if before[k]]
    log["labels_after"] = [CLASS_MAP[k] for k in range(1, NUM_LABELS + 1) if after[k]]

    os.makedirs(os.path.join(out_dir, "segmentations"), exist_ok=True)
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(new, img.affine, header), os.path.join(out_dir, "combined_labels.nii.gz"))
    for k, name in CLASS_MAP.items():
        nib.save(nib.Nifti1Image((new == k).astype(np.uint8), img.affine, header),
                 os.path.join(out_dir, "segmentations", f"{name}.nii.gz"))
    with open(os.path.join(out_dir, "postprocessing_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    return log


def main():
    parser = argparse.ArgumentParser(description="Postprocess AI vertebrae masks.")
    parser.add_argument("--input", required=True, help="AbdomenAtlasDemoPredict folder")
    parser.add_argument("--output", required=True,
                        help="output folder (must differ from --input; raw predictions are never overwritten)")
    args = parser.parse_args()
    if os.path.realpath(args.output) == os.path.realpath(args.input):
        parser.error("--output must differ from --input")
    cases = sorted(d for d in os.listdir(args.input)
                   if os.path.isfile(os.path.join(args.input, d, "combined_labels.nii.gz")))
    for case in cases:
        out_dir = os.path.join(args.output, case)
        os.makedirs(out_dir, exist_ok=True)
        ct = os.path.join(args.input, case, "ct.nii.gz")
        if os.path.isfile(ct) and not os.path.isfile(os.path.join(out_dir, "ct.nii.gz")):
            shutil.copy(ct, out_dir)
        log = process_case(os.path.join(args.input, case), out_dir)
        print(f"[{case}] removed {log.get('fragments_removed', 0)} fragments, "
              f"relabeled {len(log.get('relabeled_vertebrae', []))} vertebrae "
              f"and {log.get('arch_voxels_relabeled', 0)} arch voxels, "
              f"reassigned {log.get('pieces_reassigned', 0)} pieces, "
              f"filled {log.get('hole_voxels_filled', 0)} hole voxels, "
              f"{log['voxels_changed']} voxels changed")
        for c in log.get("relabeled_vertebrae", []):
            print("    ", c)


if __name__ == "__main__":
    main()
