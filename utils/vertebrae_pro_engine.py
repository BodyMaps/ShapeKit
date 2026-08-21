"""ShapeKit-Pro: evidence-edited, anatomy-audited post-processing for
vertebra segmentations.

ABSTRACT. Segmentation models label vertebrae well until the anatomy gets
hard: scoliosis, collapsed discs, and ankylosis make them fragment labels,
misplace whole levels, and paint one vertebra's bone with a neighbor's
color, while the standing delete-only cleanup tools remove real bone and can
never return a mislabeled process to its true owner. This file solves the
problem by RECOLORING INSIDE AN ENVELOPE instead of deleting: the raw
prediction is treated as the outer boundary and is never grown or erased,
the CT image is the only editor (HU valleys, bone corridors, and thickness
waists justify every change), and each repair stage carries its own defect
meter and reverts itself in full whenever it cannot prove improvement. Nine
gated stages repair fragmentation, disc-plane assignment, posterior-arch
ownership, and the one-level-down spinous chains that defeat every distance
and thickness race. With one parameter set on both AbdomenAtlasDemo cases
the output reaches zero structural audit flags, restores L1 from 23.3 to
62.9 cm3 where the delete-only baseline shrinks it to 11.6, and re-attaches
every spinous process to its own vertebra (upward violation 3.15 -> 0.21
cm3), all verified at native resolution.

Author:  Abhijit Das  (abhijit.das@mbzuai.ac.ae / aj.das.research@gmail.com)
Project: https://github.com/aj-das-research/jhu-bodymaps-warmup
         The repository carries the full documentation for this file: an
         illustrated README (animated raw-vs-refined comparisons, results
         tables, metric plots), two method studies with the measured
         failures of every rejected algorithm (documentation_v8/,
         documentation_v9/), per-case machine-readable QA reports
         (reports/), full-resolution diagnostic tooling and renders
         (scripts/, visualizations/), and a synthetic ground-truth unit
         test (tests/test_arch_phantom.py). Every number claimed in the
         docstrings below is backed by a QA file in that repository.

Refines SuPreM / TotalSegmentator-style vertebra predictions (C1..L5) on the
AbdomenAtlasDemo scans. Design principle: fix MODEL errors, never "fix"
PATIENT reality - the CT image is the only editor; anatomical priors only
flag suspects and freeze what is already consistent.

Pipeline (all actions QA-logged per case):

  Stage 1  evidence triage of disconnected components, per vertebra:
             speck (<30 mm3)            -> removed
             soft tissue (bone frac<=.5)-> removed (hallucination)
             bone near main (<=5 mm)    -> re-attached through a CT-bone
                                           corridor at the geometric neck of
                                           the gap (unclaimed voxels only),
                                           else kept + POSSIBLE_FRAGMENT
             bone far from main         -> removed to the unclaimed pool,
                                           logged REASSIGN_CANDIDATE
  Stage 2a island guard: a component enclosed >=80% by one other label and
           small relative to it is reassigned to that label (vertebrae do
           not interpenetrate).
  Stage 2b disc-aware boundary re-arbitration of SUSPECT bands only:
           - suspicion: audit flags (FRAGMENTED / SIZE / ORDER), inflated or
             starved size vs neighbors, robust log-volume trend residual,
             or >=20% of the raw label lost to the pool; contiguous closure
             forms the band, everything else is frozen.
           - the CT's own density segments the column: mean HU sampled on
             disks PERPENDICULAR to the body centerline dips at every
             intervertebral disc. A spacing-regularized DP picks exactly the
             expected number of minima between the frozen anchor levels
             (vertebral heights vary smoothly - a periodicity prior), so the
             image, not the labels, decides where bodies begin and end.
           - each level seeds inside its disc-bounded segment; a uniform-
             speed geodesic competition (synchronized layer growth through
             the thresholded bone domain) floods the band; fronts meet at
             low-HU clefts, which is where anatomy separates. Fragments and
             pooled mass are reclaimed by whichever level reaches them
             through bone first.
           - cut surfaces tilt with the column: segment membership uses the
             first-order arc-length projection onto the centerline tangent,
             clipped to +/-6 mm so it stays monotone off-axis.
           - arch phase: posterior elements are rebuilt from PEDICLE ROOTS.
             Every posterior element attaches to its own level through the
             two pedicles - the only bone corridors crossing the body/arch
             frontier - so per level the roots are the arch bone both
             bone-geodesically and Euclidean-within ~3 mm of that level's
             phase-1 body, and a waist-severed two-tier race regrows the
             arch from these roots: identity is decided on the supra-neck
             core (facet bridges are thinner than any true posterior
             element, so severing sub-1.5 mm bone means no level can leak
             across a joint), then the shell rejoins locally and opposing
             fronts meet mid-bridge at the facet waists.
           - guards: C1/C2 skipped (no disc plane at the atlas/axis), band
             skipped when the expected disc count cannot be resolved, and
             the whole band reverts unless the audit strictly improves.
  Stage 2c interface polish: for every adjacent pair whose interface is
           guillotine-flat (planarity RMS below trigger - anatomy
           interdigitates, so near-planar means geometric arbitration, not
           joint anatomy), the boundary is re-solved ONLY inside a small
           collar: seeds just outside the collar, HU-valley watershed so
           the line settles across the darkest surface of the cleft (each
           endplate stays with its own vertebra); accepted per pair only if
           planarity strictly improves with bounded swap and no new
           fragments, else reverted.
  Stage 2d pool reclamation (runs LAST) - envelope rule: the raw prediction
           is the outer boundary; real BONE inside it gets its class fixed,
           it is never deleted. Every dropped raw-labeled bone component is
           re-owned by the AXIAL-RING vote (in the axial projection a
           vertebra is one closed ring; a true process fragment is
           2D-bone-connected to its own ring in its own slices, while ribs
           cross slices as disconnected islands), cross-checked by 3D
           bone-geodesic linkage, then re-attached through a CT-bone
           corridor. Fragments linked to no vertebra through bone stay out,
           flagged (they are not vertebra by the label definition).
  Stage 2e multiview waist-severed recoloring (after 2d): a plate-on-plate
           contact (spinous blades at the interspinous space) is not a thin
           waist in 3D, but in the 2D cross-section of the view containing
           the structure's long axis it is a thin LINE. Each voxel is
           judged in all three orthogonal views: eroded 2D bone cores are
           identified by the level ANCHOR they contain (mass within ~28 mm
           of a body through its own label - tips can never anchor their
           captor), single-anchor components vote their level in-plane,
           ambiguous components abstain as blockers, and only unanimous
           cross-view votes recolor a voxel. Bounded per pair, loser must
           not fragment, stage reverts if the audit degrades.
  Stage 2f core-integrity surgery (after 2e): a mixed in-plane supra-neck
           piece is the defect by definition (one rigid cross-section, two
           labels): unify to majority when the internal boundary crosses
           the piece's thick interior and the minority is small, else
           relocate the boundary to the in-plane thickness valley. Gated on
           the mixed-piece meter, audit, bounded per-level shifts.
  Stage 2g imbrication repair (last relabeling stage): thoracolumbar
           spinous blades IMBRICATE - root inside its own disc band, blade
           drooping caudally past the next level - so every distance race
           hands the drooping half one level down, and on DISH the ossified
           interspinous sheet defeats every thickness signal too. The one
           surviving invariant: NOTHING in the midline posterior corridor
           grows toward the head. The corridor is re-derived in a single
           top-to-bottom consecutive-slice sweep: band-consistent ring-strip
           labels seed, the z+1 assignment flows down wherever bone
           continues, a uniform 2D watershed extends in-plane - so a label
           can only enter at its junction and flow caudally to the tip.
           Gated on the upward-violation meter (label mass above its own
           band top: anatomically impossible), mixed-piece meter, audit,
           bounded per-level shifts.
  Stage 3  interface regularization (iterated 26-neighborhood majority vote
           on label-label interfaces), orphan-component absorption, enclosed-
           hole + directional pit filling, and bounded volume-preserving smoothing
           (Gaussian-smoothed SDT, volume-preserving iso-level by bisection,
           changes confined to +/-1.5 mm, additions bone-gated; a label
           reverts if Dice vs its pre-smooth mask drops below 0.97 or its
           component count increases).
  Stage 4  audit of the result (fragmentation, size-smoothness, ordering)
           written to the QA report; flags never edit voxels.

Verified on the two AbdomenAtlasDemo cases (identical parameters):
  BDMAP_00000006 (clean): 10 fragmented + 1 ORDER -> all clean; C1 caudal
    articular tips on CT-certified bone are preserved (the delete-only
    baseline removes them).
  BDMAP_00000031 (sick):  21 fragmented, mass misassignment T8..L2 with
    L1 at 23.3 cm3 vs ~60 cm3 neighbors -> all clean; L1 restored to
    ~65 cm3 by re-arbitration at the detected disc planes; the ShapeKit
    baseline instead deepens the error (L1 11.6 cm3, SIZE 0.20). The
    imbricated spinous chain (raw painted every T7..L1 blade with mixed /
    one-off labels) reads clean after stage 2g: upward-violation
    3.15 -> 0.21 cm3, every blade root-band consistent.

Dependencies: numpy, scipy, nibabel, scikit-image, cc3d (connected-components-3d).

Usage:
  python postprocessing_vertebrae.py \
      --pred_dir AbdomenAtlasDemoPredict \
      --ct_root  data/AbdomenAtlasDemo \
      --out_dir  AbdomenAtlasDemoPredict_refined \
      [--report_dir reports] [--case BDMAP_00000031]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage, signal
from skimage.segmentation import watershed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("postprocessing_vertebrae")

# SuPreM combined_labels ids: 1=L5 ... 24=C1 (bottom-up)
NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
ID_TO_NAME = {i + 1: n for i, n in enumerate(NAMES_BOTTOM_UP)}
NAME_TO_ID = {n: i for i, n in ID_TO_NAME.items()}
TOP_DOWN = list(reversed(NAMES_BOTTOM_UP))
STRUCT6 = ndimage.generate_binary_structure(3, 1)
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

P = {
    # evidence gates
    "bone_hu": 150.0, "speck_mm3": 30.0, "soft_bone_fraction": 0.50,
    "near_mm": 5.0, "corridor_pad_mm": 6.0, "bridge_cap_mm3": 500.0,
    # islands
    "enclosure": 0.80, "max_rel_to_host": 0.5, "min_labeled_shell": 0.5,
    # suspicion / bands
    "suspect_size_lo": 0.60, "suspect_size_hi": 1.80, "suspect_logres": 0.30,
    "suspect_pool_frac": 0.20, "band_gap_close": 3,
    # disc profile / DP
    "profile_sigma_mm": 2.5, "profile_radius_mm": 10.0, "min_body_height_mm": 12.0,
    "max_body_height_mm": 45.0, "dp_beta": 2.0, "pin_lo_mm": 8.0, "pin_hi_mm": 35.0,
    # arbitration domain
    "seed_margin_mm": 3.0, "seed_radius_mm": 14.0, "halo_mm": 2.5, "band_pad_mm": 4.0,
    "arch_root_mm": 3.0, "arch_neck_mm": 1.5, "arch_cost": "core",
    "arch_overhang_mm": 16.0,
    # imbrication repair (stage 2g)
    "imb_half_mm": 14.0, "imb_strip_mm": 6.0, "imb_canal_mm": 12.0,
    "imb_max_shift_cm3": 6.0,
    # interface polish (stage 2c)
    "polish_trigger_mm": 3.5, "polish_min_contact_mm2": 80.0, "polish_collar_mm": 6.0,
    "polish_seed_shell_mm": 2.5, "polish_max_shift_frac": 0.15,
    "polish_min_gain_mm": 0.3, "polish_priority": "hu",
    # pool reclamation (stage 2d, runs last)
    "reclaim_neck_mm": 10.0, "reclaim_link_mm": 10.0,
    # multiview recoloring (stage 2e)
    "mv_neck_mm": 1.2, "mv_anchor_mm": 40.0, "mv_max_shift_frac": 0.25,
    "mv_elong": 2.5,
    # skeleton-component relabeling (stage 2f)
    "skel_cut_mm": 1.8,
    # smoothing
    "sigma_mm": 1.2, "max_dev_mm": 1.5, "vol_tol": 0.01, "min_dice": 0.97,
    "smooth_pad_mm": 5.0,
    # io
    "crop_margin_mm": 25.0,
}


# ------------------------------------------------------------------ utils --
def _bbox_pad(sl, pad_vox, shape):
    """Takes: sl (tuple of slices from find_objects), pad_vox (per-axis voxel
        padding), shape (array shape).
    Does: grows a bounding box by the padding, clamped to the array bounds.
    Returns: the padded tuple of slices.
    """
    return tuple(slice(max(s.start - int(p), 0), min(s.stop + int(p), n))
                 for s, p, n in zip(sl, pad_vox, shape))


def _components(mask):
    """Takes: mask (3D bool).
    Does: 26-connected component labeling (cc3d).
    Returns: (labels volume, per-component voxel counts; counts[0] is zeroed).
    """
    cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    return cc, counts


def audit(seg, affine, vox_mm3):
    """Repo-convention audit: FRAGMENTED / SIZE(<0.6) / ORDER / EMPTY flags.

    Takes: seg (uint8 labels), affine (voxel-to-world, orients the S-I axis),
        vox_mm3.
    Does: the independent structural audit: per level it measures components,
        volume, size ratio against neighbors, and centroid ordering, and raises
        FRAGMENTED / EMPTY / SIZE / ORDER flags. Flags never edit voxels; every
        stage gate consumes this.
    Returns: (per-level row dicts, summary dict of flag counts).
    """
    rows = []
    objs = ndimage.find_objects(seg)
    for name in TOP_DOWN:
        lid = NAME_TO_ID[name]
        sl = objs[lid - 1] if lid - 1 < len(objs) else None
        if sl is None:
            rows.append({"name": f"vertebrae_{name}", "voxels": 0, "volume_cm3": 0.0,
                         "components": 0, "largest_fraction": 0.0, "flags": ["EMPTY"]})
            continue
        mask = seg[sl] == lid
        vox = int(mask.sum())
        _, counts = _components(mask)
        com = ndimage.center_of_mass(mask)
        w = affine @ np.array([com[0] + sl[0].start, com[1] + sl[1].start,
                               com[2] + sl[2].start, 1.0])
        ncomp = int((counts > 0).sum())
        rows.append({"name": f"vertebrae_{name}", "voxels": vox,
                     "volume_cm3": round(vox * vox_mm3 / 1000.0, 1),
                     "components": ncomp,
                     "largest_fraction": round(float(counts.max()) / vox, 4),
                     "si_world": float(w[2]),
                     "flags": [f"FRAGMENTED({ncomp})"] if ncomp > 1 else []})
    si = [(r["name"], r["si_world"]) for r in rows if "si_world" in r]
    if len(si) >= 3:
        dec = si[-1][1] < si[0][1]
        for i in range(1, len(si)):
            step = si[i][1] - si[i - 1][1]
            if (dec and step > 0) or (not dec and step < 0):
                next(r for r in rows if r["name"] == si[i][0])["flags"].append("ORDER")
    present = [r for r in rows if r["voxels"] > 0]
    for i in range(1, len(present) - 1):
        smaller = min(present[i - 1]["voxels"], present[i + 1]["voxels"])
        if smaller > 0:
            ratio = round(present[i]["voxels"] / smaller, 2)
            present[i]["size_ratio_vs_neighbors"] = ratio
            if ratio < 0.6:
                present[i]["flags"].append(f"SIZE({ratio})")
    summary = {k: sum(1 for r in rows if any(f.startswith(pre) for f in r["flags"]))
               for k, pre in [("n_fragmented", "FRAG"), ("n_empty", "EMPTY"),
                              ("n_size", "SIZE"), ("n_order", "ORDER")]}
    return rows, summary


# ---------------------------------------------------------------- stage 1 --
def stage1_triage(seg, ct, zooms, vox_mm3, records):
    """Takes: seg (raw uint8 labels), ct (HU volume, same grid), zooms (mm),
        vox_mm3, records (QA list, appended per decision).
    Does: evidence triage of every disconnected component: sub-speck and
        mostly-soft-tissue components are removed as noise, bone near its main
        body re-attaches through a CT-certified corridor, and far bone is moved
        to the unclaimed pool for stage 2d instead of being deleted.
    Returns: the triaged label volume.
    """
    out = seg.copy()
    bone = ct >= P["bone_hu"]
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        main = cc == main_id
        union_sl = ndimage.find_objects((cc > 0).astype(np.uint8))[0]
        pad_vox = np.ceil(P["near_mm"] / np.asarray(zooms)) + 2
        union_sl = _bbox_pad(union_sl, pad_vox, seg.shape)
        dist_main = ndimage.distance_transform_edt(~main[union_sl], sampling=zooms)
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            vol = float(counts[comp_id]) * vox_mm3
            hu = ct[comp]
            rec = {"stage": 1, "label": ID_TO_NAME[lid], "vol_mm3": round(vol, 1),
                   "hu_mean": round(float(hu.mean()), 1),
                   "bone_fraction": round(float((hu >= P["bone_hu"]).mean()), 3)}
            if vol < P["speck_mm3"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SPECK"
            elif rec["bone_fraction"] <= P["soft_bone_fraction"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SOFT"
            else:
                d = float(dist_main[comp[union_sl]].min())
                rec["dist_to_main_mm"] = round(d, 2)
                if d <= P["near_mm"]:
                    added = _bridge(out, bone, main, comp, lid, zooms)
                    if added is not None:
                        out[comp] = lid
                        out[added] = lid
                        rec["action"] = "BRIDGED"
                        rec["bridge_added_mm3"] = round(float(added.sum()) * vox_mm3, 1)
                    else:
                        rec["action"] = "KEPT_POSSIBLE_FRAGMENT"
                else:
                    out[comp] = 0
                    rec["action"] = "REMOVED_FAR_REASSIGN_CANDIDATE"
            records.append(rec)
    return out


def _bridge(seg, bone, main, comp, lid, zooms, neck_mm=None):
    """Takes: seg (labels, painted in place), bone (CT bone mask), main / comp
        (bool masks of the label's main body and the candidate component), lid
        (label id), zooms, neck_mm (optional corridor width cap).
    Does: searches a padded box for a CT-bone corridor connecting comp to main
        and, when one exists within the volume cap, paints the corridor's
        unclaimed bone with lid, so re-attachment adds only image-certified
        bone.
    Returns: the painted corridor mask, or None when no admissible corridor
        exists.
    """
    neck_mm = P["near_mm"] if neck_mm is None else neck_mm
    nz_c = ndimage.find_objects(comp.astype(np.uint8))[0]
    nz_m = ndimage.find_objects(main.astype(np.uint8))[0]
    joint = tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                  for a, b in zip(nz_c, nz_m))
    pad_vox = np.ceil(P["corridor_pad_mm"] / np.asarray(zooms)) + 1
    joint = _bbox_pad(joint, pad_vox, seg.shape)
    seg_j, bone_j = seg[joint], bone[joint]
    comp_j, main_j = comp[joint], main[joint]
    d_comp = ndimage.distance_transform_edt(~comp_j, sampling=zooms)
    d_main = ndimage.distance_transform_edt(~main_j, sampling=zooms)
    neck = (d_comp <= neck_mm) & (d_main <= neck_mm)
    domain = (bone_j & ((seg_j == 0) | (seg_j == lid)) & neck) | comp_j | main_j
    reach = ndimage.binary_propagation(main_j, mask=domain, structure=STRUCT26)
    if not (reach & comp_j).any():
        return None
    fill = reach & (seg_j == 0) & bone_j & neck
    if float(fill.sum()) * abs(np.prod(zooms)) > P["bridge_cap_mm3"]:
        return None
    corridor = np.zeros_like(comp)
    corridor[joint] = fill
    return corridor


# --------------------------------------------------------------- stage 2a --
def stage2a_islands(seg, vox_mm3, records):
    """Takes: seg, vox_mm3, records (QA list).
    Does: island guard: a component enclosed at least 80 percent by one other
        label and small relative to that host is absorbed by it, since
        vertebrae do not interpenetrate.
    Returns: the label volume with enclosed islands absorbed.
    """
    out = seg.copy()
    vol = {int(l): int((seg == l).sum()) for l in np.unique(seg) if l != 0}
    for lid in sorted(vol):
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            sl = _bbox_pad(ndimage.find_objects(comp.astype(np.uint8))[0],
                           np.array([2, 2, 2]), seg.shape)
            comp_l = comp[sl]
            shell = ndimage.binary_dilation(comp_l, structure=STRUCT26) & ~comp_l
            neigh = out[sl][shell]
            labeled = neigh[neigh != 0]
            if neigh.size == 0 or labeled.size / neigh.size < P["min_labeled_shell"]:
                continue
            ids, cnt = np.unique(labeled, return_counts=True)
            host = int(ids[cnt.argmax()])
            frac = float(cnt.max()) / labeled.size
            rec = {"stage": "2a", "label": ID_TO_NAME[lid], "host": ID_TO_NAME[host],
                   "vol_mm3": round(float(counts[comp_id]) * vox_mm3, 1),
                   "enclosure": round(frac, 3)}
            if (host != lid and frac >= P["enclosure"]
                    and counts[comp_id] <= P["max_rel_to_host"] * vol.get(host, 0)):
                out[comp] = host
                rec["action"] = "ISLAND_REASSIGNED"
            else:
                rec["action"] = "ISLAND_FLAG_ONLY"
            records.append(rec)
    return out


# --------------------------------------------------------------- stage 2b --
def find_suspects(seg, raw, affine, vox_mm3):
    """Takes: seg, raw (original prediction, for the unclaimed pool), affine,
        vox_mm3.
    Does: runs the audit, marks suspect levels (size, fragmentation, or
        ordering anomalies, or a large nearby pool), and closes small gaps so
        contiguous suspects and their audit-clean neighbors form re-arbitration
        bands.
    Returns: (list of level-name bands, suspect names, top-down list of present
        levels).
    """
    rows, _ = audit(seg, affine, vox_mm3)
    info = {r["name"].replace("vertebrae_", ""): r for r in rows if r["voxels"] > 0}
    order = [n for n in TOP_DOWN if n in info]
    suspects = set()
    logv = {n: np.log(max(info[n]["voxels"], 1)) for n in order}
    for i, n in enumerate(order):
        w = [logv[order[j]] for j in range(max(0, i - 2), min(len(order), i + 3)) if j != i]
        if w and abs(logv[n] - float(np.median(w))) > P["suspect_logres"]:
            suspects.add(n)
    for n, r in info.items():
        ratio = r.get("size_ratio_vs_neighbors")
        if ratio is not None and (ratio < P["suspect_size_lo"] or ratio > P["suspect_size_hi"]):
            suspects.add(n)
        if r["components"] > 1 or "ORDER" in r["flags"]:
            suspects.add(n)
    for lid in np.unique(raw):
        if lid == 0:
            continue
        raw_n = int((raw == lid).sum())
        pool_n = int(((raw == lid) & (seg == 0)).sum())
        if raw_n and pool_n / raw_n >= P["suspect_pool_frac"]:
            suspects.add(ID_TO_NAME[int(lid)])
    idx = {n: i for i, n in enumerate(order)}
    s_idx = sorted(idx[n] for n in suspects if n in idx)
    bands = []
    for i in s_idx:
        if bands and i - bands[-1][-1] <= P["band_gap_close"] + 1:
            bands[-1] = list(range(bands[-1][0], i + 1))
        else:
            bands.append([i])
    return [[order[j] for j in range(b[0], b[-1] + 1)] for b in bands], sorted(suspects), order


def column_profile(seg, raw, ct, zooms):
    """Body centerline (largest in-plane bone component) + perpendicular
    mean-HU profile: bodies read high, discs dip, even under kyphosis.

    Takes: seg, raw, ct, zooms.
    Does: traces the per-slice column centroid and samples mean HU inside
        centerline disks, producing the density profile whose minima are
        intervertebral discs.
    Returns: (cx, cy per-slice centroid arrays, smoothed HU profile; the
        profile is zeros when the column is too short).
    """
    bone = ct >= P["bone_hu"]
    M = (seg > 0) | ((raw > 0) & bone)
    Mb = M & bone
    nz = seg.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in np.nonzero(M.any(axis=(0, 1)))[0]:
        sl2 = Mb[:, :, z] if Mb[:, :, z].any() else M[:, :, z]
        lab, n = ndimage.label(sl2)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        pts = np.nonzero(lab == sizes.argmax())
        cx[z], cy[z] = pts[0].mean(), pts[1].mean()
    w = max(int(40.0 / zooms[2]) | 1, 5)
    ker = np.ones(w) / w
    ok = ~np.isnan(cx)
    cxs, cys = cx.copy(), cy.copy()
    cxs[ok] = np.convolve(np.pad(cx[ok], w // 2, mode="edge"), ker, "valid")
    cys[ok] = np.convolve(np.pad(cy[ok], w // 2, mode="edge"), ker, "valid")
    zmm = np.asarray(zooms)
    Pmm = np.c_[cxs * zmm[0], cys * zmm[1], np.arange(nz) * zmm[2]]
    idx = np.nonzero(ok)[0]
    if idx.size < 5:
        return cxs, cys, np.zeros(nz)
    T = np.zeros_like(Pmm)
    grad = np.gradient(ndimage.gaussian_filter1d(Pmm[idx], sigma=8.0 / zmm[2], axis=0), axis=0)
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
    T[idx] = grad
    r = P["profile_radius_mm"]
    aa, bb = np.meshgrid(np.arange(-r, r + 1.2, 1.2), np.arange(-r, r + 1.2, 1.2))
    keep = aa ** 2 + bb ** 2 <= r ** 2
    offs = np.c_[aa[keep], bb[keep]]
    rho = np.full(nz, np.nan)
    hu = ct.astype(np.float32)
    for z in idx:
        t = T[z]
        ref = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0.0, 1, 0])
        u = np.cross(t, ref); u /= np.linalg.norm(u)
        v = np.cross(t, u)
        pts_mm = Pmm[z] + offs[:, :1] * u + offs[:, 1:2] * v
        vals = ndimage.map_coordinates(hu, (pts_mm / zmm).T, order=1, mode="nearest")
        rho[z] = float(np.clip(vals, 0, 800).mean())
    rho_s = ndimage.gaussian_filter1d(np.where(np.isnan(rho), 0.0, rho),
                                      sigma=max(P["profile_sigma_mm"] / zooms[2], 0.8))
    return cxs, cys, rho_s


def disc_minima(rho_s, z_lo, z_hi, n_cuts, zooms, pin_lo, pin_hi):
    """Exactly n_cuts minima chosen by spacing-regularized DP with anchor pins.

    Takes: rho_s (HU profile), z_lo / z_hi (search window), n_cuts (expected
        disc count), zooms, pin_lo / pin_hi (whether the window ends anchor on
        clean neighbors).
    Does: selects exactly n_cuts profile minima by spacing-regularized dynamic
        programming under body-height priors, so shallow or fused discs cannot
        collapse the count.
    Returns: sorted z indices of the chosen cuts, or None when the expected
        count cannot be resolved (the band is then skipped, flagged).
    """
    seg_rho = rho_s[z_lo:z_hi]
    if seg_rho.size < 5 or seg_rho.max() <= 0 or n_cuts <= 0:
        return [] if n_cuts == 0 else None
    dz = zooms[2]
    peaks, props = signal.find_peaks(-seg_rho, distance=max(int(6.0 / dz), 2),
                                     prominence=1e-6)
    if peaks.size < n_cuts:
        return None
    z = peaks.astype(float) * dz
    p = props["prominences"] / props["prominences"].max()
    m, K = peaks.size, n_cuts
    min_h, max_h = P["min_body_height_mm"], P["max_body_height_mm"]
    span = (z_hi - z_lo) * dz
    NEG = -1e18
    first_ok = (z >= P["pin_lo_mm"]) & (z <= P["pin_hi_mm"]) if pin_lo else np.ones(m, bool)
    last_ok = ((z >= span - P["pin_hi_mm"]) & (z <= span - P["pin_lo_mm"])
               if pin_hi else np.ones(m, bool))
    if not first_ok.any() or not last_ok.any():
        return None
    dp = np.full((K, m, m), NEG)
    bk = np.zeros((K, m, m), dtype=np.int32)
    for j in range(m):
        if first_ok[j]:
            dp[0, j, :] = p[j]
    for k in range(1, K):
        for j in range(m):
            for i in range(j):
                h = z[j] - z[i]
                if not (min_h <= h <= max_h):
                    continue
                if k == 1:
                    best_prev, best_i2 = dp[0, i, 0], 0
                else:
                    best_prev, best_i2 = NEG, 0
                    for i2 in range(i):
                        h_prev = z[i] - z[i2]
                        if not (min_h <= h_prev <= max_h):
                            continue
                        s = dp[k - 1, i, i2] - P["dp_beta"] * np.log(h / h_prev) ** 2
                        if s > best_prev:
                            best_prev, best_i2 = s, i2
                if best_prev > NEG / 2 and best_prev + p[j] > dp[k, j, i]:
                    dp[k, j, i] = best_prev + p[j]
                    bk[k, j, i] = best_i2
    flat = dp[K - 1].copy()
    flat[~last_ok, :] = NEG
    if flat.max() <= NEG / 2:
        return None
    j, i = np.unravel_index(int(flat.argmax()), flat.shape)
    picks = [j]
    k = K - 1
    while k >= 1:
        picks.append(i)
        j, i, k = i, int(bk[k, j, i]), k - 1
    return sorted(int(peaks[q]) + z_lo for q in set(picks))


def arch_repartition(labels_ws, body_zone, A, band_ids, zooms, vox_mm3):
    """Rebuild the posterior arch from PEDICLE ROOTS (pure function of the
    phase-1 flood; bodies are never edited).

    Anatomy: every posterior element (pedicles, laminae, spinous, transverse
    and articular processes) attaches to its own vertebra exclusively through
    the two pedicles - the only bone corridors crossing the body/arch
    frontier - while adjacent levels touch posteriorly only at thin
    facet-joint bridges. Two consequences drive the algorithm:

      1. SEEDS: per level, the pedicle roots are the arch-domain bone that is
         BONE-GEODESICALLY within arch_root_mm of that level's phase-1 body
         (masked dilation through own body + arch only, so Euclidean
         proximity across the foramen or a collapsed disc cannot mint a
         root) AND Euclidean-within the same radius (so voxel-step overshoot
         on anisotropic grids cannot either). Voxels reached by two bodies
         are contested and stay unseeded.
      2. COMPETITION (arch_cost="core"): a waist-severed two-tier race.
         Tier 1 decides IDENTITY on the supra-neck core (arch voxels with
         local thickness >= arch_neck_mm, plus the roots): facet bridges are
         thinner than any true posterior element, so severing them makes
         each level's core reachable only from its own pedicle roots and the
         identity race cannot leak across a joint. Tier 2 rejoins the shell
         (sub-neck voxels, including the bridges themselves) by a local
         uniform BFS from the tier-1 labels, so opposing shells meet
         mid-bridge. Phantom-measured ablations: "edt" (-EDT priority) has
         no post-neck resistance and splits hanging processes by depth
         order; "uniform" lets a root near a joint out-run a far pedicle.

    Takes: labels_ws (band race result), body_zone (bool body domain), A (bool
        arch domain), band_ids, zooms, vox_mm3.
    Does: rebuilds posterior-arch ownership from pedicle-root seeds with the
        waist-severed two-tier race (arch_cost modes: core = release, hier /
        edt / uniform = measured ablations); bodies are never edited.
    Returns: (labels with the arch re-owned, QA record dict).
    """
    body_lab = np.where(body_zone, labels_ws, 0).astype(np.int32)
    roots = np.zeros(A.shape, dtype=np.int32)
    contested = np.zeros(A.shape, dtype=bool)
    pad_vox = np.ceil((P["arch_root_mm"] + 2.0) / np.asarray(zooms)).astype(int) + 1
    root_it = max(int(np.ceil(P["arch_root_mm"] / min(zooms))), 1)
    rootless = []
    for lid in band_ids:
        body = body_lab == lid
        if not body.any():
            rootless.append(ID_TO_NAME[lid])
            continue
        sl = _bbox_pad(ndimage.find_objects(body.astype(np.uint8))[0],
                       pad_vox, A.shape)
        reach = ndimage.binary_dilation(body[sl], structure=STRUCT6,
                                        iterations=root_it,
                                        mask=A[sl] | body[sl])
        d_eu = ndimage.distance_transform_edt(~body[sl], sampling=zooms)
        r = reach & A[sl] & (d_eu <= P["arch_root_mm"] + max(zooms))
        if not r.any():
            rootless.append(ID_TO_NAME[lid])
            continue
        sub = roots[sl]
        contested[sl] |= (sub > 0) & (sub != lid) & r
        sub[r & (sub == 0)] = lid
    roots[contested] = 0
    rec = {"arch_mode": f"pedicle_roots_{P['arch_cost']}",
           "arch_roots_cm3": {ID_TO_NAME[l]: round(float((roots == l).sum())
                                                   * vox_mm3 / 1000.0, 2)
                              for l in band_ids if (roots == l).any()},
           "arch_contested_mm3": round(float(contested.sum()) * vox_mm3, 1)}
    if rootless:
        rec["arch_rootless"] = rootless
    if not (roots > 0).any():
        rec["arch_skipped"] = "no pedicle roots found"
        return labels_ws, rec
    if P["arch_cost"] == "hier":
        # MULTI-SCALE PEEL RACE (ablation - MEASURED WORSE on-case, kept for
        # the record). Idea: no single neck scale exists on DISH (facet
        # necks 1-2 mm, fusions 2-5 mm, blades 4-8 mm), so claim thick bone
        # first and let fronts cross thinner bone only later; thickness
        # measured in-plane per parasagittal slice. MEASURED OUTCOME
        # (BDMAP_00000031, T5..L2 band): upward-violation 0.33 -> 7.68 cm3
        # vs 3.16 for "core" - REVERTED by the imbrication gate. WHY IT
        # FAILS: the ossified interspinous bridge is a MIDLINE SAGITTAL
        # SHEET, continuous with both blades; its sagittal in-plane
        # cross-section is a broad blob, so in-plane thickness is LARGE at
        # exactly the bridge to be severed - the peel welds blade to blade
        # (and 3D EDT agrees). No geometric thickness signal separates a
        # blade from a fusion sheet in its own plane; identity there must
        # come from CAUDAL MONOTONICITY instead (stage 2g).
        bb = _bbox_pad(ndimage.find_objects(A.astype(np.uint8))[0],
                       np.array([2, 2, 2]), A.shape)
        Ab = A[bb]
        solid = Ab | body_zone[bb]      # anatomical thickness, not domain-cut
        t2 = np.zeros(Ab.shape, dtype=np.float32)
        zo2 = (zooms[1], zooms[2])
        for x in range(Ab.shape[0]):
            sl2 = solid[x]
            if sl2.any():
                t2[x] = ndimage.distance_transform_edt(sl2, sampling=zo2)
        lab = roots[bb].astype(np.int32)
        tmax = float(t2[Ab].max()) if Ab.any() else 0.0
        tiers = [t for t in (8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.6,
                             1.2, 0.8, 0.4) if t < tmax] + [0.0]
        for t in tiers:
            mask_t = (Ab & (t2 >= t)) | (lab > 0)
            lab = watershed(np.zeros(Ab.shape, dtype=np.uint8), markers=lab,
                            mask=mask_t)
        labels_arch = np.zeros(A.shape, dtype=np.int32)
        labels_arch[bb] = lab
    elif P["arch_cost"] == "core":
        edtA = ndimage.distance_transform_edt(A, sampling=zooms).astype(np.float32)
        core = A & ((edtA >= P["arch_neck_mm"]) | (roots > 0))
        lab1 = watershed(np.zeros(A.shape, dtype=np.uint8), markers=roots,
                         mask=core)
        labels_arch = watershed(np.zeros(A.shape, dtype=np.uint8), markers=lab1,
                                mask=A)
    elif P["arch_cost"] == "edt":
        edtA = ndimage.distance_transform_edt(A, sampling=zooms).astype(np.float32)
        labels_arch = watershed(-edtA, markers=roots, mask=A)
    else:
        labels_arch = watershed(np.zeros(A.shape, dtype=np.uint8),
                                markers=roots, mask=A)
    sel = labels_arch > 0
    changed = int((sel & (labels_arch != labels_ws)).sum())
    out = np.where(sel, labels_arch.astype(labels_ws.dtype), labels_ws)
    rec["arch_phase2_changed_mm3"] = round(changed * vox_mm3, 1)
    rec["arch_unrooted_mm3"] = round(float((A & ~sel).sum()) * vox_mm3, 1)
    return out, rec


def arbitrate_band(seg, raw, ct, band, neighbors, cxs, cys, rho_s, zooms,
                   vox_mm3, qa):
    """Re-arbitrate one contiguous suspect band (v2 geometry).

    v2 upgrades over the flat-z version:
      - ARC-LENGTH OBLIQUE CUTS: segment membership uses s = first-order
        projection onto the column centerline tangent, so cut planes tilt with
        kyphosis instead of clipping tilted bodies with axial guillotines.
      - AGREEMENT-EXTENDED SEEDS: body-core seeds grow by geodesic propagation
        through voxels whose CURRENT label already matches the segment (and
        whose s lies in-segment), giving each level arch seeds wherever the
        model already agreed; only genuinely contested bone is left to race.
      - EDT-PRIORITY WATERSHED: flood priority = -distance-to-background, the
        classic touching-object splitter: thick masses (bodies, laminae) are
        claimed first and fronts meet at thin waists, which anatomically are
        the facet joints and the pars, not arbitrary mid-arch loci.

    Takes: seg, raw, ct, band (level names), neighbors (clean levels below /
        above), cxs / cys / rho_s (column geometry from column_profile), zooms,
        vox_mm3, qa.
    Does: re-arbitrates one suspect band end to end: disc cuts by DP, oblique
        arc-length segment membership, agreement-extended body seeds, uniform
        geodesic race, then the pedicle-root arch rebuild.
    Returns: the re-arbitrated label volume, or None when disc counting is
        unresolved.
    """
    bone = ct >= P["bone_hu"]
    band_ids = [NAME_TO_ID[n] for n in band]
    band_mask = np.isin(seg, band_ids)
    pool = (np.isin(raw, band_ids)) & (seg == 0) & bone
    zsel = np.nonzero((band_mask | pool).any(axis=(0, 1)))[0]
    pad = int(P["band_pad_mm"] / zooms[2]) + 1
    z0 = max(int(zsel[0]) - pad, 0)
    z1 = min(int(zsel[-1]) + pad + 1, seg.shape[2])
    rec = {"levels": band, "z_range_crop": [int(z0), int(z1)]}
    qa["bands"].append(rec)

    def centroid_z(name):
        zs = np.nonzero((seg == NAME_TO_ID[name]).any(axis=(0, 1)))[0]
        return float(zs.mean()) if zs.size else None

    ups = sorted((NAME_TO_ID[n], centroid_z(n)) for n in band if centroid_z(n) is not None)
    z_up = True if len(ups) < 2 else ups[-1][1] >= ups[0][1]
    rec["z_increases_toward_head"] = bool(z_up)
    lo_name, hi_name = (neighbors if z_up else neighbors[::-1])
    w_lo = centroid_z(lo_name) if lo_name else None
    w_hi = centroid_z(hi_name) if hi_name else None
    win_lo = int(w_lo) if w_lo is not None else int(zsel[0]) + int(4.0 / zooms[2])
    win_hi = int(w_hi) if w_hi is not None else int(zsel[-1]) - int(4.0 / zooms[2])
    n_expect = (len(band_ids) - 1) + (w_lo is not None) + (w_hi is not None)
    found = disc_minima(rho_s, win_lo, win_hi, n_expect, zooms,
                        w_lo is not None, w_hi is not None)
    if found is None:
        qa["flags"].append(f"DISC_COUNT_UNRESOLVED band={band}: flag only, no edit")
        rec["skipped"] = True
        return None
    lo_cut = found.pop(0) if w_lo is not None else z0
    hi_cut = found.pop(-1) if w_hi is not None else z1
    cuts_z = [int(lo_cut)] + [int(f) for f in found] + [int(hi_cut)]
    rec["disc_cuts_z"] = cuts_z

    # ---- arc-length field ----------------------------------------------
    zmm = np.asarray(zooms, dtype=np.float64)
    nzc = seg.shape[2]
    okc = ~np.isnan(cxs)
    Pmm = np.c_[np.where(okc, cxs, 0) * zmm[0], np.where(okc, cys, 0) * zmm[1],
                np.arange(nzc) * zmm[2]]
    T = np.zeros_like(Pmm)
    idxc = np.nonzero(okc)[0]
    g = np.gradient(ndimage.gaussian_filter1d(Pmm[idxc], sigma=8.0 / zmm[2], axis=0), axis=0)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    T[idxc] = g
    step_len = np.zeros(nzc)
    step_len[idxc[1:]] = np.linalg.norm(np.diff(Pmm[idxc], axis=0), axis=1)
    s_axis = np.cumsum(step_len)
    s_cuts = [float(s_axis[c]) for c in cuts_z]

    tilt_pad = int(10.0 / zooms[2])
    z0e, z1e = max(cuts_z[0] - tilt_pad, 0), min(cuts_z[-1] + tilt_pad + 1, nzc)
    slab = slice(z0e, z1e)
    seg_s, bone_s = seg[:, :, slab], bone[:, :, slab]
    band_s, pool_s = band_mask[:, :, slab], pool[:, :, slab]
    xx, yy = np.meshgrid(np.arange(seg.shape[0], dtype=np.float32),
                         np.arange(seg.shape[1], dtype=np.float32), indexing="ij")
    s_f = np.empty(seg_s.shape, dtype=np.float32)
    for z in range(z0e, z1e):
        dx = (xx - np.float32(cxs[z] if okc[z] else 0)) * np.float32(zmm[0])
        dy = (yy - np.float32(cys[z] if okc[z] else 0)) * np.float32(zmm[1])
        # oblique correction, clipped to +/-6 mm: full first-order tilt at the
        # body radius (where cuts matter), saturated at rib/process radius
        # where the per-z linearization folds (validated visually)
        corr = np.clip(dx * np.float32(T[z, 0]) + dy * np.float32(T[z, 1]), -6.0, 6.0)
        s_f[:, :, z - z0e] = np.float32(s_axis[z]) + corr

    ids_by_z = sorted(band_ids) if z_up else sorted(band_ids, reverse=True)
    seg_ranges = list(zip(s_cuts[:-1], s_cuts[1:]))
    assign = dict(zip(ids_by_z, seg_ranges))
    s_lo_all, s_hi_all = s_cuts[0], s_cuts[-1]

    # ---- seeds: body core + agreement extension ------------------------
    markers = np.zeros(seg_s.shape, dtype=np.int32)
    r2 = P["seed_radius_mm"] ** 2
    margin = P["seed_margin_mm"]
    d2_slab = np.empty(seg_s.shape, dtype=np.float32)
    for z in range(z0e, z1e):
        dx = (xx - np.float32(cxs[z] if okc[z] else 0)) * np.float32(zmm[0])
        dy = (yy - np.float32(cys[z] if okc[z] else 0)) * np.float32(zmm[1])
        d2_slab[:, :, z - z0e] = dx * dx + dy * dy
    eligible_s = band_s | pool_s
    for lid, (sa, sb) in assign.items():
        seeds = (bone_s & eligible_s & (d2_slab <= r2)
                 & (s_f >= sa + margin) & (s_f <= sb - margin))
        if not seeds.any():
            er = ndimage.binary_erosion(seg_s == lid, structure=STRUCT6, iterations=2)
            seeds = er if er.any() else (seg_s == lid)
            qa["flags"].append(f"SEED_FALLBACK_{ID_TO_NAME[lid]}")
        markers[seeds] = lid
    rec["seed_cm3"] = {ID_TO_NAME[l]: round(float((markers == l).sum()) * vox_mm3 / 1000.0, 1)
                       for l in assign}

    # ---- domain + EDT-priority competition ------------------------------
    st = ndimage.generate_binary_structure(3, 1)
    halo_it = int(max(np.round(P["halo_mm"] / np.asarray(zooms))))
    grown = ndimage.binary_dilation(band_s | pool_s, structure=st, iterations=max(halo_it, 1))
    halo = grown & (seg_s == 0) & bone_s & ~pool_s
    # bodies are gated strictly at the outer disc cuts, but the ARCH domain
    # may OVERHANG them: superior/inferior articular processes and steep
    # thoracic spinous processes legitimately interdigitate 8-15 mm past the
    # disc plane, and hard s-gating them amputated real bone (measured as
    # cleared_out_of_band). The pedicle-root race decides who owns the
    # overhang; frozen neighbors' voxels are never in the domain either way.
    rb2 = (P["seed_radius_mm"] + 4.0) ** 2
    body_zone = d2_slab <= rb2
    in_s = (s_f >= s_lo_all) & (s_f <= s_hi_all)
    in_s_arch = ((s_f >= s_lo_all - P["arch_overhang_mm"])
                 & (s_f <= s_hi_all + P["arch_overhang_mm"]))
    D = (band_s | pool_s | halo) & np.where(body_zone, in_s, in_s_arch)
    markers[~D] = 0
    # uniform-speed geodesic competition: distance is the cost. (A depth
    # priority was tried and convicted: laminae form one connected deep river,
    # so the first label entering ran the whole posterior column.)
    labels_ws = watershed(np.zeros(D.shape, dtype=np.uint8), markers=markers, mask=D)

    # ---- phase 2: arch rebuild from pedicle roots ------------------------
    # The flood race misassigns posterior elements (spinous/laminae) because
    # facet joints are bone-continuous. Anatomy fix: regrow the arch from
    # each level's PEDICLE ROOTS - the only bone corridors crossing the
    # body/arch frontier - with a waist-severed identity race that cannot
    # leak across thin facet necks (see arch_repartition).
    A = D & ~body_zone
    if A.any():
        labels_ws, arch_rec = arch_repartition(labels_ws, body_zone & D, A,
                                               list(assign), zooms, vox_mm3)
        rec.update(arch_rec)
        for nm in arch_rec.get("arch_rootless", []):
            qa["flags"].append(f"ARCH_NO_ROOT_{nm}")

    out = seg.copy()
    out[band_mask] = 0  # includes out-of-band overreach, which is cleared
    out_s = out[:, :, slab]
    sel = labels_ws > 0
    out_s[sel] = labels_ws[sel].astype(seg.dtype)
    rec["unreached_pool_mm3"] = round(float((D & ~sel & (seg_s == 0)).sum()) * vox_mm3, 1)
    cleared = band_mask & (out == 0)
    rec["cleared_out_of_band_mm3"] = round(float(cleared.sum()) * vox_mm3, 1)
    rec["cleared_arch_mm3"] = round(float((cleared[:, :, slab] & ~body_zone).sum())
                                    * vox_mm3, 1)
    return out



def _spinous_frame(seg, ct, zooms):
    """Shared geometry for the spinous meter and repair (stage 2g): column
    centerline, posterior side sign, per-z corridor front = body posterior
    edge + canal depth (DISH cannot fuse the body edge away), and per-level
    body z-bands from the 18 mm centerline cylinder - bodies are
    disc-cut-validated, the trusted identity source.

    Takes: seg, ct, zooms.
    Does: computes the shared spinous-corridor geometry used by the meter and
        stage 2g.
    Returns: dict with centerline (cx, cy), posterior sign, per-slice corridor
        front y0, and per-level body bands; None when the column is too short.
    """
    cx, cy = _centerline_xy(seg)
    if cx is None:
        return None
    nz = seg.shape[2]
    rb = 18.0
    counts = np.zeros((nz, 25), dtype=np.int64)
    y_lo = np.full(nz, np.nan)
    y_hi = np.full(nz, np.nan)
    d_all = np.full(nz, np.nan)
    d_bod = np.full(nz, np.nan)
    for z in range(nz):
        i, j = np.nonzero(seg[:, :, z] > 0)
        if i.size < 20:
            continue
        d2 = (((i - cx[z]) * zooms[0]) ** 2 + ((j - cy[z]) * zooms[1]) ** 2)
        sel = d2 <= rb * rb
        if sel.sum() < 20:
            continue
        labs = seg[i[sel], j[sel], z]
        counts[z] = np.bincount(labs, minlength=25)
        y_lo[z], y_hi[z] = np.percentile(j[sel], 5), np.percentile(j[sel], 95)
        d_all[z], d_bod[z] = j.mean(), j[sel].mean()
    ok = ~np.isnan(y_lo)
    if ok.sum() < 5:
        return None
    psign = 1 if np.nanmean(d_all - d_bod) >= 0 else -1
    idx = np.arange(nz)
    yB = np.interp(idx, idx[ok], (y_hi if psign > 0 else y_lo)[ok])
    yB = ndimage.median_filter(yB, 11)
    off = P["imb_canal_mm"] / zooms[1]
    y0 = np.round(yB + off if psign > 0 else yB - off).astype(int)
    bands = {}
    for lid in range(1, 25):
        c = counts[:, lid].astype(np.float64)
        if c.sum() < 100:
            continue
        cw = np.cumsum(c) / c.sum()
        bands[lid] = (int(np.searchsorted(cw, 0.04)),
                      int(np.searchsorted(cw, 0.96)))
    return {"cx": cx, "cy": cy, "psign": psign, "y0": y0, "bands": bands}


def _imbrication_cm3(seg, ct, zooms, vox_mm3, lids=None, tol_mm=5.0,
                     frame=None):
    """Spinous UPWARD-VIOLATION meter (one-sided tripwire for the
    imbrication error class). In the MIDLINE posterior corridor (|x-cx| <=
    7 mm, posterior of body edge + canal) the only structures are spinous
    blades and interspinous bone, and blades angle strictly CAUDALLY: a
    blade droops BELOW its vertebra, never reaches above it. Corridor volume
    carrying a label whose own body band lies below (z > band_top + tol) is
    therefore anatomically impossible - it is a blade wearing the label of
    the level below (measured failure mode: the T9..L2 one-down chain).
    The drooped part of a stolen blade z-overlaps the thief's own band and
    is invisible to any z-test - ONLY root-attachment can decide it - so
    this meter is a tripwire, not a complete count.
    Returns (total_cm3, per_level_dict). Pass frame= to reuse (and to
    measure before/after with IDENTICAL geometry).

    Takes: seg, ct, zooms, vox_mm3, lids (levels to score, default all
        present), tol_mm, frame (optional precomputed _spinous_frame so before
        / after are measured in identical geometry).
    Does: scores the spinous upward-violation meter defined above.
    Returns: (total violating volume in cm3, per-level dict of offenders).
    """
    fr = frame if frame is not None else _spinous_frame(seg, ct, zooms)
    if fr is None:
        return 0.0, {}
    if lids is None:
        lids = [int(v) for v in np.unique(seg) if v != 0]
    bone = ct >= P["bone_hu"]
    cx, psign, y0 = fr["cx"], fr["psign"], fr["y0"]
    ztop = {l: fr["bands"][l][1] for l in lids if l in fr["bands"]}
    tol = int(round(tol_mm / zooms[2]))
    hw = max(int(round(7.0 / zooms[0])), 2)
    upv = {lid: 0 for lid in ztop}
    for z in range(seg.shape[2]):
        active = [l for l, zt in ztop.items() if z > zt + tol]
        if not active:
            continue
        xc = int(round(cx[z]))
        xs = slice(max(xc - hw, 0), xc + hw + 1)
        ys = (slice(y0[z], None) if psign > 0
              else slice(0, max(y0[z] + 1, 0)))
        s2 = seg[xs, ys, z]
        b2 = bone[xs, ys, z]
        v = s2[(s2 > 0) & b2]
        if not v.size:
            continue
        bc = np.bincount(v, minlength=25)
        for l in active:
            upv[l] += int(bc[l])
    per = {ID_TO_NAME[l]: round(n * vox_mm3 / 1e3, 2)
           for l, n in upv.items() if n * vox_mm3 > 20.0}
    total = round(float(sum(n for n in upv.values())) * vox_mm3 / 1e3, 2)
    return total, per


def stage2g_imbrication(seg, ct, affine, zooms, vox_mm3, qa):
    """CAUDAL-FLOW spinous repair: blade identity = ROOT ATTACHMENT,
    propagated strictly downward through consecutive slices.

    THE DEFECT CLASS (measured in the v8 full-resolution review):
    thoracolumbar spinous blades IMBRICATE - the root joins the ring inside
    its own disc band and the blade droops caudally past the next level, so
    the tip lies Euclidean-nearest the vertebra BELOW. Every distance race
    therefore hands the drooping half one level down (the T9..L2 one-down
    chain), and on DISH no thickness signal can veto it: the ossified
    interspinous bridge is a midline sagittal SHEET, bone-continuous with
    both blades and broad in its own cross-section (the "hier" multi-scale
    peel ablation measured upward-violation 0.33 -> 7.68 cm3 - worse than
    the race it tried to fix). One invariant survives all of it: NOTHING in
    the midline posterior corridor grows toward the head.

    Mechanism: the corridor's anterior STRIP (imb_strip_mm at the ring,
    where the junctions live) keeps its labels - junction identity is
    race-derived from pedicle roots and audit-verified. The corridor mass
    posterior of the strip is re-derived in ONE top-to-bottom sweep over
    consecutive axial slices: markers at slice z = strip labels at z, plus
    the z+1 assignment wherever its (x,y) footprint continues into bone at
    z, extended in-plane by a uniform 2D watershed. A label can only ENTER
    the corridor at its junction and FLOW DOWN, so a lower level can never
    claim bone above its own band, while a drooping blade carries its
    root's label to the tip; descending fronts of consecutive levels meet
    inside the interspinous sheet. (This is the consecutive-slice
    "temporal outline propagation" idea, made safe by the anatomical
    direction constraint - undirected chaining was tried and reverted, see
    stage 2f docstring.)

    Recolor-only: the raw envelope is preserved, unreached labeled voxels
    keep their labels, nothing is deleted. Gates: upward-violation must
    improve (or stay ~0 with negligible churn), the mixed-piece meter must
    not increase, the audit must not degrade, per-level net shift <=
    imb_max_shift_cm3 - else full revert.

    Takes: seg, ct, affine, zooms, vox_mm3, qa (record written under key
        'imbrication').
    Does: the caudal-flow spinous repair described above.
    Returns: the repaired label volume, or the input unchanged after a gated
        full revert.
    """
    fr = _spinous_frame(seg, ct, zooms)
    if fr is None:
        qa["imbrication"] = {"skipped": "no frame"}
        return seg
    cx, psign, y0 = fr["cx"], fr["psign"], fr["y0"]
    bone = ct >= P["bone_hu"]
    u0, up0 = _imbrication_cm3(seg, ct, zooms, vox_mm3, frame=fr)
    hw = max(int(round(P["imb_half_mm"] / zooms[0])), 2)
    sw = int(round(P["imb_strip_mm"] / zooms[1]))
    zs = np.nonzero(seg.any(axis=(0, 1)))[0]
    if zs.size == 0:
        qa["imbrication"] = {"skipped": "empty"}
        return seg
    # the slab must FOLLOW THE BLADES, not the body centroid: vertebral
    # rotation (scoliosis) swings the spinous line laterally, and a slab
    # centered on the body clips the blade - the clipped part keeps its
    # stolen label and a fresh mixed piece is minted at the slab edge
    # (measured: badcut 35 -> 36, upv only halved, on the body-centered
    # first attempt)
    nz_ = seg.shape[2]
    bx = np.full(nz_, np.nan)
    wide = max(int(round(25.0 / zooms[0])), 3)
    for z in range(int(zs[0]), int(zs[-1]) + 1):
        xc = int(round(cx[z]))
        xs = slice(max(xc - wide, 0), xc + wide + 1)
        ys = (slice(y0[z], seg.shape[1]) if psign > 0
              else slice(0, max(y0[z] + 1, 0)))
        bb = bone[xs, ys, z] & (seg[xs, ys, z] > 0)
        if bb.sum() >= 10:
            bx[z] = xs.start + np.nonzero(bb)[0].mean()
    okb = ~np.isnan(bx)
    if okb.sum() >= 5:
        idxz = np.arange(nz_)
        bx = np.interp(idxz, idxz[okb], bx[okb])
        bx = ndimage.gaussian_filter1d(bx, 4.0)
    else:
        bx = cx
    out = seg.copy()
    prev = None
    unreached = 0
    for z in range(int(zs[-1]), int(zs[0]) - 1, -1):
        xc = int(round(bx[z]))
        xs = slice(max(xc - hw, 0), xc + hw + 1)
        if psign > 0:
            cys = slice(min(y0[z], seg.shape[1]), seg.shape[1])
            sys_ = slice(min(y0[z], seg.shape[1]),
                         min(y0[z] + sw, seg.shape[1]))
        else:
            cys = slice(0, max(y0[z] + 1, 0))
            sys_ = slice(max(y0[z] + 1 - sw, 0), max(y0[z] + 1, 0))
        c2 = np.zeros(seg.shape[:2], dtype=bool)
        c2[xs, cys] = bone[xs, cys, z]
        s2m = np.zeros(seg.shape[:2], dtype=bool)
        s2m[xs, sys_] = bone[xs, sys_, z]
        lab2 = out[:, :, z]
        m2 = c2 & (lab2 > 0)
        if not m2.any():
            prev = None
            continue
        # a strip voxel SEEDS the flow only when its label is BAND-
        # CONSISTENT (junctions live at their own band by definition);
        # a mislabeled strip patch - the near-root end of a stolen blade -
        # is flow territory and gets re-derived like the rest (keeping it
        # left residual violations in the root belt and minted a mixed
        # piece per blade: badcut 35 -> 36 on the first attempt)
        seed2 = s2m & m2
        if seed2.any():
            ii, jj = np.nonzero(seed2)
            lv = lab2[ii, jj].astype(int)
            zb = np.array([fr["bands"].get(l, (z, z))[0] for l in lv])
            zt = np.array([fr["bands"].get(l, (z, z))[1] for l in lv])
            tol2 = int(round(6.0 / zooms[2]))
            bad_ = (z < zb - tol2) | (z > zt + tol2)
            seed2[ii[bad_], jj[bad_]] = False
        grow2 = m2 & ~seed2
        mk = np.where(seed2, lab2, 0).astype(np.int32)
        if prev is not None:
            take = (mk == 0) & m2 & (prev > 0)
            mk[take] = prev[take]
        if (mk > 0).any():
            ws = watershed(np.zeros(mk.shape, dtype=np.uint8), markers=mk,
                           mask=m2)
        else:
            ws = mk
        new2 = np.where(grow2 & (ws > 0), ws.astype(lab2.dtype), lab2)
        unreached += int((grow2 & (ws == 0)).sum())
        out[:, :, z] = new2
        prev = np.where(m2, new2, 0)
    # CONVERGENCE (same treatment as stage 2f): per-slice reassignment
    # leaves ragged seams and strands old-label slivers at the slab and
    # strip boundaries - unconverged, five levels fragmented into dozens
    # of components (measured: T10 -> 44). Volume-preserving interface
    # majority vote, then fuse detached slivers into their surrounding
    # label (nothing deleted), inside the changed-region bbox.
    diff = out != seg
    if diff.any():
        padv = np.ceil(10.0 / np.asarray(zooms)).astype(int)
        bb = _bbox_pad(ndimage.find_objects(diff.astype(np.uint8))[0],
                       padv, seg.shape)
        sub = majority_filter(out[bb].copy(), iters=1)
        sub = absorb_orphans(sub, vox_mm3, max_mm3=1500.0, delete_below=0.0)
        keep = out[bb] > 0
        out[bb] = np.where(keep, np.where(sub > 0, sub, out[bb]), 0)
    u1, up1 = _imbrication_cm3(out, ct, zooms, vox_mm3, frame=fr)
    cx_med = int(round(np.median(cx)))
    mix_before = _badcut_pieces(seg, bone, zooms, cx_med)
    mix_after = _badcut_pieces(out, bone, zooms, cx_med)
    changed = int((out != seg).sum())
    deltas = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        d = (int((out == lid).sum()) - int((seg == lid).sum())) * vox_mm3 / 1e3
        if abs(d) > 0.05:
            deltas[ID_TO_NAME[lid]] = round(d, 2)
    _, s_before = audit(seg, affine, vox_mm3)
    _, s_after = audit(out, affine, vox_mm3)
    bad = lambda s: (s["n_fragmented"] + s["n_empty"] + s["n_size"]
                     + s["n_order"])
    rec = {"changed_cm3": round(changed * vox_mm3 / 1e3, 2),
           "unreached_mm3": round(unreached * vox_mm3, 1),
           "upv_cm3_before": u0, "upv_cm3_after": u1,
           "upv_per_level_before": up0, "upv_per_level_after": up1,
           "badcut_pieces_before": mix_before,
           "badcut_pieces_after": mix_after,
           "volume_deltas_cm3": deltas}
    qa["imbrication"] = rec
    over = [n for n, d in deltas.items()
            if abs(d) > P["imb_max_shift_cm3"]]
    ok_meter = (u1 < u0 - 0.2) or (u0 <= 0.2 and u1 <= u0 + 0.05
                                   and changed * vox_mm3 / 1e3 <= 2.0)
    rec["gate"] = {"ok_meter": bool(ok_meter),
                   "audit_bad_before_after": [bad(s_before), bad(s_after)]}
    if (not ok_meter or mix_after > mix_before
            or bad(s_after) > bad(s_before) or over):
        rec["reverted_all"] = True
        if over:
            rec["over_shift"] = over
        qa["flags"].append("IMBRICATION_REVERTED_ALL")
        return seg
    return out


def stage2b_arbitrate(seg, raw, ct, affine, zooms, vox_mm3, qa):
    """Takes: seg, raw, ct, affine, zooms, vox_mm3, qa.
    Does: finds suspect bands and runs arbitrate_band on each; a band edit is
        kept only when its audit badness strictly improves, and the band's
        imbrication meter is recorded before and after (report-only here; stage
        2g owns the repair).
    Returns: the label volume with all accepted band edits applied.
    """
    bands, suspects, present = find_suspects(seg, raw, affine, vox_mm3)
    qa["suspects"] = suspects
    if not bands:
        return seg
    cxs, cys, rho_s = column_profile(seg, raw, ct, zooms)

    def badness(s, names):
        rows_, _ = audit(s, affine, vox_mm3)
        bad = 0
        for r in rows_:
            nm = r["name"].replace("vertebrae_", "")
            if nm in names:
                bad += sum(1 for f in r["flags"]
                           if f.startswith(("SIZE", "FRAG", "EMPTY", "ORDER")))
                ratio = r.get("size_ratio_vs_neighbors")
                if ratio is not None and ratio > P["suspect_size_hi"]:
                    bad += 1
        return bad

    out = seg
    for band in bands:
        if {"C1", "C2"} & set(band):
            qa["flags"].append(f"ATLAS_AXIS_SKIP band={band}: no disc plane at C1/C2")
            continue
        i0, i1 = present.index(band[0]), present.index(band[-1])
        above = present[i0 - 1] if i0 > 0 else None
        below = present[i1 + 1] if i1 + 1 < len(present) else None
        band_lids = [NAME_TO_ID[n] for n in band]
        u0, _ = _imbrication_cm3(out, ct, zooms, vox_mm3, band_lids)
        cand = arbitrate_band(out, raw, ct, band, (below, above), cxs, cys, rho_s,
                              zooms, vox_mm3, qa)
        if cand is None:
            continue
        b0, b1 = badness(out, set(band)), badness(cand, set(band))
        u1, _ = _imbrication_cm3(cand, ct, zooms, vox_mm3, band_lids)
        qa["bands"][-1]["badness_before_after"] = [b0, b1]
        qa["bands"][-1]["imbrication_upv_cm3_before_after"] = [u0, u1]
        if b1 >= b0:
            qa["flags"].append(f"REVERTED band={band}: badness {b0}->{b1}")
        else:
            # imbrication UPV is recorded but NOT gated here: the band edit
            # owns body identity (its win), while blade imbrication is owned
            # and repaired downstream by stage 2g - vetoing the whole band
            # for a 2-3 cm3 blade steal would throw away 40+ cm3 of body
            # fixes (measured: this exact veto reverted the T5..L2 band).
            out = cand
    return out


# --------------------------------------------------------------- stage 2c --
def _adjacent_interfaces(seg):
    """One sweep over 3 axes -> {(a,b): Nx3 voxel points}, a<b.

    Takes: seg.
    Does: collects the voxel coordinates of every label-label interface by face
        adjacency.
    Returns: dict mapping ordered label pairs (a, b) to N x 3 coordinate
        arrays.
    """
    out = {}
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        m = (x > 0) & (y > 0) & (x != y)
        if not m.any():
            continue
        i = np.nonzero(m)
        a = np.minimum(x[m], y[m]).astype(np.int32)
        b = np.maximum(x[m], y[m]).astype(np.int32)
        pts = np.c_[i[0], i[1], i[2]]
        pts[:, ax] += 1
        key = a.astype(np.int64) * 100 + b
        for k in np.unique(key):
            sel = key == k
            out.setdefault((int(k) // 100, int(k) % 100), []).append(pts[sel])
    return {k: np.vstack(v) for k, v in out.items()}


def _planarity_mm(pts, zooms):
    """Takes: pts (N x 3 interface voxel coordinates), zooms.
    Does: fits a plane by SVD in millimeter space and measures the RMS out-of-
        plane residual, the flatness meter behind stage 2c (real joints
        interdigitate; near-planar means geometric arbitration).
    Returns: RMS distance in mm, or None for degenerate point sets.
    """
    if pts is None or len(pts) < 50:
        return None
    Q = pts * np.asarray(zooms)
    Q = Q - Q.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.sqrt(((Q @ Vt[2]) ** 2).mean()))


def _pair_pts(seg, a, b):
    """Takes: seg, a, b (label ids).
    Does: collects voxel coordinates on both sides of the a | b interface (face
        adjacency).
    Returns: N x 3 int array, or None when the pair does not touch.
    """
    pts = []
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        m = ((x == a) & (y == b)) | ((x == b) & (y == a))
        if m.any():
            i = np.nonzero(m)
            pp = np.c_[i[0], i[1], i[2]]
            pp[:, ax] += 1
            pts.append(pp)
    return np.vstack(pts) if pts else None


def stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa):
    """Evidence-local re-arbitration of guillotine-flat adjacent interfaces
    (v3-plan P1b). Anatomy interdigitates, so a near-planar label interface
    is a fingerprint of geometric arbitration, not of the joint. For every
    adjacent pair whose interface planarity RMS is below the trigger:
    re-solve ONLY a small collar around the interface - domain is the two
    labels inside the collar, seeds are their voxels just outside, priority
    is the (negative) smoothed HU so the boundary settles across the darkest
    surface of the cleft (cortical endplate - disc - endplate reads
    bright-dark-bright, and the valley belongs to neither side). Each pair
    is accepted only if planarity strictly improves, the swap stays bounded,
    and neither label fragments; otherwise it reverts. Flags never edit.

    Takes: seg, ct, zooms, vox_mm3, qa (per-pair records appended under
        'polish').
    Does: the HU-valley interface re-solve described above.
    Returns: the label volume with accepted per-pair boundary re-solves
        applied.
    """
    out = seg.copy()
    pairs = _adjacent_interfaces(out)
    adj = {tuple(sorted((NAME_TO_ID[TOP_DOWN[k]], NAME_TO_ID[TOP_DOWN[k + 1]])))
           for k in range(len(TOP_DOWN) - 1)}
    for (a, b) in sorted(pairs):
        if (a, b) not in adj:
            continue
        pts = pairs[(a, b)]
        p0 = _planarity_mm(pts, zooms)
        area = len(pts) * (vox_mm3 ** (2 / 3))
        rec = {"pair": f"{ID_TO_NAME[b]}|{ID_TO_NAME[a]}",
               "contact_mm2": round(area, 0),
               "planarity_before": None if p0 is None else round(p0, 2)}
        qa["polish"].append(rec)
        if (p0 is None or area < P["polish_min_contact_mm2"]
                or p0 >= P["polish_trigger_mm"]):
            rec["action"] = "skip"
            continue
        lo = np.maximum(pts.min(0) - 2, 0)
        hi = np.minimum(pts.max(0) + 3, np.asarray(out.shape))
        pad = np.ceil((P["polish_collar_mm"] + P["polish_seed_shell_mm"] + 2)
                      / np.asarray(zooms)).astype(int)
        sl = tuple(slice(max(int(l - p), 0), min(int(h + p), n))
                   for l, h, p, n in zip(lo, hi, pad, out.shape))
        sub = out[sl]
        iface = np.zeros(sub.shape, dtype=bool)
        iface[tuple((pts - [s.start for s in sl]).T)] = True
        d_if = ndimage.distance_transform_edt(~iface, sampling=zooms)
        collar = d_if <= P["polish_collar_mm"]
        shell = (d_if > P["polish_collar_mm"]) & \
                (d_if <= P["polish_collar_mm"] + P["polish_seed_shell_mm"])
        ab = (sub == a) | (sub == b)
        R = collar & ab
        seeds = np.where(shell & ab, sub, 0).astype(np.int32)
        if not (seeds == a).any() or not (seeds == b).any() or not R.any():
            rec["action"] = "skip_no_seeds"
            continue
        if P["polish_priority"] == "hu":
            # boundary settles across the darkest surface of the cleft
            # (joint space / disc): fronts traverse bone freely, the HU
            # valley is claimed last, and each endplate stays with its own
            # vertebra. (|grad HU| puts the line on an endplate ridge
            # instead, which annexes thick discs to one side.)
            prio = -ndimage.gaussian_filter(
                np.clip(ct[sl].astype(np.float32), -200, 1500),
                sigma=[0.8 / z for z in zooms])
        else:
            prio = ndimage.gaussian_gradient_magnitude(
                ct[sl].astype(np.float32), sigma=[1.0 / z for z in zooms])
        ws = watershed(prio, markers=seeds, mask=R | (seeds > 0))
        cand = sub.copy()
        m_new = R & (ws > 0)
        cand[m_new] = ws[m_new].astype(cand.dtype)
        if _polish_gate(rec, sub, cand, a, b, zooms, "ws", p0):
            out[sl] = cand
        # (A random-walker fallback - Grady 2006, the v3-plan P1b candidate -
        # was evaluated here and dropped: scikit-image's solver returns
        # degenerate probabilities on these anisotropic thin-cleft ROIs, so
        # it never produced an acceptable candidate. Pairs whose watershed
        # candidate fails the gate simply revert.)
    return out


def _polish_gate(rec, sub, cand, a, b, zooms, tag, p0):
    """Accept iff planarity strictly improves, swap bounded, no new fragments.

    Takes: rec (QA record, mutated), sub / cand (labels in the collar box
        before and after), a, b (the pair), zooms, tag, p0 (planarity before).
    Does: the per-pair acceptance test of stage 2c: planarity must strictly
        improve by polish_min_gain_mm, the swap stays under
        polish_max_shift_frac of the smaller label, and neither label may
        fragment.
    Returns: True when the candidate is accepted.
    """
    m = (cand == a) | (cand == b)
    swapped = int(((cand != sub) & m).sum())
    n_min = min(int((sub == a).sum()), int((sub == b).sum()))
    p1 = _planarity_mm(_pair_pts(cand, a, b), zooms)
    rec[f"planarity_after_{tag}"] = None if p1 is None else round(p1, 2)
    rec[f"swapped_vox_{tag}"] = swapped
    ok = (p1 is not None and swapped > 0
          and swapped <= P["polish_max_shift_frac"] * max(n_min, 1)
          and p1 > p0 + P["polish_min_gain_mm"])
    if ok:
        for lab in (a, b):
            _, c_new = _components(cand == lab)
            _, c_old = _components(sub == lab)
            if int((c_new > 0).sum()) > int((c_old > 0).sum()):
                ok = False
                rec["fragment_guard"] = ID_TO_NAME[lab]
                break
    rec["action"] = tag if ok else f"revert_{tag}"
    return bool(ok)


# --------------------------------------------------------------- stage 2e --
def _centerline_xy(seg):
    """Takes: seg.
    Does: per-slice centroid of all labeled voxels, interpolated across empty
        slices and smoothed: the column centerline used by every sheared-frame
        stage.
    Returns: (cx, cy) float arrays over z, or (None, None) when too few labeled
        slices exist.
    """
    nz = seg.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in np.nonzero((seg > 0).any(axis=(0, 1)))[0]:
        pts = np.nonzero(seg[:, :, z])
        cx[z], cy[z] = pts[0].mean(), pts[1].mean()
    ok = ~np.isnan(cx)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return None, None
    for arr in (cx, cy):
        arr[ok] = ndimage.gaussian_filter1d(arr[ok], 10)
        m = np.isnan(arr)
        arr[m] = np.interp(np.nonzero(m)[0], idx, arr[idx])
    return cx, cy


def _shift2d_stack(vol, sx, sy):
    """Integer per-slice (x, y) shifts with zero fill - exactly invertible
    with (-sx, -sy). Shear-straightens a curved column so fixed-index
    sagittal/coronal planes follow the anatomy along the whole spine.

    Takes: vol, sx / sy (integer per-slice shifts).
    Does: shifts every z slice in-plane with zero fill; integer shifts make the
        shear-straightening exactly invertible with negated shifts.
    Returns: the shifted volume.
    """
    out = np.zeros_like(vol)
    nx, ny, nz = vol.shape
    for z in range(nz):
        dx, dy = int(sx[z]), int(sy[z])
        xs0, xs1 = max(0, -dx), min(nx, nx - dx)
        ys0, ys1 = max(0, -dy), min(ny, ny - dy)
        out[xs0:xs1, ys0:ys1, z] = vol[xs0 + dx:xs1 + dx, ys0 + dy:ys1 + dy, z]
    return out


def _mv_anchors(seg, ct, zooms, cx, cy):
    """Per-level ANCHOR mass: voxels reachable from the level's BODY CORE
    through its own label within mv_anchor_mm. Bodies are the most reliable
    identity post-refinement (disc cuts), and a mislabeled process TIP is
    45 mm+ from the wrong level's body THROUGH that level's label, so tips
    can never anchor their captor.

    Takes: seg, ct, zooms, cx / cy (column centerline).
    Does: builds the per-level ANCHOR mass described above (reach from the body
        core through the level's own label, bounded by mv_anchor_mm).
    Returns: uint8 volume of anchor labels.
    """
    rb = (P["seed_radius_mm"] + 4.0)
    anchors = np.zeros_like(seg)
    it = max(int(np.ceil(P["mv_anchor_mm"] / min(zooms))), 1)
    pad = np.ceil((P["mv_anchor_mm"] + 4.0) / np.asarray(zooms)).astype(int)
    objs = ndimage.find_objects(seg)
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        sl = _bbox_pad(objs[lid - 1], pad, seg.shape)
        lab = seg[sl] == lid
        xx = (np.arange(sl[0].start, sl[0].stop, dtype=np.float32)[:, None, None]
              - cx[None, None, sl[2].start:sl[2].stop]) * zooms[0]
        yy = (np.arange(sl[1].start, sl[1].stop, dtype=np.float32)[None, :, None]
              - cy[None, None, sl[2].start:sl[2].stop]) * zooms[1]
        core = lab & (xx * xx + yy * yy <= rb * rb)
        if not core.any():
            core = ndimage.binary_erosion(lab, structure=STRUCT6, iterations=2)
        if not core.any():
            continue
        reach = ndimage.binary_dilation(core, structure=STRUCT6, iterations=it,
                                        mask=lab)
        anchors[sl][reach] = lid
    return anchors


def stage2e_multiview_recolor(seg, ct, affine, zooms, vox_mm3, qa):
    """Multiview waist-severed recoloring (recolor-only; envelope-safe).

    Physics: a plate-on-plate contact (two spinous blades touching flat at
    the interspinous space, a process resting on a lamina) is NOT a thin
    waist in 3D - the EDT of a broad surface contact is large, so the 3D
    race cannot see the joint. But in the 2D CROSS-SECTION of the view
    whose plane contains the structure's long axis, the same contact is a
    thin LINE: one erosion severs it, while the structure's own in-plane
    core stays connected end to end. Hard from the axial view, easy from
    the sagittal - so every voxel is judged in all three orthogonal views:

      per view, per slice: erode the 2D bone cross-section by mv_neck_mm
      (plate kisses vanish); each surviving 2D core component is identified
      by which levels' ANCHOR mass it contains (see _mv_anchors); a
      component with exactly ONE anchor level votes that level onto every
      pixel it reconstructs in-plane (uniform 2D watershed); components
      with zero or multiple anchors become BLOCKERS - they claim their own
      shells and abstain, so a vote can never leak back across a severed
      joint.

    A component may only VOTE when the view sees the structure lengthwise:
    its in-plane PCA elongation must exceed mv_elong (blades vote from the
    sagittal view, transverse wings from the coronal; an axial cross-cut
    of a fused tip is compact and abstains). All slicing runs in a
    SHEAR-STRAIGHTENED frame (integer per-slice centerline shifts, exactly
    invertible): on a scoliotic column a fixed plane drifts off-midline
    and chains everything into one component, while the straightened plane
    follows the anatomy end to end. A voxel is recolored only when its
    votes are unanimous for one level different from its current label
    (any conflicting view vetoes). Gates: per adjacent pair the swap is
    bounded by mv_max_shift_frac of the smaller label, the losing label
    must not fragment, and the stage reverts globally unless the audit
    stays at least as clean.

    Takes: seg, ct, affine, zooms, vox_mm3, qa (record under 'multiview').
    Does: the unanimous three-view waist-severed recoloring described above.
    Returns: the recolored volume, or the input after a gated full revert.
    """
    cx, cy = _centerline_xy(seg)
    if cx is None:
        qa["multiview"] = {"flipped_mm3": 0.0}
        return seg
    bone = ct >= P["bone_hu"]
    anchors = _mv_anchors(seg, ct, zooms, cx, cy)
    sx = np.round(cx - np.median(cx)).astype(int)
    sy = np.round(cy - np.median(cy)).astype(int)
    bone_s = _shift2d_stack(bone.astype(np.uint8), sx, sy).astype(bool)
    anch_s = _shift2d_stack(anchors, sx, sy)
    votes = np.zeros(seg.shape, dtype=np.uint8)      # sheared frame
    conflict = np.zeros(seg.shape, dtype=bool)
    SENT = 255
    for ax in range(3):
        n_sl = seg.shape[ax]
        zo2 = tuple(z for i, z in enumerate(zooms) if i != ax)
        for k in range(n_sl):
            idx = [slice(None)] * 3
            idx[ax] = k
            idx = tuple(idx)
            b2 = bone_s[idx]
            if not b2.any():
                continue
            edt2 = ndimage.distance_transform_edt(b2, sampling=zo2)
            core2 = edt2 >= P["mv_neck_mm"]
            if not core2.any():
                continue
            lab2, n2 = ndimage.label(core2)
            if n2 == 0:
                continue
            a2 = anch_s[idx]
            sel = (a2 > 0) & (lab2 > 0)
            comp_map = np.full(n2 + 1, SENT, dtype=np.int32)  # default: blocker
            if sel.any():
                pairs = np.unique(np.stack([lab2[sel], a2[sel]]).astype(np.int64),
                                  axis=1)
                comps, first = np.unique(pairs[0], return_index=True)
                ncnt = np.bincount(pairs[0], minlength=n2 + 1)
                # elongation gate: only lengthwise-seen structures may vote
                ii, jj = np.nonzero(lab2)
                cids = lab2[ii, jj]
                for c, f in zip(comps, first):
                    if ncnt[c] != 1:
                        continue
                    m = cids == c
                    if m.sum() < 12:
                        continue
                    pts = np.c_[ii[m] * zo2[0], jj[m] * zo2[1]]
                    pts = pts - pts.mean(0)
                    cov = pts.T @ pts / len(pts)
                    w = np.linalg.eigvalsh(cov)
                    if w[1] >= P["mv_elong"] ** 2 * max(w[0], 1e-6):
                        comp_map[c] = int(pairs[1, f])
            comp_map[0] = 0
            markers = comp_map[lab2]
            if not (markers > 0).any() or not ((markers > 0) & (markers != SENT)).any():
                continue
            ws2 = watershed(np.zeros_like(markers, dtype=np.uint8),
                            markers=markers, mask=b2)
            v2 = votes[idx]
            c2 = conflict[idx]
            val = ws2.astype(np.uint8)
            good = (val > 0) & (val != SENT)
            c2 |= good & (v2 > 0) & (v2 != val)
            newv = good & (v2 == 0)
            v2[newv] = val[newv]
            votes[idx] = v2
            conflict[idx] = c2
    # back to the original frame (exact inverse shifts)
    votes = _shift2d_stack(votes, -sx, -sy)
    conflict = _shift2d_stack(conflict.astype(np.uint8), -sx, -sy).astype(bool)
    cand = (seg > 0) & (votes > 0) & ~conflict & (votes != seg)
    if not cand.any():
        qa["multiview"] = {"flipped_mm3": 0.0}
        return seg
    out = seg.copy()
    out[cand] = votes[cand]
    # gates: per-pair bound + loser-fragmentation guard
    flips = {}
    pts_from = seg[cand]
    pts_to = votes[cand]
    for a, b in zip(pts_from, pts_to):
        flips[(int(a), int(b))] = flips.get((int(a), int(b)), 0) + 1
    reverted_pairs = []
    for (a, b), n in sorted(flips.items()):
        n_min = min(int((seg == a).sum()), int((seg == b).sum()))
        undo = False
        if n > P["mv_max_shift_frac"] * max(n_min, 1):
            undo = True
        else:
            _, c_new = _components(out == a)
            _, c_old = _components(seg == a)
            if int((c_new > 0).sum()) > int((c_old > 0).sum()):
                undo = True
        if undo:
            m = cand & (seg == a) & (votes == b)
            out[m] = a
            reverted_pairs.append(f"{ID_TO_NAME[a]}->{ID_TO_NAME[b]}")
    qa["multiview"] = {
        "flipped_mm3": round(float((out != seg).sum()) * vox_mm3, 1),
        "pair_flips_mm3": {f"{ID_TO_NAME[a]}->{ID_TO_NAME[b]}":
                           round(n * vox_mm3, 1) for (a, b), n in flips.items()},
        "reverted_pairs": reverted_pairs}
    _, s_before = audit(seg, affine, vox_mm3)
    _, s_after = audit(out, affine, vox_mm3)
    bad = lambda s: s["n_fragmented"] + s["n_empty"] + s["n_size"] + s["n_order"]
    if bad(s_after) > bad(s_before):
        qa["multiview"]["reverted_all"] = True
        qa["flags"].append("MULTIVIEW_REVERTED_ALL")
        return seg
    return out


# --------------------------------------------------------------- stage 2f --
def _masscut_mm2(seg, edtb, vox_mm3, thresh=2.5):
    """Area of label-label boundary crossing bone thicker than thresh -
    the violation meter for the core-integrity invariant.

    Takes: seg, edtb (3D bone EDT in mm), vox_mm3, thresh.
    Does: measures the mass-cut meter described above.
    Returns: label-boundary area crossing bone thicker than thresh, in mm2.
    """
    m = np.zeros(seg.shape, dtype=bool)
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        b = (x > 0) & (y > 0) & (x != y)
        m[tuple(s_hi)] |= b
        m[tuple(s_lo)] |= b
    return float((m & (edtb >= thresh)).sum()) * (vox_mm3 ** (2 / 3))


def _boundary_edt2(mm, edt2):
    """Mean in-plane thickness along the internal label boundary of a
    labeled 2D patch (0 outside). Returns 0.0 when no internal boundary.

    Takes: mm (2D labels of one piece, zero elsewhere), edt2 (in-plane
        thickness map).
    Does: measures how thick the bone is along the piece's INTERNAL label
        boundary, the physics test of stages 2f and 2g.
    Returns: mean in-plane thickness along that boundary in mm (0 when there is
        none).
    """
    bmask = np.zeros(mm.shape, dtype=bool)
    for a2 in range(2):
        sh = [slice(None)] * 2
        sl_ = [slice(None)] * 2
        sh[a2] = slice(1, None)
        sl_[a2] = slice(None, -1)
        xx, yy = mm[tuple(sh)], mm[tuple(sl_)]
        bb = (xx > 0) & (yy > 0) & (xx != yy)
        bmask[tuple(sh)] |= bb
        bmask[tuple(sl_)] |= bb
    if not bmask.any():
        return 0.0
    return float(np.mean(edt2[bmask]))


def _badcut_pieces(seg, bone, zooms, cx_med, span=10, thresh=2.6):
    """Direct defect meter: number of supra-neck 2D core pieces (sagittal,
    within +/-span of midline) whose INTERNAL label boundary crosses bone
    thicker than thresh - a boundary through the middle of a rigid piece.
    Mixed pieces whose boundary sits at a thin waist (a legitimate joint
    through a PV-fused pair) do NOT count.

    Takes: seg, bone, zooms, cx_med (median centerline x), span (parasagittal
        half-width in slices), thresh.
    Does: counts the badcut meter described above: mixed supra-neck 2D pieces
        whose internal boundary crosses thick bone; boundaries at thin in-plane
        waists (legitimate fused joints) are excluded.
    Returns: the number of violating pieces.
    """
    n_bad = 0
    for x in range(cx_med - span, cx_med + span + 1):
        if not (0 <= x < seg.shape[0]):
            continue
        b2 = bone[x] | (seg[x] > 0)
        if not b2.any():
            continue
        edt2 = ndimage.distance_transform_edt(b2, sampling=(zooms[1], zooms[2]))
        l2, n2 = ndimage.label((edt2 >= 1.2) & (seg[x] > 0))
        s2 = seg[x]
        for c in range(1, n2 + 1):
            m = l2 == c
            if m.sum() * zooms[1] * zooms[2] < 60:
                continue
            vals = s2[m]
            ids, cnt = np.unique(vals[vals > 0], return_counts=True)
            if ids.size < 2 or (cnt / cnt.sum()).min() < 0.15:
                continue
            if _boundary_edt2(np.where(m, s2, 0), edt2) >= thresh:
                n_bad += 1
    return n_bad


def stage2f_skeleton_relabel(seg, ct, affine, zooms, vox_mm3, qa):
    """CORE-INTEGRITY relabeling - the invariant, enforced in 3D:
    A LABEL BOUNDARY MAY ONLY PASS THROUGH THIN BONE (joint clefts, PV
    plates, necks) - NEVER THROUGH THE INTERIOR OF A THICK MASS. A facet
    knob, a blade, a body is one rigid piece and carries one label.

    Mechanism - PER-PIECE UNIFICATION (surgical, local, no seeds, no
    propagation; the global re-derivations tried first kept leaking
    through this case's pathological fusions and were auto-reverted):

    A mixed in-plane core piece IS the defect by definition: one rigid 2D
    cross-section piece (blade, knob), in-plane separated from its
    neighbors, carrying two labels. Rule: such a piece is unified to its
    majority label IFF physics says the internal boundary is impossible -
    it crosses the piece's THICK interior (in-plane EDT at the internal
    boundary >= skel_cut_mm). If the boundary sits at the piece's thin
    waist it is a legitimate joint through a PV-fused pair - skipped. The
    minority share must be <= 40% (a 50/50 split could be two whole fused
    structures). Runs over all three orthogonal views in the
    shear-straightened frame; recolor-only. Gates: the mixed-piece meter
    must not increase, audit must not degrade, per-level shift <= 4 cm3,
    else full revert.

    Takes: seg, ct, affine, zooms, vox_mm3, qa (record under 'skeleton').
    Does: the per-piece core-integrity surgery described above.
    Returns: the repaired volume, or the input after a gated full revert.
    """
    import gc
    gc.collect()
    # the pipeline crop follows the RAW prediction bbox, which scattered
    # hallucinations inflate to most of the volume; by this stage seg is
    # spine-only, so work on its tight bbox (~5x fewer voxels)
    nzs = np.nonzero(seg)
    if nzs[0].size == 0:
        qa["skeleton"] = {"skipped": "empty"}
        return seg
    padv = np.ceil(8.0 / np.asarray(zooms)).astype(int)
    tight = tuple(slice(max(int(c.min() - p), 0), min(int(c.max() + p + 1), n))
                  for c, p, n in zip(nzs, padv, seg.shape))
    seg_orig = seg
    seg = seg[tight].copy()
    ct = np.ascontiguousarray(ct[tight])
    gc.collect()
    bone = ct >= P["bone_hu"]
    edtb = ndimage.distance_transform_edt(bone, sampling=zooms).astype(np.float32)
    gc.collect()
    mc_before = _masscut_mm2(seg, edtb, vox_mm3)
    cx, cy = _centerline_xy(seg)
    if cx is None:
        qa["skeleton"] = {"skipped": "no centerline"}
        return seg
    sx = np.round(cx - np.median(cx)).astype(int)
    sy = np.round(cy - np.median(cy)).astype(int)
    seg_h = _shift2d_stack(seg, sx, sy)
    bone_h = _shift2d_stack(bone.astype(np.uint8), sx, sy).astype(bool)
    out_h = seg_h.copy()
    for ax in range(3):
        n_sl = seg.shape[ax]
        zo2 = tuple(z for i, z in enumerate(zooms) if i != ax)
        for k in range(n_sl):
            idx = [slice(None)] * 3
            idx[ax] = k
            idx = tuple(idx)
            s2 = out_h[idx]
            lab2d = s2 > 0
            if not lab2d.any():
                continue
            edt2 = ndimage.distance_transform_edt(
                bone_h[idx] | lab2d, sampling=zo2)
            l2, n2 = ndimage.label((edt2 >= P["mv_neck_mm"]) & lab2d)
            if n2 == 0:
                continue
            for pid in range(1, n2 + 1):
                m = l2 == pid
                npx = int(m.sum())
                if npx * zo2[0] * zo2[1] < 60:
                    continue
                vals = s2[m]
                ids, cnt = np.unique(vals, return_counts=True)
                sel = ids > 0
                ids, cnt = ids[sel], cnt[sel]
                if ids.size < 2:
                    continue
                order = np.argsort(cnt)[::-1]
                major = int(ids[order[0]])
                minority_frac = 1.0 - cnt[order[0]] / cnt.sum()
                # physics test: internal boundary thickness within piece
                mm = np.where(m, s2, 0)
                bedt = _boundary_edt2(mm, edt2)
                if bedt < P["skel_cut_mm"]:
                    continue        # boundary at a thin waist: legit joint
                if minority_frac <= 0.40:
                    # split single structure: unify to majority
                    s2[m & (mm != major) & (mm > 0)] = major
                    continue
                # PV-fused multi-structure piece: both sides are real -
                # RELOCATE the internal boundary to the in-plane thickness
                # valley (the kiss/joint plane) instead
                mk = np.zeros_like(mm)
                for lid_ in ids:
                    part = mm == lid_
                    er = ndimage.binary_erosion(part, iterations=3)
                    mk[er if er.any() else part] = lid_
                ws = watershed(-edt2.astype(np.float32), markers=mk, mask=m)
                cand2 = np.where(m, ws, mm).astype(mm.dtype)
                if _boundary_edt2(np.where(m, cand2, 0), edt2) < bedt - 0.2:
                    s2[m] = cand2[m].astype(s2.dtype)
    out = _shift2d_stack(out_h, -sx, -sy)
    del out_h, seg_h, bone_h
    gc.collect()
    keep = seg > 0
    out = np.where(keep, np.where(out > 0, out, seg), 0).astype(seg.dtype)
    del keep
    # per-slice decisions leave ragged seams: converge them with the
    # volume-preserving 26-neighborhood interface majority vote, then fuse
    # any disconnected slivers into their surrounding label (no deletion)
    out = majority_filter(out, iters=2)
    gc.collect()
    out = absorb_orphans(out, vox_mm3, max_mm3=600.0, delete_below=0.0)
    # a recolored sliver left detached from its new label (no labeled
    # shell, so absorb cannot fuse it) reverts to its ORIGINAL label -
    # envelope kept, no new fragments minted
    for lid in [int(v) for v in np.unique(out) if v != 0]:
        cc_, counts_ = _components(out == lid)
        if (counts_ > 0).sum() <= 1:
            continue
        main_ = int(counts_.argmax())
        for comp_id in np.nonzero(counts_)[0]:
            if comp_id == main_ or counts_[comp_id] * vox_mm3 > 600.0:
                continue
            comp_ = cc_ == comp_id
            if counts_[comp_id] * vox_mm3 < P["speck_mm3"]:
                sh_ = ndimage.binary_dilation(comp_, structure=STRUCT26) & ~comp_
                v_ = out[sh_]
                v_ = v_[v_ > 0]
                if v_.size:
                    ids_, cnt_ = np.unique(v_, return_counts=True)
                    out[comp_] = int(ids_[cnt_.argmax()])
                    continue
            if (seg[comp_] != lid).any():
                out[comp_] = seg[comp_]
    gc.collect()
    mc_after = _masscut_mm2(out, edtb, vox_mm3)
    del edtb
    gc.collect()
    cx_med = int(round(np.median(cx)))
    mix_before = _badcut_pieces(seg, bone, zooms, cx_med)
    mix_after = _badcut_pieces(out, bone, zooms, cx_med)
    changed = int((out != seg).sum())
    deltas = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        d = (int((out == lid).sum()) - int((seg == lid).sum())) * vox_mm3 / 1e3
        if abs(d) > 0.05:
            deltas[ID_TO_NAME[lid]] = round(d, 2)
    _, s_before = audit(seg, affine, vox_mm3)
    _, s_after = audit(out, affine, vox_mm3)
    bad = lambda s: s["n_fragmented"] + s["n_empty"] + s["n_size"] + s["n_order"]
    rec = {"changed_mm3": round(changed * vox_mm3, 1),
           "masscut_before_mm2": round(mc_before, 0),
           "masscut_after_mm2": round(mc_after, 0),
           "badcut_pieces_before": mix_before,
           "badcut_pieces_after": mix_after,
           "volume_deltas_cm3": deltas}
    qa["skeleton"] = rec
    # gate on the DIRECT defect meter (split pieces) + audit + bounded
    # per-level shifts; masscut is reported only - the correct boundary
    # through a plate-fused joint is legitimately thick, so thickness
    # alone would veto right fixes.
    over = [n for n, d in deltas.items() if abs(d) > 4.0]
    if mix_after > mix_before or bad(s_after) > bad(s_before) or over:
        rec["reverted_all"] = True
        if over:
            rec["over_shift"] = over
        qa["flags"].append("SKELETON_REVERTED_ALL")
        return seg_orig
    out_full = seg_orig.copy()
    out_full[tight] = out
    return out_full


# --------------------------------------------------------------- stage 2d --
def stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa):
    """Envelope rule: the raw prediction is the outer boundary - real BONE
    inside it gets its CLASS fixed, it is never deleted. Runs last, so it
    reclaims whatever the evidence stages removed and no band re-claimed
    (the model systematically labels a transverse process with the level
    above; stage 1 then removes it as far-from-its-label and, outside a
    suspect band, nothing brought it back). Owner of each dropped bone
    component (>= speck size) is decided by two independent physics:

      1. AXIAL-RING VOTE (primary): in the axial projection a vertebra is
         one closed ring, and a true process fragment is 2D-bone-connected
         to its own ring within its own slices, while ribs cross slices as
         disconnected in-plane islands. Vote = the label sharing the most
         in-slice 2D bone-component pixels with the fragment.
      2. 3D BONE-GEODESIC LINK: masked dilation through CT bone
         (reclaim_link_mm); vote = the label with the most reached voxels.

    The ring vote wins disagreements (both are logged). The fragment is
    re-attached through the stage-1 CT-bone corridor at a widened neck
    (reclaim_neck_mm) when possible. A fragment linked to NO vertebra
    through bone is, by the label definition, not vertebra (pelvic or rib
    hallucinations wearing a vertebra label) - it stays out and is flagged,
    never silently dropped. Sub-bone-HU raw mass stays removed (evidence).

    Takes: seg, raw, ct, zooms, vox_mm3, qa (record under
        'reclaim_cm3_by_level' plus flags).
    Does: the axial-ring envelope reclamation described above.
    Returns: the volume with re-owned raw-labeled fragments attached and fused.
    """
    bone = ct >= P["bone_hu"]
    pool = (raw > 0) & (seg == 0) & bone
    cc, counts = _components(pool)
    st2 = ndimage.generate_binary_structure(2, 1)
    summary = {}
    for comp_id in np.nonzero(counts)[0]:
        vol = float(counts[comp_id]) * vox_mm3
        if vol < P["speck_mm3"]:
            continue
        comp = cc == comp_id
        objs = ndimage.find_objects(comp.astype(np.uint8))[0]
        rl = raw[comp]
        ids_r, cnt_r = np.unique(rl[rl > 0], return_counts=True)
        rec = {"stage": "2d", "vol_mm3": round(vol, 1),
               "raw_label": ID_TO_NAME[int(ids_r[cnt_r.argmax()])]}
        ring_votes = {}
        for z in range(objs[2].start, objs[2].stop):
            cz = comp[:, :, z]
            if not cz.any():
                continue
            lab2, _ = ndimage.label(bone[:, :, z], structure=st2)
            for h in np.unique(lab2[cz]):
                if h == 0:
                    continue
                labs = seg[:, :, z][lab2 == h]
                for l, c in zip(*np.unique(labs[labs > 0], return_counts=True)):
                    ring_votes[int(l)] = ring_votes.get(int(l), 0) + int(c)
        ring = max(ring_votes, key=ring_votes.get) if ring_votes else 0
        pad_vox = np.ceil((P["reclaim_link_mm"] + 2.0) / np.asarray(zooms)).astype(int)
        slc = _bbox_pad(objs, pad_vox, seg.shape)
        it = max(int(np.ceil(P["reclaim_link_mm"] / min(zooms))), 1)
        reach = ndimage.binary_dilation(comp[slc], structure=STRUCT6,
                                        iterations=it,
                                        mask=bone[slc] | comp[slc])
        labs = seg[slc][reach]
        ids3, cnt3 = np.unique(labs[labs > 0], return_counts=True)
        geo = int(ids3[cnt3.argmax()]) if ids3.size else 0
        rec["ring_vote"] = ID_TO_NAME.get(ring, "-")
        rec["geodesic_vote"] = ID_TO_NAME.get(geo, "-")
        lid = ring if ring else geo
        if lid == 0:
            rec["action"] = "UNLINKED_NOT_VERTEBRA"
            qa["flags"].append(
                f"POOL_UNRESOLVED_{rec['raw_label']}_{int(vol)}mm3")
            qa["records"].append(rec)
            continue
        mcc, mcounts = _components(seg == lid)
        main = mcc == int(mcounts.argmax())
        added = _bridge(seg, bone, main, comp, lid, zooms,
                        neck_mm=P["reclaim_neck_mm"])
        seg[comp] = lid
        if added is not None:
            seg[added] = lid
            rec["action"] = "RECLAIMED_BRIDGED"
            rec["bridge_added_mm3"] = round(float(added.sum()) * vox_mm3, 1)
        else:
            rec["action"] = "RECLAIMED_KEPT_SEPARATE"
        nm = ID_TO_NAME[lid]
        summary[nm] = round(summary.get(nm, 0.0) + vol / 1000.0, 2)
        qa["records"].append(rec)
    qa["reclaim_cm3_by_level"] = summary
    # boundary slivers that could not bridge fuse into the label they touch
    # (envelope rule: nothing deleted - delete_below=0)
    return absorb_orphans(seg, vox_mm3, max_mm3=300.0, delete_below=0.0)


def majority_filter(seg, iters=3):
    """Iterated 26-neighborhood majority vote on label-label interface voxels.

    Collapses voxel-scale interdigitation between adjacent labels (the source
    of spurious Euler handles). Label<->label swaps only; the mode must
    strictly beat the current label locally, so the filter converges and is
    approximately volume-preserving.

    Takes: seg, iters.
    Does: iterated 26-neighborhood majority voting restricted to label-label
        interfaces, the volume-preserving seam smoother shared by stages 2f,
        2g, and 3.
    Returns: the smoothed label volume.
    """
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    out = seg.copy()
    for _ in range(iters):
        diff = np.zeros(out.shape, dtype=bool)
        for ax in range(3):
            s_hi = [slice(None)] * 3; s_lo = [slice(None)] * 3
            s_hi[ax] = slice(1, None); s_lo[ax] = slice(None, -1)
            x, y = out[tuple(s_hi)], out[tuple(s_lo)]
            b = (x > 0) & (y > 0) & (x != y)
            diff[tuple(s_hi)] |= b
            diff[tuple(s_lo)] |= b
        diff &= out > 0
        pts = np.nonzero(diff)
        n = pts[0].size
        if n == 0:
            break
        counts = np.zeros((n, 25), dtype=np.uint8)
        shp = out.shape
        for dx, dy, dz in offs:
            xx = np.clip(pts[0] + dx, 0, shp[0] - 1)
            yy = np.clip(pts[1] + dy, 0, shp[1] - 1)
            zz = np.clip(pts[2] + dz, 0, shp[2] - 1)
            counts[np.arange(n), out[xx, yy, zz]] += 1
        cur = out[pts]
        lab_counts = counts[:, 1:]
        mode = lab_counts.argmax(axis=1) + 1
        swap = (mode != cur) & (lab_counts.max(axis=1) > counts[np.arange(n), cur])
        if not swap.any():
            break
        out[pts[0][swap], pts[1][swap], pts[2][swap]] = mode[swap].astype(out.dtype)
    return out


def absorb_orphans(seg, vox_mm3, max_mm3=1200.0, delete_below=100.0):
    """Absorb non-largest components below max_mm3 into the label dominating
    their shell (remove if below delete_below and shell-free; delete_below=0
    disables removal - used after pool reclamation, where the envelope rule
    forbids deleting raw-labeled bone).

    Takes: seg, vox_mm3, max_mm3 (largest non-main component that may fuse),
        delete_below (specks under this with no labeled shell are dropped; 0
        disables all deletion for envelope-safe stages).
    Does: fuses every non-main component into the label surrounding it instead
        of deleting it.
    Returns: the volume with orphan components fused.
    """
    out = seg.copy()
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id or counts[comp_id] * vox_mm3 > max_mm3:
                continue
            comp = cc == comp_id
            sl = _bbox_pad(ndimage.find_objects(comp.astype(np.uint8))[0],
                           np.array([2, 2, 2]), seg.shape)
            comp_l = comp[sl]
            shell = ndimage.binary_dilation(comp_l, iterations=2) & ~comp_l
            votes = out[sl][shell]
            votes = votes[(votes != 0) & (votes != lid)]
            if votes.size >= 5:
                ids, cnt = np.unique(votes, return_counts=True)
                out[comp] = int(ids[cnt.argmax()])
            elif counts[comp_id] * vox_mm3 < delete_below:
                out[comp] = 0
    return out


# ---------------------------------------------------------------- stage 3 --
def stage3_smooth(seg, ct, zooms, vox_mm3, qa):
    """Takes: seg, ct, zooms, vox_mm3, qa (records under 'smooth').
    Does: regularization: interface majority vote, orphan absorption, enclosed-
        hole and directional pit filling, then bounded volume-preserving SDT
        smoothing (changes within max_dev_mm, additions bone-gated, per-label
        Dice and component-count guards with self-revert).
    Returns: the regularized label volume.
    """
    seg = majority_filter(seg, iters=3)
    seg = absorb_orphans(seg, vox_mm3)
    bone = ct >= P["bone_hu"]
    pad_vox = np.ceil(P["smooth_pad_mm"] / np.asarray(zooms))
    objs = ndimage.find_objects(seg)
    per_label = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        sl = _bbox_pad(objs[lid - 1], pad_vox, seg.shape)
        m = seg[sl] == lid
        others = (seg[sl] > 0) & ~m
        m1 = ndimage.binary_fill_holes(m)
        # directional pit/window filling: 2D holes in sagittal and coronal
        # slices catch wall windows and tunnels that are not 3D-enclosed;
        # axial is EXCLUDED so the vertebral canal is never filled.
        for ax in (0, 1):
            fill2d = np.zeros_like(m1)
            for k in range(m1.shape[ax]):
                idx = [slice(None)] * 3
                idx[ax] = k
                fill2d[tuple(idx)] = ndimage.binary_fill_holes(m1[tuple(idx)])
            m1 |= fill2d & ~others
        m_f = ndimage.binary_fill_holes(m1)
        m_f &= ~others
        d_in = ndimage.distance_transform_edt(m_f, sampling=zooms)
        d_out = ndimage.distance_transform_edt(~m_f, sampling=zooms)
        sdt = (d_in - d_out).astype(np.float32)
        s = ndimage.gaussian_filter(sdt, sigma=[P["sigma_mm"] / z for z in zooms])
        target = int(m_f.sum())
        lo, hi = -2.0, 2.0
        tau = 0.0
        for _ in range(40):
            tau = 0.5 * (lo + hi)
            n = int((s > tau).sum())
            if abs(n - target) <= P["vol_tol"] * target:
                break
            lo, hi = (tau, hi) if n > target else (lo, tau)
        per_label[lid] = (sl, m_f, sdt, s, float(tau))
        qa["smooth"].append({"label": ID_TO_NAME[lid],
                             "holes_filled_mm3": round(float((m_f & ~m).sum()) * vox_mm3, 1),
                             "tau_mm": round(float(tau), 3)})
    out = np.zeros_like(seg)
    step = max(64, int(120 / zooms[2]))
    for c0 in range(0, seg.shape[2], step):
        c1 = min(c0 + step, seg.shape[2])
        score = np.full((seg.shape[0], seg.shape[1], c1 - c0), -np.inf, dtype=np.float32)
        lab = np.zeros((seg.shape[0], seg.shape[1], c1 - c0), dtype=seg.dtype)
        for lid, (sl, m_f, sdt, s, tau) in per_label.items():
            i0, i1 = max(sl[2].start, c0), min(sl[2].stop, c1)
            if i0 >= i1:
                continue
            zloc = slice(i0 - sl[2].start, i1 - sl[2].start)
            s_l, sdt_l, m_l = s[:, :, zloc], sdt[:, :, zloc], m_f[:, :, zloc]
            band = np.abs(sdt_l) <= P["max_dev_mm"]
            claim = np.where(band, s_l > tau, m_l)
            marg = np.where(band, s_l - tau, np.where(m_l, 1e3, -1e3)).astype(np.float32)
            claim &= (m_l | bone[sl[0], sl[1], i0:i1])
            sc = score[sl[0], sl[1], i0 - c0:i1 - c0]
            lb = lab[sl[0], sl[1], i0 - c0:i1 - c0]
            upd = claim & (marg > sc)
            sc[upd] = marg[upd]
            lb[upd] = lid
        out[:, :, c0:c1] = lab
    for lid, (sl, m_f, sdt, s, tau) in per_label.items():
        new = out[sl] == lid
        denom = int(new.sum()) + int(m_f.sum())
        dice = 2 * int((new & m_f).sum()) / denom if denom else 1.0
        _, c_new = _components(new)
        _, c_old = _components(m_f)
        rec = next(r for r in qa["smooth"] if r["label"] == ID_TO_NAME[lid])
        rec["dice_vs_presmooth"] = round(dice, 4)
        if dice < P["min_dice"] or int((c_new > 0).sum()) > int((c_old > 0).sum()):
            out[sl][m_f] = lid
            out[sl][(out[sl] == lid) & ~m_f] = 0
            rec["reverted"] = True
    return absorb_orphans(out, vox_mm3)


# --------------------------------------------------------------------- io --
def load_case(case_dir: Path, ct_root: Path):
    """Takes: case_dir (prediction folder holding combined_labels.nii.gz), ct_root
        (data folder holding <case>/ct.nii.gz).
    Does: loads the prediction and its CT on the shared native grid, clipping
        HU to the scanner range.
    Returns: (seg nibabel image, seg uint8 array, HU int16 array, ct nibabel
        image), or None when inputs are missing.
    """
    combined = case_dir / "combined_labels.nii.gz"
    if combined.exists():
        seg_img = nib.load(str(combined))
        seg = np.asarray(seg_img.dataobj).astype(np.uint8)
    else:
        seg_dir = case_dir / "segmentations"
        files = sorted(seg_dir.glob("vertebrae_*.nii.gz"))
        if not files:
            return None
        seg_img = nib.load(str(files[0]))
        seg = np.zeros(seg_img.shape, dtype=np.uint8)
        for f in files:
            name = f.name.replace("vertebrae_", "").replace(".nii.gz", "")
            if name in NAME_TO_ID:
                seg[np.asarray(nib.load(str(f)).dataobj) > 0] = NAME_TO_ID[name]
    ct_path = ct_root / case_dir.name / "ct.nii.gz"
    if not ct_path.exists():
        log.error("%s: no CT at %s (CT evidence is required)", case_dir.name, ct_path)
        return None
    ct_img = nib.load(str(ct_path))
    ct = np.asarray(ct_img.dataobj)
    if ct.shape != seg.shape:
        log.error("%s: CT grid %s != prediction grid %s", case_dir.name, ct.shape, seg.shape)
        return None
    return seg_img, seg, np.clip(ct, -1024, 3071).astype(np.int16), ct_img


def write_case(out_dir: Path, case_id: str, seg_full, seg_img):
    """Takes: out_dir, case_id, seg_full (full-grid labels), seg_img (reference
        nibabel image for grid and header).
    Does: writes combined_labels.nii.gz plus the 24 binary per-vertebra masks
        under segmentations/, matching the input grid exactly.
    Returns: None (files on disk).
    """
    case_out = out_dir / case_id
    (case_out / "segmentations").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg_full.astype(np.uint8), seg_img.affine, seg_img.header),
             str(case_out / "combined_labels.nii.gz"))
    for lid, name in ID_TO_NAME.items():
        m = (seg_full == lid).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, seg_img.affine, seg_img.header),
                 str(case_out / "segmentations" / f"vertebrae_{name}.nii.gz"))


def process_case(case_dir: Path, ct_root: Path, out_dir: Path, report_dir: Path):
    """Takes: case_dir, ct_root, out_dir, report_dir.
    Does: runs the full pipeline on one case: crop to the prediction bbox,
        stages 1, 2a, 2b, 2c, 3, 2d, 2e, 2f, 2g, final meters, envelope
        accounting, and audit, then writes the outputs and the QA JSON.
    Returns: (audit summary before, audit summary after), or None when inputs
        are missing.
    """
    loaded = load_case(case_dir, ct_root)
    if loaded is None:
        return None
    seg_img, seg_full, ct_full, _ = loaded
    zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    vox_mm3 = float(abs(np.linalg.det(seg_img.affine[:3, :3])))
    # crop to the spine region (+margin) for all processing
    pad = np.maximum(np.round(P["crop_margin_mm"] / np.asarray(zooms)).astype(int), 2)
    nz = np.nonzero(seg_full)
    lo = np.maximum(np.array([c.min() for c in nz]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in nz]) + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg_full[sl].copy()
    ct = np.ascontiguousarray(ct_full[sl])
    full_shape = seg_full.shape
    del seg_full, ct_full   # ~1.1 GB standing memory freed for the pipeline
    aff = seg_img.affine.copy()
    aff[:3, 3] = (seg_img.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]

    qa = {"case": case_dir.name, "params": P, "records": [], "bands": [],
          "flags": [], "polish": [], "smooth": []}
    _, s0 = audit(seg, aff, vox_mm3)
    log.info("%s input audit: %s", case_dir.name, s0)
    raw = seg.copy()
    seg = stage1_triage(seg, ct, zooms, vox_mm3, qa["records"])
    seg = stage2a_islands(seg, vox_mm3, qa["records"])
    seg = stage2b_arbitrate(seg, raw, ct, aff, zooms, vox_mm3, qa)
    seg = stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa)
    seg = stage3_smooth(seg, ct, zooms, vox_mm3, qa)
    seg = stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa)
    seg = stage2e_multiview_recolor(seg, ct, aff, zooms, vox_mm3, qa)
    seg = stage2f_skeleton_relabel(seg, ct, aff, zooms, vox_mm3, qa)
    seg = stage2g_imbrication(seg, ct, aff, zooms, vox_mm3, qa)
    upv_t, upv_p = _imbrication_cm3(seg, ct, zooms, vox_mm3)
    qa["imbrication_upv_cm3"] = {"total": upv_t, "per_level": upv_p}
    bone_f = ct >= P["bone_hu"]
    qa["envelope"] = {
        "dropped_bone_cm3": round(float(((raw > 0) & (seg == 0) & bone_f).sum())
                                  * vox_mm3 / 1e3, 2),
        "dropped_subbone_cm3": round(float(((raw > 0) & (seg == 0) & ~bone_f).sum())
                                     * vox_mm3 / 1e3, 2),
        "recolored_cm3": round(float(((raw > 0) & (seg > 0) & (raw != seg)).sum())
                               * vox_mm3 / 1e3, 2),
        "added_beyond_raw_cm3": round(float(((raw == 0) & (seg > 0)).sum())
                                      * vox_mm3 / 1e3, 2)}
    rows, s1 = audit(seg, aff, vox_mm3)
    qa["audit_before"], qa["audit_after"] = s0, s1
    qa["audit_rows_after"] = rows
    log.info("%s output audit: %s", case_dir.name, s1)

    out_full = np.zeros(full_shape, dtype=np.uint8)
    out_full[sl] = seg
    write_case(out_dir, case_dir.name, out_full, seg_img)
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{case_dir.name}_postprocessing_qa.json").write_text(
            json.dumps(qa, indent=1, default=float))
    return s0, s1


def main():
    """Takes: command-line arguments: --pred_dir, --ct_root, --out_dir,
        --report_dir, --case.
    Does: CLI entry point: processes every case folder under pred_dir (or the
        one selected) and logs the before / after audit summaries.
    Returns: None.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--ct_root", required=True, type=Path,
                    help="Root holding <case>/ct.nii.gz (evidence gates need HU).")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--report_dir", type=Path, default=Path("reports"))
    ap.add_argument("--case", action="append", default=None)
    args = ap.parse_args()
    cases = sorted(p for p in args.pred_dir.iterdir() if p.is_dir())
    if args.case:
        want = {c for v in args.case for c in v.split(",")}
        cases = [c for c in cases if c.name in want]
    if not cases:
        log.error("no case folders under %s", args.pred_dir)
        return
    for case_dir in cases:
        process_case(case_dir, args.ct_root, args.out_dir, args.report_dir)
    log.info("done: refined predictions in %s", args.out_dir)


if __name__ == "__main__":
    main()
