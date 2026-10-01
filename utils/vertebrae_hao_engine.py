"""Standalone SuPreM vertebrae post-processing: v2 + anatomical refinement.

This single file contains the complete application; no companion project
Python files are required. This is a packaging-only consolidation of the
validated v2-anatomical candidate, NOT a new segmentation algorithm.

Requirements: Python >=3.10, numpy, scipy, nibabel, scikit-image.
Tested: numpy 1.26.4, scipy 1.15.3, nibabel 5.4.2, scikit-image 0.25.2.
Install into an appropriate environment, for example:
    python -m pip install numpy==1.26.4 scipy==1.15.3 nibabel==5.4.2 scikit-image==0.25.2

Example (replace paths with your own; output must not already exist):
    python postprocessing_vertebrae.py --input AbdomenAtlasDemoPredict --ct-root AbdomenAtlasDemo --output AbdomenAtlasDemoRefined
Optional: --v2-reference STORED_V2_DIRECTORY --case BDMAP_00000031

Input layout:
    INPUT/BDMAP_*/combined_labels.nii.gz  (ORIGINAL predictions, labels 0..24)
    CT_ROOT/BDMAP_*/ct.nii.gz             (matching geometry, calibrated HU)

Stage 1 reproduces the recovered v2 cleanup (16 voxels, 1%, 12 mm).
Stage 2 uses original-union thick cores, ordered L3-to-T4 anatomy and
two-scale foreground geodesics to correct candidate labels L2 through T5.
L5-L3 and T4-C1 binary masks are exactly preserved relative to v2.
Ambiguous evidence abstains; no ground-truth DSC is calculated or claimed.
Body-core consistency does not certify posterior-process correctness.
This is inspired by the supplied architecture report and related work,
not a full reproduction of published trained systems:
    https://github.com/MrGiovanni/SuPreM/blob/main/direct_inference/vertebrae.md#related-work
    https://github.com/BodyMaps/ShapeKit
    https://arxiv.org/abs/2110.12177

Original predictions, stored v2 and existing output directories are never
overwritten. Outputs include combined labels, 24 binary masks, diagnostic
volumes, per-case reports and a manifest hashing this standalone source.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import nibabel as nib
import scipy
from scipy import ndimage
from scipy import ndimage as ndi


# === Stage 1: exact recovered v2 operations ===

VERTEBRA_NAMES = [
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2",
    "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10",
    "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
    "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2",
    "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
]


def bbox_distance_mm(a: tuple[slice, ...], b: tuple[slice, ...], spacing: np.ndarray) -> float:
    """Minimum Euclidean distance between two component boxes in millimetres."""
    gaps = []
    for axis in range(3):
        if a[axis].stop <= b[axis].start:
            gap = b[axis].start - a[axis].stop
        elif b[axis].stop <= a[axis].start:
            gap = a[axis].start - b[axis].stop
        else:
            gap = 0
        gaps.append(gap * spacing[axis])
    return float(np.linalg.norm(gaps))


def clean_label(
    mask: np.ndarray,
    spacing: np.ndarray,
    min_component_voxels: int,
    min_relative_size: float,
    max_satellite_distance_mm: float,
) -> tuple[tuple[slice, slice, slice] | None, np.ndarray, int, int]:
    """Keep the main component and only substantial nearby satellites."""
    coordinates = np.where(mask)
    if coordinates[0].size == 0:
        return None, np.zeros((0, 0, 0), dtype=bool), 0, 0
    roi = tuple(
        slice(int(axis.min()), int(axis.max()) + 1)
        for axis in coordinates
    )
    cropped_mask = mask[roi]
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    components, count = ndimage.label(cropped_mask, structure=structure)

    sizes = np.bincount(components.ravel(), minlength=count + 1)[1:]
    main_id = int(np.argmax(sizes)) + 1
    main_size = int(sizes[main_id - 1])
    boxes = ndimage.find_objects(components)
    main_box = boxes[main_id - 1]
    assert main_box is not None

    keep_ids = [main_id]
    size_threshold = max(min_component_voxels, int(np.ceil(main_size * min_relative_size)))
    for component_id, size in enumerate(sizes, start=1):
        if component_id == main_id or int(size) < size_threshold:
            continue
        box = boxes[component_id - 1]
        if box is not None and bbox_distance_mm(main_box, box, spacing) <= max_satellite_distance_mm:
            keep_ids.append(component_id)

    kept = np.isin(components, keep_ids)
    filled = ndimage.binary_fill_holes(kept)
    return roi, filled, count, int(len(keep_ids))


# === Stage 2: stable cores and foreground geodesic consensus ===

@dataclass(frozen=True)
class Config:
    first: int = 4
    last: int = 13
    core_radii: tuple = (3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7.)
    stability_step_mm: float = .5
    stability_mm: float = 3.
    # Smaller thoracic cores can shrink below 500 mm3 at separation radius.
    # Size alone never accepts a core: bone support, anchors and chain required.
    min_core_mm3: float = 100.
    bone_hu: float = 130.
    min_hu: float = -250.
    min_bone_fraction: float = .5
    anchor_purity: float = .75
    min_gap_mm: float = 12.
    max_gap_mm: float = 50.
    max_xy_step_mm: float = 25.
    max_gap_ratio: float = 1.8
    crop_margin_mm: float = 20.
    roi_lateral_margin_mm: float = 65.
    roi_ap_margin_mm: float = 90.
    core_ap_corridor_mm: float = 80.
    max_roi_voxels: int = 50_000_000
    confidence_margin: float = .08
    core_mismatch_trigger: float = .05


def _validate(raw, baseline, ct, affine, cfg):
    if raw.ndim != 3 or raw.shape != baseline.shape or raw.shape != ct.shape:
        raise ValueError('Raw, baseline and CT must have the same 3D shape')
    for a in (raw, baseline):
        if not np.all(np.isfinite(a)) or not np.all(a == np.rint(a)):
            raise ValueError('Finite integer masks required')
        if np.min(a) < 0 or np.max(a) > 24:
            raise ValueError('Only labels 0..24 are supported')
    if np.shape(affine) != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError('Finite 4x4 affine required')
    if not np.allclose(affine[3], [0,0,0,1], rtol=0, atol=1e-8):
        raise ValueError('Homogeneous affine last row must be [0,0,0,1]')
    if abs(float(np.linalg.det(affine[:3,:3]))) < 1e-10:
        raise ValueError('Nonsingular affine required')
    if not isinstance(cfg.first,int) or not isinstance(cfg.last,int):
        raise ValueError('Integer label interval required')
    if not 1 < cfg.first <= cfg.last < 24:
        raise ValueError('Mutable interval must have an external anchor at both ends')
    if not cfg.core_radii or any(not np.isfinite(r) or r <= 0 for r in cfg.core_radii):
        raise ValueError('Positive finite core radii required')
    for key in ('stability_step_mm', 'stability_mm', 'min_core_mm3',
                'min_gap_mm', 'max_gap_mm', 'max_xy_step_mm', 'crop_margin_mm',
                'roi_lateral_margin_mm', 'roi_ap_margin_mm', 'core_ap_corridor_mm'):
        if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not (0 <= cfg.confidence_margin < 1 and 0 < cfg.anchor_purity <= 1):
        raise ValueError('Invalid confidence or anchor threshold')
    if (not np.isfinite(cfg.bone_hu) or not np.isfinite(cfg.min_hu)
            or cfg.min_hu > cfg.bone_hu):
        raise ValueError('Finite ordered CT thresholds required')
    if not 0 < cfg.min_bone_fraction <= 1 or not 0 <= cfg.core_mismatch_trigger <= 1:
        raise ValueError('Invalid core evidence threshold')
    if (cfg.min_gap_mm > cfg.max_gap_mm or not np.isfinite(cfg.max_gap_ratio)
            or cfg.max_gap_ratio < 1):
        raise ValueError('Invalid spacing constraints')
    if not isinstance(cfg.max_roi_voxels,int) or cfg.max_roi_voxels < 1:
        raise ValueError('Positive integer ROI limit required')


def _anchor(mask, label, affine):
    box = ndi.find_objects(mask, max_label=24)[label-1]
    if box is None:
        return None
    cc, n = ndi.label(mask[box] == label, ndi.generate_binary_structure(3, 3))
    sizes = np.bincount(cc.ravel(), minlength=n+1); sizes[0] = 0
    k = int(sizes.argmax())
    pos = np.asarray(ndi.center_of_mass(cc == k)) + np.array([b.start for b in box])
    return nib.affines.apply_affine(affine, pos)


def _crop(raw, baseline, cfg):
    boxes = ndi.find_objects(raw, max_label=24)
    boxes2 = ndi.find_objects(baseline, max_label=24)
    selected = [b for i in range(cfg.first-2, cfg.last+1)
                for b in (boxes[i], boxes2[i]) if b is not None]
    if not selected:
        return None
    return selected


def _components(depth, radius, raw, baseline, ct, affine, spacing, cfg, lower, upper):
    """Extract union-foreground thick cores, independent of predicted IDs."""
    cc, n = ndi.label(depth > radius, ndi.generate_binary_structure(3, 3))
    counts = np.bincount(cc.ravel(), minlength=n+1)
    boxes = ndi.find_objects(cc)
    volume = float(np.prod(spacing))
    records = []
    loz, hiz = lower[2]-cfg.max_gap_mm, upper[2]+cfg.max_gap_mm
    for cid, box in enumerate(boxes, 1):
        if box is None or counts[cid] * volume < cfg.min_core_mm3:
            continue
        xyz = np.argwhere(cc[box] == cid) + np.array([s.start for s in box])
        idx = np.ravel_multi_index(xyz.T, cc.shape)
        bone = float(np.mean(ct.ravel()[idx] >= cfg.bone_hu))
        if bone < cfg.min_bone_fraction:
            continue
        center = nib.affines.apply_affine(affine, xyz.mean(axis=0))
        if not loz <= center[2] <= hiz:
            continue
        t = np.clip((center[2]-lower[2]) / (upper[2]-lower[2]), 0, 1)
        approx = lower + t*(upper-lower)
        # Broad search corridor only, NOT an estimated spine centerline.
        # A narrow straight-line AP gate rejects genuine kyphotic body cores.
        # Posterior distractors must instead pass the full stable ordered chain.
        if abs(center[0]-approx[0]) > 45 or abs(center[1]-approx[1]) > cfg.core_ap_corridor_mm:
            continue
        evidence = np.bincount(baseline.ravel()[idx].astype(int), minlength=25)
        records.append(dict(id=cid, idx=idx, center=center,
                            size=int(len(idx)), volume_mm3=len(idx)*volume,
                            bone_fraction=bone, evidence=evidence))
    return cc, records


def _stable(a_cc, a, b_cc, b, cfg):
    """One-to-one parent/child support, not just nearby centroid matching."""
    b_lookup = {r['id']: r for r in b}
    stable = []
    for r in a:
        ids, counts = np.unique(b_cc.ravel()[r['idx']], return_counts=True)
        children = [(b_lookup[int(i)], int(n)) for i, n in zip(ids, counts)
                    if int(i) in b_lookup]
        if len(children) != 1:
            continue  # split/vanished core is not a stable vertebra
        child, overlap = children[0]
        if overlap < .95*child['size'] or child['size'] < .1*r['size']:
            continue
        displacement = float(np.linalg.norm(child['center'] - r['center']))
        if displacement > cfg.stability_mm:
            continue
        stable.append(dict(r, child=child, displacement_mm=displacement))
    # A child must never authenticate two competing parent candidates.
    ids = [r['child']['id'] for r in stable]
    return [r for r in stable if ids.count(r['child']['id']) == 1]


def _ordered_chain(records, cfg):
    """Two-best-path DAG search with order, gap, XY and spacing-ratio gates."""
    nodes = sorted(records, key=lambda r: float(r['center'][2]))
    first, last = cfg.first-1, cfg.last+1
    nsteps = last-first+1
    starts = [i for i, r in enumerate(nodes) if r['evidence'][first]/r['size'] >= cfg.anchor_purity]
    ends = [i for i, r in enumerate(nodes) if r['evidence'][last]/r['size'] >= cfg.anchor_purity]
    solutions = []
    for st in starts:
        for end in ends:
            if end <= st:
                continue
            target_gap = (nodes[end]['center'][2]-nodes[st]['center'][2])/(nsteps-1)
            states = {(-1, st): [(0., (st,))]}
            for step in range(1, nsteps):
                nxt = {}
                for (_, current), histories in states.items():
                    for score, path in histories:
                        for j in range(current+1, end+1):
                            if (step == nsteps-1) != (j == end):
                                continue
                            d = nodes[j]['center']-nodes[current]['center']; gap = float(d[2])
                            if not cfg.min_gap_mm <= gap <= cfg.max_gap_mm:
                                continue
                            xy = float(np.linalg.norm(d[:2]))
                            if xy > cfg.max_xy_step_mm:
                                continue
                            cost = .2*((gap-target_gap)/max(target_gap,1))**2 + .2*(xy/cfg.max_xy_step_mm)**2
                            if len(path) > 1:
                                oldgap = nodes[current]['center'][2]-nodes[path[-2]]['center'][2]
                                ratio = max(oldgap/gap, gap/oldgap)
                                if ratio > cfg.max_gap_ratio:
                                    continue
                                cost += np.log(gap/oldgap)**2
                            key = (current, j)
                            nxt.setdefault(key, []).append((score+float(cost), path+(j,)))
                states = {key: sorted(value, key=lambda v:v[0])[:2] for key,value in nxt.items()}
                if not states:
                    break
            for histories in states.values():
                solutions.extend((score, path) for score,path in histories if len(path)==nsteps and path[-1]==end)
    if not solutions:
        return None, {'reason':'no_complete_ordered_chain', 'stable_candidates':len(nodes)}
    solutions.sort(key=lambda x:x[0])
    bestscore, path = solutions[0]
    if len(solutions)>1 and solutions[1][1] != path and solutions[1][0]-bestscore < .15:
        return None, {'reason':'ambiguous_ordered_chain', 'scores':[bestscore,solutions[1][0]]}
    return [nodes[i] for i in path], {'path_cost':bestscore, 'stable_candidates':len(nodes)}


def geodesic_partition(domain, depth, markers, spacing, label_ids, progress=lambda x:None):
    """Foreground-only accumulated path cost; runner-up margin is NOT probability."""
    from skimage.graph import MCP_Geometric
    cost = np.full(domain.shape, np.inf, dtype=np.float64)
    cost[domain] = 1. + 8. / (depth[domain]+.5)**2
    best = np.full(domain.shape, np.inf, dtype=np.float64)
    second = best.copy()
    owner = np.zeros(domain.shape, dtype=np.uint8)
    for label in label_ids:
        seeds = np.argwhere((markers == label) & domain)
        if not len(seeds):
            continue
        progress(f'  foreground geodesic label {label}: {len(seeds)} seeds')
        solver = MCP_Geometric(cost, fully_connected=True, sampling=tuple(spacing))
        distances, _ = solver.find_costs(seeds)
        wins = distances < best
        second[wins] = best[wins]
        best[wins] = distances[wins]
        owner[wins] = label
        other = ~wins
        second[other] = np.minimum(second[other], distances[other])
        del solver, distances
    margin = np.zeros(domain.shape, dtype=np.float32)
    reached = np.isfinite(best)
    margin[reached & ~np.isfinite(second)] = 1.
    both = reached & np.isfinite(second)
    margin[both] = ((second[both]-best[both])/np.maximum(second[both],1e-6)).astype(np.float32)
    return owner, margin


def refine(raw, baseline, ct, affine, cfg=Config(), progress=print):
    _validate(raw, baseline, ct, affine, cfg)
    report = dict(method='v2 + independent stable cores + geodesic consensus',
                  config=asdict(cfg), status='skipped', dsc=None,
                  limitation='Core consistency and structural checks are not ground-truth accuracy.',
                  posterior_policy='No height-based posterior seeds. Unreachable/ambiguous regions retain v2.')
    debug = {}
    def unchanged(reason):
        report.update(reason=reason, changed_voxels=0)
        return baseline.copy(), report, debug
    spacing = nib.affines.voxel_sizes(affine)
    orient = affine[:3,:3]/spacing
    if not np.allclose(orient.T@orient, np.eye(3), atol=1e-4):
        return unchanged('unsupported_sheared_affine')
    lower = _anchor(baseline, cfg.first-1, affine)
    upper = _anchor(baseline, cfg.last+1, affine)
    if lower is None or upper is None:
        return unchanged('missing_external_anchor')
    if upper[2] <= lower[2]+cfg.min_gap_mm:
        return unchanged('invalid_anatomical_anchor_order')
    boxes = _crop(raw, baseline, cfg)
    if boxes is None:
        return unchanged('empty_region')
    margin = np.ceil(cfg.crop_margin_mm/spacing).astype(int)
    roi = tuple(slice(max(0,min(b[i].start for b in boxes)-int(margin[i])),
                      min(raw.shape[i],max(b[i].stop for b in boxes)+int(margin[i]))) for i in range(3))
    # A remote mislabelled island must not expand the expensive dense ROI to
    # the whole scan. Clip in physical RAS space around the reliable endpoints;
    # voxels outside this conservative search box retain baseline unchanged.
    physical_margin = np.array([cfg.roi_lateral_margin_mm, cfg.roi_ap_margin_mm,
                                cfg.max_gap_mm])
    world_lo = np.minimum(lower, upper)-physical_margin
    world_hi = np.maximum(lower, upper)+physical_margin
    corners = np.array([[x,y,z] for x in (world_lo[0],world_hi[0])
                        for y in (world_lo[1],world_hi[1])
                        for z in (world_lo[2],world_hi[2])])
    voxel_corners = nib.affines.apply_affine(np.linalg.inv(affine),corners)
    vmin = np.floor(voxel_corners.min(axis=0)).astype(int)
    vmax = np.ceil(voxel_corners.max(axis=0)).astype(int)+1
    unbounded_roi = roi
    roi = tuple(slice(max(s.start,int(vmin[i])),min(s.stop,int(vmax[i])))
                for i,s in enumerate(roi))
    report['roi_policy'] = 'Anchor-bounded search only; all voxels outside ROI retain v2.'
    report['roi_before_anchor_bound'] = [[s.start,s.stop] for s in unbounded_roi]
    if any(s.stop <= s.start for s in roi):
        return unchanged('empty_anchor_bounded_roi')
    if np.prod([s.stop-s.start for s in roi]) > cfg.max_roi_voxels:
        return unchanged('roi_memory_safety_limit')
    r = raw[roi]; b = baseline[roi]; hu = ct[roi]
    local_affine = affine.copy()
    local_affine[:3,3] = nib.affines.apply_affine(affine, [s.start for s in roi])
    report['roi'] = [[s.start,s.stop] for s in roi]
    shape_domain = (r > 0) & np.isfinite(hu) & (hu >= cfg.min_hu)
    path_domain = ((r > 0) | (b > 0)) & np.isfinite(hu) & (hu >= cfg.min_hu)
    progress(f'  ROI {r.shape}: extracting label-independent foreground thickness')
    depth = ndi.distance_transform_edt(np.pad(shape_domain,1), sampling=spacing)[1:-1,1:-1,1:-1].copy()
    chain = None; report['scale_search'] = []
    for radius in cfg.core_radii:
        cc, recs = _components(depth,radius,r,b,hu,local_affine,spacing,cfg,lower,upper)
        cc2, recs2 = _components(depth,radius+cfg.stability_step_mm,r,b,hu,local_affine,spacing,cfg,lower,upper)
        stable = _stable(cc,recs,cc2,recs2,cfg)
        candidate, info = _ordered_chain(stable,cfg)
        progress(f'  radius {radius:g} mm: {len(recs)} cores, {len(stable)} stable; '
                 + ('ordered chain found' if candidate is not None else info['reason']))
        report['scale_search'].append(dict(radius_mm=radius, candidates=len(recs), **info))
        del cc, cc2
        if candidate is not None:
            chain = candidate; report['selected_radius_mm'] = radius
            break
    if chain is None:
        return unchanged('no_stable_unambiguous_core_chain')
    centers = np.array([c['center'] for c in chain])
    label_ids = list(range(cfg.first-1,cfg.last+2))
    markers = np.zeros(r.shape,dtype=np.uint8); markers2 = markers.copy()
    mismatch = 0; total = 0; core_rows = []
    for label,c in zip(label_ids,chain):
        markers.ravel()[c['idx']] = label
        markers2.ravel()[c['child']['idx']] = label
        purity = float(c['evidence'][label]/c['size'])
        if cfg.first <= label <= cfg.last:
            mismatch += c['size']-int(c['evidence'][label]); total += c['size']
        core_rows.append(dict(label=label, center_world_mm=c['center'].tolist(),
                              volume_mm3=c['volume_mm3'], baseline_purity=purity,
                              dominant_baseline_label=int(c['evidence'].argmax()),
                              bone_fraction=c['bone_fraction'], stability_mm=c['displacement_mm']))
    report['cores'] = core_rows
    report['baseline_core_mismatch_fraction'] = mismatch/max(total,1)
    debug.update(roi=roi, markers=markers, centers=centers, affine=local_affine)
    # Do not force new boundaries on an already consistent scan. This gate is
    # limited to BODY-core evidence; it does not certify posterior accuracy.
    interior_purities = [c['baseline_purity'] for c in core_rows[1:-1]]
    if mismatch/max(total,1)<cfg.core_mismatch_trigger and min(interior_purities)>=.9:
        return unchanged('body_core_chain_consistent_posterior_not_certified')
    progress('  independently seeded geodesic partition at two erosion scales')
    owner, confidence = geodesic_partition(path_domain,depth,markers,spacing,label_ids,progress)
    owner2, confidence2 = geodesic_partition(path_domain,depth,markers2,spacing,label_ids,progress)
    mutable = (b>=cfg.first)&(b<=cfg.last)
    recoverable = (b==0)&(r>=cfg.first)&(r<=cfg.last)
    consensus = (owner==owner2)&(owner>=cfg.first)&(owner<=cfg.last)
    support = path_domain & consensus & (confidence>=cfg.confidence_margin) & (confidence2>=cfg.confidence_margin)
    eligible = (mutable|recoverable)&support
    proposal = b.copy(); proposal[eligible] = owner[eligible]
    report.update(reachable_voxels=int(np.count_nonzero(owner)),
                  scale_disagreement_voxels=int(np.count_nonzero((owner!=owner2)&(mutable|recoverable))),
                  anchor_winner_held_voxels=int(np.count_nonzero((mutable|recoverable)&np.isin(owner,[cfg.first-1,cfg.last+1]))),
                  recovered_voxels=int(np.count_nonzero(recoverable & (proposal>0))),
                  unreachable_target_voxels=int(np.count_nonzero((mutable|recoverable)&(owner==0))))
    debug.update(owner=owner, confidence=confidence, scale_disagreement=(owner!=owner2))
    # Exact binary-mask preservation prohibits BOTH losses and gains for anchors
    # and all other protected labels, unlike the friend's looser writeback rule.
    protected = [i for i in range(1,25) if not cfg.first <= i <= cfg.last]
    for label in protected:
        if not np.array_equal(proposal==label, b==label):
            raise AssertionError(f'Protected label {label} changed')
    if np.any((b>0)&(proposal==0)):
        raise AssertionError('Stage 2 deleted foreground')
    final_purities = []
    for label,c in zip(label_ids[1:-1],chain[1:-1]):
        final_purities.append(float(np.mean(proposal.ravel()[c['idx']]==label)))
    report['final_core_purities'] = final_purities
    # Structural gate, NOT a learned classifier or accuracy certificate.
    if min(final_purities)<.90 or np.mean(final_purities)<np.mean(interior_purities):
        return unchanged('proposal_failed_core_support_check')
    final = baseline.copy(); final[roi] = proposal
    report.update(status='applied', reason='stable_core_mismatch_corrected_with_geodesic_consensus',
                  changed_voxels=int(np.count_nonzero(proposal!=b)))
    report['per_label_volume_mm3'] = {
        str(i): {'baseline':float(np.count_nonzero(b==i)*np.prod(spacing)),
                 'final':float(np.count_nonzero(proposal==i)*np.prod(spacing))}
        for i in range(cfg.first,cfg.last+1)}
    report['protected_labels_exact'] = True
    return final, report, debug


# === Input/output, validation and command-line entry point ===

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):
            h.update(data)
    return h.hexdigest()


def load_labels(path):
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim!=3 or not np.all(np.isfinite(data)) or not np.all(data==np.rint(data)):
        raise ValueError(f'Finite integer 3D label map required: {path}')
    if data.min()<0 or data.max()>24:
        raise ValueError(f'Labels outside 0..24: {path}')
    return image, data.astype(np.uint8,copy=False)


def run_v2(raw, spacing):
    """Same function, order and default settings as recovered v2 attachment."""
    baseline = np.zeros(raw.shape,dtype=np.uint8)
    for label in range(1,25):
        roi,cleaned,_,_ = clean_label(raw==label,spacing,16,.01,12.)
        if roi is not None:
            baseline[roi][cleaned] = label
        if label%4==0:
            print(f'  v2 baseline {label}/24 labels',flush=True)
    return baseline


def save_volume(data, image, path):
    hdr=image.header.copy(); hdr.set_data_dtype(data.dtype)
    out=nib.Nifti1Image(data,image.affine,hdr)
    qform,qcode=image.get_qform(coded=True); sform,scode=image.get_sform(coded=True)
    out.set_qform(qform,int(qcode)); out.set_sform(sform,int(scode))
    nib.save(out,str(path))


def execute_case(case, args):
    started=time.monotonic()
    source=case/'combined_labels.nii.gz'
    print(f'[{case.name}] reading predictions and reproducing original v2',flush=True)
    im,raw=load_labels(source)
    baseline=run_v2(raw,np.asarray(im.header.get_zooms()[:3],float))
    report_v2={'parameters':{'min_component_voxels':16,'min_relative_size':.01,
                             'max_satellite_distance_mm':12.},'reference_equal':None}
    if args.v2_reference:
        refpath=args.v2_reference/case.name/'combined_labels.nii.gz'
        refim,reference=load_labels(refpath)
        samegeom=reference.shape==raw.shape and np.allclose(refim.affine,im.affine,rtol=0,atol=1e-5)
        same=samegeom and np.array_equal(baseline,reference)
        report_v2.update(reference_equal=bool(same),reference_path=str(refpath),reference_sha256=sha256(refpath))
        del reference
        if not same:
            raise ValueError(f'Replayed v2 does not exactly match reference: {case.name}')
        print('  v2 replay matches stored v2 voxel-for-voxel',flush=True)
    ctpath=args.ct_root/case.name/'ct.nii.gz'
    ctim=nib.load(str(ctpath))
    if ctim.shape!=im.shape or not np.allclose(ctim.affine,im.affine,rtol=0,atol=1e-4):
        raise ValueError('CT geometry does not match prediction')
    ct=ctim.get_fdata(dtype=np.float32)
    cfg=Config()
    final,report,debug=refine(raw,baseline,ct,im.affine,cfg)
    protected=[i for i in range(1,25) if not cfg.first<=i<=cfg.last]
    protected_checks={str(i):bool(np.array_equal(final==i,baseline==i)) for i in protected}
    if not all(protected_checks.values()):
        raise AssertionError('Protected binary mask changed')
    counts_before=np.bincount(baseline.ravel(),minlength=25)
    counts_after=np.bincount(final.ravel(),minlength=25)
    if np.any((counts_before[1:]>0)&(counts_after[1:]==0)):
        raise AssertionError('A previously present label disappeared')
    outcase=args.output/case.name
    outcase.mkdir(parents=True,exist_ok=False)
    segdir=outcase/'segmentations';segdir.mkdir()
    print(f'[{case.name}] {report["status"]}: {report["reason"]}; writing exports',flush=True)
    save_volume(final,im,outcase/'combined_labels.nii.gz')
    for label,name in enumerate(VERTEBRA_NAMES,1):
        save_volume((final==label).astype(np.uint8),im,segdir/(name+'.nii.gz'))
        if label%6==0:
            print(f'  exported {label}/24 binary masks',flush=True)
    # Audit artifacts are visibly separate from the deliverable masks.
    qa=outcase/'qa';qa.mkdir()
    if debug:
        cropped_affine=debug['affine']
        for key in ('markers','owner','scale_disagreement'):
            if key in debug:
                nib.save(nib.Nifti1Image(debug[key].astype(np.uint8),cropped_affine),str(qa/(key+'.nii.gz')))
        roi=debug['roi']
        changes=(final[roi]!=baseline[roi]).astype(np.uint8)
        nib.save(nib.Nifti1Image(changes,cropped_affine),str(qa/'changes_from_v2.nii.gz'))
    report.update(v2_baseline=report_v2,case=case.name,
                  protected_binary_masks_exact=protected_checks,
                  label_counts_v2=counts_before.tolist(),label_counts_final=counts_after.tolist(),
                  input_sha256=sha256(source),output_sha256=sha256(outcase/'combined_labels.nii.gz'),
                  runtime_seconds=round(time.monotonic()-started,2))
    (outcase/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--ct-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--v2-reference',type=Path,help='Optional stored v2 directory, checked voxel-for-voxel')
    p.add_argument('--case',action='append',help='Restrict to named BDMAP case(s)')
    args=p.parse_args()
    args.input=args.input.resolve();args.ct_root=args.ct_root.resolve();args.output=args.output.resolve()
    if args.v2_reference: args.v2_reference=args.v2_reference.resolve()
    for source in (args.input,args.ct_root,args.v2_reference):
        if source and (source==args.output or source in args.output.parents or args.output in source.parents):
            raise ValueError('Output must be disjoint from source directories')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing output; choose a new output path')
    cases=sorted(d for d in args.input.glob('BDMAP_*') if d.is_dir())
    if args.case:
        cases=[d for d in cases if d.name in args.case]
        if {d.name for d in cases}!=set(args.case): raise ValueError('Requested case not found')
    if not cases:raise FileNotFoundError('No BDMAP cases found')
    from skimage.graph import MCP_Geometric  # dependency check before expensive work
    import skimage
    args.output.mkdir(parents=True)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='incomplete',
                  sources={str(Path(__file__).resolve()):sha256(Path(__file__))},
                  versions={'python':platform.python_version(),'numpy':np.__version__,
                            'scipy':scipy.__version__,'nibabel':nib.__version__,'scikit_image':skimage.__version__},
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  dsc=None,cases=[])
    manifestpath=args.output/'run_manifest.json'
    manifestpath.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for case in cases:
        manifest['cases'].append(execute_case(case,args))
        manifestpath.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['status']='complete'
    manifestpath.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed; DSC remains unknown without reference masks.',flush=True)


if __name__=='__main__':main()

