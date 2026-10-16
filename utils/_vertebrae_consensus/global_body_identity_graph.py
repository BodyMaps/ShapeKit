"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
import nibabel as nib

import numpy as np

from scipy import ndimage as ndi

from skimage.graph import MCP_Geometric

from .component_identity_transfer import source_cores

from .postprocessing_causal import NAMES

def refine(base, anchors, ct, affine, labels=tuple(range(1, 23))):
    if base.ndim != 3 or base.shape != anchors.shape or base.shape != ct.shape or not np.isfinite(ct).all():
        raise ValueError('Matching finite3D inputs required')
    for image in (base, anchors):
        if not np.issubdtype(image.dtype, np.integer) or image.min() < 0 or image.max() > 24:
            raise ValueError('Integer labels0..24 required')
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3]/spacing
    if not np.allclose(direction.T@direction, np.eye(3), atol=1e-5):
        raise ValueError('Sheared grids unsupported')
    report = {'method': 'Scale-consistent independent body-seeded global physical geodesics',
        'reference_labels_used': False, 'foreground_exact': True, 'original_C1_C2_masks_exact': True,
        'source_scale_offsets_mm': [-1., 0., 1.], 'source_chain_gap_range_mm': [5., 60.],
        'minimum_CT_seed_HU': 100., 'minimum_CT_foreground_contrast_HU': 80.,
        'physical_cost_powers': [1., 1.5, 2.], 'each_scale_runner_up_margin': .10,
        'minimum_target_CT_HU': None, 'marrow_identity_can_change': True,
        'whole_body_identity_changes_can_exceed_adjacent_or_original_volume_gates': True,
        'proposed_after_development_error_review': True, 'DSC': None,
        'credit': 'Neural foreground: TotalSegmentator; independent anchors: SuPreM plus credited Hao and original roots; global graph: original experimental code'}
    domain = (base > 0) & (base < 23)
    pos = np.where(domain)
    if not len(pos[0]):
        return base.copy(), dict(report, status='abstained', reason='empty_domain', changed_voxels=0)
    pad = np.ceil(5/spacing).astype(int)
    starts = np.array([max(0, int(q.min())-int(v)) for q,v in zip(pos,pad)])
    roi = tuple(slice(int(starts[i]), min(base.shape[i], int(q.max())+int(pad[i])+1)) for i,q in enumerate(pos))
    d = domain[roi]; hu = ct[roi]; del domain, pos
    outside = ndi.distance_transform_edt(~d, sampling=spacing)
    shell = (outside > 0) & (outside <= 2.5)
    contrast = float(np.median(hu[d])-np.median(hu[shell])) if shell.any() else 0.
    report['median_CT_foreground_shell_contrast_HU'] = contrast
    del outside,shell
    if contrast < 80:
        return base.copy(), dict(report, status='abstained', reason='insufficient_CT_contrast', changed_voxels=0)
    try:
        cores, centers = source_cores(anchors, affine, labels)
    except ValueError as error:
        return base.copy(), dict(report, status='abstained', reason=str(error), changed_voxels=0)
    origin = centers.mean(0); _,_,vh = np.linalg.svd(centers-origin,full_matrices=False)
    tangent = vh[0].copy()
    if tangent[2] < 0: tangent = -tangent
    anterior = np.array([0.,1.,0.])-tangent[1]*tangent
    if np.linalg.norm(anterior) < .2:
        return base.copy(), dict(report,status='abstained',reason='ambiguous_anterior_direction',changed_voxels=0)
    anterior /= np.linalg.norm(anterior)
    if np.any(np.diff((centers-origin)@tangent) < 5):
        return base.copy(), dict(report,status='abstained',reason='intrinsic_core_order_invalid',changed_voxels=0)
    marker_tiers = [dict() for _ in range(3)]
    for index, body in enumerate(cores):
        for tier, coordinates in enumerate(body['tiers']):
            world = nib.affines.apply_affine(affine, coordinates)
            local = coordinates-starts
            inside = np.all((local>=0)&(local<np.array(d.shape)), axis=1)
            front = (world-centers[index])@anterior >= -body['radius_mm']
            local = local[inside & front]
            if len(local): local = local[d[tuple(local.T)] & (hu[tuple(local.T)]>=100)]
            if len(local)*np.prod(spacing) < body['minimum_core_mm3']/3:
                return base.copy(),dict(report,status='abstained',reason=f'insufficient_trusted_seed_{body["label"]}_tier_{tier}',changed_voxels=0)
            marker_tiers[tier][body['label']] = [tuple(int(v) for v in q) for q in local]
    depth = ndi.distance_transform_edt(d,sampling=spacing).astype(np.float32)
    penalty = 1+np.maximum(0,150-hu)/150
    votes=[]; confidence = d.copy()
    for tier,power in enumerate((1.,1.5,2.)):
        print('global geodesic scale',tier,'power',power,'ROI',d.shape,flush=True)
        costs = np.full(d.shape,np.inf,np.float32)
        costs[d] = penalty[d]/(depth[d]+.5)**power
        best = np.full(d.shape,np.inf,np.float64); second = best.copy(); owner=np.zeros(d.shape,np.uint8)
        for label,seeds in marker_tiers[tier].items():
            solver=MCP_Geometric(costs,sampling=tuple(spacing))
            distances,_=solver.find_costs(seeds)
            better=distances<best
            second[better]=best[better]; best[better]=distances[better]; owner[better]=label
            alternate=(~better)&(distances<second); second[alternate]=distances[alternate]
            del solver,distances,better,alternate
        unique=np.isfinite(best)&~np.isfinite(second)
        competing=np.isfinite(best)&np.isfinite(second)
        strong=unique.copy()
        strong[competing]=(second[competing]-best[competing])/np.maximum(second[competing],1e-6)>=.10
        confidence &= strong; votes.append(owner)
        del best,second,owner,costs,unique,competing,strong
    consensus=(votes[0]==votes[1])&(votes[0]==votes[2])
    certified=d & consensus & confidence & (votes[0]>0)
    out=base.copy(); out[roi][certified]=votes[0][certified]
    assert np.array_equal(out>0,base>0)
    assert np.array_equal(out==23,base==23) and np.array_equal(out==24,base==24)
    changed=int(np.count_nonzero(out!=base))
    return out,dict(report,status='applied' if changed else 'abstained', changed_voxels=changed,
        source_body_centers_RAS_mm=centers.tolist(),native_ROI_shape=list(d.shape),
        seed_counts_by_tier=[{str(k):len(v) for k,v in tier.items()} for tier in marker_tiers],
        certified_foreground_voxels=int(certified.sum()))
