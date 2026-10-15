"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
import nibabel as nib

import numpy as np

from .component_identity_transfer import source_cores

from .postprocessing_causal import NAMES

def filter_proposal(base,anchors,proposal,affine,labels=tuple(range(1,23))):
    if base.ndim!=3 or base.shape!=anchors.shape or base.shape!=proposal.shape:
        raise ValueError('Matching3D arrays required')
    for image in (base,anchors,proposal):
        if not np.issubdtype(image.dtype,np.integer) or image.min()<0 or image.max()>24:
            raise ValueError('Integer labels0..24 required')
    if not np.array_equal(proposal>0,base>0):raise ValueError('Graph must preserve foreground')
    if not np.array_equal(proposal==23,base==23) or not np.array_equal(proposal==24,base==24):
        raise ValueError('Graph must preserve originalC1/C2 masks')
    spacing=nib.affines.voxel_sizes(affine)
    direction=affine[:3,:3]/spacing
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5):raise ValueError('Sheared grids unsupported')
    report={'method':'Coherent body-numbering offset gates independent global graph',
        'reference_labels_used':False,'foreground_exact':True,'original_C1_C2_masks_exact':True,
        'required_adjacent_source_body_pairs':1,'same_nonzero_model_minus_source_offset':True,
        'required_erosion_scales':3,'dominant_ID_requires_strict_majority':True,
        'isolated_identity_conflicts_do_not_trigger':True,
        'selected_after_first_case_development_regression':True,'DSC':None}
    try:cores,centers=source_cores(anchors,affine,labels)
    except ValueError as error:
        return base.copy(),dict(report,status='abstained',reason=str(error),changed_voxels=0)
    body_rows=[]; offsets={}
    for body in cores:
        tiers=[]
        for coords in body['tiers']:
            values=base[tuple(coords.T)];counts=np.bincount(values,minlength=25)
            foreground=int(counts[1:23].sum())
            dominant=int(counts[1:23].argmax()+1) if foreground else 0
            supported=foreground*np.prod(spacing)>=body['minimum_core_mm3']/3
            majority=supported and counts[dominant]>foreground/2
            tiers.append({'dominant_model_ID':dominant,'strict_majority':bool(majority),
                'foreground_core_voxels':foreground,'dominant_voxels':int(counts[dominant])})
        ids=[t['dominant_model_ID'] for t in tiers]
        stable=all(t['strict_majority'] for t in tiers) and len(set(ids))==1
        offset=ids[0]-body['label'] if stable else None
        offsets[body['label']]=offset
        body_rows.append({'source_ID':body['label'],'scale_stable_dominant_offset':offset,'tiers':tiers})
    pairs=[]
    for first,second in zip(labels,labels[1:]):
        if second==first+1 and offsets[first] not in (None,0) and offsets[first]==offsets[second]:
            pairs.append({'source_bodies':[first,second],'model_minus_source_ID_offset':offsets[first]})
    report.update(body_core_model_identity=body_rows,coherent_offset_pairs=pairs,
                  source_body_centers_RAS_mm=centers.tolist())
    if not pairs:
        return base.copy(),dict(report,status='abstained',reason='no_coherent_multilevel_numbering_offset',changed_voxels=0)
    changed=int(np.count_nonzero(base!=proposal))
    return proposal.copy(),dict(report,status='applied' if changed else 'abstained',changed_voxels=changed,
        gate_passed=True,scope='All frozen global graph proposals accepted when coherent body-chain error detected')
