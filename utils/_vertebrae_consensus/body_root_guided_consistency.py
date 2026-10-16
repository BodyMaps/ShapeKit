"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
from dataclasses import asdict

import nibabel as nib

import numpy as np

from .body_root_guided import REGIONS, centers_and_seeds, refine_region

from .postprocessing_causal import NAMES

def chain_audit(labels, affine, region):
    positions = np.where((labels >= region.first_anchor) & (labels <= region.last_anchor))
    if not len(positions[0]):
        raise ValueError('empty_anchor_band')
    spacing = nib.affines.voxel_sizes(affine)
    padding = np.ceil(5/spacing).astype(int)
    roi = tuple(slice(max(0, int(p.min())-int(padding[i])),
                      min(labels.shape[i], int(p.max())+int(padding[i])+1))
                for i, p in enumerate(positions))
    cropped_affine = affine.copy()
    cropped_affine[:3, 3] = nib.affines.apply_affine(affine, [s.start for s in roi])
    centers, _, _, _, _ = centers_and_seeds(labels[roi], cropped_affine, region)
    return {'body_centers_RAS_mm': centers.tolist(), 'minimum_world_z_gap_mm': float(np.diff(centers[:, 2]).min())}

def refine(base, ct, affine, anchors, regions=tuple(REGIONS)):
    if base.ndim != 3 or base.shape != anchors.shape or base.shape != ct.shape or not np.isfinite(ct).all():
        raise ValueError('Matching finite 3D CT and prediction grids required')
    for labels in (base, anchors):
        if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 24:
            raise ValueError('Integer prediction labels0..24 required')
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3]/spacing
    if not np.allclose(direction.T@direction, np.eye(3), atol=1e-5):
        raise ValueError('Sheared physical grids unsupported')
    checks = []
    for region in regions:
        for role, labels in [('neural_guide', base), ('independent_SuPreM_anchors', anchors)]:
            row = {'region': asdict(region), 'role': role}
            try:
                row.update(chain_audit(labels, affine, region), status='valid')
            except ValueError as error:
                row.update(status='invalid', reason=str(error))
            checks.append(row)
    report = {'method': 'Dual ordered body-chain gate followed by original guided 3D identity graph',
              'reference_labels_used': False, 'global_prerequisite': 'all configured bands ordered in both predictions',
              'chain_checks': checks, 'foreground_exact': True, 'DSC': None,
              'proposed_after_development_DSC_inspection': True,
              'new_model_used': True, 'pure_original_SuPreM_postprocessing': False,
              'credit': 'TotalSegmentator foreground plus unchanged Hao; original SuPreM anchors plus Hao and fixed roots; own graph and consistency gate.'}
    if not checks or any(row['status'] != 'valid' for row in checks):
        return base.copy(), dict(report, status='abstained', reason='dual_body_chain_prerequisite_failed',
                                 changed_voxels=0, regions=[])
    actual = base.copy()
    reports = []
    for region in regions:
        try:
            actual, row = refine_region(actual, ct, affine, region, anchors)
        except ValueError as error:
            row = {'region': asdict(region), 'status': 'abstained', 'reason': str(error), 'changed_voxels': 0}
        reports.append(row)
    assert np.array_equal(actual > 0, base > 0)
    assert np.array_equal(actual == 24, base == 24)
    assert np.all(actual[base == 23] == 23)
    return actual, dict(report, status='applied' if np.any(actual != base) else 'abstained',
                        changed_voxels=int(np.count_nonzero(actual != base)), regions=reports)
