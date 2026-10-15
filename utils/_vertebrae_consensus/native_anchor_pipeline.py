"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
import nibabel as nib

import numpy as np

from . import postprocessing_causal as caudal

from . import cervicothoracic_causal as cervical

def crop_band(base, affine, lo, hi):
    positions = np.where((base >= lo) & (base <= hi))
    if not len(positions[0]):
        raise ValueError('empty_anatomical_band')
    roi = tuple(slice(max(0, int(v.min())-2), min(base.shape[i], int(v.max())+3))
                for i, v in enumerate(positions))
    local = affine.copy()
    local[:3, 3] = nib.affines.apply_affine(affine, [s.start for s in roi])
    return roi, local

def refine(raw, base, ct, affine):
    if raw.shape != base.shape or ct.shape != base.shape or base.ndim != 3 or not np.isfinite(ct).all():
        raise ValueError('Matching finite 3D inputs required')
    for labels in (raw, base):
        if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 24:
            raise ValueError('Integer labels0..24 required')
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3]/spacing
    if not (np.allclose(np.abs(direction), np.eye(3), atol=1e-5) and direction[1, 1] > 0 and direction[2, 2] > 0):
        raise ValueError('Native LAS/RAS required; no scanner-axis adapter applied')
    out = base.copy()
    target = (base >= 4) & (base <= 13)
    mismatch = np.count_nonzero(target & (base != raw))/max(np.count_nonzero(target), 1)
    stages = []
    for name, module, band in [('caudal', caudal, (3, 14)), ('cervical', cervical, (16, 23))]:
        try:
            if name == 'caudal' and mismatch < .02:
                raise ValueError('no_body_repair_trigger')
            roi, local = crop_band(out, affine, *band)
            centers = module.body_anchors(out[roi], local)
            result, report = module.refine(out[roi], ct[roi], local, centers)
            out[roi] = result
            report.update(status='applied' if report['changed_voxels'] else 'abstained',
                          body_centers_world_mm=centers.tolist())
        except ValueError as error:
            report = {'status': 'abstained', 'reason': str(error), 'changed_voxels': 0}
        stages.append({'stage': name, **report})
    assert np.array_equal(out > 0, base > 0)
    assert np.array_equal(out == 24, base == 24)
    assert np.all(out[base == 23] == 23)
    return out, {'stages': stages, 'changed_voxels': int(np.count_nonzero(out != base)),
                 'raw_baseline_disagreement_fraction': float(mismatch),
                 'reference_labels_used': False, 'foreground_exact': True,
                 'coordinate_adaptation': 'none; original native LAS/RAS indexing',
                 'CT_boundary_stage_used': False, 'algorithm_parameters_changed': False}
