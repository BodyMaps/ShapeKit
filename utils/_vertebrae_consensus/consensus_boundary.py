"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
import nibabel as nib

import numpy as np

from scipy import ndimage as ndi

from .postprocessing_causal import NAMES

def refine(base, source, auxiliary, ct, affine, ct_guard=True):
    if base.ndim != 3 or any(x.shape != base.shape for x in (source, auxiliary, ct)):
        raise ValueError('Matching3D arrays required')
    for x in (base, source, auxiliary):
        if not np.issubdtype(x.dtype, np.integer) or x.min() < 0 or x.max() > 24:
            raise ValueError('Integer labels0..24 required')
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3, :3]/spacing
    if not np.allclose(direction.T@direction, np.eye(3), atol=1e-5):
        raise ValueError('Sheared grids unsupported')
    if not np.isfinite(ct).all():
        raise ValueError('Finite CT intensities required')
    report = {'method': 'Two-source boundary agreement with physical band and CT guard',
              'reference_labels_used': False, 'band_mm': 2.0, 'bone_threshold_HU': 100,
              'ct_guard': bool(ct_guard), 'incumbent_foreground_id_changes': 0,
              'C1_C2_masks_exact': True, 'DSC': None}
    out = base.copy()
    foreground = base > 0
    axes = [np.flatnonzero(np.any(foreground, axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if not axes[0].size:
        return out, dict(report, status='abstained', reason='incumbent_has_no_foreground', changed_voxels=0)
    # Crop by incumbent extent plus3mm; artificial crop boundaries are outside
    # the2mm acceptance band. One extra zero layer models the scan exterior.
    pad = np.ceil(3/spacing).astype(int)+1
    starts = np.maximum(0, [x[0] for x in axes]-pad)
    stops = np.minimum(base.shape, np.array([x[-1]+1 for x in axes])+pad)
    crop = tuple(slice(int(i), int(j)) for i, j in zip(starts, stops))
    b, s, v, h = [x[crop] for x in (base, source, auxiliary, ct)]
    fg = b > 0
    distances, nearest = ndi.distance_transform_edt(~fg, sampling=spacing, return_indices=True)
    nearest_label = b[tuple(nearest)]
    add = (b == 0) & (s == v) & (s > 0) & (s <= 22) & (s == nearest_label) & (distances <= 2.0+1e-7)
    del distances, nearest, nearest_label
    inside = ndi.distance_transform_edt(np.pad(fg, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    remove = fg & (b <= 22) & (s == 0) & (v == 0) & (inside <= 2.0+1e-7)
    del inside
    agreement_add, agreement_remove = int(add.sum()), int(remove.sum())
    if ct_guard:
        add &= h >= 100
        remove &= h < 100
    # Retain any class for which the accepted deletions would erase its mask.
    erased = []
    for label in range(1, 23):
        if np.any(b == label) and not np.any((b == label) & ~remove):
            remove[b == label] = False
            erased.append(label)
    target = out[crop]
    target[add] = s[add]
    target[remove] = 0
    assert not np.any((base > 0) & (out > 0) & (base != out))
    for label in (23, 24):
        assert np.array_equal(base == label, out == label)
    rows = [{'id': label, 'name': name, 'added_voxels': int(np.count_nonzero(add & (s == label))),
             'removed_voxels': int(np.count_nonzero(remove & (b == label)))} for label, name in enumerate(NAMES, 1)]
    changed = int(add.sum()+remove.sum())
    return out, dict(report, status='applied' if changed else 'abstained', changed_voxels=changed,
                     foreground_added_voxels=int(add.sum()), foreground_removed_voxels=int(remove.sum()),
                     agreement_band_add_candidates=agreement_add, agreement_band_remove_candidates=agreement_remove,
                     empty_class_deletion_abstentions=erased, per_class_edits=rows,
                     native_crop_start=starts.tolist(), native_crop_end_exclusive=stops.tolist())
