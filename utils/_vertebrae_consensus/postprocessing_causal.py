"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
from __future__ import annotations

import nibabel as nib

import numpy as np

from scipy import ndimage as ndi

from skimage.segmentation import watershed

NAMES = ['L5', 'L4', 'L3', 'L2', 'L1'] + [f'T{i}' for i in range(12, 0, -1)] + [f'C{i}' for i in range(7, 0, -1)]

def body_anchors(base, affine):
    """Derive body seeds directly; no experimental-report dependency."""
    spacing = nib.affines.voxel_sizes(affine)
    centers = []
    for label in range(3, 15):
        depth = ndi.distance_transform_edt(base == label, sampling=spacing)
        components, count = ndi.label(depth >= 5.0, np.ones((3, 3, 3)))
        sizes = np.bincount(components.ravel()); sizes[0] = 0
        if count == 0 or sizes.max() * np.prod(spacing) < 300:
            raise ValueError(f'Insufficient anatomical body seed {label}')
        center = ndi.center_of_mass(components == sizes.argmax())
        centers.append(nib.affines.apply_affine(affine, center))
    centers = np.array(centers)
    if np.any(np.diff(centers[:, 2]) < 10) or np.any(np.diff(centers[:, 2]) > 55):
        raise ValueError('Body anchors are not anatomically ordered')
    return centers

def sweep_one(base, hu, affine, centers, width_mm, root_depth_mm, min_hu=100):
    spacing = nib.affines.voxel_sizes(affine)
    # This experiment handles native axis-aligned superior-positive scans;
    # other orientations abstain rather than silently reversing causal flow.
    direction = affine[:3, :3] / spacing
    if not (np.allclose(np.abs(direction), np.eye(3), atol=1e-5) and direction[2, 2] > 0 and direction[1, 1] > 0):
        raise ValueError('Causal experiment requires native axial LAS/RAS, superior-positive')
    x = affine[0, 3] + affine[0, 0] * np.arange(base.shape[0])[:, None]
    y = affine[1, 3] + affine[1, 1] * np.arange(base.shape[1])[None, :]
    z_world = affine[2, 3] + affine[2, 2] * np.arange(base.shape[2])
    center_z, center_x, center_y = centers[:, 2], centers[:, 0], centers[:, 1]
    labels = np.arange(3, 15)
    boundary_z = (center_z[1:] + center_z[:-1]) / 2
    previous = np.zeros(base.shape[:2], np.uint8)
    out = base.copy()
    for k in range(base.shape[2]-1, -1, -1):
        z = z_world[k]
        cx, cy = np.interp(z, center_z, center_x), np.interp(z, center_z, center_y)
        corridor = (np.abs(x-cx) <= width_mm) & (y < cy-22) & (y >= cy-80)
        foreground = (base[:, :, k] >= 3) & (base[:, :, k] <= 14)
        domain = corridor & foreground & (hu[:, :, k] >= min_hu)
        expected = int(labels[np.clip(np.searchsorted(boundary_z, z), 0, len(labels)-1)])
        # Roots enter through the anterior corridor edge only at their body
        # level; long posterior tips cannot create an arbitrary new identity.
        expected_bone = (base[:, :, k] == expected) & (hu[:, :, k] >= min_hu)
        components, _ = ndi.label(expected_bone, np.ones((3, 3)))
        body_cross_section = (np.abs(x-cx) <= 18) & (y >= cy-15) & (y <= cy+20)
        body_components = np.unique(components[body_cross_section & expected_bone])
        body_components = body_components[body_components > 0]
        body_connected = np.isin(components, body_components) if body_components.size else np.zeros(domain.shape, bool)
        root_zone = domain & (y >= cy-root_depth_mm) & body_connected
        markers = np.where(domain, previous, 0).astype(np.uint8)
        markers[root_zone] = expected
        # The top boundary is an unchanged anatomical anchor, never an edit.
        if expected == 14:
            markers[domain & (base[:, :, k] == 14)] = 14
        if markers.any():
            depth = ndi.distance_transform_edt(domain, sampling=spacing[:2])
            owner = watershed(-depth, markers=markers, mask=domain, connectivity=np.ones((3, 3))).astype(np.uint8)
        else:
            owner = np.zeros(domain.shape, np.uint8)
        # Unreachable pieces retain baseline and do not seed subsequent slices.
        editable = (base[:, :, k] >= 4) & (base[:, :, k] <= 13)
        proposed = editable & domain & (owner >= 4) & (owner <= 13)
        out[:, :, k][proposed] = owner[proposed]
        previous = owner
    return out

def refine(base, hu, affine, centers, max_fraction=.10):
    # Three independently specified corridors. Consensus belongs to the
    # smallest common midline region; disagreement abstains.
    settings = [(8., 34.), (10., 38.), (12., 42.)]
    votes = []
    for width, root_depth in settings:
        print('  causal corridor', width, 'mm; root depth', root_depth, 'mm', flush=True)
        votes.append(sweep_one(base, hu, affine, centers, width, root_depth))
    owner = votes[0]
    agrees = (owner == votes[1]) & (owner == votes[2])
    # A caudally drooping process may be captured one level below its true
    # owner, so its corrected bottom-up id can increase by one. Opposite
    # shifts are not justified by this experiment's causal evidence.
    change = agrees & (owner.astype(int) - base.astype(int) == 1)
    out = base.copy()
    out[change] = owner[change]
    capped = []
    for label in range(4, 14):
        affected = (out != base) & ((out == label) | (base == label))
        if np.count_nonzero(affected) / max(np.count_nonzero(base == label), 1) > max_fraction:
            capped.append(label)
    out[np.isin(base, capped) | np.isin(out, capped)] = base[np.isin(base, capped) | np.isin(out, capped)]
    assert np.array_equal(out > 0, base > 0)
    assert all(np.array_equal(out == i, base == i) for i in list(range(1, 4)) + list(range(14, 25)))
    return out, {'method': 'original three-corridor body-connected cranial-root-to-caudal sweep',
                 'settings_width_root_depth_mm': settings, 'max_fraction': max_fraction,
                 'changed_voxels': int(np.count_nonzero(out != base)), 'capped_labels': capped,
                 'dsc': None, 'ground_truth_used': False,
                 'foreground_exact': True, 'protected_labels_exact': True,
                 'baseline_credit': 'Hao Yu / BodyMaps ShapeKit, unmodified engine',
                 'anatomical_inspiration': 'Abhijit Das / ShapeKit-Pro, caudal spinous continuity; independent implementation'}
