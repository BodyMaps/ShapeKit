"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
import nibabel as nib

import numpy as np

from scipy import ndimage as ndi

from .postprocessing_causal import NAMES

def source_cores(source, affine, labels):
    spacing = nib.affines.voxel_sizes(affine)
    rows = []; centers = []
    for label in labels:
        positions = np.where(source == label)
        if not len(positions[0]): raise ValueError(f'missing_source_body_{label}')
        radius, minimum = (5., 300.) if label <= 14 else (3., 60.)
        padding = np.ceil((radius+2)/spacing).astype(int)
        starts = np.array([max(0, int(p.min())-int(q)) for p, q in zip(positions, padding)])
        roi = tuple(slice(int(starts[i]), min(source.shape[i], int(p.max())+int(padding[i])+1))
                    for i, p in enumerate(positions))
        mask = source[roi] == label
        depth = ndi.distance_transform_edt(mask, sampling=spacing)
        tiers = []
        for offset in (-1., 0., 1.):
            cc, count = ndi.label(depth >= radius+offset, np.ones((3, 3, 3)))
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            required = minimum if offset == 0 else minimum/3
            if not count or sizes.max()*np.prod(spacing) < required:
                raise ValueError(f'insufficient_scale_stable_body_{label}_radius_{radius+offset}')
            coordinates = np.array(np.where(cc == sizes.argmax())).T+starts
            tiers.append(coordinates)
            if offset == 0: centers.append(nib.affines.apply_affine(affine, coordinates.mean(0)))
        rows.append({'label': int(label), 'radius_mm': radius, 'minimum_core_mm3': minimum, 'tiers': tiers})
    centers = np.array(centers)
    gaps = np.diff(centers[:, 2])
    if np.any(gaps < 5) or np.any(gaps > 60):
        raise ValueError('source_body_chain_not_anatomically_ordered')
    return rows, centers
