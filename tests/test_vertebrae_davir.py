"""Synthetic tests for the ShapeKit-DAVIR vertebrae engine (no data download needed)."""
import logging

import nibabel as nib
import numpy as np

from utils import vertebrae_davir as adapter
from utils import vertebrae_davir_engine as davir

ZOOMS = (1.0, 1.0, 1.5)


def _phantom():
    """24 cylindrical bodies (L5 at the bottom) separated by 5 mm discs, each with an arch behind it."""
    pitch = [davir.GAP_MU.get(k, 17.0) for k in range(1, 25)]
    nz = int(sum(pitch) / ZOOMS[2]) + 20
    lab = np.zeros((70, 80, nz), np.uint8)
    x, y = np.meshgrid(np.arange(70), np.arange(80), indexing="ij")
    body = (x - 35) ** 2 + (y - 30) ** 2 <= 12 ** 2
    arch = (np.abs(x - 35) <= 4) & (y >= 42) & (y <= 60)
    pedicle = (np.abs(x - 35) <= 7) & (y > 39) & (y < 44)
    z = 10.0
    for k in range(1, 25):
        z0, z1 = int(z / ZOOMS[2]), int((z + pitch[k - 1] - 5.0) / ZOOMS[2])
        lab[:, :, z0:z1][body | pedicle | arch] = k
        z += pitch[k - 1]
    affine = np.diag(list(ZOOMS) + [1.0])
    return lab, affine


def _shift(lab):
    """The error seen on BDMAP_00000031: L1..T9 named one level too low."""
    out = lab.copy()
    for k in range(5, 10):
        out[lab == k] = k + 1
    return out


def _mean_dice(a, b):
    ds = [2 * ((a == k) & (b == k)).sum() / ((a == k).sum() + (b == k).sum()) for k in np.unique(b) if k]
    return float(np.mean(ds))


def test_clean_spine_is_unchanged():
    lab, affine = _phantom()
    out = davir.refine(lab, affine)
    assert _mean_dice(out, lab) > 0.999


def test_shifted_names_are_corrected():
    lab, affine = _phantom()
    log = {}
    out = davir.refine(_shift(lab), affine, log=log)
    assert log["mode"] == "full" and log["n_instances"] == 24
    assert _mean_dice(_shift(lab), lab) < 0.85
    assert _mean_dice(out, lab) > 0.99


def test_shapekit_adapter_matches_engine():
    lab, affine = _phantom()
    shifted = _shift(lab)
    seg = {name: (shifted == k).astype(np.uint8) for k, name in enumerate(adapter.VERTEBRA_NAMES, 1)}
    ref = nib.Nifti1Image(np.zeros(lab.shape, np.uint8), affine)
    seg = adapter.postprocessing_vertebrae_davir("phantom", seg, ref, None, logging.getLogger(__name__))
    out = np.zeros(lab.shape, np.uint8)
    for k, name in enumerate(adapter.VERTEBRA_NAMES, 1):
        out[seg[name] > 0] = k
    assert np.array_equal(out, davir.refine(shifted, affine))
