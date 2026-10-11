"""Small deterministic tests for the ShapeKit-Asfand vertebrae engine and adapter."""

import logging

import nibabel as nib
import numpy as np

from utils import vertebrae_asfand as adapter
from utils import vertebrae_asfand_engine as engine

LOGGER = logging.getLogger(__name__)
SPACING = (1.0, 1.0, 2.0)
N_BODIES = 12            # bottom-up: L5, L4, L3, L2, L1, T12, T11, T10, T9, T8, T7, T6
BODY, DISC = 8, 3        # slices per body / per unlabelled disc


def _body_mask(shape):
    m = np.zeros(shape, bool)
    m[8:42, 30:66, :] = True
    return m


def _spine(drift=False):
    """Synthetic spine (superior axis last, superior = higher index).

    Bodies are separated by unlabelled discs; posterior elements bridge each
    disc, so the whole spine is one connected component, as in real masks.
    With drift=True the model 'skips' T11: bodies L1, T12, T11 carry the label
    one level too high (body only; their posterior elements keep the true label).
    """
    z_total = N_BODIES * (BODY + DISC) + 6
    truth = np.zeros((50, 80, z_total), np.uint8)
    model = np.zeros_like(truth)
    for i in range(N_BODIES):
        level = i + 1                                   # engine id: L5 = 1
        z0 = 3 + i * (BODY + DISC)
        z1 = z0 + BODY
        body = (slice(8, 42), slice(30, 66), slice(z0, z1))  # ~35 mm body, like thoracolumbar
        post = (slice(22, 28), slice(8, 30), slice(max(z0 - DISC, 0), z1))
        for arr in (truth, model):
            arr[post] = level
            arr[body] = level
        if drift and i in (4, 5, 6):                    # L1, T12, T11 bodies shifted up by one
            model[body] = level + 1
    return truth, model


def test_clean_spine_is_unchanged():
    truth, model = _spine(drift=False)
    out, report = engine.refine(model, SPACING, log=lambda m: None)
    assert report["relabelled_bodies"] == 0
    assert np.array_equal(out, truth)


def test_drifted_bodies_are_relabelled():
    truth, model = _spine(drift=True)
    assert (model != truth).sum() > 0
    out, report = engine.refine(model, SPACING, log=lambda m: None)
    assert report["relabelled_bodies"] == 3
    bodies = _body_mask(truth.shape) & (truth > 0)
    assert np.array_equal(out[bodies], truth[bodies])            # every body gets its counted level
    # posterior elements stacked directly over the level below are the known weak
    # spot of the geodesic growth (see docs/VERTEBRAE_ASFAND.md, limitations)
    assert np.mean(out[truth > 0] == truth[truth > 0]) > 0.95
    for level in range(1, N_BODIES + 1):              # every level is one connected piece
        assert engine.cc3d.connected_components(out == level).max() == 1


def test_floating_fragment_is_removed():
    truth, model = _spine(drift=False)
    model[1:3, 1:3, 1:3] = 7                           # tiny stray T11 voxels far from the spine
    out, report = engine.refine(model, SPACING, log=lambda m: None)
    assert report["fragments_removed"] == 1
    assert not out[1:3, 1:3, 1:3].any()


def _segmentation_dict(combined):
    return {name: (combined == i).astype(np.uint8) for i, name in enumerate(adapter.VERTEBRA_NAMES, 1)}


def test_adapter_handles_any_si_axis():
    truth, model = _spine(drift=True)
    # move the superior axis to the front and make it point down (I), as in some scanners
    moved = np.flip(np.moveaxis(model, 2, 0), axis=0).copy()
    affine = np.array([[0, 1.0, 0, 0], [0, 0, 1.0, 0], [-2.0, 0, 0, 0], [0, 0, 0, 1]])  # axcodes I, R, A
    ref = nib.Nifti1Image(moved, affine)
    assert nib.aff2axcodes(affine)[0] == "I"
    out_dict = adapter.postprocessing_vertebrae_asfand("synthetic", _segmentation_dict(moved), ref, LOGGER)
    combined = np.zeros_like(moved)
    for i, name in enumerate(adapter.VERTEBRA_NAMES, 1):
        combined[out_dict[name] > 0] = i
    expected = np.flip(np.moveaxis(truth, 2, 0), axis=0)
    bodies = np.flip(np.moveaxis(_body_mask(truth.shape), 2, 0), axis=0) & (expected > 0)
    assert np.array_equal(combined[bodies], expected[bodies])


def test_adapter_failure_leaves_masks_unchanged():
    truth, model = _spine(drift=True)
    seg = _segmentation_dict(model)
    seg_out = adapter.postprocessing_vertebrae_asfand("broken", seg, None, LOGGER)  # no reference -> error path
    assert seg_out is seg


def test_adapter_skips_cases_without_vertebrae():
    seg = {"liver": np.ones((4, 4, 4), np.uint8)}
    assert adapter.postprocessing_vertebrae_asfand("noverts", seg, None, LOGGER) is seg
