"""Contract tests for the ShapeKit-Aaron vertebrae adapter on a synthetic spine."""

import logging

import nibabel as nib
import numpy as np
import pytest

from utils import vertebrae_aaron as adapter

LOGGER = logging.getLogger(__name__)
NAMES = adapter.VERTEBRA_NAMES        # vertebrae_L5 ... vertebrae_C1
N_BODIES = 6                          # L5 ... T12
BODY_Z, DISC_Z = 18, 6                # mm, 1 mm voxels


def _spine(ai_labels, top_sliver=None):
    """Stacked 24 x 24 x 18 mm bodies along +z (RAS), labelled with ai_labels.

    Returns the label volume and the z-slices of each body. top_sliver adds a
    vertebra cut off by the scan edge (a few slices, no usable core).
    """
    depth = N_BODIES * (BODY_Z + DISC_Z) + 10
    lab = np.zeros((40, 40, depth), np.uint8)
    slices = []
    for i, k in enumerate(ai_labels):
        z0 = 4 + i * (BODY_Z + DISC_Z)
        lab[8:32, 8:32, z0:z0 + BODY_Z] = k
        slices.append(slice(z0, z0 + BODY_Z))
    if top_sliver is not None:
        lab[8:32, 8:32, depth - 4:] = top_sliver
    return lab, slices


def _as_dict(lab, extra=None):
    d = {name: (lab == k).astype(np.uint8) for k, name in enumerate(NAMES, start=1)}
    d.update(extra or {})
    return d


def _labels_from_dict(d, shape):
    out = np.zeros(shape, np.uint8)
    for k, name in enumerate(NAMES, start=1):
        if d.get(name) is not None:
            out[d[name] > 0] = k
    return out


def _body_labels(lab, slices):
    """The single label of each body; fails if a body carries several labels."""
    found = []
    for sl in slices:
        values = np.unique(lab[8:32, 8:32, sl])
        assert len(values) == 1, f"body split between labels {values}"
        found.append(int(values[0]))
    return found


def test_duplicate_label_is_resolved_into_anatomical_order():
    # the AI used L3 (label 3) on two bodies; everything above is one level low
    lab, slices = _spine([1, 2, 3, 3, 4, 5])
    ref = nib.Nifti1Image(lab, np.eye(4))
    out = adapter.postprocessing_vertebrae_aaron("synthetic", _as_dict(lab), ref, LOGGER)
    labels = _body_labels(_labels_from_dict(out, lab.shape), slices)
    assert labels == sorted(set(labels)), labels   # strictly increasing L5 -> C1
    assert labels[0] == 1


def test_result_does_not_depend_on_voxel_orientation():
    lab, _ = _spine([1, 2, 3, 3, 4, 5])
    ras = adapter.postprocessing_vertebrae_aaron(
        "ras", _as_dict(lab), nib.Nifti1Image(lab, np.eye(4)), LOGGER)
    expected = _labels_from_dict(ras, lab.shape)

    # same anatomy stored as L, P, I (all three axes flipped)
    flipped = lab[::-1, ::-1, ::-1].copy()
    affine = np.diag([-1.0, -1.0, -1.0, 1.0])
    affine[:3, 3] = np.array(lab.shape) - 1
    lpi = adapter.postprocessing_vertebrae_aaron(
        "lpi", _as_dict(flipped), nib.Nifti1Image(flipped, affine), LOGGER)
    got = _labels_from_dict(lpi, lab.shape)[::-1, ::-1, ::-1]
    np.testing.assert_array_equal(got, expected)


def test_other_organs_are_untouched():
    lab, _ = _spine([1, 2, 3, 4, 5, 6])
    liver = np.zeros(lab.shape, np.uint8)
    liver[0:5, 0:5, 0:5] = 1
    out = adapter.postprocessing_vertebrae_aaron(
        "organs", _as_dict(lab, {"liver": liver.copy()}), nib.Nifti1Image(lab, np.eye(4)), LOGGER)
    np.testing.assert_array_equal(out["liver"], liver)


def test_cut_off_vertebra_keeps_its_ai_label():
    # T11 (label 7) only has 4 slices inside the scan: no body core, keep the AI label
    lab, _ = _spine([1, 2, 3, 4, 5, 6], top_sliver=7)
    out = adapter.postprocessing_vertebrae_aaron(
        "cutoff", _as_dict(lab), nib.Nifti1Image(lab, np.eye(4)), LOGGER)
    result = _labels_from_dict(out, lab.shape)
    np.testing.assert_array_equal(result[lab == 7], 7)


@pytest.mark.parametrize("n_present", [0, 2])
def test_too_few_vertebrae_returns_dict_unchanged(n_present):
    lab, _ = _spine(list(range(1, n_present + 1)) or [1])
    if n_present == 0:
        lab[:] = 0
    d = _as_dict(lab)
    before = {k: v.copy() for k, v in d.items()}
    out = adapter.postprocessing_vertebrae_aaron("few", d, nib.Nifti1Image(lab, np.eye(4)), LOGGER)
    for k in before:
        np.testing.assert_array_equal(out[k], before[k])
