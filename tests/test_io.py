from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from utils.utils import read_all_segmentations, save_and_combine_segmentations


def _save_mask(path: Path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.uint8), affine), str(path))


def test_each_mask_is_aligned_to_reference_grid(tmp_path):
    shape = (4, 5, 6)
    reference_data = np.zeros(shape, dtype=np.uint8)
    reference_data[1, 2, 3] = 1
    reference_affine = np.eye(4)
    reference_path = tmp_path / 'segmentations' / 'liver.nii.gz'
    _save_mask(reference_path, reference_data, reference_affine)

    flipped_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    flipped_affine[0, 3] = shape[0] - 1
    _save_mask(
        tmp_path / 'segmentations' / 'spleen.nii.gz',
        np.flip(reference_data, axis=0),
        flipped_affine,
    )

    reference_img = nib.load(str(reference_path))
    masks = read_all_segmentations(
        str(tmp_path), ['liver', 'spleen'], reference_img=reference_img
    )

    np.testing.assert_array_equal(masks['liver'], reference_data)
    np.testing.assert_array_equal(masks['spleen'], reference_data)


def test_non_overlapping_mask_is_rejected(tmp_path):
    data = np.ones((3, 3, 3), dtype=np.uint8)
    reference_path = tmp_path / 'segmentations' / 'liver.nii.gz'
    _save_mask(reference_path, data, np.eye(4))
    distant_affine = np.eye(4)
    distant_affine[:3, 3] = 1000
    _save_mask(tmp_path / 'segmentations' / 'spleen.nii.gz', data, distant_affine)

    with pytest.raises(ValueError, match='does not overlap'):
        read_all_segmentations(
            str(tmp_path),
            ['liver', 'spleen'],
            reference_img=nib.load(str(reference_path)),
        )


def test_save_combines_in_memory_and_preserves_grid(tmp_path):
    reference = nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.uint8), np.diag([2, 2, 2, 1]))
    liver = np.zeros(reference.shape, dtype=np.uint8)
    spleen = np.zeros(reference.shape, dtype=np.uint8)
    liver[0, 0, 0] = 1
    spleen[1, 1, 1] = 1

    save_and_combine_segmentations(
        {'liver': liver, 'spleen': spleen},
        {1: 'liver', 2: 'spleen'},
        reference,
        str(tmp_path),
        True,
    )

    combined = nib.load(str(tmp_path / 'combined_labels.nii.gz'))
    assert combined.get_data_dtype() == np.dtype(np.uint8)
    np.testing.assert_allclose(combined.affine, reference.affine)
    data = np.asanyarray(combined.dataobj)
    assert data[0, 0, 0] == 1
    assert data[1, 1, 1] == 2


def test_save_rejects_wrong_shape(tmp_path):
    reference = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
    with pytest.raises(ValueError, match='expected'):
        save_and_combine_segmentations(
            {'liver': np.ones((2, 2, 2))},
            {1: 'liver'},
            reference,
            str(tmp_path),
            True,
        )
