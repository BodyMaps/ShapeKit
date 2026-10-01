"""The regression checker must reject changes, not merely run successfully."""

import json

import nibabel as nib
import numpy as np

from tools.verify_vertebrae_hao import compare_case
from utils.vertebrae_hao_engine import VERTEBRA_NAMES


def make_outputs(tmp_path):
    actual, reference = tmp_path / 'actual', tmp_path / 'reference'
    case = 'synthetic'
    raw = np.zeros((8, 8, 8), dtype=np.uint8)
    raw.ravel()[:25] = np.arange(25)
    mapped = np.where(raw > 0, raw + 25, 0).astype(np.uint8)
    for root, data in ((actual, mapped), (reference, raw)):
        folder = root / case
        (folder / 'segmentations').mkdir(parents=True)
        nib.save(nib.Nifti1Image(data, np.eye(4)), folder / 'combined_labels.nii.gz')
        for label, name in enumerate(VERTEBRA_NAMES, 1):
            nib.save(nib.Nifti1Image((raw == label).astype(np.uint8), np.eye(4)),
                     folder / 'segmentations' / f'{name}.nii.gz')
    report = {'integration_status': 'complete',
              'label_map': {str(i): i + 25 for i in range(1, 25)}}
    (actual / case / 'vertebrae_hao_report.json').write_text(json.dumps(report))
    return actual, reference, case


def test_checker_accepts_exact_named_masks_after_mapping(tmp_path):
    actual, reference, case = make_outputs(tmp_path)
    result = compare_case(actual, reference, case)
    assert result['passed']
    assert result['changed_combined_voxels'] == 0
    assert len(result['masks']) == 24


def test_checker_accepts_configured_actual_mask_subfolder(tmp_path):
    actual, reference, case = make_outputs(tmp_path)
    (actual / case / 'segmentations').rename(actual / case / 'masks')
    result = compare_case(actual, reference, case, actual_subfolder='masks')
    assert result['passed']
    assert len(result['masks']) == 24


def test_checker_rejects_wrong_binary_mask(tmp_path):
    actual, reference, case = make_outputs(tmp_path)
    path = actual / case / 'segmentations' / 'vertebrae_L1.nii.gz'
    image = nib.load(path)
    data = np.asanyarray(image.dataobj).copy()
    data[7, 7, 7] = 1
    nib.save(nib.Nifti1Image(data, image.affine, image.header), path)
    result = compare_case(actual, reference, case)
    assert not result['passed']
    assert result['masks'][4]['changed_voxels'] == 1
    assert not result['masks'][4]['actual_combined_consistent']


def test_checker_rejects_geometry_change_with_identical_voxels(tmp_path):
    actual, reference, case = make_outputs(tmp_path)
    path = actual / case / 'segmentations' / 'vertebrae_L5.nii.gz'
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 1
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), path)
    result = compare_case(actual, reference, case)
    assert not result['passed']
    assert result['masks'][0]['changed_voxels'] == 0
    assert not result['masks'][0]['geometry']['affine']


def test_checker_rejects_wrong_combined_labels(tmp_path):
    actual, reference, case = make_outputs(tmp_path)
    path = actual / case / 'combined_labels.nii.gz'
    image = nib.load(path)
    data = np.asanyarray(image.dataobj).copy()
    data.ravel()[1] = 27
    nib.save(nib.Nifti1Image(data, image.affine, image.header), path)
    result = compare_case(actual, reference, case)
    assert not result['passed']
    assert result['changed_combined_voxels'] == 1
