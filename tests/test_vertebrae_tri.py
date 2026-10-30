"""Regression/contract tests for the standalone numerical engine and adapter."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi

from utils.vertebrae_tri_engine import (
    correct_global_numbering, find_connected_components, refine_vertebrae,
)

try:
    import nibabel as nib
    from utils.vertebrae_tri import NAMES, process_case
    HAVE_NIB = True
except ImportError:
    HAVE_NIB = False


def sample(shape=(48, 48, 144)):
    raw = np.zeros(shape, dtype=np.uint8)
    for level, z in enumerate((12, 42, 72, 102), 1):
        raw[17:31, 18:30, z:z + 14] = level
    return raw


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.affine = np.diag([1., 1., 1., 1.])

    def test_empty_case_is_unchanged(self):
        raw = np.zeros((12, 12, 12), np.uint8)
        actual, stages = refine_vertebrae(raw, self.affine, return_stages=True)
        self.assertTrue(np.array_equal(actual, raw))
        self.assertTrue(np.array_equal(stages['numbering'], raw))

    def test_one_vertebra_abstains(self):
        raw = np.zeros((32, 32, 40), np.uint8)
        raw[8:22, 8:22, 12:30] = 1
        self.assertTrue(np.array_equal(refine_vertebrae(raw, self.affine), raw))

    def test_regular_case_preserves_all_labels(self):
        raw = sample()
        output, stages = refine_vertebrae(raw, self.affine, return_stages=True)
        self.assertEqual(output.shape, raw.shape)
        self.assertTrue(np.array_equal(output != 0, raw != 0))
        self.assertTrue(np.array_equal(stages['numbering'], raw))
        self.assertTrue(np.array_equal(output, raw))

    def test_invalid_segmentation_rejected(self):
        raw = sample()
        raw[0, 0, 0] = 25
        with self.assertRaises(ValueError):
            refine_vertebrae(raw, self.affine)
        with self.assertRaises(ValueError):
            refine_vertebrae(raw.astype(float), self.affine)

    def test_invalid_affine_rejected(self):
        with self.assertRaises(ValueError):
            refine_vertebrae(sample(), np.zeros((4, 4)))

    def test_cc_matches_full_volume_reference(self):
        mask = sample()
        mask[1:3, 1:3, 1:3] = 2
        components = find_connected_components(mask, self.affine)
        labeled, n = ndi.label(mask == 2, structure=np.ones((3, 3, 3)))
        self.assertEqual(n, len(components[2]))
        for found in components[2]:
            cc = labeled == found['id']
            self.assertEqual(found['num_voxels'], int(cc.sum()))
            self.assertTrue(np.allclose(found['centroid_voxel'], ndi.center_of_mass(cc)))

    def test_anisotropic_and_axis_flipped(self):
        raw = sample()
        affine = np.diag([0.85, 1.3, -2.0, 1.0])
        affine[2, 3] = 300
        output = refine_vertebrae(raw, affine)
        self.assertEqual(output.shape, raw.shape)
        self.assertFalse(np.any((output < 0) | (output > 24)))

    def test_non_mutating(self):
        raw = sample()
        saved = raw.copy()
        refine_vertebrae(raw, self.affine)
        self.assertTrue(np.array_equal(saved, raw))


@unittest.skipUnless(HAVE_NIB, 'NiBabel required for NIfTI adapter tests')
class AdapterTests(unittest.TestCase):
    def make_case(self, directory, overlap=False):
        src = Path(directory) / 'input' / 'CASE'
        seg = src / 'segmentations'
        seg.mkdir(parents=True)
        raw = sample()
        affine = np.diag([1., 1., 1., 1.])
        for i, name in enumerate(NAMES[:4], 1):
            mask = raw == i
            if overlap and i == 2:
                mask[raw == 1] = True
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(seg / f'{name}.nii.gz'))
        return src

    def class_map(self):
        return {i + 26: name for i, name in enumerate(NAMES)}

    def test_case_roundtrip_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_case(tmp)
            output = Path(tmp) / 'output' / 'CASE'
            report = process_case(source, output, None, self.class_map())
            self.assertTrue((output / 'combined_labels.nii.gz').exists())
            self.assertEqual(len(list((output / 'segmentations').glob('vertebrae_*.nii.gz'))), 24)
            combined = np.asarray(nib.load(str(output / 'combined_labels.nii.gz')).dataobj)
            self.assertEqual(set(np.unique(combined)), {0, 26, 27, 28, 29})
            self.assertEqual(len(report['missing_masks']), 20)
            self.assertTrue((output / 'vertebrae_tri_report.json').exists())
            # Source is left untouched and a second run cannot overwrite outputs.
            self.assertEqual(len(list((source / 'segmentations').glob('vertebrae_*.nii.gz'))), 4)
            with self.assertRaises(FileExistsError):
                process_case(source, output, None, self.class_map())

    def test_overlapping_labels_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_case(tmp, overlap=True)
            output = Path(tmp) / 'output' / 'CASE'
            with self.assertRaises(ValueError):
                process_case(source, output, None, self.class_map())
            self.assertFalse(output.exists())

    def test_grid_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_case(tmp)
            path = source / 'segmentations' / f'{NAMES[1]}.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((40, 40, 40), np.uint8), np.eye(4)), str(path))
            with self.assertRaises(ValueError):
                process_case(source, Path(tmp) / 'output' / 'CASE', None, self.class_map())


if __name__ == '__main__':
    unittest.main()
