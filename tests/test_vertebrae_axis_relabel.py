import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from utils import vertebrae_axis_relabel_engine as engine
from utils.vertebrae_axis_relabel import NAMES, process_case


def synthetic_spine(n=6, height=22, gap=5, radius=13):
    """n vertebrae (body + arch + spinous process) stacked along z, 1 mm voxels, labels 1..n."""
    shape = (64, 72, n * (height + gap) + 20)
    x, y, z = np.meshgrid(*[np.arange(s) + 0.5 for s in shape], indexing='ij')
    labels = np.zeros(shape, np.uint8)
    ct = np.zeros(shape, np.float32)
    for k in range(n):
        z0 = 10 + k * (height + gap)
        level = (z >= z0) & (z < z0 + height)
        body = level & ((x - 32) ** 2 + (y - 30) ** 2 <= radius ** 2)
        upper = level & (z < z0 + height - 6)
        spinous = upper & (np.abs(x - 32) <= 4) & (y > 30 + radius - 2) & (y < 30 + radius + 18)
        arch = upper & (np.abs(np.hypot(x - 32, y - 30 - radius - 6) - 7) <= 2)
        vertebra = body | spinous | arch
        labels[vertebra] = k + 1
        ct[vertebra] = 400
        ct[vertebra & ~ndi.binary_erosion(vertebra, iterations=2)] = 900   # cortical shell
    ct += np.random.default_rng(0).normal(0, 10, shape).astype(np.float32)
    return labels, ct


def mislabel(labels):
    """give the upper half of vertebra 3 the label of vertebra 4"""
    wrong = labels.copy()
    wrong[(labels == 3) & (np.arange(labels.shape[2])[None, None, :] > 75)] = 4
    return wrong


class EngineTests(unittest.TestCase):

    def test_fixes_mislabelled_body(self):
        truth, ct = synthetic_spine()
        wrong = mislabel(truth)
        result, status = engine.relabel_volume(wrong, ct, np.eye(4), (1, 1, 1))
        self.assertEqual(status, 'ok')
        self.assertGreater((wrong != truth).sum(), 1000)
        self.assertEqual((result != truth).sum(), 0)

    def test_orientation_does_not_change_result(self):
        truth, ct = synthetic_spine()
        wrong = mislabel(truth)
        ras, _ = engine.relabel_volume(wrong, ct, np.eye(4), (1, 1, 1))
        las_affine = np.diag([-1.0, 1, 1, 1])
        las, _ = engine.relabel_volume(wrong[::-1], ct[::-1], las_affine, (1, 1, 1))
        np.testing.assert_array_equal(las[::-1], ras)

    def test_no_spine_keeps_input(self):
        labels = np.zeros((30, 30, 30), np.uint8)
        labels[10:13, 10:13, 10:13] = 5
        result, status = engine.relabel_volume(labels, np.zeros(labels.shape, np.float32), np.eye(4), (1, 1, 1))
        self.assertEqual(status, 'spine not found')
        np.testing.assert_array_equal(result, labels)

    def test_empty_input(self):
        labels = np.zeros((10, 10, 10), np.uint8)
        result, status = engine.relabel_volume(labels, np.zeros(labels.shape), np.eye(4), (1, 1, 1))
        self.assertEqual(status, 'empty')
        self.assertFalse(result.any())


class AdapterTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'input' / 'CASE'
        (self.source / 'segmentations').mkdir(parents=True)
        self.output = self.root / 'output' / 'CASE'
        self.class_map = {i + 26: name for i, name in enumerate(NAMES)}
        self.class_map[5] = 'liver'
        self.truth, ct = synthetic_spine()
        self.ct = self.root / 'ct.nii.gz'
        nib.save(nib.Nifti1Image(ct, np.eye(4)), self.ct)

    def tearDown(self):
        self.temp.cleanup()

    def write_mask(self, name, data, affine=None):
        nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4) if affine is None else affine),
                 self.source / 'segmentations' / f'{name}.nii.gz')

    def write_labels(self, labels):
        for label in range(1, 25):
            self.write_mask(NAMES[label - 1], labels == label)

    def run_case(self):
        return process_case(self.source, self.output, self.ct, self.class_map)

    def test_case_is_relabelled_and_other_masks_kept(self):
        self.write_labels(mislabel(self.truth))
        liver = np.zeros(self.truth.shape, np.uint8)
        liver[0:3, 0:3, 0:3] = 1
        self.write_mask('liver', liver)
        liver_bytes = (self.source / 'segmentations/liver.nii.gz').read_bytes()

        self.assertEqual(self.run_case(), 'ok')
        self.assertEqual(liver_bytes, (self.output / 'segmentations/liver.nii.gz').read_bytes())
        self.assertEqual(len(list((self.output / 'segmentations').glob('vertebrae_*.nii.gz'))), 24)
        combined = np.asarray(nib.load(self.output / 'combined_labels.nii.gz').dataobj)
        expected = np.where(self.truth > 0, self.truth + 25, 0)
        expected[liver > 0] = 5
        np.testing.assert_array_equal(combined, expected)
        l3 = np.asarray(nib.load(self.output / 'segmentations/vertebrae_L3.nii.gz').dataobj)
        np.testing.assert_array_equal(l3, (self.truth == 3).astype(np.uint8))
        self.assertFalse(any((self.root / 'output' / '.incomplete').iterdir()))

    def test_overlap_rejected(self):
        self.write_mask(NAMES[0], self.truth == 1)
        self.write_mask(NAMES[1], self.truth == 1)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            self.run_case()

    def test_nonbinary_rejected(self):
        self.write_mask(NAMES[0], np.full(self.truth.shape, 2))
        with self.assertRaisesRegex(ValueError, 'Binary'):
            self.run_case()

    def test_ct_geometry_rejected(self):
        self.write_labels(self.truth)
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), np.eye(4)), self.ct)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.run_case()

    def test_missing_ct_rejected(self):
        self.write_labels(self.truth)
        self.ct.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_case()

    def test_no_vertebra_masks_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No vertebra masks'):
            self.run_case()

    def test_existing_output_rejected(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_case()


if __name__ == '__main__':
    unittest.main()
