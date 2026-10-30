import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from utils.vertebrae_landmark import NAMES, REPORT_NAME, check_orientation, process_case
from utils import vertebrae_landmark_engine as engine


class LandmarkAdapterTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        (self.source / 'segmentations').mkdir(parents=True)
        self.mapping = {i + 26: n for i, n in enumerate(NAMES)}
        self.ct = self.root / 'ct.nii.gz'
        nib.save(nib.Nifti1Image(np.full((24, 24, 24), 300, np.float32), np.eye(4)), self.ct)

    def tearDown(self):
        self.temp.cleanup()

    def mask(self, name, data, affine=None):
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine),
                 self.source / 'segmentations' / f'{name}.nii.gz')

    def run_case(self):
        return process_case(self.source, self.root / 'output', self.ct, self.mapping)

    def test_all_24_masks_report_and_mapping(self):
        a = np.zeros((24, 24, 24), np.uint8)
        a[8:16, 8:16, 8:16] = 1
        self.mask(NAMES[0], a)
        report = self.run_case()
        out = self.root / 'output'
        self.assertEqual(len(list((out / 'segmentations').glob('*.nii.gz'))), 24)
        self.assertEqual(report['label_map']['1'], 26)
        self.assertEqual(len(report['missing_input_masks']), 23)
        self.assertEqual(json.loads((out / REPORT_NAME).read_text())['engine'], 'shapekit_landmark')
        combined = np.asarray(nib.load(out / 'combined_labels.nii.gz').dataobj)
        self.assertEqual(combined[10, 10, 10], 26)

    def test_never_adds_bone(self):
        a = np.zeros((24, 24, 24), np.uint8)
        a[4:10, 4:10, 4:10] = 1
        b = np.zeros_like(a)
        b[14:20, 14:20, 14:20] = 1
        self.mask(NAMES[0], a)
        self.mask(NAMES[1], b)
        self.run_case()
        combined = np.asarray(nib.load(self.root / 'output/combined_labels.nii.gz').dataobj)
        self.assertFalse(np.any((combined > 0) & ((a | b) == 0)))

    def test_other_masks_copied_unchanged(self):
        a = np.zeros((24, 24, 24), np.uint8)
        a[8:16, 8:16, 8:16] = 1
        self.mask(NAMES[0], a)
        other = np.zeros_like(a)
        other[0, 0, 0] = 1
        self.mask('liver', other)
        self.mapping[1] = 'liver'
        data = (self.source / 'segmentations/liver.nii.gz').read_bytes()
        self.run_case()
        self.assertEqual(data, (self.root / 'output/segmentations/liver.nii.gz').read_bytes())
        out = np.asarray(nib.load(self.root / 'output/combined_labels.nii.gz').dataobj)
        self.assertEqual(out[0, 0, 0], 1)

    def test_far_stray_removed(self):
        # a small stray piece of "C1" far from its vertebra and from every other vertebra
        c1 = np.zeros((24, 24, 60), np.uint8)
        c1[8:16, 8:16, 50:58] = 1
        c1[10:12, 10:12, 2:4] = 1
        self.mask(NAMES[-1], c1)
        nib.save(nib.Nifti1Image(np.full((24, 24, 60), 400, np.float32), np.eye(4)), self.ct)
        self.run_case()
        out = np.asarray(nib.load(self.root / 'output/segmentations/vertebrae_C1.nii.gz').dataobj)
        self.assertEqual(out[10:12, 10:12, 2:4].sum(), 0)
        self.assertEqual(out[8:16, 8:16, 50:58].sum(), 8 * 8 * 8)

    def test_overlap_rejected(self):
        a = np.zeros((24, 24, 24), np.uint8)
        a[10, 10, 10] = 1
        self.mask(NAMES[0], a)
        self.mask(NAMES[1], a)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            self.run_case()

    def test_nonbinary_rejected(self):
        self.mask(NAMES[0], np.full((24, 24, 24), 2, np.uint8))
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.run_case()

    def test_geometry_rejected(self):
        self.mask(NAMES[0], np.zeros((24, 24, 24), np.uint8))
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), np.eye(4)), self.ct)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.run_case()

    def test_oblique_rejected(self):
        c, s = np.cos(0.3), np.sin(0.3)
        affine = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with self.assertRaisesRegex(ValueError, 'Oblique'):
            check_orientation(nib.Nifti1Image(np.zeros((2, 2, 2)), affine))

    def test_flipped_axes_accepted(self):
        check_orientation(nib.Nifti1Image(np.zeros((2, 2, 2)), np.diag([-1, 1, 1, 1])))

    def test_missing_ct_rejected(self):
        self.mask(NAMES[0], np.zeros((24, 24, 24), np.uint8))
        self.ct.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_case()

    def test_existing_output_rejected(self):
        (self.root / 'output').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_case()


class LandmarkEngineTests(unittest.TestCase):

    def test_assign_names_keeps_order(self):
        # two bodies both mostly "T8" (label 10): the order constraint forces 10 and 11
        names = engine.assign_names([{9: 1.0}, {10: 0.6, 11: 0.4}, {10: 0.64, 11: 0.36}], 0.3)
        self.assertEqual(names, [9, 10, 11])

    def test_order_ok(self):
        lab = np.zeros((4, 4, 10), np.int16)
        lab[:, :, 1:3] = 1
        lab[:, :, 5:7] = 2
        self.assertTrue(engine.order_ok(lab))
        lab2 = np.zeros_like(lab)
        lab2[:, :, 1:3] = 2
        lab2[:, :, 5:7] = 1
        self.assertFalse(engine.order_ok(lab2))

    def test_refine_does_not_modify_input(self):
        lab = np.zeros((20, 20, 20), np.int16)
        lab[5:15, 5:15, 5:15] = 1
        before = lab.copy()
        engine.refine(lab, np.full(lab.shape, 300, np.float32), np.eye(4), (1.0, 1.0, 1.0))
        self.assertTrue(np.array_equal(lab, before))


if __name__ == '__main__':
    unittest.main()
