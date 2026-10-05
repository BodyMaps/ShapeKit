import json
from pathlib import Path
import tempfile
import unittest
import nibabel as nib
import numpy as np
from utils.vertebrae_ct_refinement import process_case, load_masks, check_orientation, NAMES

class CTRefinementAdapterTests(unittest.TestCase):

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
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), self.source / 'segmentations' / f'{name}.nii.gz')

    def run_case(self):
        return process_case(self.source, self.root / 'output', self.ct, self.mapping)

    def test_empty_output_and_mapping(self):
        self.mask(NAMES[0], np.zeros((24, 24, 24), np.uint8))
        result = self.run_case()
        self.assertEqual(result['identity']['status'], 'RAW_EMPTY')
        self.assertEqual(result['label_map']['1'], 26)
        self.assertEqual(len(list((self.root / 'output/segmentations').glob('*.nii.gz'))), 24)

    def test_nonempty_growth_and_preserved_other_mask(self):
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
        self.assertEqual(out[10, 10, 10], 26)
        self.assertEqual(out[0, 0, 0], 1)

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

    def test_orientation_rejected(self):
        affine = np.diag([1, -1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'orientation'):
            check_orientation(nib.Nifti1Image(np.zeros((2, 2, 2)), affine))

    def test_missing_ct_rejected(self):
        self.mask(NAMES[0], np.zeros((24, 24, 24), np.uint8))
        self.ct.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_case()

    def test_existing_output_rejected(self):
        (self.root / 'output').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_case()
if __name__ == '__main__':
    unittest.main()
