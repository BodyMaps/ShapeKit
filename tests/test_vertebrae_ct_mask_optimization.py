"""Tests for the shapekit_ct_mask_optimization adapter (utils/vertebrae_ct_mask_optimization.py).

The fast tests use small synthetic cases and check the adapter's contract: inputs it refuses, the output layout,
label IDs, other organs copied unchanged, and the missing-CT policy. The demo-case test checks that the adapter
reproduces a reference output voxel for voxel; it runs only when the data is available:
    CTMO_DEMO_PRED=<folder with BDMAP_*/segmentations/vertebrae_*.nii.gz>
    CTMO_DEMO_CT=<folder with BDMAP_*/ct.nii.gz>
    CTMO_DEMO_REFERENCE=<folder with the expected BDMAP_*/segmentations/vertebrae_*.nii.gz>
Run from the repo root: python -m unittest tests.test_vertebrae_ct_mask_optimization
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from utils.vertebrae_ct_mask_optimization import REPORT_FILE, VERTEBRA_NAMES, process_case

SHAPE = (24, 24, 24)


class CTMaskOptimizationAdapterTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input" / "case"
        (self.source / "segmentations").mkdir(parents=True)
        self.output = self.root / "output" / "case"
        self.class_map = {26 + i: name for i, name in enumerate(VERTEBRA_NAMES)}
        self.ct = self.root / "ct" / "ct.nii.gz"
        self.ct.parent.mkdir()
        self.save_ct(np.eye(4))

    def tearDown(self):
        self.temp.cleanup()

    def save_ct(self, affine, shape=SHAPE):
        nib.save(nib.Nifti1Image(np.full(shape, 300, np.int16), affine), self.ct)

    def mask(self, name, data, affine=None):
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine),
                 self.source / "segmentations" / f"{name}.nii.gz")

    def cube(self):
        data = np.zeros(SHAPE, np.uint8)
        data[8:16, 8:16, 8:16] = 1
        return data

    def run_case(self, **kwargs):
        return process_case(self.source, self.output, self.ct, self.class_map, **kwargs)

    def test_empty_masks_give_24_empty_outputs_and_the_label_map(self):
        self.mask(VERTEBRA_NAMES[0], np.zeros(SHAPE, np.uint8))
        report = self.run_case()
        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["label_map"]["1"], 26)
        self.assertEqual(report["label_map"]["24"], 49)
        self.assertEqual(len(report["missing_input_masks"]), 23)
        self.assertEqual(len(list((self.output / "segmentations").glob("vertebrae_*.nii.gz"))), 24)
        self.assertEqual(json.loads((self.output / REPORT_FILE).read_text())["case"], "case")

    def test_other_organs_copied_unchanged_and_combined_uses_shapekit_ids(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        liver = np.zeros(SHAPE, np.uint8)
        liver[0, 0, 0] = 1
        self.mask("liver", liver)
        self.class_map[5] = "liver"
        original = (self.source / "segmentations" / "liver.nii.gz").read_bytes()
        self.run_case()
        self.assertEqual((self.output / "segmentations" / "liver.nii.gz").read_bytes(), original)
        combined = np.asarray(nib.load(self.output / "combined_labels.nii.gz").dataobj)
        self.assertEqual(combined[12, 12, 12], 26)
        self.assertEqual(combined[0, 0, 0], 5)

    def test_output_keeps_the_input_grid_in_any_orientation(self):
        affine = np.diag([-1.0, -1.0, 2.0, 1.0])                 # LPS, 2 mm slices
        self.mask(VERTEBRA_NAMES[0], self.cube(), affine)
        self.save_ct(affine)
        self.run_case()
        out = nib.load(self.output / "segmentations" / f"{VERTEBRA_NAMES[0]}.nii.gz")
        self.assertTrue(np.allclose(out.affine, affine))
        self.assertEqual(nib.aff2axcodes(out.affine), ("L", "P", "S"))

    def test_overlapping_masks_rejected(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        self.mask(VERTEBRA_NAMES[1], self.cube())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.run_case()

    def test_non_binary_mask_rejected(self):
        self.mask(VERTEBRA_NAMES[0], np.full(SHAPE, 2, np.uint8))
        with self.assertRaisesRegex(ValueError, "binary"):
            self.run_case()

    def test_masks_on_different_grids_rejected(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        shifted = np.eye(4)
        shifted[0, 3] = 5.0
        self.mask(VERTEBRA_NAMES[1], np.zeros(SHAPE, np.uint8), shifted)
        with self.assertRaisesRegex(ValueError, "voxel grid"):
            self.run_case()

    def test_ct_on_another_grid_rejected(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        shifted = np.eye(4)
        shifted[0, 3] = 5.0
        self.save_ct(shifted)                                   # same shape, different position
        with self.assertRaisesRegex(ValueError, "voxel grid"):
            self.run_case()
        self.save_ct(np.eye(4), shape=(10, 10, 10))
        with self.assertRaisesRegex(ValueError, "voxel grid"):
            self.run_case()

    def test_sheared_affine_rejected(self):
        sheared = np.eye(4)
        sheared[0, 1] = 0.5
        self.mask(VERTEBRA_NAMES[0], self.cube(), sheared)
        with self.assertRaisesRegex(ValueError, "sheared"):
            self.run_case()

    def test_missing_ct_fails_by_default_and_writes_nothing(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        self.ct.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_case()
        self.assertFalse(self.output.exists())

    def test_missing_ct_can_hand_off_to_shapekit_songlin(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        self.ct.unlink()
        report = self.run_case(missing_ct="shapekit_songlin")
        self.assertIn("shapekit_songlin", report["status"])
        self.assertIsNone(report["ct"])
        self.assertEqual(len(list((self.output / "segmentations").glob("vertebrae_*.nii.gz"))), 24)

    def test_invalid_settings_rejected(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        with self.assertRaisesRegex(ValueError, "missing_ct"):
            self.run_case(missing_ct="ignore")
        del self.class_map[49]
        with self.assertRaisesRegex(ValueError, "24 vertebrae"):
            self.run_case()

    def test_existing_or_nested_output_rejected(self):
        self.mask(VERTEBRA_NAMES[0], self.cube())
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_case()
        with self.assertRaisesRegex(ValueError, "separate"):
            process_case(self.source, self.source / "out", self.ct, self.class_map)


@unittest.skipUnless(all(os.environ.get(k) for k in ("CTMO_DEMO_PRED", "CTMO_DEMO_CT", "CTMO_DEMO_REFERENCE")),
                     "demo data not configured (CTMO_DEMO_PRED, CTMO_DEMO_CT, CTMO_DEMO_REFERENCE)")
class CTMaskOptimizationDemoEquivalence(unittest.TestCase):

    def test_demo_cases_match_the_reference_voxel_for_voxel(self):
        pred, ct, reference = (Path(os.environ[k]) for k in ("CTMO_DEMO_PRED", "CTMO_DEMO_CT", "CTMO_DEMO_REFERENCE"))
        class_map = {26 + i: name for i, name in enumerate(VERTEBRA_NAMES)}
        with tempfile.TemporaryDirectory() as temp:
            for case in sorted(p.name for p in reference.iterdir() if p.is_dir()):
                process_case(pred / case, Path(temp) / case, ct / case / "ct.nii.gz", class_map)
                for name in VERTEBRA_NAMES:
                    got = nib.load(Path(temp) / case / "segmentations" / f"{name}.nii.gz")
                    want = nib.load(reference / case / "segmentations" / f"{name}.nii.gz")
                    self.assertTrue(np.array_equal(np.asarray(got.dataobj), np.asarray(want.dataobj)), f"{case} {name}")
                    self.assertTrue(np.allclose(got.affine, want.affine), f"{case} {name} affine")


if __name__ == "__main__":
    unittest.main()
