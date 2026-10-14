import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import shutil

import nibabel as nib
import numpy as np

from utils import vertebrae_components as adapter
from utils import vertebrae_components_engine as engine


class ComponentsTests(unittest.TestCase):
    def test_cleanup_remote_but_preserve_boundary(self):
        labels = np.zeros((140, 12, 12), np.uint8)
        labels[2:10, 2:10, 2:10] = 1
        labels[120, 5, 5] = 1
        labels[139, 5, 5] = 1
        after, changes = engine.conservative_cleanup(labels, np.ones(3))
        self.assertEqual(after[120, 5, 5], 0)
        self.assertEqual(after[139, 5, 5], 1)
        self.assertEqual(sum(c['removed_voxels'] for c in changes), 1)

    def test_intact_component_and_unseeded_fragment(self):
        labels = np.zeros((8, 8, 8), np.uint8)
        labels[1:4, 1:4, 1:4] = 6
        labels[6, 6, 6] = 6
        labels[5, 2, 2] = 5
        seeds = np.zeros_like(labels)
        seeds[2, 2, 2] = 5
        after, changes = engine.correct_components(labels, seeds, np.ones(3))
        self.assertTrue(np.all(after[1:4, 1:4, 1:4] == 5))
        self.assertEqual(after[6, 6, 6], 6)
        self.assertEqual(changes[0]['modified_voxels'], 27)

    def test_geodesic_anisotropy_and_markers(self):
        mask = np.ones((7, 7, 3), bool)
        seeds = np.zeros(mask.shape, np.int32)
        seeds[0, 3, 1] = 4
        seeds[3, 0, 1] = 5
        x_heavy = engine.split_component(mask, seeds, np.array([5., 1., 1.]))
        y_heavy = engine.split_component(mask, seeds, np.array([1., 5., 1.]))
        self.assertEqual(x_heavy[3, 3, 1], 5)
        self.assertEqual(y_heavy[3, 3, 1], 4)
        self.assertTrue(np.all(x_heavy[mask] > 0))
        np.testing.assert_array_equal(x_heavy[seeds > 0], seeds[seeds > 0])

    def test_unreachable_component_rejected(self):
        mask = np.zeros((8, 3, 3), bool)
        mask[1:3] = True
        mask[5:7] = True
        seeds = np.zeros(mask.shape, np.int32)
        seeds[1, 1, 1] = 1
        with self.assertRaisesRegex(RuntimeError, 'not assigned'):
            engine.split_component(mask, seeds, np.ones(3))

    def test_contact_relabels_only_supported_neighbor(self):
        original = np.zeros((9, 9, 9), np.uint8)
        original[1:4, 1:6, 1:6] = 5
        original[4:7, 1:6, 1:6] = 6
        seeds = np.zeros_like(original)
        seeds[2, 2, 2] = 5
        after, changes = engine.correct_contacts(original, original, seeds, np.array([2., 2., 2.]))
        self.assertTrue(np.all(after[4:7, 1:6, 1:6] == 5))
        self.assertEqual(len(changes), 1)

    def test_empty_cleanup(self):
        raw = np.zeros((3, 3, 3), np.uint8)
        after, _ = engine.refine(raw, np.ones(3), 1., {'mode': 'cleanup_only'})
        np.testing.assert_array_equal(after, raw)

    def test_reviewed_boundary_selection_is_checked(self):
        original = np.zeros((8, 8, 8), np.uint8)
        original[2:6, 2:6, 2:6] = 5
        with self.assertRaisesRegex(ValueError, 'size changed'):
            engine.split_reviewed_fragment(original, original, np.ones(3),
                                            {'original_label': 5, 'rank': 1, 'voxels': 63, 'targets': [4, 5]})

    def test_orientation_and_plan_mismatch(self):
        raw = np.zeros((3, 3, 3), np.uint8)
        ct = nib.Nifti1Image(raw, np.diag([-1., 1., 1., 1.]))
        ct.header.set_xyzt_units('mm')
        adapter.check_ct(ct)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            adapter.validate_plan({'schema_version': 1, 'mode': 'cleanup_only', 'source_sha256': 'wrong'}, raw, ct)
        with self.assertRaisesRegex(ValueError, 'LAS'):
            ras = nib.Nifti1Image(raw, np.eye(4))
            ras.header.set_xyzt_units('mm')
            adapter.check_ct(ras)

    def test_io_maps_labels_preserves_other_mask_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'input'
            masks = source / 'segmentations'
            masks.mkdir(parents=True)
            raw = np.zeros((6, 6, 6), np.uint8)
            raw[2:4, 2:4, 2:4] = 1
            ct = nib.Nifti1Image(raw.astype(np.int16), np.diag([-1., 1., 1., 1.]))
            ct.header.set_xyzt_units('mm')
            ct_path = root / 'ct.nii.gz'
            nib.save(ct, ct_path)
            for label, name in enumerate(adapter.NAMES, 1):
                nib.save(nib.Nifti1Image((raw == label).astype(np.uint8), ct.affine, ct.header), masks / (name + '.nii.gz'))
            extra = np.zeros(raw.shape, np.uint8)
            extra[0, 0, 0] = 1
            nib.save(nib.Nifti1Image(extra, ct.affine, ct.header), masks / 'liver.nii.gz')
            original_bytes = (masks / 'liver.nii.gz').read_bytes()
            plan = {'schema_version': 1, 'mode': 'cleanup_only', 'source_sha256': adapter.fingerprint(raw, ct.affine)}
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            mapping = {i + 26: n for i, n in enumerate(adapter.NAMES)}
            mapping[1] = 'liver'
            output = root / 'out'
            adapter.process_case(source, output, ct_path, mapping, plan_path=plan_path)
            combined = np.asanyarray(nib.load(output / 'combined_labels.nii.gz').dataobj)
            self.assertEqual(combined[2, 2, 2], 26)
            self.assertEqual(combined[0, 0, 0], 1)
            self.assertEqual((output / 'segmentations/liver.nii.gz').read_bytes(), original_bytes)
            self.assertEqual(len(list((output / 'segmentations').glob('vertebrae_*.nii.gz'))), 24)
            with self.assertRaises(FileExistsError):
                adapter.process_case(source, output, ct_path, mapping, plan_path=plan_path)

    def test_batch_cli_reports_missing_plan_and_processes_valid_case(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'input' / 'sample'
            masks = source / 'segmentations'
            masks.mkdir(parents=True)
            raw = np.zeros((5, 5, 5), np.uint8)
            raw[2, 2, 2] = 1
            ct = nib.Nifti1Image(raw.astype(np.int16), np.diag([-1., 1., 1., 1.]))
            ct.header.set_xyzt_units('mm')
            nib.save(ct, source / 'ct.nii.gz')
            for label, name in enumerate(adapter.NAMES, 1):
                nib.save(nib.Nifti1Image((raw == label).astype(np.uint8), ct.affine, ct.header), masks / (name + '.nii.gz'))
            plans = root / 'plans'
            plans.mkdir()
            repo = Path(__file__).resolve().parents[1]
            command = [sys.executable, str(repo / 'main.py'), '--input_folder', str(source.parent),
                       '--output_folder', str(root / 'output'), '--component_plan_root', str(plans),
                       '--vertebrae_engine', 'shapekit_components', '--vertebrae_only',
                       '--cpu_count', '1', '--log_folder', str(root / 'logs')]
            failed = subprocess.run(command, cwd=str(repo), capture_output=True, text=True, timeout=120)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Failed cases: sample', failed.stdout)
            self.assertFalse((root / 'output' / 'sample').exists())
            plan = {'schema_version': 1, 'mode': 'cleanup_only', 'source_sha256': adapter.fingerprint(raw, ct.affine)}
            (plans / 'sample.json').write_text(json.dumps(plan))
            passed = subprocess.run(command, cwd=str(repo), capture_output=True, text=True, timeout=120)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertTrue((root / 'output/sample/vertebrae_components_report.json').is_file())
            shutil.copytree(source, source.parent / 'second')
            (plans / 'second.json').write_text(json.dumps(plan))
            command[command.index('--output_folder') + 1] = str(root / 'parallel')
            command[command.index('--cpu_count') + 1] = '2'
            parallel = subprocess.run(command, cwd=str(repo), capture_output=True, text=True, timeout=120)
            self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
            for case in ('sample', 'second'):
                self.assertTrue((root / 'parallel' / case / 'vertebrae_components_report.json').is_file())


if __name__ == '__main__':
    unittest.main()
