import json
from pathlib import Path
import shutil
import tempfile
import unittest
import nibabel as nib
import numpy as np
from utils import vertebrae_driftfix as adapter
from utils import vertebrae_driftfix_engine as engine
from utils.vertebrae_driftfix import NAMES, process_case, shapekit_to_engine, engine_to_shapekit

DEMO = Path(__file__).resolve().parents[3] / 'data' / 'shapekit_demo'
MAPPING = {i + 26: n for i, n in enumerate(NAMES)}


def phantom_labels(shape=(40, 40, 140)):
    """A small column of 6 stacked boxes (labels 1..6, L5 at the bottom) plus a far speck and a detached piece."""
    lab = np.zeros(shape, np.uint8)
    for k in range(6):
        z = 10 + 18 * k
        lab[10:30, 10:30, z:z + 14] = k + 1
    lab[2, 2, 2] = 24  # debris far from the spine
    lab[10:13, 10:13, 60:63] = 2  # small piece of L4 sitting inside another vertebra
    return lab


def save_masks(root, lab, affine, subfolder='segmentations'):
    seg = root / subfolder
    seg.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(NAMES, 1):
        nib.save(nib.Nifti1Image((lab == i).astype(np.uint8), affine), seg / f'{name}.nii.gz')


def run_standalone(lab, affine, ct_array, root, relabel):
    """Run the engine function directly on a combined label map (what the script's CLI does)."""
    (root / 'in' / 'c').mkdir(parents=True)
    nib.save(nib.Nifti1Image(lab, affine), root / 'in' / 'c' / 'combined_labels.nii.gz')
    ct_root = None
    if ct_array is not None:
        (root / 'ct' / 'c').mkdir(parents=True)
        nib.save(nib.Nifti1Image(ct_array, affine), root / 'ct' / 'c' / 'ct.nii.gz')
        ct_root = str(root / 'ct')
    engine.process_case('c', str(root / 'in'), ct_root, str(root / 'out'), relabel=relabel)
    return np.asanyarray(nib.load(root / 'out' / 'c' / 'combined_labels.nii.gz').dataobj)


def to_shapekit(ref):
    return np.where(ref > 0, ref + 25, 0)


class LabelMapTests(unittest.TestCase):
    def test_round_trip(self):
        for engine_label in range(1, 25):
            self.assertEqual(shapekit_to_engine(engine_to_shapekit(engine_label)), engine_label)
        for shapekit_id in range(26, 50):
            self.assertEqual(engine_to_shapekit(shapekit_to_engine(shapekit_id)), shapekit_id)
        self.assertEqual(engine_to_shapekit(1), 26)
        self.assertEqual(engine_to_shapekit(24), 49)

    def test_names_order(self):
        self.assertEqual(len(NAMES), 24)
        self.assertEqual(NAMES[0], 'vertebrae_L5')
        self.assertEqual(NAMES[4], 'vertebrae_L1')
        self.assertEqual(NAMES[5], 'vertebrae_T12')
        self.assertEqual(NAMES[-1], 'vertebrae_C1')


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        self.affine = np.diag([1.5, 1.5, 2.0, 1.0])
        self.affine[:3, 3] = [-30, -20, 10]
        self.lab = phantom_labels()
        save_masks(self.source, self.lab, self.affine)
        self.ct = self.root / 'ct.nii.gz'
        self.ct_array = np.full(self.lab.shape, 300, np.int16)
        nib.save(nib.Nifti1Image(self.ct_array, self.affine), self.ct)

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, ct='default', **kw):
        ct = self.ct if ct == 'default' else ct
        return process_case(self.source, self.root / 'output', ct, MAPPING, **kw)

    def combined_out(self):
        return np.asanyarray(nib.load(self.root / 'output' / 'combined_labels.nii.gz').dataobj)

    def test_equals_standalone_phantom(self):
        for relabel in (False, True):
            with self.subTest(relabel=relabel):
                out = self.root / 'output'
                if out.exists():
                    shutil.rmtree(out)
                ref = run_standalone(self.lab, self.affine, self.ct_array, self.root / f'ref{int(relabel)}', relabel)
                self.run_case(relabel=relabel)
                got = self.combined_out()
                self.assertTrue(np.array_equal(got, to_shapekit(ref)))
                self.assertEqual(got.dtype, np.uint8)
                for i, name in enumerate(NAMES, 1):
                    m = np.asanyarray(nib.load(out / 'segmentations' / f'{name}.nii.gz').dataobj)
                    self.assertTrue(np.array_equal(m, (ref == i).astype(np.uint8)), name)

    def test_phantom_is_changed_by_engine(self):
        report = self.run_case(relabel=False)
        self.assertGreater(report['voxels_changed'], 0)
        self.assertEqual(self.combined_out()[2, 2, 2], 0)

    def test_outputs_and_report(self):
        report = self.run_case(relabel=False)
        out = self.root / 'output'
        self.assertEqual(len(list((out / 'segmentations').glob('*.nii.gz'))), 24)
        saved = json.loads((out / 'vertebrae_driftfix_report.json').read_text())
        self.assertEqual(saved['label_map']['1'], 26)
        self.assertEqual(saved['label_map']['24'], 49)
        self.assertTrue(saved['ct_used'])
        self.assertFalse(saved['relabel'])
        self.assertIn('changes', saved)
        self.assertIn('review_flags', saved)
        self.assertEqual(saved['changes'], json.loads(json.dumps(report['changes'])))
        ref = nib.load(self.source / 'segmentations' / f'{NAMES[0]}.nii.gz')
        got = nib.load(out / 'combined_labels.nii.gz')
        self.assertTrue(np.array_equal(got.affine, ref.affine))
        self.assertEqual(int(got.header['sform_code']), int(ref.header['sform_code']))

    def test_missing_ct_runs_mask_only(self):
        report = self.run_case(ct=self.root / 'nope.nii.gz', relabel=False)
        self.assertFalse(report['ct_used'])
        self.assertIsNone(report['ct'])
        self.assertIn('missing', report['inputs']['ct_status'])
        ref = run_standalone(self.lab, self.affine, None, self.root / 'ref', False)
        self.assertTrue(np.array_equal(self.combined_out(), to_shapekit(ref)))

    def test_none_ct_runs_mask_only(self):
        report = self.run_case(ct=None, relabel=False)
        self.assertFalse(report['ct_used'])

    def test_existing_output_refused(self):
        (self.root / 'output').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_case(relabel=False)

    def test_grid_mismatch_raises(self):
        other = np.zeros(self.lab.shape, np.uint8)
        other[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(other, np.diag([2.0, 2.0, 2.0, 1.0])),
                 self.source / 'segmentations' / f'{NAMES[3]}.nii.gz')
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            self.run_case(relabel=False)
        self.assertFalse((self.root / 'output').exists())

    def test_ct_grid_mismatch_raises(self):
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.int16), np.eye(4)), self.ct)
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            self.run_case(relabel=False)

    def test_overlap_and_nonbinary_rejected(self):
        dup = (self.lab == 1).astype(np.uint8)
        nib.save(nib.Nifti1Image(dup, self.affine), self.source / 'segmentations' / f'{NAMES[1]}.nii.gz')
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            self.run_case(relabel=False)
        nib.save(nib.Nifti1Image(dup * 2, self.affine), self.source / 'segmentations' / f'{NAMES[1]}.nii.gz')
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.run_case(relabel=False)

    def test_no_masks_rejected(self):
        for p in (self.source / 'segmentations').glob('*.nii.gz'):
            p.unlink()
        with self.assertRaisesRegex(ValueError, 'No vertebra masks'):
            self.run_case(relabel=False)

    def test_bad_class_map(self):
        bad = dict(MAPPING)
        bad[99] = bad.pop(26)
        with self.assertRaises(ValueError):
            process_case(self.source, self.root / 'output', self.ct, bad, relabel=False)

    def test_relabel_default_is_off(self):
        self.assertFalse(adapter.relabel_from_config(str(self.root / 'no_such_config.yaml')))
        cfg = self.root / 'c.yaml'
        cfg.write_text('vertebrae_driftfix_relabel: true\n')
        self.assertTrue(adapter.relabel_from_config(str(cfg)))
        cfg.write_text('other: 1\n')
        self.assertFalse(adapter.relabel_from_config(str(cfg)))


@unittest.skipUnless((DEMO / 'input' / 'BDMAP_00000006' / 'segmentations').is_dir()
                     and (DEMO / 'ct_root' / 'BDMAP_00000006' / 'ct.nii.gz').is_file(),
                     'cropped demo case not available')
class DemoCaseTest(unittest.TestCase):
    def test_equals_standalone_on_cropped_demo(self):
        case = 'BDMAP_00000006'
        src = DEMO / 'input' / case
        ct = DEMO / 'ct_root' / case / 'ct.nii.gz'
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ref_img, raw, _ = adapter.load_masks(src, 'segmentations')
            for relabel in (False, True):
                tag = int(relabel)
                (t / f'in{tag}' / case).mkdir(parents=True)
                nib.save(nib.Nifti1Image(raw, ref_img.affine, ref_img.header), t / f'in{tag}' / case / 'combined_labels.nii.gz')
                engine.process_case(case, str(t / f'in{tag}'), str(ct.parents[1]), str(t / f'ref{tag}'), relabel=relabel)
                expected = np.asanyarray(nib.load(t / f'ref{tag}' / case / 'combined_labels.nii.gz').dataobj)
                process_case(src, t / f'out{tag}', ct, MAPPING, relabel=relabel)
                got = np.asanyarray(nib.load(t / f'out{tag}' / 'combined_labels.nii.gz').dataobj)
                self.assertTrue(np.array_equal(got, to_shapekit(expected)))
                del got, expected


if __name__ == '__main__':
    unittest.main()
