"""Integration tests for the shapekit_anchor_graph adapter."""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import yaml

from utils import vertebrae_anchor_graph as adapter
from utils.utils import save_and_combine_segmentations

SHAPE = (40, 40, 90)
AFFINE = np.eye(4)
LOG = logging.getLogger('test_anchor_graph')
NAMES = adapter.VERTEBRA_NAMES  # 1 = vertebrae_L5 ... 24 = vertebrae_C1


def blocks(labels):
    """14x14x10 blocks stacked upwards with 2-voxel discs; the block at position i carries labels[i]."""
    label = np.zeros(SHAPE, np.uint8)
    for i, k in enumerate(labels):
        z = 2 + 12 * i
        label[13:27, 13:27, z:z + 10] = k
    return label


def spine(n=6, speck=True):
    """n stacked vertebrae plus, optionally, a far 8-voxel T11 speck."""
    label = blocks(range(1, n + 1))
    if speck:
        label[1:3, 1:3, 84:86] = 7  # T11, far from every vertebra -> removed by the engine
    return label


def as_dict(label):
    d = {name: (label == k).astype(np.uint8) for k, name in NAMES.items()}
    d['liver'] = np.zeros(SHAPE, np.uint8)
    d['liver'][30:38, 2:10, 40:50] = 1
    return d


def ref_img():
    return nib.Nifti1Image(np.zeros(SHAPE, np.uint8), AFFINE)


def run(d, ct_path=None, report_dir=None):
    return adapter.postprocessing_vertebrae_anchor_graph('case', d, ref_img(), ct_path, LOG, report_dir=report_dir)


def read_report(folder):
    with open(os.path.join(folder, adapter.REPORT_NAME)) as f:
        return json.load(f)


class AdapterTest(unittest.TestCase):

    def test_far_fragment_removed_and_other_organs_untouched(self):
        d = as_dict(spine())
        liver = d['liver'].copy()
        with tempfile.TemporaryDirectory() as tmp:
            out = run(d, report_dir=tmp)
            report = read_report(tmp)
        self.assertEqual(int(out['vertebrae_T11'].sum()), 0)
        for k in range(1, 7):
            self.assertEqual(int(out[NAMES[k]].sum()), 14 * 14 * 10, NAMES[k])
        np.testing.assert_array_equal(out['liver'], liver)
        self.assertEqual(set(NAMES.values()) - set(out), set())  # all 24 names written back
        self.assertEqual(report['status'], 'ok')
        self.assertFalse(report['ct_used'])
        self.assertIn('delete_isolated', {c['step'] for c in report['changes']})
        self.assertEqual(report['engine_sha256'], adapter.ENGINE_SHA256)

    def test_no_vertebrae_is_a_no_op(self):
        d = {'liver': np.ones(SHAPE, np.uint8)}
        self.assertEqual(set(run(d)), {'liver'})

    def test_misaligned_ct_falls_back_to_mask_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct = os.path.join(tmp, 'ct.nii.gz')
            nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.int16), AFFINE), ct)
            with self.assertLogs(LOG, 'WARNING'):
                run(as_dict(spine()), ct_path=ct, report_dir=tmp)
            report = read_report(tmp)
        self.assertEqual(report['status'], 'ok')
        self.assertFalse(report['ct_used'])

    def test_aligned_ct_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct = os.path.join(tmp, 'ct.nii.gz')
            hu = np.where(spine(speck=False) > 0, 400, -100).astype(np.int16)
            nib.save(nib.Nifti1Image(hu, AFFINE), ct)
            run(as_dict(spine()), ct_path=ct, report_dir=tmp)
            report = read_report(tmp)
        self.assertEqual(report['status'], 'ok')
        self.assertTrue(report['ct_used'])

    def test_4d_ct_with_singleton_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct = os.path.join(tmp, 'ct.nii.gz')
            hu = np.where(spine(speck=False) > 0, 400, -100).astype(np.int16)[..., None]  # (x, y, z, 1)
            nib.save(nib.Nifti1Image(hu, AFFINE), ct)
            out = run(as_dict(spine()), ct_path=ct, report_dir=tmp)
            report = read_report(tmp)
        self.assertEqual(report['status'], 'ok')
        self.assertTrue(report['ct_used'])
        self.assertEqual(int(out['vertebrae_T11'].sum()), 0)  # cleanup ran: far speck removed

    def test_unreadable_ct_falls_back_to_mask_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ct = os.path.join(tmp, 'ct.nii.gz')
            with open(ct, 'wb') as f:
                f.write(b'not a nifti file')
            with self.assertLogs(LOG, 'WARNING'):
                out = run(as_dict(spine()), ct_path=ct, report_dir=tmp)
            report = read_report(tmp)
        self.assertEqual(report['status'], 'ok')
        self.assertFalse(report['ct_used'])
        self.assertEqual(int(out['vertebrae_T11'].sum()), 0)

    def test_engine_failure_keeps_input(self):
        d = as_dict(spine())
        before = {k: v.copy() for k, v in d.items()}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(adapter.engine, 'postprocess_array', side_effect=RuntimeError('boom')), \
                self.assertLogs(LOG, 'ERROR'):
            out = run(d, report_dir=tmp)
            report = read_report(tmp)
        for k, v in before.items():
            np.testing.assert_array_equal(out[k], v, k)
        self.assertEqual(report['status'], 'engine_failed')

    def test_matches_standalone_script(self):
        label = spine()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'pred', 'c1'))
            nib.save(nib.Nifti1Image(label, AFFINE), os.path.join(tmp, 'pred', 'c1', 'combined_labels.nii.gz'))
            p = adapter.engine.PARAMS
            args = argparse.Namespace(pred_root=os.path.join(tmp, 'pred'), save_root=os.path.join(tmp, 'save'),
                                      ct_root=None, near_mm=p['near_mm'], min_anchor_ml=p['min_anchor_ml'],
                                      merge_contact_ratio=p['merge_contact_ratio'], max_merges=p['max_merges'],
                                      no_merge=False, no_fill=False, disc_score=p['disc_score'],
                                      max_ct_relabels=p['max_ct_relabels'])
            adapter.engine.process_case('c1', args)
            standalone = np.asanyarray(nib.load(os.path.join(tmp, 'save', 'c1', 'combined_labels.nii.gz')).dataobj)
        out = run(as_dict(label))
        integrated = np.zeros(SHAPE, np.uint8)
        for k, name in NAMES.items():
            integrated[out[name] > 0] = k
        np.testing.assert_array_equal(integrated, standalone)


def engine_run(label, zooms=(1.0, 1.0, 1.0), ct=None):
    return adapter.engine.postprocess_array(label.copy(), np.diag([*zooms, 1.0]), zooms, ct=ct)


def steps(report, step):
    return [(c['from'], c['to'], c['n_voxels']) for c in report['changes'] if c['step'] == step]


def rim_case():
    """Six vertebrae on a CT where L4 has a one-voxel bone rim the label map missed."""
    label = blocks(range(1, 7))
    hu = np.where(label > 0, 400, -100).astype(np.int16)
    shell = (slice(12, 28), slice(12, 28), slice(13, 25))  # L4 grown by one voxel on every side
    hu[shell][label[shell] != 2] = 400
    return label, hu


class EngineTest(unittest.TestCase):
    """The engine's own correction steps on synthetic label maps (no file I/O)."""

    def test_touching_wrong_label_fragment_is_reassigned(self):
        label = blocks(range(1, 7))
        label[27:31, 15:19, 28:32] = 9  # 64-voxel T9 piece stuck to the side of L3
        out, report = engine_run(label)
        self.assertEqual(int((out == 9).sum()), 0)
        self.assertEqual(int((out == 3).sum()), 14 * 14 * 10 + 64)
        self.assertEqual(steps(report, 'reassign'), [('T9', 'L3', 64)])

    def test_split_vertebra_is_merged_and_gap_flagged(self):
        label = blocks([1, 2, 3, 5, 6, 7])  # the bone at position 2 is L3 below and L2 above
        label[13:27, 13:27, 31:36] = 4
        out, report = engine_run(label, zooms=(1.5, 1.5, 1.5))  # 1.5 mm voxels: each half is >= 1 mL
        self.assertEqual(int((out == 4).sum()), 0)
        self.assertEqual(int((out == 3).sum()), 14 * 14 * 10)
        self.assertEqual([(c['from'], c['to']) for c in report['changes'] if c['step'] == 'merge'], [('L2', 'L3')])
        self.assertIn(['L2'], [f['labels'] for f in report['flags'] if f['type'] == 'missing_label'])

    def test_bone_rim_is_grown_from_ct(self):
        label, hu = rim_case()
        out, report = engine_run(label, ct=hu)
        self.assertEqual(steps(report, 'grow_rim'), [('background', 'L4', 16 * 16 * 12 - 14 * 14 * 10)])
        self.assertEqual(int((out == 2).sum()), 16 * 16 * 12)
        np.testing.assert_array_equal(out[out != 2], label[out != 2])  # nothing else changed

    def test_second_pass_changes_nothing(self):
        label, hu = rim_case()
        first, _ = engine_run(label, ct=hu)
        second, report = engine_run(first, ct=hu)
        self.assertEqual(report['voxels_changed'], 0)
        np.testing.assert_array_equal(second, first)


class SaverTest(unittest.TestCase):
    class_map = {1: 'liver', 31: 'vertebrae_T12', 32: 'vertebrae_T11'}

    def _save(self, tmp, write_empty):
        seg = os.path.join(tmp, 'segmentations')
        os.makedirs(seg)
        raw = np.zeros(SHAPE, np.uint8)
        raw[1:3, 1:3, 1:3] = 1
        nib.save(nib.Nifti1Image(raw, AFFINE), os.path.join(seg, 'vertebrae_T11.nii.gz'))  # main()'s input copy
        d = {'liver': np.ones(SHAPE, np.uint8), 'vertebrae_T11': np.zeros(SHAPE, np.uint8)}
        save_and_combine_segmentations(d, self.class_map, ref_img(), tmp, if_save_combined=True,
                                       write_empty=write_empty)
        return int(np.asanyarray(nib.load(os.path.join(seg, 'vertebrae_T11.nii.gz')).dataobj).sum())

    def test_write_empty_overwrites_input_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._save(tmp, {'vertebrae_T11'}), 0)
            combined = np.asanyarray(nib.load(os.path.join(tmp, 'combined_labels.nii.gz')).dataobj)
            self.assertFalse(np.any(combined == 32))

    def test_default_behaviour_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._save(tmp, ()), 8)  # empty masks are still skipped by default


REPO = Path(__file__).resolve().parents[1]


class MainSmokeTest(unittest.TestCase):
    """main.py end to end on a tiny case: engine dispatch, report_dir and write_empty wiring."""

    def test_main_runs_anchor_graph_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(REPO / 'config.yaml') as f:
                config = yaml.safe_load(f)
            config['affine_reference_file_name'] = 'vertebrae_L1.nii.gz'  # the tiny case has no liver
            with open(os.path.join(tmp, 'config.yaml'), 'w') as f:  # main.py reads config.yaml from its cwd
                yaml.safe_dump(config, f)
            seg = os.path.join(tmp, 'input', 'case1', 'segmentations')
            os.makedirs(seg)
            label = spine()
            for k, name in NAMES.items():
                if (label == k).any():
                    nib.save(nib.Nifti1Image((label == k).astype(np.uint8), AFFINE), os.path.join(seg, name + '.nii.gz'))
            out, logs = os.path.join(tmp, 'output'), os.path.join(tmp, 'logs')
            proc = subprocess.run(
                [sys.executable, str(REPO / 'main.py'), '--input_folder', os.path.join(tmp, 'input'),
                 '--output_folder', out, '--log_folder', logs, '--vertebrae_only',
                 '--vertebrae_engine', 'shapekit_anchor_graph', '--cpu_count', '1'],
                cwd=tmp, capture_output=True, text=True, timeout=300)
            with open(os.path.join(logs, 'postprocessing.log')) as f:
                post_log = f.read()
            case = os.path.join(out, 'case1')
            report = read_report(case) if os.path.exists(os.path.join(case, adapter.REPORT_NAME)) else {}
            t11 = np.asanyarray(nib.load(os.path.join(case, 'segmentations', 'vertebrae_T11.nii.gz')).dataobj)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertNotIn('Traceback', proc.stderr, proc.stderr[-2000:])
        self.assertIn('Successfully processed case1', post_log)
        self.assertEqual(report.get('status'), 'ok')
        self.assertFalse(t11.any())  # speck removed and the input copy of the T11 file overwritten


if __name__ == '__main__':
    unittest.main()
