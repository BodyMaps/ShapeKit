import logging
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

import main as pipeline
from utils.organs_postprocessing import (
    post_processing_liver,
    post_processing_lung,
    post_processing_pancreas,
    post_processing_spleen,
    post_processing_stomach,
)
from utils.utils import reassign_left_right_based_on_liver


def _make_case(root: Path, case_id='case_001'):
    case = root / case_id
    seg_dir = case / 'segmentations'
    seg_dir.mkdir(parents=True)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(seg_dir / 'liver.nii.gz'))
    (case / 'metadata.txt').write_text('preserve me')
    return case


@pytest.fixture
def minimal_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, 'class_map', {1: 'liver'})
    monkeypatch.setattr(pipeline, 'organ_list', ['liver'])
    monkeypatch.setattr(pipeline, 'target_organs', set())
    monkeypatch.setattr(pipeline, 'save_combined_label_bool', True)
    monkeypatch.setattr(
        pipeline,
        'process_organs',
        lambda segmentation_dict, *args, **kwargs: segmentation_dict,
    )


def test_successful_case_replaces_stale_output_transactionally(tmp_path, minimal_pipeline):
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    case = _make_case(input_root)
    stale = output_root / case.name / 'segmentations'
    stale.mkdir(parents=True)
    (stale / 'stale.nii.gz').write_text('stale')

    pipeline.main(str(case), case.name, str(output_root))

    result = output_root / case.name
    assert (result / pipeline.COMPLETION_MARKER).is_file()
    assert (result / 'segmentations' / 'liver.nii.gz').is_file()
    assert not (result / 'segmentations' / 'stale.nii.gz').exists()
    assert (result / 'metadata.txt').read_text() == 'preserve me'
    assert not list(output_root.glob(f'.{case.name}.tmp-*'))
    assert not list(output_root.glob(f'.{case.name}.backup-*'))


def test_failed_case_preserves_previous_output(tmp_path, minimal_pipeline, monkeypatch):
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    case = _make_case(input_root)
    old_result = output_root / case.name
    old_result.mkdir(parents=True)
    sentinel = old_result / 'previous-result.txt'
    sentinel.write_text('still valid')

    def fail_processing(*args, **kwargs):
        raise RuntimeError('deliberate failure')

    monkeypatch.setattr(pipeline, 'process_organs', fail_processing)
    with pytest.raises(RuntimeError, match='deliberate failure'):
        pipeline.main(str(case), case.name, str(output_root))

    assert sentinel.read_text() == 'still valid'
    assert not list(output_root.glob(f'.{case.name}.tmp-*'))


def test_commit_failure_restores_previous_output(tmp_path, monkeypatch):
    final_path = tmp_path / 'case_001'
    staging_path = tmp_path / '.case_001.tmp-test'
    final_path.mkdir()
    staging_path.mkdir()
    (final_path / 'old.txt').write_text('old')
    (staging_path / 'new.txt').write_text('new')
    real_replace = pipeline.os.replace
    call_count = 0

    def fail_second_replace(source, destination):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError('deliberate commit failure')
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, 'replace', fail_second_replace)
    with pytest.raises(OSError, match='deliberate commit failure'):
        pipeline._commit_case_output(staging_path, final_path)

    assert (final_path / 'old.txt').read_text() == 'old'
    assert (staging_path / 'new.txt').read_text() == 'new'
    assert not list(tmp_path.glob('.case_001.backup-*'))


def test_resume_requires_completion_marker(tmp_path):
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    _make_case(input_root, 'complete')
    _make_case(input_root, 'partial')
    (output_root / 'complete').mkdir(parents=True)
    (output_root / 'complete' / pipeline.COMPLETION_MARKER).write_text('{}')
    partial_seg = output_root / 'partial' / 'segmentations'
    partial_seg.mkdir(parents=True)
    (partial_seg / 'liver.nii.gz').write_text('partial')

    pending = pipeline.check_unprocessed_cases(
        str(input_root), str(output_root), str(tmp_path / 'continue.csv')
    )
    assert pending == ['partial']


def test_worker_reports_failure(monkeypatch):
    monkeypatch.setattr(pipeline, 'main', lambda *args: (_ for _ in ()).throw(ValueError('bad case')))
    case_id, error = pipeline.process_case_wrapper(('case_001', 'input', 'output'))
    assert case_id == 'case_001'
    assert error == 'ValueError: bad case'


def test_missing_optional_organs_do_not_crash():
    logger = logging.getLogger('test')
    assert post_processing_liver({}) == {}
    assert post_processing_pancreas({}) == {}
    assert post_processing_stomach({}) == {}
    assert post_processing_spleen({}) == {}
    assert post_processing_lung({}, {'x': 0, 'z': 2}, None, 'case', logger) == {}


def test_missing_liver_does_not_swap_left_and_right():
    right = np.zeros((3, 3, 3), dtype=np.uint8)
    left = np.zeros_like(right)
    right[0, 0, 0] = 1
    left[2, 2, 2] = 1
    corrected_right, corrected_left = reassign_left_right_based_on_liver(
        right, left, None
    )
    np.testing.assert_array_equal(corrected_right, right)
    np.testing.assert_array_equal(corrected_left, left)


def test_worker_count_handles_empty_and_invalid_inputs():
    assert pipeline.resolve_worker_count(4, 0) == 0
    assert pipeline.resolve_worker_count(4, 1) == 1
    with pytest.raises(ValueError, match='cpu_count'):
        pipeline.resolve_worker_count(0, 1)


def test_cli_multiprocessing_smoke(tmp_path):
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    log_root = tmp_path / 'logs'
    _make_case(input_root)

    result = subprocess.run(
        [
            sys.executable,
            str(Path(pipeline.__file__)),
            '--input_folder', str(input_root),
            '--output_folder', str(output_root),
            '--log_folder', str(log_root),
            '--cpu_count', '1',
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / 'case_001' / pipeline.COMPLETION_MARKER).is_file()
    assert (output_root / 'case_001' / 'combined_labels.nii.gz').is_file()
