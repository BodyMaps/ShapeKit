"""Exercise the adapter and real CLI with tiny independent source fixtures."""

import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from tools.run_ordered_core import run_case

ALGORITHM = """
import shutil
import numpy as np
from tools.import_vertebrae import save_like as save
def get_array(img):
    return np.asanyarray(img.dataobj)
def refine_ordered_cores(img, raw, mapping):
    return raw.copy(), {"applied": False}
def process_case_cleanup(case, out, mapping):
    shutil.copytree(case, out / case.name)
    return {"changed_voxels": 0}
def save_like(ref, data, path, dtype):
    save(np.asarray(data, dtype=dtype), ref, path)
"""
from tools.import_vertebrae import NAMES

CLASS_MAP = {i: name for i, name in enumerate(NAMES, 26)}
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch(tmp_path):
    source = tmp_path / "raw"
    impl = tmp_path / "source.py"
    impl.write_text(ALGORITHM, encoding="utf-8")
    return source, tmp_path / "out", impl


def make_case(root, name, *, label=1, empty=False):
    case = root / name
    (case / "segmentations").mkdir(parents=True)
    arr = np.zeros((9, 9, 9), np.uint8)
    if not empty:
        arr[3:6, 3:6, 3:6] = label
    affine = np.diag([-1.0, 1.0, 2.5, 1.0])
    nib.save(nib.Nifti1Image(arr, affine), case / "combined_labels.nii.gz")
    nib.save(
        nib.Nifti1Image((arr != 0).astype(np.uint8), affine),
        case / "segmentations/vertebrae_L5.nii.gz",
    )
    return case, arr, affine


def test_native_cli_batch_and_verified_resume(batch, tmp_path):
    source, output, implementation = batch
    case, arr, affine = make_case(source, "case_1")
    empty, _, _ = make_case(source, "case_2", empty=True)
    cmd = [
        sys.executable,
        "main.py",
        "--input_folder",
        str(source),
        "--output_folder",
        str(output),
        "--log_folder",
        str(tmp_path / "log"),
        "--vertebrae_only",
        "--vertebrae_engine",
        "shapekit_ordered_core",
        "--ordered_core_source",
        str(implementation),
    ]
    result = subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    assert result.stdout.index("case_1") < result.stdout.index("case_2")
    combined = nib.load(output / case.name / "combined_labels.nii.gz")
    np.testing.assert_array_equal(combined.affine, affine)
    np.testing.assert_array_equal(np.asanyarray(combined.dataobj), arr * 26)
    assert (output / empty.name / "segmentations/vertebrae_C1.nii.gz").is_file()
    assert (output / case.name / "segmentations/vertebrae_L5.nii.gz").read_bytes() == (
        case / "segmentations/vertebrae_L5.nii.gz"
    ).read_bytes()
    resumed = subprocess.run(
        [*cmd, "--continue_prediction"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    assert resumed.stdout.count('"resumed": true') == 2


def test_reject_bad_inputs_before_executing_source(batch):
    source, output, impl = batch
    case, _, _ = make_case(source, "bad_id", label=99)
    impl.write_text('raise RuntimeError("must not execute")', encoding="utf-8")
    with pytest.raises(ValueError, match="E_LABELS"):
        run_case(case, output, impl, CLASS_MAP)
    assert not output.exists()


@pytest.mark.parametrize("difference", ["shape", "affine"])
def test_wrong_ct_grid(batch, tmp_path, difference):
    source, output, impl = batch
    case, raw, affine = make_case(source, "bad_ct")
    ct = tmp_path / "ct.nii.gz"
    shape = (7, 7, 7) if difference == "shape" else raw.shape
    if difference == "affine":
        affine[0, 3] += 1
    nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), affine), ct)
    with pytest.raises(ValueError, match="E_GEOMETRY"):
        run_case(case, output, impl, CLASS_MAP, ct=ct)
    assert not output.exists()


def test_partial_run_is_unpublished_and_can_retry(batch):
    source, output, impl = batch
    case, _, _ = make_case(source, "retry")
    failing = (
        ALGORITHM
        + '\ndef process_case_cleanup(*args):\n    raise RuntimeError("failed")\n'
    )
    impl.write_text(failing, encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed"):
        run_case(case, output, impl, CLASS_MAP)
    assert not output.exists()
    impl.write_text(ALGORITHM, encoding="utf-8")
    result = run_case(case, output, impl, CLASS_MAP, resume=True)
    assert result["status"] == "complete"


@pytest.mark.parametrize("changed", ["code", "mask", "output"])
def test_resume_rejects_changed_sources_and_outputs(batch, changed):
    source, output, impl = batch
    case, _, _ = make_case(source, "resume")
    run_case(case, output, impl, CLASS_MAP)
    target = {
        "code": impl,
        "mask": case / "combined_labels.nii.gz",
        "output": output / "combined_labels.nii.gz",
    }[changed]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="E_RESUME"):
        run_case(case, output, impl, CLASS_MAP, resume=True)
