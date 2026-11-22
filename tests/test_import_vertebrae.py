"""IO contracts for importing existing predictions, independent of any engine."""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "import_vertebrae",
    Path(__file__).resolve().parents[1] / "tools/import_vertebrae.py",
)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)
CLASS_MAP = {i + 26: name for i, name in enumerate(bridge.NAMES)} | {5: "liver"}
AFFINE = np.diag([-1.25, 1.5, 2.5, 1.0])


def save(path, data, affine=AFFINE):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 2)
    nib.save(image, path)


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "input/case"
    raw = np.zeros((8, 9, 10), np.uint8)
    raw[1:3, 2:4, 3:5] = 1
    save(source / "combined_labels.nii.gz", raw)
    save(source / "segmentations/vertebrae_L5.nii.gz", (raw == 1).astype(np.uint8))
    return source, tmp_path / "output/case", raw


def test_preserves_binary_and_maps_combined_with_metadata(case):
    source, output, raw = case
    report = bridge.import_case(source, output, CLASS_MAP)
    image = nib.load(output / "combined_labels.nii.gz")
    assert np.array_equal(np.asanyarray(image.dataobj), (raw > 0) * 26)
    assert image.get_data_dtype() == np.dtype("uint8")
    assert np.array_equal(image.affine, AFFINE)
    assert int(image.header["qform_code"]) == 1
    assert int(image.header["sform_code"]) == 2
    assert len(report["missing_empty_classes"]) == 23
    assert len(list((output / "segmentations").glob("*.nii.gz"))) == 24
    assert bridge.sha256(source / "segmentations/vertebrae_L5.nii.gz") == bridge.sha256(
        output / "segmentations/vertebrae_L5.nii.gz"
    )
    assert report["dsc"] is None


def test_no_overwrite_and_verified_resume(case):
    source, output, _ = case
    bridge.import_case(source, output, CLASS_MAP)
    with pytest.raises(FileExistsError, match="E_EXISTS"):
        bridge.import_case(source, output, CLASS_MAP)
    assert bridge.import_case(source, output, CLASS_MAP, resume=True)["resumed"]
    (output / "segmentations/vertebrae_L5.nii.gz").write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="E_RESUME"):
        bridge.import_case(source, output, CLASS_MAP, resume=True)


def test_missing_populated_mask(case):
    source, output, raw = case
    raw[5, 5, 5] = 2
    save(source / "combined_labels.nii.gz", raw)
    with pytest.raises(ValueError, match="E_MISSING"):
        bridge.import_case(source, output, CLASS_MAP)
    assert not output.exists()


@pytest.mark.parametrize(
    "fault", ["affine", "shape", "binary", "consistency", "transforms", "unknown"]
)
def test_invalid_masks_fail_without_publishing(case, fault):
    source, output, raw = case
    path = source / "segmentations/vertebrae_L5.nii.gz"
    if fault == "affine":
        affine = AFFINE.copy()
        affine[0, 3] = 10
        save(path, (raw > 0).astype(np.uint8), affine)
    elif fault == "shape":
        save(path, np.zeros((3, 4, 5), np.uint8))
    elif fault == "binary":
        save(path, (raw > 0).astype(np.float32) * np.nan)
    elif fault == "consistency":
        save(path, np.zeros_like(raw))
    elif fault == "transforms":
        image = nib.load(path)
        q = AFFINE.copy()
        q[1, 3] = 100
        image.set_qform(q, 1)
        nib.save(image, path)
    else:
        save(source / "segmentations/vertebrae_L6.nii.gz", np.zeros_like(raw))
    with pytest.raises(ValueError, match="E_"):
        bridge.import_case(source, output, CLASS_MAP)
    assert not output.exists()


def test_unrelated_binary_copied_exactly_and_overlap_rejected(case):
    source, output, raw = case
    extra = np.zeros_like(raw)
    extra[6, 6, 6] = 1
    path = source / "segmentations/liver.nii.gz"
    save(path, extra)
    bridge.import_case(source, output, CLASS_MAP)
    assert bridge.sha256(path) == bridge.sha256(output / "segmentations/liver.nii.gz")
    assert (
        np.asanyarray(nib.load(output / "combined_labels.nii.gz").dataobj)[6, 6, 6] == 5
    )
    extra[1, 2, 3] = 1
    save(path, extra)
    with pytest.raises(ValueError, match="E_OVERLAP"):
        bridge.import_case(source, output.parent / "conflict", CLASS_MAP)


def test_ct_mismatch(case, tmp_path):
    source, output, raw = case
    ct = tmp_path / "ct.nii.gz"
    save(ct, np.zeros_like(raw), np.eye(4))
    with pytest.raises(ValueError, match="E_GEOMETRY"):
        bridge.import_case(source, output, CLASS_MAP, ct_path=ct)


@pytest.mark.parametrize("offset", [0, 25])
def test_mapping_offsets_and_uint16_export(case, offset):
    source, output, raw = case
    shifted = np.where(raw, raw + offset, 0).astype(np.uint8)
    save(source / "combined_labels.nii.gz", shifted)
    cmap = {i + 1000: name for i, name in enumerate(bridge.NAMES)}
    bridge.import_case(source, output, cmap, label_offset=offset)
    image = nib.load(output / "combined_labels.nii.gz")
    assert image.get_data_dtype() == np.dtype("uint16")
    assert np.max(np.asanyarray(image.dataobj)) == 1000


@pytest.mark.parametrize("mode", ["inside", "parent", "same"])
def test_source_preserved_by_path_guard(case, mode):
    source, _, _ = case
    output = {"inside": source / "out", "parent": source.parent, "same": source}[mode]
    with pytest.raises(ValueError, match="E_PATH"):
        bridge.import_case(source, output, CLASS_MAP)


def test_batch_continues_after_failure_and_resume(case):
    source, output, _ = case
    bad = source.parent / "bad"
    bad.mkdir()
    config = source.parent.parent / "config.yaml"
    config.write_text(__import__("yaml").safe_dump({"class_map": CLASS_MAP}))
    args = [
        "--input-root",
        str(source.parent),
        "--output-root",
        str(output.parent),
        "--config",
        str(config),
    ]
    assert bridge.main(args) == 1
    assert (
        json.loads((output / "import_report.json").read_text())["status"] == "complete"
    )
    assert bridge.main(args + ["--resume"]) == 1


def test_interrupted_write_never_publishes(case, monkeypatch):
    source, output, _ = case

    def fail(*args, **kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(bridge.shutil, "copy2", fail)
    with pytest.raises(OSError, match="interruption"):
        bridge.import_case(source, output, CLASS_MAP)
    assert not output.exists()
    assert list((output.parent / ".incomplete").iterdir())


def test_clean_case_and_empty_prediction_are_unchanged(case):
    source, output, raw = case
    raw[:] = 0
    save(source / "combined_labels.nii.gz", raw)
    save(source / "segmentations/vertebrae_L5.nii.gz", raw)
    bridge.import_case(source, output, CLASS_MAP)
    assert not np.any(
        np.asanyarray(nib.load(output / "combined_labels.nii.gz").dataobj)
    )


def test_reimport_is_voxel_idempotent(case):
    source, output, _ = case
    bridge.import_case(source, output, CLASS_MAP)
    second = output.parent / "second"
    bridge.import_case(output, second, CLASS_MAP, label_offset=25)
    assert np.array_equal(
        np.asanyarray(nib.load(output / "combined_labels.nii.gz").dataobj),
        np.asanyarray(nib.load(second / "combined_labels.nii.gz").dataobj),
    )
