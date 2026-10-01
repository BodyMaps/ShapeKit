"""Small, deterministic contract tests for the native-grid Hao adapter."""

import hashlib
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from utils import vertebrae_hao as adapter
from utils import vertebrae_hao_engine as engine


SOURCE_SHA256 = "9213f51295c91b3fa4b8315af3640b709bfa89430e59047bbff0b98d924a8cc9"
SHAPE = (6, 7, 8)
LOGGER = logging.getLogger(__name__)


def _save(path, data, affine=None, zooms=None, slope=None, intercept=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag([-1.25, 1.5, 2.0, 1.0])
        affine[:3, 3] = [31.0, -42.0, 53.0]
    image = nib.Nifti1Image(data, affine)
    if zooms is not None:
        image.header.set_zooms(zooms)
    if slope is not None:
        image.header.set_slope_inter(slope, intercept)
    nib.save(image, str(path))
    return nib.load(str(path))


def _mask(voxel=(1, 2, 3), shape=SHAPE):
    data = np.zeros(shape, dtype=np.uint8)
    data[voxel] = 1
    return data


def _mask_path(case, label=1, subfolder="segmentations"):
    return case / subfolder / (engine.VERTEBRA_NAMES[label - 1] + ".nii.gz")


def _fake_engine(monkeypatch, *, baseline=None, final=None, observations=None):
    """Replace only the expensive numerical stages, retaining adapter checks."""
    calls = {} if observations is None else observations

    def run_v2(raw, spacing):
        calls["raw"] = raw.copy()
        calls["spacing"] = np.asarray(spacing).copy()
        return raw.copy() if baseline is None else baseline.copy()

    def refine(raw, v2, ct, affine, cfg, progress=None):
        calls["baseline"] = v2.copy()
        calls["ct"] = ct.copy()
        calls["affine"] = affine.copy()
        calls["cfg"] = cfg
        calls["progress"] = progress
        result = v2.copy() if final is None else final.copy()
        return result, {"status": "skipped", "reason": "test_double"}, {}

    monkeypatch.setattr(engine, "run_v2", run_v2)
    monkeypatch.setattr(engine, "refine", refine)
    return calls


def test_vendored_engine_is_the_exact_validated_source():
    assert hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest() == SOURCE_SHA256


def test_load_missing_masks_and_native_orientation(tmp_path):
    case = tmp_path / "case"
    first = _save(_mask_path(case, 1), _mask((1, 2, 3)))
    _save(_mask_path(case, 24), _mask((4, 5, 6)))

    reference, raw, missing = adapter.load_vertebrae(case)

    expected = np.zeros(SHAPE, dtype=np.uint8)
    expected[1, 2, 3] = 1
    expected[4, 5, 6] = 24
    assert raw.dtype == np.uint8
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(reference.affine, first.affine)
    assert nib.aff2axcodes(reference.affine) == ("L", "A", "S")
    assert list(missing) == engine.VERTEBRA_NAMES[1:23]


def test_load_custom_subfolder_and_present_empty_mask(tmp_path):
    case = tmp_path / "case"
    _save(_mask_path(case, 3, "masks"), np.zeros(SHAPE, dtype=np.uint8))

    reference, raw, missing = adapter.load_vertebrae(case, subfolder_name="masks")

    assert reference.shape == SHAPE
    assert not raw.any()
    assert len(missing) == 23
    assert engine.VERTEBRA_NAMES[2] not in missing


def test_load_requires_at_least_one_vertebra_file(tmp_path):
    (tmp_path / "segmentations").mkdir()
    _save(tmp_path / "segmentations" / "liver.nii.gz", _mask())
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.load_vertebrae(tmp_path)


@pytest.mark.parametrize("bad_value", [-1.0, 0.5, 2.0, 255.0, np.nan, np.inf, -np.inf])
def test_load_rejects_nonbinary_or_nonfinite_mask(tmp_path, bad_value):
    data = _mask().astype(np.float32)
    data[0, 0, 0] = bad_value
    _save(_mask_path(tmp_path), data)

    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


def test_load_checks_calibrated_not_only_stored_mask_values(tmp_path):
    _save(_mask_path(tmp_path), _mask(), slope=2.0, intercept=0.0)
    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


@pytest.mark.parametrize("shape", [(6, 7), (6, 7, 8, 1)])
def test_load_rejects_non_3d_mask(tmp_path, shape):
    _save(_mask_path(tmp_path), np.zeros(shape, dtype=np.uint8))
    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


def test_load_checks_empty_masks_for_shape_mismatch(tmp_path):
    _save(_mask_path(tmp_path, 1), _mask())
    _save(_mask_path(tmp_path, 2), np.zeros((6, 7, 9), dtype=np.uint8))
    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


@pytest.mark.parametrize("difference", [0.001, 1.0])
def test_load_rejects_affine_mismatch_with_zero_relative_tolerance(tmp_path, difference):
    affine = np.eye(4)
    affine[0, 3] = 100000.0
    _save(_mask_path(tmp_path, 1), _mask(), affine)
    changed = affine.copy()
    # The small displacement is placed at the origin; the large one would
    # incorrectly pass np.allclose's default relative tolerance at 100000 mm.
    changed[1 if difference < 1 else 0, 3] += difference
    _save(_mask_path(tmp_path, 2), _mask((4, 5, 6)), changed)
    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


def test_load_rejects_header_zooms_mismatch_even_with_equal_affines(tmp_path):
    first = _save(_mask_path(tmp_path, 1), _mask())
    second = _save(
        _mask_path(tmp_path, 2), _mask((4, 5, 6)), first.affine,
        zooms=(1.25, 1.5, 3.0),
    )
    np.testing.assert_array_equal(first.affine, second.affine)
    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


def test_load_accepts_affine_rounding_within_absolute_tolerance(tmp_path):
    first = _save(_mask_path(tmp_path, 1), _mask())
    changed = first.affine.copy()
    changed[0, 3] += 0.00005
    _save(_mask_path(tmp_path, 2), _mask((4, 5, 6)), changed)
    _, raw, _ = adapter.load_vertebrae(tmp_path)
    assert raw[1, 2, 3] == 1
    assert raw[4, 5, 6] == 2


def test_load_rejects_overlapping_vertebrae(tmp_path):
    _save(_mask_path(tmp_path, 1), _mask())
    _save(_mask_path(tmp_path, 2), _mask())
    with pytest.raises(ValueError):
        adapter.load_vertebrae(tmp_path)


def test_refine_uses_calibrated_float32_ct_and_header_spacing(tmp_path, monkeypatch):
    raw = _mask() * 4
    reference = _save(tmp_path / "reference.nii.gz", raw)
    stored_ct = np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE) * 17
    ct_path = tmp_path / "ct.nii.gz"
    _save(ct_path, stored_ct, reference.affine, slope=2.5, intercept=-2000.0)
    observations = _fake_engine(monkeypatch)

    final, report = adapter.refine_labels(raw, reference, ct_path, LOGGER)

    assert observations["ct"].dtype == np.float32
    np.testing.assert_array_equal(observations["ct"], stored_ct.astype(np.float32) * 2.5 - 2000.0)
    np.testing.assert_array_equal(observations["raw"], raw)
    np.testing.assert_array_equal(observations["affine"], reference.affine)
    np.testing.assert_array_equal(observations["spacing"], reference.header.get_zooms()[:3])
    assert isinstance(observations["cfg"], engine.Config)
    assert callable(observations["progress"])
    np.testing.assert_array_equal(final, raw)
    assert report["reason"] == "test_double"


@pytest.mark.parametrize("mismatch", ["shape", "affine", "relative_affine", "4d"])
def test_refine_rejects_misaligned_ct_before_engine_runs(tmp_path, monkeypatch, mismatch):
    raw = _mask() * 4
    affine = np.eye(4)
    affine[0, 3] = 100000.0
    reference = _save(tmp_path / "reference.nii.gz", raw, affine)
    ct_shape = SHAPE
    ct_affine = reference.affine.copy()
    if mismatch == "shape":
        ct_shape = (6, 7, 9)
    elif mismatch == "4d":
        ct_shape = (*SHAPE, 1)
    elif mismatch == "relative_affine":
        ct_affine[0, 3] += 1.0
    else:
        ct_affine[1, 3] += 0.001
    ct_path = tmp_path / "ct.nii.gz"
    _save(ct_path, np.zeros(ct_shape, dtype=np.int16), ct_affine)

    def must_not_run(*args, **kwargs):
        pytest.fail("Engine called before CT geometry was validated")

    monkeypatch.setattr(engine, "run_v2", must_not_run)
    monkeypatch.setattr(engine, "refine", must_not_run)
    with pytest.raises(ValueError):
        adapter.refine_labels(raw, reference, ct_path, LOGGER)


def test_refine_rejects_missing_ct_without_fallback(tmp_path, monkeypatch):
    raw = _mask() * 4
    reference = _save(tmp_path / "reference.nii.gz", raw)

    def must_not_run(*args, **kwargs):
        pytest.fail("Missing CT must fail, not invoke a fallback engine")

    monkeypatch.setattr(engine, "run_v2", must_not_run)
    with pytest.raises((FileNotFoundError, ValueError)):
        adapter.refine_labels(raw, reference, tmp_path / "missing.nii.gz", LOGGER)


@pytest.mark.parametrize("protected_label", [1, 2, 3, 14, 24])
def test_refine_rejects_protected_mask_change(tmp_path, monkeypatch, protected_label):
    raw = _mask() * protected_label
    reference = _save(tmp_path / "reference.nii.gz", raw)
    ct_path = tmp_path / "ct.nii.gz"
    _save(ct_path, np.zeros(SHAPE, dtype=np.int16), reference.affine)
    final = np.zeros_like(raw)
    final[4, 5, 6] = protected_label  # Same count does not mean same mask.
    _fake_engine(monkeypatch, final=final)
    with pytest.raises((AssertionError, ValueError)):
        adapter.refine_labels(raw, reference, ct_path, LOGGER)


def test_refine_rejects_disappearance_of_mutable_label(tmp_path, monkeypatch):
    raw = _mask() * 4
    raw[4, 5, 6] = 5
    reference = _save(tmp_path / "reference.nii.gz", raw)
    ct_path = tmp_path / "ct.nii.gz"
    _save(ct_path, np.zeros(SHAPE, dtype=np.int16), reference.affine)
    final = raw.copy()
    final[final == 4] = 5
    _fake_engine(monkeypatch, final=final)
    with pytest.raises((AssertionError, ValueError)):
        adapter.refine_labels(raw, reference, ct_path, LOGGER)


def test_protected_masks_are_compared_to_v2_not_original_prediction(tmp_path, monkeypatch):
    raw = _mask() * 1
    raw[4, 5, 6] = 1
    baseline = raw.copy()
    baseline[4, 5, 6] = 0
    reference = _save(tmp_path / "reference.nii.gz", raw)
    ct_path = tmp_path / "ct.nii.gz"
    _save(ct_path, np.zeros(SHAPE, dtype=np.int16), reference.affine)
    _fake_engine(monkeypatch, baseline=baseline)

    final, _ = adapter.refine_labels(raw, reference, ct_path, LOGGER)

    np.testing.assert_array_equal(final, baseline)


def test_real_engine_abstains_with_missing_external_anchor(tmp_path):
    raw = np.zeros(SHAPE, dtype=np.uint8)
    raw[1:5, 1:5, 1:5] = 4
    reference = _save(tmp_path / "reference.nii.gz", raw)
    ct_path = tmp_path / "ct.nii.gz"
    _save(ct_path, np.full(SHAPE, 300, dtype=np.int16), reference.affine)

    final, report = adapter.refine_labels(raw, reference, ct_path, LOGGER)

    np.testing.assert_array_equal(final, raw)
    assert report["status"] == "skipped"
    assert report["reason"] == "missing_external_anchor"
    assert report["changed_voxels"] == 0


def _case_fixture(tmp_path, monkeypatch, subfolder="segmentations", class_id_start=26):
    case = tmp_path / "inputs" / "case"
    first_data = _mask((1, 2, 3))
    first_path = _mask_path(case, 1, subfolder)
    initial = _save(first_path, first_data)
    # Non-default form codes and header metadata must survive new-mask export.
    first = nib.Nifti1Image(first_data, initial.affine)
    first.set_qform(initial.affine, code=1)
    first.set_sform(initial.affine, code=4)
    first.header.set_xyzt_units("mm")
    first.header["descrip"] = b"native-grid header"
    nib.save(first, str(first_path))
    reference = nib.load(str(first_path))
    _save(_mask_path(case, 4, subfolder), _mask((2, 3, 4)), reference.affine)
    _save(_mask_path(case, 24, subfolder), _mask((4, 5, 6)), reference.affine)
    # The known organ intentionally overlaps one vertebra voxel. Its file
    # remains untouched, while the combined map follows class-ID priority.
    organ = _mask((1, 2, 3))
    organ[0, 1, 2] = 1
    _save(case / subfolder / "liver.nii.gz", organ, reference.affine)
    _save(case / subfolder / "research_aux.nii.gz", _mask((5, 6, 7)), reference.affine)
    ct_path = tmp_path / "cts" / "case" / "ct.nii.gz"
    _save(ct_path, np.zeros(SHAPE, dtype=np.int16), reference.affine)
    output = tmp_path / "outputs" / "case"
    mapping = {i + class_id_start: name for i, name in enumerate(engine.VERTEBRA_NAMES)}
    mapping[5] = "liver"
    _fake_engine(monkeypatch)
    return case, output, ct_path, mapping, reference


@pytest.mark.parametrize("subfolder", ["segmentations", "masks"])
@pytest.mark.parametrize("class_id_start", [26, 301])
def test_process_exports_all_masks_native_header_combined_mapping_and_report(
    tmp_path, monkeypatch, subfolder, class_id_start,
):
    case, output, ct_path, mapping, reference = _case_fixture(
        tmp_path, monkeypatch, subfolder, class_id_start,
    )
    input_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in case.rglob("*") if p.is_file()}
    ct_hash = hashlib.sha256(ct_path.read_bytes()).hexdigest()

    report = adapter.process_case(case, output, ct_path, mapping, subfolder_name=subfolder)

    assert report["integration_status"] == "complete"
    assert report["engine_sha256"] == SOURCE_SHA256
    assert json.loads((output / "vertebrae_hao_report.json").read_text(encoding="utf-8")) == report
    assert report["dsc"] is None
    assert len(report["missing_input_masks"]) == 21
    assert report["label_map"] == {str(i + 1): i + class_id_start for i in range(24)}
    expected_labels = np.zeros(SHAPE, dtype=np.uint8)
    expected_labels[1, 2, 3] = 1
    expected_labels[2, 3, 4] = 4
    expected_labels[4, 5, 6] = 24
    for label, name in enumerate(engine.VERTEBRA_NAMES, 1):
        image = nib.load(str(output / subfolder / f"{name}.nii.gz"))
        assert image.get_data_dtype() == np.dtype(np.uint8)
        np.testing.assert_array_equal(np.asanyarray(image.dataobj), expected_labels == label)
        np.testing.assert_array_equal(image.affine, reference.affine)
        np.testing.assert_array_equal(image.get_qform(), reference.get_qform())
        np.testing.assert_array_equal(image.get_sform(), reference.get_sform())
        assert image.header["qform_code"] == reference.header["qform_code"]
        assert image.header["sform_code"] == reference.header["sform_code"]
        assert image.header["descrip"] == reference.header["descrip"]
        assert image.header.get_xyzt_units() == reference.header.get_xyzt_units()
        assert image.header.get_zooms() == reference.header.get_zooms()
    for unrelated in ("liver.nii.gz", "research_aux.nii.gz"):
        assert (output / subfolder / unrelated).read_bytes() == (case / subfolder / unrelated).read_bytes()
    combined = nib.load(str(output / "combined_labels.nii.gz"))
    expected_combined = np.zeros(SHAPE, dtype=np.uint16)
    expected_combined[0, 1, 2] = 5
    expected_combined[1, 2, 3] = class_id_start
    expected_combined[2, 3, 4] = class_id_start + 3
    expected_combined[4, 5, 6] = class_id_start + 23
    np.testing.assert_array_equal(np.asanyarray(combined.dataobj), expected_combined)
    np.testing.assert_array_equal(combined.affine, reference.affine)
    assert combined.get_data_dtype() == np.dtype(np.uint8 if class_id_start == 26 else np.uint16)
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_hashes} == input_hashes
    assert hashlib.sha256(ct_path.read_bytes()).hexdigest() == ct_hash


def test_process_refuses_existing_output_without_modifying_it(tmp_path, monkeypatch):
    case, output, ct_path, mapping, _ = _case_fixture(tmp_path, monkeypatch)
    output.mkdir(parents=True)
    marker = output / "existing.txt"
    marker.write_text("existing result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        adapter.process_case(case, output, ct_path, mapping)
    assert marker.read_text(encoding="utf-8") == "existing result"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("relation", ["same", "input_child", "input_parent", "ct_parent"])
def test_process_requires_output_disjoint_from_input_and_ct(tmp_path, monkeypatch, relation):
    case, output, ct_path, mapping, _ = _case_fixture(tmp_path, monkeypatch)
    output = {
        "same": case,
        "input_child": case / "postprocessed",
        "input_parent": case.parent,
        "ct_parent": ct_path.parent,
    }[relation]
    with pytest.raises(ValueError):
        adapter.process_case(case, output, ct_path, mapping)


def test_process_does_not_publish_partial_result_when_export_fails(tmp_path, monkeypatch):
    case, output, ct_path, mapping, _ = _case_fixture(tmp_path, monkeypatch)
    original_save = engine.save_volume
    calls = []

    def fail_second_save(data, image, path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated export failure")
        original_save(data, image, path)

    monkeypatch.setattr(engine, "save_volume", fail_second_save)
    with pytest.raises(OSError, match="simulated export failure"):
        adapter.process_case(case, output, ct_path, mapping)
    assert not output.exists()
    incomplete = list((output.parent / ".incomplete").iterdir())
    assert len(incomplete) == 1
    assert not (incomplete[0] / "vertebrae_hao_report.json").exists()


def test_process_writes_report_only_after_all_volumes(tmp_path, monkeypatch):
    case, output, ct_path, mapping, _ = _case_fixture(tmp_path, monkeypatch)
    original_save = engine.save_volume
    saves = []

    def record_save(data, image, path):
        path = Path(path)
        staging = path.parent.parent if path.parent.name == "segmentations" else path.parent
        assert not (staging / "vertebrae_hao_report.json").exists()
        assert not output.exists()
        saves.append(path.name)
        original_save(data, image, path)

    monkeypatch.setattr(engine, "save_volume", record_save)
    adapter.process_case(case, output, ct_path, mapping)
    assert len(saves) == 25
    assert saves[-1] == "combined_labels.nii.gz"
    assert (output / "vertebrae_hao_report.json").is_file()
