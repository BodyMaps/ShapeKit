"""Tests for the rib-anchored vertebral identity stage.

Everything here runs on synthetic phantoms; no imaging data is required. The
phantom is a column of uniform blocks at a known pitch with rib slabs placed at
their anatomically correct costovertebral attachments, so the expected answer is
known in closed form.

Run from the repository root:

    python -m pytest tests/test_vertebrae_rib_identity.py -v
"""

import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import vertebrae_rib_identity as adapter          # noqa: E402
from utils import vertebrae_rib_identity_engine as engine     # noqa: E402


# --------------------------------------------------------------------------
# Phantom construction
# --------------------------------------------------------------------------

ZOOM = 2.0                 # isotropic mm
PITCH_SLICES = 12          # 24 mm between level centres
BODY_SLICES = 8            # block height; the remaining 4 slices are the disc
N_LEVELS = 20              # engine ids 1..20, covering the 4..17 solve domain
Z0 = 10                    # first block starts here
SHAPE = (64, 64, Z0 + N_LEVELS * PITCH_SLICES + 20)


def level_slices(k):
    """Inclusive z-slice range occupied by the block for engine label k."""
    start = Z0 + (k - 1) * PITCH_SLICES
    return start, start + BODY_SLICES - 1


def level_centroid_mm(k):
    """Centroid the engine will measure for a uniform block, in mm."""
    lo, hi = level_slices(k)
    return (lo + hi) / 2.0 * ZOOM


def expected_attachment_mm(rib_number):
    """Costovertebral attachment predicted by the canonical anatomy model."""
    if rib_number == 1 or rib_number >= 10:
        return level_centroid_mm(engine.k_of_T(rib_number))
    above = level_centroid_mm(engine.k_of_T(rib_number - 1))
    below = level_centroid_mm(engine.k_of_T(rib_number))
    return (above + below) / 2.0


def build_phantom(shift_levels=0, rib_lr_offset_mm=0.0, ribs=range(1, 13)):
    """Takes: an optional whole-column label shift, an optional left/right rib
        disagreement in mm, and which ribs to draw.
    Does: builds a label volume, a matching CT, and a TotalSegmentator-style
        rib volume on one grid.
    Returns: (labels uint8, ct int16, rib volume int16, affine)."""
    labels = np.zeros(SHAPE, dtype=np.uint8)
    ct = np.full(SHAPE, -200, dtype=np.int16)
    ribs_vol = np.zeros(SHAPE, dtype=np.int16)

    cx = cy = SHAPE[0] // 2
    half = 10                                   # 20 voxels = 40 mm across

    for k in range(1, N_LEVELS + 1):
        lo, hi = level_slices(k)
        stored = k + shift_levels
        if not 1 <= stored <= engine.N_CLASSES:
            continue
        labels[cx - half:cx + half, cy - half:cy + half, lo:hi + 1] = stored
        ct[cx - half:cx + half, cy - half:cy + half, lo:hi + 1] = 400

    for rib_number in ribs:
        z_mm = expected_attachment_mm(rib_number)
        for side, sign, offset in (("left", 1, 0.0),
                                   ("right", -1, rib_lr_offset_mm)):
            z_idx = int(round((z_mm + offset) / ZOOM))
            if not 0 <= z_idx < SHAPE[2]:
                continue
            label_id = engine.TS_RIB_IDS[f"rib_{side}_{rib_number}"]
            if sign > 0:
                x0, x1 = cx + half, cx + half + 22
            else:
                x0, x1 = cx - half - 22, cx - half
            ribs_vol[x0:x1, cy - 3:cy + 3, z_idx:z_idx + 2] = label_id

    affine = np.diag([ZOOM, ZOOM, ZOOM, 1.0])   # already RAS
    return labels, ct, ribs_vol, affine


def phantom_dict(labels):
    """Turn a phantom label volume into a ShapeKit segmentation dict."""
    return {name: (labels == engine.CLASS_MAP_INV[name]).astype(np.uint8)
            for name in adapter.VERTEBRA_NAMES
            if np.any(labels == engine.CLASS_MAP_INV[name])}


# the engine does not expose a reverse map; build one for the tests
engine.CLASS_MAP_INV = {v: k for k, v in engine.CLASS_MAP.items()}


@pytest.fixture
def logger():
    log = logging.getLogger("rib_identity_test")
    log.handlers = []
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.DEBUG)
    return log


class _RefImg:
    """Minimal stand-in for the nibabel reference image the stage receives."""

    def __init__(self, affine, zooms=(ZOOM, ZOOM, ZOOM)):
        self.affine = affine

        class _H:
            def get_zooms(self_inner):
                return zooms

        self.header = _H()


# --------------------------------------------------------------------------
# 1. k_of_T
# --------------------------------------------------------------------------

def test_k_of_T_endpoints_and_monotonicity():
    assert engine.k_of_T(12) == 6           # T12
    assert engine.k_of_T(1) == 17           # T1
    values = [engine.k_of_T(n) for n in range(1, 13)]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == 12


def test_k_of_T_agrees_with_class_map():
    for n in range(1, 13):
        assert engine.CLASS_MAP[engine.k_of_T(n)] == f"vertebrae_T{n}"


# --------------------------------------------------------------------------
# 2. complete label mapping
# --------------------------------------------------------------------------

def test_label_mapping_endpoints():
    assert adapter.engine_id_to_shapekit_id(1) == 26
    assert engine.CLASS_MAP[1] == "vertebrae_L5"
    assert adapter.SHAPEKIT_VERTEBRA_LABELS[26] == "vertebrae_L5"

    assert adapter.engine_id_to_shapekit_id(24) == 49
    assert engine.CLASS_MAP[24] == "vertebrae_C1"
    assert adapter.SHAPEKIT_VERTEBRA_LABELS[49] == "vertebrae_C1"


def test_label_mapping_is_a_bijection_over_all_levels():
    for engine_id in range(1, 25):
        shapekit_id = adapter.engine_id_to_shapekit_id(engine_id)
        assert adapter.shapekit_id_to_engine_id(shapekit_id) == engine_id
        assert (engine.CLASS_MAP[engine_id]
                == adapter.SHAPEKIT_VERTEBRA_LABELS[shapekit_id])
    assert sorted(adapter.SHAPEKIT_VERTEBRA_LABELS) == list(range(26, 50))
    assert len(adapter.VERTEBRA_NAMES) == 24


def test_label_mapping_matches_upstream_table():
    from utils.vertebrae_postprocessing import all_labels
    assert {int(k): v for k, v in all_labels.items()} == \
        adapter.SHAPEKIT_VERTEBRA_LABELS


def test_label_mapping_matches_repository_config():
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml")) as handle:
        class_map = yaml.safe_load(handle)["class_map"]
    vertebrae = {int(k): v for k, v in class_map.items()
                 if str(v).startswith("vertebrae_")}
    assert vertebrae == adapter.SHAPEKIT_VERTEBRA_LABELS


# --------------------------------------------------------------------------
# 3 / 4. left-right rib measurement acceptance and rejection
# --------------------------------------------------------------------------

def _measure(rib_lr_offset_mm):
    labels, _, ribs_vol, _ = build_phantom(rib_lr_offset_mm=rib_lr_offset_mm)
    zooms = np.array([ZOOM, ZOOM, ZOOM])
    centre, _, _ = engine.spine_centreline(engine.debris_filtered(labels), zooms)
    log = {}
    found = engine.rib_attachments(ribs_vol, centre, zooms, log)
    return found, log


def test_symmetric_ribs_are_accepted():
    found, log = _measure(0.0)
    assert sorted(found) == list(range(1, 13))
    assert log["ribs_excluded_lr"] == []


def test_left_right_disagreement_inside_tolerance_is_accepted():
    found, log = _measure(engine.RIB_LR_TOL_MM - 3.0)
    assert sorted(found) == list(range(1, 13))
    assert log["ribs_excluded_lr"] == []


def test_left_right_disagreement_beyond_tolerance_is_rejected():
    found, log = _measure(engine.RIB_LR_TOL_MM + 9.0)
    assert found == {}
    excluded = {entry["rib"] for entry in log["ribs_excluded_lr"]}
    assert excluded == set(range(1, 13))
    for entry in log["ribs_excluded_lr"]:
        assert entry["lr_mm"] > engine.RIB_LR_TOL_MM


def test_rib_tolerance_constant_is_canonical():
    assert engine.RIB_LR_TOL_MM == 15.0


# --------------------------------------------------------------------------
# 5. costovertebral rib equations
# --------------------------------------------------------------------------

def _row_for(rib_number, measurement=0.0):
    """Extract the single rib row the solver builds for one rib."""
    ks = list(range(engine.K_LO, engine.K_HI + 1))
    centroid = {k: 0.0 for k in ks}
    captured = {}
    real_lstsq = np.linalg.lstsq

    def spy(A, b, rcond=None):
        captured["A"] = A
        captured["b"] = b
        return real_lstsq(A, b, rcond=rcond)

    np.linalg.lstsq = spy
    try:
        engine.solve_centroids({rib_number: measurement}, centroid,
                               {k: 1.0 for k in ks}, {})
    finally:
        np.linalg.lstsq = real_lstsq
    # the rib row is the first row, weighted by W_RIB
    return captured["A"][0] / engine.W_RIB, list(
        range(engine.K_LO, engine.K_HI + 1))


def test_rib_one_articulates_with_a_single_body():
    row, ks = _row_for(1)
    assert row[ks.index(engine.k_of_T(1))] == pytest.approx(1.0)
    assert np.count_nonzero(row) == 1


@pytest.mark.parametrize("rib_number", list(range(2, 10)))
def test_ribs_two_to_nine_span_two_bodies(rib_number):
    row, ks = _row_for(rib_number)
    above = ks.index(engine.k_of_T(rib_number - 1))
    below = ks.index(engine.k_of_T(rib_number))
    assert row[above] == pytest.approx(0.5)
    assert row[below] == pytest.approx(0.5)
    assert np.count_nonzero(row) == 2


@pytest.mark.parametrize("rib_number", [10, 11, 12])
def test_ribs_ten_to_twelve_articulate_with_their_own_body(rib_number):
    row, ks = _row_for(rib_number)
    assert row[ks.index(engine.k_of_T(rib_number))] == pytest.approx(1.0)
    assert np.count_nonzero(row) == 1


# --------------------------------------------------------------------------
# 6. least-squares construction and solution
# --------------------------------------------------------------------------

def _consistent_inputs():
    ks = list(range(engine.K_LO, engine.K_HI + 1))
    centroid = {k: level_centroid_mm(k) for k in ks}
    ribs = {n: expected_attachment_mm(n) for n in range(1, 13)}
    frac = {k: 1.0 for k in ks}
    return ribs, centroid, frac, ks


def test_solve_reproduces_a_self_consistent_column():
    ribs, centroid, frac, ks = _consistent_inputs()
    solved, delta = engine.solve_centroids(ribs, centroid, frac, {})
    for k in ks:
        assert solved[k] == pytest.approx(centroid[k], abs=1e-6)
        assert delta[k] == pytest.approx(0.0, abs=1e-6)


def test_pinned_lumbar_anchor_barely_moves():
    ribs, centroid, frac, _ = _consistent_inputs()
    for n in ribs:
        ribs[n] += 30.0                       # push every rib superiorly
    solved, delta = engine.solve_centroids(ribs, centroid, frac, {})
    assert abs(delta[engine.K_LO]) < 2.0      # W_PIN holds L2 in place
    assert abs(delta[engine.K_HI]) > abs(delta[engine.K_LO])


def test_solve_domain_is_l2_to_t1():
    assert (engine.K_LO, engine.K_HI) == (4, 17)
    assert engine.CLASS_MAP[engine.K_LO] == "vertebrae_L2"
    assert engine.CLASS_MAP[engine.K_HI] == "vertebrae_T1"


def test_solve_weights_are_canonical():
    assert (engine.W_RIB, engine.W_SMOOTH, engine.W_TIE, engine.W_PIN) == \
        (1.0, 0.9, 0.12, 50.0)


# --------------------------------------------------------------------------
# 12. threshold semantics  (controlled synthetic deltas, no patient data)
# --------------------------------------------------------------------------

def test_min_delta_constant_is_canonical():
    assert engine.MIN_DELTA_MM == 12.0


def test_threshold_operator_in_source_is_strictly_greater():
    """Guard the exact gating expression against an accidental >= or epsilon."""
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "utils", "vertebrae_rib_identity_engine.py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "abs(v) > MIN_DELTA_MM" in text
    assert "abs(v) >= MIN_DELTA_MM" not in text
    assert "MIN_DELTA_MM -" not in text and "MIN_DELTA_MM +" not in text


@pytest.mark.parametrize("delta_mm, corrected", [
    (0.0, False),
    (11.9, False),
    (12.0, False),          # exactly on the threshold: NOT corrected
    (-12.0, False),         # sign must not change the decision
    (12.000001, True),
    (12.1, True),
    (-12.1, True),
    (30.0, True),
])
def test_gate_decision_for_controlled_deltas(delta_mm, corrected, monkeypatch,
                                             logger):
    """Drive the real gate inside process() with a controlled delta."""
    labels, ct, ribs_vol, affine = build_phantom()
    target = 10                                   # engine id inside the domain

    real_solve = engine.solve_centroids

    def controlled(ribs, centroid, frac, log):
        solved, _ = real_solve(ribs, centroid, frac, log)
        delta = {k: 0.0 for k in solved}
        delta[target] = delta_mm
        log["delta_mm"] = {engine.SHORT[k]: v for k, v in delta.items()}
        return solved, delta

    monkeypatch.setattr(engine, "solve_centroids", controlled)
    _, log = engine.process(labels, affine, (ZOOM, ZOOM, ZOOM), ct, ribs_vol)

    assert log.get("levels_corrected") is not None
    was_corrected = engine.SHORT[target] in log["levels_corrected"]
    assert was_corrected is corrected


# --------------------------------------------------------------------------
# 11. a self-consistent column must be left completely alone
# --------------------------------------------------------------------------

def test_correct_column_is_not_modified():
    labels, ct, ribs_vol, affine = build_phantom()
    out, log = engine.process(labels, affine, (ZOOM, ZOOM, ZOOM), ct, ribs_vol)
    assert log["levels_corrected"] == []
    assert log["voxels_moved"] == 0
    assert np.array_equal(out[labels > 0], labels[labels > 0])


def test_shifted_column_is_detected_and_corrected():
    labels, ct, ribs_vol, affine = build_phantom(shift_levels=1)
    _, log = engine.process(labels, affine, (ZOOM, ZOOM, ZOOM), ct, ribs_vol)
    assert log["levels_corrected"], "a one-level shift should be detected"
    assert log["voxels_moved"] > 0


# --------------------------------------------------------------------------
# 10. too few usable ribs
# --------------------------------------------------------------------------

def test_too_few_ribs_makes_no_correction():
    labels, ct, ribs_vol, affine = build_phantom(shift_levels=1,
                                                 ribs=[4, 5])
    out, log = engine.process(labels, affine, (ZOOM, ZOOM, ZOOM), ct, ribs_vol)
    assert "too few usable ribs" in log["status"]
    assert np.array_equal(out, labels)


# --------------------------------------------------------------------------
# 7 / 8 / 9. adapter fallbacks
# --------------------------------------------------------------------------

def _dict_and_ref():
    labels, ct, ribs_vol, affine = build_phantom()
    return phantom_dict(labels), _RefImg(affine), labels, ct, ribs_vol, affine


def _save(tmp_path, name, array, affine):
    import nibabel as nib
    path = str(tmp_path / name)
    nib.save(nib.Nifti1Image(array, affine), path)
    return path


def test_missing_ct_is_a_safe_noop(tmp_path, logger):
    seg, ref, labels, _, ribs_vol, affine = _dict_and_ref()
    rib_path = _save(tmp_path, "total.nii.gz", ribs_vol, affine)
    before = {k: v.copy() for k, v in seg.items()}
    out, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", seg, ref, str(tmp_path / "absent_ct.nii.gz"), rib_path, logger)
    assert "CT not found" in log["status"]
    assert all(np.array_equal(out[k], before[k]) for k in before)


def test_missing_rib_volume_is_a_safe_noop(tmp_path, logger):
    seg, ref, labels, ct, _, affine = _dict_and_ref()
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)
    before = {k: v.copy() for k, v in seg.items()}
    out, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", seg, ref, ct_path, str(tmp_path / "absent_ribs.nii.gz"), logger)
    assert "rib volume not found" in log["status"]
    assert all(np.array_equal(out[k], before[k]) for k in before)


def test_incompatible_grid_is_a_safe_noop(tmp_path, logger):
    seg, ref, labels, ct, ribs_vol, affine = _dict_and_ref()
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)
    small = ribs_vol[:32, :32, :32]
    rib_path = _save(tmp_path, "total.nii.gz", small, affine)
    before = {k: v.copy() for k, v in seg.items()}
    out, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", seg, ref, ct_path, rib_path, logger)
    assert "incompatible" in log["status"]
    assert all(np.array_equal(out[k], before[k]) for k in before)


def test_mismatched_affine_is_a_safe_noop(tmp_path, logger):
    seg, ref, labels, ct, ribs_vol, affine = _dict_and_ref()
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)
    shifted = affine.copy()
    shifted[0, 3] += 25.0                      # same shape, different grid
    rib_path = _save(tmp_path, "total.nii.gz", ribs_vol, shifted)
    before = {k: v.copy() for k, v in seg.items()}
    out, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", seg, ref, ct_path, rib_path, logger)
    assert "incompatible" in log["status"]
    assert all(np.array_equal(out[k], before[k]) for k in before)


def test_too_few_masks_is_a_safe_noop(tmp_path, logger):
    seg, ref, labels, ct, ribs_vol, affine = _dict_and_ref()
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)
    rib_path = _save(tmp_path, "total.nii.gz", ribs_vol, affine)
    thin = {k: seg[k] for k in list(seg)[:2]}
    out, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", thin, ref, ct_path, rib_path, logger)
    assert "identity correction skipped" in log["status"]
    assert out is thin


def test_adapter_never_raises_on_garbage(tmp_path, logger):
    seg, ref, labels, ct, ribs_vol, affine = _dict_and_ref()
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)
    rib_path = _save(tmp_path, "total.nii.gz", ribs_vol, affine)

    class Broken:
        affine = "not-an-affine"
        header = None

    out, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", seg, Broken(), ct_path, rib_path, logger)
    assert "failed" in log["status"] or "skipped" in log["status"]
    assert out is seg


# --------------------------------------------------------------------------
# assembly / scatter round trip and overlap precedence
# --------------------------------------------------------------------------

def test_assemble_scatter_round_trip():
    labels, _, _, _ = build_phantom()
    seg = phantom_dict(labels)
    volume, present = adapter.assemble_label_volume(seg)
    assert volume is not None and len(present) == N_LEVELS
    assert np.array_equal(volume, labels)
    rebuilt = adapter.scatter_label_volume(volume, dict(seg))
    again, _ = adapter.assemble_label_volume(rebuilt)
    assert np.array_equal(again, labels)


def test_overlapping_masks_resolve_to_the_more_superior_level():
    shape = (4, 4, 4)
    seg = {"vertebrae_L5": np.ones(shape, dtype=np.uint8),
           "vertebrae_L4": np.ones(shape, dtype=np.uint8)}
    volume, _ = adapter.assemble_label_volume(seg)
    assert np.all(volume == adapter.name_to_engine_id("vertebrae_L4"))


def test_rib_path_resolution_prefers_the_case_directory(tmp_path):
    case = tmp_path / "CASE"
    case.mkdir()
    (case / "total.nii.gz").write_bytes(b"x")
    root = tmp_path / "ribs"
    (root / "CASE").mkdir(parents=True)
    (root / "CASE" / "total.nii.gz").write_bytes(b"x")
    found = adapter.resolve_rib_path(str(case), "CASE", "total.nii.gz", str(root))
    assert found == os.path.join(str(case), "total.nii.gz")


def test_rib_path_resolution_falls_back_to_the_external_root(tmp_path):
    case = tmp_path / "CASE"
    case.mkdir()
    root = tmp_path / "ribs"
    root.mkdir()
    (root / "CASE.nii.gz").write_bytes(b"x")
    found = adapter.resolve_rib_path(str(case), "CASE", "total.nii.gz", str(root))
    assert found == os.path.join(str(root), "CASE.nii.gz")


def test_rib_path_resolution_returns_none_when_absent(tmp_path):
    case = tmp_path / "CASE"
    case.mkdir()
    assert adapter.resolve_rib_path(str(case), "CASE", "total.nii.gz", None) is None


# --------------------------------------------------------------------------
# Configuration validation: the rib_anchor / engine pairing
#
# This is a configuration error, not a per-case condition. It is checked once
# at startup and stops the run; the per-case fallbacks above must stay no-ops.
# --------------------------------------------------------------------------

def test_rib_anchor_with_default_engine_is_rejected():
    with pytest.raises(adapter.IdentityConfigError) as excinfo:
        adapter.validate_identity_config("rib_anchor", "shapekit")
    message = str(excinfo.value)
    assert "cannot be used with" in message
    assert "shapekit_pro" in message
    assert "vertebrae_identity: none" in message
    assert "reassign contiguous" in message


def test_rib_anchor_with_pro_engine_is_accepted():
    assert adapter.validate_identity_config("rib_anchor", "shapekit_pro") is None


@pytest.mark.parametrize("identity", ["none", None, ""])
@pytest.mark.parametrize("engine_name", ["shapekit", "shapekit_pro"])
def test_disabled_identity_accepts_any_engine(identity, engine_name):
    assert adapter.validate_identity_config(identity, engine_name) is None


def test_unknown_identity_value_is_rejected():
    with pytest.raises(adapter.IdentityConfigError) as excinfo:
        adapter.validate_identity_config("rib_anchr", "shapekit_pro")
    assert "Unknown vertebrae_identity" in str(excinfo.value)


def test_unknown_engine_is_also_rejected_when_identity_is_on():
    """An engine we have not verified must not be assumed compatible."""
    with pytest.raises(adapter.IdentityConfigError):
        adapter.validate_identity_config("rib_anchor", "some_future_engine")


def test_validation_runs_before_any_case_is_processed():
    """Guard the call site: validation must precede the processing call."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "main.py"), encoding="utf-8") as handle:
        source = handle.read()
    guard = source.index("if __name__ == '__main__':")
    validate_at = source.index("validate_identity_config(", guard)
    run_at = source.index("run_in_parallel(", guard)
    assert validate_at < run_at, "config validation must run before processing"
    assert "sys.exit(2)" in source[guard:run_at]


def test_missing_inputs_are_not_configuration_errors(tmp_path, logger):
    """A case that lacks a CT or ribs must no-op, never raise IdentityConfigError."""
    seg, ref, labels, ct, ribs_vol, affine = _dict_and_ref()
    rib_path = _save(tmp_path, "total.nii.gz", ribs_vol, affine)
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)

    for missing_ct, missing_rib in ((True, False), (False, True), (True, True)):
        working = {k: v.copy() for k, v in seg.items()}
        out, log = adapter.postprocessing_vertebrae_rib_identity(
            "CASE", working, ref,
            str(tmp_path / "absent.nii.gz") if missing_ct else ct_path,
            str(tmp_path / "absent.nii.gz") if missing_rib else rib_path,
            logger)
        assert "skipped" in log["status"]
        assert all(np.array_equal(out[k], seg[k]) for k in seg)


# --------------------------------------------------------------------------
# End to end: the correction must survive the downstream engine
# --------------------------------------------------------------------------

def _run_stage_then(engine_callable, tmp_path):
    """Identity stage, then a downstream engine; returns (before, after) volumes."""
    labels, ct, ribs_vol, affine = build_phantom(shift_levels=1)
    seg = phantom_dict(labels)
    ref = _RefImg(affine)
    ct_path = _save(tmp_path, "ct.nii.gz", ct, affine)
    rib_path = _save(tmp_path, "total.nii.gz", ribs_vol, affine)

    def volume(d):
        out = np.zeros(labels.shape, dtype=np.uint8)
        for name in adapter.VERTEBRA_NAMES:
            mask = d.get(name)
            if mask is not None:
                out[mask > 0] = adapter.ENGINE_ID_BY_NAME[name]
        return out

    before = volume(seg)
    seg, log = adapter.postprocessing_vertebrae_rib_identity(
        "CASE", seg, ref, ct_path, rib_path, logging.getLogger("e2e"))
    corrected = volume(seg)
    seg = engine_callable(seg, ref, ct_path)
    return before, corrected, volume(seg), log


def test_shapekit_pro_retains_the_identity_correction(tmp_path):
    """The supported pairing must keep the correction after downstream cleanup."""
    from utils.vertebrae_pro import postprocessing_vertebrae_pro

    def run_pro(seg, ref, ct_path):
        return postprocessing_vertebrae_pro(
            "CASE", seg, ref, ct_path, logging.getLogger("e2e"))

    before, corrected, after, log = _run_stage_then(run_pro, tmp_path)

    assert log["voxels_moved"] == 38400
    assert int((before != corrected).sum()) == 38400
    # the correction is still present once shapekit_pro has run
    assert int((before != after).sum()) == 38400


def test_default_engine_would_revert_it(tmp_path):
    """Why the pairing is rejected: the default engine undoes the correction.

    This documents the measured behaviour that motivates
    ``validate_identity_config``. It asserts the revert, so the day the default
    engine stops reverting, this test fails and the restriction can be revisited.
    """
    from utils.vertebrae_postprocessing import postprocessing_vertebrae

    def run_default(seg, ref, ct_path):
        return postprocessing_vertebrae("CASE", seg, logger=logging.getLogger("e2e"))

    before, corrected, after, log = _run_stage_then(run_default, tmp_path)

    assert int((before != corrected).sum()) == 38400   # stage did its work
    assert int((before != after).sum()) == 0           # engine undid all of it
