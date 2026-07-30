"""Tests for the opt-in vertebral instance batch audit adapter."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import yaml

from utils.vertebrae_instance_audit import (
    VertebraeInstanceAuditConfig,
    audit_output_path,
    ordered_vertebral_anatomical_names,
    parse_vertebrae_instance_audit_config,
    run_vertebrae_instance_audit,
)


NAMES = (
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
)


class _ReferenceImage:
    def __init__(self, data: np.ndarray) -> None:
        self.affine = np.eye(4, dtype=np.float64)
        self.dataobj = data


class _GeometryOnlyReference:
    affine = np.eye(4, dtype=np.float64)

    @property
    def dataobj(self):
        raise AssertionError("disabled CT data was accessed")


def _ellipsoid(
    shape=(40, 40, 80),
    center=(20.0, 20.0, 40.0),
) -> np.ndarray:
    grid = np.ogrid[tuple(slice(0, length) for length in shape)]
    normalized = (
        ((grid[0] - center[0]) / 11.0) ** 2
        + ((grid[1] - center[1]) / 12.0) ** 2
        + ((grid[2] - center[2]) / 8.0) ** 2
    )
    return normalized <= 1.0


def _masks() -> dict:
    return {
        NAMES[0]: _ellipsoid(center=(20.0, 20.0, 24.0)),
        NAMES[1]: _ellipsoid(center=(20.0, 20.0, 54.0)),
    }


def _array_digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_legacy_fixture(
    masks: dict,
    class_map: dict,
    reference: nib.Nifti1Image,
    output_folder: Path,
) -> None:
    segmentation_folder = output_folder / "segmentations"
    segmentation_folder.mkdir(parents=True)
    combined = np.zeros(reference.shape, dtype=np.uint8)
    for label_id, anatomical_name in sorted(class_map.items()):
        mask = masks.get(anatomical_name)
        if mask is None or not np.any(mask):
            continue
        binary = mask.astype(np.uint8, copy=False)
        nib.save(
            nib.Nifti1Image(binary, reference.affine),
            segmentation_folder / f"{anatomical_name}.nii.gz",
        )
        combined[binary > 0] = label_id
    nib.save(
        nib.Nifti1Image(combined, reference.affine),
        output_folder / "combined_labels.nii.gz",
    )


def _spawn_audit_worker(payload) -> str:
    output_root, patient_id = payload
    masks = _masks()
    reference = _GeometryOnlyReference()
    status = run_vertebrae_instance_audit(
        masks,
        reference_img=reference,
        ordered_anatomical_names=NAMES,
        output_root=output_root,
        patient_id=patient_id,
        config=VertebraeInstanceAuditConfig(enabled=True),
        logger=logging.getLogger(f"audit-test-{patient_id}"),
    )
    return status


class VertebraeInstanceAuditTests(unittest.TestCase):
    def test_repository_config_is_disabled_by_default(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        raw = yaml.safe_load(
            (repository / "config.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            raw["vertebrae_instance_analysis"],
            {
                "enabled": False,
                "output_dir_name": "vertebrae_analysis",
                "use_reference_as_ct": False,
            },
        )
        self.assertFalse(
            parse_vertebrae_instance_audit_config(
                raw["vertebrae_instance_analysis"]
            ).enabled
        )

    def test_missing_and_disabled_config_do_nothing(self) -> None:
        masks = _masks()
        reference = _GeometryOnlyReference()
        for config in (
            parse_vertebrae_instance_audit_config(None),
            VertebraeInstanceAuditConfig(enabled=False),
        ):
            with self.subTest(config=config):
                with tempfile.TemporaryDirectory() as temporary:
                    output_root = Path(temporary) / "output"
                    with mock.patch(
                        "utils.vertebrae_instance_audit._load_analyzer",
                        side_effect=AssertionError(
                            "analyzer imported while disabled"
                        ),
                    ):
                        status = run_vertebrae_instance_audit(
                            masks,
                            reference_img=reference,
                            ordered_anatomical_names=NAMES,
                            output_root=output_root,
                            patient_id="case_001",
                            config=config,
                            logger=mock.Mock(),
                        )
                    self.assertEqual(status, "disabled")
                    self.assertFalse(output_root.exists())

    def test_enabled_geometry_only_writes_canonical_json(self) -> None:
        masks = _masks()
        before = {
            name: (id(mask), _array_digest(mask))
            for name, mask in masks.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            config = VertebraeInstanceAuditConfig(enabled=True)
            status = run_vertebrae_instance_audit(
                masks,
                reference_img=_GeometryOnlyReference(),
                ordered_anatomical_names=NAMES,
                output_root=output_root,
                patient_id="case_001",
                config=config,
                logger=mock.Mock(),
            )
            self.assertEqual(status, "written")
            path = audit_output_path(
                output_root,
                config,
                "case_001",
            )
            payload = path.read_bytes()
            report = json.loads(payload)
            expected = (
                json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            self.assertEqual(payload, expected)
            self.assertEqual(report["ct_evidence"], "unavailable")
            self.assertNotIn("segmentations", path.parts)
        after = {
            name: (id(mask), _array_digest(mask))
            for name, mask in masks.items()
        }
        self.assertEqual(before, after)

    def test_ct_is_passed_only_when_explicitly_requested(self) -> None:
        masks = _masks()
        ct = np.full(
            next(iter(masks.values())).shape,
            300.0,
            dtype=np.float32,
        )
        captured = []

        def fake_analyzer(
            segmentation_dict,
            *,
            affine,
            ordered_anatomical_names,
            ct=None,
        ):
            captured.append(ct)
            return {"ct_supplied": ct is not None}

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "utils.vertebrae_instance_audit._load_analyzer",
                return_value=fake_analyzer,
            ):
                for index, use_ct in enumerate((False, True)):
                    status = run_vertebrae_instance_audit(
                        masks,
                        reference_img=_ReferenceImage(ct),
                        ordered_anatomical_names=NAMES,
                        output_root=temporary,
                        patient_id=f"case_{index}",
                        config=VertebraeInstanceAuditConfig(
                            enabled=True,
                            use_reference_as_ct=use_ct,
                        ),
                        logger=mock.Mock(),
                    )
                    self.assertEqual(status, "written")
        self.assertIsNone(captured[0])
        self.assertIs(captured[1], ct)

    def test_invalid_requested_ct_falls_back_to_geometry(self) -> None:
        masks = _masks()
        shape = next(iter(masks.values())).shape
        invalid_ct_values = (
            np.zeros((3, 4, 5), dtype=np.float32),
            np.full(shape, np.nan, dtype=np.float32),
        )
        for index, invalid_ct in enumerate(invalid_ct_values):
            captured = []

            def fake_analyzer(
                segmentation_dict,
                *,
                affine,
                ordered_anatomical_names,
                ct=None,
            ):
                captured.append(ct)
                return {"ct_supplied": ct is not None}

            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as temporary:
                    logger = mock.Mock()
                    with mock.patch(
                        "utils.vertebrae_instance_audit._load_analyzer",
                        return_value=fake_analyzer,
                    ):
                        status = run_vertebrae_instance_audit(
                            masks,
                            reference_img=_ReferenceImage(invalid_ct),
                            ordered_anatomical_names=NAMES,
                            output_root=temporary,
                            patient_id=f"case_{index}",
                            config=VertebraeInstanceAuditConfig(
                                enabled=True,
                                use_reference_as_ct=True,
                            ),
                            logger=logger,
                        )
                    self.assertEqual(status, "written")
                    self.assertEqual(captured, [None])
                    logger.warning.assert_called()

    def test_failure_removes_stale_output_and_preserves_handoff(
        self,
    ) -> None:
        masks = _masks()
        before = {
            name: (id(mask), _array_digest(mask))
            for name, mask in masks.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            config = VertebraeInstanceAuditConfig(enabled=True)
            path = audit_output_path(temporary, config, "case_001")
            path.parent.mkdir(parents=True)
            path.write_text("stale", encoding="utf-8")
            logger = mock.Mock()
            with mock.patch(
                "utils.vertebrae_instance_audit._load_analyzer",
                side_effect=RuntimeError("synthetic analysis failure"),
            ):
                status = run_vertebrae_instance_audit(
                    masks,
                    reference_img=_GeometryOnlyReference(),
                    ordered_anatomical_names=NAMES,
                    output_root=temporary,
                    patient_id="case_001",
                    config=config,
                    logger=logger,
                )
            self.assertEqual(status, "failed")
            self.assertFalse(path.exists())
            logger.exception.assert_called()

            reference = nib.Nifti1Image(
                np.zeros(
                    next(iter(masks.values())).shape,
                    dtype=np.uint8,
                ),
                np.eye(4),
            )
            class_map = {
                index: name
                for index, name in enumerate(masks, start=1)
            }
            baseline_output = Path(temporary) / "baseline_output"
            failure_output = Path(temporary) / "failure_output"
            for output_folder in (
                baseline_output,
                failure_output,
            ):
                _save_legacy_fixture(
                    masks={
                        name: mask.copy()
                        for name, mask in masks.items()
                    },
                    class_map=class_map,
                    reference=reference,
                    output_folder=output_folder,
                )
            baseline_files = sorted(
                path.relative_to(baseline_output)
                for path in baseline_output.rglob("*")
                if path.is_file()
            )
            failure_files = sorted(
                path.relative_to(failure_output)
                for path in failure_output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(baseline_files, failure_files)
            for relative_path in baseline_files:
                self.assertEqual(
                    _file_digest(baseline_output / relative_path),
                    _file_digest(failure_output / relative_path),
                )

        handoff = {}

        def downstream(segmentation_dict):
            handoff["object"] = segmentation_dict
            handoff["hashes"] = {
                name: _array_digest(mask)
                for name, mask in segmentation_dict.items()
            }

        downstream(masks)
        self.assertIs(handoff["object"], masks)
        self.assertEqual(
            before,
            {
                name: (id(mask), handoff["hashes"][name])
                for name, mask in masks.items()
            },
        )

    def test_write_failure_removes_stale_and_temporary_output(
        self,
    ) -> None:
        masks = _masks()
        with tempfile.TemporaryDirectory() as temporary:
            config = VertebraeInstanceAuditConfig(enabled=True)
            path = audit_output_path(temporary, config, "case_001")
            path.parent.mkdir(parents=True)
            path.write_text("stale", encoding="utf-8")
            with mock.patch(
                "utils.vertebrae_instance_audit.os.replace",
                side_effect=OSError("synthetic write failure"),
            ):
                status = run_vertebrae_instance_audit(
                    masks,
                    reference_img=_GeometryOnlyReference(),
                    ordered_anatomical_names=NAMES,
                    output_root=temporary,
                    patient_id="case_001",
                    config=config,
                    logger=mock.Mock(),
                )
            self.assertEqual(status, "failed")
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_repeated_runs_write_byte_identical_json(self) -> None:
        masks = _masks()
        with tempfile.TemporaryDirectory() as temporary:
            config = VertebraeInstanceAuditConfig(enabled=True)
            path = audit_output_path(temporary, config, "case_001")
            payloads = []
            for _ in range(2):
                self.assertEqual(
                    run_vertebrae_instance_audit(
                        masks,
                        reference_img=_GeometryOnlyReference(),
                        ordered_anatomical_names=NAMES,
                        output_root=temporary,
                        patient_id="case_001",
                        config=config,
                        logger=mock.Mock(),
                    ),
                    "written",
                )
                payloads.append(path.read_bytes())
            self.assertEqual(payloads[0], payloads[1])

    def test_spawned_cases_use_distinct_deterministic_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payloads = [
                (temporary, "case_001"),
                (temporary, "case_002"),
            ]
            context = multiprocessing.get_context("spawn")
            with context.Pool(processes=2) as pool:
                statuses = pool.map(_spawn_audit_worker, payloads)
            self.assertEqual(statuses, ["written", "written"])
            output_dir = Path(temporary) / "vertebrae_analysis"
            paths = sorted(output_dir.glob("*.json"))
            self.assertEqual(
                [path.name for path in paths],
                ["case_001.json", "case_002.json"],
            )
            for path in paths:
                payload = path.read_bytes()
                report = json.loads(payload)
                self.assertEqual(
                    payload,
                    (
                        json.dumps(
                            report,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )

    def test_ordering_uses_names_not_numeric_class_keys(self) -> None:
        first = {
            900: "vertebrae_C1",
            -4: "vertebrae_L1",
            12: "liver",
            3: "vertebrae_T12",
            1: "vertebrae_L5",
        }
        second = {
            1: "vertebrae_T12",
            800: "vertebrae_L5",
            2: "vertebrae_C1",
            44: "vertebrae_L1",
            9: "liver",
        }
        expected = (
            "vertebrae_L5",
            "vertebrae_L1",
            "vertebrae_T12",
            "vertebrae_C1",
        )
        self.assertEqual(
            ordered_vertebral_anatomical_names(first.values()),
            expected,
        )
        self.assertEqual(
            ordered_vertebral_anatomical_names(second.values()),
            expected,
        )

    def test_malformed_configuration_is_rejected(self) -> None:
        invalid = (
            [],
            {"enabled": "false"},
            {"use_reference_as_ct": 1},
            {"output_dir_name": ""},
            {"output_dir_name": ".."},
            {"output_dir_name": "/absolute"},
            {"output_dir_name": "nested/path"},
            {"output_dir_name": " leading"},
            {"unknown": True},
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_vertebrae_instance_audit_config(raw)

    def test_main_calls_audit_before_combination_as_unused_result(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (repository / "main.py").read_text(encoding="utf-8")
        )
        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

        call_positions = {}
        audit_statement = None
        for statement_index, statement in enumerate(main_function.body):
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if isinstance(function, ast.Name):
                    call_positions.setdefault(
                        function.id,
                        statement_index,
                    )
                    if function.id == "run_vertebrae_instance_audit":
                        audit_statement = statement

        self.assertLess(
            call_positions["read_all_segmentations"],
            call_positions["run_vertebrae_instance_audit"],
        )
        self.assertLess(
            call_positions["run_vertebrae_instance_audit"],
            call_positions["combine_segmentation_dict"],
        )
        self.assertLess(
            call_positions["combine_segmentation_dict"],
            call_positions["process_organs"],
        )
        self.assertIsInstance(audit_statement, ast.Expr)

    def test_disabled_mode_preserves_legacy_output_bytes(self) -> None:
        shape = (20, 20, 24)
        masks = {
            "vertebrae_L1": _ellipsoid(
                shape=shape,
                center=(10.0, 10.0, 12.0),
            ),
            "liver": np.zeros(shape, dtype=bool),
        }
        masks["liver"][3:8, 3:8, 3:8] = True
        class_map = {1: "liver", 2: "vertebrae_L1"}
        reference = nib.Nifti1Image(
            np.zeros(shape, dtype=np.uint8),
            np.eye(4),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline" / "case_001"
            disabled = root / "disabled" / "case_001"
            _save_legacy_fixture(
                masks={
                    name: mask.copy()
                    for name, mask in masks.items()
                },
                class_map=class_map,
                reference=reference,
                output_folder=baseline,
            )
            self.assertEqual(
                run_vertebrae_instance_audit(
                    masks,
                    reference_img=_GeometryOnlyReference(),
                    ordered_anatomical_names=("vertebrae_L1",),
                    output_root=root / "disabled",
                    patient_id="case_001",
                    config=VertebraeInstanceAuditConfig(enabled=False),
                    logger=mock.Mock(),
                ),
                "disabled",
            )
            _save_legacy_fixture(
                masks={
                    name: mask.copy()
                    for name, mask in masks.items()
                },
                class_map=class_map,
                reference=reference,
                output_folder=disabled,
            )

            baseline_files = sorted(
                path.relative_to(baseline)
                for path in baseline.rglob("*")
                if path.is_file()
            )
            disabled_files = sorted(
                path.relative_to(disabled)
                for path in disabled.rglob("*")
                if path.is_file()
            )
            self.assertEqual(baseline_files, disabled_files)
            for relative_path in baseline_files:
                self.assertEqual(
                    _file_digest(baseline / relative_path),
                    _file_digest(disabled / relative_path),
                )
            self.assertFalse(
                (root / "disabled" / "vertebrae_analysis").exists()
            )


if __name__ == "__main__":
    unittest.main()
