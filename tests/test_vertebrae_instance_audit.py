"""Tests for the opt-in vertebral instance batch audit adapter."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
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


_ACTUAL_MAIN_SCRIPT = r"""
import json
import os
import sys
import types

# scikit-image is optional in the review environment.  These import-only
# stubs fail loudly if ShapeKit executes any stubbed operation.  target_organs
# is empty in this integration fixture, so no such operation is expected.
skimage = types.ModuleType("skimage")
morphology = types.ModuleType("skimage.morphology")
measure = types.ModuleType("skimage.measure")

def _unexpected_stub_call(*args, **kwargs):
    raise AssertionError("an import-only scikit-image stub was executed")

morphology.disk = _unexpected_stub_call
morphology.convex_hull_image = _unexpected_stub_call
measure.label = _unexpected_stub_call
measure.regionprops = _unexpected_stub_call
skimage.morphology = morphology
skimage.measure = measure
sys.modules["skimage"] = skimage
sys.modules["skimage.morphology"] = morphology
sys.modules["skimage.measure"] = measure

sys.path.insert(0, os.environ["SHAPEKIT_REPOSITORY"])
sys.argv = [
    "main.py",
    "--log_folder",
    os.environ["SHAPEKIT_LOG_FOLDER"],
]

import main

analyzer_before = "utils.vertebrae_instance_analysis" in sys.modules
if os.environ["SHAPEKIT_AUDIT_MODE"] == "failure":
    import utils.vertebrae_instance_audit as audit_adapter

    def _synthetic_analyzer_failure():
        raise RuntimeError("synthetic actual-main analyzer failure")

    audit_adapter._load_analyzer = _synthetic_analyzer_failure

main.main(
    os.environ["SHAPEKIT_CASE_PATH"],
    "case_001",
    os.environ["SHAPEKIT_OUTPUT_ROOT"],
)
analyzer_after = "utils.vertebrae_instance_analysis" in sys.modules
print(
    "__SHAPEKIT_TEST_RESULT__"
    + json.dumps(
        {
            "analyzer_before": analyzer_before,
            "analyzer_after": analyzer_after,
        },
        sort_keys=True,
    )
)
"""


def _create_actual_main_case(root: Path) -> Path:
    segmentation_folder = root / "input" / "case_001" / "segmentations"
    segmentation_folder.mkdir(parents=True)
    shape = (40, 40, 80)
    affine = np.diag([1.0, 1.0, 1.5, 1.0])
    reference = np.zeros(shape, dtype=np.float32)
    reference[1:3, 1:3, 1:3] = 300.0
    vertebra = _ellipsoid(shape=shape, center=(20.0, 20.0, 40.0))
    nib.save(
        nib.Nifti1Image(reference, affine),
        segmentation_folder / "liver.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(vertebra.astype(np.uint8), affine),
        segmentation_folder / "vertebrae_L1.nii.gz",
    )
    return segmentation_folder.parent


def _run_actual_main(
    *,
    repository: Path,
    root: Path,
    case_path: Path,
    mode: str,
) -> dict:
    raw_config = yaml.safe_load(
        (repository / "config.yaml").read_text(encoding="utf-8")
    )
    raw_config["target_organs"] = []
    if mode == "absent":
        raw_config.pop("vertebrae_instance_analysis", None)
    else:
        raw_config["vertebrae_instance_analysis"] = {
            "enabled": mode != "disabled",
            "output_dir_name": "vertebrae_analysis",
            "use_reference_as_ct": mode == "ct",
        }

    run_folder = root / f"run_{mode}"
    output_root = root / f"output_{mode}"
    log_folder = root / f"logs_{mode}"
    cache_folder = root / f"cache_{mode}"
    run_folder.mkdir()
    output_root.mkdir()
    (run_folder / "config.yaml").write_text(
        yaml.safe_dump(raw_config, sort_keys=False),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(cache_folder),
            "SHAPEKIT_REPOSITORY": str(repository),
            "SHAPEKIT_LOG_FOLDER": str(log_folder),
            "SHAPEKIT_AUDIT_MODE": mode,
            "SHAPEKIT_CASE_PATH": str(case_path),
            "SHAPEKIT_OUTPUT_ROOT": str(output_root),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _ACTUAL_MAIN_SCRIPT],
        cwd=run_folder,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"actual main.main() failed for {mode}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    marker = "__SHAPEKIT_TEST_RESULT__"
    result_line = next(
        line for line in completed.stdout.splitlines()
        if line.startswith(marker)
    )
    nifti_hashes = {
        str(path.relative_to(output_root)): _file_digest(path)
        for path in sorted(output_root.rglob("*.nii.gz"))
    }
    audit_path = (
        output_root / "vertebrae_analysis" / "case_001.json"
    )
    return {
        "output_root": output_root,
        "nifti_hashes": nifti_hashes,
        "audit_path": audit_path,
        "process_result": json.loads(result_line[len(marker):]),
        "log_folder": log_folder,
    }


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

    def test_failure_preserves_existing_report_and_handoff(
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
            existing_payload = b'{"previous_success":true}\n'
            path.write_bytes(existing_payload)
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
            self.assertEqual(path.read_bytes(), existing_payload)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            logger.exception.assert_called()

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

    def test_write_failure_preserves_final_and_removes_temporary_output(
        self,
    ) -> None:
        masks = _masks()
        with tempfile.TemporaryDirectory() as temporary:
            config = VertebraeInstanceAuditConfig(enabled=True)
            path = audit_output_path(temporary, config, "case_001")
            path.parent.mkdir(parents=True)
            existing_payload = b'{"previous_success":true}\n'
            path.write_bytes(existing_payload)
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
            self.assertEqual(path.read_bytes(), existing_payload)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_same_patient_concurrent_failure_cannot_delete_success(
        self,
    ) -> None:
        masks = _masks()
        config = VertebraeInstanceAuditConfig(enabled=True)
        failure_started = threading.Event()
        success_finished = threading.Event()
        statuses = {}
        logger = mock.Mock()

        def coordinated_analyzer(*args, **kwargs):
            if threading.current_thread().name == "failing-audit":
                failure_started.set()
                if not success_finished.wait(timeout=10):
                    raise AssertionError("successful worker did not finish")
                raise RuntimeError("synthetic concurrent failure")
            if not failure_started.wait(timeout=10):
                raise AssertionError("failing worker did not start")
            return {"worker": "successful-audit"}

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            path = audit_output_path(
                output_root,
                config,
                "case_001",
            )

            def successful_worker():
                statuses["success"] = run_vertebrae_instance_audit(
                    masks,
                    reference_img=_GeometryOnlyReference(),
                    ordered_anatomical_names=NAMES,
                    output_root=output_root,
                    patient_id="case_001",
                    config=config,
                    logger=logger,
                )
                success_finished.set()

            def failing_worker():
                statuses["failure"] = run_vertebrae_instance_audit(
                    masks,
                    reference_img=_GeometryOnlyReference(),
                    ordered_anatomical_names=NAMES,
                    output_root=output_root,
                    patient_id="case_001",
                    config=config,
                    logger=logger,
                )

            with mock.patch(
                "utils.vertebrae_instance_audit._load_analyzer",
                return_value=coordinated_analyzer,
            ):
                failure_thread = threading.Thread(
                    target=failing_worker,
                    name="failing-audit",
                )
                success_thread = threading.Thread(
                    target=successful_worker,
                    name="successful-audit",
                )
                failure_thread.start()
                success_thread.start()
                failure_thread.join(timeout=15)
                success_thread.join(timeout=15)

            self.assertFalse(failure_thread.is_alive())
            self.assertFalse(success_thread.is_alive())
            self.assertEqual(
                statuses,
                {"success": "written", "failure": "failed"},
            )
            payload = path.read_bytes()
            report = json.loads(payload)
            self.assertEqual(report, {"worker": "successful-audit"})
            self.assertEqual(
                payload,
                b'{"worker":"successful-audit"}\n',
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_later_failure_preserves_same_and_other_patient_reports(
        self,
    ) -> None:
        masks = _masks()
        config = VertebraeInstanceAuditConfig(enabled=True)
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            with mock.patch(
                "utils.vertebrae_instance_audit._load_analyzer",
                return_value=lambda *args, **kwargs: {"success": True},
            ):
                for patient_id in ("case_001", "case_002"):
                    self.assertEqual(
                        run_vertebrae_instance_audit(
                            masks,
                            reference_img=_GeometryOnlyReference(),
                            ordered_anatomical_names=NAMES,
                            output_root=output_root,
                            patient_id=patient_id,
                            config=config,
                            logger=mock.Mock(),
                        ),
                        "written",
                    )
            first_path = audit_output_path(
                output_root,
                config,
                "case_001",
            )
            second_path = audit_output_path(
                output_root,
                config,
                "case_002",
            )
            first_payload = first_path.read_bytes()
            second_payload = second_path.read_bytes()

            with mock.patch(
                "utils.vertebrae_instance_audit._load_analyzer",
                side_effect=RuntimeError("later failure"),
            ):
                self.assertEqual(
                    run_vertebrae_instance_audit(
                        masks,
                        reference_img=_GeometryOnlyReference(),
                        ordered_anatomical_names=NAMES,
                        output_root=output_root,
                        patient_id="case_001",
                        config=config,
                        logger=mock.Mock(),
                    ),
                    "failed",
                )
            self.assertEqual(first_path.read_bytes(), first_payload)
            self.assertEqual(second_path.read_bytes(), second_payload)
            self.assertEqual(list(first_path.parent.glob("*.tmp")), [])

    def test_symlinked_audit_directories_are_rejected(self) -> None:
        masks = _masks()
        before = {
            name: _array_digest(mask)
            for name, mask in masks.items()
        }
        config = VertebraeInstanceAuditConfig(enabled=True)
        for destination_kind in ("segmentation", "outside"):
            with self.subTest(destination_kind=destination_kind):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output_root = root / "output"
                    output_root.mkdir()
                    if destination_kind == "segmentation":
                        target = (
                            output_root
                            / "case_001"
                            / "segmentations"
                        )
                    else:
                        target = root / "outside"
                    target.mkdir(parents=True)
                    audit_directory = (
                        output_root / config.output_dir_name
                    )
                    try:
                        audit_directory.symlink_to(
                            target,
                            target_is_directory=True,
                        )
                    except (NotImplementedError, OSError) as error:
                        self.skipTest(
                            f"symbolic links are unavailable: {error}"
                        )

                    status = run_vertebrae_instance_audit(
                        masks,
                        reference_img=_GeometryOnlyReference(),
                        ordered_anatomical_names=NAMES,
                        output_root=output_root,
                        patient_id="case_001",
                        config=config,
                        logger=mock.Mock(),
                    )
                    self.assertEqual(status, "failed")
                    self.assertEqual(list(target.glob("*.json")), [])
                    self.assertEqual(list(target.glob("*.tmp")), [])
                    self.assertEqual(
                        before,
                        {
                            name: _array_digest(mask)
                            for name, mask in masks.items()
                        },
                    )

    def test_existing_real_audit_directory_is_accepted(self) -> None:
        masks = _masks()
        config = VertebraeInstanceAuditConfig(enabled=True)
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            audit_directory = (
                output_root / config.output_dir_name
            )
            audit_directory.mkdir()
            self.assertEqual(
                run_vertebrae_instance_audit(
                    masks,
                    reference_img=_GeometryOnlyReference(),
                    ordered_anatomical_names=NAMES,
                    output_root=output_root,
                    patient_id="case_001",
                    config=config,
                    logger=mock.Mock(),
                ),
                "written",
            )
            self.assertTrue((audit_directory / "case_001.json").is_file())

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

    def test_patient_identifiers_with_control_characters_are_rejected(
        self,
    ) -> None:
        config = VertebraeInstanceAuditConfig(enabled=True)
        for patient_id in (
            "case\ninjected",
            "case\tinjected",
            "case\x1finjected",
            "case\u0085injected",
        ):
            with self.subTest(patient_id=repr(patient_id)):
                with self.assertRaises(ValueError):
                    audit_output_path(".", config, patient_id)

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

    def test_actual_main_modes_preserve_segmentation_output_bytes(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_path = _create_actual_main_case(root)
            input_hashes = {
                str(path.relative_to(case_path)): _file_digest(path)
                for path in sorted(case_path.rglob("*.nii.gz"))
            }
            results = {
                mode: _run_actual_main(
                    repository=repository,
                    root=root,
                    case_path=case_path,
                    mode=mode,
                )
                for mode in (
                    "absent",
                    "disabled",
                    "geometry",
                    "ct",
                    "failure",
                )
            }

            expected_hashes = results["absent"]["nifti_hashes"]
            self.assertTrue(expected_hashes)
            for mode, result in results.items():
                with self.subTest(mode=mode):
                    self.assertEqual(
                        result["nifti_hashes"],
                        expected_hashes,
                    )

            for mode in ("absent", "disabled"):
                result = results[mode]
                self.assertFalse(
                    (
                        result["output_root"]
                        / "vertebrae_analysis"
                    ).exists()
                )
                self.assertFalse(
                    result["process_result"]["analyzer_before"]
                )
                self.assertFalse(
                    result["process_result"]["analyzer_after"]
                )

            for mode, expected_ct_evidence in (
                ("geometry", "unavailable"),
                ("ct", "used"),
            ):
                audit_path = results[mode]["audit_path"]
                payload = audit_path.read_bytes()
                report = json.loads(payload)
                self.assertEqual(
                    report["ct_evidence"],
                    expected_ct_evidence,
                )
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

            self.assertFalse(results["failure"]["audit_path"].exists())
            self.assertEqual(
                list(
                    results["failure"]["audit_path"].parent.glob(
                        "*.tmp"
                    )
                ),
                [],
            )
            failure_log = (
                results["failure"]["log_folder"] / "debug.log"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "synthetic actual-main analyzer failure",
                failure_log,
            )
            self.assertIn("patient=case_001", failure_log)
            self.assertIn("report=", failure_log)
            self.assertIn("requested_ct_mode=geometry-only", failure_log)
            self.assertIn("effective_ct_mode=geometry-only", failure_log)

            self.assertEqual(
                input_hashes,
                {
                    str(path.relative_to(case_path)): _file_digest(path)
                    for path in sorted(case_path.rglob("*.nii.gz"))
                },
            )


if __name__ == "__main__":
    unittest.main()
