"""Opt-in batch adapter for read-only vertebral instance audit reports."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VertebraeInstanceAuditConfig:
    """Validated batch-integration configuration."""

    enabled: bool = False
    output_dir_name: str = "vertebrae_analysis"
    use_reference_as_ct: bool = False


AuditRunStatus = Literal["disabled", "written", "failed"]

_CONFIG_KEYS = {
    "enabled",
    "output_dir_name",
    "use_reference_as_ct",
}
_VERTEBRA_NAME = re.compile(r"^vertebrae_([LTC])([1-9][0-9]*)$")
_REGION_ORDER = {"L": 0, "T": 1, "C": 2}
_REGION_MAXIMUM = {"L": 5, "T": 12, "C": 7}
_REPORT_SUFFIX = ".json"


def parse_vertebrae_instance_audit_config(
    raw_config: object,
) -> VertebraeInstanceAuditConfig:
    """Validate the optional nested configuration block."""

    if raw_config is None:
        return VertebraeInstanceAuditConfig()
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            "vertebrae_instance_analysis must be a mapping"
        )

    unknown = sorted(
        (key for key in raw_config if key not in _CONFIG_KEYS),
        key=str,
    )
    if unknown:
        raise ValueError(
            "Unknown vertebrae_instance_analysis configuration keys: "
            + ", ".join(str(key) for key in unknown)
        )

    enabled = raw_config.get("enabled", False)
    use_reference_as_ct = raw_config.get(
        "use_reference_as_ct",
        False,
    )
    if type(enabled) is not bool:
        raise ValueError(
            "vertebrae_instance_analysis.enabled must be a boolean"
        )
    if type(use_reference_as_ct) is not bool:
        raise ValueError(
            "vertebrae_instance_analysis.use_reference_as_ct "
            "must be a boolean"
        )

    output_dir_name = raw_config.get(
        "output_dir_name",
        "vertebrae_analysis",
    )
    if not isinstance(output_dir_name, str):
        raise ValueError(
            "vertebrae_instance_analysis.output_dir_name "
            "must be a string"
        )
    if (
        not output_dir_name
        or output_dir_name != output_dir_name.strip()
        or output_dir_name in {".", ".."}
        or os.path.isabs(output_dir_name)
        or "/" in output_dir_name
        or "\\" in output_dir_name
    ):
        raise ValueError(
            "vertebrae_instance_analysis.output_dir_name must be "
            "one nonempty relative path component"
        )

    return VertebraeInstanceAuditConfig(
        enabled=enabled,
        output_dir_name=output_dir_name,
        use_reference_as_ct=use_reference_as_ct,
    )


def ordered_vertebral_anatomical_names(
    anatomical_names: Iterable[str],
) -> Tuple[str, ...]:
    """Return canonical vertebral names in inferior-to-superior order."""

    parsed = []
    seen = set()
    for anatomical_name in anatomical_names:
        if not isinstance(anatomical_name, str):
            raise ValueError("class-map anatomical names must be strings")
        if not anatomical_name.startswith("vertebrae_"):
            continue
        match = _VERTEBRA_NAME.fullmatch(anatomical_name)
        if match is None:
            raise ValueError(
                "Unsupported vertebral anatomical name: "
                f"{anatomical_name}"
            )
        region, level_text = match.groups()
        level = int(level_text)
        if level > _REGION_MAXIMUM[region]:
            raise ValueError(
                "Unsupported vertebral anatomical name: "
                f"{anatomical_name}"
            )
        if anatomical_name in seen:
            raise ValueError(
                "Duplicate vertebral anatomical name: "
                f"{anatomical_name}"
            )
        seen.add(anatomical_name)
        parsed.append((region, level, anatomical_name))

    if not parsed:
        raise ValueError(
            "No canonical vertebral anatomical names were configured"
        )

    parsed.sort(
        key=lambda item: (
            _REGION_ORDER[item[0]],
            -item[1],
            item[2],
        )
    )
    return tuple(item[2] for item in parsed)


def audit_output_path(
    output_root: os.PathLike,
    config: VertebraeInstanceAuditConfig,
    patient_id: str,
) -> Path:
    """Build the deterministic report path outside case segmentations."""

    if (
        not isinstance(patient_id, str)
        or not patient_id
        or patient_id in {".", ".."}
        or "/" in patient_id
        or "\\" in patient_id
        or "\x00" in patient_id
        or any(not character.isprintable() for character in patient_id)
    ):
        raise ValueError(
            "patient_id must be one printable nonempty path component"
        )
    return (
        Path(output_root)
        / config.output_dir_name
        / f"{patient_id}{_REPORT_SUFFIX}"
    )


def _prepare_audit_output_path(
    output_root: os.PathLike,
    config: VertebraeInstanceAuditConfig,
    patient_id: str,
) -> Tuple[Path, Path]:
    """Create and validate a real audit directory below output_root."""

    lexical_path = audit_output_path(output_root, config, patient_id)
    resolved_output_root = Path(output_root).resolve(strict=False)
    audit_directory = (
        resolved_output_root / config.output_dir_name
    )
    if audit_directory.is_symlink():
        raise ValueError("audit output directory must not be a symlink")

    audit_directory.mkdir(parents=True, exist_ok=True)
    if audit_directory.is_symlink():
        raise ValueError("audit output directory must not be a symlink")

    resolved_audit_directory = audit_directory.resolve(strict=True)
    try:
        resolved_audit_directory.relative_to(resolved_output_root)
    except ValueError as error:
        raise ValueError(
            "audit output directory escapes the resolved output root"
        ) from error

    report_path = (
        resolved_audit_directory / lexical_path.name
    )
    return report_path, resolved_output_root


def _load_analyzer():
    # Keep the analyzer and its SciPy imports out of disabled batch runs.
    from .vertebrae_instance_analysis import analyze_vertebral_instances

    return analyze_vertebral_instances


def _reference_ct_or_none(
    *,
    reference_img,
    segmentation_dict: Mapping[str, np.ndarray],
    ordered_names: Sequence[str],
    logger,
    patient_id: str,
):
    requested_masks = [
        np.asarray(segmentation_dict[name])
        for name in ordered_names
        if name in segmentation_dict
    ]
    if not requested_masks:
        logger.warning(
            "[ShapeKit][vertebrae-instance-analysis] "
            "CT requested for %s, but no vertebral mask is available "
            "for alignment validation; using geometry-only mode.",
            patient_id,
        )
        return None

    try:
        ct_array = np.asanyarray(reference_img.dataobj)
    except Exception:
        logger.warning(
            "[ShapeKit][vertebrae-instance-analysis] "
            "Failed to load the requested CT reference for %s; "
            "using geometry-only mode.",
            patient_id,
            exc_info=True,
        )
        return None

    expected_shape = requested_masks[0].shape
    try:
        valid_masks = all(
            mask.ndim == 3 and mask.shape == expected_shape
            for mask in requested_masks
        )
        valid_ct = (
            valid_masks
            and ct_array.ndim == 3
            and ct_array.shape == expected_shape
            and bool(np.all(np.isfinite(ct_array)))
        )
    except Exception:
        valid_ct = False
    if not valid_ct:
        logger.warning(
            "[ShapeKit][vertebrae-instance-analysis] "
            "The explicitly requested CT reference for %s does not "
            "satisfy the 3-D shape/finiteness contract; using "
            "geometry-only mode.",
            patient_id,
        )
        return None
    return ct_array


def _canonical_json_bytes(report: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    resolved_output_root: Path,
) -> None:
    if path.parent.is_symlink():
        raise ValueError("audit output directory must not be a symlink")
    resolved_parent = path.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_output_root)
    except ValueError as error:
        raise ValueError(
            "audit output directory escapes the resolved output root"
        ) from error
    if resolved_parent != path.parent:
        raise ValueError("audit output directory must not be a symlink")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(resolved_parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def run_vertebrae_instance_audit(
    segmentation_dict: Mapping[str, np.ndarray],
    *,
    reference_img,
    ordered_anatomical_names: Sequence[str],
    output_root: os.PathLike,
    patient_id: str,
    config: VertebraeInstanceAuditConfig,
    logger,
) -> AuditRunStatus:
    """Run and save an optional audit without affecting segmentation."""

    if not config.enabled:
        return "disabled"

    requested_ct_mode = (
        "reference-as-ct"
        if config.use_reference_as_ct
        else "geometry-only"
    )
    effective_ct_mode = "not-evaluated"
    report_path = None
    try:
        report_path = audit_output_path(
            output_root,
            config,
            patient_id,
        )
        report_path, resolved_output_root = _prepare_audit_output_path(
            output_root,
            config,
            patient_id,
        )
    except Exception:
        logger.exception(
            "[ShapeKit][vertebrae-instance-analysis] "
            "Invalid audit output path for patient=%s, report=%s, "
            "requested_ct_mode=%s, effective_ct_mode=%s; segmentation "
            "postprocessing will continue unchanged.",
            patient_id,
            report_path,
            requested_ct_mode,
            effective_ct_mode,
        )
        return "failed"

    try:
        ct_array = None
        if config.use_reference_as_ct:
            ct_array = _reference_ct_or_none(
                reference_img=reference_img,
                segmentation_dict=segmentation_dict,
                ordered_names=ordered_anatomical_names,
                logger=logger,
                patient_id=patient_id,
            )
        effective_ct_mode = (
            "CT-supported"
            if ct_array is not None
            else "geometry-only"
        )
        analyze_vertebral_instances = _load_analyzer()
        report = analyze_vertebral_instances(
            segmentation_dict,
            affine=reference_img.affine,
            ordered_anatomical_names=ordered_anatomical_names,
            ct=ct_array,
        )
        payload = _canonical_json_bytes(report)
        _atomic_write(
            report_path,
            payload,
            resolved_output_root=resolved_output_root,
        )
        logger.info(
            "[ShapeKit][vertebrae-instance-analysis] "
            "Wrote %s audit for %s to %s.",
            effective_ct_mode,
            patient_id,
            report_path,
        )
        return "written"
    except Exception:
        logger.exception(
            "[ShapeKit][vertebrae-instance-analysis] "
            "Audit failed for patient=%s, report=%s, "
            "requested_ct_mode=%s, effective_ct_mode=%s; segmentation "
            "postprocessing will continue unchanged.",
            patient_id,
            report_path,
            requested_ct_mode,
            effective_ct_mode,
        )
        return "failed"


__all__ = [
    "AuditRunStatus",
    "VertebraeInstanceAuditConfig",
    "audit_output_path",
    "ordered_vertebral_anatomical_names",
    "parse_vertebrae_instance_audit_config",
    "run_vertebrae_instance_audit",
]
