"""Read-only physical vertebral-instance and sequence analysis.

This module deliberately does not post-process segmentations.  It accepts
binary masks keyed by anatomical name and returns a deterministic,
JSON-compatible diagnostic report.  No function in this module writes image
files or returns a corrected label map.

``ordered_anatomical_names`` is always interpreted inferior-to-superior.
Geometry is measured in physical millimetres from the supplied affine.  CT is
optional and, when present, contributes confidence evidence only; it never
changes instance boundaries.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np
from scipy import ndimage
from scipy import signal as scipy_signal


class NameCount(TypedDict):
    anatomical_name: str
    voxel_count: int
    fraction_of_core: float


class VertebralInstanceReport(TypedDict):
    instance_id: str
    rank_inferior_to_superior: int
    centroid_world_mm: List[float]
    core_voxel_count: int
    core_volume_mm3: float
    maximum_internal_thickness_mm: float
    persistence_mm: float
    compactness: float
    trajectory_distance_mm: float
    current_name_composition: List[NameCount]
    protected_core_identity: Optional[str]
    identity_confidence: float
    confidence_mode: Literal["geometry_only", "geometry_and_ct"]
    ct_bone_support_fraction: Optional[float]
    inferior_core_separation_mm: Optional[float]
    superior_core_separation_mm: Optional[float]
    status: Literal[
        "protected_high_confidence",
        "unresolved_low_confidence",
        "unresolved_mixed_identity",
        "unresolved_overlap",
        "unresolved_boundary_truncated",
    ]
    reasons: List[str]


class SequenceAnomalyReport(TypedDict):
    anomaly_code: Literal[
        "duplicate_identity",
        "missing_internal_identity",
        "nonmonotonic_identity",
        "off_trajectory_core",
        "abnormal_spacing",
        "ambiguous_identity",
        "overlapping_input_masks",
        "unsupported_affine",
    ]
    affected_instance_ids: List[str]
    affected_candidate_ids: List[str]
    affected_anatomical_names: List[str]
    status: Literal["detected", "unresolved"]
    explanation: str


class RejectedCandidateReport(TypedDict):
    candidate_id: str
    centroid_world_mm: List[float]
    core_voxel_count: int
    core_volume_mm3: float
    trajectory_distance_mm: float
    status: Literal["rejected_unresolved"]
    reasons: List[str]


class VertebralAnalysisReport(TypedDict):
    schema_version: str
    ordered_names_direction: Literal["inferior_to_superior"]
    shape: List[int]
    spacing_mm: List[float]
    orientation_axcodes: List[str]
    ct_evidence: Literal["used", "unavailable"]
    field_of_view_status: Literal[
        "not_truncated_at_array_boundary",
        "inferior_boundary_truncated",
        "superior_boundary_truncated",
        "both_boundaries_truncated",
        "extent_uncertain",
    ]
    effective_config: Dict[str, float]
    input_overlap_voxel_count: int
    instances: List[VertebralInstanceReport]
    rejected_candidates: List[RejectedCandidateReport]
    observed_sequence_inferior_to_superior: List[Optional[str]]
    anomalies: List[SequenceAnomalyReport]
    overall_status: Literal[
        "continuous_sequence",
        "anomaly_detected",
        "unresolved",
        "empty_input",
    ]


@dataclass(frozen=True)
class VertebralInstanceAnalysisConfig:
    """Configurable physical and confidence thresholds.

    Defaults are conservative starting values for diagnostics and synthetic
    validation.  They are not population-level performance claims.
    """

    trajectory_tube_radius_mm: float = 55.0
    trajectory_outlier_mm: float = 35.0
    core_radius_mm: float = 5.0
    min_core_volume_mm3: float = 120.0
    min_core_persistence_mm: float = 5.0
    max_core_persistence_mm: float = 45.0
    profile_smoothing_mm: float = 2.0
    min_instance_peak_distance_mm: float = 14.0
    min_instance_peak_prominence: float = 0.08
    min_compactness: float = 0.30
    max_trajectory_distance_mm: float = 30.0
    min_label_vote_fraction: float = 0.55
    min_label_vote_margin: float = 0.15
    geometry_confidence_threshold: float = 0.75
    ct_augmented_confidence_threshold: float = 0.70
    bone_hu_threshold: float = 150.0
    min_bone_support_fraction: float = 0.35
    spacing_outlier_mad: float = 3.5
    max_si_axis_obliquity_degrees: float = 20.0


@dataclass
class _Trajectory:
    x_coefficients: np.ndarray
    y_coefficients: np.ndarray

    def point_at_world_z(self, world_z: float) -> np.ndarray:
        return np.asarray(
            [
                np.polyval(self.x_coefficients, world_z),
                np.polyval(self.y_coefficients, world_z),
                world_z,
            ],
            dtype=np.float64,
        )


@dataclass
class _Candidate:
    centroid_world: np.ndarray
    core_voxel_count: int
    core_volume_mm3: float
    maximum_internal_thickness_mm: float
    persistence_mm: float
    compactness: float
    trajectory_distance_mm: float
    composition: List[NameCount]
    proposed_identity: Optional[str]
    confidence: float
    ct_bone_support_fraction: Optional[float]
    status: str
    reasons: List[str]
    min_world_z: float
    max_world_z: float
    center_si_index: float
    inferior_core_separation_mm: Optional[float] = None
    superior_core_separation_mm: Optional[float] = None


@dataclass
class _RejectedCandidate:
    centroid_world: np.ndarray
    core_voxel_count: int
    core_volume_mm3: float
    trajectory_distance_mm: float
    reasons: List[str]


_SCHEMA_VERSION = "1.1"
_ORTHOGONALITY_TOLERANCE = 1e-3
_MAX_SUPPORTED_SI_OBLIQUITY_DEGREES = 20.0
_ROUND_DECIMALS = 6


def _rounded(value: float) -> float:
    return round(float(value), _ROUND_DECIMALS)


def _rounded_vector(values: np.ndarray) -> List[float]:
    return [_rounded(value) for value in np.asarray(values).tolist()]


def _validate_config(config: VertebralInstanceAnalysisConfig) -> None:
    if not np.isfinite(config.bone_hu_threshold):
        raise ValueError(
            "Configuration values must be finite: bone_hu_threshold"
        )
    positive = {
        "trajectory_tube_radius_mm": config.trajectory_tube_radius_mm,
        "trajectory_outlier_mm": config.trajectory_outlier_mm,
        "core_radius_mm": config.core_radius_mm,
        "min_core_volume_mm3": config.min_core_volume_mm3,
        "min_core_persistence_mm": config.min_core_persistence_mm,
        "max_core_persistence_mm": config.max_core_persistence_mm,
        "profile_smoothing_mm": config.profile_smoothing_mm,
        "min_instance_peak_distance_mm": config.min_instance_peak_distance_mm,
        "max_trajectory_distance_mm": config.max_trajectory_distance_mm,
        "spacing_outlier_mad": config.spacing_outlier_mad,
    }
    invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError("Configuration values must be positive: " + ", ".join(invalid))
    fractions = {
        "min_instance_peak_prominence": config.min_instance_peak_prominence,
        "min_compactness": config.min_compactness,
        "min_label_vote_fraction": config.min_label_vote_fraction,
        "min_label_vote_margin": config.min_label_vote_margin,
        "geometry_confidence_threshold": config.geometry_confidence_threshold,
        "ct_augmented_confidence_threshold": config.ct_augmented_confidence_threshold,
        "min_bone_support_fraction": config.min_bone_support_fraction,
    }
    invalid = [
        name
        for name, value in fractions.items()
        if not np.isfinite(value) or value < 0 or value > 1
    ]
    if invalid:
        raise ValueError("Configuration values must lie in [0, 1]: " + ", ".join(invalid))
    if config.max_core_persistence_mm < config.min_core_persistence_mm:
        raise ValueError("max_core_persistence_mm must be >= min_core_persistence_mm")
    if not (
        0
        < config.max_si_axis_obliquity_degrees
        <= _MAX_SUPPORTED_SI_OBLIQUITY_DEGREES
    ):
        raise ValueError(
            "max_si_axis_obliquity_degrees must lie in (0, 20]"
        )


def _validate_affine(
    affine: np.ndarray, config: VertebralInstanceAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray, List[str], int, int, Optional[str]]:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("affine must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("affine contains non-finite values")
    expected_homogeneous_row = np.asarray([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(
        matrix[3, :],
        expected_homogeneous_row,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "affine homogeneous row must be approximately [0, 0, 0, 1]"
        )
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if abs(determinant) < 1e-8:
        raise ValueError("affine is singular")
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0):
        raise ValueError("affine contains a zero-length voxel axis")
    directions = linear / spacing
    gram = directions.T @ directions
    off_diagonal = gram - np.eye(3)
    shear = float(np.max(np.abs(off_diagonal)))
    orientation = _orientation_axcodes(directions)
    si_axis = int(np.argmax(np.abs(directions[2, :])))
    si_sign = 1 if directions[2, si_axis] >= 0 else -1
    alignment = float(np.clip(abs(directions[2, si_axis]), 0.0, 1.0))
    obliquity_degrees = math.degrees(math.acos(alignment))
    unsupported_reason = None
    if shear > _ORTHOGONALITY_TOLERANCE:
        unsupported_reason = (
            "Affine voxel axes are materially non-orthogonal "
            f"(maximum normalized dot product {shear:.6f})."
        )
    elif (
        obliquity_degrees
        > config.max_si_axis_obliquity_degrees + 1e-6
    ):
        unsupported_reason = (
            "No voxel axis is sufficiently aligned with physical superior-inferior "
            f"direction (obliquity {obliquity_degrees:.3f} degrees)."
        )
    return matrix, spacing, orientation, si_axis, si_sign, unsupported_reason


def _orientation_axcodes(directions: np.ndarray) -> List[str]:
    negative = ("L", "P", "I")
    positive = ("R", "A", "S")
    # Assign each voxel axis to a unique world axis.  Independent argmax calls
    # can emit duplicate codes for valid in-plane rotations near 45 degrees.
    assignment = max(
        itertools.permutations(range(3)),
        key=lambda candidate: sum(
            abs(float(directions[candidate[voxel_axis], voxel_axis]))
            for voxel_axis in range(3)
        ),
    )
    codes: List[str] = []
    for voxel_axis in range(3):
        world_axis = assignment[voxel_axis]
        sign_positive = directions[world_axis, voxel_axis] >= 0
        codes.append(positive[world_axis] if sign_positive else negative[world_axis])
    return codes


def _shape_and_masks(
    segmentation_dict: Mapping[str, np.ndarray],
    ordered_names: Sequence[str],
) -> Tuple[List[int], Dict[str, np.ndarray]]:
    requested: Dict[str, np.ndarray] = {}
    shape: Optional[Tuple[int, ...]] = None
    for name in ordered_names:
        if name not in segmentation_dict:
            continue
        array = np.asarray(segmentation_dict[name])
        if array.ndim != 3:
            raise ValueError(f"Mask {name!r} is not three-dimensional")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"Mask {name!r} shape differs from other vertebral masks")
        requested[name] = array
    if shape is None:
        for value in segmentation_dict.values():
            array = np.asarray(value)
            if array.ndim == 3:
                shape = array.shape
                break
    return list(shape) if shape is not None else [], requested


def _apply_affine(affine: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    return coords @ affine[:3, :3].T + affine[:3, 3]


def _largest_component_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    structure = ndimage.generate_binary_structure(3, 1)
    components, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component_id = int(np.argmax(sizes))
    centroid = ndimage.center_of_mass(mask, components, component_id)
    return np.asarray(centroid, dtype=np.float64)


def _fit_trajectory(
    masks: Mapping[str, np.ndarray],
    ordered_names: Sequence[str],
    affine: np.ndarray,
    outlier_mm: float,
) -> Optional[_Trajectory]:
    seeds: List[np.ndarray] = []
    for name in ordered_names:
        mask = masks.get(name)
        if mask is None or not np.any(mask):
            continue
        centroid = _largest_component_centroid(mask != 0)
        if centroid is not None:
            seeds.append(_apply_affine(affine, centroid[None, :])[0])
    if not seeds:
        return None
    points = np.asarray(seeds, dtype=np.float64)
    unique_z = np.unique(np.round(points[:, 2], decimals=6))
    degree = min(2, len(unique_z) - 1)
    if degree <= 0:
        return _Trajectory(
            x_coefficients=np.asarray([float(np.median(points[:, 0]))]),
            y_coefficients=np.asarray([float(np.median(points[:, 1]))]),
        )
    x_coefficients = np.polyfit(points[:, 2], points[:, 0], degree)
    y_coefficients = np.polyfit(points[:, 2], points[:, 1], degree)
    predicted = np.column_stack(
        [
            np.polyval(x_coefficients, points[:, 2]),
            np.polyval(y_coefficients, points[:, 2]),
        ]
    )
    residuals = np.linalg.norm(points[:, :2] - predicted, axis=1)
    keep = residuals <= outlier_mm
    if int(np.count_nonzero(keep)) >= degree + 1 and not np.all(keep):
        kept = points[keep]
        kept_unique_z = np.unique(np.round(kept[:, 2], decimals=6))
        kept_degree = min(degree, len(kept_unique_z) - 1)
        if kept_degree > 0:
            x_coefficients = np.polyfit(kept[:, 2], kept[:, 0], kept_degree)
            y_coefficients = np.polyfit(kept[:, 2], kept[:, 1], kept_degree)
    return _Trajectory(x_coefficients, y_coefficients)


def _bbox(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    low = np.min(coordinates, axis=0)
    high = np.max(coordinates, axis=0) + 1
    return tuple(  # type: ignore[return-value]
        slice(int(low[axis]), int(high[axis])) for axis in range(3)
    )


def _compactness(world: np.ndarray) -> float:
    if len(world) < 5:
        return 0.0
    transverse = np.asarray(world[:, :2], dtype=np.float64)
    covariance = np.cov(transverse, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[-1] <= 1e-8:
        return 0.0
    return float(math.sqrt(max(float(eigenvalues[0]), 0.0) / float(eigenvalues[-1])))


def _clip_ratio(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return float(np.clip(value / reference, 0.0, 1.0))


def _confidence(
    *,
    max_radius_mm: float,
    volume_mm3: float,
    persistence_mm: float,
    compactness: float,
    trajectory_distance_mm: float,
    vote_fraction: float,
    vote_margin: float,
    bone_support: Optional[float],
    config: VertebralInstanceAnalysisConfig,
) -> float:
    thickness_score = _clip_ratio(
        max_radius_mm - config.core_radius_mm,
        config.core_radius_mm,
    )
    volume_score = _clip_ratio(volume_mm3, 2.0 * config.min_core_volume_mm3)
    persistence_score = _clip_ratio(
        persistence_mm, config.min_core_persistence_mm
    )
    compactness_score = _clip_ratio(compactness, 2.0 * config.min_compactness)
    trajectory_score = 1.0 - _clip_ratio(
        trajectory_distance_mm, config.max_trajectory_distance_mm
    )
    vote_score = 0.65 * vote_fraction + 0.35 * _clip_ratio(
        vote_margin, config.min_label_vote_margin
    )
    geometry = (
        0.15 * thickness_score
        + 0.20 * volume_score
        + 0.15 * persistence_score
        + 0.15 * compactness_score
        + 0.15 * trajectory_score
        + 0.20 * vote_score
    )
    if bone_support is None:
        return float(np.clip(geometry, 0.0, 1.0))
    bone_score = _clip_ratio(bone_support, config.min_bone_support_fraction)
    return float(np.clip(0.85 * geometry + 0.15 * bone_score, 0.0, 1.0))


def _profile(
    thick: np.ndarray,
    spacing: np.ndarray,
    si_axis: int,
    config: VertebralInstanceAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    transverse_axes = tuple(axis for axis in range(3) if axis != si_axis)
    area = np.count_nonzero(thick, axis=transverse_axes).astype(np.float64)
    scale = max(float(np.percentile(area, 95)), 1.0)
    normalized = np.clip(area / scale, 0.0, 1.5)
    sigma = config.profile_smoothing_mm / max(float(spacing[si_axis]), 1e-6)
    smoothed = ndimage.gaussian_filter1d(
        normalized, sigma=max(sigma, 0.5), mode="nearest"
    )
    minimum_distance = max(
        1,
        int(
            round(
                config.min_instance_peak_distance_mm
                / max(float(spacing[si_axis]), 1e-6)
            )
        ),
    )
    peaks, _ = scipy_signal.find_peaks(
        smoothed,
        distance=minimum_distance,
        prominence=config.min_instance_peak_prominence,
    )
    return smoothed, np.asarray(sorted(int(peak) for peak in peaks), dtype=int)


def _core_valley_quality(
    profile: np.ndarray, left_index: float, right_index: float
) -> float:
    low = int(round(min(left_index, right_index)))
    high = int(round(max(left_index, right_index)))
    low = max(low, 0)
    high = min(high, len(profile) - 1)
    if high <= low:
        return 0.0
    valley = float(np.min(profile[low : high + 1]))
    endpoint = min(float(profile[low]), float(profile[high]))
    if endpoint <= 1e-8:
        return 0.0
    return float(np.clip(1.0 - valley / endpoint, 0.0, 1.0))


def _field_of_view_status(
    union: np.ndarray,
    si_axis: int,
    si_sign: int,
    protected_identities: Sequence[Optional[str]],
    ordered_names: Sequence[str],
) -> str:
    inferior_touch, superior_touch = _si_boundary_touches(
        union, si_axis, si_sign
    )
    if inferior_touch and superior_touch:
        return "both_boundaries_truncated"
    if inferior_touch:
        return "inferior_boundary_truncated"
    if superior_touch:
        return "superior_boundary_truncated"
    confident = [name for name in protected_identities if name is not None]
    if confident and ordered_names:
        order = {name: index for index, name in enumerate(ordered_names)}
        observed_indices = [order[name] for name in confident]
        if min(observed_indices) > 0 or max(observed_indices) < len(ordered_names) - 1:
            return "extent_uncertain"
    return "not_truncated_at_array_boundary"


def _si_boundary_touches(
    union: np.ndarray, si_axis: int, si_sign: int
) -> Tuple[bool, bool]:
    inferior_index = 0 if si_sign > 0 else union.shape[si_axis] - 1
    superior_index = union.shape[si_axis] - 1 if si_sign > 0 else 0
    inferior_touch = bool(
        np.any(np.take(union, indices=inferior_index, axis=si_axis))
    )
    superior_touch = bool(
        np.any(np.take(union, indices=superior_index, axis=si_axis))
    )
    return inferior_touch, superior_touch


def _anomaly(
    code: str,
    instance_ids: Sequence[str],
    names: Sequence[str],
    status: str,
    explanation: str,
    candidate_ids: Sequence[str] = (),
) -> SequenceAnomalyReport:
    return {
        "anomaly_code": code,  # type: ignore[typeddict-item]
        "affected_instance_ids": list(instance_ids),
        "affected_candidate_ids": list(candidate_ids),
        "affected_anatomical_names": list(names),
        "status": status,  # type: ignore[typeddict-item]
        "explanation": explanation,
    }


def _empty_report(
    *,
    shape: Sequence[int],
    spacing: np.ndarray,
    orientation: Sequence[str],
    ct_used: bool,
    config: VertebralInstanceAnalysisConfig,
    anomaly: Optional[SequenceAnomalyReport] = None,
) -> VertebralAnalysisReport:
    anomalies = [] if anomaly is None else [anomaly]
    return {
        "schema_version": _SCHEMA_VERSION,
        "ordered_names_direction": "inferior_to_superior",
        "shape": [int(value) for value in shape],
        "spacing_mm": _rounded_vector(spacing),
        "orientation_axcodes": list(orientation),
        "ct_evidence": "used" if ct_used else "unavailable",
        "field_of_view_status": "extent_uncertain",
        "effective_config": {
            key: _rounded(value) for key, value in asdict(config).items()
        },
        "input_overlap_voxel_count": 0,
        "instances": [],
        "rejected_candidates": [],
        "observed_sequence_inferior_to_superior": [],
        "anomalies": anomalies,
        "overall_status": "unresolved" if anomalies else "empty_input",
    }


def analyze_vertebral_instances(
    segmentation_dict: Mapping[str, np.ndarray],
    *,
    affine: np.ndarray,
    ordered_anatomical_names: Sequence[str],
    ct: Optional[np.ndarray] = None,
    config: Optional[VertebralInstanceAnalysisConfig] = None,
) -> VertebralAnalysisReport:
    """Return a deterministic read-only vertebral-instance audit.

    The returned object contains JSON-compatible primitives only.  It never
    contains segmentation arrays, voxel-index lists, or proposed corrections.
    """

    effective = config or VertebralInstanceAnalysisConfig()
    _validate_config(effective)
    names = tuple(str(name) for name in ordered_anatomical_names)
    if not names:
        raise ValueError("ordered_anatomical_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("ordered_anatomical_names must contain unique names")

    (
        affine_array,
        spacing,
        orientation,
        si_axis,
        si_sign,
        unsupported_reason,
    ) = _validate_affine(affine, effective)
    shape, arrays = _shape_and_masks(segmentation_dict, names)
    if ct is not None:
        ct_array = np.asarray(ct)
        if ct_array.ndim != 3:
            raise ValueError("ct must be a three-dimensional array")
        if shape and list(ct_array.shape) != shape:
            raise ValueError("ct shape differs from vertebral masks")
        if not np.all(np.isfinite(ct_array)):
            raise ValueError("ct contains non-finite values")
    else:
        ct_array = None

    if unsupported_reason is not None:
        return _empty_report(
            shape=shape,
            spacing=spacing,
            orientation=orientation,
            ct_used=ct_array is not None,
            config=effective,
            anomaly=_anomaly(
                "unsupported_affine",
                [],
                [],
                "unresolved",
                unsupported_reason,
            ),
        )
    if not shape:
        return _empty_report(
            shape=[],
            spacing=spacing,
            orientation=orientation,
            ct_used=ct_array is not None,
            config=effective,
        )

    occupancy = np.zeros(tuple(shape), dtype=np.uint16)
    binary_masks: Dict[str, np.ndarray] = {}
    for name in names:
        array = arrays.get(name)
        if array is None:
            continue
        mask = np.asarray(array != 0)
        binary_masks[name] = mask
        occupancy += mask
    union = occupancy > 0
    overlap = occupancy > 1
    overlap_count = int(np.count_nonzero(overlap))
    if not np.any(union):
        report = _empty_report(
            shape=shape,
            spacing=spacing,
            orientation=orientation,
            ct_used=ct_array is not None,
            config=effective,
        )
        report["input_overlap_voxel_count"] = overlap_count
        return report

    trajectory = _fit_trajectory(
        binary_masks,
        names,
        affine_array,
        effective.trajectory_outlier_mm,
    )
    if trajectory is None:
        return _empty_report(
            shape=shape,
            spacing=spacing,
            orientation=orientation,
            ct_used=ct_array is not None,
            config=effective,
            anomaly=_anomaly(
                "ambiguous_identity",
                [],
                [],
                "unresolved",
                "No trajectory seeds could be estimated from the named masks.",
            ),
        )

    crop = _bbox(union)
    starts = np.asarray([axis_slice.start or 0 for axis_slice in crop], dtype=int)
    union_crop = np.asarray(union[crop])
    distance = ndimage.distance_transform_edt(union_crop, sampling=spacing)
    thick = distance >= effective.core_radius_mm
    profile, peaks = _profile(thick, spacing, si_axis, effective)
    structure = ndimage.generate_binary_structure(3, 1)
    component_map, component_count = ndimage.label(thick, structure=structure)
    component_slices = ndimage.find_objects(
        component_map,
        max_label=component_count,
    )
    voxel_volume = float(abs(np.linalg.det(affine_array[:3, :3])))
    name_order = {name: index for index, name in enumerate(names)}
    candidates: List[_Candidate] = []
    rejected_candidates: List[_RejectedCandidate] = []

    for component_id, component_slice in enumerate(
        component_slices,
        start=1,
    ):
        if component_slice is None:
            continue
        component_starts = np.asarray(
            [axis_slice.start or 0 for axis_slice in component_slice],
            dtype=int,
        )
        component_region = component_map[component_slice]
        component_local_coords = np.argwhere(
            component_region == component_id
        )
        local_coords = component_local_coords + component_starts
        if not len(local_coords):
            continue
        global_coords = local_coords + starts
        world = _apply_affine(affine_array, global_coords)
        centroid_world = np.mean(world, axis=0)
        trajectory_point = trajectory.point_at_world_z(float(centroid_world[2]))
        trajectory_distance = float(
            np.linalg.norm(centroid_world[:2] - trajectory_point[:2])
        )
        count = int(len(global_coords))
        volume = count * voxel_volume
        if trajectory_distance > effective.trajectory_tube_radius_mm:
            rejected_candidates.append(
                _RejectedCandidate(
                    centroid_world=centroid_world,
                    core_voxel_count=count,
                    core_volume_mm3=volume,
                    trajectory_distance_mm=trajectory_distance,
                    reasons=["core_outside_trajectory_tube"],
                )
            )
            continue

        component_distance = distance[tuple(local_coords.T)]
        max_radius = float(np.max(component_distance))
        # Include one voxel's physical SI extent.  A coordinate range measures
        # center-to-center distance and otherwise underestimates persistence
        # more severely on coarse/anisotropic grids.
        persistence = float(
            np.ptp(world[:, 2])
            + np.sum(np.abs(affine_array[2, :3]))
        )
        compactness = _compactness(world)
        core_overlap = bool(np.any(overlap[tuple(global_coords.T)]))
        composition: List[NameCount] = []
        counts: List[Tuple[str, int]] = []
        for name in names:
            mask = binary_masks.get(name)
            if mask is None:
                continue
            name_count = int(np.count_nonzero(mask[tuple(global_coords.T)]))
            if name_count:
                counts.append((name, name_count))
                composition.append(
                    {
                        "anatomical_name": name,
                        "voxel_count": name_count,
                        "fraction_of_core": _rounded(name_count / count),
                    }
                )
        counts.sort(key=lambda item: (-item[1], name_order[item[0]], item[0]))
        composition.sort(
            key=lambda item: (
                name_order[item["anatomical_name"]],
                item["anatomical_name"],
            )
        )
        winner = counts[0][0] if counts else None
        winner_count = counts[0][1] if counts else 0
        runner_count = counts[1][1] if len(counts) > 1 else 0
        vote_fraction = winner_count / count if count else 0.0
        vote_margin = (winner_count - runner_count) / count if count else 0.0

        bone_support: Optional[float]
        if ct_array is None:
            bone_support = None
        else:
            ct_values = ct_array[tuple(global_coords.T)]
            bone_support = float(
                np.count_nonzero(ct_values >= effective.bone_hu_threshold) / count
            )
        confidence = _confidence(
            max_radius_mm=max_radius,
            volume_mm3=volume,
            persistence_mm=persistence,
            compactness=compactness,
            trajectory_distance_mm=trajectory_distance,
            vote_fraction=vote_fraction,
            vote_margin=vote_margin,
            bone_support=bone_support,
            config=effective,
        )
        reasons: List[str] = []
        if volume < effective.min_core_volume_mm3:
            reasons.append("core_volume_below_threshold")
        if persistence < effective.min_core_persistence_mm:
            reasons.append("core_persistence_below_threshold")
        if persistence > effective.max_core_persistence_mm:
            reasons.append("possible_merged_instance")
        if compactness < effective.min_compactness:
            reasons.append("core_compactness_below_threshold")
        if trajectory_distance > effective.max_trajectory_distance_mm:
            reasons.append("core_far_from_trajectory")
        if winner is None:
            reasons.append("no_anatomical_name_vote")
        if vote_fraction < effective.min_label_vote_fraction:
            reasons.append("mixed_identity_vote_fraction")
        if vote_margin < effective.min_label_vote_margin:
            reasons.append("mixed_identity_vote_margin")
        if bone_support is not None and bone_support < effective.min_bone_support_fraction:
            reasons.append("low_ct_bone_support")
        local_si_min = int(np.min(local_coords[:, si_axis]))
        local_si_max = int(np.max(local_coords[:, si_axis]))
        contained_peaks = peaks[
            (peaks >= local_si_min) & (peaks <= local_si_max)
        ]
        if len(contained_peaks) > 1:
            reasons.append("multiple_body_profile_peaks_in_one_core")

        threshold = (
            effective.geometry_confidence_threshold
            if ct_array is None
            else effective.ct_augmented_confidence_threshold
        )
        if core_overlap:
            status = "unresolved_overlap"
            reasons.append("overlapping_input_masks")
        elif (
            "mixed_identity_vote_fraction" in reasons
            or "mixed_identity_vote_margin" in reasons
            or "no_anatomical_name_vote" in reasons
        ):
            status = "unresolved_mixed_identity"
        elif reasons or confidence < threshold:
            status = "unresolved_low_confidence"
            if confidence < threshold:
                reasons.append("confidence_below_threshold")
        else:
            status = "protected_high_confidence"

        candidates.append(
            _Candidate(
                centroid_world=centroid_world,
                core_voxel_count=count,
                core_volume_mm3=volume,
                maximum_internal_thickness_mm=2.0 * max_radius,
                persistence_mm=persistence,
                compactness=compactness,
                trajectory_distance_mm=trajectory_distance,
                composition=composition,
                proposed_identity=winner if status == "protected_high_confidence" else None,
                confidence=confidence,
                ct_bone_support_fraction=bone_support,
                status=status,
                reasons=sorted(set(reasons)),
                min_world_z=float(np.min(world[:, 2])),
                max_world_z=float(np.max(world[:, 2])),
                center_si_index=float(np.mean(local_coords[:, si_axis])),
            )
        )

    candidates.sort(
        key=lambda item: (
            float(item.centroid_world[2]),
            float(item.centroid_world[1]),
            float(item.centroid_world[0]),
        )
    )
    inferior_touch, superior_touch = _si_boundary_touches(
        union, si_axis, si_sign
    )
    boundary_candidates: List[Tuple[_Candidate, str]] = []
    if candidates and inferior_touch:
        boundary_candidates.append((candidates[0], "inferior"))
    if candidates and superior_touch:
        boundary_candidates.append((candidates[-1], "superior"))
    for candidate, boundary_name in boundary_candidates:
        candidate.status = "unresolved_boundary_truncated"
        candidate.proposed_identity = None
        candidate.reasons = sorted(
            set(
                candidate.reasons
                + [f"physical_instance_touches_{boundary_name}_scan_boundary"]
            )
        )

    for index in range(len(candidates) - 1):
        inferior = candidates[index]
        superior = candidates[index + 1]
        core_separation = max(
            0.0,
            superior.min_world_z - inferior.max_world_z,
        )
        inferior.superior_core_separation_mm = core_separation
        superior.inferior_core_separation_mm = core_separation
        quality = _core_valley_quality(
            profile, inferior.center_si_index, superior.center_si_index
        )
        if quality < effective.min_instance_peak_prominence:
            for candidate in (inferior, superior):
                if candidate.status == "protected_high_confidence":
                    candidate.status = "unresolved_low_confidence"
                    candidate.proposed_identity = None
                candidate.reasons = sorted(
                    set(
                        candidate.reasons
                        + ["weak_core_separation_evidence"]
                    )
                )

    instance_reports: List[VertebralInstanceReport] = []
    for rank, candidate in enumerate(candidates, start=1):
        instance_reports.append(
            {
                "instance_id": f"instance_{rank:03d}",
                "rank_inferior_to_superior": rank,
                "centroid_world_mm": _rounded_vector(candidate.centroid_world),
                "core_voxel_count": candidate.core_voxel_count,
                "core_volume_mm3": _rounded(candidate.core_volume_mm3),
                "maximum_internal_thickness_mm": _rounded(
                    candidate.maximum_internal_thickness_mm
                ),
                "persistence_mm": _rounded(candidate.persistence_mm),
                "compactness": _rounded(candidate.compactness),
                "trajectory_distance_mm": _rounded(
                    candidate.trajectory_distance_mm
                ),
                "current_name_composition": candidate.composition,
                "protected_core_identity": candidate.proposed_identity,
                "identity_confidence": _rounded(candidate.confidence),
                "confidence_mode": (
                    "geometry_only" if ct_array is None else "geometry_and_ct"
                ),
                "ct_bone_support_fraction": (
                    None
                    if candidate.ct_bone_support_fraction is None
                    else _rounded(candidate.ct_bone_support_fraction)
                ),
                "inferior_core_separation_mm": (
                    None
                    if candidate.inferior_core_separation_mm is None
                    else _rounded(candidate.inferior_core_separation_mm)
                ),
                "superior_core_separation_mm": (
                    None
                    if candidate.superior_core_separation_mm is None
                    else _rounded(candidate.superior_core_separation_mm)
                ),
                "status": candidate.status,  # type: ignore[typeddict-item]
                "reasons": candidate.reasons,
            }
        )

    rejected_candidates.sort(
        key=lambda item: (
            float(item.centroid_world[2]),
            float(item.centroid_world[1]),
            float(item.centroid_world[0]),
            item.core_voxel_count,
        )
    )
    rejected_candidate_reports: List[RejectedCandidateReport] = []
    for rank, candidate in enumerate(rejected_candidates, start=1):
        rejected_candidate_reports.append(
            {
                "candidate_id": f"rejected_candidate_{rank:03d}",
                "centroid_world_mm": _rounded_vector(
                    candidate.centroid_world
                ),
                "core_voxel_count": candidate.core_voxel_count,
                "core_volume_mm3": _rounded(candidate.core_volume_mm3),
                "trajectory_distance_mm": _rounded(
                    candidate.trajectory_distance_mm
                ),
                "status": "rejected_unresolved",
                "reasons": candidate.reasons,
            }
        )

    identities = [item["protected_core_identity"] for item in instance_reports]
    fov_status = _field_of_view_status(
        union, si_axis, si_sign, identities, names
    )
    anomalies: List[SequenceAnomalyReport] = []
    if rejected_candidate_reports:
        anomalies.append(
            _anomaly(
                "off_trajectory_core",
                [],
                [],
                "unresolved",
                "One or more thick-core candidates lie outside the "
                "configured trajectory tube and were not accepted as "
                "vertebral instances.",
                candidate_ids=[
                    item["candidate_id"]
                    for item in rejected_candidate_reports
                ],
            )
        )
    if overlap_count:
        affected = [
            item["instance_id"]
            for item in instance_reports
            if item["status"] == "unresolved_overlap"
        ]
        anomalies.append(
            _anomaly(
                "overlapping_input_masks",
                affected,
                [],
                "unresolved",
                f"{overlap_count} voxels belong to more than one anatomical-name mask.",
            )
        )

    identity_to_instances: Dict[str, List[str]] = {}
    for item in instance_reports:
        identity = item["protected_core_identity"]
        if identity is not None:
            identity_to_instances.setdefault(identity, []).append(item["instance_id"])
    for identity in names:
        matching = identity_to_instances.get(identity, [])
        if len(matching) > 1:
            anomalies.append(
                _anomaly(
                    "duplicate_identity",
                    matching,
                    [identity],
                    "detected",
                    f"Multiple high-confidence physical cores vote for {identity}.",
                )
            )

    for left, right in zip(instance_reports, instance_reports[1:]):
        left_name = left["protected_core_identity"]
        right_name = right["protected_core_identity"]
        if left_name is None or right_name is None:
            continue
        left_index = name_order[left_name]
        right_index = name_order[right_name]
        if right_index < left_index:
            anomalies.append(
                _anomaly(
                    "nonmonotonic_identity",
                    [left["instance_id"], right["instance_id"]],
                    [left_name, right_name],
                    "detected",
                    "Confident identities decrease while physical instances move superiorly.",
                )
            )
        elif right_index - left_index > 1:
            missing = list(names[left_index + 1 : right_index])
            anomalies.append(
                _anomaly(
                    "missing_internal_identity",
                    [left["instance_id"], right["instance_id"]],
                    missing,
                    "detected",
                    "Confident neighboring physical cores skip internal anatomical names.",
                )
            )

    sequence_codes = {item["anomaly_code"] for item in anomalies}
    if (
        "duplicate_identity" in sequence_codes
        and "missing_internal_identity" in sequence_codes
    ):
        anomalies.append(
            _anomaly(
                "ambiguous_identity",
                [item["instance_id"] for item in instance_reports],
                [
                    name
                    for name in identities
                    if name is not None
                ],
                "unresolved",
                "Combined duplicate and missing identities may reflect "
                "transitional anatomy or another nonstandard sequence.",
            )
        )

    unresolved_instances = [
        item for item in instance_reports if item["status"] != "protected_high_confidence"
    ]
    for item in unresolved_instances:
        anomalies.append(
            _anomaly(
                "ambiguous_identity",
                [item["instance_id"]],
                [
                    entry["anatomical_name"]
                    for entry in item["current_name_composition"]
                ],
                "unresolved",
                "Physical core identity remains unresolved: "
                + ", ".join(item["reasons"]),
            )
        )
    if union.any() and not instance_reports:
        anomalies.append(
            _anomaly(
                "ambiguous_identity",
                [],
                [],
                "unresolved",
                "Vertebral foreground exists but no thick core passed candidate selection.",
            )
        )

    if len(instance_reports) >= 5:
        centroids = np.asarray(
            [item["centroid_world_mm"] for item in instance_reports], dtype=np.float64
        )
        spacings = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        median = float(np.median(spacings))
        mad = float(np.median(np.abs(spacings - median)))
        scale = max(1.4826 * mad, 1.0)
        for index, value in enumerate(spacings):
            if abs(float(value) - median) > effective.spacing_outlier_mad * scale:
                left = instance_reports[index]
                right = instance_reports[index + 1]
                anomalies.append(
                    _anomaly(
                        "abnormal_spacing",
                        [left["instance_id"], right["instance_id"]],
                        [
                            name
                            for name in (
                                left["protected_core_identity"],
                                right["protected_core_identity"],
                            )
                            if name is not None
                        ],
                        "unresolved",
                        f"Inter-core spacing {_rounded(value)} mm is a robust local outlier.",
                    )
                )

    anomaly_order = {
        "abnormal_spacing": 0,
        "ambiguous_identity": 1,
        "duplicate_identity": 2,
        "missing_internal_identity": 3,
        "nonmonotonic_identity": 4,
        "off_trajectory_core": 5,
        "overlapping_input_masks": 6,
        "unsupported_affine": 7,
    }
    anomalies.sort(
        key=lambda item: (
            anomaly_order[item["anomaly_code"]],
            item["affected_instance_ids"],
            item["affected_anatomical_names"],
        )
    )
    if any(item["status"] == "unresolved" for item in anomalies):
        overall_status = "unresolved"
    elif any(item["status"] == "detected" for item in anomalies):
        overall_status = "anomaly_detected"
    elif fov_status != "not_truncated_at_array_boundary":
        overall_status = "unresolved"
    else:
        overall_status = "continuous_sequence"

    return {
        "schema_version": _SCHEMA_VERSION,
        "ordered_names_direction": "inferior_to_superior",
        "shape": shape,
        "spacing_mm": _rounded_vector(spacing),
        "orientation_axcodes": orientation,
        "ct_evidence": "used" if ct_array is not None else "unavailable",
        "field_of_view_status": fov_status,  # type: ignore[typeddict-item]
        "effective_config": {
            key: _rounded(value) for key, value in asdict(effective).items()
        },
        "input_overlap_voxel_count": overlap_count,
        "instances": instance_reports,
        "rejected_candidates": rejected_candidate_reports,
        "observed_sequence_inferior_to_superior": identities,
        "anomalies": anomalies,
        "overall_status": overall_status,  # type: ignore[typeddict-item]
    }


__all__ = [
    "VertebralInstanceAnalysisConfig",
    "VertebralAnalysisReport",
    "VertebralInstanceReport",
    "RejectedCandidateReport",
    "SequenceAnomalyReport",
    "analyze_vertebral_instances",
]
