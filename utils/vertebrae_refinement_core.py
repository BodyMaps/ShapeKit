from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def connected_components(binary, connectivity=26, out_dtype=np.uint32):
    if connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=bool)
    elif connectivity == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    else:
        structure = ndimage.generate_binary_structure(3, 2)
    labels, _ = ndimage.label(np.asarray(binary, dtype=bool), structure=structure)
    return labels.astype(out_dtype, copy=False)


def component_statistics(labels):
    labels = np.asarray(labels)
    count = int(labels.max())
    voxel_counts = np.bincount(labels.ravel(), minlength=count + 1)
    objects = ndimage.find_objects(labels, max_label=count)
    bounding_boxes = [tuple((slice(0, int(n)) for n in labels.shape))]
    bounding_boxes.extend(
        (
            item if item is not None else tuple((slice(0, 0) for _ in labels.shape))
            for item in objects
        )
    )
    centroids = np.zeros((count + 1, labels.ndim), dtype=float)
    if count:
        centers = ndimage.center_of_mass(
            np.ones(labels.shape, dtype=np.uint8), labels, range(1, count + 1)
        )
        centroids[1:] = np.asarray(centers, dtype=float)
    return {
        "voxel_counts": voxel_counts,
        "bounding_boxes": bounding_boxes,
        "centroids": centroids,
    }


LABEL_NAMES: Dict[int, str] = {
    **{i: f"vertebrae_L{6 - i}" for i in range(1, 6)},
    **{i: f"vertebrae_T{18 - i}" for i in range(6, 18)},
    **{i: f"vertebrae_C{25 - i}" for i in range(18, 25)},
}


@dataclass
class Component:
    raw_label: int
    component_id: int
    voxel_count: int
    volume_mm3: float
    centroid_voxel: np.ndarray
    centroid_world: np.ndarray
    bbox: Tuple[slice, slice, slice]


def _io_validate_labels(data: np.ndarray, path: Path) -> None:
    values = np.unique(data)
    if values.size and (values.min() < 0 or values.max() > 24):
        raise ValueError(f"Expected labels 0..24 in {path}, found {values.tolist()}")


def _io_save_like(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), reference.affine, header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None:
        image.set_sform(sform, int(scode))
    nib.save(image, str(path))


def _io_save_case(
    data: np.ndarray, reference: nib.Nifti1Image, output_case: Path
) -> None:
    _io_save_like(data, reference, output_case / "combined_labels.nii.gz")
    for label, name in LABEL_NAMES.items():
        _io_save_like(
            data == label, reference, output_case / "segmentations" / f"{name}.nii.gz"
        )


def _io_iter_case_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "combined_labels.nii.gz").exists():
            yield path


def _io_bbox_from_points(points: np.ndarray) -> Tuple[slice, slice, slice]:
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    return tuple((slice(int(a), int(b)) for a, b in zip(lower, upper)))


def _io_extract_components(
    segmentation: np.ndarray,
    affine: np.ndarray,
    labels: Sequence[int] = tuple(range(1, 25)),
) -> Dict[int, List[Component]]:
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    foreground_tuple = np.nonzero(segmentation)
    foreground_points = np.column_stack(foreground_tuple)
    foreground_labels = segmentation[foreground_tuple]
    output: Dict[int, List[Component]] = {}
    for label in labels:
        points = foreground_points[foreground_labels == label]
        if points.size == 0:
            continue
        outer = _io_bbox_from_points(points)
        connected = connected_components(
            np.asarray(segmentation[outer] == label, dtype=np.uint8),
            connectivity=26,
            out_dtype=np.uint32,
        )
        stats = component_statistics(connected)
        outer_start = np.asarray([part.start for part in outer], dtype=float)
        components: List[Component] = []
        for component_id, count in enumerate(stats["voxel_counts"][1:], start=1):
            if not count:
                continue
            local_bbox = stats["bounding_boxes"][component_id]
            bbox = tuple(
                (
                    slice(int(a.start + b.start), int(a.start + b.stop))
                    for a, b in zip(outer, local_bbox)
                )
            )
            centroid_voxel = outer_start + np.asarray(
                stats["centroids"][component_id], dtype=float
            )
            centroid_world = nib.affines.apply_affine(affine, centroid_voxel)
            components.append(
                Component(
                    raw_label=int(label),
                    component_id=int(component_id),
                    voxel_count=int(count),
                    volume_mm3=float(count * voxel_volume),
                    centroid_voxel=centroid_voxel,
                    centroid_world=np.asarray(centroid_world, dtype=float),
                    bbox=bbox,
                )
            )
        components.sort(key=lambda item: (-item.voxel_count, item.component_id))
        output[int(label)] = components
    return output


def _io_component_mask(segmentation: np.ndarray, component: Component) -> np.ndarray:
    local = np.asarray(
        segmentation[component.bbox] == component.raw_label, dtype=np.uint8
    )
    connected = connected_components(local, connectivity=26, out_dtype=np.uint32)
    identifiers = np.unique(connected)
    identifiers = identifiers[identifiers > 0]
    if not identifiers.size:
        return np.zeros(local.shape, dtype=bool)
    stats = component_statistics(connected)
    starts = np.asarray([part.start for part in component.bbox], dtype=float)
    expected = component.centroid_voxel - starts
    target = min(
        (int(value) for value in identifiers),
        key=lambda value: float(
            np.linalg.norm(np.asarray(stats["centroids"][value]) - expected)
        ),
    )
    return connected == target


def _io_robust_spacing_mm(components: Iterable[Component]) -> float:
    z_values = np.asarray(
        sorted((item.centroid_world[2] for item in components)), dtype=float
    )
    gaps = np.diff(z_values)
    plausible = gaps[(gaps >= 8.0) & (gaps <= 60.0)]
    return float(np.median(plausible)) if plausible.size else 25.0


@dataclass
class IdentityConfig:
    fragment_min_mm3: float = 10.0
    seed_min_mm3: float = 1000.0
    seed_separation_fraction: float = 0.42
    assignment_z_fraction: float = 0.55
    assignment_transverse_mm: float = 70.0
    min_sequence_matches: int = 18
    min_endpoint_support: float = 0.55
    min_span_in_spacing_units: float = 18.0
    max_span_in_spacing_units: float = 30.0
    max_merge_gap_ratio: float = 0.65
    very_close_merge_gap_ratio: float = 0.4
    max_merge_transverse_mm: float = 35.0
    max_merged_volume_ratio: float = 1.65
    small_piece_volume_ratio: float = 0.7
    min_hypothesis_margin: float = 0.75
    max_component_label_distance: int = 1
    label_distance_weight: float = 2.0
    exact_support_weight: float = 1.0
    geometry_weight: float = 0.75
    merge_weight: float = 0.5


@dataclass
class VertebralInstance:
    components: List[Component] = field(default_factory=list)
    centroid_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    volume_mm3: float = 0.0
    raw_votes: np.ndarray = field(default_factory=lambda: np.zeros(25, dtype=float))

    def update(self) -> None:
        weights = np.asarray([item.volume_mm3 for item in self.components], dtype=float)
        points = np.asarray(
            [item.centroid_world for item in self.components], dtype=float
        )
        self.volume_mm3 = float(weights.sum())
        self.centroid_world = np.average(points, axis=0, weights=weights)
        self.raw_votes = np.zeros(25, dtype=float)
        for component in self.components:
            self.raw_votes[component.raw_label] += component.volume_mm3

    @property
    def dominant_label(self) -> int:
        return int(np.argmax(self.raw_votes[1:]) + 1)

    def support(self, label: int) -> float:
        return float(self.raw_votes[label] / max(self.raw_votes.sum(), 1e-08))


@dataclass
class MergeEvidence:
    boundary: int
    left_index: int
    right_index: int
    gap_mm: float
    gap_ratio: float
    transverse_mm: float
    left_volume_ratio: float
    right_volume_ratio: float
    merged_volume_ratio: float
    same_dominant_label: bool
    close_enough: bool
    transverse_consistent: bool
    volume_consistent: bool
    fragment_evidence: bool
    supported: bool


@dataclass
class IdentityHypothesis:
    removed_boundaries: Tuple[int, ...]
    groups: List[Tuple[int, ...]]
    target_length_compatible: bool
    score: Optional[float]
    score_breakdown: Dict[str, float]
    raw_path: List[int]
    match_count: Optional[int]
    maximum_dominant_label_distance: Optional[int]
    endpoint_support: Optional[Tuple[float, float]]
    span_in_spacing_units: Optional[float]
    merge_evidence: List[MergeEvidence]
    all_merges_supported: bool


def _identity_build_instances_without_merging(
    components_by_label: Dict[int, List[Component]], config: IdentityConfig
) -> Tuple[List[VertebralInstance], List[Component], float]:
    components = [
        item
        for values in components_by_label.values()
        for item in values
        if item.volume_mm3 >= config.fragment_min_mm3
    ]
    if not components:
        return ([], [], 25.0)
    dominant = [values[0] for values in components_by_label.values() if values]
    spacing = _io_robust_spacing_mm(dominant)
    separation = max(8.0, config.seed_separation_fraction * spacing)
    dominant_ids = {id(item) for item in dominant}
    candidates = sorted(
        {
            id(item): item
            for item in components
            if item.volume_mm3 >= config.seed_min_mm3 or id(item) in dominant_ids
        }.values(),
        key=lambda item: -item.volume_mm3,
    )
    seeds: List[Component] = []
    for component in candidates:
        if all(
            (
                abs(component.centroid_world[2] - seed.centroid_world[2]) >= separation
                for seed in seeds
            )
        ):
            seeds.append(component)
    instances = [VertebralInstance(components=[seed]) for seed in seeds]
    for instance in instances:
        instance.update()
    seeded_ids = {id(item) for item in seeds}
    unassigned: List[Component] = []
    for component in sorted(components, key=lambda item: -item.volume_mm3):
        if id(component) in seeded_ids:
            continue
        distances = []
        for index, instance in enumerate(instances):
            dz = abs(component.centroid_world[2] - instance.centroid_world[2])
            dxy = float(
                np.linalg.norm(
                    component.centroid_world[:2] - instance.centroid_world[:2]
                )
            )
            score = (dz / max(spacing, 1.0)) ** 2 + (
                dxy / max(config.assignment_transverse_mm, 1.0)
            ) ** 2
            distances.append((score, dz, dxy, index))
        if not distances:
            unassigned.append(component)
            continue
        _, dz, dxy, best_index = min(distances)
        if (
            dz <= config.assignment_z_fraction * spacing
            and dxy <= config.assignment_transverse_mm
        ):
            instances[best_index].components.append(component)
            instances[best_index].update()
        elif component.volume_mm3 >= config.seed_min_mm3:
            new_instance = VertebralInstance(components=[component])
            new_instance.update()
            instances.append(new_instance)
        else:
            unassigned.append(component)
    instances.sort(key=lambda item: item.centroid_world[2])
    return (instances, unassigned, spacing)


def _identity_groups_after_removing_boundaries(
    count: int, removed_boundaries: Sequence[int]
) -> List[Tuple[int, ...]]:
    removed = set((int(value) for value in removed_boundaries))
    groups: List[List[int]] = [[0]] if count else []
    for index in range(1, count):
        if index - 1 in removed:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [tuple(group) for group in groups]


def _identity_combine_group(
    instances: Sequence[VertebralInstance], group: Sequence[int]
) -> VertebralInstance:
    combined = VertebralInstance(
        components=[
            component for index in group for component in instances[index].components
        ]
    )
    combined.update()
    return combined


def _identity_local_reference_volume(
    instances: Sequence[VertebralInstance], left_index: int
) -> float:
    excluded = {left_index, left_index + 1}
    local = [
        instances[index].volume_mm3
        for index in range(max(0, left_index - 3), min(len(instances), left_index + 5))
        if index not in excluded
    ]
    if not local:
        local = [
            instance.volume_mm3
            for index, instance in enumerate(instances)
            if index not in excluded
        ]
    if not local:
        return max(
            instances[left_index].volume_mm3, instances[left_index + 1].volume_mm3, 1.0
        )
    return float(np.median(np.asarray(local, dtype=float)))


def _identity_merge_evidence(
    instances: Sequence[VertebralInstance],
    boundary: int,
    spacing: float,
    config: IdentityConfig,
) -> MergeEvidence:
    left = instances[boundary]
    right = instances[boundary + 1]
    reference_volume = max(_identity_local_reference_volume(instances, boundary), 1.0)
    gap_mm = float(abs(right.centroid_world[2] - left.centroid_world[2]))
    gap_ratio = gap_mm / max(spacing, 1.0)
    transverse = float(
        np.linalg.norm(right.centroid_world[:2] - left.centroid_world[:2])
    )
    left_ratio = left.volume_mm3 / reference_volume
    right_ratio = right.volume_mm3 / reference_volume
    merged_ratio = (left.volume_mm3 + right.volume_mm3) / reference_volume
    same_label = left.dominant_label == right.dominant_label
    close_enough = gap_ratio <= config.max_merge_gap_ratio
    transverse_consistent = transverse <= config.max_merge_transverse_mm
    volume_consistent = merged_ratio <= config.max_merged_volume_ratio
    fragment_evidence = bool(
        same_label
        or min(left_ratio, right_ratio) <= config.small_piece_volume_ratio
        or gap_ratio <= config.very_close_merge_gap_ratio
    )
    supported = bool(
        close_enough
        and transverse_consistent
        and volume_consistent
        and fragment_evidence
    )
    return MergeEvidence(
        boundary=int(boundary),
        left_index=int(boundary),
        right_index=int(boundary + 1),
        gap_mm=gap_mm,
        gap_ratio=float(gap_ratio),
        transverse_mm=transverse,
        left_volume_ratio=float(left_ratio),
        right_volume_ratio=float(right_ratio),
        merged_volume_ratio=float(merged_ratio),
        same_dominant_label=bool(same_label),
        close_enough=bool(close_enough),
        transverse_consistent=bool(transverse_consistent),
        volume_consistent=bool(volume_consistent),
        fragment_evidence=bool(fragment_evidence),
        supported=bool(supported),
    )


def _identity_huber(value: float, delta: float = 0.35) -> float:
    absolute = abs(value)
    if absolute <= delta:
        return 0.5 * absolute * absolute / delta
    return absolute - 0.5 * delta


def _identity_score_full_path(
    grouped: Sequence[VertebralInstance],
    merge_evidence: Sequence[MergeEvidence],
    spacing: float,
    config: IdentityConfig,
) -> Tuple[float, Dict[str, float]]:
    if len(grouped) != 24:
        raise ValueError(
            "A complete BodyMaps path requires exactly 24 grouped instances"
        )
    label_distance = 0.0
    exact_support_loss = 0.0
    for target, instance in enumerate(grouped, start=1):
        total = max(float(instance.raw_votes.sum()), 1e-08)
        distances = np.asarray(
            [abs(label - target) for label in range(25)], dtype=float
        )
        distances = np.minimum(distances, 3.0)
        label_distance += float(np.dot(instance.raw_votes, distances) / total)
        exact_support_loss += 1.0 - instance.support(target)
    geometry = 0.0
    for left, right in zip(grouped[:-1], grouped[1:]):
        gap = float(right.centroid_world[2] - left.centroid_world[2])
        geometry += _identity_huber(gap / max(spacing, 1.0) - 1.0)
    merge_cost = 0.0
    for evidence in merge_evidence:
        merge_cost += evidence.gap_ratio
        merge_cost += (
            0.25 * evidence.transverse_mm / max(config.max_merge_transverse_mm, 1.0)
        )
        merge_cost += max(0.0, evidence.merged_volume_ratio - 1.0)
    breakdown = {
        "label_distance": float(label_distance),
        "exact_support_loss": float(exact_support_loss),
        "geometry": float(geometry),
        "merge": float(merge_cost),
    }
    total = (
        config.label_distance_weight * label_distance
        + config.exact_support_weight * exact_support_loss
        + config.geometry_weight * geometry
        + config.merge_weight * merge_cost
    )
    breakdown["weighted_total"] = float(total)
    return (float(total), breakdown)


def _identity_evaluate_hypothesis(
    instances: Sequence[VertebralInstance],
    removed_boundaries: Tuple[int, ...],
    spacing: float,
    config: IdentityConfig,
) -> IdentityHypothesis:
    groups = _identity_groups_after_removing_boundaries(
        len(instances), removed_boundaries
    )
    evidence = [
        _identity_merge_evidence(instances, boundary, spacing, config)
        for boundary in removed_boundaries
    ]
    compatible = len(groups) == 24
    if not compatible:
        return IdentityHypothesis(
            removed_boundaries=removed_boundaries,
            groups=groups,
            target_length_compatible=False,
            score=None,
            score_breakdown={},
            raw_path=[],
            match_count=None,
            maximum_dominant_label_distance=None,
            endpoint_support=None,
            span_in_spacing_units=None,
            merge_evidence=evidence,
            all_merges_supported=bool(evidence)
            and all((item.supported for item in evidence)),
        )
    grouped = [_identity_combine_group(instances, group) for group in groups]
    score, breakdown = _identity_score_full_path(grouped, evidence, spacing, config)
    raw_path = [instance.dominant_label for instance in grouped]
    differences = [
        abs(observed - target) for target, observed in enumerate(raw_path, start=1)
    ]
    endpoint_support = (grouped[0].support(1), grouped[-1].support(24))
    span = float(
        (grouped[-1].centroid_world[2] - grouped[0].centroid_world[2])
        / max(spacing, 1.0)
    )
    return IdentityHypothesis(
        removed_boundaries=removed_boundaries,
        groups=groups,
        target_length_compatible=True,
        score=score,
        score_breakdown=breakdown,
        raw_path=raw_path,
        match_count=int(sum((value == 0 for value in differences))),
        maximum_dominant_label_distance=int(max(differences) if differences else 0),
        endpoint_support=(float(endpoint_support[0]), float(endpoint_support[1])),
        span_in_spacing_units=span,
        merge_evidence=evidence,
        all_merges_supported=all((item.supported for item in evidence)),
    )


def _identity_enumerate_hypotheses(
    instances: Sequence[VertebralInstance],
    spacing: float,
    config: Optional[IdentityConfig] = None,
) -> List[IdentityHypothesis]:
    active = config or IdentityConfig()
    hypotheses: List[IdentityHypothesis] = []
    boundaries = tuple(range(max(0, len(instances) - 1)))
    for merge_count in range(3):
        for removed in combinations(boundaries, merge_count):
            hypotheses.append(
                _identity_evaluate_hypothesis(instances, removed, spacing, active)
            )
    return hypotheses


def _identity_component_record(component: Component) -> Dict[str, object]:
    return {
        "raw_label": int(component.raw_label),
        "raw_name": LABEL_NAMES[component.raw_label],
        "component_id_within_raw_label": int(component.component_id),
        "voxel_count": int(component.voxel_count),
        "volume_mm3": float(component.volume_mm3),
        "centroid_world_mm": [float(value) for value in component.centroid_world],
        "bbox": [[int(part.start), int(part.stop)] for part in component.bbox],
    }


def _identity_instance_record(
    index: int, instance: VertebralInstance
) -> Dict[str, object]:
    return {
        "index": int(index),
        "dominant_raw_label": int(instance.dominant_label),
        "dominant_raw_name": LABEL_NAMES[instance.dominant_label],
        "volume_mm3": float(instance.volume_mm3),
        "centroid_world_mm": [float(value) for value in instance.centroid_world],
        "raw_label_support": {
            LABEL_NAMES[label]: float(instance.support(label))
            for label in range(1, 25)
            if instance.raw_votes[label] > 0
        },
        "components": [
            _identity_component_record(item) for item in instance.components
        ],
    }


def _identity_hypothesis_record(
    rank: Optional[int],
    hypothesis: IdentityHypothesis,
    structurally_supported_rank: Optional[int] = None,
) -> Dict[str, object]:
    return {
        "rank_among_compatible": rank,
        "rank_among_structurally_supported": structurally_supported_rank,
        "merge_count": len(hypothesis.removed_boundaries),
        "removed_boundaries": [int(value) for value in hypothesis.removed_boundaries],
        "groups": [[int(value) for value in group] for group in hypothesis.groups],
        "target_length_compatible": bool(hypothesis.target_length_compatible),
        "score": hypothesis.score,
        "score_breakdown": hypothesis.score_breakdown,
        "raw_path_labels": [int(value) for value in hypothesis.raw_path],
        "raw_path_names": [LABEL_NAMES[value] for value in hypothesis.raw_path],
        "match_count": hypothesis.match_count,
        "maximum_dominant_label_distance": hypothesis.maximum_dominant_label_distance,
        "endpoint_support": None
        if hypothesis.endpoint_support is None
        else {
            "L5": float(hypothesis.endpoint_support[0]),
            "C1": float(hypothesis.endpoint_support[1]),
        },
        "span_in_spacing_units": hypothesis.span_in_spacing_units,
        "merge_evidence": [asdict(item) for item in hypothesis.merge_evidence],
        "all_merges_supported": bool(hypothesis.all_merges_supported),
    }


def correct_identity(
    segmentation: np.ndarray,
    affine: np.ndarray,
    config: Optional[IdentityConfig] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    active = config or IdentityConfig()
    components = _io_extract_components(segmentation, affine)
    instances, unassigned, spacing = _identity_build_instances_without_merging(
        components, active
    )
    hypotheses = _identity_enumerate_hypotheses(instances, spacing, active)
    compatible = [item for item in hypotheses if item.target_length_compatible]
    compatible.sort(
        key=lambda item: float(item.score if item.score is not None else math.inf)
    )
    ranks = {id(item): index + 1 for index, item in enumerate(compatible)}
    unconstrained_best = compatible[0] if compatible else None
    unconstrained_second = compatible[1] if len(compatible) > 1 else None
    unconstrained_margin = (
        float(unconstrained_second.score - unconstrained_best.score)
        if unconstrained_best is not None
        and unconstrained_second is not None
        and (unconstrained_best.score is not None)
        and (unconstrained_second.score is not None)
        else None
    )
    supported = [item for item in compatible if item.all_merges_supported]
    supported.sort(
        key=lambda item: float(item.score if item.score is not None else math.inf)
    )
    supported_ranks = {id(item): index + 1 for index, item in enumerate(supported)}
    best = supported[0] if supported else None
    second = supported[1] if len(supported) > 1 else None
    margin = (
        float(second.score - best.score)
        if best is not None
        and second is not None
        and (best.score is not None)
        and (second.score is not None)
        else None
    )
    preliminary_count_ok = 24 <= len(instances) <= 26
    complete_candidate = best is not None
    correction_needed = bool(
        best is not None and best.match_count is not None and (best.match_count < 24)
    )
    endpoint_supported = bool(
        best is not None
        and best.endpoint_support is not None
        and (min(best.endpoint_support) >= active.min_endpoint_support)
        and best.raw_path
        and (best.raw_path[0] == 1)
        and (best.raw_path[-1] == 24)
    )
    sequence_supported = bool(
        best is not None
        and best.match_count is not None
        and (best.match_count >= active.min_sequence_matches)
        and (best.maximum_dominant_label_distance is not None)
        and (best.maximum_dominant_label_distance <= 1)
    )
    span_supported = bool(
        best is not None
        and best.span_in_spacing_units is not None
        and (
            active.min_span_in_spacing_units
            <= best.span_in_spacing_units
            <= active.max_span_in_spacing_units
        )
    )
    merges_supported = bool(best is not None and best.all_merges_supported)
    required_merge_count = max(0, len(instances) - 24)
    margin_required = required_merge_count > 0
    unique_supported_hypothesis = len(supported) == 1
    margin_supported = bool(
        not margin_required
        or unique_supported_hypothesis
        or (margin is not None and margin >= active.min_hypothesis_margin)
    )
    gates = {
        "preliminary_instance_count_24_to_26": bool(preliminary_count_ok),
        "complete_24_group_hypothesis_exists": bool(complete_candidate),
        "correction_is_needed": bool(correction_needed),
        "L5_and_C1_endpoints_supported": bool(endpoint_supported),
        "raw_path_close_to_full_sequence": bool(sequence_supported),
        "physical_span_plausible": bool(span_supported),
        "every_proposed_merge_has_independent_structural_support": bool(
            merges_supported
        ),
        "best_vs_second_margin_supported_when_merges_are_needed": bool(
            margin_supported
        ),
    }
    accepted = all(gates.values())
    abstention_reasons = [name for name, passed in gates.items() if not passed]
    if len(instances) < 24:
        abstention_reasons.append("partial_field_or_missing_instance_is_not_handled")
    elif len(instances) > 26:
        abstention_reasons.append("more_than_two_merges_would_be_required")
    if required_merge_count > 0 and (not merges_supported):
        abstention_reasons.append(
            "extra_instances_may_represent_a_variant_or_distinct_vertebrae"
        )
    refined = segmentation.copy()
    reassignments: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    if accepted and best is not None:
        grouped = [_identity_combine_group(instances, group) for group in best.groups]
        for group_index, (group, instance, target) in enumerate(
            zip(best.groups, grouped, range(1, 25))
        ):
            for component in instance.components:
                if component.raw_label == target:
                    continue
                distance = abs(component.raw_label - target)
                record = {
                    "target_group_index": int(group_index),
                    "source_preliminary_instances": [int(value) for value in group],
                    "from_label": int(component.raw_label),
                    "from_name": LABEL_NAMES[component.raw_label],
                    "proposed_label": int(target),
                    "proposed_name": LABEL_NAMES[target],
                    "label_distance": int(distance),
                    "voxel_count": int(component.voxel_count),
                    "volume_mm3": float(component.volume_mm3),
                }
                if distance > active.max_component_label_distance:
                    record["reason"] = (
                        "component_raw_label_more_than_one_level_from_target"
                    )
                    rejected.append(record)
                    continue
                mask = _io_component_mask(segmentation, component)
                view = refined[component.bbox]
                changed = int(np.count_nonzero(mask & (view != target)))
                view[mask] = target
                record["changed_voxels"] = changed
                record["support"] = (
                    "accepted_complete_path_and_component_distance_at_most_one"
                )
                reassignments.append(record)
    enumeration_counts = {
        str(merge_count): int(
            sum((len(item.removed_boundaries) == merge_count for item in hypotheses))
        )
        for merge_count in range(3)
    }
    report: Dict[str, object] = {
        "method": "evidence_gated_complete_sequence_candidate",
        "status": "corrected" if accepted else "abstained",
        "configuration": asdict(active),
        "estimated_spacing_mm": float(spacing),
        "premerge_instance_count": int(len(instances)),
        "required_merge_count_for_24_groups": int(required_merge_count),
        "unassigned_component_count": int(len(unassigned)),
        "premerge_instances": [
            _identity_instance_record(index, item)
            for index, item in enumerate(instances)
        ],
        "unassigned_components": [
            _identity_component_record(item) for item in unassigned
        ],
        "hypothesis_enumeration": {
            "enumerated_counts_by_merge_count": enumeration_counts,
            "compatible_complete_path_count": int(len(compatible)),
            "structurally_supported_complete_path_count": int(len(supported)),
            "unconstrained_best_score": None
            if unconstrained_best is None
            else unconstrained_best.score,
            "unconstrained_second_best_score": None
            if unconstrained_second is None
            else unconstrained_second.score,
            "unconstrained_best_vs_second_margin": unconstrained_margin,
            "best_score": None if best is None else best.score,
            "second_best_score": None if second is None else second.score,
            "best_vs_second_margin": margin,
            "margin_required": bool(margin_required),
            "unique_structurally_supported_hypothesis": bool(
                unique_supported_hypothesis
            ),
            "hypotheses": [
                _identity_hypothesis_record(
                    ranks.get(id(item)), item, supported_ranks.get(id(item))
                )
                for item in hypotheses
            ],
        },
        "decision_gates": gates,
        "abstention_reasons": abstention_reasons,
        "selected_hypothesis": None
        if best is None
        else _identity_hypothesis_record(
            ranks.get(id(best)), best, supported_ranks.get(id(best))
        ),
        "reassignments": reassignments,
        "rejected_component_reassignments": rejected,
        "changed_voxels": int(np.count_nonzero(refined != segmentation)),
        "preserved_nonadjacent_component_voxels": int(
            sum((int(item["voxel_count"]) for item in rejected))
        ),
    }
    return (refined, report)


@dataclass
class ShapeConfig:
    tiny_component_mm3: float = 10.0
    tiny_min_bbox_gap_mm: float = 0.0
    gross_si_spacing_factor: float = 2.25
    gross_min_si_mm: float = 55.0
    gross_max_relative_volume: float = 0.25
    focus_min_label: int = 6
    focus_max_label: int = 13
    span_outlier_factor: float = 1.0
    min_span_mm: float = 40.0
    min_mode_separation_spacing_fraction: float = 0.72
    min_partition_fraction: float = 0.1
    min_expected_center_advantage_spacing_fraction: float = 0.08
    max_valley_area_ratio: float = 0.7
    max_shallow_fallback_valley_ratio: float = 0.95
    shallow_fallback_minor_fraction: float = 0.3
    shallow_fallback_min_identity_voxels: int = 20000
    valley_smoothing_bins: int = 2


def _shape_voxel_sizes(affine: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(affine[:3, :3], dtype=float), axis=0)


def _shape_bbox_gap_mm(left: Component, right: Component, affine: np.ndarray) -> float:
    spacing = _shape_voxel_sizes(affine)
    squared = 0.0
    for axis, (a, b) in enumerate(zip(left.bbox, right.bbox)):
        gap_voxels = max(a.start - b.stop, b.start - a.stop, 0)
        squared += float(gap_voxels * spacing[axis]) ** 2
    return float(np.sqrt(squared))


def _shape_dominant_centers(
    segmentation: np.ndarray, affine: np.ndarray
) -> Dict[int, np.ndarray]:
    return {
        label: values[0].centroid_world
        for label, values in _io_extract_components(segmentation, affine).items()
        if values
    }


def _shape_sequence_spacing(centers: Dict[int, np.ndarray]) -> float:
    gaps = [
        float(np.linalg.norm(centers[label + 1] - centers[label]))
        for label in range(1, 24)
        if label in centers and label + 1 in centers
    ]
    plausible = [value for value in gaps if 8.0 <= value <= 60.0]
    return float(np.median(plausible)) if plausible else 25.0


def _shape_cleanup_components(
    segmentation: np.ndarray, affine: np.ndarray, config: ShapeConfig
) -> Tuple[np.ndarray, Dict[str, object]]:
    refined = segmentation.copy()
    components = _io_extract_components(segmentation, affine)
    centers = _shape_dominant_centers(segmentation, affine)
    spacing = _shape_sequence_spacing(centers)
    actions: List[Dict[str, object]] = []
    for label, values in components.items():
        if len(values) < 2:
            continue
        _shape_main = values[0]
        for component in values[1:]:
            bbox_gap = _shape_bbox_gap_mm(component, _shape_main, affine)
            si_distance = abs(
                float(component.centroid_world[2] - _shape_main.centroid_world[2])
            )
            relative_volume = component.volume_mm3 / max(_shape_main.volume_mm3, 1e-08)
            tiny_isolated = bool(
                component.volume_mm3 < config.tiny_component_mm3
                and bbox_gap >= config.tiny_min_bbox_gap_mm
            )
            gross_displacement = bool(
                si_distance
                >= max(config.gross_min_si_mm, config.gross_si_spacing_factor * spacing)
                and relative_volume <= config.gross_max_relative_volume
            )
            if not (tiny_isolated or gross_displacement):
                continue
            local_mask = _io_component_mask(segmentation, component)
            view = refined[component.bbox]
            removed = int(np.count_nonzero(local_mask & (view == label)))
            view[local_mask] = 0
            actions.append(
                {
                    "label": int(label),
                    "name": LABEL_NAMES[label],
                    "voxels": removed,
                    "volume_mm3": float(component.volume_mm3),
                    "bbox_gap_mm": bbox_gap,
                    "si_distance_mm": si_distance,
                    "relative_volume": relative_volume,
                    "reason": "tiny_and_isolated"
                    if tiny_isolated
                    else "gross_longitudinal_displacement",
                }
            )
    return (
        refined,
        {
            "estimated_spacing_mm": spacing,
            "removed_component_count": len(actions),
            "removed_voxels": int(sum((int(item["voxels"]) for item in actions))),
            "actions": actions,
        },
    )


def _shape_local_axis(
    centers: Dict[int, np.ndarray], label: int
) -> Optional[np.ndarray]:
    if label - 1 not in centers or label + 1 not in centers:
        return None
    direction = centers[label + 1] - centers[label - 1]
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-08 else None


def _shape_two_means(values: np.ndarray) -> Tuple[float, float, np.ndarray]:
    lower, upper = np.percentile(values, [25.0, 75.0])
    membership = values <= 0.5 * (lower + upper)
    for _ in range(40):
        threshold = 0.5 * (lower + upper)
        membership = values <= threshold
        if not np.any(membership) or np.all(membership):
            break
        updated_lower = float(values[membership].mean())
        updated_upper = float(values[~membership].mean())
        if abs(updated_lower - lower) + abs(updated_upper - upper) < 0.0001:
            lower, upper = (updated_lower, updated_upper)
            break
        lower, upper = (updated_lower, updated_upper)
    return (float(lower), float(upper), membership)


def _shape_valley_between_modes(
    values: np.ndarray, lower_center: float, upper_center: float, smoothing_bins: int
) -> Tuple[float, Dict[str, float]]:
    width = max(float(values.max() - values.min()), 1.0)
    bins = max(16, min(192, int(np.ceil(width))))
    counts, edges = np.histogram(values, bins=bins)
    kernel_width = max(1, int(smoothing_bins))
    kernel = np.arange(1, kernel_width + 2, dtype=float)
    kernel = np.concatenate([kernel, kernel[-2::-1]])
    kernel /= kernel.sum()
    smooth = np.convolve(counts.astype(float), kernel, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])
    search = (centers > lower_center) & (centers < upper_center)
    indices = np.flatnonzero(search)
    if not indices.size:
        return (float(0.5 * (lower_center + upper_center)), {"valley_area_ratio": 1.0})
    midpoint = 0.5 * (lower_center + upper_center)
    objective = smooth[indices] / max(float(np.median(smooth[indices])), 1.0)
    objective += (
        0.08
        * np.abs(centers[indices] - midpoint)
        / max(upper_center - lower_center, 1e-08)
    )
    best = int(indices[int(np.argmin(objective))])
    peak_scale = max(
        float(np.max(smooth[centers <= lower_center]))
        if np.any(centers <= lower_center)
        else 1.0,
        float(np.max(smooth[centers >= upper_center]))
        if np.any(centers >= upper_center)
        else 1.0,
        1.0,
    )
    return (
        float(centers[best]),
        {
            "valley_area_ratio": float(smooth[best] / peak_scale),
            "valley_smoothed_area": float(smooth[best]),
        },
    )


def _shape_component_projection(
    segmentation: np.ndarray, component: Component, affine: np.ndarray, axis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _io_component_mask(segmentation, component)
    local_points = np.argwhere(mask)
    starts = np.asarray([part.start for part in component.bbox])
    world = nib.affines.apply_affine(affine, local_points + starts)
    return (mask, local_points, world @ axis)


def _shape_split_long_components(
    segmentation: np.ndarray,
    affine: np.ndarray,
    config: ShapeConfig,
    shallow_fallback_labels: Sequence[int] = (),
) -> Tuple[np.ndarray, Dict[str, object]]:
    refined = segmentation.copy()
    components = _io_extract_components(segmentation, affine)
    centers = {
        label: values[0].centroid_world
        for label, values in components.items()
        if values
    }
    spacing = _shape_sequence_spacing(centers)
    spans: Dict[int, float] = {}
    projection_cache: Dict[
        int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for label in range(config.focus_min_label, config.focus_max_label + 1):
        if label not in components:
            continue
        axis = _shape_local_axis(centers, label)
        if axis is None:
            continue
        mask, points, projection = _shape_component_projection(
            segmentation, components[label][0], affine, axis
        )
        spans[label] = float(
            np.percentile(projection, 99.0) - np.percentile(projection, 1.0)
        )
        projection_cache[label] = (mask, points, projection, axis)
    reference_spans = [value for value in spans.values() if 20.0 <= value <= 80.0]
    reference_span = float(np.median(reference_spans)) if reference_spans else 40.0
    span_threshold = max(
        config.min_span_mm, config.span_outlier_factor * reference_span
    )
    actions: List[Dict[str, object]] = []
    for label in range(config.focus_min_label, config.focus_max_label + 1):
        if label not in projection_cache or spans[label] <= span_threshold:
            continue
        _shape_main = components[label][0]
        mask, points, projection, axis = projection_cache[label]
        lower_center, upper_center, lower_membership = _shape_two_means(projection)
        mode_separation = upper_center - lower_center
        lower_fraction = float(np.mean(lower_membership))
        upper_fraction = 1.0 - lower_fraction
        expected_center = float(
            0.5 * (centers[label - 1] @ axis + centers[label + 1] @ axis)
        )
        lower_distance = abs(lower_center - expected_center)
        upper_distance = abs(upper_center - expected_center)
        retain_lower = lower_distance < upper_distance
        retained_distance = min(lower_distance, upper_distance)
        alternate_distance = max(lower_distance, upper_distance)
        target_label = label + 1 if retain_lower else label - 1
        contaminant_center = upper_center if retain_lower else lower_center
        target_center = float(centers[target_label] @ axis)
        current_center = float(centers[label] @ axis)
        target_advantage = abs(contaminant_center - current_center) - abs(
            contaminant_center - target_center
        )
        gates = {
            "span_is_outlier": True,
            "two_modes_are_separated": bool(
                mode_separation >= config.min_mode_separation_spacing_fraction * spacing
            ),
            "both_partitions_are_substantial": bool(
                min(lower_fraction, upper_fraction) >= config.min_partition_fraction
            ),
            "expected_center_prefers_one_mode": bool(
                alternate_distance - retained_distance
                >= config.min_expected_center_advantage_spacing_fraction * spacing
            ),
        }
        threshold, valley = _shape_valley_between_modes(
            projection, lower_center, upper_center, config.valley_smoothing_bins
        )
        record: Dict[str, object] = {
            "label": label,
            "name": LABEL_NAMES[label],
            "target_label": target_label,
            "target_name": LABEL_NAMES[target_label],
            "span_mm": spans[label],
            "reference_span_mm": reference_span,
            "span_threshold_mm": span_threshold,
            "mode_centers_mm": [lower_center, upper_center],
            "mode_separation_mm": mode_separation,
            "partition_fractions": [lower_fraction, upper_fraction],
            "expected_center_mm": expected_center,
            "retain_side": "inferior" if retain_lower else "superior",
            "target_center_advantage_mm": target_advantage,
            "threshold_mm": threshold,
            "valley": valley,
            "gates": gates,
            "accepted": bool(all(gates.values())),
            "changed_voxels": 0,
        }
        deep_valley = bool(valley["valley_area_ratio"] <= config.max_valley_area_ratio)
        identity_fallback = bool(
            label in set(shallow_fallback_labels)
            and valley["valley_area_ratio"] <= config.max_shallow_fallback_valley_ratio
        )
        gates["valley_or_identity_shift_fallback_is_supported"] = bool(
            deep_valley or identity_fallback
        )
        record["threshold_rule"] = "histogram_valley"
        if identity_fallback and (not deep_valley):
            quantile = (
                1.0 - config.shallow_fallback_minor_fraction
                if retain_lower
                else config.shallow_fallback_minor_fraction
            )
            threshold = float(np.quantile(projection, quantile))
            record["threshold_mm"] = threshold
            record["threshold_rule"] = "identity_shift_guarded_minor_lobe_quantile"
            record["fallback_quantile"] = quantile
        record["accepted"] = bool(all(gates.values()))
        if all(gates.values()):
            move = projection > threshold if retain_lower else projection < threshold
            local_candidate = np.zeros(mask.shape, dtype=bool)
            local_candidate[tuple(points[move].T)] = True
            target_local = segmentation[_shape_main.bbox] == target_label
            touches_target = local_candidate & ndimage.binary_dilation(
                target_local, structure=np.ones((3, 3, 3), dtype=bool)
            )
            if np.any(touches_target):
                candidate_cc, count = ndimage.label(
                    local_candidate, structure=np.ones((3, 3, 3), dtype=bool)
                )
                touching_ids = np.unique(candidate_cc[touches_target])
                touching_ids = touching_ids[touching_ids > 0]
                local_candidate = np.isin(candidate_cc, touching_ids)
                view = refined[_shape_main.bbox]
                changed = int(np.count_nonzero(local_candidate & (view == label)))
                view[local_candidate] = target_label
                record["changed_voxels"] = changed
                record["accepted"] = bool(changed > 0)
                if not changed:
                    record["rejection_reason"] = (
                        "no_target_connected_voxels_after_partition"
                    )
            else:
                record["accepted"] = False
                record["rejection_reason"] = (
                    "candidate_side_does_not_touch_target_label"
                )
        actions.append(record)
    return (
        refined,
        {
            "estimated_spacing_mm": spacing,
            "reference_span_mm": reference_span,
            "span_threshold_mm": span_threshold,
            "candidate_count": len(actions),
            "accepted_count": int(sum((bool(item["accepted"]) for item in actions))),
            "changed_voxels": int(
                sum((int(item["changed_voxels"]) for item in actions))
            ),
            "actions": actions,
        },
    )


def refine_shape(
    segmentation: np.ndarray, affine: np.ndarray, config: Optional[ShapeConfig] = None
) -> Tuple[np.ndarray, Dict[str, object]]:
    active = config or ShapeConfig()
    identity_refined, identity_report = correct_identity(segmentation, affine)
    cleaned_before, cleanup_before = _shape_cleanup_components(
        identity_refined, affine, active
    )
    shallow_fallback_labels = sorted(
        {
            int(item["proposed_label"])
            for item in identity_report.get("reassignments", [])
            if int(item.get("changed_voxels", 0))
            >= active.shallow_fallback_min_identity_voxels
            and abs(int(item["from_label"]) - int(item["proposed_label"])) == 1
        }
    )
    boundary_refined, boundary_report = _shape_split_long_components(
        cleaned_before, affine, active, shallow_fallback_labels=shallow_fallback_labels
    )
    boundary_report["identity_shift_fallback_labels"] = shallow_fallback_labels
    final, cleanup_after = _shape_cleanup_components(boundary_refined, affine, active)
    return (
        final,
        {
            "method": "evidence_gated_identity_with_guarded_shape_refinement",
            "configuration": asdict(active),
            "identity_stage": identity_report,
            "cleanup_before_boundary": cleanup_before,
            "boundary_stage": boundary_report,
            "cleanup_after_boundary": cleanup_after,
            "input_foreground_voxels": int(np.count_nonzero(segmentation)),
            "output_foreground_voxels": int(np.count_nonzero(final)),
            "added_foreground_voxels": int(
                np.count_nonzero((segmentation == 0) & (final != 0))
            ),
            "removed_foreground_voxels": int(
                np.count_nonzero((segmentation != 0) & (final == 0))
            ),
            "changed_voxels": int(np.count_nonzero(segmentation != final)),
        },
    )


def _shape_process_case(
    input_case: Path,
    output_case: Path,
    config: Optional[ShapeConfig] = None,
    save_individual_labels: bool = True,
) -> Dict[str, object]:
    path = input_case / "combined_labels.nii.gz"
    image = nib.load(str(path))
    segmentation = np.asarray(image.dataobj, dtype=np.uint8)
    _io_validate_labels(segmentation, path)
    refined, report = refine_shape(segmentation, image.affine, config=config)
    if save_individual_labels:
        _io_save_case(refined, image, output_case)
    else:
        _io_save_like(refined, image, output_case / "combined_labels.nii.gz")
    report.update(
        {
            "case_id": input_case.name,
            "input": str(path),
            "output": str(output_case / "combined_labels.nii.gz"),
            "shape": [int(value) for value in image.shape],
            "orientation": [str(value) for value in nib.aff2axcodes(image.affine)],
        }
    )
    return report


def _shape_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--disable-boundary", action="store_true")
    args = parser.parse_args()
    config = ShapeConfig()
    reports: List[Dict[str, object]] = []
    for case_dir in _io_iter_case_dirs(args.input_dir):
        if args.disable_boundary:
            image = nib.load(str(case_dir / "combined_labels.nii.gz"))
            source = np.asarray(image.dataobj, dtype=np.uint8)
            identity_refined, identity_report = correct_identity(source, image.affine)
            refined, cleanup_report = _shape_cleanup_components(
                identity_refined, image.affine, config
            )
            output_case = args.output_dir / case_dir.name
            if args.combined_only:
                _io_save_like(refined, image, output_case / "combined_labels.nii.gz")
            else:
                _io_save_case(refined, image, output_case)
            report = {
                "method": "identity_with_cleanup_only_ablation",
                "identity_stage": identity_report,
                "cleanup_before_boundary": cleanup_report,
                "case_id": case_dir.name,
                "changed_voxels": int(np.count_nonzero(source != refined)),
                "input_foreground_voxels": int(np.count_nonzero(source)),
                "output_foreground_voxels": int(np.count_nonzero(refined)),
                "added_foreground_voxels": 0,
                "removed_foreground_voxels": int(
                    np.count_nonzero((source != 0) & (refined == 0))
                ),
            }
        else:
            report = _shape_process_case(
                case_dir,
                args.output_dir / case_dir.name,
                config=config,
                save_individual_labels=not args.combined_only,
            )
        reports.append(report)
        print(
            f"{case_dir.name}: changed={report['changed_voxels']}; removed={report['removed_foreground_voxels']}; added={report['added_foreground_voxels']}"
        )
    report_path = args.report_json or args.output_dir / "shape_aware_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"cases": reports}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_path}")


GEOMETRY_TARGET_LABELS = tuple(range(7, 13))
GEOMETRY_LABEL_NAMES = {label: f"T{18 - label}" for label in range(6, 18)}
GEOMETRY_FULL = np.ones((3, 3, 3), dtype=bool)
GEOMETRY_FACES = ndimage.generate_binary_structure(3, 1)


def _geometry_crop_for(mask: np.ndarray, margin: int = 5) -> Tuple[slice, slice, slice]:
    points = np.argwhere(mask)
    low = np.maximum(points.min(axis=0) - margin, 0)
    high = np.minimum(points.max(axis=0) + margin + 1, mask.shape)
    return tuple((slice(int(a), int(b)) for a, b in zip(low, high)))


def _geometry_world_grid(
    shape: Tuple[int, int, int], crop: Tuple[slice, slice, slice], affine: np.ndarray
) -> np.ndarray:
    points = np.indices(shape, dtype=np.float32).reshape(3, -1).T
    points += np.asarray([part.start for part in crop], dtype=np.float32)
    return nib.affines.apply_affine(affine, points).reshape(*shape, 3)


def _geometry_body_anchor(
    mask: np.ndarray, world: np.ndarray, spacing: np.ndarray
) -> Dict[str, object]:
    distance = ndimage.distance_transform_edt(mask, sampling=spacing)
    values = distance[mask]
    threshold = max(3.0, float(np.percentile(values, 82.0)))
    core = mask & (distance >= threshold)
    if np.count_nonzero(core) < 100:
        threshold = max(2.0, float(np.percentile(values, 70.0)))
        core = mask & (distance >= threshold)
    core_y = world[..., 1][core]
    anterior_cut = float(np.percentile(core_y, 35.0))
    anterior_core = core & (world[..., 1] >= anterior_cut)
    weights = np.square(distance[anterior_core]).astype(np.float64)
    points = world[anterior_core].astype(np.float64)
    anchor = np.average(points, axis=0, weights=weights)
    return {
        "anchor": anchor,
        "distance": distance,
        "core": anterior_core,
        "core_threshold_mm": threshold,
        "anterior_cut_y_mm": anterior_cut,
        "core_voxels": int(np.count_nonzero(anterior_core)),
    }


def _geometry_body_valley(
    union: np.ndarray,
    world: np.ndarray,
    anchors: Dict[int, Dict[str, object]],
    low_label: int,
    high_label: int,
) -> Dict[str, object]:
    low = np.asarray(anchors[low_label]["anchor"], dtype=float)
    high = np.asarray(anchors[high_label]["anchor"], dtype=float)
    axis = high - low
    separation = float(np.linalg.norm(axis))
    axis /= max(separation, 1e-08)
    projection = np.einsum("...i,i->...", world - low, axis)
    clipped = np.clip(projection / max(separation, 1e-08), 0.0, 1.0)
    centerline = low + clipped[..., None] * (high - low)
    delta = world - centerline
    axial_distance = np.linalg.norm(
        delta - np.einsum("...i,i->...", delta, axis)[..., None] * axis, axis=-1
    )
    anterior_floor = min(low[1], high[1]) - 13.0
    body_roi = union & (axial_distance <= 27.0) & (world[..., 1] >= anterior_floor)
    bin_width = 0.7
    edges = np.arange(0.0, separation + bin_width, bin_width)
    if edges[-1] < separation:
        edges = np.append(edges, separation)
    counts, edges = np.histogram(projection[body_roi], bins=edges)
    smooth = ndimage.gaussian_filter1d(counts.astype(float), sigma=1.2, mode="nearest")
    centers = 0.5 * (edges[:-1] + edges[1:])
    search = (centers >= 0.3 * separation) & (centers <= 0.7 * separation)
    if not np.any(search):
        threshold = 0.5 * separation
        valley_ratio = 1.0
    else:
        ids = np.flatnonzero(search)
        index = int(ids[np.argmin(smooth[ids])])
        threshold = float(centers[index])
        endpoint_scale = max(
            float(np.percentile(smooth[centers < 0.3 * separation], 75))
            if np.any(centers < 0.3 * separation)
            else 1.0,
            float(np.percentile(smooth[centers > 0.7 * separation], 75))
            if np.any(centers > 0.7 * separation)
            else 1.0,
            1.0,
        )
        valley_ratio = float(smooth[index] / endpoint_scale)
    return {
        "axis": axis,
        "projection": projection,
        "separation_mm": separation,
        "threshold_mm_from_low_anchor": threshold,
        "valley_ratio": valley_ratio,
        "body_roi_voxels": int(np.count_nonzero(body_roi)),
    }


@dataclass
class BoundaryConfig:
    boundary_margin_mm: float = 0.7
    posterior_offset_mm: float = 15.0
    primary_min_fraction: float = 0.01
    primary_min_voxels: int = 100
    primary_min_contact_ratio: float = 1.1
    companion_min_voxels: int = 20
    companion_min_contact_ratio: float = 1.0
    maximum_lobe_fraction: float = 0.15
    maximum_valley_ratio: float = 0.68
    body_min_fraction: float = 0.03
    body_min_contact_ratio: float = 1.5
    body_maximum_valley_ratio: float = 0.45
    protected_anchor_radius_mm: float = 8.0
    competing_distance_bias_mm: float = 0.0
    minimum_trimmed_volume_mm3: float = 500.0
    fragment_reattachment_minimum_large_actions: int = 3
    fragment_reattachment_maximum_volume_mm3: float = 2000.0
    branch_radius_mm: float = 30.0
    branch_min_voxels: int = 80
    branch_lateral_offset_mm: float = 15.0
    maximum_total_changed_fraction: float = 0.12
    maximum_main_component_fraction_drop: float = 0.002


def _boundary_anchor_distance(world: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.linalg.norm(world - anchor, axis=-1)


def _boundary_component_records(
    data: np.ndarray,
    world: np.ndarray,
    pair: Dict[str, object],
    source: int,
    target: int,
    low_label: int,
    anchors: Dict[int, Dict[str, object]],
    config: BoundaryConfig,
) -> List[Dict[str, object]]:
    projection = np.asarray(pair["projection"])
    threshold = float(pair["threshold_mm_from_low_anchor"])
    target_side = (
        projection > threshold + config.boundary_margin_mm
        if target > low_label
        else projection < threshold - config.boundary_margin_mm
    )
    candidate = (data == source) & target_side
    components, count = ndimage.label(candidate, structure=GEOMETRY_FULL)
    target_mask = data == target
    retained_source = (data == source) & ~candidate
    anchor = np.asarray(anchors[source]["anchor"], dtype=float)
    source_voxels = max(int(np.count_nonzero(data == source)), 1)
    posterior_limit = (
        min(
            float(np.asarray(anchors[source]["anchor"])[1]),
            float(np.asarray(anchors[target]["anchor"])[1]),
        )
        - config.posterior_offset_mm
    )
    records: List[Dict[str, object]] = []
    for component_id in range(1, count + 1):
        mask = components == component_id
        voxels = int(np.count_nonzero(mask))
        if voxels < 8:
            continue
        shell = ndimage.binary_dilation(mask, structure=GEOMETRY_FACES) & ~mask
        target_contact = int(np.count_nonzero(shell & target_mask))
        source_neck = int(np.count_nonzero(shell & retained_source))
        if not target_contact:
            continue
        points = world[mask]
        centroid = np.mean(points, axis=0)
        protected = mask & (
            _boundary_anchor_distance(world, anchor)
            <= config.protected_anchor_radius_mm
        )
        records.append(
            {
                "mask": mask,
                "source": source,
                "source_name": GEOMETRY_LABEL_NAMES[source],
                "target": target,
                "target_name": GEOMETRY_LABEL_NAMES[target],
                "voxels": voxels,
                "fraction_of_source": float(voxels / source_voxels),
                "target_contact": target_contact,
                "source_neck": source_neck,
                "contact_ratio": float(target_contact / max(source_neck, 1)),
                "centroid_world_xyz": centroid.tolist(),
                "posterior": bool(float(centroid[1]) < posterior_limit),
                "protected_anchor_voxels": int(np.count_nonzero(protected)),
                "projection_quantiles_mm": np.percentile(
                    projection[mask], [5, 50, 95]
                ).tolist(),
            }
        )
    return sorted(records, key=lambda item: int(item["voxels"]), reverse=True)


def _boundary_branch_signature(
    data: np.ndarray,
    label: int,
    world: np.ndarray,
    spacing: np.ndarray,
    config: BoundaryConfig,
) -> Dict[str, object]:
    mask = data == label
    anchor_record = _geometry_body_anchor(mask, world, spacing)
    anchor = np.asarray(anchor_record["anchor"], dtype=float)
    distance_from_core = ndimage.distance_transform_edt(
        ~np.asarray(anchor_record["core"], dtype=bool), sampling=spacing
    )
    distal = mask & (distance_from_core > config.branch_radius_mm)
    components, count = ndimage.label(distal, structure=GEOMETRY_FULL)
    component_sizes = np.bincount(components.ravel())[1:]
    branches: List[Dict[str, object]] = []
    category_counts = {"left": 0, "midline": 0, "right": 0}
    for component_id in range(1, count + 1):
        voxels = int(component_sizes[component_id - 1])
        if voxels < config.branch_min_voxels:
            continue
        centroid = np.mean(world[components == component_id], axis=0)
        lateral = float(centroid[0] - anchor[0])
        if lateral < -config.branch_lateral_offset_mm:
            category = "left"
        elif lateral > config.branch_lateral_offset_mm:
            category = "right"
        else:
            category = "midline"
        category_counts[category] += 1
        branches.append(
            {
                "voxels": voxels,
                "centroid_world_xyz": centroid.tolist(),
                "category": category,
            }
        )
    total = len(branches)
    score = 2.0 * max(total - 3, 0) + 0.5 * max(3 - total, 0)
    score += 1.0 * abs(category_counts["midline"] - 1)
    score += 0.5 * abs(category_counts["left"] - 1)
    score += 0.5 * abs(category_counts["right"] - 1)
    return {
        "label": label,
        "name": GEOMETRY_LABEL_NAMES[label],
        "branch_count": total,
        "category_counts": category_counts,
        "score": float(score),
        "branches": branches,
    }


def _boundary_pair_branch_score(
    data: np.ndarray,
    source: int,
    target: int,
    world: np.ndarray,
    spacing: np.ndarray,
    config: BoundaryConfig,
) -> Tuple[float, List[Dict[str, object]]]:
    signatures = [
        _boundary_branch_signature(data, label, world, spacing, config)
        for label in (source, target)
    ]
    return (float(sum((float(item["score"]) for item in signatures))), signatures)


def _boundary_local_axis(
    anchors: Dict[int, Dict[str, object]], label: int
) -> np.ndarray:
    lower = max(min(GEOMETRY_TARGET_LABELS), label - 1)
    upper = min(max(GEOMETRY_TARGET_LABELS), label + 1)
    if lower == upper:
        return np.asarray([0.0, 0.0, 1.0])
    direction = np.asarray(anchors[upper]["anchor"]) - np.asarray(
        anchors[lower]["anchor"]
    )
    return direction / max(float(np.linalg.norm(direction)), 1e-08)


def _boundary_label_span(
    data: np.ndarray,
    label: int,
    world: np.ndarray,
    anchors: Dict[int, Dict[str, object]],
) -> float:
    axis = _boundary_local_axis(anchors, label)
    values = np.einsum("...i,i->...", world, axis)[data == label]
    return float(np.percentile(values, 99.0) - np.percentile(values, 1.0))


def _boundary_main_component_fraction(mask: np.ndarray) -> float:
    components, count = ndimage.label(mask, structure=GEOMETRY_FULL)
    if not count:
        return 0.0
    sizes = np.bincount(components.ravel())[1:]
    return float(np.max(sizes) / max(np.sum(sizes), 1))


def _boundary_reattach_secondary_components(
    data: np.ndarray, baseline: np.ndarray, spacing: np.ndarray, config: BoundaryConfig
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    result = data.copy()
    proposals: List[Tuple[np.ndarray, int, int, int]] = []
    voxel_volume = float(np.prod(spacing))
    for label in GEOMETRY_TARGET_LABELS:
        mask = data == label
        components, count = ndimage.label(mask, structure=GEOMETRY_FULL)
        if count < 2:
            continue
        sizes = np.bincount(components.ravel())[1:]
        main_id = int(np.argmax(sizes)) + 1
        for component_id in range(1, count + 1):
            if component_id == main_id:
                continue
            component = components == component_id
            voxels = int(np.count_nonzero(component))
            if not voxels:
                continue
            if voxels * voxel_volume > config.fragment_reattachment_maximum_volume_mm3:
                continue
            if not np.all(baseline[component] == label):
                continue
            shell = (
                ndimage.binary_dilation(component, structure=GEOMETRY_FULL) & ~component
            )
            contacts = {
                neighbor: int(np.count_nonzero(shell & (data == neighbor)))
                for neighbor in (label - 1, label + 1)
                if neighbor in GEOMETRY_TARGET_LABELS
            }
            touching = [neighbor for neighbor, value in contacts.items() if value > 0]
            if len(touching) != 1:
                continue
            target = touching[0]
            proposals.append((component, label, target, contacts[target]))
    actions: List[Dict[str, object]] = []
    for component, source, target, contact in proposals:
        valid = component & (result == source)
        voxels = int(np.count_nonzero(valid))
        if not voxels:
            continue
        result[valid] = target
        actions.append(
            {
                "source": source,
                "source_name": GEOMETRY_LABEL_NAMES[source],
                "target": target,
                "target_name": GEOMETRY_LABEL_NAMES[target],
                "kind": "secondary_fragment_reattachment",
                "changed_voxels": voxels,
                "volume_mm3": float(voxels * voxel_volume),
                "target_contact_voxels": contact,
            }
        )
    return (result, actions)


def _boundary_trim_by_competing_distance(
    data: np.ndarray,
    move: np.ndarray,
    source: int,
    target: int,
    spacing: np.ndarray,
    bias_mm: float,
) -> np.ndarray:
    if not np.any(move):
        return move
    target_mask = data == target
    retained_source = (data == source) & ~move
    distance_to_target = ndimage.distance_transform_edt(~target_mask, sampling=spacing)
    distance_to_source = ndimage.distance_transform_edt(
        ~retained_source, sampling=spacing
    )
    trimmed = move & (distance_to_target <= distance_to_source + float(bias_mm))
    if not np.any(trimmed):
        return trimmed
    components, count = ndimage.label(trimmed, structure=GEOMETRY_FULL)
    touches = trimmed & ndimage.binary_dilation(target_mask, structure=GEOMETRY_FULL)
    identifiers = np.unique(components[touches])
    identifiers = identifiers[identifiers > 0]
    return (
        np.isin(components, identifiers) if identifiers.size else np.zeros_like(trimmed)
    )


def refine_boundary_window(
    segmentation: np.ndarray,
    affine: np.ndarray,
    config: Optional[BoundaryConfig] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    active = config or BoundaryConfig()
    original = np.asarray(segmentation, dtype=np.uint8)
    missing_labels = [
        label for label in GEOMETRY_TARGET_LABELS if not np.any(original == label)
    ]
    if missing_labels:
        return (
            original.copy(),
            {
                "method": "evidence_gated_thoracic_topology_refinement",
                "configuration": asdict(active),
                "abstained": True,
                "abstention_reason": "one_or_more_T6_to_T11_labels_are_missing",
                "missing_labels": missing_labels,
                "actions": [],
                "changed_voxels": 0,
                "case_checks": {
                    "foreground_preserved": True,
                    "only_target_labels_changed": True,
                    "only_adjacent_exchanges": True,
                    "bounded_total_change": True,
                    "main_components_preserved": True,
                },
                "topology_stage_accepted": False,
                "rollback_reason": None,
            },
        )
    target_mask = np.isin(original, GEOMETRY_TARGET_LABELS)
    crop = _geometry_crop_for(target_mask)
    data = original[crop].copy()
    baseline = data.copy()
    world = _geometry_world_grid(data.shape, crop, affine)
    spacing = np.linalg.norm(np.asarray(affine[:3, :3], dtype=float), axis=0)
    union = np.isin(data, GEOMETRY_TARGET_LABELS)
    anchors = {
        label: _geometry_body_anchor(data == label, world, spacing)
        for label in GEOMETRY_TARGET_LABELS
    }
    baseline_spans = {
        label: _boundary_label_span(data, label, world, anchors)
        for label in GEOMETRY_TARGET_LABELS
    }
    reference_span = float(np.median(list(baseline_spans.values())))
    pair_records: List[Dict[str, object]] = []
    accepted_masks: List[Tuple[np.ndarray, int, int, str]] = []
    reserved = np.zeros(data.shape, dtype=bool)
    for low_label in range(7, 12):
        high_label = low_label + 1
        pair = _geometry_body_valley(union, world, anchors, low_label, high_label)
        pair_record: Dict[str, object] = {
            "labels": [low_label, high_label],
            "names": [
                GEOMETRY_LABEL_NAMES[low_label],
                GEOMETRY_LABEL_NAMES[high_label],
            ],
            "valley_ratio": float(pair["valley_ratio"]),
            "separation_mm": float(pair["separation_mm"]),
            "directions": [],
        }
        for source, target in ((low_label, high_label), (high_label, low_label)):
            components = _boundary_component_records(
                data, world, pair, source, target, low_label, anchors, active
            )
            public_components = [
                {key: value for key, value in item.items() if key != "mask"}
                for item in components
            ]
            direction: Dict[str, object] = {
                "source": source,
                "source_name": GEOMETRY_LABEL_NAMES[source],
                "target": target,
                "target_name": GEOMETRY_LABEL_NAMES[target],
                "components": public_components,
                "accepted": False,
                "reason": "no_supported_crossing_lobe",
            }
            posterior_primary = [
                item
                for item in components
                if bool(item["posterior"])
                and int(item["voxels"]) >= active.primary_min_voxels
                and (float(item["fraction_of_source"]) >= active.primary_min_fraction)
                and (float(item["fraction_of_source"]) <= active.maximum_lobe_fraction)
                and (float(item["contact_ratio"]) >= active.primary_min_contact_ratio)
                and (int(item["protected_anchor_voxels"]) == 0)
            ]
            if (
                posterior_primary
                and float(pair["valley_ratio"]) <= active.maximum_valley_ratio
            ):
                selected = [
                    item
                    for item in components
                    if bool(item["posterior"])
                    and int(item["voxels"]) >= active.companion_min_voxels
                    and (
                        float(item["fraction_of_source"])
                        <= active.maximum_lobe_fraction
                    )
                    and (
                        float(item["contact_ratio"])
                        >= active.companion_min_contact_ratio
                    )
                    and (int(item["protected_anchor_voxels"]) == 0)
                ]
                move = np.logical_or.reduce(
                    [np.asarray(item["mask"]) for item in selected]
                )
                pretrim_voxels = int(np.count_nonzero(move))
                move = _boundary_trim_by_competing_distance(
                    data,
                    move,
                    source,
                    target,
                    spacing,
                    active.competing_distance_bias_mm,
                )
                move &= ~reserved
                selected_volume_mm3 = float(np.count_nonzero(move) * np.prod(spacing))
                before_score, before_signature = _boundary_pair_branch_score(
                    data, source, target, world, spacing, active
                )
                trial = data.copy()
                trial[move] = target
                after_score, after_signature = _boundary_pair_branch_score(
                    trial, source, target, world, spacing, active
                )
                direction.update(
                    {
                        "candidate_type": "posterior_process_group",
                        "selected_voxels_before_distance_trim": pretrim_voxels,
                        "selected_voxels": int(np.count_nonzero(move)),
                        "selected_volume_mm3": selected_volume_mm3,
                        "branch_score_before": before_score,
                        "branch_score_after": after_score,
                        "branch_signatures_before": before_signature,
                        "branch_signatures_after": after_signature,
                    }
                )
                if selected_volume_mm3 >= active.minimum_trimmed_volume_mm3:
                    accepted_masks.append(
                        (move, source, target, "posterior_process_group")
                    )
                    reserved |= move
                    direction["accepted"] = True
                    direction["reason"] = "anatomical_boundary_and_interface_supported"
                else:
                    direction["reason"] = "trimmed_lobe_below_physical_volume_gate"
            body_candidates = [
                item
                for item in components
                if not bool(item["posterior"])
                and int(item["voxels"]) >= active.primary_min_voxels
                and (float(item["fraction_of_source"]) >= active.body_min_fraction)
                and (float(item["fraction_of_source"]) <= active.maximum_lobe_fraction)
                and (float(item["contact_ratio"]) >= active.body_min_contact_ratio)
                and (int(item["protected_anchor_voxels"]) == 0)
            ]
            if (
                body_candidates
                and float(pair["valley_ratio"]) <= active.body_maximum_valley_ratio
            ):
                body = body_candidates[0]
                move = np.asarray(body["mask"]) & ~reserved
                pretrim_voxels = int(np.count_nonzero(move))
                move = _boundary_trim_by_competing_distance(
                    data,
                    move,
                    source,
                    target,
                    spacing,
                    active.competing_distance_bias_mm,
                )
                selected_volume_mm3 = float(np.count_nonzero(move) * np.prod(spacing))
                trial = data.copy()
                trial[move] = target
                trial_anchors = {
                    label: _geometry_body_anchor(trial == label, world, spacing)
                    for label in GEOMETRY_TARGET_LABELS
                }
                source_after = _boundary_label_span(trial, source, world, trial_anchors)
                target_after = _boundary_label_span(trial, target, world, trial_anchors)
                reduction = baseline_spans[source] - source_after
                direction.update(
                    {
                        "candidate_type": "vertebral_body_lobe",
                        "selected_voxels_before_distance_trim": pretrim_voxels,
                        "selected_voxels": int(np.count_nonzero(move)),
                        "selected_volume_mm3": selected_volume_mm3,
                        "source_span_before_mm": baseline_spans[source],
                        "source_span_after_mm": source_after,
                        "target_span_before_mm": baseline_spans[target],
                        "target_span_after_mm": target_after,
                        "source_span_reduction_mm": reduction,
                    }
                )
                if selected_volume_mm3 >= active.minimum_trimmed_volume_mm3:
                    accepted_masks.append((move, source, target, "vertebral_body_lobe"))
                    reserved |= move
                    direction["accepted"] = True
                    direction["reason"] = "deep_body_valley_and_interface_supported"
                elif not bool(direction["accepted"]):
                    direction["reason"] = "trimmed_body_lobe_below_physical_volume_gate"
            pair_record["directions"].append(direction)
        pair_records.append(pair_record)
    refined_local = baseline.copy()
    actions: List[Dict[str, object]] = []
    for move, source, target, kind in accepted_masks:
        valid = move & (baseline == source)
        refined_local[valid] = target
        actions.append(
            {
                "source": source,
                "source_name": GEOMETRY_LABEL_NAMES[source],
                "target": target,
                "target_name": GEOMETRY_LABEL_NAMES[target],
                "kind": kind,
                "changed_voxels": int(np.count_nonzero(valid)),
            }
        )
    large_action_count = int(sum((int(item["changed_voxels"] > 0) for item in actions)))
    fragment_gate_activated = bool(
        large_action_count >= active.fragment_reattachment_minimum_large_actions
    )
    if fragment_gate_activated:
        refined_local, fragment_actions = _boundary_reattach_secondary_components(
            refined_local, baseline, spacing, active
        )
        actions.extend(fragment_actions)
    changed_local = refined_local != baseline
    target_voxels = max(
        int(np.count_nonzero(np.isin(baseline, GEOMETRY_TARGET_LABELS))), 1
    )
    baseline_main_fractions = {
        label: _boundary_main_component_fraction(baseline == label)
        for label in GEOMETRY_TARGET_LABELS
    }
    refined_main_fractions = {
        label: _boundary_main_component_fraction(refined_local == label)
        for label in GEOMETRY_TARGET_LABELS
    }
    checks = {
        "foreground_preserved": bool(np.array_equal(refined_local != 0, baseline != 0)),
        "only_target_labels_changed": bool(
            np.all(np.isin(baseline[changed_local], GEOMETRY_TARGET_LABELS))
            and np.all(np.isin(refined_local[changed_local], GEOMETRY_TARGET_LABELS))
        ),
        "only_adjacent_exchanges": bool(
            np.all(
                np.abs(
                    refined_local[changed_local].astype(int)
                    - baseline[changed_local].astype(int)
                )
                == 1
            )
        ),
        "bounded_total_change": bool(
            np.count_nonzero(changed_local) / target_voxels
            <= active.maximum_total_changed_fraction
        ),
        "main_components_preserved": bool(
            all(
                (
                    refined_main_fractions[label]
                    >= baseline_main_fractions[label]
                    - active.maximum_main_component_fraction_drop
                    for label in GEOMETRY_TARGET_LABELS
                )
            )
        ),
    }
    passed = bool(all(checks.values()))
    output = original.copy()
    if passed:
        output[crop] = refined_local
    else:
        actions = []
    return (
        output,
        {
            "method": "evidence_gated_thoracic_topology_refinement",
            "configuration": asdict(active),
            "crop": [[part.start, part.stop] for part in crop],
            "reference_span_mm": reference_span,
            "baseline_spans_mm": {
                GEOMETRY_LABEL_NAMES[label]: value
                for label, value in baseline_spans.items()
            },
            "baseline_main_component_fractions": {
                GEOMETRY_LABEL_NAMES[label]: value
                for label, value in baseline_main_fractions.items()
            },
            "refined_main_component_fractions": {
                GEOMETRY_LABEL_NAMES[label]: value
                for label, value in refined_main_fractions.items()
            },
            "pairs": pair_records,
            "fragment_reattachment_gate": {
                "large_action_count": large_action_count,
                "minimum_required": active.fragment_reattachment_minimum_large_actions,
                "activated": fragment_gate_activated,
            },
            "actions": actions,
            "changed_voxels": int(np.count_nonzero(output != original)),
            "case_checks": checks,
            "topology_stage_accepted": passed,
            "rollback_reason": None if passed else "one_or_more_case_checks_failed",
        },
    )


ADJACENCY_ALL_LABELS = tuple(range(1, 25))
ADJACENCY_FULL = np.ones((3, 3, 3), dtype=bool)


def _adjacency_label_name(label: int) -> str:
    if 1 <= label <= 5:
        return f"L{6 - label}"
    if 6 <= label <= 17:
        return f"T{18 - label}"
    if 18 <= label <= 24:
        return f"C{25 - label}"
    raise ValueError(f"unsupported vertebral label: {label}")


def _adjacency_anatomical_region(label: int) -> str:
    if label <= 5:
        return "lumbar"
    if label <= 17:
        return "thoracic"
    return "cervical"


def _adjacency_pair_region(low_label: int) -> str:
    low_region = _adjacency_anatomical_region(low_label)
    high_region = _adjacency_anatomical_region(low_label + 1)
    return low_region if low_region == high_region else "transition"


@dataclass
class RefinementConfig:
    context_levels: int = 6
    maximum_total_changed_fraction: float = 0.12
    maximum_main_component_fraction_drop: float = 0.002
    lumbar_minimum_volume_mm3: float = 700.0
    thoracic_minimum_volume_mm3: float = 500.0
    cervical_minimum_volume_mm3: float = 180.0
    transition_minimum_volume_mm3: float = 350.0
    lumbar_posterior_offset_mm: float = 18.0
    thoracic_posterior_offset_mm: float = 15.0
    cervical_posterior_offset_mm: float = 8.0
    transition_posterior_offset_mm: float = 11.0
    minimum_consecutive_supported_pairs: int = 3
    supported_pair_extension: int = 1


def _adjacency_supported_pair_ids(
    candidate_pairs: Iterable[int], active: RefinementConfig
) -> Tuple[List[int], List[List[int]]]:
    ordered = sorted(set((int(pair) for pair in candidate_pairs)))
    runs: List[List[int]] = []
    for pair in ordered:
        if (
            runs
            and pair == runs[-1][-1] + 1
            and (_adjacency_pair_region(pair) == _adjacency_pair_region(runs[-1][-1]))
        ):
            runs[-1].append(pair)
        else:
            runs.append([pair])
    cores = [
        run for run in runs if len(run) >= active.minimum_consecutive_supported_pairs
    ]
    supported = {pair for run in cores for pair in run}
    for run in cores:
        region = _adjacency_pair_region(run[0])
        for distance in range(1, active.supported_pair_extension + 1):
            for pair in (run[0] - distance, run[-1] + distance):
                if 1 <= pair <= 23 and _adjacency_pair_region(pair) == region:
                    supported.add(pair)
    return (sorted(supported), cores)


def _adjacency_contiguous_runs(labels: Iterable[int]) -> List[List[int]]:
    ordered = sorted(set((int(x) for x in labels if x in ADJACENCY_ALL_LABELS)))
    if not ordered:
        return []
    runs: List[List[int]] = [[ordered[0]]]
    for label in ordered[1:]:
        if label == runs[-1][-1] + 1:
            runs[-1].append(label)
        else:
            runs.append([label])
    return runs


def _adjacency_context_windows(
    run: Sequence[int], width: int
) -> List[Dict[str, object]]:
    if len(run) < width:
        return []
    windows: List[Dict[str, object]] = []
    next_pair = int(run[0])
    final_pair = int(run[-1]) - 1
    while next_pair <= final_pair:
        start = min(next_pair, int(run[-1]) - width + 1)
        start = max(start, int(run[0]))
        levels = list(range(start, start + width))
        last_owned = min(start + width - 2, final_pair)
        if start < next_pair:
            first_owned = next_pair
        else:
            first_owned = start
        owned_pairs = [
            (label, label + 1) for label in range(first_owned, last_owned + 1)
        ]
        windows.append({"levels": levels, "owned_pairs": owned_pairs})
        next_pair = last_owned + 1
    return windows


def _adjacency_crop_for_labels(
    data: np.ndarray, labels: Sequence[int], margin: int = 5
) -> Tuple[slice, slice, slice]:
    points = np.argwhere(np.isin(data, labels))
    low = np.maximum(points.min(axis=0) - margin, 0)
    high = np.minimum(points.max(axis=0) + margin + 1, data.shape)
    return tuple((slice(int(a), int(b)) for a, b in zip(low, high)))


def _adjacency_window_config(
    levels: Sequence[int],
    owned_pairs: Sequence[Tuple[int, int]],
    active: RefinementConfig,
) -> BoundaryConfig:
    regions = {
        _adjacency_anatomical_region(label) for pair in owned_pairs for label in pair
    }
    if regions == {"lumbar"}:
        minimum_volume = active.lumbar_minimum_volume_mm3
        posterior_offset = active.lumbar_posterior_offset_mm
        branch_radius, branch_lateral, primary_voxels = (34.0, 18.0, 120)
        distance_bias = -1.0
    elif regions == {"thoracic"}:
        minimum_volume = active.thoracic_minimum_volume_mm3
        posterior_offset = active.thoracic_posterior_offset_mm
        branch_radius, branch_lateral, primary_voxels = (30.0, 15.0, 100)
        distance_bias = 0.0
    elif regions == {"cervical"}:
        minimum_volume = active.cervical_minimum_volume_mm3
        posterior_offset = active.cervical_posterior_offset_mm
        branch_radius, branch_lateral, primary_voxels = (20.0, 9.0, 45)
        distance_bias = -0.5
    else:
        minimum_volume = active.transition_minimum_volume_mm3
        posterior_offset = active.transition_posterior_offset_mm
        branch_radius, branch_lateral, primary_voxels = (24.0, 11.0, 60)
        distance_bias = -0.5
    return BoundaryConfig(
        minimum_trimmed_volume_mm3=minimum_volume,
        posterior_offset_mm=posterior_offset,
        primary_min_voxels=primary_voxels,
        competing_distance_bias_mm=distance_bias,
        branch_radius_mm=branch_radius,
        branch_lateral_offset_mm=branch_lateral,
        maximum_total_changed_fraction=active.maximum_total_changed_fraction,
        maximum_main_component_fraction_drop=active.maximum_main_component_fraction_drop,
    )


def _adjacency_main_component_fraction_local(mask: np.ndarray) -> float:
    points = np.argwhere(mask)
    if not len(points):
        return 0.0
    low = np.maximum(points.min(axis=0) - 1, 0)
    high = np.minimum(points.max(axis=0) + 2, mask.shape)
    crop = tuple((slice(int(a), int(b)) for a, b in zip(low, high)))
    components, count = ndimage.label(mask[crop], structure=ADJACENCY_FULL)
    if not count:
        return 0.0
    sizes = np.bincount(components.ravel())[1:]
    return float(np.max(sizes) / max(np.sum(sizes), 1))


def refine_adjacent_boundaries(
    segmentation: np.ndarray, affine: np.ndarray, config: RefinementConfig | None = None
) -> Tuple[np.ndarray, Dict[str, object]]:
    active = config or RefinementConfig()
    original = np.asarray(segmentation, dtype=np.uint8)
    present = [label for label in ADJACENCY_ALL_LABELS if np.any(original == label)]
    runs = _adjacency_contiguous_runs(present)
    windows = [
        window
        for run in runs
        for window in _adjacency_context_windows(run, active.context_levels)
    ]
    proposal = np.zeros(original.shape, dtype=np.uint8)
    proposal_pair = np.zeros(original.shape, dtype=np.uint8)
    conflicts = np.zeros(original.shape, dtype=bool)
    window_reports: List[Dict[str, object]] = []
    candidate_pair_ids: List[int] = []
    for window_id, window in enumerate(windows, start=1):
        levels = list(window["levels"])
        owned_pairs = [tuple(pair) for pair in window["owned_pairs"]]
        crop = _adjacency_crop_for_labels(original, levels)
        local = original[crop]
        encoded = np.zeros(local.shape, dtype=np.uint8)
        actual_to_surrogate = {actual: 7 + index for index, actual in enumerate(levels)}
        surrogate_to_actual = {value: key for key, value in actual_to_surrogate.items()}
        for actual, surrogate in actual_to_surrogate.items():
            encoded[local == actual] = surrogate
        crop_start = np.asarray([part.start for part in crop], dtype=float)
        local_affine = np.asarray(affine, dtype=float) @ nib.affines.from_matvec(
            np.eye(3), crop_start
        )
        boundary_config = _adjacency_window_config(levels, owned_pairs, active)
        refined, report = refine_boundary_window(encoded, local_affine, boundary_config)
        changed = refined != encoded
        accepted_voxels = 0
        translated_actions: List[Dict[str, object]] = []
        for source_surrogate in range(7, 13):
            source_actual = surrogate_to_actual[source_surrogate]
            for target_surrogate in (source_surrogate - 1, source_surrogate + 1):
                if target_surrogate not in surrogate_to_actual:
                    continue
                target_actual = surrogate_to_actual[target_surrogate]
                pair = tuple(sorted((source_actual, target_actual)))
                if pair not in owned_pairs:
                    continue
                move = (
                    changed
                    & (encoded == source_surrogate)
                    & (refined == target_surrogate)
                )
                count = int(np.count_nonzero(move))
                if not count:
                    continue
                local_proposal = proposal[crop]
                local_proposal_pair = proposal_pair[crop]
                local_conflicts = conflicts[crop]
                disagreement = (
                    move & (local_proposal != 0) & (local_proposal != target_actual)
                )
                local_conflicts[disagreement] = True
                available = move & (local_proposal == 0) & ~local_conflicts
                local_proposal[available] = target_actual
                local_proposal_pair[available] = pair[0]
                local_proposal[local_conflicts] = 0
                local_proposal_pair[local_conflicts] = 0
                accepted = int(np.count_nonzero(available))
                accepted_voxels += accepted
                candidate_pair_ids.append(pair[0])
                translated_actions.append(
                    {
                        "source": source_actual,
                        "source_name": _adjacency_label_name(source_actual),
                        "target": target_actual,
                        "target_name": _adjacency_label_name(target_actual),
                        "proposed_voxels": count,
                        "accepted_nonconflicting_voxels": accepted,
                    }
                )
        window_reports.append(
            {
                "window_id": window_id,
                "levels": [
                    {"label": x, "name": _adjacency_label_name(x)} for x in levels
                ],
                "owned_pairs": [list(pair) for pair in owned_pairs],
                "regions": sorted({_adjacency_anatomical_region(x) for x in levels}),
                "boundary_configuration": asdict(boundary_config),
                "boundary_stage_accepted": bool(report["topology_stage_accepted"]),
                "boundary_changed_voxels_before_pair_ownership": int(
                    report["changed_voxels"]
                ),
                "accepted_voxels": accepted_voxels,
                "actions": translated_actions,
            }
        )
    supported_pairs, supporting_runs = _adjacency_supported_pair_ids(
        candidate_pair_ids, active
    )
    output = original.copy()
    raw_change_mask = (proposal > 0) & ~conflicts
    change_mask = raw_change_mask & np.isin(proposal_pair, supported_pairs)
    output[change_mask] = proposal[change_mask]
    changed_labels = sorted(
        set(original[change_mask].astype(int).tolist())
        | set(output[change_mask].astype(int).tolist())
    )
    foreground_voxels = max(
        int(np.count_nonzero(np.isin(original, ADJACENCY_ALL_LABELS))), 1
    )
    baseline_fractions = {
        label: _adjacency_main_component_fraction_local(original == label)
        for label in changed_labels
    }
    refined_fractions = {
        label: _adjacency_main_component_fraction_local(output == label)
        for label in changed_labels
    }
    source = original[change_mask]
    target = output[change_mask]
    checks = {
        "foreground_preserved": bool(np.all(source > 0) and np.all(target > 0)),
        "only_known_vertebral_labels_changed": bool(
            np.all(np.isin(source, ADJACENCY_ALL_LABELS))
            and np.all(np.isin(target, ADJACENCY_ALL_LABELS))
        ),
        "only_adjacent_exchanges": bool(
            np.all(np.abs(target.astype(int) - source.astype(int)) == 1)
        ),
        "no_conflicting_proposal_applied": bool(np.all(~(change_mask & conflicts))),
        "bounded_total_change": bool(
            np.count_nonzero(change_mask) / foreground_voxels
            <= active.maximum_total_changed_fraction
        ),
        "main_components_preserved": bool(
            all(
                (
                    refined_fractions[label]
                    >= baseline_fractions[label]
                    - active.maximum_main_component_fraction_drop
                    for label in changed_labels
                )
            )
        ),
    }
    passed = bool(all(checks.values()))
    if not passed:
        output = original.copy()
        change_mask = np.zeros(original.shape, dtype=bool)
    short_runs = [run for run in runs if len(run) < active.context_levels]
    return (
        output,
        {
            "method": "evidence_gated_general_vertebral_topology_refinement",
            "configuration": asdict(active),
            "present_labels": [
                {"label": x, "name": _adjacency_label_name(x)} for x in present
            ],
            "contiguous_runs": runs,
            "short_runs_abstained": short_runs,
            "windows": window_reports,
            "candidate_pair_ids": candidate_pair_ids,
            "supporting_pair_runs": supporting_runs,
            "supported_pair_ids": supported_pairs,
            "suppressed_isolated_candidate_voxels": int(
                np.count_nonzero(raw_change_mask & ~change_mask)
            ),
            "conflicting_proposal_voxels": int(np.count_nonzero(conflicts)),
            "changed_voxels": int(np.count_nonzero(change_mask)),
            "changed_labels": [
                {"label": x, "name": _adjacency_label_name(x)} for x in changed_labels
            ],
            "baseline_main_component_fractions": {
                _adjacency_label_name(label): value
                for label, value in baseline_fractions.items()
            },
            "refined_main_component_fractions": {
                _adjacency_label_name(label): value
                for label, value in refined_fractions.items()
            },
            "case_checks": checks,
            "topology_stage_accepted": passed,
            "rollback_reason": None if passed else "one_or_more_global_checks_failed",
        },
    )


def process_array(segmentation, affine, stage_one_config=None, stage_two_config=None):
    source = np.asarray(segmentation, dtype=np.uint8)
    intermediate, first_report = refine_shape(source, affine, config=stage_one_config)
    final, second_report = refine_adjacent_boundaries(
        intermediate, affine, config=stage_two_config
    )
    return (
        final,
        {
            "method": "global_sequence_then_general_adjacent_boundary_refinement",
            "global_sequence_stage": first_report,
            "adjacent_boundary_stage": second_report,
            "input_foreground_voxels": int(np.count_nonzero(source)),
            "output_foreground_voxels": int(np.count_nonzero(final)),
            "changed_voxels": int(np.count_nonzero(source != final)),
        },
    )


def process_case(input_case, output_case, combined_only=False):
    source_path = Path(input_case) / "combined_labels.nii.gz"
    image = nib.load(str(source_path))
    source = np.asarray(image.dataobj, dtype=np.uint8)
    _io_validate_labels(source, source_path)
    final, report = process_array(source, image.affine)
    output_case = Path(output_case)
    if combined_only:
        _io_save_like(final, image, output_case / "combined_labels.nii.gz")
    else:
        _io_save_case(final, image, output_case)
    report.update(
        {
            "case_id": Path(input_case).name,
            "input": str(source_path),
            "output": str(output_case / "combined_labels.nii.gz"),
            "shape": [int(value) for value in image.shape],
            "orientation": [str(value) for value in nib.aff2axcodes(image.affine)],
        }
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Save only combined_labels.nii.gz instead of all 24 binary masks.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path for a detailed JSON processing record.",
    )
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--input-dir and --output-dir must be different")
    reports = []
    for case_dir in _io_iter_case_dirs(args.input_dir):
        report = process_case(
            case_dir, args.output_dir / case_dir.name, combined_only=args.combined_only
        )
        reports.append(report)
        print(
            f"{case_dir.name}: changed={report['changed_voxels']} output={report['output']}"
        )
    if not reports:
        parser.error(
            f"No case folders containing combined_labels.nii.gz were found under {args.input_dir}"
        )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps({"cases": reports}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
