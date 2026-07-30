"""Synthetic and API-contract tests for vertebrae_instance_analysis."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing
import unittest
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from utils.vertebrae_instance_analysis import (
    VertebralInstanceAnalysisConfig,
    analyze_vertebral_instances,
)


NAMES = ("vertebrae_L3", "vertebrae_L2", "vertebrae_L1")


def _canonical(report: Mapping[str, object]) -> str:
    return json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _ellipsoid(
    shape: Sequence[int],
    center: Sequence[float],
    radii: Sequence[float] = (11.0, 12.0, 8.0),
) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64)
    normalized = np.zeros(shape, dtype=np.float64)
    for axis in range(3):
        normalized += ((grid[axis] - center[axis]) / radii[axis]) ** 2
    return normalized <= 1.0


def _physical_ellipsoid(
    shape: Sequence[int],
    affine: np.ndarray,
    center_world: Sequence[float],
    radii_mm: Sequence[float] = (11.0, 12.0, 8.0),
) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    world = grid @ affine[:3, :3].T + affine[:3, 3]
    normalized = np.zeros(len(world), dtype=np.float64)
    for axis in range(3):
        normalized += (
            (world[:, axis] - center_world[axis]) / radii_mm[axis]
        ) ** 2
    return (normalized <= 1.0).reshape(tuple(shape))


def _body_with_thin_posterior(
    shape: Sequence[int], center: Sequence[float]
) -> np.ndarray:
    body = _ellipsoid(shape, center)
    x, y, z = np.indices(shape)
    process = (
        (np.abs(x - center[0]) <= 2)
        & (y >= int(round(center[1] + 9)))
        & (y <= int(round(center[1] + 20)))
        & (np.abs(z - center[2]) <= 2)
    )
    return body | process


def _standard_masks(
    *,
    shape: Sequence[int] = (48, 48, 112),
    centers_z: Sequence[float] = (24.0, 52.0, 80.0),
    names: Sequence[str] = NAMES,
    posterior: bool = False,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name, z_value in zip(names, centers_z):
        center = (24.0, 22.0, z_value)
        result[name] = (
            _body_with_thin_posterior(shape, center)
            if posterior
            else _ellipsoid(shape, center)
        )
    return result


def _fragmented_mask(
    *,
    shape: Sequence[int] = (112, 112, 112),
    positions: Sequence[float] = (12.0, 38.0, 64.0, 90.0),
) -> np.ndarray:
    mask = np.zeros(tuple(shape), dtype=bool)
    for center in itertools.product(positions, repeat=3):
        mask |= _ellipsoid(shape, center, radii=(9.0, 9.0, 9.0))
    return mask


def _rotation_about_x_affine(
    angle_degrees: float,
    pivot_world: Sequence[float],
) -> np.ndarray:
    angle = math.radians(angle_degrees)
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation
    pivot = np.asarray(pivot_world, dtype=np.float64)
    affine[:3, 3] = pivot - rotation @ pivot
    return affine


def _codes(report: Mapping[str, object]) -> Sequence[str]:
    return [item["anomaly_code"] for item in report["anomalies"]]  # type: ignore[index]


def _semantic_signature(report: Mapping[str, object]) -> Tuple[object, ...]:
    instances = report["instances"]  # type: ignore[index]
    return (
        report["observed_sequence_inferior_to_superior"],
        [item["status"] for item in instances],
        tuple(_codes(report)),
        report["overall_status"],
    )


def _array_digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _reorient(
    array: np.ndarray,
    affine: np.ndarray,
    permutation: Sequence[int],
    flips: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    transformed = np.transpose(array, axes=permutation)
    for axis, should_flip in enumerate(flips):
        if should_flip:
            transformed = np.flip(transformed, axis=axis)
    mapping = np.eye(4, dtype=np.float64)
    mapping[:3, :3] = 0.0
    mapping[:3, 3] = 0.0
    for new_axis, old_axis in enumerate(permutation):
        if flips[new_axis]:
            mapping[old_axis, new_axis] = -1.0
            mapping[old_axis, 3] = transformed.shape[new_axis] - 1
        else:
            mapping[old_axis, new_axis] = 1.0
    return transformed, affine @ mapping


def _multiprocessing_worker(payload: Tuple[Dict[str, np.ndarray], np.ndarray]) -> str:
    masks, affine = payload
    return _canonical(
        analyze_vertebral_instances(
            masks,
            affine=affine,
            ordered_anatomical_names=NAMES,
        )
    )


class VertebralInstanceAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.affine = np.eye(4, dtype=np.float64)

    def analyze(
        self,
        masks: Mapping[str, np.ndarray],
        *,
        names: Sequence[str] = NAMES,
        affine: np.ndarray = None,
        ct: np.ndarray = None,
        config: VertebralInstanceAnalysisConfig = None,
    ) -> Mapping[str, object]:
        return analyze_vertebral_instances(
            masks,
            affine=self.affine if affine is None else affine,
            ordered_anatomical_names=names,
            ct=ct,
            config=config,
        )

    def test_thick_core_survives_thin_posterior_appendage(self) -> None:
        masks = _standard_masks(centers_z=(52.0,), names=(NAMES[1],), posterior=True)
        report = self.analyze(masks)
        self.assertEqual(len(report["instances"]), 1)
        instance = report["instances"][0]
        self.assertEqual(instance["protected_core_identity"], NAMES[1])
        self.assertEqual(instance["status"], "protected_high_confidence")

    def test_two_separated_vertebrae_and_core_separation(self) -> None:
        masks = _standard_masks(centers_z=(32.0, 68.0), names=NAMES[:2])
        report = self.analyze(masks, names=NAMES[:2])
        self.assertEqual(
            report["observed_sequence_inferior_to_superior"], list(NAMES[:2])
        )
        self.assertGreater(
            report["instances"][0]["superior_core_separation_mm"],
            0.0,
        )
        self.assertEqual(report["overall_status"], "continuous_sequence")

    def test_core_separation_is_not_foreground_gap(self) -> None:
        masks = _standard_masks(
            centers_z=(32.0, 68.0),
            names=NAMES[:2],
        )
        report = self.analyze(masks, names=NAMES[:2])
        lower_z = np.flatnonzero(np.any(masks[NAMES[0]], axis=(0, 1)))
        upper_z = np.flatnonzero(np.any(masks[NAMES[1]], axis=(0, 1)))
        foreground_center_gap_mm = float(upper_z.min() - lower_z.max())
        core_separation = report["instances"][0][
            "superior_core_separation_mm"
        ]
        self.assertIsNotNone(core_separation)
        self.assertGreater(core_separation, foreground_center_gap_mm)
        self.assertNotIn("superior_gap_mm", report["instances"][0])

    def test_off_trajectory_thick_candidate_is_reported(self) -> None:
        shape = (150, 64, 80)
        central = _ellipsoid(
            shape,
            (30.0, 30.0, 40.0),
            radii=(13.0, 13.0, 9.0),
        )
        remote = _ellipsoid(
            shape,
            (105.0, 30.0, 40.0),
            radii=(10.0, 10.0, 8.0),
        )
        report = self.analyze(
            {NAMES[0]: central | remote},
            names=NAMES[:1],
        )
        self.assertEqual(len(report["instances"]), 1)
        self.assertEqual(len(report["rejected_candidates"]), 1)
        rejected = report["rejected_candidates"][0]
        self.assertEqual(
            rejected["reasons"],
            ["core_outside_trajectory_tube"],
        )
        self.assertGreater(rejected["core_voxel_count"], 0)
        self.assertGreater(rejected["core_volume_mm3"], 0.0)
        self.assertGreater(
            rejected["trajectory_distance_mm"],
            VertebralInstanceAnalysisConfig().trajectory_tube_radius_mm,
        )
        self.assertEqual(len(rejected["centroid_world_mm"]), 3)
        self.assertIn("off_trajectory_core", _codes(report))
        anomaly = next(
            item
            for item in report["anomalies"]
            if item["anomaly_code"] == "off_trajectory_core"
        )
        self.assertEqual(anomaly["affected_instance_ids"], [])
        self.assertEqual(
            anomaly["affected_candidate_ids"],
            [rejected["candidate_id"]],
        )
        self.assertEqual(report["overall_status"], "unresolved")

    def test_duplicate_identity(self) -> None:
        shape = (48, 48, 112)
        duplicate = _ellipsoid(shape, (24.0, 22.0, 30.0))
        duplicate |= _ellipsoid(shape, (24.0, 22.0, 72.0))
        report = self.analyze({NAMES[0]: duplicate})
        self.assertIn("duplicate_identity", _codes(report))
        self.assertEqual(
            report["observed_sequence_inferior_to_superior"],
            [NAMES[0], NAMES[0]],
        )

    def test_internal_missing_identity(self) -> None:
        masks = _standard_masks(
            centers_z=(28.0, 76.0), names=(NAMES[0], NAMES[2])
        )
        report = self.analyze(masks)
        self.assertIn("missing_internal_identity", _codes(report))
        missing = [
            anomaly
            for anomaly in report["anomalies"]
            if anomaly["anomaly_code"] == "missing_internal_identity"
        ]
        self.assertEqual(missing[0]["affected_anatomical_names"], [NAMES[1]])

    def test_endpoint_missing_labels_are_not_internal_missing(self) -> None:
        names = ("vertebrae_L4",) + NAMES + ("vertebrae_T12",)
        masks = _standard_masks()
        report = self.analyze(masks, names=names)
        self.assertNotIn("missing_internal_identity", _codes(report))
        self.assertEqual(report["field_of_view_status"], "extent_uncertain")

    def test_nonmonotonic_identity(self) -> None:
        shape = (48, 48, 100)
        masks = {
            NAMES[1]: _ellipsoid(shape, (24.0, 22.0, 28.0)),
            NAMES[0]: _ellipsoid(shape, (24.0, 22.0, 70.0)),
        }
        report = self.analyze(masks, names=NAMES[:2])
        self.assertIn("nonmonotonic_identity", _codes(report))

    def test_transitional_duplicate_missing_pattern_is_unresolved(self) -> None:
        shape = (48, 48, 120)
        masks = {
            NAMES[0]: (
                _ellipsoid(shape, (24.0, 22.0, 24.0))
                | _ellipsoid(shape, (24.0, 22.0, 58.0))
            ),
            NAMES[2]: _ellipsoid(shape, (24.0, 22.0, 92.0)),
        }
        report = self.analyze(masks)
        self.assertIn("duplicate_identity", _codes(report))
        self.assertIn("missing_internal_identity", _codes(report))
        self.assertEqual(report["overall_status"], "unresolved")
        ambiguous = [
            anomaly
            for anomaly in report["anomalies"]
            if anomaly["anomaly_code"] == "ambiguous_identity"
            and "transitional anatomy" in anomaly["explanation"]
        ]
        self.assertEqual(len(ambiguous), 1)
        self.assertEqual(ambiguous[0]["status"], "unresolved")

    def test_mixed_adjacent_identity_is_unresolved(self) -> None:
        shape = (48, 48, 96)
        body = _ellipsoid(shape, (24.0, 22.0, 48.0))
        x = np.indices(shape)[0]
        masks = {
            NAMES[0]: body & (x <= 24),
            NAMES[1]: body & (x > 24),
        }
        report = self.analyze(masks, names=NAMES[:2])
        self.assertEqual(len(report["instances"]), 1)
        self.assertIsNone(report["instances"][0]["protected_core_identity"])
        self.assertEqual(
            report["instances"][0]["status"], "unresolved_mixed_identity"
        )
        self.assertIn("ambiguous_identity", _codes(report))

    def test_overlapping_masks_are_unresolved(self) -> None:
        shape = (48, 48, 96)
        body = _ellipsoid(shape, (24.0, 22.0, 48.0))
        report = self.analyze(
            {NAMES[0]: body, NAMES[1]: body.copy()}, names=NAMES[:2]
        )
        self.assertGreater(report["input_overlap_voxel_count"], 0)
        self.assertEqual(report["instances"][0]["status"], "unresolved_overlap")
        self.assertIn("overlapping_input_masks", _codes(report))

    def test_partially_overlapping_masks_are_unresolved(self) -> None:
        shape = (48, 48, 96)
        body = _ellipsoid(shape, (24.0, 22.0, 48.0))
        x = np.indices(shape)[0]
        masks = {
            NAMES[0]: body & (x <= 26),
            NAMES[1]: body & (x >= 22),
        }
        report = self.analyze(masks, names=NAMES[:2])
        expected_overlap = int(
            np.count_nonzero(masks[NAMES[0]] & masks[NAMES[1]])
        )
        self.assertGreater(expected_overlap, 0)
        self.assertEqual(
            report["input_overlap_voxel_count"],
            expected_overlap,
        )
        self.assertEqual(
            report["instances"][0]["status"],
            "unresolved_overlap",
        )

    def test_merged_or_weak_core_separation_is_unresolved(self) -> None:
        shape = (56, 56, 112)
        lower = _ellipsoid(shape, (28.0, 26.0, 30.0))
        upper = _ellipsoid(shape, (28.0, 26.0, 78.0))
        x, y, z = np.indices(shape)
        bridge = (
            ((x - 28.0) ** 2 + (y - 26.0) ** 2 <= 7.0**2)
            & (z >= 30)
            & (z <= 78)
        )
        masks = {
            NAMES[0]: lower | (bridge & (z < 54)),
            NAMES[1]: upper | (bridge & (z >= 54)),
        }
        report = self.analyze(masks, names=NAMES[:2])
        self.assertEqual(len(report["instances"]), 1)
        self.assertNotEqual(
            report["instances"][0]["status"], "protected_high_confidence"
        )
        self.assertTrue(
            {
                "possible_merged_instance",
                "multiple_body_profile_peaks_in_one_core",
            }
            & set(report["instances"][0]["reasons"])
        )

    def test_thick_posterior_like_candidate_is_unresolved(self) -> None:
        shape = (72, 88, 96)
        elongated = _ellipsoid(
            shape,
            (36.0, 44.0, 48.0),
            radii=(7.0, 20.0, 8.0),
        )
        report = self.analyze(
            {NAMES[1]: elongated},
            names=(NAMES[1],),
        )
        self.assertEqual(len(report["instances"]), 1)
        instance = report["instances"][0]
        self.assertNotEqual(
            instance["status"],
            "protected_high_confidence",
        )
        self.assertIsNone(instance["protected_core_identity"])
        self.assertIn(
            "core_compactness_below_threshold",
            instance["reasons"],
        )

    def test_inferior_partial_field_of_view(self) -> None:
        shape = (48, 48, 96)
        masks = {
            NAMES[0]: _ellipsoid(shape, (24.0, 22.0, 2.0)),
            NAMES[1]: _ellipsoid(shape, (24.0, 22.0, 38.0)),
        }
        report = self.analyze(masks, names=NAMES[:2])
        self.assertEqual(
            report["field_of_view_status"], "inferior_boundary_truncated"
        )
        self.assertEqual(
            report["instances"][0]["status"], "unresolved_boundary_truncated"
        )
        self.assertIsNone(report["instances"][0]["protected_core_identity"])
        self.assertNotIn("missing_internal_identity", _codes(report))

    def test_superior_partial_field_of_view(self) -> None:
        shape = (48, 48, 96)
        masks = {
            NAMES[0]: _ellipsoid(shape, (24.0, 22.0, 52.0)),
            NAMES[1]: _ellipsoid(shape, (24.0, 22.0, 94.0)),
        }
        report = self.analyze(masks, names=NAMES[:2])
        self.assertEqual(
            report["field_of_view_status"], "superior_boundary_truncated"
        )
        self.assertEqual(
            report["instances"][-1]["status"], "unresolved_boundary_truncated"
        )
        self.assertIsNone(report["instances"][-1]["protected_core_identity"])
        self.assertNotIn("missing_internal_identity", _codes(report))

    def test_empty_input(self) -> None:
        report = self.analyze({})
        self.assertEqual(report["overall_status"], "empty_input")
        self.assertEqual(report["instances"], [])
        self.assertEqual(report["shape"], [])

    def test_ct_absent_and_present(self) -> None:
        masks = _standard_masks(centers_z=(52.0,), names=(NAMES[1],))
        geometry = self.analyze(masks)
        ct = np.full(next(iter(masks.values())).shape, -100.0, dtype=np.float32)
        ct[next(iter(masks.values()))] = 300.0
        supported = self.analyze(masks, ct=ct)
        self.assertEqual(geometry["ct_evidence"], "unavailable")
        self.assertEqual(supported["ct_evidence"], "used")
        self.assertEqual(len(geometry["instances"]), len(supported["instances"]))
        for geometry_instance, supported_instance in zip(
            geometry["instances"], supported["instances"]
        ):
            for field in (
                "centroid_world_mm",
                "core_voxel_count",
                "core_volume_mm3",
                "maximum_internal_thickness_mm",
                "persistence_mm",
                "inferior_core_separation_mm",
                "superior_core_separation_mm",
            ):
                self.assertEqual(
                    geometry_instance[field], supported_instance[field]
                )
        self.assertIsNone(geometry["instances"][0]["ct_bone_support_fraction"])
        self.assertGreaterEqual(
            supported["instances"][0]["ct_bone_support_fraction"], 0.99
        )

    def test_ct_shape_mismatch(self) -> None:
        masks = _standard_masks(centers_z=(52.0,), names=(NAMES[1],))
        with self.assertRaisesRegex(ValueError, "ct shape differs"):
            self.analyze(masks, ct=np.zeros((3, 4, 5), dtype=np.float32))

    def test_low_ct_bone_support_is_explicit_and_geometry_is_stable(
        self,
    ) -> None:
        masks = _standard_masks(
            centers_z=(52.0,),
            names=(NAMES[1],),
        )
        geometry = self.analyze(masks)
        low_ct = np.full(
            next(iter(masks.values())).shape,
            -100.0,
            dtype=np.float32,
        )
        low_support = self.analyze(masks, ct=low_ct)
        geometry_instance = geometry["instances"][0]
        low_instance = low_support["instances"][0]
        for field in (
            "centroid_world_mm",
            "core_voxel_count",
            "core_volume_mm3",
            "maximum_internal_thickness_mm",
            "persistence_mm",
            "inferior_core_separation_mm",
            "superior_core_separation_mm",
        ):
            self.assertEqual(
                geometry_instance[field],
                low_instance[field],
            )
        self.assertEqual(low_support["ct_evidence"], "used")
        self.assertEqual(
            low_instance["ct_bone_support_fraction"],
            0.0,
        )
        self.assertIn("low_ct_bone_support", low_instance["reasons"])
        self.assertNotEqual(
            low_instance["status"],
            "protected_high_confidence",
        )

    def test_nonfinite_ct_is_rejected(self) -> None:
        masks = _standard_masks(
            centers_z=(52.0,),
            names=(NAMES[1],),
        )
        ct = np.zeros(
            next(iter(masks.values())).shape,
            dtype=np.float32,
        )
        ct[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.analyze(masks, ct=ct)

    def test_anisotropic_spacing_is_physically_stable(self) -> None:
        center_worlds = ((24.0, 22.0, 28.0), (24.0, 22.0, 72.0))
        affine_a = np.eye(4, dtype=np.float64)
        shape_a = (48, 48, 104)
        masks_a = {
            name: _physical_ellipsoid(shape_a, affine_a, center)
            for name, center in zip(NAMES[:2], center_worlds)
        }
        affine_b = np.diag([2.0, 1.0, 2.0, 1.0])
        shape_b = (24, 48, 52)
        masks_b = {
            name: _physical_ellipsoid(shape_b, affine_b, center)
            for name, center in zip(NAMES[:2], center_worlds)
        }
        report_a = self.analyze(masks_a, names=NAMES[:2], affine=affine_a)
        report_b = self.analyze(masks_b, names=NAMES[:2], affine=affine_b)
        self.assertEqual(_semantic_signature(report_a), _semantic_signature(report_b))
        for first, second in zip(report_a["instances"], report_b["instances"]):
            error = np.linalg.norm(
                np.asarray(first["centroid_world_mm"])
                - np.asarray(second["centroid_world_mm"])
            )
            self.assertLessEqual(error, np.linalg.norm([2.0, 1.0, 2.0]))

    def test_axis_permutations_and_sign_flips(self) -> None:
        base_masks = _standard_masks(centers_z=(30.0, 76.0), names=NAMES[:2])
        base = self.analyze(base_masks, names=NAMES[:2])
        base_centroids = [
            np.asarray(instance["centroid_world_mm"]) for instance in base["instances"]
        ]
        for permutation in itertools.permutations(range(3)):
            for flips in itertools.product((False, True), repeat=3):
                transformed: Dict[str, np.ndarray] = {}
                transformed_affine = None
                for name, mask in base_masks.items():
                    new_mask, new_affine = _reorient(
                        mask, self.affine, permutation, flips
                    )
                    transformed[name] = new_mask
                    transformed_affine = new_affine
                report = self.analyze(
                    transformed,
                    names=NAMES[:2],
                    affine=transformed_affine,
                )
                self.assertEqual(_semantic_signature(base), _semantic_signature(report))
                for expected, instance in zip(base_centroids, report["instances"]):
                    np.testing.assert_allclose(
                        expected,
                        instance["centroid_world_mm"],
                        atol=1e-6,
                        rtol=0,
                    )

    def test_unsupported_affine_shear(self) -> None:
        affine = np.eye(4, dtype=np.float64)
        affine[0, 1] = 0.25
        report = self.analyze(
            _standard_masks(centers_z=(52.0,), names=(NAMES[1],)),
            affine=affine,
        )
        self.assertEqual(report["overall_status"], "unresolved")
        self.assertEqual(_codes(report), ["unsupported_affine"])
        self.assertEqual(report["instances"], [])

    def test_equivalent_29_degree_phantom_is_unresolved(self) -> None:
        shape = (64, 64, 80)
        affine = _rotation_about_x_affine(
            29.0,
            pivot_world=(32.0, 32.0, 40.0),
        )
        centers = ((32.0, 32.0, 28.0), (32.0, 32.0, 52.0))
        masks = {
            name: _physical_ellipsoid(shape, affine, center)
            for name, center in zip(NAMES[:2], centers)
        }
        report = self.analyze(
            masks,
            names=NAMES[:2],
            affine=affine,
        )
        self.assertEqual(_codes(report), ["unsupported_affine"])
        self.assertEqual(report["overall_status"], "unresolved")
        self.assertEqual(report["instances"], [])

    def test_invalid_affine_homogeneous_row_raises(self) -> None:
        affine = np.eye(4, dtype=np.float64)
        affine[3, 0] = 0.2
        with self.assertRaisesRegex(ValueError, "homogeneous row"):
            self.analyze(
                _standard_masks(
                    centers_z=(52.0,),
                    names=(NAMES[1],),
                ),
                affine=affine,
            )

    def test_orthogonal_in_plane_rotation_has_unique_axis_codes(self) -> None:
        angle = math.radians(35.0)
        affine = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0, 0.0],
                [math.sin(angle), math.cos(angle), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        report = self.analyze(
            _standard_masks(centers_z=(30.0, 76.0), names=NAMES[:2]),
            names=NAMES[:2],
            affine=affine,
        )
        axis_families = [
            "x" if code in ("L", "R") else "y" if code in ("P", "A") else "z"
            for code in report["orientation_axcodes"]
        ]
        self.assertEqual(set(axis_families), {"x", "y", "z"})
        self.assertEqual(
            report["observed_sequence_inferior_to_superior"], list(NAMES[:2])
        )

    def test_equivalent_axis_aligned_and_20_degree_phantoms_match(
        self,
    ) -> None:
        shape = (64, 64, 80)
        aligned_affine = np.eye(4, dtype=np.float64)
        oblique_affine = _rotation_about_x_affine(
            20.0,
            pivot_world=(32.0, 32.0, 40.0),
        )
        centers = ((32.0, 32.0, 24.0), (32.0, 32.0, 56.0))
        aligned_masks = {
            name: _physical_ellipsoid(shape, aligned_affine, center)
            for name, center in zip(NAMES[:2], centers)
        }
        oblique_masks = {
            name: _physical_ellipsoid(shape, oblique_affine, center)
            for name, center in zip(NAMES[:2], centers)
        }
        aligned = self.analyze(
            aligned_masks,
            names=NAMES[:2],
            affine=aligned_affine,
        )
        oblique = self.analyze(
            oblique_masks,
            names=NAMES[:2],
            affine=oblique_affine,
        )
        self.assertEqual(
            _semantic_signature(aligned),
            _semantic_signature(oblique),
        )
        self.assertNotIn("unsupported_affine", _codes(oblique))
        self.assertEqual(
            oblique["observed_sequence_inferior_to_superior"],
            list(NAMES[:2]),
        )

    def test_fragmented_volume_reports_every_thick_component(self) -> None:
        fragmented = _fragmented_mask()
        report = self.analyze(
            {NAMES[0]: fragmented},
            names=NAMES[:1],
        )
        total_candidates = (
            len(report["instances"])
            + len(report["rejected_candidates"])
        )
        self.assertEqual(total_candidates, 64)
        for candidate in report["rejected_candidates"]:
            self.assertEqual(
                candidate["reasons"],
                ["core_outside_trajectory_tube"],
            )

    def test_fragmented_candidate_order_is_deterministic(self) -> None:
        fragmented = _fragmented_mask(
            shape=(88, 88, 88),
            positions=(12.0, 38.0, 64.0),
        )
        masks = {NAMES[0]: fragmented}
        first = self.analyze(masks, names=NAMES[:1])
        second = self.analyze(masks, names=NAMES[:1])
        self.assertEqual(_canonical(first), _canonical(second))
        accepted_centroids = [
            tuple(item["centroid_world_mm"])
            for item in first["instances"]
        ]
        rejected_centroids = [
            tuple(item["centroid_world_mm"])
            for item in first["rejected_candidates"]
        ]
        self.assertEqual(
            accepted_centroids,
            sorted(
                accepted_centroids,
                key=lambda value: (value[2], value[1], value[0]),
            ),
        )
        self.assertEqual(
            rejected_centroids,
            sorted(
                rejected_centroids,
                key=lambda value: (value[2], value[1], value[0]),
            ),
        )

    def test_all_raised_exception_paths_preserve_inputs(self) -> None:
        base_masks = {
            name: mask.astype(np.uint8) * (index + 2)
            for index, (name, mask) in enumerate(
                _standard_masks().items()
            )
        }
        bad_shape_masks = dict(base_masks)
        bad_shape_masks[NAMES[1]] = np.zeros(
            (4, 5, 6),
            dtype=np.uint8,
        )
        bad_dimension_masks = dict(base_masks)
        bad_dimension_masks[NAMES[1]] = np.zeros(
            (4, 5),
            dtype=np.uint8,
        )
        nonfinite_ct = np.zeros(
            next(iter(base_masks.values())).shape,
            dtype=np.float32,
        )
        nonfinite_ct[0, 0, 0] = np.inf
        bad_row = np.eye(4, dtype=np.float64)
        bad_row[3, 1] = 0.5
        singular = np.zeros((4, 4), dtype=np.float64)
        singular[3, 3] = 1.0
        cases = (
            (
                "ct_shape",
                base_masks,
                np.eye(4),
                np.zeros((3, 4, 5), dtype=np.float32),
                NAMES,
                None,
            ),
            (
                "ct_nonfinite",
                base_masks,
                np.eye(4),
                nonfinite_ct,
                NAMES,
                None,
            ),
            (
                "homogeneous_row",
                base_masks,
                bad_row,
                None,
                NAMES,
                None,
            ),
            (
                "singular_affine",
                base_masks,
                singular,
                None,
                NAMES,
                None,
            ),
            (
                "mask_shape",
                bad_shape_masks,
                np.eye(4),
                None,
                NAMES,
                None,
            ),
            (
                "mask_dimension",
                bad_dimension_masks,
                np.eye(4),
                None,
                NAMES,
                None,
            ),
            (
                "empty_names",
                base_masks,
                np.eye(4),
                None,
                (),
                None,
            ),
            (
                "invalid_config",
                base_masks,
                np.eye(4),
                None,
                NAMES,
                VertebralInstanceAnalysisConfig(core_radius_mm=-1.0),
            ),
            (
                "unsupported_configured_obliquity",
                base_masks,
                np.eye(4),
                None,
                NAMES,
                VertebralInstanceAnalysisConfig(
                    max_si_axis_obliquity_degrees=29.0
                ),
            ),
        )
        for (
            case_name,
            masks,
            affine,
            ct,
            names,
            config,
        ) in cases:
            arrays = list(masks.values())
            if ct is not None:
                arrays.append(ct)
            arrays.append(affine)
            before = [
                (id(array), array.shape, array.dtype, _array_digest(array))
                for array in arrays
            ]
            with self.subTest(case=case_name):
                with self.assertRaises(ValueError):
                    self.analyze(
                        masks,
                        affine=affine,
                        ct=ct,
                        names=names,
                        config=config,
                    )
                after = [
                    (
                        id(array),
                        array.shape,
                        array.dtype,
                        _array_digest(array),
                    )
                    for array in arrays
                ]
                self.assertEqual(before, after)

    def test_nonfinite_bone_thresholds_preserve_inputs(self) -> None:
        masks = {
            name: mask.astype(np.uint8) * (index + 2)
            for index, (name, mask) in enumerate(
                _standard_masks().items()
            )
        }
        ct = np.full(
            next(iter(masks.values())).shape,
            300.0,
            dtype=np.float32,
        )
        arrays = [*masks.values(), ct]
        before = [
            (id(array), array.shape, array.dtype, _array_digest(array))
            for array in arrays
        ]
        for threshold in (
            float("nan"),
            float("inf"),
            float("-inf"),
        ):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError,
                    "Configuration values must be finite: "
                    "bone_hu_threshold",
                ):
                    self.analyze(
                        masks,
                        ct=ct,
                        config=VertebralInstanceAnalysisConfig(
                            bone_hu_threshold=threshold
                        ),
                    )
                after = [
                    (
                        id(array),
                        array.shape,
                        array.dtype,
                        _array_digest(array),
                    )
                    for array in arrays
                ]
                self.assertEqual(before, after)

    def test_finite_negative_bone_threshold_is_accepted(self) -> None:
        masks = _standard_masks(
            centers_z=(52.0,),
            names=(NAMES[1],),
        )
        ct = np.full(
            next(iter(masks.values())).shape,
            -100.0,
            dtype=np.float32,
        )
        report = self.analyze(
            masks,
            ct=ct,
            config=VertebralInstanceAnalysisConfig(
                bone_hu_threshold=-200.0
            ),
        )
        self.assertEqual(
            report["effective_config"]["bone_hu_threshold"],
            -200.0,
        )
        self.assertEqual(
            report["instances"][0]["ct_bone_support_fraction"],
            1.0,
        )
        _canonical(report)

    def test_zero_input_mutation(self) -> None:
        masks = {
            name: mask.astype(np.uint8) * (index + 2)
            for index, (name, mask) in enumerate(_standard_masks().items())
        }
        before = {
            name: (id(array), array.shape, array.dtype, _array_digest(array))
            for name, array in masks.items()
        }
        self.analyze(masks)
        after = {
            name: (id(array), array.shape, array.dtype, _array_digest(array))
            for name, array in masks.items()
        }
        self.assertEqual(before, after)

    def test_twenty_repeated_serializations_are_identical(self) -> None:
        masks = _standard_masks()
        serializations = {_canonical(self.analyze(masks)) for _ in range(20)}
        self.assertEqual(len(serializations), 1)

    def test_numeric_label_independence(self) -> None:
        binary = _standard_masks()
        low_values = {
            name: mask.astype(np.uint8) * (index + 1)
            for index, (name, mask) in enumerate(binary.items())
        }
        high_values = {
            name: mask.astype(np.uint16) * (200 + index)
            for index, (name, mask) in enumerate(binary.items())
        }
        self.assertEqual(
            _canonical(self.analyze(low_values)),
            _canonical(self.analyze(high_values)),
        )

    def test_multiprocessing_consistency(self) -> None:
        payload = (_standard_masks(), self.affine)
        expected = _multiprocessing_worker(payload)
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=2) as pool:
            observed = pool.map(_multiprocessing_worker, [payload, payload])
        self.assertEqual(observed, [expected, expected])

    def test_report_contains_only_primitives_and_no_corrected_output(
        self,
    ) -> None:
        report = self.analyze(_standard_masks())
        forbidden_key_terms = ("corrected", "segmentation", "voxel_indices")

        def inspect(value: object) -> None:
            self.assertNotIsInstance(
                value,
                (np.ndarray, np.generic, tuple, set),
            )
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertIsInstance(key, str)
                    self.assertFalse(
                        any(term in key.lower() for term in forbidden_key_terms)
                    )
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
            else:
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))
                self.assertIsInstance(
                    value, (str, int, float, bool, type(None))
                )

        inspect(report)
        self.assertIsInstance(_canonical(report), str)

    def test_module_has_no_image_writing_api(self) -> None:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "utils"
            / "vertebrae_instance_analysis.py"
        )
        source = module_path.read_text(encoding="utf-8")
        forbidden = (
            "nib.save",
            "nibabel.save",
            "to_filename(",
            "Nifti1Image(",
            "SimpleITK.WriteImage",
        )
        for token in forbidden:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
