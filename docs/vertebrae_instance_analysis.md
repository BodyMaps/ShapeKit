# Read-only vertebral instance analysis

`utils.vertebrae_instance_analysis` provides a diagnostic API for examining
physical vertebral instances, protected thick-core candidates, separation
between EDT-derived core extents, and anatomical-name sequence consistency.

The analyzer is intentionally separate from ShapeKit's vertebra
post-processing pipeline. It does not delete, fill, merge, relabel, or save
segmentation voxels, and importing it does not change ShapeKit runtime
behavior.

## Public API

```python
from utils.vertebrae_instance_analysis import (
    VertebralInstanceAnalysisConfig,
    analyze_vertebral_instances,
)

report = analyze_vertebral_instances(
    segmentation_dict,
    affine=affine,
    ordered_anatomical_names=(
        "vertebrae_L5",
        "vertebrae_L4",
        "vertebrae_L3",
        "vertebrae_L2",
        "vertebrae_L1",
        "vertebrae_T12",
        "vertebrae_T11",
        "vertebrae_T10",
        "vertebrae_T9",
        "vertebrae_T8",
        "vertebrae_T7",
        "vertebrae_T6",
        "vertebrae_T5",
        "vertebrae_T4",
        "vertebrae_T3",
        "vertebrae_T2",
        "vertebrae_T1",
        "vertebrae_C7",
        "vertebrae_C6",
        "vertebrae_C5",
        "vertebrae_C4",
        "vertebrae_C3",
        "vertebrae_C2",
        "vertebrae_C1",
    ),
    ct=None,
    config=VertebralInstanceAnalysisConfig(),
)
```

`ordered_anatomical_names` is always interpreted
**inferior-to-superior**. The API uses anatomical names rather than combined
label values, so it does not depend on a particular numeric class map.

## Input contract

### `segmentation_dict`

- A mapping from anatomical names to three-dimensional arrays.
- Nonzero values are treated as foreground.
- All requested masks must have the same shape.
- Missing and empty masks are accepted.
- Keys not listed in `ordered_anatomical_names` are ignored.
- Input arrays are never modified.
- Overlapping requested masks are reported as ambiguous. Their overlap is not
  resolved using key order or a numeric-label priority.

### `affine`

- A finite, invertible 4×4 voxel-to-world affine.
- Physical spacing is derived from the affine.
- Axis permutations, axis sign changes, and in-plane rotations that preserve
  superior-inferior alignment are supported.
- Mild superior-inferior obliquity up to 20 degrees is supported. Material
  shear or greater SI obliquity produces an `unsupported_affine` unresolved
  report instead of silently using an inappropriate voxel axis.
- The final affine row must be approximately `[0, 0, 0, 1]`; a malformed
  homogeneous row raises `ValueError`.
- The 20-degree limit is intentionally conservative because discrete EDT and
  persistence measurements can change near decision thresholds on more
  oblique grids. Arbitrary orthogonal rotations are not claimed as supported.

### `ct`

CT means Computed Tomography input and is optional. When supplied, it must
already be aligned voxel-for-voxel with the segmentation masks, have the same
shape, and contain only finite intensity values. Shape mismatch or non-finite
values raise `ValueError`.

CT values contribute only bone-support confidence. CT never changes:

- the vertebral union;
- thick-core boundaries;
- physical-instance boundaries;
- thick-core candidate boundaries or core-separation measurements.

When CT is absent, the analyzer reports `ct_evidence: "unavailable"`, sets
per-instance confidence mode to `geometry_only`, and uses a stricter
geometry-only confidence threshold. When CT is present, low bone support is
reported explicitly as `low_ct_bone_support`; it can keep a candidate
unresolved, but it still cannot change candidate boundaries.

### `config`

All anatomical distances and volumes are configurable in physical units.
Dimensionless confidence and shape thresholds are also explicit in
`VertebralInstanceAnalysisConfig`.

All numeric configuration values must be finite. In particular,
`bone_hu_threshold` accepts finite positive, zero, or negative CT intensity
thresholds, while non-finite values raise `ValueError`.

The defaults are provisional, conservative diagnostic starting values. They
are not population-validated constants or claims of clinical validity or
population-level generalization. The effective configuration is included in
every report.

## Analysis stages

1. Validate masks, anatomical-name order, CT shape, and affine geometry.
2. Construct the unmodified union of requested vertebral masks.
3. Estimate a smooth physical spine trajectory from the largest component of
   each available named mask, using a polynomial least-squares fit followed by
   at most one distance-thresholded refit. This is a deterministic provisional
   estimator, not a fully robust trajectory estimator.
4. Run a spacing-aware distance transform and identify thick interior
   candidates.
5. Measure volume, persistence, compactness, trajectory distance, name
   composition, and optional CT bone support.
6. Analyze a label-independent thick-area profile for candidate peaks and
   valleys between EDT-derived thick cores.
7. Sort selected candidates by physical inferior-to-superior position.
8. Report duplicate, internal-missing, nonmonotonic, abnormal-spacing, overlap,
   and ambiguous-identity findings.

No stage produces a proposed correction.

## Protected and unresolved cores

A thick-core candidate is reported as `protected_high_confidence` only when
multiple
independent signals agree:

- sufficient thick-core physical volume;
- sufficient but not implausibly long superior-inferior persistence;
- compact transverse geometry;
- proximity to the estimated physical spine trajectory;
- a decisive anatomical-name vote;
- adequate CT bone support when CT is supplied;
- adequate separation evidence from neighboring thick cores.

Otherwise it remains unresolved. Unresolved statuses include:

- `unresolved_low_confidence`;
- `unresolved_mixed_identity`;
- `unresolved_overlap`;
- `unresolved_boundary_truncated`.

The words “protected” and “core” are diagnostic. They mean that a thick,
compact interior candidate has high-confidence identity evidence. They do not
guarantee that the region is a clinical vertebral-body core, and this module
still does not edit any mask.

## Rejected thick-core candidates

Every EDT-derived thick component is accounted for. A component farther than
`trajectory_tube_radius_mm` from the provisional trajectory is excluded from
accepted vertebral instances but retained in `rejected_candidates` with:

- a deterministic candidate ID;
- centroid in world millimetres;
- thick-core voxel count and physical volume;
- trajectory distance;
- status `rejected_unresolved`;
- reason `core_outside_trajectory_tube`.

At least one rejected candidate also produces an unresolved
`off_trajectory_core` anomaly, so the report cannot claim an unqualified
`continuous_sequence`. The anomaly links to the rejected records through
`affected_candidate_ids`; it does not represent them as accepted instances.

## Core-separation terminology

`inferior_core_separation_mm` and `superior_core_separation_mm` measure the
physical separation between extents of neighboring EDT-derived thick cores.
They are threshold-dependent diagnostic measurements. They are not anatomical
intervertebral foreground gaps or disc-space measurements.

## Partial field of view and missing identities

Missing endpoint names are not automatically treated as missing anatomy.

The analyzer reports an internal missing identity only when two consecutive,
high-confidence physical cores skip one or more anatomical names between
them. It does not infer missing identities:

- above the most superior confident core;
- below the most inferior confident core;
- through an unresolved physical core.

Foreground touching the physical inferior or superior array face is reported
as possible boundary truncation. Absence of boundary contact does not prove
that the anatomical field of view is complete; endpoint omissions can
therefore produce `extent_uncertain`.

Duplicate or missing patterns that could reflect transitional anatomy remain
diagnostic findings. The analyzer does not force a standard identity onto
transitional anatomy.

## Output contract

The return value contains JSON-compatible Python primitives only:

- dictionaries with string keys;
- lists;
- strings;
- integers;
- finite floats;
- booleans;
- `None`.

It contains no NumPy arrays, masks, voxel-index lists, NIfTI objects, or
corrected labels.

For canonical deterministic serialization:

```python
import json

canonical_json = json.dumps(
    report,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)
```

Instances are ordered inferior-to-superior and receive stable report-local IDs
such as `instance_001`. Anomalies and name-composition entries also use stable
sorting.

## Relationship to automatic vertebra re-identification

This module is an independent audit layer. It neither imports nor wraps an
automatic vertebra re-identification implementation.

It conceptually overlaps with upstream PR #5 in constructing a joint vertebral
mask, applying a physical distance transform, and ordering thick cores. PR #5
uses those signals for automatic offset voting, watershed rebuilding, cleanup,
and runtime relabeling. This module instead accepts configurable anatomical
names, never relabels voxels, preserves uncertainty, and returns deterministic
structured diagnostic evidence. It does not duplicate PR #5's automatic
correction path.

Potential future callers may use the report to decide whether additional
human review or a separately validated correction method is appropriate. Such
integration is outside this module's contract.

In particular, this analyzer does not:

- run a fixed global label offset;
- perform nearest-core watershed relabeling;
- suppress secondary connected components;
- fill holes;
- rebuild individual or combined NIfTI outputs;
- approve an automatic correction.

## Tests

The synthetic and API-contract suite requires no medical data:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -B -m unittest discover -s tests \
  -p 'test_vertebrae_instance_analysis.py' -v
```

The suite covers thick cores with thin appendages, posterior-like negative
controls, separated and merged instances, rejected off-trajectory candidates,
sequence anomalies, complete and partial overlaps, partial field of view,
optional and invalid CT, anisotropic spacing, supported mild obliquity,
unsupported geometry, fragmented masks, input non-mutation, deterministic
serialization, numeric-label independence, and multiprocessing consistency.

## Limitations

- Geometry alone cannot establish clinical anatomical identity in every case.
- CT evidence is limited to local intensity support; it is not a learned shape
  model.
- Strongly oblique or sheared grids are reported as unsupported rather than
  resampled internally.
- Distance, occupancy, and connected-component arrays scale with the cropped
  foreground volume. Component coordinates are extracted from deterministic
  `ndimage.find_objects` slices rather than rescanning the complete crop once
  per component. Clinical runtime and multiprocessing capacity still require
  workload-specific benchmarking.
- Transitional anatomy, merged vertebrae, severe leakage, fractures, implants,
  and incomplete scans may remain unresolved.
- The analyzer has not been demonstrated to generalize to unseen clinical
  populations.
- Diagnostic sequence anomalies are not equivalent to known segmentation
  errors or ground truth.
