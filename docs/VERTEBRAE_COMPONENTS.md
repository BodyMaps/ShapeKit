# Reviewed component refinement

`shapekit_components` integrates Chandan Krishna Yalavarthy's vertebra warmup
postprocessing as an optional ShapeKit engine. Existing engine defaults are
unchanged. It operates on the native voxel grid without model inference or
resampling. It requires an explicit review plan for each case.

## Scope and evidence

This is a **review-assisted implementation**, not an automatic vertebral
identification method. The warmup used reviewed eroded-core assignments and a
reviewed boundary split. Those decisions are represented as input data rather
than embedded case names or machine-specific paths. Plan creation is not visual
review, and neither seed order nor component size establishes anatomical truth.

The original two-case submission received a professor-reported average DSC of
93.4%, compared with 77.4% for the earlier submission. These are external
evaluation results supplied by the contributor, not scores computed by ShapeKit.
They do not establish accuracy on new cases. Reproducing masks is a software
regression check, not an independent accuracy evaluation.

Local regression against the submitted archive passed for both warmup cases:

| Case | Binary masks | Combined labels | Modified voxels vs baseline |
| --- | --- | --- | ---: |
| BDMAP_00000006 | All 24 identical | Identical after ID conversion | 484 |
| BDMAP_00000031 | All 24 identical | Identical after ID conversion | 364,964 |

Shape, affine, spacing, spatial units, and qform/sform metadata also matched.
All 50 reference NIfTI files were verified byte-for-byte against the submitted
archive. The submitted orchestration script also matched the local original
script byte-for-byte. The source archive SHA-256 was
`87c2998324bbb7bdb4c24419fc7863959e7c7c0c1886f91b39cb74f579086cac`; see the local regression report for provenance.

The adapter calls took approximately 141 and 605 seconds, including input/output
verification, on the contributor's local Windows environment. These timings are
observations from two cases, not a throughput guarantee.

## Algorithm

1. Remove a secondary component only when it occupies at most 5% of its label,
   the primary component occupies at least 80%, the bounding-box separation
   lower bound is at least 100 mm, and the secondary component does not touch
   the image boundary. Components use 26-connectivity.
2. In reviewed mode, reconstruct the reviewed L5-to-T1 seed cores by physical
   erosion of the cleaned foreground. Check component IDs, volumes, and original
   dominant names against the review metadata.
3. Relabel a component touching one reviewed seed identity; split a component
   touching multiple identities using minimum-cost paths restricted to it.
4. Reassign unseeded fragments only through a unique adjacent-level face
   contact with seed-supported regions: volume >=100 mm3 and contact >=20 mm2.
   Newly corrected fragments do not become support for further corrections.
5. Apply only explicitly reviewed boundary splits. Select the original component
   by label, rank, and expected voxel count. Create exclusive face-contact
   markers for the two reviewed adjacent target levels and split locally.

Geodesic costs are `1 + 4/(distance_mm + 0.5)^2`, with physical sampling along
each axis. After cleanup, relabeling preserves foreground and cervical labels.
The cleanup itself can remove cervical false-positive components.

## Requirements and input contract

- Use the normal ShapeKit environment. Dependencies are NumPy, SciPy, NiBabel,
  connected-components-3d, and scikit-image (already in requirements.txt).
- Regression environment: Python 3.9, NumPy 1.23.5, SciPy 1.13.1,
  scikit-image 0.24.0. Dependency changes can affect component numbering or
  shortest-path tie breaking; rerun regressions after upgrades.
- Supply all 24 binary, nonoverlapping vertebra masks under `segmentations/`.
  Empty masks are permitted for cleanup; reviewed correction needs its seed
  sequence. Every mask and CT must share geometry.
- The evaluated implementation requires axis-aligned LAS orientation and CT
  spatial units in millimetres. Other orientations are rejected rather than
  silently reinterpreted. CT voxel intensities are not used by this method.
- The adapter uses mask names to construct internal labels 1=L5 through 24=C1.
  Outputs use ShapeKit's configured class map (normally 26=L5 through 49=C1).
  Additional organ masks are copied unchanged; combined-map precedence follows
  ShapeKit's ascending class-ID convention.
- Use a new, disjoint output root. The engine requires `--vertebrae_only` and
  does not support `--continue_prediction`.

## Prepare review plans

Store plans outside the source checkout, named `<case_folder>.json`. A plan
contains a SHA-256 fingerprint of canonical input labels, shape, and affine.
A plan from a different input fails before corrections or output writing.

For a case explicitly selected for cleanup only:

```sh
python tools/prepare_component_plan.py --input-case /data/predictions/case_a --ct /data/ct/case_a/ct.nii.gz --mode cleanup_only --output /data/plans/case_a.json
```

For reviewed correction, supply the existing reviewed seed CSV:

```sh
python tools/prepare_component_plan.py --input-case /data/predictions/case_b --ct /data/ct/case_b/ct.nii.gz --mode reviewed --reviewed-seeds /data/review/case_b/proposed_seeds.csv --erosion-mm 5 --boundary-splits /data/review/case_b/splits.json --output /data/plans/case_b.json
```

CSV columns: `component_id`, `proposed_label`, `dominant_original_label`,
`core_volume_mm3`. Exactly 17 rows, ordered L5 through T1, are required. Component
IDs refer to 26-connected components of `distance > erosion_mm` within the
tightly cropped, zero-padded cleaned foreground. This CSV must come from review
of the same baseline with these cleanup settings. The preparation tool checks
the sequence; the engine validates the reconstructed cores when run.

The optional split file is a JSON list. Each entry specifies `original_label`
(internal integer ID), one-based size `rank`, exact expected `voxels`, and two
adjacent internal label IDs in `targets`. An empty list disables boundary splits.
Do not copy reviewed fragment selections or seed assignments to unrelated cases.

## Run through ShapeKit

Run from the repository root:

```sh
python main.py --input_folder /data/predictions --output_folder /data/components_output --ct_root /data/ct --component_plan_root /data/plans --vertebrae_engine shapekit_components --vertebrae_only --cpu_count 1 --log_folder /data/components_logs
```

The existing batch runner discovers case folders. Each requires a matching plan;
missing plans fail visibly, with no silent fallback. Start with one case worker:
3D arrays and distance transforms can require several GB of RAM per case.
Increase workers only after measuring memory use on representative volumes.
This CPU implementation does not require GPU inference.

Each successful case contains 24 vertebra masks, a combined label image, and
`vertebrae_components_report.json` with changes, hashes, timing, and scope.
Output files are reopened and verified before the case directory is finalized.
Failures do not leave a final case directory. Inputs are not modified.

## Tests

```sh
python -m unittest discover -s tests -p test_vertebrae_components.py -v
```

Tests cover cleanup boundary protection, seed-supported relabeling, anisotropic
path costs, unreachable components, contact assignment, plan mismatch, geometry,
label conversion, preservation of additional masks, and refusal to overwrite.
All 10 tests passed, including execution through ShapeKit's command line, a
missing-plan failure, and two concurrent synthetic case workers.
Keep CTs, predictions, local review plans, and experiment outputs outside Git.

For comparison with a standalone warmup prediction folder (internal IDs 1..24):

```sh
python tools/verify_component_predictions.py --actual /data/components_output --expected /data/verified_warmup --report /data/component_regression.json
```

The verifier checks every binary mask, affine/spacing/units and mask transform
metadata, and the combined-map label conversion. It does not calculate DSC
against ground truth.

## Remaining work for unattended deployment

Automatic seed identification and deciding when boundary splits are anatomically
justified remain research problems. A reviewed-plan batch workflow removes local
script dependencies but does not remove review effort. Partial fields of view,
anatomical variants, different orientations, and new scanners require additional
development and evaluation. Do not describe this engine as having proven 93.4%
accuracy across a deployment population.
