# Multi-model vertebrae consensus backend

`shapekit_consensus` integrates Hongkang Chu's submitted warmup CPU
postprocessing into ShapeKit's case selection, worker pool, logging, label
mapping, and output layout. It is optional; the existing default is unchanged.

The method uses predictions from **TotalSegmentator v2, SuPreM, and
TotalSegmentator v3**. It does not train or run these networks. Predictions are
not reference annotations. No ground-truth mask, case-specific rule, or score
is an algorithm input.

## Method

Raw mode runs the following chain with the submitted numerical parameters:

1. Original upstream Hao component cleanup and anatomical correction on SuPreM.
2. The native-grid caudal and cervicothoracic anchor refinements.
3. Original Hao refinement on TotalSegmentator v2, followed by the dual ordered
   body-chain identity check using the corrected SuPreM anchors.
4. Global identity proposals on **raw v3**, then the coherent-offset gate.
   V3 is not passed through Hao in this path.
5. Two-source agreement for boundary additions/deletions within 2 mm of the
   incumbent foreground. Additions must agree with the nearest incumbent
   identity; retained foreground IDs and complete C1/C2 masks are protected.
   The submitted variant has `ct_guard=False`. CT is still required by the
   upstream stages and by the physical-grid contract.

`utils/_vertebrae_consensus/` contains the numerical functions extracted from
the submission, with cohort-specific CLIs removed and imports made package
relative. Function bodies are unchanged. `SOURCE_PROVENANCE.json` records
original source hashes and AST fingerprints. Hao is reused from the existing
`utils/vertebrae_hao_engine.py`, not copied or modified.

## Installation

Use the existing small CPU environment; PyTorch and model weights are not needed:

```console
python -m pip install -r requirements-vertebrae-hao.txt
python -m pip install pytest
python -m pytest tests -q
```

Run ShapeKit commands from the repository root: its existing entrypoint reads
`config.yaml` from the working directory.

## Raw prediction input

All three roots have the same case IDs and mask naming:

```text
v2/CASE/segmentations/vertebrae_L5.nii.gz
v2/CASE/segmentations/vertebrae_L4.nii.gz
... through vertebrae_C1.nii.gz
suprem/CASE/segmentations/vertebrae_L5.nii.gz
v3/CASE/segmentations/vertebrae_L5.nii.gz
ct/CASE/ct.nii.gz
```

Use **original raw predictions** for each model in this mode. The adapter reads
named binary masks and does not infer label conventions from combined maps.
Missing mask files are reported and treated as empty; insufficient anatomical
cores cause the numerical stages to abstain. Additional vertebra levels such
as L6 or T13 are rejected, not dropped. Each mask must be binary, nonoverlapping,
and on the same grid as the CT. No resampling or orientation conversion occurs.

Raw mode supports the original **axis-aligned LAS/RAS** grids. Other scanner
orientations fail explicitly; they have not been adapted or validated.
All grids must be orthogonal, finite, and have matching shape, affine, and
spacing. The CT must declare millimeter units and contain calibrated HU after
NIfTI scaling. Legacy prediction headers with unknown units are accepted only
on the matching explicitly-mm CT grid; their original flags are retained and
reported. Meter/micron prediction coordinates are rejected.

```console
python main.py --vertebrae_engine shapekit_consensus --vertebrae_only \
  --input_folder /data/v2 --suprem_root /data/suprem \
  --totalseg_v3_root /data/v3 --ct_root /data/ct \
  --output_folder /data/consensus --cpu_count 1 \
  --log_folder /data/consensus_logs
```

Repeat `--case CASE` to select cases, or use the existing `--csv` filter.
Without either filter, all primary-root case directories are selected.
Use `--consensus_save_stages` to save intermediate combined volumes for
regression checks. They use **internal IDs 1=L5 through 24=C1**, unlike the
ShapeKit combined output.

## Prepared boundary-stage replay

If the upstream chain has already run, set `--consensus_input_stage prepared`.
The roles change explicitly:

| Argument | Prepared input |
| --- | --- |
| `--input_folder` | V2 + Hao + dual-chain incumbent |
| `--suprem_root` | Corrected SuPreM anchors |
| `--totalseg_v3_root` | Global-gated v3 prediction |

Each root still supplies named binary masks in `CASE/segmentations/`.
Prepared mode runs **only** the final boundary agreement stage. Reports state
the input stage; a prepared run must not be described as a full replay.

## Outputs and batch behavior

- All 24 binary masks are written, including empty masks. Unrelated organ
  masks from the primary input are copied unchanged.
- `combined_labels.nii.gz` follows `class_map`, normally 26=L5 through 49=C1.
  Existing ascending-ID overlap priority is retained for combined multi-organ
  output. Per-vertebra masks remain the authoritative binary exports.
- qform, sform, codes, affine, spacing, orientation, and header units are
  retained from the primary reference mask.
- `vertebrae_consensus_report.json` records completion, input hashes, source
  hashes, missing masks, label conversion, unit provenance, stage decisions,
  and runtime. `dsc` remains null: this adapter does not read ground truth.
- New output directories must be disjoint from **all** input roots, including
  both auxiliary models and CT. Existing case outputs are never overwritten.
- A case is written under `.incomplete` and renamed only after every output
  and its completion report has been saved. Failed writes remain available
  for diagnosis; a failed case does not appear as a completed case directory.
- The existing worker pool continues processing other selected cases and
  exits nonzero if any case fails. Do not use `--continue_prediction`; its
  legacy nonempty-folder check does not establish completion of this backend.

This backend defaults to **one worker**; use `--cpu_count` to opt into concurrency.
Full-resolution 3D graph stages can require substantial
RAM; concurrency should follow measured peak RSS on the target CT resolution.
The adapter writes one binary mask at a time rather than constructing 24
full-volume mask arrays simultaneously. This is a memory-management change in
the writer, not a change to the numerical method.

## Regression and evidence limits

Run the full chain on both supplied cases and compare **all 48 binary masks**
and metadata with the submitted standalone results. Convert ShapeKit combined
IDs using the report's `label_map` before comparing internal combined maps.
Also compare the saved `suprem_anchors`, `incumbent`, and `aligned_v3` stages
against the submitted frozen upstream maps. Unit tests alone cannot establish
real-case equivalence.

```console
python tools/verify_vertebrae_consensus.py \
  --actual /data/consensus --reference /data/submitted_predictions \
  --case CASE1 --case CASE2 \
  --prepared-inputs /data/submitted_boundary_inputs \
  --report /data/consensus_equivalence.json
```

The optional `--prepared-inputs` folder has `SuPreM_anchors/CASE/`,
`incumbent/CASE/`, and `aligned_v3/CASE/` directories, each with a standalone
`combined_labels.nii.gz` in 0..24 encoding. It requires `--consensus_save_stages`
on the full run. The checker compares all masks, combined-map conversion,
units, qform/sform, affine, and spacing, and returns a nonzero exit on a mismatch.

The previous public-reference two-case development mean was 90.106%; the
instructor's separate evaluation reported 91.5%. These are different protocols.
On three additional public cases, the candidate scored 88.640% versus 89.313%
for the stronger v2 + Hao baseline. The development improvement did **not**
generalize to that cohort. Packaging and output equivalence do not establish
clinical accuracy or large-cohort generalization.

Credit: SuPreM authors; TotalSegmentator / Jakob Wasserthal and collaborators;
Hao Yu / BodyMaps ShapeKit for the unchanged Hao engine; Hongkang Chu for the
submitted anchor, identity-consistency, and boundary-consensus refinements.
The causal-root idea was informed by the public ShapeKit-Pro discussion; no
ShapeKit-Pro source was copied into these numerical modules.

Keep CT volumes, predictions, reference masks, checkpoints, and private reports
outside the repository. Local validation artifacts are not PR source files.

## Local integration validation — 15 October 2026

- All **84 tests** passed, including the existing suite, malformed-input checks,
  partial-write handling, failed-case isolation, and verifier corruption tests.
- Both supplied warmup cases ran through the complete **raw** CPU chain using
  ShapeKit's entrypoint and two concurrent case workers.
- All **48 exported binary masks** matched the submitted standalone predictions
  voxel-for-voxel. Both combined maps matched after class-ID conversion, and all
  **six frozen intermediate maps** matched. Shape, affine, spacing, units, qform,
  and sform checks passed.
- Numerical function fingerprints matched the submitted source, and the run
  reports' source hashes matched the current integration files.
- On this host, case 06 took 4.68 minutes and case 31 took 31.87 minutes. The
  concurrent batch took 31 minutes 55 seconds. GNU time reported a maximum
  process RSS of about **14.5 GiB**; this is not aggregate concurrent memory or
  a large-cohort throughput benchmark. BLAS/OMP threads were set to one per
  worker. The backend still defaults to one case worker.

These are integration and output-equivalence checks. They do not rerun neural
inference, recompute the instructor's DSC, or establish accuracy on new cases.
Local CTs, predictions, logs, and regression reports are kept outside the Git
working tree.
