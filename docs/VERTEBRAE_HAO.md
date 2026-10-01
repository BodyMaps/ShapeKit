# Native-grid two-stage vertebrae backend

`shapekit_hao` integrates Hao Yu's evaluated standalone warm-up algorithm.
It does **not** replace the default `shapekit_songlin` backend or change the
evaluated algorithm. `utils/vertebrae_hao_engine.py` is an unchanged source copy
(SHA256 `9213f51295c91b3fa4b8315af3640b709bfa89430e59047bbff0b98d924a8cc9`).

## Method and scope

1. Reproduce the v2 baseline: per-label component filtering and hole filling
   with the original 16-voxel, 1%, 12-mm parameters.
2. Correct L2 through T5 using stable foreground cores, ordered anatomical
   anchors, and two-scale foreground geodesic consensus. Ambiguous evidence
   retains the baseline. L5-L3 and T4-C1 remain exactly equal to the baseline.

The external warm-up evaluation reported a mean DSC of 92.0% on two demo
cases for the standalone submission. This is **not** a validation on a new
dataset and does not establish generalization. Local regression checks compare
outputs with that submission, not with ground truth; they cannot compute DSC.
Related-work inspiration is described in the engine header, not claimed as a
full reproduction of a trained published system.

## Installation

In a separate Python 3.10 environment:

```console
python -m pip install -r requirements-vertebrae-hao.txt
python -m pip install pytest
python -m pytest tests -q
```

No GPU, Torch or MONAI is needed by this backend. Run commands from the
repository root because the existing entrypoint loads `config.yaml` there.

## Input and command

Use **original predictions**, not previously refined outputs:

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz
predictions/CASE/segmentations/vertebrae_L4.nii.gz
... (through C1)
ct_root/CASE/ct.nii.gz
```

Masks must be nonoverlapping binary 3D NIfTI files on the same grid as the
calibrated-HU CT. Missing named masks are treated as empty and listed in the
report; missing anchors cause the original algorithm to abstain. CT is required.
Wrong grids, nonbinary masks, and overlaps are errors, not silent fallbacks.
No mask/CT resampling, axis reorientation, or HU clipping is performed.

```console
python main.py --vertebrae_engine shapekit_hao --vertebrae_only --input_folder /data/predictions --ct_root /data/ct_root --output_folder /data/new_hao_results --case CASE --cpu_count 1 --log_folder /data/hao_logs
```

CLI overrides leave `config.yaml` defaults unchanged. Repeat `--case` to select
more cases or omit it to process all case directories. Start with one worker;
each worker loads full-resolution masks and float32 CT plus cropped working
arrays. Peak RAM and throughput must be measured before raising concurrency.
The adapter processes one case at a time per worker and never changes inputs.

## Outputs and safeguards

- All 24 binary vertebra masks are written, including empty masks.
- Other input segmentation files are copied byte-for-byte; organ cleanup is
  deliberately not stacked with this algorithm.
- `combined_labels.nii.gz` uses ShapeKit's `class_map` (normally L5=26 through
  C1=49), **not** the standalone 1-24 encoding. Compare masks by name, or use the
  `label_map` in `vertebrae_hao_report.json` when comparing combined maps.
- qform, sform and image geometry are preserved from the reference mask.
- The case report records stage-two status/reason, protected-mask checks,
  input missing-mask list, source hashes, label mapping, and runtime.
- Output case directories must be new and disjoint from inputs. Successful
  cases are published by renaming a staging directory after all writes finish.
  Failed writes remain under `.incomplete` for inspection. The batch exits
  nonzero if any case fails. Do not use the legacy `--continue_prediction` check.

## Required regression before submitting a PR

Run both original demo cases with one worker. For each case, compare all 24
output masks voxel-for-voxel with the evaluated standalone output, including
shape, affine, spacing and qform/sform. Confirm combined-map label conversion
and protected-mask assertions. Unit tests alone do not establish real-case
equivalence or accuracy. Do not commit CT volumes, predictions or private CVs.

```console
python tools/verify_vertebrae_hao.py --actual /data/new_hao_results --reference /data/passed_standalone_results --case CASE --report /data/new_equivalence_report.json
```

Repeat `--case` for multiple cases. The checker requires all 24 binary masks
and accepts `--actual-subfolder NAME` if `subfolder_name` was customized.
It compares geometry including qform/sform, converts combined IDs using the
case report, and exits nonzero on any discrepancy. It reports `dsc: null`
deliberately.

### Local regression record (2026-09-23)

- 50 synthetic/contract tests passed in the isolated Python 3.10 environment.
- Both demo cases ran through `main.py` with this backend and one worker.
- `BDMAP_00000006`: stage two abstained on a consistent body-core chain;
  all 24 masks matched the evaluated standalone output, with zero combined-map
  voxel differences after label conversion and matching geometry.
- `BDMAP_00000031`: stage two applied the stable-core geodesic correction;
  all 24 masks matched the evaluated standalone output, with zero combined-map
  voxel differences after label conversion and matching geometry.

These checks establish output equivalence on these two cases only. They do not
recompute the externally reported DSC, measure peak RAM/concurrent throughput,
or validate clinical accuracy on unseen cases.
