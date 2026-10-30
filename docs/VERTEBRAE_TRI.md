# Mask-only anatomical vertebra refinement backend

An optional ShapeKit backend implementing global anatomical numbering,
connected-component reassignment, and posterior-aware plane refinement.
It operates **only on segmentation masks** (CT images are not required).

## Input and output contract

Input (per case):

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz
predictions/CASE/segmentations/vertebrae_L4.nii.gz
...
predictions/CASE/segmentations/vertebrae_C1.nii.gz
```

Each input is a binary `0/1` NIfTI mask. All masks must have exactly the same
shape, affine, and voxel spacing, and must not overlap. Missing named masks
are treated as empty and recorded in the case report. At least one vertebral
mask must be present. Output folders must be new, separate from input folders.

The numerical engine uses IDs `1=L5` through `24=C1`. The adapter converts to
and from binary masks; the saved `combined_labels.nii.gz` uses IDs from
ShapeKit's `class_map` (normally `26=L5` through `49=C1`). Non-vertebral
segmentation files are copied unchanged and included in the combined map.
NIfTI affine, qform and sform are preserved from the reference input.

## Installation in an existing ShapeKit clone

Runtime requirements: NumPy, SciPy, and NiBabel (already included in the
ShapeKit requirements). The numerical engine needs no GPU and no CT.
A Python 3.10+ environment with these three packages is sufficient for
this backend's own tests and processing. Running `main.py` additionally needs
all its import-time dependencies; follow the existing ShapeKit environment setup.

## Convert 0..24 standalone combined masks (optional)

If the input is currently `original/CASE/combined_labels.nii.gz` with IDs
`0..24`, convert it into the per-vertebra mask format before calling ShapeKit:

```bash
python tools/prepare_vertebrae_inputs.py \
  --input_root /data/original \
  --output_root /data/shapekit_predictions \
  --case BDMAP_00000006 \
  --case BDMAP_00000031
```

Do not feed a `26..49` ShapeKit combined map or an already refined output to
this converter. Original masks are not modified. Output directories must not
already contain `segmentations`.

## Run

One case first:

```bash
python main.py \
  --vertebrae_engine shapekit_tri \
  --vertebrae_only \
  --input_folder /data/shapekit_predictions \
  --output_folder /data/shapekit_results \
  --case BDMAP_00000006 \
  --cpu_count 1 \
  --log_folder /data/shapekit_logs
```

For batch processing, omit `--case`; start with one worker and increase
`--cpu_count` only after measuring peak RAM and run time on representative
full-resolution cases. Each worker handles an independent case. The output
case directory is published atomically, and existing output cases are never
overwritten. Avoid `--continue_prediction` with this backend.

## Validation

Run tests from the ShapeKit root:

```bash
python -m unittest discover -s tests -p 'test_vertebrae_anatomical.py' -v
```

Other essential checks:

- All 24 output binary masks exist (including empty masks).
- No input files have changed.
- The reported label mapping matches `config.yaml`.
- Output shape, affine, qform, and sform match the input reference.
- A missing class or a case with one vertebra does not crash.
- Overlapping, nonbinary, or misregistered input masks are rejected.
- Run a small multi-case batch and verify distinct per-case reports.
