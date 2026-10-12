# CT-guided vertebra refinement

The optional `shapekit_ct_refinement` engine refines vertebra predictions using multiscale body cores, endpoint anchors, CT evidence, conservative component reassignment, and local boundary recovery. Existing default engines remain unchanged.

## Installation

Use Python 3.10 or later in a dedicated environment:

```sh
python -m pip install -r requirements-vertebrae-ct_refinement.txt
```

## Input

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz
predictions/CASE/segmentations/vertebrae_L4.nii.gz
...
predictions/CASE/segmentations/vertebrae_C1.nii.gz
ct_root/CASE/ct.nii.gz
```

Vertebra masks must be nonoverlapping binary 3D volumes. CT must contain calibrated HU and match the mask grid. Missing individual masks are treated as empty and recorded. Missing CT, inconsistent grids, overlapping masks, and nonbinary masks are errors.

RAS and LAS orientations are supported. The method assumes the anterior direction along positive Y and the superior direction along positive Z. It uses symmetric X distances. Sheared affines are rejected. No resampling or orientation conversion is performed. Oblique scans require further anatomical validation.

## Run

Run from the ShapeKit repository root:

```sh
python main.py \
  --vertebrae_engine shapekit_ct_refinement \
  --vertebrae_only \
  --input_folder /path/to/predictions \
  --ct_root /path/to/ct_root \
  --output_folder /path/to/new_results \
  --case CASE \
  --cpu_count 1 \
  --log_folder /path/to/logs
```

Repeat `--case` for additional cases or omit it to select all cases. Output directories must be new and disjoint from input directories. `--continue_prediction` is not supported. Start with one worker and measure memory use before increasing concurrency. KD-tree queries use one worker to avoid nested parallelism.

## Output

Each case contains 24 binary vertebra masks, `combined_labels.nii.gz`, and `vertebrae_ct_refinement_report.json`. Internal labels 1 through 24 map to the configured ShapeKit labels, normally 26 through 49. Other organ masks remain unchanged. Combined-map overlap precedence follows ascending label IDs.

The reference grid, qform, and sform are preserved. Reports include correction status, voxel changes, missing masks, label mapping, runtime, and an engine checksum. Completed cases are published after successful writes. Failed cases retain temporary output under `.incomplete` and cause a nonzero batch exit status.

## Validation

```sh
python -m unittest discover -s tests -p test_vertebrae_ct_refinement.py -v
```

Eight tests cover empty input, output mapping, preservation of other masks, overlapping masks, nonbinary input, geometry mismatch, unsupported orientation, missing CT, and existing outputs.

On BDMAP_00000006 and BDMAP_00000031, all 24 output masks per case matched the original standalone implementation voxel for voxel. Combined maps matched after label conversion, and image geometry matched. ShapeKit elapsed times were approximately 56 seconds and 221 seconds with one case worker on the local test machine.

These checks establish integration equivalence on two cases. They do not recompute DSC, establish generalization, or measure peak memory and concurrent throughput. No accuracy improvement is inferred from the integration checks.
