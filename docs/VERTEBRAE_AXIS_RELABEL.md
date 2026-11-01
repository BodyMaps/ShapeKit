# Axis-based vertebra relabelling (`shapekit_axis_relabel`)

An optional vertebrae engine that fixes **wrong vertebra numbers** in a prediction. A common
problem is that part of a vertebral body carries the label of the vertebra above it, or a
spinous process or lamina carries the label of its neighbour. This engine re-derives the
vertebrae from the CT and the shape of the mask, using the predicted labels only as a hint.
It is classic image processing only (no learned model, no GPU), and the default engines are
unchanged.

## How it works

1. **Cleanup** – small specks far away from the spine are removed.
2. **Spine axis** – in every axial slice the vertebral body is the thickest solid part of the
   mask. Its centre is tracked from slice to slice with dynamic programming and then smoothed.
   Disc gaps are bridged and small holes filled first, but never the spinal canal.
3. **Levels** – along the axis, disc evidence is collected: dark bands in the CT, gaps in the
   mask, and places where the predicted label changes. A dynamic program splits the column into
   levels. The expected vertebra height is taken from the spacing of the predicted labels, so
   lumbar and cervical levels both fit. Levels get consecutive numbers, shifted to agree best
   with the prediction.
4. **Bodies and posterior elements** – each body gets the label of its level. Posterior bone
   (pedicles, laminae, spinous and transverse processes) is grown from each vertebra's own
   pedicle roots through the bone with two independent watershed floods:
   - Where they agree, that label is used.
   - Where they disagree, the prediction decides.
   - Dense cortical bone cores are then made single-label.
5. **Safeguards**:
   - C1 (which has no body) and slices beyond the ends of the detected axis keep their
     predicted labels.
   - Pieces cut off from their vertebra join the vertebra they touch.
   - Distant small fragments are removed.

If no spine can be found (for example, almost no vertebra in the scan), the case is written
with its input labels unchanged and a warning is logged.

## Installation

```sh
python -m pip install -r requirements-vertebrae-axis_relabel.txt
```

The engine itself needs numpy, scipy, nibabel and scikit-image. The other packages in the
file are needed because `main.py` imports the organ modules.

## Input

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz ... vertebrae_C1.nii.gz   (+ any other masks)
ct_root/CASE/ct.nii.gz          (or predictions/CASE/ct.nii.gz)
```

- Vertebra masks must be binary and must not overlap. A missing mask is treated as empty and
  logged.
- The CT must be in HU and on the same grid as the masks.
- Any axis-aligned orientation works. The engine reorients a crop around the spine to RAS
  internally and maps the result back, so the input is never resampled.

## Run

```sh
python main.py \
  --vertebrae_engine shapekit_axis_relabel \
  --vertebrae_only \
  --input_folder /path/to/predictions \
  --ct_root /path/to/ct_root \
  --output_folder /path/to/new_results \
  --cpu_count 4 \
  --log_folder /path/to/logs
```

Or set `vertebrae_engine: shapekit_axis_relabel` in `config.yaml` and pass `--vertebrae_only`.
Use `--case CASE` (repeatable) or `--csv` to select cases.

## Running at scale

- **Memory:** about 2 GB per worker for a 0.7 mm whole-spine case (measured: 2.0 GB peak).
  Set `--cpu_count` to roughly *available RAM / 2.5 GB*. One case uses one core.
- **Time:** about 2 minutes for a 0.7 mm whole-spine case (147 × 236 × 930 voxels after
  cropping) and about 25 seconds for a thoracolumbar segment, on one core.
- **Resume:** each case is written to `<output>/.incomplete/` and only moved into place once
  every file is saved, so a case folder in the output is always complete. After an
  interruption, rerun the same command with `--continue_prediction` and only the missing
  cases are processed.
- **Failures:** a case with a missing CT, a grid mismatch, or overlapping or non-binary masks
  fails on its own. The other cases continue, and the failed ones are listed at the end with a
  non-zero exit code. The output folder must be separate from the input and CT folders.

## Output

```text
new_results/CASE/segmentations/vertebrae_*.nii.gz   relabelled masks (all 24)
new_results/CASE/segmentations/<other>.nii.gz       other masks, copied unchanged
new_results/CASE/combined_labels.nii.gz             combined map with the class_map IDs (26 = L5 ... 49 = C1)
```

The affine, qform and sform of the input masks are kept.

## Validation

```sh
python -m unittest discover -s tests -p test_vertebrae_axis_relabel.py -v
```

There are 11 tests:

- A synthetic spine with a mislabelled body is corrected exactly.
- An LAS copy of a case gives the same result as the RAS original.
- Other masks are copied byte for byte, and the combined map uses the class_map IDs.
- A case with no detectable spine keeps its labels.
- Invalid input is rejected: overlapping masks, non-binary masks, a CT grid mismatch, a
  missing CT, no vertebra masks, or an existing output folder.

End to end through `main.py` on AbdomenAtlasDemo BDMAP_00000031 (SuPreM prediction, 0.7 mm,
24 levels), the output contains all 24 vertebrae (L5 to C1). An LAS-flipped copy of the case gave
the same labels voxel for voxel, and resuming an interrupted batch with `--continue_prediction`
reproduced byte-identical output.

No accuracy against ground truth (DSC) is reported, and the end-to-end checks used one patient.
