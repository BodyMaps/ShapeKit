# Landmark-numbering vertebrae engine

`shapekit_landmark` integrates Wajiha Rahim Khan's warm-up solution as an
opt-in vertebrae backend. It does not replace or change the default engine or
any other backend. `utils/vertebrae_landmark_engine.py` is the evaluated
script, refactored only so that its steps can be called as `refine()`; on both
demo cases the refactored engine gives voxel-for-voxel the same labels as the
evaluated submission (see the regression record below).

## Method

The SuPreM vertebra predictions outline bone well; their errors are mostly
about *which* vertebra a piece of bone is given. The engine fixes the naming
with anatomy, and every step that renames bone is kept only if its checks pass.

1. **Clean fragments.** Each label is split into connected pieces. The largest
   piece is the vertebra. Other pieces are removed if they are not bone (mean CT
   under 150 HU) or far from every vertebra; small pieces embedded in another
   vertebra take that vertebra's name.
2. **Count the bodies.** Discs are found as gaps or dips in labelled coverage
   along the spine's centre line, which cuts the column into vertebral bodies.
   When exactly 17 bodies lie between L5 (bottom) and C7 (directly above the
   top body), and the 6th body from the bottom carries ribs while the two below
   it do not, the bodies are numbered by position (landmark-based labelling, as
   in TotalSpineSeg and Meng et al.). Otherwise one name per body is chosen by
   dynamic programming on the model's own labels, with names rising strictly up
   the spine. Kept only if head-to-feet order holds and the vertebra size curve
   does not get rougher.
3. **Arches follow their body.** Each posterior voxel is assigned to the body
   it reaches most cheaply through labelled bone (geodesic cost; darker CT voxels
   such as joint spaces cost more). An arch label mostly attached to the body
   below is renamed piece by piece. Kept only if order holds, the size curve
   does not get rougher and arch/body agreement rises.
4. **Fragments join their neighbour** (after the AtlasDataset post-processing).
   A piece of a label that is detached from that label's main piece
   (6-connectivity) and lies against the main piece of the vertebra directly
   above or below takes that neighbour's name.
5. **Checks.** Labels only 0-24, no background turned into bone, all masks
   consistent with the combined label map.

The engine never adds bone: voxels are only renamed or cleared.

## Installation

Python 3.10 or newer, CPU only:

```console
python -m pip install -r requirements-vertebrae-landmark.txt
python -m pip install pytest
python -m pytest tests -q
```

## Input and command

Use the **original predictions**, not previously refined outputs:

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz ... vertebrae_C1.nii.gz
ct_root/CASE/ct.nii.gz          (or predictions/CASE/ct.nii.gz)
```

Masks must be non-overlapping binary 3D NIfTI files on the same grid as the
calibrated-HU CT. Missing masks are treated as empty and listed in the report.
Any axis-aligned orientation is accepted (the engine reorients internally and
writes back on the input grid); oblique or sheared affines, missing CT, grid
mismatches, non-binary or overlapping masks are errors, not silent fallbacks.

```console
python main.py --vertebrae_engine shapekit_landmark --vertebrae_only \
  --input_folder /data/predictions --ct_root /data/ct_root \
  --output_folder /data/landmark_results --cpu_count 1 --log_folder /data/landmark_logs
```

`--case CASE` (repeatable) restricts the run. The adapter works on a box
around the main piece of every vertebra plus 30 mm, so memory stays modest:
the 512 x 512 x 1394 demo scan peaked at 2.3 GB and took about 6 minutes on
one CPU core. Labelled pieces outside that box (for example pelvic bone
labelled C1) are far strays, which the engine's Step 1 removes anyway; the
adapter clears them and counts them in the report.

## Outputs

- All 24 binary vertebra masks, including empty ones; any other masks in the
  input folder copied byte for byte.
- `combined_labels.nii.gz` with ShapeKit's `class_map` IDs (normally L5 = 26
  through C1 = 49), not the engine's internal 1-24 encoding.
- `vertebrae_landmark_report.json`: what each step did and why (bodies found,
  landmark and rib checks, arches renamed, fragments moved), input missing
  masks, runtime and an engine checksum. It reports `dsc: null`: no ground
  truth is used.
- Each case is built in `<output>/.incomplete/` and published by renaming once
  every file is written.

## Results on the demo cases

| Case | What the engine did |
|:--|:--|
| BDMAP_00000006 | Removed three pelvic pieces labelled C1-C3 and renamed small islands; the numbering checks did not apply, so no body or arch was renamed. |
| BDMAP_00000031 | 17 bodies between L5 and C7, last rib on the 6th body from the bottom: numbered by counting. Arches of T9-L1 renamed to follow their body; 14 detached fragments moved, mostly T4-T7. |

The BodyMaps warm-up evaluation of this output gave an average DSC of 92.2%
over the 24 vertebrae (first, mask-cleaning-only version: 83.3%); the largest
gains were at T9-L1. This is two demo cases only and does not establish
generalisation. The landmark count needs both L5 and C7 in view; on partial
scans it falls back to the model's own labels.

## Regression record

On both demo cases at full size (512 x 512 x 391 and 512 x 512 x 1394), the
adapter reproduced the evaluated submission exactly: all 24 masks per case
matched voxel for voxel, with matching shape and affine. `main.py
--vertebrae_engine shapekit_landmark` gave the same result on cropped copies of
both cases. `python -m pytest tests -q`: 72 passed.
