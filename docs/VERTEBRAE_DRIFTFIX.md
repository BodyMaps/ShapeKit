# Drift-fix vertebrae post-processing (`shapekit_driftfix`)

An optional engine that cleans up 24-level vertebra masks (L5 ... C1) and flags what still looks wrong.
It is not the default and changes nothing else in ShapeKit. The algorithm is a vendored script,
`utils/vertebrae_driftfix_engine.py`, kept byte-for-byte as it was written. `utils/vertebrae_driftfix.py`
is a thin adapter that connects it to ShapeKit.

## What it does (five stages)

1. **Load and crop.** Read the labels (and the CT if there is one), cut away the empty space around the spine.
2. **Clean-up.** Delete tiny specks, decide what happens to detached fragments and far-away debris
   (for example hip bone labelled C1), and drop label blobs that are far off the spine.
3. **Label drift.** If label boundaries cut through vertebral bodies instead of the discs, split the bone
   at the discs and give each vertebra one label. Needs the CT.
4. **Sacrum and lost level.** Find the sacrum below the lowest vertebra. If it confirms L5 and the chain looks
   one level too low above a duplicated label, restore the lost level.
5. **Order check.** Look at the vertebra centres along the head-foot axis and flag inverted or oversized steps.
   This stage never changes a voxel.

Every change and every flag is written to the JSON report with its reason.

## Install

No new packages: it needs numpy, scipy, nibabel (and PyYAML for the config key), all already in
`requirements-vertebrae-hao.txt`. No separate requirements file is added.

## Input

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz
...
predictions/CASE/segmentations/vertebrae_C1.nii.gz
ct_root/CASE/ct.nii.gz            (optional)
```

Masks must be binary, 3D, on one grid, and must not overlap. A missing level file is treated as empty and
recorded. If a CT is given and its grid differs from the masks, the run stops with an error. Any orientation is
accepted (the engine works in millimetres through the affine). No resampling is done.

If the CT is **missing**, the engine runs its mask-only path: the HU checks and the sacrum anchor are skipped,
and the report says `"ct_used": false` and `inputs.ct_status` starts with `missing`. This differs
from `shapekit_hao` and `shapekit_ct_refinement`, which raise an error. Check `ct_used` in the report.

## Run

Run from the ShapeKit root. Use a new output folder.

```sh
python main.py \
  --vertebrae_engine shapekit_driftfix \
  --vertebrae_only \
  --input_folder /path/to/predictions \
  --ct_root /path/to/ct_root \
  --output_folder /path/to/new_results \
  --case CASE \
  --cpu_count 1 \
  --log_folder /path/to/logs
```

`--vertebrae_only` is required, as for the other case engines.

## Options

One option, in `config.yaml`:

```yaml
vertebrae_driftfix_relabel: false   # default
```

- `false` (default, the safe choice): stages 3 and 4, the two that can change which level a voxel belongs to,
  are skipped. This is the script's `--no_relabel`. Clean-up and the order check still run.
- `true`: all five stages run.

Why false is the default: on spines with an extra vertebra (L6 or T13) the lost-level repair can move many
vertebrae to the wrong level (see Known limits). From Python you can also pass `relabel=True` or `False`
to `process_case`.

## Output

- All 24 binary masks (empty ones too) with the header, qform and sform of the input.
- `combined_labels.nii.gz` with ShapeKit ids (26 = L5 ... 49 = C1). The engine's own numbering is
  1 = L5 ... 24 = C1 (`id = label + 25`).
- `vertebrae_driftfix_report.json`: the engine log (`changes`, `order_flags`, `qa_before`, `qa_after`,
  `lost_level`, `label_drift`, `sacrum`, `timing_s`), plus `review_flags` (the entries that need a person to
  look), `ct_used`, `relabel`, `label_map`, `missing_input_masks` and `engine_sha256_lf`.
- Other files in the input segmentation folder are not copied.
- The case is written into `.incomplete` and renamed at the end. An existing output folder is refused.

## Runtime and memory

Measured on a Windows laptop with 8 GB RAM, one worker, on the cropped demo case BDMAP_00000006
(278 x 322 x 290 voxels): about 25 s inside the adapter, about 28 s through `main.py`.
The author's notes record about 5 s for the full case 06 and about 30 s for the large case 31 on a larger
machine, and a median of 1.6 s per case (maximum 19 s) on a cloud CPU. Peak memory was not measured.
The adapter holds the 24 masks as one uint8 volume and the engine works on a crop around the spine, so memory is
roughly a few copies of the full uint8 volume plus the CT as int16. Large 0.7 mm whole-spine scans were not
run on this laptop.

## Known limits

Numbers below come from the author's held-out test of 100 CTs with ground truth (frozen script, scored once).
The write-up (`REPORT.md`, `eval/heldout2_results.md`) is in the author's own project, not in this repository.

- On 90 spines with typical anatomy: Dice 0.880 to 0.888 (gain +0.009, 95% CI +0.003 to +0.016).
  Order violations per case fell from 12.4 to 3.9. The Dice gain is small; most of the gain is anatomical validity.
- On 10 spines with an L6 or T13: Dice fell from 0.577 to 0.506. Almost all of that is one scan where the
  lost-level repair took the lowest vertebra to be L5 and moved 14 vertebrae one level the wrong way.
  So **the lost-level repair can wreck a spine with an L6**. The label-drift repair also hurt one L6 scan.
- The model has only 24 classes, so L6 and T13 cannot be written.
- A simple rule that keeps only the largest piece of each label scored about as well on Dice (0.876 on the
  90 typical spines, against 0.888 here); this engine leaves fewer order violations but more small pieces.
- On the two demo cases there is no ground truth. Checks there are about consistency (volumes, pieces, order),
  not accuracy.
- On one scan a 21-voxel edge piece of a vertebra was removed as a speck, which took that level's Dice to 0.
- Not tested: large 0.7 mm scans on this laptop; Python versions other than 3.13.

## Validation done for this integration

- `tests/test_vertebrae_driftfix.py`: label round trip, synthetic phantom, missing CT, existing output,
  grid mismatch, overlap, non-binary, and voxel-for-voxel equality with the engine run directly, on the
  phantom and on a cropped demo case (that test is skipped if `data/shapekit_demo` is not present).
- BDMAP_00000006 (cropped) through `main.py`: 794 voxels changed, 1 review flag. BDMAP_00000031 was not run
  on this machine.

These checks show the integration reproduces the standalone script. They are not an accuracy test.

## Benchmark against other engines

> **PLACEHOLDER. To be filled in later. No numbers have been measured for this table yet.**

| engine | identification rate | mean Dice | whole-spine Dice | HD95 (mm) | time | peak RAM |
|---|---|---|---|---|---|---|
| shapekit_driftfix | TBD | TBD | TBD | TBD | TBD | TBD |
| other engines | TBD | TBD | TBD | TBD | TBD | TBD |
