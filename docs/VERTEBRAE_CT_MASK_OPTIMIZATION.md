# CT-guided vertebra mask optimization

The optional `shapekit_ct_mask_optimization` engine corrects the 24 vertebra masks (C1-L5) of a case against the case's CT. It fixes labels that are shifted by one level, vertebrae split between two labels, arches and spinous processes labelled as a neighbouring vertebra, and stray pieces far from the spine. Default engines and `config.yaml` defaults remain unchanged.

`utils/vertebrae_ct_mask_optimization_engine.py` is an unchanged copy of the standalone implementation (SHA256 `d2c9bc3a5788b5725ff42efd0674c5f0a8ee2b661157820c63d4fe83036d7803`). `utils/vertebrae_ct_mask_optimization.py` is the ShapeKit adapter.

## Method

All work is done in the CT's own voxel grid, reoriented internally to RAS and converted back before saving. Steps 1-4 correct L5 to T4; C1-T3 keep the model's labels apart from the clean-up in step 5.

1. **Vertebral bodies.** Bodies are found from the shape of the spine mask (a 6 mm erosion separates them at the discs) and numbered as one consecutive sequence. When the scan contains the vertebra just above the range (T3), the bodies are numbered down from it; otherwise the numbering that most body voxels agree with is used.
2. **Arches.** Each arch is attached to its body through the pedicles. Where a body's pedicles carry another level's label, that arch is moved to the body it is attached to. An arch label shared by two vertebrae is divided along the dense cortical bone where the two arches meet.
3. **Thoracic spinous processes.** Each process is isolated from its neighbours using the thin low-density gaps in the CT and takes the label of its root, where it leaves the arch.
4. **Lumbar spinous processes.** Each process takes the level of its position along the curve of the spine.
5. **Clean-up (all 24 levels).** Pieces more than 50 mm from their vertebra are removed as false detections. Small detached pieces join the vertebra they touch or the root of the process they lie in. Enclosed holes of bone density (150 HU or more) are filled; darker joint spaces stay open.

**Safety checks.** Steps 1-4 renumber the spine from the bodies found in step 1, so the bodies are checked before any label changes. Steps 1-4 are skipped, and the model's labels kept, when:

- fewer than 3 bodies are found;
- the numbering does not fit the vertebra above the range;
- a body's new number is more than one level from the model's label;
- the spacing between neighbouring bodies indicates a missed or split body;
- fewer bodies are found than the model has complete levels.

The clean-up still removes far pieces and fills bone-dense holes, but it does not move pieces between labels. A vertebra cut off by the scan edge, with no complete body found, keeps the model's labels. The report gives the reason for every skip.

## Accuracy

The BodyMaps warm-up evaluation on the two AbdomenAtlasDemo cases gave a mean DSC of 94.0% over the 24 vertebrae, from 87.2% (T7) to 99.0% (L5). These are the cases the method was developed on, so this is not a held-out result and does not establish generalization.

## Installation

Use Python 3.9 or later in a dedicated environment:

```sh
python -m pip install -r requirements-vertebrae-ct_mask_optimization.txt
```

Output was verified identical in two environments: Python 3.9.23 with numpy 1.23.5, scipy 1.13.1, nibabel 5.3.3, connected-components-3d 4.1.0, scikit-image 0.24.0; and Python 3.13.1 with numpy 2.5.4, scipy 1.18.1, nibabel 5.4.2, connected-components-3d 4.1.0, scikit-image 0.26.0. No GPU is needed.

## Input

```text
predictions/CASE/segmentations/vertebrae_L5.nii.gz
...
predictions/CASE/segmentations/vertebrae_C1.nii.gz
predictions/CASE/segmentations/<other organs>.nii.gz      (optional, copied unchanged)
ct_root/CASE/ct.nii.gz                                    (or predictions/CASE/ct.nii.gz)
```

Use the model's original predictions. Vertebra masks must be non-overlapping binary 3D volumes on one voxel grid, and the CT, in HU, must be on the same grid. Missing vertebra masks are treated as empty and listed in the report. Overlapping or non-binary masks, masks or a CT on another grid, and sheared affines are errors.

Any axis orientation is accepted (RAS, LAS, LPS and others) because the engine reorients internally; no resampling is performed. Orientation independence was checked by storing a case in RAS, LPS and with a 10° rotated affine: the labels were identical to the normal run.

**Missing CT.** By default a case without a CT fails with a message, and the other cases continue. With `ct_mask_optimization_missing_ct: shapekit_songlin` in `config.yaml`, such a case is processed by the mask-only `shapekit_songlin` engine instead, and its report says so.

## Run

Run from the ShapeKit repository root:

```sh
python main.py \
  --vertebrae_engine shapekit_ct_mask_optimization \
  --vertebrae_only \
  --input_folder /path/to/predictions \
  --ct_root /path/to/ct_root \
  --output_folder /path/to/new_results \
  --cpu_count 4 \
  --log_folder /path/to/logs
```

Add `--case CASE` (repeatable) to select cases, or omit it to process all of them. Output directories must be new and disjoint from the input and CT directories. `--continue_prediction` is not supported.

**Memory and runtime.** One worker peaks at about 3 GB for a 512 × 512 × 1400 CT at 0.7 mm slices and about 2.2 GB for a 512 × 512 × 391 CT at 2.5 mm. One case takes about 3 minutes and 50 seconds respectively on one CPU core. The engine is single-threaded, so `--cpu_count` can be set to roughly the available memory divided by 3 GB, up to the number of cores. Output is deterministic: two runs, including runs in a multiprocessing pool, give byte-identical files.

## Output

Each case contains the 24 binary vertebra masks, the other organ masks copied unchanged, `combined_labels.nii.gz`, and `vertebrae_ct_mask_optimization_report.json`. Internal labels 1 through 24 (L5 to C1) map to the configured ShapeKit labels, normally 26 through 49. The reference grid, qform and sform are preserved.

The report contains:

- the status: `refined`, `steps 1-4 skipped (...)`, `empty`, or `CT missing: shapekit_songlin applied`;
- the engine's log, one line per correction;
- missing input masks;
- changed and total labelled volume;
- the label map and orientation;
- the engine checksum, package versions and runtime.

Cases are written to a staging directory under `.incomplete` and published by renaming only when complete; a failed case causes a nonzero batch exit status.

## Validation

```sh
python -m unittest tests.test_vertebrae_ct_mask_optimization -v
```

Twelve synthetic tests cover:

- empty input and label mapping;
- preservation of other masks;
- output geometry in a non-RAS orientation;
- overlapping, non-binary and mismatched masks;
- a CT on another grid;
- sheared affines;
- both missing-CT policies;
- invalid settings;
- existing or nested outputs.

A thirteenth test compares the adapter's output with a reference voxel for voxel when demo data is configured. The reference can be made with the engine's own command line, which reads each case's `combined_labels.nii.gz` as written by the model:

```sh
python utils/vertebrae_ct_mask_optimization_engine.py --input /path/to/predictions --output /path/to/reference --ct_root /path/to/ct_root
CTMO_DEMO_PRED=/path/to/predictions CTMO_DEMO_CT=/path/to/ct_root CTMO_DEMO_REFERENCE=/path/to/reference \
  python -m unittest tests.test_vertebrae_ct_mask_optimization -v
```

On BDMAP_00000006 and BDMAP_00000031, all 24 masks per case matched the standalone output in data, affine and header, both through the adapter and through `main.py` with two workers. Combined maps matched after label conversion. The full ShapeKit test suite passes with the engine added.

Further checks on modified copies of the two demo cases:

- **Cropped scans:** chest-only, pelvis-only and a middle section. Results agree with the full-case result on 99.1-100% of the bone.
- **Orientations:** RAS, LPS and a 10° rotated affine.
- **Thick slices:** 4.9 mm and 5 mm slices.
- **Edge inputs:** empty masks, a missing CT and a CT on another grid.

## Limitations

- Validated on two cases and modified copies of them only; no generalization claim.
- Thick slices: at about 5 mm, the thin thoracic discs can vanish so that two bodies merge. The case is then caught by the body-count check and keeps the model's labels. In the 5 mm test of BDMAP_00000006 this happened at T8/T9; BDMAP_00000031 at 4.9 mm was corrected normally.
- When the scan does not contain T3, the numbering relies on the model's majority. A short scan in which the model is one level off on most bodies could then be numbered one level off.
- Only L5-T4 are renumbered; C1-T3 receive the clean-up only.
