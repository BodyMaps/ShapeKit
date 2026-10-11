# ShapeKit-Asfand vertebrae engine

`vertebrae_engine: shapekit_asfand` (or `--vertebrae_engine shapekit_asfand`)

A mask-only vertebrae engine that fixes **label drift**: runs of vertebral bodies where the model's labels are shifted by one level, so that single bodies carry two labels. No CT, no new dependencies, CPU only.

## The error it targets

In AbdomenAtlas `BDMAP_00000031` (SuPreM vertebrae model) the model put 12 thoracic labels on 13 vertebral bodies. Between T8 and L2, several bodies carried two labels and every level was shifted by one. For example, the L1 body was labelled T12 while its posterior elements were labelled L1. Mask-only cleanup that keeps the largest component per label cannot fix this, because each wrong label is a large, plausible component.

## Method

| Step | What it does | Why |
|---|---|---|
| 1. Fragments | Delete whole-spine components < `min_frag_ml` (2 mL) | Removes stray voxels, e.g. C1 labels in the pelvis |
| 2. Body centre line | Per axial slice, centroid of the deepest part of the mask (distance transform); median filtered and smoothed | The body is the thickest bone. The centroid of the whole slice lies in the spinal canal, because body, pedicles and lamina form a ring |
| 3. Discs | Unlabelled fraction in a 6 mm disk around the centre line; peaks are discs | The model leaves intervertebral discs unlabelled |
| 4. Counting | From the lowest body upward, each body gets the level it must have; a body is trusted if the model's majority label matches with ≥ 90 % share. Stops after 3 trusted bodies following a mismatch, or when a segment is implausibly long | Only the drifted range is changed; the rest of the spine keeps the model's labels |
| 5. Relabel | Seed each body in the drifted range with its counted level and grow through the mask only (geodesic) | Posterior elements reach their own body through the pedicles; straight-line distance would hand sloping spinous processes to the level below |
| 6. Detached pieces | Each label keeps its largest piece; other pieces (≤ 10 mL) join the vertebra they touch most | One connected component per vertebra |

All parameters live in `Config` in `utils/vertebrae_asfand_engine.py`.

## Results

AbdomenAtlas demo, SuPreM `swin_unetr_totalsegmentator_vertebrae.pth` predictions:

| Case | Model pieces | Refined pieces | Bodies relabelled |
|---|---|---|---|
| BDMAP_00000006 | 45 | 24 | 0 |
| BDMAP_00000031 | 122 | 24 | 5 (T8 to L2) |

Average DSC against the reference labels (evaluated by the BodyMaps team): **92.9 %** over the 24 vertebrae of both cases. The weakest band is T7 to L1 (86 to 89 %), which is the range this engine relabels: the level counting is right, but boundaries drawn by the regrowth are less precise than the model's original boundaries.

Runtime on 2 CPU cores: about 15 s for a 2.5 mm case and about 90 s for a 0.7 mm whole-spine case. Memory is dominated by ShapeKit holding the 24 masks (twice, input and output): budget about 10 GB per worker for a 0.7 mm whole-spine case and set `--cpu_count` accordingly.

The ShapeKit integration reproduces the standalone submission exactly: on both demo cases the refined labels are voxel-identical to the evaluated output.

## Usage

```bash
python main.py --input_folder /path/to/predictions --output_folder /path/to/out \
  --vertebrae_engine shapekit_asfand --vertebrae_only
```

Vertebrae-only prediction folders have no `liver.nii.gz`; `main.py` then uses any mask in the case as the grid reference.

## Limitations

- **Anchor at the bottom.** Counting starts from the model's label of the lowest body. If that label is itself wrong, or the anatomy is transitional (L6, sacralized L5), the counted levels are shifted too.
- **Discs must be unlabelled.** If a model labels discs as bone, step 3 finds fewer discs; climbing stops at the first implausibly long segment and nothing above it is changed.
- **Boundary placement.** Geodesic growth assigns contested voxels to the higher level when two fronts arrive in the same step (`grey_dilation` takes the maximum). Posterior elements stacked directly over the level below can be pulled up one level.
- **Seed radius.** Seeds are a fixed 12 mm disk around the body centre; in small cervical bodies the disk can reach posterior elements.

## Tests

```bash
python -m pytest -q tests/test_vertebrae_asfand.py
```

Synthetic spines with unlabelled discs and connected posterior elements: clean spine unchanged, drifted bodies relabelled, fragment removal, arbitrary superior-inferior axis, fail-safe fallback.

## Credits

Builds on ShapeKit (Liu et al.) and the mask-based vertebra post-processing of Meng et al. (2022). Engine and adapter by Asfand Yar.
