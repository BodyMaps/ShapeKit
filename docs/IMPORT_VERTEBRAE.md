# Import refined vertebra masks

Use this command when you already have the refined masks. To run the warm-up
from raw SuPreM predictions, use the [ordered-core engine](ORDERED_CORE.md).

Each input case needs a vertebra-only `combined_labels.nii.gz` and matching
binary masks in `segmentations/`. Use 1–24 for L5 through C1, or pass
`--label-offset 25` if the combined map already uses IDs 26–49.

```bash
python -m pip install numpy nibabel PyYAML

python tools/import_vertebrae.py \
  --input-root /runs/refined --output-root /runs/shapekit \
  --ct-root /data/ct
```

The importer checks that the masks and combined map agree on the same grid.
If provided, the CT must match that grid too. Existing binary files are copied
as-is, missing empty levels get zero masks, and the combined map uses the
IDs in ShapeKit's class map. Other known organ masks can be copied if they
have the same grid and do not overlap another class.

Use `--case` to select individual cases. After an interruption, rerun with
`--resume` to check and skip completed cases. Each case has an
`import_report.json` alongside its outputs. A failed case is reported while
the rest of the batch continues.
