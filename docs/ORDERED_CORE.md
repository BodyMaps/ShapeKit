# Run the vertebra warm-up in ShapeKit

Use this engine to run the ordered-core warm-up on raw SuPreM vertebra masks.
You'll need a copy of the warm-up's `postprocessing_vertebrae.py`, supplied
separately. Pass the path to a copy you trust, since ShapeKit runs that script.

Use Python 3.12. If you're setting up an environment just for this engine:

```bash
python -m pip install numpy scipy scikit-image nibabel PyYAML connected-components-3d matplotlib tqdm
```

## Run one case

From the ShapeKit checkout, run the command below for a case named `case_a`.
Each case needs a vertebra-only `combined_labels.nii.gz` and the corresponding
binary masks in `segmentations/`. The raw combined map uses 0 for background
and 1–24 for L5 through C1. Keep `if_save_combined_label: true` in `config.yaml`.

```bash
python main.py --input_folder /data/raw --output_folder /runs/ordered_core \
  --vertebrae_only --vertebrae_engine shapekit_ordered_core \
  --ordered_core_source /path/to/postprocessing_vertebrae.py \
  --case case_a --log_folder /runs/ordered_core_logs
```

ShapeKit writes the binary masks, a combined map and `workflow_report.json` in
the output case folder. The combined map uses ShapeKit's IDs (26–49 by default).
If you pass `--ct_root`, each CT must match its masks' grid.

## Run a batch

Leave out `--case` to process the input folder, or use `--csv` with an
`Inference ID` column to select cases.

```bash
python main.py --input_folder /data/raw --output_folder /runs/ordered_core \
  --vertebrae_only --vertebrae_engine shapekit_ordered_core \
  --ordered_core_source /path/to/postprocessing_vertebrae.py \
  --cpu_count 2 --log_folder /runs/ordered_core_logs --continue_prediction
```

One worker is the default. Start there for a new dataset and measure memory
before adding workers. After an interruption, rerun with `--continue_prediction`.
ShapeKit checks completed outputs before skipping them and starts unfinished
cases again. Check `ordered_core.jsonl` in the log folder for failures. Use a
new output folder if the inputs, script, environment or label mapping change.

To compare with the standalone script, run it on the same raw masks. Compare the
binary masks directly. For the default combined map, subtract 25 from each
positive ShapeKit ID before comparing; leave background at zero.
