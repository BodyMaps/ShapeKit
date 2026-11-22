# Run the independent ordered-core vertebra method

The opt-in `shapekit_ordered_core` engine executes a user-supplied copy of the
independent warm-up implementation. The adapter uses its array loader, ordered
core watershed, conservative cleanup and NIfTI writer. It reproduces the
standalone glue sequence that combines the identity and cleanup stages. The
actual repair rules stay in the original source.

## Install and obtain the implementation

In a Python 3.12 environment, install these versions of existing ShapeKit
dependencies for the ordered-core CLI and tests:

```bash
python -m pip install numpy==1.26.4 scipy==1.14.1 scikit-image==0.24.0 nibabel==5.3.2 PyYAML==6.0.4 connected-components-3d==4.1.1 matplotlib==3.9.4 tqdm==4.70.1
python -m pip install pytest==9.2.0
```

Obtain the original submitted `postprocessing_vertebrae.py` from the warm-up
submission author. This external prerequisite is not distributed by ShapeKit.
The source shared with the evaluated submission has SHA-256
`8854c4d4c4c3d9b078238ed0cfc9501c73296b79e4d0f3edf99e93add683396d`.
Use `sha256sum` or PowerShell `Get-FileHash -Algorithm SHA256` to verify it.
The CLI executes the trusted source you provide and calls its `get_array`,
`refine_ordered_cores`, `process_case_cleanup`, and `save_like` functions.
Algorithm thresholds and cleanup defaults come from that file.

## Run one case

From the ShapeKit root, with case `case_a` under `/data/raw`:

```bash
python main.py --input_folder /data/raw --output_folder /runs/ordered_core \
  --vertebrae_only --vertebrae_engine shapekit_ordered_core \
  --ordered_core_source /path/to/submission_independent/postprocessing_vertebrae.py \
  --case case_a --cpu_count 1 --log_folder /runs/ordered_core_logs
```

Input is one directory per case with a vertebra-only integer
`combined_labels.nii.gz` (background 0, L5 through C1 in IDs 1–24) and
`segmentations/vertebrae_*.nii.gz` binary files. Mixed organ inputs fail
explicitly. Named masks and the combined map must agree on the native grid.
The optional CT is checked for geometry only. Missing empty masks become
explicit zeros at export. ShapeKit writes its configured IDs (26–49 by default),
the named binary masks and `workflow_report.json` in a new case directory.
Keep `if_save_combined_label: true` and `subfolder_name: segmentations` in
`config.yaml`. The configured class map must include all 24 vertebrae.

## Run and recover a batch

```bash
python main.py --input_folder /data/raw --output_folder /runs/ordered_core \
  --vertebrae_only --vertebrae_engine shapekit_ordered_core \
  --ordered_core_source /path/to/postprocessing_vertebrae.py \
  --ct_root /data/ct --cpu_count 2 --log_folder /runs/ordered_core_logs \
  --continue_prediction
```

Omit `--case` to select all case folders, repeat it to select specific cases, or
pass a ShapeKit `--csv` with an `Inference ID` column. One worker is the default.
Larger values run separate processes, one volume at a time per worker, within
CPU and platform limits. Choose the worker count from measured peak memory on
representative volumes and leave disk space for intermediates. A worker limit
caps simultaneous cases but cannot enforce a RAM budget.

The optional `--ct_root` above requires matching CT geometry for each case.
The output root must be separate from the input and CT roots. The final case
folder appears by rename on the same filesystem after validation. An interruption
can leave scratch under `.incomplete`. Resume verifies completed cases and restarts
unfinished cases from raw input. Changed inputs, source, dependencies or label
mapping require a fresh output location. Corrupted outputs fail verification.
Case errors allow other cases to continue and make the batch exit nonzero. A
worker-process crash can abort the pool, with completed cases available for
restart. The manifest records source and input digests, class mapping, key
runtime versions, and output digests. Structured events go to stdout and
`ordered_core.jsonl` in the log folder. Use a separate log folder for each job.

This branch also retains [the importer](IMPORT_VERTEBRAE.md) for cases that
have already been refined. For the raw-input path use the engine above. It
leaves the default backends unchanged. The method can abstain on missing,
merged or implausible cores, and a plausible false core can still pass the gate.
Case-level reports record that decision. Two demo cases cannot establish
accuracy or throughput on new clinical datasets.

## Verify output equivalence

```bash
python /path/to/postprocessing_vertebrae.py --in_dir /data/raw \
  --out_dir /runs/standalone --report_dir /runs/standalone_reports
```

Using the same cases and source as the ShapeKit batch, compare every decoded
binary mask and the native spatial metadata. In the combined map, keep
background at zero and map ShapeKit IDs back by vertebra name. With the default
class map this subtracts 25 from positive IDs. Compare the normalized
combined arrays voxel for voxel. This checks prediction agreement with the
standalone method. Measuring accuracy requires separate reference labels.

```bash
python -m pytest tests/test_ordered_core_workflow.py tests/test_import_vertebrae.py -q
```
