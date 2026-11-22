# Run the independent ordered-core vertebra method

The opt-in `shapekit_ordered_core` engine executes a user-supplied copy of the
independent warm-up implementation. The adapter uses its array loader, ordered
core watershed, conservative cleanup and NIfTI writer. It reproduces the
standalone glue sequence that combines the identity and cleanup stages. The
actual repair rules stay in the original source.

From the ShapeKit root, after installing the normal ShapeKit dependencies:

```bash
python main.py --input_folder /data/raw --output_folder /runs/ordered_core \
  --vertebrae_only --vertebrae_engine shapekit_ordered_core \
  --ordered_core_source /path/to/submission_independent/postprocessing_vertebrae.py \
  --ct_root /data/ct --cpu_count 1
```

The argument executes local Python code. Use a trusted, local copy of the
independent script. The adapter expects `get_array`, `refine_ordered_cores`,
`process_case_cleanup`, and `save_like` with their submitted signatures. The
source also requires SciPy and scikit-image. The integration was tested with
Python 3.12, NumPy 1.26.4, SciPy 1.14.1, scikit-image 0.24.0, and nibabel 5.3.2.

Input is one directory per case with a vertebra-only integer
`combined_labels.nii.gz` (background 0, L5 through C1 in IDs 1–24) and
`segmentations/vertebrae_*.nii.gz` binary files. Mixed organ inputs fail
explicitly. Named masks and the combined map must agree on the native grid.
The optional CT is checked for geometry only. Missing empty masks become
explicit zeros at export. ShapeKit writes its configured IDs (26–49 by default),
the named binary masks and `workflow_report.json` in a new case directory.

Cases run in sorted order, one at a time. Use repeated `--case` or standard
`--csv` to select cases. Add `--continue_prediction` to verify every recorded
input and output hash before skipping a completed case. Changed input,
algorithm, adapter, mapping or output requires a new output location. A failed
case never appears as complete and causes a nonzero batch exit. The output
root must be separate from input and CT roots.

This branch also retains [the importer](IMPORT_VERTEBRAE.md) for cases that
have already been refined. For the raw-input path use the engine above. It
leaves the default backends unchanged. The method can abstain on missing,
merged or implausible cores, and a plausible false core can still pass the gate.
Case-level reports record that decision. Two demo cases cannot establish
accuracy or throughput on new clinical datasets.

```bash
python -m pytest tests/test_ordered_core_workflow.py tests/test_import_vertebrae.py -q
```
