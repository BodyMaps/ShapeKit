# Anchor-graph vertebrae engine (`shapekit_anchor_graph`)

It is an optional vertebrae engine. It does not change the default (`shapekit_songlin`) and runs inside the normal
ShapeKit pipeline, so the other organs are still post-processed as usual.

`utils/vertebrae_anchor_graph_engine.py` is the standalone warm-up script, kept unchanged; its SHA256 is
written into every case report. `utils/vertebrae_anchor_graph.py` is the ShapeKit adapter.

## Method

All steps work in a superior-up view of a crop around the spine; nothing is resampled.

1. **Order check** – each label's main body must follow anatomical order (L5 lowest … C1 highest); a body out
   of order is demoted to a fragment (a correctly placed piece of the same label replaces it when one exists).
2. **Contact-graph relabelling** – fragments touching the spine take the label they share the most surface with.
3. **Isolated fragments** – kept (as the nearest vertebra) only within `near_mm`, otherwise removed.
4. **Split-bone merge** – one bone split across two labels is merged into one label; nothing is renumbered,
   the resulting gap is reported as a flag instead (at most `max_merges` per case).
5. **CT step (only with a CT)** – discs are detected from the label area and endplate density; a run of
   shifted labels enclosed by two agreeing blocks is renumbered, and each repair is rejected unless every
   affected vertebra is one piece with a plausible volume. Then labelled fat/air reached from outside is
   trimmed, the missed cortical rim is added and bone-dense notches are filled.
6. **Hole fill** – enclosed holes inside each vertebra.

Every voxel change is logged in the report; the engine asserts that no change goes unlogged and that voxels
are only added by fill, notch and rim steps.

## Enable

```yaml
vertebrae_engine: shapekit_anchor_graph
ct_file_name: ct.nii.gz
```

or per run: `python main.py --vertebrae_engine shapekit_anchor_graph --input_folder ... --output_folder ...`.
Add `--vertebrae_only` to skip the other organs.

## Input

The usual ShapeKit layout, `CASE/segmentations/vertebrae_L5.nii.gz … vertebrae_C1.nii.gz`. The CT is **optional**: it is used only when it has the same shape and affine as the masks;
a missing or misaligned CT is logged and the case runs mask-only. Overlapping vertebra masks are resolved
in favour of the higher label.

## Output

- All 24 vertebra masks are written for every case, **including empty ones**, so a vertebra removed or merged
  by the engine does not reappear from the input copy that `main.py` places in the output folder first.
- `CASE/vertebrae_anchor_graph_report.json`: `status` (`ok` / `engine_failed`), `ct_used`, per-label metrics
  before/after, every change with location and reason, `flags` (missing labels, pitch/volume outliers,
  truncation by the field of view, large reassignments, rejected CT repairs, …), runtime, engine SHA256.
- If the engine raises for a case, the error is logged, the input vertebrae are kept unchanged and the report
  says `engine_failed`; the batch continues. Such a case is still listed as "Successfully processed" in
  `postprocessing.log` (that line is written for every engine once `main()` returns), so check `status` in the
  report or search the log for `engine failed`.
- A missing, unreadable or misaligned CT runs the case mask-only; a CT with a trailing singleton axis
  `(x, y, z, 1)` is accepted.

## Validation

```sh
python -m unittest tests.test_vertebrae_anchor_graph -v
```

The tests use synthetic volumes. Integration: fragment removal, empty-label overwrite, saver regression,
no-vertebrae, misaligned, unreadable and 4D CT paths, engine failure, and voxel-for-voxel equality with the
standalone script. Engine steps: a touching wrong-label fragment is reassigned, a vertebra split across two labels
is merged and the gap flagged, a missed bone rim is grown from the CT, and a second pass changes nothing. They
check behaviour, not accuracy: no DSC against ground truth is claimed here.

## Scale

CPU only; one case per worker through `--cpu_count`. The engine works on the spine crop and reads only the
matching CT block, so memory is dominated by ShapeKit loading all organ masks of a case. Measure peak memory
on a few real cases before raising `--cpu_count`.
