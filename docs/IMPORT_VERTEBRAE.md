# Import an existing standalone vertebra refinement

ShapeKit already provides several refinement engines. `tools/import_vertebrae.py`
is a separate, opt-in IO bridge for a standalone algorithm whose binary predictions
must survive integration unchanged. It does not add another refinement engine or
change `main.py`, `config.yaml`, or existing defaults.

## Input contract

Each case contains `combined_labels.nii.gz` and named binary files under
`segmentations/vertebrae_L5.nii.gz` through `vertebrae_C1.nii.gz`. The combined
source must contain only background and vertebra IDs: 1=L5 through 24=C1 by default,
or 26=L5 through 49=C1 with `--label-offset 25`. Declare the scheme explicitly.
The checked-in ShapeKit class map exports 26=L5 through 49=C1.

The tool checks 3D shape, finite nonsingular affine, positive spacing, conflicting
coded qform/sform transforms, binary values, exact binary/combined correspondence,
and overlaps. It rejects unaligned CT or masks without interpolation. Empty missing
classes become zero files; a missing populated class fails. Other named organ masks
are copied unchanged and included in the combined export only if configured and
nonoverlapping. Existing ShapeKit multi-organ combined maps are not a supported
source: export a vertebra-only combined map first.

## Reproducible environment

Python 3.12 CPU environment used for the bridge:

```bash
python -m venv .venv-import
# Activate the virtual environment for your platform, then:
python -m pip install numpy==1.26.4 nibabel==5.3.2 PyYAML==6.0.4 pytest==9.2.0
```

Use your standalone algorithm's additional pinned dependencies separately.

## Example: the independent SuPreM warm-up

Run the existing standalone script once against untouched raw predictions, into a
new output directory. The script is an external prerequisite, not vendored here.

```bash
python /path/to/submission_independent/postprocessing_vertebrae.py \
  --in_dir /data/AbdomenAtlasDemoPredict_raw \
  --out_dir /runs/independent \
  --report_dir /runs/independent_reports
python tools/import_vertebrae.py \
  --input-root /runs/independent --output-root /runs/shapekit \
  --ct-root /data/AbdomenAtlasDemo
```

The source masks keep their compressed bytes, dtype, qform and sform. The combined
map intentionally changes numeric IDs into ShapeKit's configured scheme. The tool
does not claim a DSC score; the report records `dsc: null`. An evaluated standalone
prediction and an integrated prediction are distinct artifacts even when named
binary masks are identical.

## Batch and interruption behavior

Cases run sequentially in sorted order for bounded memory. `--case` can repeat.
Every case validates before writing into a unique `.incomplete` staging folder.
Only a complete case is renamed into the final location. Batch failures are reported
as JSON and exit status 1 while other cases continue. Argument errors exit 2.
Error messages use `E_*` identifiers. A failed stage stays available for diagnosis.

```bash
python tools/import_vertebrae.py --input-root /runs/independent \
  --output-root /runs/shapekit --ct-root /data/AbdomenAtlasDemo --resume
python -m pytest tests/test_import_vertebrae.py -q
```

`--resume` skips a completed case only after checking input/configuration/importer
fingerprints and every output hash. Corrupt or changed output fails. Re-importing
an exported case with `--label-offset 25` is voxel-idempotent for the standard
class map. The morphology of the external algorithm has its own idempotence and
accuracy requirements, which the importer cannot certify.

No patient data or masks are bundled with this change. The two available warm-up
cases are a compatibility check, not evidence of production scaling or clinical
performance. For larger runs, measure throughput, memory and failure rates on a
representative cohort before choosing worker counts.
