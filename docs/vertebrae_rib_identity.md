<h1 align="center">Rib-Anchored Vertebral Identity</h1>

An optional stage that adjudicates **which vertebral level is which** before the
configured `vertebrae_engine` performs its usual shape cleanup. It is disabled by
default; with the default configuration ShapeKit behaves exactly as it did before
this stage existed.

---

## 1. What it does

A vertebra segmentation can delineate bone well while assigning level *names* that
are displaced, so that a run of vertebrae carries the identity of a neighbouring
level. The bone is present and its shape is plausible, but the labels are wrong.

This stage estimates each level's superior position from **costovertebral rib
attachment geometry** and relabels only those levels whose current position
disagrees with that estimate by more than a fixed threshold.

The cue it uses is external: it reads the ribs, rather than deriving level identity
from the vertebral predictions themselves, their ordering, spacing, connected
components or internal consistency.

The anatomy it encodes:

| rib | articulation | predicted attachment |
|---|---|---|
| 1 | single facet on the T1 body | centroid of T1 |
| 2–9 | facets on T(N−1) and TN | midpoint of those two centroids |
| 10–12 | single facet on their own body | centroid of TN |

Left and right ribs are measured independently. A pair whose two sides disagree by
more than the canonical tolerance is discarded rather than trusted. The remaining
pairs form a least-squares system, together with a smoothness term, a soft tie to
the model's own centroids, and a strongly weighted lumbar anchor. Levels displaced
beyond
the threshold are relabelled at boundaries placed on CT-detected disc planes.

## 2. Why it is optional

It requires two inputs ShapeKit does not otherwise need — the case CT and a
precomputed rib segmentation. Mask-only inputs cannot supply them, so the stage is
off unless it is explicitly enabled and its inputs are present.

It is also deliberately conservative. It relabels only where the rib evidence
requires it, and leaves a case untouched when the evidence is absent, insufficient,
or in agreement with the existing labels.

## 3. Required inputs

| Input | Where it comes from | Requirement |
|---|---|---|
| Vertebra masks | the usual `segmentations/` folder | at least 3 non-empty |
| Case CT | `ct_file_name` / `ct_root` (shared with `shapekit_pro`) | same voxel grid as the prediction |
| Rib volume | `rib_file_name` / `rib_root` | TotalSegmentator `total` multilabel, same voxel grid |

"Same voxel grid" means identical array shape **and** a matching affine. Volumes
that do not already share the grid are rejected; this stage performs no resampling
and no reorientation.

Only the geometry of the rib masks (label ids 92–115) is read. TotalSegmentator's
vertebral labels are not consulted and play no part in any decision.

**ShapeKit does not run TotalSegmentator.** The rib volume is a precomputed input,
generated once per dataset outside this pipeline. Keeping it external avoids adding
a GPU inference dependency to a CPU post-processor, and keeps the rib label
convention pinned to a file you control rather than to an installed version.

## 4. Configuration

```yaml
# optional identity stage, applied BEFORE vertebrae_engine
vertebrae_identity: rib_anchor      # 'none' (default) disables the stage

# CT lookup, shared with shapekit_pro
ct_file_name: ct.nii.gz
# ct_root: /path/to/original/ct/cases

# rib volume lookup
rib_file_name: total.nii.gz
# rib_root: /path/to/totalsegmentator/cases

# optional: one small JSON QA record per case
# rib_qa_dir: /path/to/qa
```

The rib volume is looked up in the same order the CT is: inside the case directory
first, then under an external root.

1. `<input_case>/<rib_file_name>`
2. `<rib_root>/<case_id>/<rib_file_name>`
3. `<rib_root>/<case_id>.nii.gz`

## 5. Interaction with `vertebrae_engine`

The two settings are independent and compose:

```
vertebra prediction
        │
        ▼
vertebrae_identity   (optional — decides which level is which)
        │
        ▼
vertebrae_engine     (shapekit or shapekit_pro — cleans up shape)
        │
        ▼
   ShapeKit output
```

`vertebrae_identity` is not a value of `vertebrae_engine` and does not replace it.
This stage performs no shape cleanup of its own, so the selected engine still runs
afterwards exactly as configured.

## 6. Fallback behaviour

Every condition below leaves the masks exactly as they arrived, logs the reason,
and lets processing continue. A case that cannot be adjudicated is never degraded,
and a batch run is never halted by one.

| Condition | Behaviour |
|---|---|
| `vertebrae_identity: none` | stage does not run at all |
| Fewer than 3 vertebra masks | no correction |
| CT missing or unreadable | no correction |
| Rib volume missing or unreadable | no correction |
| CT or rib grid differs in shape or affine | no correction, no resampling |
| Spine centreline cannot be derived | no correction |
| Fewer than 4 usable rib pairs | no correction |
| A rib pair's two sides disagree beyond tolerance | that pair is excluded; the rest are used |
| No level exceeds the correction threshold | labels left unchanged |
| Any unexpected error | caught, logged, masks returned unchanged |

## 7. Directory layout

Rib volumes alongside each case:

```
INPUT (--input_folder)
└── BDMAP_00000031
    ├── ct.nii.gz
    ├── total.nii.gz            <- rib_file_name
    └── segmentations
        ├── vertebrae_L5.nii.gz
        ...
        └── vertebrae_C1.nii.gz
```

Or held separately, leaving the prediction folders untouched:

```
rib_root/
├── BDMAP_00000006/total.nii.gz
└── BDMAP_00000031/total.nii.gz
```

## 8. Example

```bash
python -W ignore main.py \
    --input_folder  /path/to/predictions \
    --output_folder /path/to/output \
    --log_folder    logs/rib_identity \
    --cpu_count     8
```

With `vertebrae_identity: rib_anchor` set in `config.yaml`, each case logs one line:

```
[ShapeKit-RibIdentity] BDMAP_00000031: ribs_used=[1,...,10] rejected_lr=[11, 12]
    corrected=['L1','T12','T11','T10','T9','T8'] moved=366151 untouched=0.7479
```

and, when a CT or rib volume is absent:

```
[ShapeKit-RibIdentity] BDMAP_00000006: rib volume not found (...); identity
    correction skipped
```

## 9. Resource notes

The stage holds the CT and the rib volume in memory in addition to the masks
ShapeKit already loads. On a 0.7 mm whole-spine case those are roughly 0.7 GB each,
so budget workers accordingly with `--cpu_count`; high worker counts on
high-resolution cohorts are memory-bound rather than CPU-bound. Runtime is dominated
by connected-component and profile computations over the volume, not by the
least-squares solve, which has fourteen unknowns.

## 10. Scope of validation

The method was developed and validated on the two-case BodyMaps vertebrae warm-up
set. The tests in `tests/test_vertebrae_rib_identity.py` cover the label mapping,
the rib measurement and rejection rules, the costovertebral equations, the solver,
the correction threshold, and every fallback path, using synthetic phantoms that
require no imaging data.

Behaviour on larger and more varied cohorts has not been measured. The stage is
default-off and conservative for that reason.

## 11. Why `rib_anchor` requires `shapekit_pro`

> [!IMPORTANT]
> `vertebrae_identity: rib_anchor` is only valid with
> `vertebrae_engine: shapekit_pro`. Pairing it with the default `shapekit`
> engine is rejected at startup, before any case is processed.

The default `shapekit` vertebra engine may reassign contiguous cranio-caudal
identities during its own cleanup. When the identity stage has deliberately
changed which level a body belongs to, that reassignment can undo the identity
stage and restore the original labelling. The run would appear to succeed while
discarding the correction on every case, which is worse than not running it at
all — so the combination is refused rather than warned about.

Measured on the synthetic shifted phantom, where the identity stage relabels
38,400 voxels:

| engine after the identity stage | voxels still differing from the input | outcome |
|---|---|---|
| `shapekit` (default) | 0 | correction lost |
| `shapekit_pro` | 38,400 | correction retained |

ShapeKit exits with status 2 and an explanatory message:

```
[ERROR] Incompatible configuration: vertebrae_identity: 'rib_anchor' cannot be
used with vertebrae_engine: 'shapekit'.
...
Resolve it in one of two ways:
  - set vertebrae_engine: shapekit_pro, which preserves the corrected
    identities; or
  - set vertebrae_identity: none to disable rib-anchored correction and keep
    the default engine.
```

No existing ShapeKit algorithm was modified to make the combination work. The
default engine's behaviour is left exactly as it is; only the unsupported
pairing is refused.

This check is a *configuration* error and is distinct from the per-case
conditions in §6. A missing CT, a missing rib volume, too few usable ribs or an
incompatible grid are properties of one case: they are logged, that case is
skipped, and the batch continues. Only a configuration that would silently
discard the correction for every case stops the run.
