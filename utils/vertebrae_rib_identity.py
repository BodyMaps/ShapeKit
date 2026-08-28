"""ShapeKit adapter for rib-anchored vertebral identity correction.

This module is the integration layer only. It performs no anatomical
reasoning: every scientific decision is made by ``vertebrae_rib_identity_engine``,
which is a byte-for-byte copy of the accepted warm-up implementation. The
adapter is responsible for

  * discovering the case CT and the precomputed rib volume,
  * checking that they share the prediction's voxel grid,
  * converting between ShapeKit's per-organ binary masks (label ids 26-49)
    and the engine's combined label volume (ids 1-24),
  * logging one case-level record, and
  * failing safely, so a batch run is never interrupted by a case that
    lacks the inputs this stage needs.

The stage runs *before* the configured ``vertebrae_engine`` and is orthogonal
to it: it adjudicates which level is which, then hands the masks on for the
usual shape cleanup. It is disabled unless ``vertebrae_identity`` is set.

The rib volume is a precomputed TotalSegmentator ``total`` multilabel output.
Nothing here runs TotalSegmentator, and only the geometry of its rib masks is
read - its vertebral labels are not consulted at any point.
"""

import json
import os

import nibabel as nib
import numpy as np

from . import vertebrae_rib_identity_engine as engine

LOG_PREFIX = "[ShapeKit-RibIdentity]"

# Value of the ``vertebrae_identity`` config key that selects this stage.
IDENTITY_RIB_ANCHOR = "rib_anchor"

#: ``vertebrae_engine`` values this stage can run in front of. The default
#: engine is excluded deliberately - see ``validate_identity_config``.
COMPATIBLE_ENGINES = ("shapekit_pro",)


INCOMPATIBLE_ENGINE_MESSAGE = """\
Incompatible configuration: vertebrae_identity: '{identity}' cannot be used \
with vertebrae_engine: '{engine}'.

The rib-anchored identity stage changes which vertebral level a body is
assigned to. The default ShapeKit vertebra engine may then reassign contiguous
cranio-caudal identities during its own cleanup, which can undo the identity
stage and silently restore the original labelling. The run would appear to
succeed while discarding the correction on every case.

Resolve it in one of two ways:
  - set vertebrae_engine: shapekit_pro, which preserves the corrected
    identities; or
  - set vertebrae_identity: none to disable rib-anchored correction and keep
    the default engine.

See docs/vertebrae_rib_identity.md for details."""


class IdentityConfigError(ValueError):
    """Raised for a configuration that cannot produce a correct result.

    Distinct from the per-case conditions this stage tolerates: a missing CT,
    a missing rib volume, too few usable ribs or an incompatible grid are
    properties of one case, are handled by skipping that case, and must never
    stop a batch. This exception is for a combination of settings that would
    silently discard the identity correction for every case, which is worth
    refusing before any work starts."""


def validate_identity_config(vertebrae_identity, vertebrae_engine):
    """Takes: the configured ``vertebrae_identity`` and ``vertebrae_engine``.
    Does: checks the two settings can produce a meaningful result together.
        Intended to be called once at startup, before any case is processed.
    Returns: None when the configuration is usable.
    Raises: IdentityConfigError when the identity stage is enabled in front of
        an engine that would discard its output."""
    if vertebrae_identity in (None, "none", "None", ""):
        return
    if vertebrae_identity != IDENTITY_RIB_ANCHOR:
        raise IdentityConfigError(
            f"Unknown vertebrae_identity: {vertebrae_identity!r}. "
            f"Valid values are 'none' (default) and '{IDENTITY_RIB_ANCHOR}'.")
    if vertebrae_engine in COMPATIBLE_ENGINES:
        return
    raise IdentityConfigError(
        INCOMPATIBLE_ENGINE_MESSAGE.format(
            identity=IDENTITY_RIB_ANCHOR, engine=vertebrae_engine))


# --------------------------------------------------------------------------
# Label-space contract
#
# Two identity spaces exist and must never be confused:
#
#   engine space    1 .. 24   L5 .. C1   (the accepted warm-up convention)
#   ShapeKit space  26 .. 49  L5 .. C1   (config.yaml class_map)
#
# The two are related by a constant offset, but no arithmetic is used to
# convert between them anywhere in this integration. Both directions go
# through the explicit tables below, keyed by anatomical name, so a future
# change to either convention surfaces as a failed consistency check at
# import time rather than as silently mislabelled anatomy.
# --------------------------------------------------------------------------

#: ShapeKit combined-label id -> anatomical name, ordered L5 (26) to C1 (49).
SHAPEKIT_VERTEBRA_LABELS = {
    26: "vertebrae_L5", 27: "vertebrae_L4", 28: "vertebrae_L3",
    29: "vertebrae_L2", 30: "vertebrae_L1", 31: "vertebrae_T12",
    32: "vertebrae_T11", 33: "vertebrae_T10", 34: "vertebrae_T9",
    35: "vertebrae_T8", 36: "vertebrae_T7", 37: "vertebrae_T6",
    38: "vertebrae_T5", 39: "vertebrae_T4", 40: "vertebrae_T3",
    41: "vertebrae_T2", 42: "vertebrae_T1", 43: "vertebrae_C7",
    44: "vertebrae_C6", 45: "vertebrae_C5", 46: "vertebrae_C4",
    47: "vertebrae_C3", 48: "vertebrae_C2", 49: "vertebrae_C1",
}

#: anatomical name -> engine label id (1 = L5 ... 24 = C1)
ENGINE_ID_BY_NAME = {name: k for k, name in engine.CLASS_MAP.items()}
#: anatomical name -> ShapeKit combined-label id
SHAPEKIT_ID_BY_NAME = {name: i for i, name in SHAPEKIT_VERTEBRA_LABELS.items()}

#: vertebra names in inferior-to-superior order, L5 first.
VERTEBRA_NAMES = [engine.CLASS_MAP[k] for k in range(1, engine.N_CLASSES + 1)]


def _check_label_tables():
    """Fail loudly at import if the two identity spaces ever diverge."""
    if set(ENGINE_ID_BY_NAME) != set(SHAPEKIT_ID_BY_NAME):
        raise RuntimeError(
            "vertebra name sets differ between engine and ShapeKit label tables")
    engine_order = sorted(ENGINE_ID_BY_NAME, key=ENGINE_ID_BY_NAME.get)
    shapekit_order = sorted(SHAPEKIT_ID_BY_NAME, key=SHAPEKIT_ID_BY_NAME.get)
    if engine_order != shapekit_order:
        raise RuntimeError(
            "engine and ShapeKit label tables disagree on vertebra ordering")
    # Cross-check against the table the existing vertebrae module already
    # carries, when that module can be imported (it pulls optional deps).
    try:
        from .vertebrae_postprocessing import all_labels as upstream
    except Exception:  # noqa: BLE001 - the check is advisory, not required
        return
    if {int(k): v for k, v in upstream.items()} != SHAPEKIT_VERTEBRA_LABELS:
        raise RuntimeError(
            "SHAPEKIT_VERTEBRA_LABELS disagrees with vertebrae_postprocessing."
            "all_labels")


_check_label_tables()


def engine_id_to_shapekit_id(engine_id):
    """Takes: an engine label id (1..24).
    Returns: the corresponding ShapeKit combined-label id (26..49)."""
    return SHAPEKIT_ID_BY_NAME[engine.CLASS_MAP[engine_id]]


def shapekit_id_to_engine_id(shapekit_id):
    """Takes: a ShapeKit combined-label id (26..49).
    Returns: the corresponding engine label id (1..24)."""
    return ENGINE_ID_BY_NAME[SHAPEKIT_VERTEBRA_LABELS[shapekit_id]]


def name_to_engine_id(name):
    """Takes: an anatomical name such as ``vertebrae_T9``.
    Returns: the engine label id."""
    return ENGINE_ID_BY_NAME[name]


# --------------------------------------------------------------------------
# Input discovery and validation
# --------------------------------------------------------------------------

def resolve_rib_path(input_path, patient_id, rib_file_name, rib_root=None):
    """Takes: the case input directory, the case id, the configured rib file
        name, and an optional external rib root.
    Does: looks for the rib volume inside the case directory first, then under
        ``<rib_root>/<patient_id>/<rib_file_name>``, then for a flat
        ``<rib_root>/<patient_id>.nii.gz``. This mirrors how ShapeKit already
        locates the case CT.
    Returns: the first path that exists, else None."""
    if not rib_file_name:
        return None
    candidates = [os.path.join(input_path, rib_file_name)]
    if rib_root:
        candidates.append(os.path.join(rib_root, patient_id, rib_file_name))
        candidates.append(os.path.join(rib_root, f"{patient_id}.nii.gz"))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _load_aligned(path, reference_img, shape, kind, patient_id, logger):
    """Takes: a volume path, the case reference image, the mask array shape, a
        short label for messages, the case id and a logger.
    Does: loads the volume and requires it to sit on exactly the prediction's
        grid - identical array shape and an affine matching the reference. No
        resampling and no reorientation is performed: a volume that does not
        already share the grid is rejected rather than silently transformed.
    Returns: the array as int16, or None with the reason logged."""
    try:
        img = nib.load(path)
    except Exception as exc:  # noqa: BLE001 - batch runs must not stall
        logger.warning(f"{LOG_PREFIX} {patient_id}: {kind} unreadable "
                       f"({path}: {exc}); identity correction skipped")
        return None
    if tuple(img.shape) != tuple(shape):
        logger.warning(f"{LOG_PREFIX} {patient_id}: {kind} grid {tuple(img.shape)} "
                       f"does not match prediction {tuple(shape)}; "
                       f"identity correction skipped")
        return None
    if not np.allclose(img.affine, reference_img.affine, atol=1e-4):
        logger.warning(f"{LOG_PREFIX} {patient_id}: {kind} affine differs from the "
                       f"prediction affine; identity correction skipped "
                       f"(no resampling is performed)")
        return None
    try:
        return np.asarray(img.dataobj).astype(np.int16)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{LOG_PREFIX} {patient_id}: {kind} could not be read into "
                       f"memory ({exc}); identity correction skipped")
        return None


# --------------------------------------------------------------------------
# Label-volume assembly
# --------------------------------------------------------------------------

def assemble_label_volume(segmentation_dict):
    """Takes: the ShapeKit segmentation dict.
    Does: builds the engine's combined label volume from the individual binary
        vertebra masks. Levels are written in ascending id order, so where two
        masks claim the same voxel the more superior level wins - the same
        precedence ShapeKit's own ``combine_segmentation_dict`` applies.
    Returns: (uint8 label volume, list of names that were present), or
        (None, []) when no vertebra mask is present."""
    present = [n for n in VERTEBRA_NAMES
               if segmentation_dict.get(n) is not None
               and np.any(segmentation_dict[n])]
    if not present:
        return None, []
    shape = segmentation_dict[present[0]].shape
    volume = np.zeros(shape, dtype=np.uint8)
    for name in VERTEBRA_NAMES:          # ascending engine id == ascending ShapeKit id
        mask = segmentation_dict.get(name)
        if mask is None:
            continue
        volume[mask > 0] = ENGINE_ID_BY_NAME[name]
    return volume, present


def scatter_label_volume(volume, segmentation_dict):
    """Takes: an engine label volume and the ShapeKit segmentation dict.
    Does: writes each level back as a binary mask, replacing what was there.
        A level that was absent before and is still empty stays absent.
    Returns: the segmentation dict."""
    for name in VERTEBRA_NAMES:
        mask = (volume == ENGINE_ID_BY_NAME[name]).astype(np.uint8)
        if mask.any() or segmentation_dict.get(name) is not None:
            segmentation_dict[name] = mask
    return segmentation_dict


# --------------------------------------------------------------------------
# Stage entry point
# --------------------------------------------------------------------------

def _summarise(patient_id, log, logger):
    """Emit exactly one case-level line; the detail lives in the QA record."""
    if "status" in log:
        logger.info(f"{LOG_PREFIX} {patient_id}: {log['status']}")
        return
    churn = log.get("churn", {})
    logger.info(
        f"{LOG_PREFIX} {patient_id}: ribs_used={log.get('ribs_used', [])} "
        f"rejected_lr={[r['rib'] for r in log.get('ribs_excluded_lr', [])]} "
        f"corrected={log.get('levels_corrected', [])} "
        f"moved={log.get('voxels_moved', 0)} "
        f"untouched={churn.get('fraction_untouched')}")


def _write_qa(qa_dir, patient_id, log, logger):
    if not qa_dir:
        return
    try:
        os.makedirs(qa_dir, exist_ok=True)
        with open(os.path.join(qa_dir, f"{patient_id}.json"), "w") as handle:
            json.dump(log, handle, indent=2)
    except Exception as exc:  # noqa: BLE001 - QA output is never load-bearing
        logger.warning(f"{LOG_PREFIX} {patient_id}: could not write QA record ({exc})")


def postprocessing_vertebrae_rib_identity(patient_id, segmentation_dict,
                                          reference_img, ct_path, rib_path,
                                          logger, qa_dir=None):
    """Takes: the case id, the ShapeKit segmentation dict, the case reference
        image, the resolved CT path, the resolved rib-volume path, a logger,
        and an optional directory for per-case QA records.
    Does: runs rib-anchored identity correction on the vertebra masks. The CT
        and the rib volume must both be present and share the prediction's
        voxel grid; when either is missing or incompatible the masks are
        returned untouched and the reason is logged. All anatomical decisions
        are made by the vendored engine, unchanged.
    Returns: (segmentation dict, per-case log dict). The dict is returned
        unmodified on every fallback path, and this function does not raise."""
    log = {}
    try:
        volume, present = assemble_label_volume(segmentation_dict)
        if volume is None or len(present) < 3:
            log["status"] = (f"{len(present)} vertebra masks present; "
                             f"identity correction skipped")
            _summarise(patient_id, log, logger)
            return segmentation_dict, log

        shape = volume.shape

        if not ct_path or not os.path.exists(ct_path):
            log["status"] = (f"CT not found ({ct_path}); "
                             f"identity correction skipped")
            _summarise(patient_id, log, logger)
            return segmentation_dict, log

        if not rib_path or not os.path.exists(rib_path):
            log["status"] = (f"rib volume not found ({rib_path}); "
                             f"identity correction skipped")
            _summarise(patient_id, log, logger)
            return segmentation_dict, log

        ct = _load_aligned(ct_path, reference_img, shape, "CT", patient_id, logger)
        if ct is None:
            log["status"] = "CT incompatible; identity correction skipped"
            return segmentation_dict, log

        ribs = _load_aligned(rib_path, reference_img, shape, "rib volume",
                             patient_id, logger)
        if ribs is None:
            log["status"] = "rib volume incompatible; identity correction skipped"
            return segmentation_dict, log

        log["ct_path"] = ct_path
        log["rib_path"] = rib_path

        corrected, engine_log = engine.process(
            volume,
            reference_img.affine,
            reference_img.header.get_zooms()[:3],
            ct,
            ribs,
        )
        log.update(engine_log)

        if corrected is not None:
            segmentation_dict = scatter_label_volume(corrected, segmentation_dict)

        _summarise(patient_id, log, logger)
        _write_qa(qa_dir, patient_id, log, logger)
        return segmentation_dict, log

    except Exception as exc:  # noqa: BLE001 - one bad case must not stop a batch
        logger.error(f"{LOG_PREFIX} {patient_id}: identity stage failed ({exc}); "
                     f"masks left unchanged")
        log["status"] = f"identity stage failed ({exc}); masks left unchanged"
        return segmentation_dict, log
