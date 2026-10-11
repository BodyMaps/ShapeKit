"""ShapeKit adapter for the ShapeKit-Asfand vertebrae engine (vertebrae_engine: shapekit_asfand).

Bridges ShapeKit's per-organ binary masks (class_map ids 26 = L5 ... 49 = C1)
and the engine's combined array (ids 1 = L5 ... 24 = C1, superior axis last).
Mask-only: no CT needed. On any internal error the case is logged and the
vertebra masks are returned unchanged, so batch runs never stall.
See docs/VERTEBRAE_ASFAND.md.
"""

import nibabel as nib
import numpy as np

from . import vertebrae_asfand_engine as engine

VERTEBRA_NAMES = [f"vertebrae_{n}" for n in engine.NAMES_BOTTOM_UP]  # engine id i -> VERTEBRA_NAMES[i - 1]


def _si_axis(reference_img):
    """Index of the superior-inferior axis and whether a higher index is more superior."""
    codes = nib.aff2axcodes(reference_img.affine)
    for ax, code in enumerate(codes):
        if code in ("S", "I"):
            return ax, code == "S"
    raise ValueError(f"No superior-inferior axis in orientation {''.join(codes)}")


def postprocessing_vertebrae_asfand(patient_id, segmentation_dict, reference_img, logger, cfg=None):
    """Takes: patient_id, ShapeKit segmentation dict (organ name -> binary array,
        already reoriented to the reference axcodes), the reference image, logger.
    Returns: a new segmentation dict with refined vertebra masks."""
    present = [n for n in VERTEBRA_NAMES if n in segmentation_dict]
    if not present:
        logger.info(f"[ShapeKit-Asfand] {patient_id}: no vertebra masks, skipping")
        return segmentation_dict
    try:
        shape = segmentation_dict[present[0]].shape
        comb = np.zeros(shape, dtype=np.uint8)
        for engine_id, name in enumerate(VERTEBRA_NAMES, 1):
            mask = segmentation_dict.get(name)
            if mask is not None:
                comb[(mask > 0) & (comb == 0)] = engine_id

        si, superior_up = _si_axis(reference_img)
        zooms = [float(z) for z in reference_img.header.get_zooms()[:3]]
        order = [a for a in range(3) if a != si] + [si]
        spacing = [zooms[a] for a in order]

        refined, report = engine.refine(
            np.transpose(comb, order), spacing, superior_up=superior_up, cfg=cfg,
            log=lambda msg: logger.info(f"[ShapeKit-Asfand] {patient_id}: {msg}"))
        refined = np.transpose(refined, np.argsort(order))
    except Exception as e:  # noqa: BLE001 - batch runs must not stall
        logger.warning(f"[ShapeKit-Asfand] {patient_id}: failed ({e}); vertebrae left unchanged")
        return segmentation_dict

    processed = segmentation_dict.copy()
    for engine_id, name in enumerate(VERTEBRA_NAMES, 1):
        processed[name] = (refined == engine_id).astype(np.uint8)
    logger.info(f"[ShapeKit-Asfand] {patient_id}: done, {report['relabelled_bodies']} bodies relabelled, "
                f"{report['fragments_removed']} fragments removed, {report['orphans_moved']} pieces reattached")
    return processed
