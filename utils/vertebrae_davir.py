"""ShapeKit-DAVIR vertebrae engine (vertebrae_engine: shapekit_davir).

Adapter between ShapeKit (one binary mask per vertebra, ids 26..49) and the DAVIR engine in
vertebrae_davir_engine.py (one label map, 1 = L5 ... 24 = C1). DAVIR re-identifies vertebrae
from disc positions along the spine and changes only the labels that disagree with that naming.
The case CT is optional: with it, disc detection also uses attenuation and the mask surface is
refined; without it, DAVIR runs on the masks alone.
"""
import os

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, ornt_transform

from . import vertebrae_davir_engine as davir

VERTEBRA_NAMES = [f"vertebrae_{davir.NAMES[k]}" for k in range(1, davir.NUM_LABELS + 1)]


def _load_ct(ct_path, reference_img, shape, logger, patient_id):
    """CT reoriented like the masks, or None if it is missing or on another grid."""
    if not ct_path or not os.path.exists(ct_path):
        logger.info(f"[DAVIR] {patient_id}: no CT found, running on masks only")
        return None
    ct = nib.load(ct_path)
    ct = ct.as_reoriented(ornt_transform(io_orientation(ct.affine), io_orientation(reference_img.affine)))
    if ct.shape[:3] != shape or not np.allclose(ct.affine, reference_img.affine, atol=1e-2):
        logger.warning(f"[DAVIR] {patient_id}: CT grid differs from the masks, running on masks only")
        return None
    return np.asarray(ct.dataobj).astype(np.int16)


def postprocessing_vertebrae_davir(patient_id, segmentation_dict, reference_img, ct_path, logger):
    present = [n for n in VERTEBRA_NAMES if n in segmentation_dict]
    if not present:
        return segmentation_dict
    shape = segmentation_dict[present[0]].shape
    if any(segmentation_dict[n].shape != shape for n in present):
        logger.warning(f"[DAVIR] {patient_id}: vertebra masks have different shapes, skipped")
        return segmentation_dict

    labels = np.zeros(shape, np.uint8)
    for k, name in enumerate(VERTEBRA_NAMES, 1):
        if name in segmentation_dict:
            labels[segmentation_dict[name] > 0] = k
    ct = _load_ct(ct_path, reference_img, shape, logger, patient_id)

    log = {"case": patient_id}
    refined = davir.refine(labels, reference_img.affine, ct, log=log)
    renamed = [i["raw_majority"] + "->" + i["name"] for i in log.get("naming", {}).get("instances", [])
               if i["name"] != i["raw_majority"]]
    logger.info(f"[DAVIR] {patient_id}: mode={log.get('mode')}, bodies={log.get('n_instances')}, "
                f"relabelled={log.get('voxels_relabelled_frac')}, renamed={renamed}")

    for k, name in enumerate(VERTEBRA_NAMES, 1):
        segmentation_dict[name] = (refined == k).astype(np.uint8)
    return segmentation_dict
