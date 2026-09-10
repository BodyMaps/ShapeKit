import numpy as np

from utils.vertebrae_refinement_core import LABEL_NAMES, process_array


def postprocessing_vertebrae_refinement(
    patient_id, segmentation_dict, reference_img, logger=None
):
    shape = reference_img.shape
    affine = np.asarray(reference_img.affine, dtype=float)
    if len(shape) != 3:
        raise ValueError("The reference image must be three-dimensional")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("The reference image must have a finite, invertible affine")
    combined = np.zeros(shape, dtype=np.uint8)
    present = set()
    for label, name in LABEL_NAMES.items():
        if name not in segmentation_dict:
            continue
        mask = np.asarray(segmentation_dict[name])
        if mask.shape != shape:
            raise ValueError(f"{name}: mask shape {mask.shape} differs from {shape}")
        if not np.isfinite(mask).all():
            raise ValueError(f"{name}: mask contains non-finite values")
        foreground = mask > 0
        if np.any(foreground & (combined != 0)):
            raise ValueError("Vertebral masks overlap; provide exclusive input labels")
        combined[foreground] = label
        present.add(name)
    if not np.any(combined):
        if logger is not None:
            logger.info("[%s] No vertebral foreground; refinement skipped", patient_id)
        return segmentation_dict
    refined, report = process_array(combined, affine)
    result = dict(segmentation_dict)
    for label, name in LABEL_NAMES.items():
        mask = refined == label
        if name in present or np.any(mask):
            result[name] = mask.astype(np.uint8)
    if logger is not None:
        logger.info(
            "[%s] Anatomical vertebrae refinement: changed=%d, foreground=%d -> %d",
            patient_id,
            report["changed_voxels"],
            report["input_foreground_voxels"],
            report["output_foreground_voxels"],
        )
    return result
