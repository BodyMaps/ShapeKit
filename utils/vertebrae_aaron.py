"""ShapeKit-Aaron vertebrae engine: anatomical re-identification from the masks alone.

Adapter between the ShapeKit pipeline (per-organ binary masks, ids 26..49) and the
label-map engine in vertebrae_aaron_engine.py (combined uint8 labels, 1 = L5 ...
24 = C1, closest-canonical RAS orientation).

What the engine does (details in docs/VERTEBRAE_AARON.md):
  - removes components not attached to the spinal column;
  - finds one core per vertebral body by a 5 mm erosion, splitting merged bodies;
  - re-identifies the bodies by dynamic programming so labels follow L5 -> C1,
    which fixes runs shifted by one level and labels used on two vertebrae;
  - gives each body one label, then gives every posterior-element voxel the label
    of the body it connects to through the pedicles (cheapest path in the mask);
  - keeps one connected piece per label and fills holes.
Vertebrae without a usable body core (cervical, or cut off at the edge of the
scan) keep their AI labels.

Deployment notes:
  - No CT needed and no new dependencies (numpy, scipy, nibabel, scikit-image,
    connected-components-3d).
  - CPU only, one core per case: the engine takes about 1.5 min on a 2.5 mm case
    and 4.5 min on a 0.7 mm whole-spine case (AbdomenAtlasDemo 006 and 031),
    before ShapeKit's own reading and writing of the per-organ masks.
"""

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

from . import vertebrae_aaron_engine as engine

# engine ids 1..24 in order: vertebrae_L5 ... vertebrae_C1 (same names as ShapeKit)
VERTEBRA_NAMES = [engine.CLASS_MAP[k] for k in range(1, engine.NUM_LABELS + 1)]


def postprocessing_vertebrae_aaron(patient_id, segmentation_dict, reference_img, logger):
    """Replace the vertebra masks in the ShapeKit segmentation dict with the refined ones.

    segmentation_dict maps organ names to binary masks reoriented to the axcodes of
    reference_img, whose affine therefore describes every mask. Masks of other organs
    are left untouched.
    """
    present = [n for n in VERTEBRA_NAMES
               if segmentation_dict.get(n) is not None and np.any(segmentation_dict[n])]
    if len(present) < 3:
        logger.info(f"[ShapeKit-Aaron] {patient_id}: {len(present)} vertebra masks "
                    f"present, nothing to refine")
        return segmentation_dict

    shape = segmentation_dict[present[0]].shape
    lab = np.zeros(shape, np.uint8)
    for k, name in enumerate(VERTEBRA_NAMES, start=1):
        if name in present:
            lab[segmentation_dict[name] > 0] = k

    # the engine works in closest-canonical (RAS) orientation: axis 2 is superior
    img = nib.Nifti1Image(lab, reference_img.affine)
    can = nib.as_closest_canonical(img)
    zooms = np.asarray(can.header.get_zooms()[:3], float)
    new, log = engine.postprocess(np.asarray(can.dataobj).astype(np.uint8), zooms, can.affine)
    new = apply_orientation(new, ornt_transform(io_orientation(can.affine),
                                                io_orientation(img.affine))).astype(np.uint8)

    logger.info(
        f"[ShapeKit-Aaron] {patient_id}: removed {log.get('fragments_removed', 0)} fragments, "
        f"relabeled {len(log.get('relabeled_vertebrae', []))} vertebrae and "
        f"{log.get('arch_voxels_relabeled', 0)} arch voxels, "
        f"filled {log.get('hole_voxels_filled', 0)} hole voxels")
    for change in log.get("relabeled_vertebrae", []):
        logger.info(f"[ShapeKit-Aaron] {patient_id}:     {change}")

    for k, name in enumerate(VERTEBRA_NAMES, start=1):
        mask = (new == k).astype(np.uint8)
        if mask.any() or segmentation_dict.get(name) is not None:
            segmentation_dict[name] = mask
    return segmentation_dict
