"""ShapeKit-Pro vertebrae engine: evidence-gated repair of vertebra labels.

Adapter between the ShapeKit pipeline (per-organ binary masks, ids 26..49)
and the evidence-gated vertebrae engine in vertebrae_pro_engine.py
(combined uint8 labels, ids 1..24, plus the case CT).

Why a second engine: the default vertebrae module operates on the masks
alone. This engine additionally reads the case CT and repairs label errors
by RECOLORING inside the prediction envelope instead of deleting:
fragments are re-attached through CT-certified bone corridors, level-band
mass misassignment is re-arbitrated at image-detected disc planes, the
posterior arch is rebuilt from pedicle roots, and one-level-down spinous
chains are repaired by a caudal-flow re-derivation. Every risky stage
carries its own defect meter and reverts itself when it cannot prove
improvement, so the engine is safe to run at scale with one parameter set:
on clean cases the repair stages self-revert and the output is unchanged
in structure.

Deployment notes:
  - Zero new dependencies: numpy, scipy, nibabel, scikit-image, and
    connected-components-3d are already ShapeKit requirements.
  - CPU only. Runtime is dominated by resolution: ~2 min for a 2.5 mm
    case, ~30 min for a 0.7 mm whole-spine case, on 2 cores.
  - Memory: the engine crops to the vertebrae bounding box; peak is
    roughly 9 GB on a 0.7 mm whole-spine case. Budget workers
    accordingly (see --cpu_count).
  - Graceful degradation: when the case CT cannot be found or aligned,
    the function logs the reason and falls back to the default
    vertebrae module, so batch runs never stall on a missing CT.

Validation and full documentation (method report, per-stage QA schema,
native-resolution verification tooling):
    https://github.com/aj-das-research/jhu-bodymaps-warmup
"""

import os

import nibabel as nib
import numpy as np
from nibabel.orientations import (apply_orientation, axcodes2ornt,
                                  io_orientation, ornt_transform)

from . import vertebrae_pro_engine as engine
from .vertebrae_postprocessing import postprocessing_vertebrae as \
    legacy_postprocessing_vertebrae

# ShapeKit organ names <-> engine ids (engine: 1 = L5 ... 24 = C1)
VERTEBRA_NAMES = [f"vertebrae_{n}" for n in engine.NAMES_BOTTOM_UP]


def _load_ct_aligned(ct_path, reference_img, logger, patient_id):
    """Takes: ct_path, the case reference image (whose axcodes every mask in
        the segmentation dict was reoriented to), logger, patient_id.
    Does: loads the CT and reorients its array to the reference axcodes with
        the same nibabel orientation transform ShapeKit applies to masks, so
        CT voxels correspond one-to-one with the mask arrays.
    Returns: (ct int16 array, voxel zooms in mm, voxel volume in mm3), or
        None when the CT is missing, unreadable, or a different grid."""
    if ct_path is None or not os.path.exists(ct_path):
        logger.warning(
            f"[ShapeKit-Pro] {patient_id}: CT not found "
            f"({ct_path}); falling back to the default vertebrae module")
        return None
    try:
        ct_img = nib.load(ct_path)
        target_axcodes = nib.aff2axcodes(reference_img.affine)
        transform = ornt_transform(io_orientation(ct_img.affine),
                                   axcodes2ornt(target_axcodes))
        ct = np.asanyarray(ct_img.dataobj)
        ct = apply_orientation(ct, transform)
        ct = np.clip(ct, -1024, 3071).astype(np.int16)
    except Exception as e:  # noqa: BLE001 - batch runs must not stall
        logger.warning(
            f"[ShapeKit-Pro] {patient_id}: CT load failed ({e}); "
            f"falling back to the default vertebrae module")
        return None
    zooms = tuple(float(z) for z in reference_img.header.get_zooms()[:3])
    vox_mm3 = float(abs(np.linalg.det(reference_img.affine[:3, :3])))
    return ct, zooms, vox_mm3


def postprocessing_vertebrae_pro(patient_id, segmentation_dict,
                                 reference_img, ct_path, logger):
    """Takes: patient_id, the ShapeKit segmentation dict (organ name ->
        binary mask, all reoriented to the reference axcodes), the reference
        nibabel image, the path to the case CT, and a logger.
    Does: assembles the 24 vertebra masks into one labeled volume, runs the
        evidence-gated repair pipeline against the CT (triage, island guard,
        disc-band re-arbitration with the pedicle-root arch race, interface
        polish, bounded smoothing, envelope reclamation, multiview recolor,
        core-integrity surgery, caudal-flow imbrication repair, audit), and
        writes the repaired labels back into the dict. Recolor-only: no bone
        the model predicted is deleted, nothing is added beyond a
        CT-certified bridge. Falls back to the default vertebrae module when
        the CT is unavailable.
    Returns: the segmentation dict with the vertebrae masks replaced."""
    present = [n for n in VERTEBRA_NAMES
               if segmentation_dict.get(n) is not None
               and np.any(segmentation_dict[n])]
    if len(present) < 3:
        logger.info(
            f"[ShapeKit-Pro] {patient_id}: {len(present)} vertebra masks "
            f"present, nothing to repair")
        return segmentation_dict

    loaded = _load_ct_aligned(ct_path, reference_img, logger, patient_id)
    if loaded is None:
        return legacy_postprocessing_vertebrae(
            patient_id, segmentation_dict, logger=logger)
    ct_full, zooms, vox_mm3 = loaded

    shape = segmentation_dict[present[0]].shape
    if ct_full.shape != shape:
        logger.warning(
            f"[ShapeKit-Pro] {patient_id}: CT grid {ct_full.shape} does not "
            f"match masks {shape}; falling back to the default module")
        return legacy_postprocessing_vertebrae(
            patient_id, segmentation_dict, logger=logger)

    # ---- assemble engine labels (1 = L5 ... 24 = C1) --------------------
    seg_full = np.zeros(shape, dtype=np.uint8)
    for name in present:
        lid = engine.NAME_TO_ID[name.replace("vertebrae_", "")]
        seg_full[segmentation_dict[name] > 0] = lid

    # ---- crop and run the engine stage sequence -------------------------
    pad = np.maximum(np.round(
        engine.P["crop_margin_mm"] / np.asarray(zooms)).astype(int), 2)
    nz = np.nonzero(seg_full)
    lo = np.maximum(np.array([c.min() for c in nz]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in nz]) + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg_full[sl].copy()
    ct = np.ascontiguousarray(ct_full[sl])
    del ct_full

    aff = reference_img.affine.copy()
    aff[:3, 3] = (reference_img.affine
                  @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]

    qa = {"case": patient_id, "records": [], "bands": [], "flags": [],
          "polish": [], "smooth": []}
    raw = seg.copy()
    _, s_before = engine.audit(seg, aff, vox_mm3)
    seg = engine.stage1_triage(seg, ct, zooms, vox_mm3, qa["records"])
    seg = engine.stage2a_islands(seg, vox_mm3, qa["records"])
    seg = engine.stage2b_arbitrate(seg, raw, ct, aff, zooms, vox_mm3, qa)
    seg = engine.stage2c_interface_polish(seg, ct, zooms, vox_mm3, qa)
    seg = engine.stage3_smooth(seg, ct, zooms, vox_mm3, qa)
    seg = engine.stage2d_reclaim_pool(seg, raw, ct, zooms, vox_mm3, qa)
    seg = engine.stage2e_multiview_recolor(seg, ct, aff, zooms, vox_mm3, qa)
    seg = engine.stage2f_skeleton_relabel(seg, ct, aff, zooms, vox_mm3, qa)
    seg = engine.stage2g_imbrication(seg, ct, aff, zooms, vox_mm3, qa)
    _, s_after = engine.audit(seg, aff, vox_mm3)

    logger.info(
        f"[ShapeKit-Pro] {patient_id}: vertebrae audit "
        f"{s_before} -> {s_after}; flags={qa['flags']}")

    # ---- write repaired labels back into the ShapeKit dict --------------
    for name in VERTEBRA_NAMES:
        lid = engine.NAME_TO_ID[name.replace("vertebrae_", "")]
        mask = np.zeros(shape, dtype=np.uint8)
        mask[sl] = (seg == lid).astype(np.uint8)
        if mask.any() or segmentation_dict.get(name) is not None:
            segmentation_dict[name] = mask
    return segmentation_dict
