"""ShapeKit-Anchor vertebrae engine: adapter between ShapeKit and the engine.

ShapeKit hands each engine a dict of per-organ binary masks (vertebrae_L5 ...
vertebrae_C1, all reoriented to the reference image's axcodes). The engine in
vertebrae_anchor_engine.py works on one combined uint8 volume (1 = L5 ... 24 =
C1). This module converts in both directions, finds the optional inputs (the
case CT, the aorta and celiac trunk masks used as naming landmarks), and
writes a one-line summary to the ShapeKit log.

What the engine does, in one paragraph: it re-derives vertebra names from
geometry instead of trusting the network's per-voxel labels. On a clean spine
that is a rank renumbering anchored to the network's own votes, which changes
nothing when the network was right. On a scrambled spine (split and merged
bodies, repeated labels) it finds the levels the network got right, fits a
smooth curve of position against level through them, reads off where the
other levels must sit, and cuts the spine mask into slabs there, placing each
cut by volume share snapped to the disc waist. Anchor levels keep the
network's own boundary. Optionally the planar boundary of each rebuilt level
is replaced by a learned one from nnInteractive (see vertebrae_anchor_prompt).

Deployment notes:
  - No new required dependencies: numpy, scipy, nibabel and
    connected-components-3d are already ShapeKit requirements.
  - CPU path: about 35 s for a 2.5 mm abdominal case and about 2.5 min for
    a 0.7 mm whole-spine case, single core. The engine crops to the
    vertebrae bounding box first and adds about 2 GB above the masks
    ShapeKit already holds (24 full-size masks are about 9 GB on the 0.7 mm
    case).
  - Optional GPU path (vertebrae_prompt_model: nninteractive): needs a CUDA
    GPU with 8 to 10 GB free and `pip install nninteractive`. Run with
    --cpu_count 1 when it is on, since every worker would otherwise load its
    own copy of the model. When the package or the GPU is missing the engine
    logs it and returns the CPU result.
  - Graceful degradation: a missing or misaligned CT disables the bone check
    and the prompt stage, and nothing else. The naming repair never needs
    the CT.
  - A case whose naming cannot be resolved with confidence (too few reliable
    levels to fit the position model) keeps the network's names, cleaned of
    islands, and is flagged needs_review in the log rather than guessed.

Author: Ura Modi (ura.modi24@gmail.com)
"""

import os

import nibabel as nib
import numpy as np
from nibabel.orientations import (apply_orientation, axcodes2ornt,
                                  io_orientation, ornt_transform)

from . import vertebrae_anchor_engine as engine

# ShapeKit organ names <-> engine ids (engine: 1 = L5 ... 24 = C1)
VERTEBRA_NAMES = [engine.CLASS_MAP[k] for k in sorted(engine.CLASS_MAP)]
NAME_TO_ID = {v: k for k, v in engine.CLASS_MAP.items()}
LANDMARK_ORGANS = ("aorta", "celiac_trunk")


def _load_ct_aligned(ct_path, reference_img, logger, patient_id, shape):
    """Load the case CT reoriented to the reference axcodes, the same way
    ShapeKit reorients the masks, so CT voxels match mask voxels one to one.
    Returns an int16 array or None (missing, unreadable, different grid)."""
    if ct_path is None or not os.path.exists(ct_path):
        logger.info(
            f"[ShapeKit-Anchor] {patient_id}: no CT at {ct_path}; "
            f"bone check and prompt stage off, naming repair unaffected")
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
            f"[ShapeKit-Anchor] {patient_id}: CT load failed ({e}); "
            f"bone check and prompt stage off")
        return None
    if ct.shape != shape:
        logger.warning(
            f"[ShapeKit-Anchor] {patient_id}: CT grid {ct.shape} does not "
            f"match masks {shape}; bone check and prompt stage off")
        return None
    return ct


def _prompt_available(prompt_model, prompt_device, logger, patient_id):
    """Check the optional GPU stage before the engine tries to use it, so the
    log says why it was skipped rather than a traceback fragment."""
    if prompt_model != "nninteractive":
        return "none"
    try:
        import torch
        if prompt_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                f"[ShapeKit-Anchor] {patient_id}: prompt stage requested but "
                f"CUDA is not available; using planar cuts")
            return "none"
        import nnInteractive  # noqa: F401
    except ImportError as e:
        logger.warning(
            f"[ShapeKit-Anchor] {patient_id}: prompt stage requested but "
            f"{e}; using planar cuts (pip install nninteractive)")
        return "none"
    return "nninteractive"


def postprocessing_vertebrae_anchor(patient_id, segmentation_dict,
                                    reference_img, ct_path, logger,
                                    prompt_model="none",
                                    prompt_device="cuda:0",
                                    trust_landmarks=False):
    """Takes: patient_id, the ShapeKit segmentation dict (organ name ->
        binary mask, all reoriented to the reference axcodes), the reference
        nibabel image, the path to the case CT (may be None or missing), a
        logger, and the engine options from config.yaml.
    Does: assembles the vertebra masks into one labelled volume, runs the
        anchor-and-slab engine (island and leakage cleanup, naming from
        geometry, slab cuts by volume share and disc waist, thoracic
        watershed, staircase smoothing, optional learned boundaries), and
        writes the result back into the dict. Every label in the output is
        one connected component and the levels present form a contiguous run.
    Returns: the segmentation dict with the vertebrae masks replaced."""
    present = [n for n in VERTEBRA_NAMES
               if segmentation_dict.get(n) is not None
               and np.any(segmentation_dict[n])]
    if len(present) < 3:
        logger.info(
            f"[ShapeKit-Anchor] {patient_id}: {len(present)} vertebra masks "
            f"present, nothing to repair")
        return segmentation_dict

    shape = segmentation_dict[present[0]].shape
    affine = np.asarray(reference_img.affine, dtype=np.float64)

    # ---- assemble engine labels (1 = L5 ... 24 = C1) --------------------
    # Descending order, so that where two masks claim a voxel the more
    # inferior label does not silently win by dict ordering.
    seg = np.zeros(shape, dtype=np.uint8)
    for name in sorted(present, key=lambda n: -NAME_TO_ID[n]):
        m = segmentation_dict[name] > 0
        seg[m & (seg == 0)] = NAME_TO_ID[name]

    ct = _load_ct_aligned(ct_path, reference_img, logger, patient_id, shape)
    prompt_model = _prompt_available(prompt_model, prompt_device, logger,
                                     patient_id)

    # Landmarks come from the same dict when the pipeline segments them; they
    # share the reference grid, so they carry the reference affine.
    organ_masks = {}
    for organ in LANDMARK_ORGANS:
        m = segmentation_dict.get(organ)
        if m is not None and np.any(m):
            organ_masks[organ] = (m > 0, affine)

    out, stats = engine.refine_case(
        seg, affine, ct=ct, organ_masks=organ_masks or None,
        trust_anchor=trust_landmarks, prompt_model=prompt_model,
        prompt_device=prompt_device,
        log=lambda msg: logger.info(f"[ShapeKit-Anchor] {patient_id}: {msg}"))

    # ---- one-line summary for postprocessing.log ------------------------
    resolver = stats.get("resolver", "rank" if stats.get("labels_renumbered", 0)
                         or stats.get("order_violations_fixed", 0) else "none")
    rebuilt = stats.get("slab_rejected_levels", [])
    summary = (
        f"[ShapeKit-Anchor] {patient_id}: resolver={resolver}, "
        f"bodies={stats.get('bodies_found')}, "
        f"islands_removed={stats.get('islands_removed')}, "
        f"labels_renumbered={stats.get('labels_renumbered')}, "
        f"rebuilt_levels={[n.replace('vertebrae_', '') for n in rebuilt]}, "
        f"prompted={[n.replace('vertebrae_', '') for n in stats.get('prompted_levels_accepted', [])]}, "
        f"levels={len(stats.get('levels_present', []))} "
        f"contiguous={stats.get('contiguous')}")
    if stats.get("needs_review"):
        summary += " NEEDS_REVIEW"
    if stats.get("anchor_disagreement"):
        summary += f" landmark_disagreement={stats['anchor_disagreement']}"
    logger.info(summary)
    for w in stats.get("warnings", []):
        logger.warning(f"[ShapeKit-Anchor] {patient_id}: {w}")

    # ---- write repaired labels back into the ShapeKit dict --------------
    for name in VERTEBRA_NAMES:
        mask = (out == NAME_TO_ID[name]).astype(np.uint8)
        if mask.any() or segmentation_dict.get(name) is not None:
            segmentation_dict[name] = mask
    return segmentation_dict
