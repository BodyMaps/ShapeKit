"""ShapeKit-Anchor, optional stage: learned boundaries for rebuilt levels.

The anchor-and-slab resolver in vertebrae_anchor_engine.py fixes the naming
of a scrambled thoracolumbar spine and places every level within a couple of
millimetres. What it cannot do is draw the boundary: a planar cut through the
disc gets the body right and hands part of every spinous process to the level
below, because a thoracic process slopes downward. No rule on the label map
recovers that; it needs a model that has learned what one vertebra looks like.

nnInteractive (Isensee et al. 2025) is a promptable 3D segmenter trained on
120+ datasets. Given one point inside a vertebra it returns that vertebra,
processes included, separated from its neighbours. It knows nothing about
level names, and it does not need to: the resolver supplies one prompt per
corrected level, and the model supplies the boundary. Anatomical reasoning
names the vertebrae; a foundation model delineates them.

Where it is applied. Only to levels the resolver rebuilt. Where the network's
own boundary survived (the anchors), that boundary is left alone. Learned
boundaries go only where the network's were lost.

Safeguards. Every prompted mask is checked before it is accepted: it must lie
mostly inside the union of everything the network called vertebra (dilated
slightly), it must be within a plausible volume ratio of the slab it replaces,
and where two accepted masks overlap the voxel goes to the nearer prompt along
the spine axis. A level whose prompt fails any check keeps its slab.

Deployment. This stage is off by default (vertebrae_prompt_model: none in
config.yaml) because it needs a CUDA GPU and the optional nnInteractive
package, which is not in requirements.txt and needs torch 2.x. When it is
switched on and either is missing, the engine logs the reason and returns the
planar-cut result, so batch runs never stall. Weights download on first use.
About 8 to 10 GB of VRAM; roughly one minute for nine levels on a 0.7 mm
whole-spine case.

Install (optional):  pip install nninteractive
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

PROMPT_MIN_RATIO = 0.55      # accepted mask volume / slab volume, lower bound
PROMPT_MAX_RATIO = 1.90      # upper bound
PROMPT_ENVELOPE_DILATION = 2 # voxels the network's envelope is grown by
PROMPT_MIN_INSIDE = 0.80     # share of the mask that must lie in the envelope
PROMPT_EXCLUDE = set()       # no level is excluded. C1 and C2 were excluded for one run on
                             # a measurement that turned out to compare a re-prompt against
                             # the first prompt; the first prompt on a rebuilt C2 beat the cut
                             # by six points and the exclusion cost C1 four more. Prompting
                             # anchors, which is where C1 and C2 really did go wrong, is
                             # already ruled out by the caller.


def _session(device: str = "cuda:0"):
    import torch
    from nnInteractive.model_management import ensure_model_available, get_default_model_id
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    path = ensure_model_available(get_default_model_id())
    s = nnInteractiveInferenceSession(device=torch.device(device), use_torch_compile=False,
                                      verbose=False, torch_n_threads=8, do_autozoom=True)
    s.initialize_from_trained_model_folder(str(path))
    return s


def _prompt_point(level_voxels: np.ndarray) -> tuple[int, int, int]:
    """A voxel of the level nearest its centroid, so the click lands in bone."""
    c = level_voxels.mean(axis=0)
    j = int(np.argmin(((level_voxels - c) ** 2).sum(axis=1)))
    return tuple(int(v) for v in level_voxels[j])


def refine_levels(ct: np.ndarray, seg: np.ndarray, levels: list[int],
                  axis_pos_of_voxel, log=print, device: str = "cuda:0") -> tuple[np.ndarray, dict]:
    """Replace the given levels in `seg` with prompted masks where they pass checks.

    ct, seg          full-resolution arrays on the same grid
    levels           label indices to re-delineate (the rebuilt ones)
    axis_pos_of_voxel  callable (N,3) voxel idx -> (N,) position along the spine
                       axis in mm, used to settle overlaps
    Returns the new label map and a per-level record.
    """
    import torch

    t0 = time.time()
    session = _session(device)
    session.set_image(ct.astype(np.float32)[None])
    log(f"    nnInteractive ready in {time.time() - t0:.0f}s")

    envelope = ndimage.binary_dilation(seg > 0, iterations=PROMPT_ENVELOPE_DILATION)
    out = seg.copy()
    record = {}
    masks: dict[int, np.ndarray] = {}
    centres: dict[int, float] = {}

    for k in levels:
        if k in PROMPT_EXCLUDE:
            record[k] = {"status": "skipped: atlas or axis"}
            continue
        idx = np.argwhere(seg == k)
        if not len(idx):
            record[k] = {"status": "absent"}
            continue
        pt = _prompt_point(idx)
        buf = torch.zeros(ct.shape, dtype=torch.uint8)
        session.set_target_buffer(buf)
        session.reset_interactions()
        session.add_point_interaction(pt, include_interaction=True)
        m = buf.cpu().numpy().astype(bool)
        n_mask, n_slab = int(m.sum()), int(len(idx))
        inside = float((m & envelope).sum() / max(n_mask, 1))
        ratio = n_mask / max(n_slab, 1)
        rec = {"prompt_voxel": pt, "mask_voxels": n_mask, "slab_voxels": n_slab,
               "volume_ratio": round(ratio, 3), "inside_envelope": round(inside, 3)}
        if n_mask == 0:
            rec["status"] = "rejected: empty"
        elif not (PROMPT_MIN_RATIO <= ratio <= PROMPT_MAX_RATIO):
            rec["status"] = "rejected: volume ratio"
        elif inside < PROMPT_MIN_INSIDE:
            rec["status"] = "rejected: outside envelope"
        else:
            rec["status"] = "accepted"
            masks[k] = m & envelope
            centres[k] = float(axis_pos_of_voxel(idx).mean())
        record[k] = rec

    if not masks:
        return out, record

    # Clear the slab regions of accepted levels, then paint the prompted masks.
    # Overlaps between two accepted masks go to the nearer prompt along the
    # axis. A voxel a prompted mask takes from a level that was not prompted
    # (an anchor, or a rejected level) is taken: that is the point of the
    # exercise, the process belonged to the prompted vertebra all along.
    accepted = sorted(masks)
    for k in accepted:
        out[out == k] = 0
    claimed = np.zeros(seg.shape, dtype=np.uint8)
    owner_pos = np.full(seg.shape, np.nan, dtype=np.float32)
    for k in accepted:
        m = masks[k]
        idx = np.argwhere(m)
        s = axis_pos_of_voxel(idx).astype(np.float32)
        cur_owner = claimed[idx[:, 0], idx[:, 1], idx[:, 2]]
        cur_pos = owner_pos[idx[:, 0], idx[:, 1], idx[:, 2]]
        take = (cur_owner == 0) | (np.abs(s - centres[k]) < np.abs(s - cur_pos))
        sel = idx[take]
        claimed[sel[:, 0], sel[:, 1], sel[:, 2]] = k
        owner_pos[sel[:, 0], sel[:, 1], sel[:, 2]] = centres[k]
    paint = claimed > 0
    out[paint] = claimed[paint]

    # Anything of a prompted level's old slab that no mask covered keeps its
    # slab label, so no bone the network predicted is deleted.
    for k in accepted:
        lost = (seg == k) & (out == 0)
        out[lost] = k

    record["_summary"] = {"accepted": [int(k) for k in accepted],
                          "rejected": [int(k) for k in levels if k in record and record[k]["status"] != "accepted"],
                          "seconds": round(time.time() - t0, 1)}
    return out, record
