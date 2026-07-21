"""
Atlas-Based Registration Solution for Vertebrae Label Refinement

This module provides a sophisticated atlas-based registration pipeline for vertebrae
label refinement. Unlike simple morphological post-processing, this approach uses
deformable image registration to align input vertebrae to a reference atlas, enabling
intelligent label refinement based on spatial and anatomical consistency.

Pipeline Stages:
  1. Cleanup: Remove small fragments and pool noise
  2. Registration: ANTsPy-based deformable registration (rigid → affine → SyN)
     to reference atlas for spatial alignment
  3. Relabel: Assign components based on atlas overlap and spatial distance
  4. Finalize: Consolidate to target vertebrae count via intelligent merging/splitting

Key Innovation:
  - Registration-based approach enables atlas-guided label refinement
  - Generic technique applicable to any anatomical structure with atlas
  - Currently implemented for vertebrae (C1-L5, 24 labels)
  - Could extend to other organs (organs, brain structures, etc.)

Architecture:
  - Configuration-driven with YAML support
  - Supports flexible target counts (7, 12, 24, or custom)
  - Generates detailed CSV audit trails for each processing step
  - Integrates with ShapeKit's parallel batch processing
  - Parametric throughout for easy adaptation to other anatomies
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

try:
    import ants
except ImportError as exc:
    raise SystemExit("ANTsPy is required for vertebrae warmup mode. Install with: pip install antspyx") from exc


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupConfig:
    min_fragment_voxels: int = 64
    min_fragment_ratio: float = 0.01


@dataclass(frozen=True)
class RegistrationConfig:
    mask_dilate_mm: float = 6.0
    crop_margin_voxels: int = 24
    affine_mask_metric: str = "meansquares"
    enable_rigid_foreground_stage: bool = True
    label_weight: float = 1.0
    use_centroid_anchor: bool = True
    anchor_axis: str = "all"
    centroid_shift_clamp_voxels: int = 40
    reg_iterations: tuple[int, ...] = (30, 10, 0)
    grad_step: float = 0.05
    flow_sigma: float = 4.0
    total_sigma: float = 2.0


@dataclass(frozen=True)
class RelabelConfig:
    min_overlap_voxels: int = 25
    centroid_z_weight: float = 2.5
    overlap_weight: float = 0.15
    connectivity: int = 3


@dataclass
class Piece:
    id: int
    source_label: int
    mask: np.ndarray
    volume: int
    centroid_z: float


@dataclass(frozen=True)
class FinalizeConfig:
    min_fragment_voxels: int = 20
    split_size_ratio: float = 0.35
    split_distance_ratio: float = 1.5
    pair_small_ratio: float = 0.70
    pair_sum_min_ratio: float = 0.70
    pair_sum_max_ratio: float = 1.40
    nearby_spacing_ratio: float = 0.80
    max_edit_label: int = 12
    low_label_aggressiveness: float = 0.90
    high_label_distance_penalty: float = 1.60
    target_labels: int = 24
    connectivity: int = 3


def structure_for(connectivity: int) -> np.ndarray:
    if connectivity == 1:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 2:
        return ndimage.generate_binary_structure(3, 2)
    return ndimage.generate_binary_structure(3, 3)


def cleanup_labels(seg: np.ndarray, config: CleanupConfig, max_label: int) -> np.ndarray:
    """Remove small fragments and pool noise-like components."""
    out = np.zeros_like(seg, dtype=np.int16)
    pooled = np.zeros_like(seg, dtype=bool)
    st = structure_for(3)

    centroids: dict[int, float] = {}

    for label_id in range(1, max_label + 1):
        mask = seg == label_id
        if not np.any(mask):
            continue
        cc, n_cc = ndimage.label(mask, structure=st)
        sizes = np.bincount(cc.ravel())[1:]
        if sizes.size == 0:
            continue

        main_local = int(np.argmax(sizes)) + 1
        main_size = int(sizes[main_local - 1])
        main_mask = cc == main_local
        out[main_mask] = label_id

        main_coords = np.argwhere(main_mask)
        if main_coords.size:
            centroids[label_id] = float(main_coords[:, 2].mean())

        size_thr = max(config.min_fragment_voxels, int(round(config.min_fragment_ratio * main_size)))
        for local_id in range(1, n_cc + 1):
            if local_id == main_local:
                continue
            frag = cc == local_id
            frag_size = int(np.count_nonzero(frag))
            if frag_size >= size_thr:
                out[frag] = label_id
            else:
                pooled |= frag

    pooled_coords = np.argwhere(pooled)
    if pooled_coords.size and centroids:
        label_ids = np.asarray(sorted(centroids.keys()), dtype=np.int16)
        zc = np.asarray([centroids[int(k)] for k in label_ids], dtype=np.float64)
        for xyz in pooled_coords:
            z = float(xyz[2])
            idx = int(np.argmin(np.abs(zc - z)))
            out[tuple(xyz)] = int(label_ids[idx])

    out[seg <= 0] = 0
    return out


def foreground_mask(label_map: np.ndarray, dilate_mm: float) -> np.ndarray:
    mask = label_map > 0
    ants_mask = ants.from_numpy(mask.astype(np.float32))
    if dilate_mm > 0:
        mask = ants.iMath(ants_mask, "MD", int(round(dilate_mm))).numpy() > 0
    return mask


def mask_to_ants(mask: np.ndarray, reference: ants.ANTsImage) -> ants.ANTsImage:
    return ants.from_numpy(
        mask.astype(np.float32),
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def centroid(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise RuntimeError("Cannot compute centroid of empty mask")
    return pts.mean(axis=0)


def clamp_shift(shift_xyz: np.ndarray, shape: tuple[int, int, int], clamp_voxels: int) -> tuple[int, int, int]:
    out: list[int] = []
    for axis in range(3):
        value = int(round(float(shift_xyz[axis])))
        low = -(shape[axis] - 1)
        high = shape[axis] - 1
        value = max(low, min(high, value))
        value = max(-clamp_voxels, min(clamp_voxels, value))
        out.append(value)
    return out[0], out[1], out[2]


def integer_shift_3d(data: np.ndarray, shift_xyz: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(data)
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for axis, shift in enumerate(shift_xyz):
        size = data.shape[axis]
        if shift >= 0:
            src_start, src_end = 0, size - shift
            dst_start, dst_end = shift, size
        else:
            src_start, src_end = -shift, size
            dst_start, dst_end = 0, size + shift
        if src_end <= src_start or dst_end <= dst_start:
            return out
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = data[tuple(src_slices)]
    return out


def crop_to_mask(image: ants.ANTsImage, mask: ants.ANTsImage, margin_voxels: int) -> ants.ANTsImage:
    coords = np.argwhere(mask.numpy() > 0)
    if coords.size == 0:
        return image
    shape = np.array(mask.numpy().shape, dtype=np.int64)
    lower = np.maximum(coords.min(axis=0) - margin_voxels, 0)
    upper = np.minimum(coords.max(axis=0) + margin_voxels + 1, shape)
    return ants.crop_indices(image, lower.tolist(), upper.tolist())


def run_registration(
    fixed_ct_path: Path,
    moving_ct_path: Path,
    fixed_labels: np.ndarray,
    moving_labels: np.ndarray,
    output_dir: Path,
    config: RegistrationConfig,
    verbose: bool,
) -> tuple[np.ndarray, list[str], tuple[int, int, int]]:
    """Register moving labels to fixed atlas space."""
    fixed_ct = ants.image_read(str(fixed_ct_path))
    moving_ct = ants.image_read(str(moving_ct_path))

    fixed_fg_np = foreground_mask(fixed_labels, config.mask_dilate_mm)
    moving_fg_np = foreground_mask(moving_labels, config.mask_dilate_mm)

    moving_ct_np = np.asanyarray(nib.load(str(moving_ct_path)).dataobj)

    shift_xyz = (0, 0, 0)
    moving_ct_anchor = moving_ct
    moving_labels_anchor = moving_labels.copy()
    if config.use_centroid_anchor:
        delta = centroid(fixed_fg_np) - centroid(moving_fg_np)
        if config.anchor_axis == "z":
            delta = np.asarray([0.0, 0.0, delta[2]], dtype=np.float64)
        shift_xyz = clamp_shift(delta, moving_labels.shape, config.centroid_shift_clamp_voxels)
        moving_ct_anchor_np = integer_shift_3d(moving_ct_np, shift_xyz)
        moving_labels_anchor = integer_shift_3d(moving_labels, shift_xyz)
        moving_ct_anchor = ants.from_numpy(
            moving_ct_anchor_np.astype(np.float32),
            origin=moving_ct.origin,
            spacing=moving_ct.spacing,
            direction=moving_ct.direction,
        )

    fixed_fg = mask_to_ants(fixed_fg_np, fixed_ct)
    moving_fg = mask_to_ants(moving_labels_anchor > 0, moving_ct)

    fixed_ct_reg = crop_to_mask(fixed_ct, fixed_fg, config.crop_margin_voxels)
    moving_ct_reg = crop_to_mask(moving_ct_anchor, moving_fg, config.crop_margin_voxels)
    fixed_fg_reg = crop_to_mask(fixed_fg, fixed_fg, config.crop_margin_voxels)
    moving_fg_reg = crop_to_mask(moving_fg, moving_fg, config.crop_margin_voxels)

    rigid_fwd: list[str] = []
    if config.enable_rigid_foreground_stage:
        logger.info("registration stage 1/4: rigid foreground")
        rigid = ants.registration(
            fixed=fixed_fg_reg,
            moving=moving_fg_reg,
            type_of_transform="Rigid",
            aff_metric=config.affine_mask_metric,
            mask=fixed_fg_reg,
            moving_mask=moving_fg_reg,
            mask_all_stages=True,
            outprefix=str(output_dir / "rigid_fg_"),
            verbose=verbose,
        )
        rigid_fwd = list(rigid["fwdtransforms"])

    logger.info("registration stage 2/4: affine foreground")
    affine = ants.registration(
        fixed=fixed_fg_reg,
        moving=moving_fg_reg,
        type_of_transform="Affine",
        initial_transform=rigid_fwd if rigid_fwd else None,
        aff_metric=config.affine_mask_metric,
        mask=fixed_fg_reg,
        moving_mask=moving_fg_reg,
        mask_all_stages=True,
        outprefix=str(output_dir / "affine_"),
        verbose=verbose,
    )

    logger.info("registration stage 3/4: SyN refinement")
    refine = ants.registration(
        fixed=fixed_ct_reg,
        moving=moving_ct_reg,
        type_of_transform="SyNOnly",
        initial_transform=affine["fwdtransforms"],
        mask=fixed_fg_reg,
        moving_mask=moving_fg_reg,
        mask_all_stages=True,
        syn_metric="meansquares",
        grad_step=config.grad_step,
        flow_sigma=config.flow_sigma,
        total_sigma=config.total_sigma,
        reg_iterations=config.reg_iterations,
        outprefix=str(output_dir / "refine_"),
        multivariate_extras=[("MeanSquares", fixed_fg_reg.clone("float"), moving_fg_reg.clone("float"), config.label_weight, 0)],
        verbose=verbose,
    )
    forward = list(refine["fwdtransforms"])

    logger.info("registration stage 4/4: warp labels")
    moving_labels_ants = ants.from_numpy(
        moving_labels.astype(np.float32),
        origin=moving_ct.origin,
        spacing=moving_ct.spacing,
        direction=moving_ct.direction,
    )
    warped_labels = ants.apply_transforms(
        fixed=fixed_ct,
        moving=moving_labels_ants,
        transformlist=forward,
        interpolator="genericLabel",
    )
    warped_labels_np = np.rint(np.asanyarray(warped_labels.numpy())).astype(np.int16)

    return warped_labels_np, forward, shift_xyz


def compute_label_centroids(label_map: np.ndarray, max_label: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for label_id in range(1, max_label + 1):
        coords = np.argwhere(label_map == label_id)
        if coords.size:
            out[label_id] = coords.mean(axis=0)
    return out


def fill_missing_centroids(centroids: dict[int, np.ndarray], num_labels: int) -> dict[int, np.ndarray]:
    known = sorted(centroids.items())
    if not known:
        return {label_id: np.asarray([0.0, 0.0, float(label_id)], dtype=np.float64) for label_id in range(1, num_labels + 1)}

    out: dict[int, np.ndarray] = {}
    for label_id in range(1, num_labels + 1):
        if label_id in centroids:
            out[label_id] = np.asarray(centroids[label_id], dtype=np.float64)
            continue
        left = [x for x in known if x[0] < label_id]
        right = [x for x in known if x[0] > label_id]
        if left and right:
            l_id, l_val = left[-1]
            r_id, r_val = right[0]
            t = (label_id - l_id) / max(1, (r_id - l_id))
            out[label_id] = (1.0 - t) * np.asarray(l_val, dtype=np.float64) + t * np.asarray(r_val, dtype=np.float64)
        elif left:
            out[label_id] = np.asarray(left[-1][1], dtype=np.float64)
        else:
            out[label_id] = np.asarray(right[0][1], dtype=np.float64)
    return out


def weighted_centroid_distance(a: np.ndarray, b: np.ndarray, z_weight: float) -> float:
    d = a - b
    return float(np.linalg.norm(d * np.asarray([1.0, 1.0, z_weight], dtype=np.float64)))


def build_parent_assignment(
    case31_labels: np.ndarray, atlas_labels: np.ndarray, cfg: RelabelConfig, num_labels: int
) -> tuple[dict[int, int], list[str]]:
    c31 = fill_missing_centroids(compute_label_centroids(case31_labels, num_labels), num_labels)
    atl = fill_missing_centroids(compute_label_centroids(atlas_labels, num_labels), num_labels)

    ids = list(range(1, num_labels + 1))
    cost = np.zeros((num_labels, num_labels), dtype=np.float64)
    for r, lbl31 in enumerate(ids):
        mask31 = case31_labels == lbl31
        ctr31 = c31[lbl31]
        for c, lbl_a in enumerate(ids):
            maska = atlas_labels == lbl_a
            ctra = atl[lbl_a]
            dist = weighted_centroid_distance(ctr31, ctra, cfg.centroid_z_weight)
            overlap = int(np.count_nonzero(mask31 & maska))
            cost[r, c] = dist - cfg.overlap_weight * overlap

    ri, ci = linear_sum_assignment(cost)
    mapping = {int(ids[r]): int(ids[c]) for r, c in zip(ri.tolist(), ci.tolist())}

    rows = ["case31_label,assigned_label,cost"]
    for i in ids:
        j = mapping[i]
        rows.append(f"{i},{j},{cost[i-1, j-1]:.6f}")
    return mapping, rows


def split_connected_components_per_label(labels: np.ndarray, connectivity: int, max_label: int) -> tuple[np.ndarray, dict[int, int]]:
    st = structure_for(connectivity)
    component_labels = np.zeros_like(labels, dtype=np.int32)
    parent: dict[int, int] = {}
    next_id = 1

    for label_id in range(1, max_label + 1):
        mask = labels == label_id
        if not np.any(mask):
            continue
        cc, n_cc = ndimage.label(mask, structure=st)
        for local_id in range(1, n_cc + 1):
            m = cc == local_id
            if not np.any(m):
                continue
            component_labels[m] = next_id
            parent[next_id] = label_id
            next_id += 1

    return component_labels, parent


def assign_components_strict(
    component_labels: np.ndarray,
    parent_by_component: dict[int, int],
    atlas_labels: np.ndarray,
    parent_assignment: dict[int, int],
    cfg: RelabelConfig,
) -> tuple[np.ndarray, list[str]]:
    out = np.zeros_like(component_labels, dtype=np.int16)
    rows = ["component_id,parent_label,assigned_label,assignment_method,size,atlas_overlap"]

    ids = np.unique(component_labels)
    ids = ids[ids > 0]
    comp_slices = ndimage.find_objects(component_labels)

    for comp_id in ids.tolist():
        sl = comp_slices[comp_id - 1]
        if sl is None:
            continue
        local = component_labels[sl]
        m = local == comp_id
        size = int(np.count_nonzero(m))
        parent_label = int(parent_by_component.get(comp_id, 0))
        fallback = int(parent_assignment.get(parent_label, 0))

        atlas_local = atlas_labels[sl]
        overlap_labels = atlas_local[m]
        overlap_labels = overlap_labels[overlap_labels > 0]

        assigned = fallback
        method = "parent_fallback"
        overlap = 0
        if overlap_labels.size > 0:
            u, c = np.unique(overlap_labels, return_counts=True)
            idx = int(np.argmax(c))
            best_label = int(u[idx])
            best_count = int(c[idx])
            if best_count >= cfg.min_overlap_voxels:
                assigned = best_label
                method = "component_overlap"
                overlap = best_count

        out_local = out[sl]
        out_local[m] = assigned
        out[sl] = out_local
        rows.append(f"{comp_id},{parent_label},{assigned},{method},{size},{overlap}")

    return out, rows


def median_spacing_from_labels(labels: np.ndarray, max_label: int) -> float:
    zs = []
    for label_id in sorted(int(v) for v in np.unique(labels) if v > 0 and v <= max_label):
        coords = np.argwhere(labels == label_id)
        if coords.size:
            zs.append(float(coords[:, 2].mean()))
    if len(zs) < 2:
        return 1.0
    zs.sort()
    spacing = [abs(zs[i + 1] - zs[i]) for i in range(len(zs) - 1)]
    return float(np.median(np.asarray(spacing, dtype=np.float64))) if spacing else 1.0


def extract_pieces_with_split(labels: np.ndarray, cfg: FinalizeConfig, max_label: int) -> tuple[list[Piece], list[str], np.ndarray]:
    rows = ["source_label,component_local_id,volume,centroid_z,action"]
    pool = np.zeros_like(labels, dtype=bool)
    pieces: list[Piece] = []
    st = structure_for(cfg.connectivity)
    med_space = median_spacing_from_labels(labels, max_label)
    pid = 0

    for label_id in sorted(int(v) for v in np.unique(labels) if v > 0 and v <= max_label):
        mask = labels == label_id
        cc, n_cc = ndimage.label(mask, structure=st)
        if n_cc == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        if sizes.size == 0:
            continue

        main_id = int(np.argmax(sizes)) + 1
        main_size = int(sizes[main_id - 1])
        main_coords = np.argwhere(cc == main_id)
        main_z = float(main_coords[:, 2].mean()) if main_coords.size else 0.0

        for comp_local in range(1, n_cc + 1):
            cm = cc == comp_local
            csize = int(np.count_nonzero(cm))
            if csize == 0:
                continue
            cz = float(np.argwhere(cm)[:, 2].mean())

            if comp_local == main_id:
                pid += 1
                pieces.append(Piece(pid, label_id, cm.copy(), csize, cz))
                rows.append(f"{label_id},{comp_local},{csize},{cz:.3f},main")
                continue

            is_big = csize >= int(cfg.split_size_ratio * main_size)
            far = abs(cz - main_z) >= (cfg.split_distance_ratio * med_space)
            if is_big and far:
                pid += 1
                pieces.append(Piece(pid, label_id, cm.copy(), csize, cz))
                rows.append(f"{label_id},{comp_local},{csize},{cz:.3f},split_new_piece")
            else:
                pool |= cm
                rows.append(f"{label_id},{comp_local},{csize},{cz:.3f},fragment_pool")

    return pieces, rows, pool


def update_piece_stats(pieces: list[Piece]) -> None:
    for p in pieces:
        coords = np.argwhere(p.mask)
        p.volume = int(coords.shape[0])
        p.centroid_z = float(coords[:, 2].mean()) if coords.size else -1.0


def assign_pool_to_nearest_piece(pieces: list[Piece], pool_mask: np.ndarray) -> list[str]:
    rows = ["reassigned_voxels"]
    coords = np.argwhere(pool_mask)
    if coords.size == 0 or not pieces:
        rows.append("0")
        return rows

    pz = np.asarray([p.centroid_z for p in pieces], dtype=np.float64)
    ps = np.asarray([p.source_label for p in pieces], dtype=np.float64)
    span = float(max(np.max(ps) - 1.0, 1.0))
    for z in np.unique(coords[:, 2]):
        dz = np.abs(pz - float(z))
        penalty = 1.0 + 0.20 * ((ps - 1.0) / span)
        idx = int(np.argmin(dz * penalty))
        pieces[idx].mask[:, :, int(z)] |= pool_mask[:, :, int(z)]

    rows.append(str(int(coords.shape[0])))
    update_piece_stats(pieces)
    return rows


def merge_small_pairs(pieces: list[Piece], cfg: FinalizeConfig) -> tuple[list[Piece], list[str]]:
    rows = ["piece_a,piece_b,src_a,src_b,vol_a,vol_b,sum,delta_z,action"]
    if len(pieces) < 2:
        return pieces, rows

    while True:
        update_piece_stats(pieces)
        pieces = [p for p in pieces if p.volume > 0]
        pieces.sort(key=lambda p: p.centroid_z)
        if len(pieces) < 2 or len(pieces) <= cfg.target_labels:
            break

        vols = np.asarray([p.volume for p in pieces], dtype=np.float64)
        src = np.asarray([p.source_label for p in pieces], dtype=np.float64)
        w = (cfg.target_labels + 1.0 - src) ** (1.0 + cfg.low_label_aggressiveness)
        ref_vol = float(np.average(vols, weights=w)) if vols.size else 1.0
        spacings = np.asarray([abs(pieces[i + 1].centroid_z - pieces[i].centroid_z) for i in range(len(pieces) - 1)], dtype=np.float64)
        med_spacing = float(np.median(spacings)) if spacings.size else 1.0

        best_i = None
        best_score = None
        for i in range(len(pieces) - 1):
            a = pieces[i]
            b = pieces[i + 1]
            if a.source_label > cfg.max_edit_label or b.source_label > cfg.max_edit_label:
                continue

            combined = a.volume + b.volume
            dz = abs(b.centroid_z - a.centroid_z)
            scale_a = 1.0 + cfg.low_label_aggressiveness * ((cfg.target_labels - a.source_label) / max(cfg.target_labels - 1, 1))
            scale_b = 1.0 + cfg.low_label_aggressiveness * ((cfg.target_labels - b.source_label) / max(cfg.target_labels - 1, 1))

            cond_small = (a.volume <= cfg.pair_small_ratio * scale_a * ref_vol) and (b.volume <= cfg.pair_small_ratio * scale_b * ref_vol)
            cond_sum = cfg.pair_sum_min_ratio * ref_vol <= combined <= cfg.pair_sum_max_ratio * ref_vol
            cond_near = dz <= cfg.nearby_spacing_ratio * med_spacing
            if cond_small and cond_sum and cond_near:
                hi_penalty = 1.0 + cfg.high_label_distance_penalty * (max(a.source_label, b.source_label) / max(cfg.target_labels, 1))
                score = (dz * hi_penalty, abs(combined - ref_vol), max(a.source_label, b.source_label))
                if best_score is None or score < best_score:
                    best_score = score
                    best_i = i

        if best_i is None:
            break

        a = pieces[best_i]
        b = pieces[best_i + 1]
        target = a if a.volume >= b.volume else b
        source = b if target is a else a
        target.mask |= source.mask
        target.source_label = min(target.source_label, source.source_label)
        source.mask[:] = False
        rows.append(f"{a.id},{b.id},{a.source_label},{b.source_label},{a.volume},{b.volume},{a.volume+b.volume},{abs(b.centroid_z-a.centroid_z):.3f},merged")

    pieces = [p for p in pieces if np.any(p.mask)]
    update_piece_stats(pieces)
    pieces.sort(key=lambda p: p.centroid_z)
    return pieces, rows


def fix_false_positive_splits(pieces: list[Piece], cfg: FinalizeConfig) -> tuple[list[Piece], list[str]]:
    rows = ["piece_id,src_label,fragment_voxels,target_piece_id,target_src_label"]
    st = structure_for(cfg.connectivity)

    for p in pieces:
        if p.source_label > cfg.max_edit_label:
            continue

        cc, n_cc = ndimage.label(p.mask, structure=st)
        if n_cc <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        if sizes.size == 0:
            continue

        main_id = int(np.argmax(sizes)) + 1
        for local_id in range(1, n_cc + 1):
            if local_id == main_id:
                continue
            frag = cc == local_id
            frag_size = int(np.count_nonzero(frag))
            if frag_size < cfg.min_fragment_voxels:
                continue

            frag_z = float(np.argwhere(frag)[:, 2].mean())
            other = [q for q in pieces if q.id != p.id and q.volume > 0]
            if not other:
                continue

            target = min(
                other,
                key=lambda q: abs(q.centroid_z - frag_z)
                * (1.0 + cfg.high_label_distance_penalty * (q.source_label / max(cfg.target_labels, 1))),
            )
            p.mask[frag] = False
            target.mask[frag] = True
            rows.append(f"{p.id},{p.source_label},{frag_size},{target.id},{target.source_label}")

    pieces = [p for p in pieces if np.any(p.mask)]
    update_piece_stats(pieces)
    pieces.sort(key=lambda p: p.centroid_z)
    return pieces, rows


def enforce_target_count(pieces: list[Piece], cfg: FinalizeConfig) -> tuple[list[Piece], list[str]]:
    rows = ["action,source_piece,target_piece,source_label,target_label,source_volume,target_volume"]

    while len(pieces) > cfg.target_labels:
        pieces.sort(key=lambda p: p.centroid_z)
        best_i = None
        best_score = None
        for i in range(len(pieces) - 1):
            a = pieces[i]
            b = pieces[i + 1]
            if a.source_label > cfg.max_edit_label or b.source_label > cfg.max_edit_label:
                continue
            high_penalty = 1.0 + cfg.high_label_distance_penalty * (max(a.source_label, b.source_label) / max(cfg.target_labels, 1))
            score = (abs(b.centroid_z - a.centroid_z) * high_penalty, a.volume + b.volume, max(a.source_label, b.source_label))
            if best_score is None or score < best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        a = pieces[best_i]
        b = pieces[best_i + 1]
        target = a if a.volume >= b.volume else b
        source = b if target is a else a
        target.mask |= source.mask
        target.source_label = min(target.source_label, source.source_label)
        rows.append(
            f"merge_to_target_count,{source.id},{target.id},{source.source_label},{target.source_label},{source.volume},{target.volume}"
        )
        pieces = [p for p in pieces if p.id != source.id]
        update_piece_stats(pieces)

    pieces = [p for p in pieces if np.any(p.mask)]
    update_piece_stats(pieces)
    pieces.sort(key=lambda p: p.centroid_z)
    return pieces, rows


def pieces_to_label_map(shape: tuple[int, int, int], pieces: list[Piece]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int16)
    pieces.sort(key=lambda p: p.centroid_z)
    for idx, p in enumerate(pieces, start=1):
        out[p.mask] = int(idx)
    return out


def run_finalize(input_labels: np.ndarray, cfg: FinalizeConfig, out_dir: Path, max_label: int) -> tuple[np.ndarray, list[str], list[str], list[str], list[str], list[str]]:
    pieces, split_rows, pool = extract_pieces_with_split(input_labels, cfg, max_label)
    pool_rows = assign_pool_to_nearest_piece(pieces, pool)
    pieces, merge_rows = merge_small_pairs(pieces, cfg)
    pieces, fp_rows = fix_false_positive_splits(pieces, cfg)
    pieces, enforce_rows = enforce_target_count(pieces, cfg)
    final_labels = pieces_to_label_map(input_labels.shape, pieces)
    return final_labels, split_rows, pool_rows, merge_rows, fp_rows, enforce_rows


def process_vertebrae_warmup(
    input_labels: np.ndarray,
    fixed_ct_path: Path,
    fixed_labels_path: Path,
    moving_ct_path: Path,
    output_dir: Path,
    target_vertebrae_count: int = 24,
    cleanup_cfg: Optional[CleanupConfig] = None,
    registration_cfg: Optional[RegistrationConfig] = None,
    relabel_cfg: Optional[RelabelConfig] = None,
    finalize_cfg: Optional[FinalizeConfig] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """
    Main orchestrator for vertebrae warmup postprocessing pipeline.

    Args:
        input_labels: Input segmentation (labels 1 to target_vertebrae_count)
        fixed_ct_path: Path to fixed (atlas) CT image
        fixed_labels_path: Path to fixed (atlas) labels
        moving_ct_path: Path to moving (input) CT image
        output_dir: Output directory for intermediate files and CSVs
        target_vertebrae_count: Expected vertebrae count (7, 12, 24, etc.)
        cleanup_cfg: Cleanup configuration
        registration_cfg: Registration configuration
        relabel_cfg: Relabel configuration
        finalize_cfg: Finalize configuration
        verbose: Verbose ANTsPy logging

    Returns:
        final_labels: Processed labels (1 to target_vertebrae_count)
        audit_trails: Dictionary of CSV audit trails (split, pool, merge, etc.)
    """
    if cleanup_cfg is None:
        cleanup_cfg = CleanupConfig()
    if registration_cfg is None:
        registration_cfg = RegistrationConfig()
    if relabel_cfg is None:
        relabel_cfg = RelabelConfig()
    if finalize_cfg is None:
        finalize_cfg = FinalizeConfig(target_labels=target_vertebrae_count)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Cleanup
    logger.info("Step 1/4: Cleanup labels")
    fixed_labels = nib.load(str(fixed_labels_path))
    fixed_labels_data = np.asanyarray(fixed_labels.dataobj).astype(np.int16)
    moving_labels_clean = cleanup_labels(input_labels, cleanup_cfg, target_vertebrae_count)

    # Step 2: Registration
    logger.info("Step 2/4: Registration")
    warped_labels, forward_transforms, shift_xyz = run_registration(
        fixed_ct_path=Path(fixed_ct_path),
        moving_ct_path=Path(moving_ct_path),
        fixed_labels=fixed_labels_data,
        moving_labels=moving_labels_clean,
        output_dir=output_dir,
        config=registration_cfg,
        verbose=verbose,
    )

    # Step 3: Relabel (split-aware strict)
    logger.info("Step 3/4: Split-aware strict relabel")
    comp_labels, parent_map = split_connected_components_per_label(
        fixed_labels_data, relabel_cfg.connectivity, target_vertebrae_count
    )
    parent_assignment, parent_rows = build_parent_assignment(
        fixed_labels_data, warped_labels, relabel_cfg, target_vertebrae_count
    )
    relabeled, comp_rows = assign_components_strict(comp_labels, parent_map, warped_labels, parent_assignment, relabel_cfg)
    relabeled[fixed_labels_data <= 0] = 0

    # Step 4: Finalize
    logger.info("Step 4/4: Finalize label set")
    final_labels, split_rows, pool_rows, merge_rows, fp_rows, enforce_rows = run_finalize(
        relabeled, finalize_cfg, output_dir, target_vertebrae_count
    )
    final_labels[relabeled <= 0] = 0

    # Check vertebrae count and warn if mismatch
    present_labels = sorted(int(v) for v in np.unique(final_labels) if v > 0)
    if len(present_labels) != target_vertebrae_count:
        logger.warning(
            f"Configured for {target_vertebrae_count} vertebrae but output contains {len(present_labels)} labels. "
            f"Proceeding with available labels."
        )

    # Return audit trails
    audit_trails = {
        "split_components": split_rows,
        "pool_assignment": pool_rows,
        "volume_merge_pairs": merge_rows,
        "false_positive_repairs": fp_rows,
        "enforce_target_count": enforce_rows,
        "parent_label_assignment": parent_rows,
        "component_assignments": comp_rows,
    }

    return final_labels, audit_trails
