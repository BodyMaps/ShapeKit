"""
Axis-based vertebra relabelling (the `shapekit_axis_relabel` engine).

Segmentation models often mix up neighbouring vertebrae: part of a body gets the label of the
one above, or an arch ends up with the wrong number. This engine re-derives the vertebrae from
the CT and the shape of the mask, using the predicted labels only as a hint. It uses classic
image processing only (no learned model).

Labels inside this module are 1 (L5) ... 24 (C1); the adapter in vertebrae_axis_relabel.py
maps them to and from ShapeKit's class_map.

How it works, in short:
  1. Remove small specks that are far away from the spine.
  2. Find the spine axis. In every axial slice the vertebral body is the biggest solid
     blob of the mask, so we track its centre from slice to slice.
  3. Find the discs along that axis (dark gaps in the CT, gaps in the mask, places where
     the predicted label changes) and split the column into vertebral levels.
  4. Give every vertebral body the label of its level. Then work out which vertebra each
     piece of posterior bone (pedicles, laminae, spinous and transverse processes) belongs
     to, starting from the pedicles and growing outwards through the bone.
  5. A few safety checks: C1 (which has no body) and anything above/below the detected axis
     keep the predicted labels, and pieces that ended up cut off from their vertebra are
     given to the vertebra they are attached to.

Entry point: relabel_volume(labels, ct, affine, zooms) for a whole scan in any orientation.
"""
import math

import numpy as np
from nibabel import orientations as nio
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

# ----------------------------------------------------------------------------
# Settings (all distances in mm)
# ----------------------------------------------------------------------------
GRID_MM = 1.0             # isotropic grid used for finding the axis and the levels
CROP_MARGIN_MM = 20.0     # margin kept around the mask when cropping the CT
TUBE_RADIUS_MM = 70.0     # only voxels this close to the axis are used for the level profile
BIN_MM = 1.0              # bin size of the profile along the axis
DISC_GAP_MM = 14.0        # gaps along the spine up to this length are bridged (discs)
SMALL_HOLE_MM2 = 60.0     # holes up to this area are filled (never the spinal canal)

MIN_LEVEL_MM = 12.0       # shortest normal vertebra
MIN_END_LEVEL_MM = 5.0    # the first/last level may be cut off by the scan
MAX_LEVEL_MM = 48.0
MAX_EXPECTED_MM = 55.0
DISC_THRESHOLD = 0.6      # how much disc evidence is needed to place a boundary
MAX_LEVELS = 24

FACE_NEIGHBOURS = ndi.generate_binary_structure(3, 1)   # 6-connectivity
ALL_NEIGHBOURS = ndi.generate_binary_structure(3, 3)    # 26-connectivity


KDTREE_WORKERS = 1        # ShapeKit already runs one process per case; avoid nested threads

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def shift(values, k):
    """values shifted by k positions, repeating the edge value"""
    idx = np.clip(np.arange(len(values)) - k, 0, len(values) - 1)
    return values[idx]


def disk(radius):
    n = int(np.ceil(radius))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (yy ** 2 + xx ** 2) <= radius ** 2


def ball(radius):
    grid = np.mgrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
    return (grid ** 2).sum(0) <= radius * radius


def odd_size(length_mm, spacing):
    """odd filter size (at least 3) covering roughly length_mm"""
    return max(3, int(round(length_mm / spacing)) | 1)


def fill_small_holes(slice_mask, pixel_area):
    """Fill the small holes SuPreM leaves inside bodies, but not big ones like the spinal canal."""
    holes = ndi.binary_fill_holes(slice_mask) & ~slice_mask
    hole_ids, n_holes = ndi.label(holes)
    if n_holes == 0:
        return slice_mask
    is_small = np.bincount(hole_ids.ravel()) * pixel_area <= SMALL_HOLE_MM2
    is_small[0] = False
    return slice_mask | is_small[hole_ids]


def bridge_discs(mask, z_spacing):
    """Close gaps along z so the vertebral column becomes one continuous column."""
    column = np.ones((1, 1, odd_size(DISC_GAP_MM, z_spacing)), bool)
    return ndi.binary_closing(mask, structure=column) | mask


# ----------------------------------------------------------------------------
# Step 1: remove specks far from the spine
# ----------------------------------------------------------------------------
def remove_distant_specks(seg, spacing):
    mask = seg > 0
    components, n = ndi.label(mask, structure=ALL_NEIGHBOURS)
    if n == 0:
        return seg.copy()
    volumes = np.bincount(components.ravel(), minlength=n + 1)[1:].astype(np.float64) * float(np.prod(spacing))

    # rough centre line of the spine from the large components only
    large = np.concatenate([[False], volumes >= 0.2 * volumes.max()])[components]
    n_slices = seg.shape[2]
    centre = np.full((n_slices, 2), np.nan)
    for z in range(n_slices):
        pts = np.argwhere(large[:, :, z])
        if pts.shape[0] >= 5:
            centre[z] = np.median(pts, axis=0) * spacing[:2] + 0.5 * spacing[:2]
    zs = np.arange(n_slices)
    known = ~np.isnan(centre[:, 0])
    centre = np.stack([np.interp(zs, zs[known], centre[known, k]) for k in (0, 1)], 1)
    centre = ndi.median_filter(centre, size=(odd_size(60.0, spacing[2]), 1), mode="nearest")

    # distance of every component to that centre line
    idx = np.nonzero(mask)
    dist = np.hypot((idx[0] + 0.5) * spacing[0] - centre[idx[2], 0],
                    (idx[1] + 0.5) * spacing[1] - centre[idx[2], 1])
    closest = np.asarray(ndi.minimum(dist, components[idx], index=np.arange(1, n + 1)))

    remove = ((volumes < 2000.0) & (closest > 60.0)) | ((volumes < 100.0) & (closest > 25.0))
    remove[int(np.argmax(volumes))] = False
    cleaned = seg.copy()
    cleaned[np.concatenate([[False], remove])[components]] = 0
    return cleaned


# ----------------------------------------------------------------------------
# Step 2: spine axis
# ----------------------------------------------------------------------------
def body_radius_map(mask):
    """For every axial slice: distance to the edge of the (gap-bridged, hole-filled) mask.
    The vertebral body is the thickest part of every slice, so this peaks in the body."""
    column = bridge_discs(mask, GRID_MM)
    radius = np.zeros(mask.shape, np.float32)
    for z in range(mask.shape[2]):
        if column[:, :, z].any():
            filled = fill_small_holes(column[:, :, z], GRID_MM * GRID_MM)
            radius[:, :, z] = ndi.distance_transform_edt(filled, sampling=GRID_MM)
    return radius


def candidate_centres(radius):
    """Up to 5 local maxima per slice, each moved to the centre of its flat top."""
    nx, ny, nz = radius.shape
    win = odd_size(8.0, GRID_MM)
    is_peak = (radius == ndi.maximum_filter(radius, size=(win, win, 1))) & (radius >= 4.0)
    half = max(2, int(round(15.0 / GRID_MM)))

    slices, candidates = [], []
    for z in range(nz):
        peaks = np.argwhere(is_peak[:, :, z])
        if len(peaks) == 0:
            continue
        heights = radius[peaks[:, 0], peaks[:, 1], z]
        order = np.argsort(-heights)
        chosen = []
        for p in order:
            if heights[p] < 0.5 * heights[order[0]]:
                break
            if all(np.hypot(*(peaks[p] - peaks[q])) * GRID_MM >= 6.0 for q in chosen):
                chosen.append(p)
            if len(chosen) >= 5:
                break

        centres = []
        for p in chosen:
            px, py = peaks[p]
            x0, x1 = max(0, px - half), min(nx, px + half + 1)
            y0, y1 = max(0, py - half), min(ny, py + half + 1)
            patch = radius[x0:x1, y0:y1, z]
            weight = np.where(patch >= 0.85 * heights[p], patch, 0.0)
            gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
            centres.append([(weight * gx).sum() / weight.sum(), (weight * gy).sum() / weight.sum()])
        slices.append(z)
        candidates.append((np.array(centres) * GRID_MM + GRID_MM / 2.0, heights[chosen].astype(float)))
    return slices, candidates


def best_path(slices, candidates):
    """Dynamic programming: pick one candidate per slice, preferring thick candidates and
    penalising sideways jumps."""
    n = len(slices)
    score = [candidates[0][1].copy()] + [None] * (n - 1)
    came_from = [None] * n
    for i in range(1, n):
        prev_xy, _ = candidates[i - 1]
        cur_xy, cur_height = candidates[i]
        jump = np.hypot(cur_xy[:, None, 0] - prev_xy[None, :, 0], cur_xy[:, None, 1] - prev_xy[None, :, 1])
        too_far = jump > 0.35 * (slices[i] - slices[i - 1]) * GRID_MM
        total = score[i - 1][None, :] - (1.5 * jump + 1e3 * too_far)
        best_prev = np.argmax(total, axis=1)
        score[i] = cur_height + total[np.arange(len(cur_height)), best_prev]
        came_from[i] = best_prev

    path = [int(np.argmax(score[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(came_from[i][path[-1]]))
    path = path[::-1]

    x = np.array([candidates[i][0][path[i], 0] for i in range(n)])
    y = np.array([candidates[i][0][path[i], 1] for i in range(n)])
    r = np.array([candidates[i][1][path[i]] for i in range(n)])
    return np.asarray(slices), x, y, r


def trim_thin_ends(z, x, y, r):
    """Drop axis ends that are much thinner than the vertebrae just inside them
    (e.g. a spinous process sticking out below the last body)."""
    r_smooth = ndi.median_filter(r, size=odd_size(10.0, GRID_MM), mode="nearest")
    z_mm = z * GRID_MM
    start, stop = 0, len(z)
    while start < stop - 5:
        inside = r_smooth[(z_mm >= z_mm[start] + 20) & (z_mm <= z_mm[start] + 80)]
        if inside.size < 5 or r_smooth[start] >= 0.6 * np.median(inside):
            break
        start += 1
    while stop - 1 > start + 5:
        inside = r_smooth[(z_mm <= z_mm[stop - 1] - 20) & (z_mm >= z_mm[stop - 1] - 80)]
        if inside.size < 5 or r_smooth[stop - 1] >= 0.6 * np.median(inside):
            break
        stop -= 1
    return z[start:stop], x[start:stop], y[start:stop], r[start:stop]


class SpineAxis:
    """Smoothed spine axis sampled every 0.5 mm along z, with a local coordinate frame."""

    def __init__(self, z, x, y, r):
        z_all = np.arange(z[0], z[-1] + 1)
        x_all, y_all, r_all = [np.interp(z_all, z, v) for v in (x, y, r)]
        self.z_mm = (z_all + 0.5) * GRID_MM
        z_fine = np.arange(self.z_mm[0], self.z_mm[-1] + 1e-6, 0.5)

        # smooth more and more until the axis is no longer wiggly (at most 5 % longer than the spine)
        for factor in (1.0, 2.5, 5.0, 10.0):
            size = odd_size(factor * 10.0, GRID_MM)
            sigma = max(0.5, factor * 10.0 / GRID_MM / 2.0)
            self.x, self.y, radius = [
                gaussian_filter1d(ndi.median_filter(v, size=size, mode="nearest"), sigma, mode="nearest")
                for v in (x_all, y_all, r_all)]
            points = np.stack([np.interp(z_fine, self.z_mm, self.x), np.interp(z_fine, self.z_mm, self.y), z_fine], 1)
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
            if arc[-1] / max(z_fine[-1] - z_fine[0], 1e-6) < 1.05:
                break

        tangent = np.gradient(points, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        lateral = np.array([1.0, 0, 0])[None, :] - tangent[:, 0:1] * tangent
        lateral /= np.linalg.norm(lateral, axis=1, keepdims=True) + 1e-9

        self.points = points                      # (n, 3) positions in mm
        self.arc = arc                            # distance along the axis
        self.tangent = tangent
        self.lateral = lateral
        self.anterior = np.cross(tangent, lateral)
        self.body_radius = np.maximum(np.interp(z_fine, self.z_mm, radius), 4.0)
        self.tree = cKDTree(points)

    def project(self, pts):
        """position along the axis, lateral and front-back offset, local body radius, offset along the axis"""
        _, i = self.tree.query(pts, workers=KDTREE_WORKERS)
        rel = pts - self.points[i]
        return (self.arc[i],
                np.einsum("ij,ij->i", rel, self.lateral[i]),
                np.einsum("ij,ij->i", rel, self.anterior[i]),
                self.body_radius[i],
                np.einsum("ij,ij->i", rel, self.tangent[i]))


def find_axis(mask):
    slices, candidates = candidate_centres(body_radius_map(mask))
    if len(slices) < 5:
        return None
    return SpineAxis(*trim_thin_ends(*best_path(slices, candidates)))


# ----------------------------------------------------------------------------
# Step 3: split the column into vertebral levels
# ----------------------------------------------------------------------------
def valley_depth(signal, half=4, gap=2):
    """how much lower each sample is than the maxima on both sides of it"""
    local_max = ndi.maximum_filter1d(signal, size=2 * half + 1, mode="nearest")
    return np.minimum(shift(local_max, gap + half), shift(local_max, -(gap + half))) - signal


def split_into_levels(evidence, threshold, expected_height):
    """Choose disc positions along the profile with dynamic programming.

    Each boundary earns its disc evidence minus `threshold`; levels are penalised for being
    far from the expected height and for being very different from their neighbour.
    expected_height (mm per bin) comes from SuPreM's label spacing; None = use 24 mm."""
    n = len(evidence)
    min_end = max(2, int(round(MIN_END_LEVEL_MM / BIN_MM)))
    min_len = int(round(MIN_LEVEL_MM / BIN_MM))
    longest = MAX_LEVEL_MM if expected_height is None else max(MAX_LEVEL_MM, 1.4 * float(np.max(expected_height)))
    max_len = int(round(longest / BIN_MM))
    if n - 1 <= max_len:
        return []

    lengths = np.arange(min_end, max_len + 1)
    n_len = len(lengths)
    NONE = -1e18
    best = np.full((n, n_len), NONE)               # best score of a level ending at bin b with length index k
    prev_len = np.full((n, n_len), -1, dtype=np.int64)
    height_penalty = -0.3 * ((lengths * BIN_MM - 24.0) / 10.0) ** 2 - 4.0 * (lengths < min_len)
    neighbour_penalty = -1.0 * (np.log(lengths[None, :] / lengths[:, None]) / 0.4) ** 2

    for k, length in enumerate(lengths):            # the first level
        if length < n - 1:
            best[length, k] = evidence[length] - threshold

    for b in range(n):
        if best[b].max() <= NONE / 2:
            continue
        gain = best[b][:, None] + np.where(((b - lengths) > 0)[:, None], neighbour_penalty, 0.0)
        if expected_height is not None:
            mid = expected_height[np.minimum(b + lengths // 2, n - 1)]
            height_penalty = -0.3 * ((lengths * BIN_MM - mid) / (0.35 * mid)) ** 2 - 4.0 * (lengths < min_len)
        best_k = np.argmax(gain, axis=0)
        best_gain = gain[best_k, np.arange(n_len)]
        for k in range(n_len):
            end = b + lengths[k]
            if end >= n - 1:
                break
            value = best_gain[k] + evidence[end] - threshold + height_penalty[k]
            if value > best[end, k]:
                best[end, k] = value
                prev_len[end, k] = best_k[k]

    # the last level runs to the end of the profile
    total, last_b, last_k = NONE, None, None
    for b in range(n):
        if min_end <= (n - 1) - b <= max_len:
            k = int(np.argmax(best[b]))
            if best[b, k] > total:
                total, last_b, last_k = best[b, k], b, k
    if last_b is None:
        return []

    boundaries, b, k = [], last_b, last_k
    while True:
        boundaries.append(b)
        start = b - lengths[k]
        if start <= 0:
            break
        k, b = int(prev_len[b, k]), int(start)
    return boundaries[::-1]


def find_levels(seg, ct, spacing):
    """Find the axis and the vertebral levels on a 1 mm grid.

    Returns (axis, level labels, label of every voxel's level, body-core seeds), with the last
    two at the original resolution, or None if the spine cannot be analysed."""
    # nearest-neighbour mapping between the original grid and the 1 mm grid
    grid_shape = tuple(int(math.ceil(n * s / GRID_MM)) for n, s in zip(seg.shape, spacing))
    to_grid = [np.clip(np.floor((np.arange(g) + 0.5) * GRID_MM / s).astype(np.int64), 0, n - 1)
               for g, s, n in zip(grid_shape, spacing, seg.shape)]
    to_orig = [np.clip(np.floor((np.arange(n) + 0.5) * s / GRID_MM).astype(np.int64), 0, g - 1)
               for g, s, n in zip(grid_shape, spacing, seg.shape)]

    def downsample(a):
        return a[np.ix_(*to_grid)]

    def upsample(a):
        return a[np.ix_(*to_orig)]

    labels = downsample(seg)
    mask = labels > 0
    if mask.sum() < 200:
        return None
    sigma = [0.4 * s if s > 1.3 else 0.0 for s in GRID_MM / np.asarray(spacing)]
    ct_smooth = ndi.gaussian_filter(ct, sigma) if any(s > 0 for s in sigma) else ct
    hu = ndi.gaussian_filter(downsample(ct_smooth).astype(np.float32), 0.6)
    mask_thickness = ndi.distance_transform_edt(mask, sampling=GRID_MM).astype(np.float32)

    axis = find_axis(mask)
    if axis is None:
        return None

    # --- describe every voxel within a tube around the axis in axis coordinates
    centres = [(np.arange(n) + 0.5) * GRID_MM for n in grid_shape]
    axis_x = np.interp(centres[2], axis.z_mm, axis.x).astype(np.float32)
    axis_y = np.interp(centres[2], axis.z_mm, axis.y).astype(np.float32)
    in_tube = ((centres[0][:, None, None].astype(np.float32) - axis_x[None, None, :]) ** 2 +
               (centres[1][None, :, None].astype(np.float32) - axis_y[None, None, :]) ** 2) <= TUBE_RADIUS_MM ** 2
    tube = np.flatnonzero(in_tube)
    del in_tube
    ijk = np.unravel_index(tube, grid_shape)
    along, side, front, body_r, off_axis = [np.empty(len(tube), np.float32) for _ in range(5)]
    for start in range(0, len(tube), 1_000_000):     # in chunks to limit memory
        part = slice(start, min(len(tube), start + 1_000_000))
        pts = np.stack([(ijk[k][part] + 0.5) * GRID_MM for k in range(3)], 1).astype(np.float64)
        along[part], side[part], front[part], body_r[part], off_axis[part] = axis.project(pts)
    del ijk

    # --- 1-D profiles along the axis, measured in the core of the vertebral bodies
    n_bins = int(np.ceil(float(axis.arc[-1]) / BIN_MM)) + 1
    tube_hu, tube_mask, tube_label = hu.ravel()[tube], mask.ravel()[tube], labels.ravel()[tube]
    core_r = 0.6 * body_r
    in_core = ((side / core_r) ** 2 + (front / core_r) ** 2 <= 1.0) & (np.abs(off_axis) < 1.5)
    core_bin = np.minimum((along[in_core] / BIN_MM).astype(np.int64), n_bins - 1)
    count = np.bincount(core_bin, minlength=n_bins).astype(np.float64)

    mean_hu = np.bincount(core_bin, weights=np.clip(tube_hu[in_core], -200, 700), minlength=n_bins) / np.maximum(count, 1)
    empty = count == 0
    if empty.all():
        mean_hu[:] = 0
    elif empty.any():
        mean_hu[empty] = np.interp(np.flatnonzero(empty), np.flatnonzero(~empty), mean_hu[~empty])
    mask_fraction = np.bincount(core_bin, weights=tube_mask[in_core].astype(float), minlength=n_bins) / np.maximum(count, 1)

    core_labels = tube_label[in_core]
    labelled = core_labels > 0
    label_counts = np.bincount(core_bin[labelled] * 25 + np.minimum(core_labels[labelled], 24),
                               minlength=n_bins * 25).reshape(n_bins, 25)    # bins x SuPreM label
    labelled_per_bin = label_counts.sum(1)
    usable = np.flatnonzero(labelled_per_bin >= 3)
    if len(usable) < 10:
        return None
    first, last = int(usable[0]), int(usable[-1])

    candidates = mask & (hu > -100) & (hu < 1500)
    bone_hu = threshold_otsu(hu[candidates]) if candidates.sum() > 1000 else 120.0
    bone_hu = float(np.clip(bone_hu, 80.0, 160.0))

    # --- disc evidence: dark band in the CT, gap in the mask, change of SuPreM label
    profile = gaussian_filter1d(mean_hu, 1.5 / BIN_MM, mode="nearest")
    spread = np.percentile(profile[first:last + 1], 90) - np.percentile(profile[first:last + 1], 10)
    ct_evidence = np.clip(valley_depth(profile) / (0.5 * spread + 1e-6), 0, 1.5)
    mask_evidence = np.clip(valley_depth(gaussian_filter1d(mask_fraction, 1.0 / BIN_MM, mode="nearest")) / 0.5, 0, 1.5)
    majority = np.where(labelled_per_bin >= 3, np.argmax(label_counts[:, 1:], axis=1) + 1, 0)
    below, above = shift(majority, 3), shift(majority, -3)
    label_change = ((below > 0) & (above > 0) & (below != above)).astype(float)
    evidence = 1.0 * ct_evidence + 0.5 * mask_evidence + 0.4 * ndi.uniform_filter1d(label_change, 3)
    evidence = gaussian_filter1d(evidence, 0.7, mode="nearest")

    # --- expected vertebra height = spacing between SuPreM's label centres (fine even if numbers are off)
    expected_height = None
    label_centres = np.sort(np.array([
        float((np.arange(n_bins) * label_counts[:, l]).sum() / label_counts[:, l].sum()) * BIN_MM
        for l in range(1, 25) if label_counts[:, l].sum() > 30]))
    if len(label_centres) >= 3 and (np.diff(label_centres) > 5).sum() >= 2:
        real_gap = np.diff(label_centres) > 5
        gaps = ndi.median_filter(np.diff(label_centres)[real_gap], size=min(3, int(real_gap.sum())), mode="nearest")
        midpoints = 0.5 * (label_centres[1:] + label_centres[:-1])[real_gap]
        expected_height = np.interp((np.arange(n_bins) + 0.5) * BIN_MM, midpoints, gaps)[first:last + 1]
        expected_height = np.clip(expected_height, MIN_LEVEL_MM, MAX_EXPECTED_MM)

    # --- split; ask for stronger discs if we get more levels than a spine can have
    threshold = DISC_THRESHOLD
    boundaries = split_into_levels(evidence[first:last + 1], threshold, expected_height)
    for _ in range(12):
        if len(boundaries) + 1 <= MAX_LEVELS:
            break
        threshold += 0.15
        boundaries = split_into_levels(evidence[first:last + 1], threshold, expected_height)
    n_levels = len(boundaries) + 1
    if n_levels > MAX_LEVELS:
        return None

    edges = [0] + list(boundaries) + [last - first]
    disc_positions = np.array([(first + b + 0.5) * BIN_MM for b in boundaries], dtype=np.float32)
    start_mm, end_mm = first * BIN_MM, (last + 1) * BIN_MM
    level_of = np.searchsorted(disc_positions, along, side="right").astype(np.int16)

    # --- number the levels: consecutive labels, shifted to agree best with SuPreM
    votes = np.array([label_counts[max(first + edges[j], 0):first + edges[j + 1] + 1].sum(0)
                      for j in range(n_levels)], float)
    present = [l for l in range(1, 25) if label_counts[:, l].sum() > 100]
    direction = 1
    if len(present) >= 3:
        positions = [(np.arange(n_bins) * label_counts[:, l]).sum() / label_counts[:, l].sum() for l in present]
        direction = 1 if np.corrcoef(present, positions)[0, 1] >= 0 else -1
    best_score, numbering = -1, None
    for lowest in range(1, 25):
        candidate = [lowest + direction * j for j in range(n_levels)]
        if min(candidate) < 1 or max(candidate) > 24:
            continue
        agreement = sum(votes[j, candidate[j]] for j in range(n_levels))
        if agreement > best_score:
            best_score, numbering = agreement, candidate
    if numbering is None:
        return None
    level_labels = np.array(numbering, dtype=np.uint8)

    # --- seeds in the middle of every body, and the level of every tube voxel
    in_body = (side / (1.2 * body_r)) ** 2 + (front / (0.85 * body_r)) ** 2 <= 1.0
    padded = np.concatenate([[-1e9], disc_positions, [1e9]]).astype(np.float32)
    to_disc = np.minimum(along - padded[level_of], padded[level_of + 1] - along)
    is_seed = (tube_mask & in_body & (np.abs(off_axis) < 1.5) & (along >= start_mm) & (along <= end_mm)
               & (to_disc >= 1.5))
    seeds = np.zeros(grid_shape, np.int32)
    seeds.ravel()[tube[is_seed]] = level_of[is_seed] + 1
    level = -np.ones(grid_shape, np.int16)
    level.ravel()[tube] = np.where((along >= start_mm - 5) & (along <= end_mm + 5), level_of, -1)

    # --- outside the tube: grow the body seeds through the bone
    bone_thickness = ndi.distance_transform_edt(mask & (hu > bone_hu), sampling=GRID_MM).astype(np.float32)
    grown = watershed(-(0.9 * bone_thickness + (1 - 0.9) * mask_thickness), seeds, mask=mask)
    fallback = np.zeros(grid_shape, np.uint8)
    fallback[grown > 0] = level_labels[grown[grown > 0] - 1]

    level_up, seeds_up = upsample(level), upsample(seeds)
    level_label = np.where(level_up >= 0, level_labels[np.clip(level_up, 0, n_levels - 1)], upsample(fallback))
    seed_label = np.where(seeds_up > 0, level_labels[np.clip(seeds_up - 1, 0, n_levels - 1)], 0)
    return axis, level_labels, level_label.astype(np.uint8), seed_label.astype(np.uint8)


# ----------------------------------------------------------------------------
# Step 4: bodies and posterior elements, at the original resolution
# ----------------------------------------------------------------------------
def vertebral_bodies(mask, spacing, axis_xy, body_radius):
    """Keep only the vertebral bodies: in every slice, an opening with a disk sized to the
    local body removes the thinner pedicles and arch, and we keep the blob on the axis."""
    column = bridge_discs(mask, spacing[2])
    bodies = np.zeros(mask.shape, bool)
    disks = {}
    for z in range(mask.shape[2]):
        radius_mm = float(np.clip(0.55 * body_radius[z], 2.5, 10.0))
        key = int(round(radius_mm / spacing[0] * 2))
        if key not in disks:
            disks[key] = disk(key / 2.0)
        if not column[:, :, z].any():
            continue
        opened = ndi.binary_opening(fill_small_holes(column[:, :, z], spacing[0] * spacing[1]), structure=disks[key])
        blobs, n = ndi.label(opened)
        if n == 0:
            continue
        i = int(np.clip(round(axis_xy[z][0]), 0, mask.shape[0] - 1))
        j = int(np.clip(round(axis_xy[z][1]), 0, mask.shape[1] - 1))
        if blobs[i, j] > 0:
            bodies[:, :, z] = blobs == blobs[i, j]
        else:   # axis just misses the body: take the nearest blob
            dist, nearest = ndi.distance_transform_edt(blobs == 0, return_indices=True)
            if dist[i, j] * spacing[0] <= 10.0:
                bodies[:, :, z] = blobs == blobs[nearest[0][i, j], nearest[1][i, j]]
    return bodies & mask


def one_label_per_bone_core(labels, posterior, ct):
    """Dense cortical bone (spinous process, lamina, articular process) is one bone, so it should
    carry one label. Cores above 650 HU that are at least 80 % one label get that label; mixed
    cores are split again at 750 HU (they usually span a joint) and otherwise left alone."""
    core_labels = np.zeros(labels.shape, np.uint8)

    def check(region, level, offset):
        cores, _ = ndi.label(region & (ct[offset] > (650.0, 750.0)[level]), structure=FACE_NEIGHBOURS)
        for core_id, box in enumerate(ndi.find_objects(cores), start=1):
            if box is None:
                continue
            core = cores[box] == core_id
            if core.sum() < 60:
                continue
            where = tuple(slice(b.start + o.start, b.stop + o.start) for b, o in zip(box, offset))
            votes = labels[where][core]
            votes = votes[votes > 0]
            if votes.size == 0:
                continue
            counts = np.bincount(votes)
            if counts.max() / votes.size >= 0.8:
                core_labels[where][core] = int(counts.argmax())
            elif level == 0:
                check(core.copy(), 1, where)

    check(posterior & (labels > 0), 0, tuple(slice(0, n) for n in posterior.shape))
    return core_labels


def label_vertebrae(seg, ct, spacing, axis, level_label, seed_label, keep_as_is):
    """Bodies get the label of their level. Posterior bone is grown from each vertebra's own
    pedicles with two independent floods; where they disagree, SuPreM's label decides."""
    mask = seg > 0
    z_mm = (np.arange(mask.shape[2]) + 0.5) * spacing[2]
    axis_xy = list(zip(np.interp(z_mm, axis.points[:, 2], axis.points[:, 0]) / spacing[0] - 0.5,
                       np.interp(z_mm, axis.points[:, 2], axis.points[:, 1]) / spacing[1] - 0.5))
    body_radius = np.interp(z_mm, axis.points[:, 2], axis.body_radius)
    bodies = vertebral_bodies(mask, spacing, axis_xy, body_radius)
    bodies = ndi.binary_dilation(bodies, structure=FACE_NEIGHBOURS, iterations=3) & mask   # give back the rim
    body_labels = np.where(bodies, level_label, 0).astype(np.int32)
    posterior = mask & ~bodies

    # starting points: the bodies, posterior bone right next to a body (pedicle roots),
    # and the core seeds of every level (in case a body was missed)
    reach = max(1, int(round(2.0 / float(np.min(spacing)))))
    next_to_body = ndi.grey_dilation(body_labels.astype(np.uint8), footprint=ball(reach))
    seeds = np.where(bodies, body_labels, np.where(posterior, next_to_body, 0)).astype(np.int32)
    del next_to_body
    missing = (seeds == 0) & (seed_label > 0) & mask
    seeds[missing] = seed_label[missing]

    # labels without a level (C1 has no body) stay exactly as SuPreM segmented them
    untouched = mask & np.isin(seg, keep_as_is)
    seeds[untouched] = seg[untouched]
    bodies &= ~untouched
    posterior &= ~untouched

    # flood 1 follows bright bone; flood 2 follows distance but stops at joint gaps and soft tissue
    ct = np.clip(ndi.gaussian_filter(ct.astype(np.float32), 0.6), -200, 1500)
    by_brightness = watershed(-ct, seeds, mask=mask)
    joint_gaps = ndi.grey_closing(ct, footprint=ball(2)) - ct
    barriers = np.round(np.maximum(joint_gaps, np.clip(200.0 - ct, 0, None)) / 10.0)
    by_distance = watershed(barriers, seeds, mask=mask)
    del joint_gaps, barriers
    agree = by_brightness == by_distance

    # translate SuPreM's numbering into ours, learned where both floods agree
    translate = np.zeros(256, np.int32)
    reference = posterior & agree
    for suprem_label in np.unique(seg[reference]):
        region = reference & (seg == suprem_label)
        if suprem_label == 0 or region.sum() < 300:
            continue
        counts = np.bincount(by_brightness[region], minlength=32)
        if counts.max() / region.sum() >= 0.6:
            translate[suprem_label] = int(np.argmax(counts))
    suprem = translate[seg]

    # agreement wins; otherwise take the flood SuPreM agrees with; otherwise SuPreM itself
    result = np.where(agree, by_brightness,
                      np.where(suprem == by_brightness, by_brightness,
                               np.where(suprem == by_distance, by_distance, 0)))
    undecided = posterior & (result == 0)
    use_suprem = undecided & (suprem > 0)
    result = np.where(use_suprem, suprem, result)
    undecided &= ~use_suprem
    result = np.where(bodies, body_labels, result).astype(np.uint8)
    result[untouched] = seg[untouched]
    del by_brightness, by_distance, suprem, body_labels, agree

    # anything still open takes the label of its neighbours
    if undecided.any():
        result[undecided] = 0
        for _ in range(50):
            grown = ndi.grey_dilation(result, footprint=FACE_NEIGHBOURS)
            reached = undecided & (grown > 0)
            result[reached] = grown[reached]
            undecided &= ~reached
            if not undecided.any():
                break

    # one label per dense bone core, then re-grow the softer bone around cores that changed
    cores = one_label_per_bone_core(result, posterior, ct)
    in_core = cores > 0
    if in_core.any():
        seeds[in_core] = cores[in_core]
        regrown = watershed(-ct, seeds, mask=mask)
        near_change = ndi.binary_dilation(in_core & (cores != result), structure=FACE_NEIGHBOURS,
                                          iterations=max(1, int(round(8.0 / float(np.min(spacing))))))
        update = posterior & ~in_core & (regrown > 0) & near_change & ~untouched
        result[in_core] = cores[in_core]
        result[update] = regrown[update]
        del regrown, near_change
    del cores, ct, seeds

    # tiny posterior islands that do not touch the rest of their label join their surroundings
    for label in np.unique(result[posterior]):
        if label == 0:
            continue
        pieces, n = ndi.label((result == label) & posterior, structure=FACE_NEIGHBOURS)
        if n <= 1:
            continue
        sizes, boxes = np.bincount(pieces.ravel()), ndi.find_objects(pieces)
        for piece in range(1, n + 1):
            if sizes[piece] >= 200:
                continue
            box = tuple(slice(max(0, b.start - 1), min(dim, b.stop + 1)) for b, dim in zip(boxes[piece - 1], pieces.shape))
            this = pieces[box] == piece
            around = result[box][ndi.binary_dilation(this, structure=FACE_NEIGHBOURS) & ~this]
            around = around[(around > 0) & (around != label)]
            if around.size:
                result[box][this] = np.bincount(around).argmax()
    result[~mask] = 0
    return result


# ----------------------------------------------------------------------------
# Step 5: final checks
# ----------------------------------------------------------------------------
def reattach_loose_pieces(labels):
    """A vertebra is one bone. A piece of a label that does not touch the rest of that label
    (and is at most 10 % of it) is given to the vertebra it is attached to."""
    for _ in range(3):
        changed = False
        for label in [int(v) for v in np.unique(labels) if v > 0]:
            pieces, n = ndi.label(labels == label, structure=FACE_NEIGHBOURS)
            if n <= 1:
                continue
            sizes, boxes = np.bincount(pieces.ravel()), ndi.find_objects(pieces)
            main = int(np.argmax(sizes[1:])) + 1
            for piece in range(1, n + 1):
                if piece == main or sizes[piece] > 0.10 * int(sizes[1:].sum()):
                    continue
                box = tuple(slice(max(0, b.start - 1), min(dim, b.stop + 1)) for b, dim in zip(boxes[piece - 1], labels.shape))
                this = pieces[box] == piece
                around = labels[box][ndi.binary_dilation(this, structure=FACE_NEIGHBOURS) & ~this]
                around = around[(around > 0) & (around != label)]
                if around.size:
                    labels[box][this] = int(np.bincount(around).argmax())
                    changed = True
        if not changed:
            break
    return labels


def remove_far_fragments(labels, spacing):
    """Pieces smaller than 500 mm^3 lying more than 25 mm from the rest of their label are removed."""
    result = labels.copy()
    for label in np.unique(labels[labels > 0]):
        pieces, n = ndi.label(labels == label, structure=ALL_NEIGHBOURS)
        if n <= 1:
            continue
        volumes = np.bincount(pieces.ravel(), minlength=n + 1)[1:].astype(np.float64) * float(np.prod(spacing))
        main = int(np.argmax(volumes)) + 1
        boxes = ndi.find_objects(pieces)
        main_pts = np.argwhere(pieces == main) * spacing
        tree = cKDTree(main_pts[::max(1, len(main_pts) // 150000)])
        for piece in range(1, n + 1):
            if piece == main or volumes[piece - 1] >= 500.0:
                continue
            box = boxes[piece - 1]
            pts = (np.argwhere(pieces[box] == piece) + np.array([b.start for b in box])) * spacing
            if float(tree.query(pts[::max(1, len(pts) // 2000)])[0].min()) > 25.0:
                result[box][pieces[box] == piece] = 0
    return result


def relabel(seg, ct, spacing):
    """Full pipeline for one case (arrays in RAS orientation). Returns None if it cannot run."""
    seg = remove_distant_specks(np.ascontiguousarray(seg), spacing)
    found = find_levels(seg, ct, spacing)
    if found is None:
        return None
    axis, level_labels, level_label, seed_label = found
    level_label[seg == 0] = 0
    seed_label[seg == 0] = 0
    no_level = sorted(v for v in np.unique(seg[seg > 0]).tolist()
                      if v < level_labels.min() or v > level_labels.max())

    result = label_vertebrae(seg, ct, spacing, axis, level_label, seed_label, no_level)
    del level_label, seed_label

    # slices above or below the axis could not be analysed: keep SuPreM's labels there
    z_mm = (np.arange(seg.shape[2]) + 0.5) * spacing[2]
    outside = (z_mm < axis.points[0, 2] - 3.0) | (z_mm > axis.points[-1, 2] + 3.0)
    result[:, :, outside] = seg[:, :, outside]

    result = reattach_loose_pieces(result)
    return remove_far_fragments(result, spacing)




# ----------------------------------------------------------------------------
# Whole-scan entry point
# ----------------------------------------------------------------------------
def relabel_volume(labels, ct, affine, zooms):
    """Relabel the vertebrae of one scan.

    labels: integer array, 0 = background, 1 (L5) ... 24 (C1)
    ct:     CT in HU on the same grid (array or array proxy; only the crop is read)
    affine, zooms: geometry of that grid

    Returns (result, status). result is a uint8 array with the input's shape; status is "ok",
    "empty" or "spine not found" (in the last two cases the input labels are returned unchanged).
    """
    seg = np.asarray(labels).astype(np.uint8)
    result = seg.copy()
    if not seg.any():
        return result, "empty"

    # crop to the mask (plus a margin) and turn it to RAS so "up" is always +z
    voxel_size = np.array(zooms[:3], float)
    margin = np.ceil(CROP_MARGIN_MM / voxel_size).astype(int)
    box = ndi.find_objects((seg > 0).astype(np.uint8))[0]
    crop = tuple(slice(max(0, b.start - m), min(n, b.stop + m)) for b, m, n in zip(box, margin, seg.shape))

    ras = nio.axcodes2ornt(("R", "A", "S"))
    to_ras = nio.ornt_transform(nio.io_orientation(affine), ras)
    from_ras = nio.ornt_transform(ras, nio.io_orientation(affine))
    spacing = np.zeros(3)
    for i in range(3):
        spacing[int(to_ras[i, 0])] = voxel_size[i]

    seg_ras = np.ascontiguousarray(nio.apply_orientation(seg[crop], to_ras))
    ct_ras = np.ascontiguousarray(nio.apply_orientation(np.asarray(ct[crop], dtype=np.float32), to_ras))
    relabelled = relabel(seg_ras, ct_ras, spacing)
    if relabelled is None:
        return result, "spine not found"
    result[crop] = nio.apply_orientation(relabelled, from_ras)
    return result, "ok"
