"""ShapeKit-Anchor vertebrae engine: anatomy-derived names, learned boundaries.

Library half of the engine. The ShapeKit adapter in vertebrae_anchor.py turns
the per-organ segmentation dict into the combined label volume this module
works on (1 = L5 ... 24 = C1, background 0) and back again.

What the network gets wrong, and why a rule-based repair is possible
---------------------------------------------------------------------
A per-voxel argmax over 25 classes gives every voxel one label, but nothing
in the network relates a voxel's label to its neighbours' or to the order of
levels along the spine. The errors that follow are therefore not random
noise, they are violations of a few facts about how a spine is built:

    A1  The label index increases monotonically toward the superior direction.
    A2  One vertebra is one connected component.
    A3  The visible levels form a contiguous run. A field of view truncates
        the spine at its ends, never punches a hole in the middle.
    A4  Consecutive vertebrae are evenly spaced, with a pitch that tapers
        smoothly from lumbar to cervical.
    A5  One voxel belongs to one vertebra.
    A6  A component that is a tiny fraction of its neighbours is a fragment.

The network is much better at finding vertebrae than at naming them. So the
engine re-derives the naming from the geometry instead of repairing labels
one at a time.

Two regimes
-----------
Clean spine (one body per label, consistent order): rank the bodies along the
spine axis and renumber with one integer offset chosen to agree with the
network's own votes. The output differs from the input only where geometry
and network disagree, and then the whole sequence moves coherently.

Scrambled spine (labels repeat, bodies split or merged): the component count
carries no usable information, and the only thing that can be trusted is the
clean part of the spine. The anchor-and-slab resolver finds the anchors (one
body per label, plausible volume, on a smooth curve of position against
level), fits that curve, reads off where every other level must sit, and
cuts the spine mask into slabs between anchors. Anchor levels keep the
network's own boundary. Cut placement uses a volume-share prior (each
rebuilt level gets its expected fraction of the volume between anchors)
snapped to the mask waist at the disc where one exists. A balanced watershed
then lets thoracic levels take back posterior elements a planar cut handed
to the level below, and a one-voxel Gaussian argmax removes the resampling
staircase. Optionally (vertebrae_anchor_prompt.py, needs a GPU and the
nnInteractive package) the planar boundary of each rebuilt level is
replaced by a learned one, prompted once at the position the resolver
found. Anchors are never prompted.

Everything geometric is computed in millimetres from the NIfTI affine,
never in voxels: vertebra spacing is a physical quantity, and a threshold in
voxels silently changes meaning when the slice thickness changes.

Validation: on the AbdomenAtlasDemo warm-up cases the conservative first
version (keep the network's names when the count cannot be trusted) scored
76.7% mean DSC against the human-revised labels, and this engine scored
92.8% with every level at or above 90%. Details in the README section.

Author: Ura Modi (ura.modi24@gmail.com)
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

try:
    import cc3d
except ImportError:  # pragma: no cover
    cc3d = None
from scipy import ndimage

# --------------------------------------------------------------------------
# Anatomy
# --------------------------------------------------------------------------

# SuPreM class_map_part_vertebrae. Index increases toward the head.
CLASS_MAP = {
    1: "vertebrae_L5", 2: "vertebrae_L4", 3: "vertebrae_L3", 4: "vertebrae_L2",
    5: "vertebrae_L1", 6: "vertebrae_T12", 7: "vertebrae_T11", 8: "vertebrae_T10",
    9: "vertebrae_T9", 10: "vertebrae_T8", 11: "vertebrae_T7", 12: "vertebrae_T6",
    13: "vertebrae_T5", 14: "vertebrae_T4", 15: "vertebrae_T3", 16: "vertebrae_T2",
    17: "vertebrae_T1", 18: "vertebrae_C7", 19: "vertebrae_C6", 20: "vertebrae_C5",
    21: "vertebrae_C4", 22: "vertebrae_C3", 23: "vertebrae_C2", 24: "vertebrae_C1",
}
N_LABELS = 24

# Physical priors, in millimetres. Deliberately loose: they exist to reject
# absurdities, not to impose a template on a real patient.
MIN_VERTEBRA_VOLUME_MM3 = 800.0     # a true body is several cm^3, this is a floor
FRAGMENT_VOLUME_FRACTION = 0.10     # island < 10% of its label's main body
LOCAL_SPACING_FRACTION = 0.45       # closer than this * local pitch = one vertebra
LOCAL_SPACING_WINDOW = 2            # neighbours each side used for the local pitch
SLIVER_VOLUME_FRACTION = 0.25       # a body this much smaller than its peers is a sliver
MIN_PLAUSIBLE_SPACING_MM = 8.0      # cervical discs are small but not this small
MAX_PLAUSIBLE_SPACING_MM = 60.0     # beyond this, a level is missing
GAP_ROUNDING_TOLERANCE = 0.35       # how far a gap may sit from an integer multiple
ANCHOR_MIN_COUNT = 5                # fewer reliable levels than this: no fit, keep model
ANCHOR_VOLUME_HI = 1.55             # a body this much bigger than the trend is two vertebrae
ANCHOR_VOLUME_LO = 0.45             # this much smaller is a fragment, not an anchor
ANCHOR_VOLUME_HI_C2 = 2.4           # C2 carries the dens and sits well above the trend
ANCHOR_RESID_FRACTION = 0.35        # position residual allowed, as a fraction of local pitch
ANCHOR_EXTENT_MAX = 2.6             # taller than this * local pitch is two vertebrae
CUT_MODE = "volume"                 # 'volume': equal share of the volume trend, 'disc': mask waist, 'midpoint': halfway
CUT_SEARCH_FRACTION = 0.22          # search window for the waist, as a fraction of pitch, each side of the midpoint
REFINE_PROCESSES = True             # balanced watershed pass to reattach posterior elements
REFINE_LEVELS = range(6, 18)        # T12 (6) to T1 (17): only thoracic levels may take voxels back
CORE_FRACTION = 0.40                # central share of each level, along the axis, used as its marker
SMOOTH_STAIRCASE = True             # final pass: remove the resampling staircase
SMOOTH_SIGMA_VOX = 1.0              # Gaussian sigma in voxels for that pass

# Vertebrae are bone. Cancellous bone starts around +150 HU and cortical bone is
# far higher, while the muscle and fat a mask leaks into sit below +100. The
# threshold is deliberately below the textbook value: the aim is to strip
# obvious spill into soft tissue, not to erode the trabecular interior or the
# partial-volume rim, which would look worse in ITK-SNAP than the leak did.
BONE_HU_MIN = 100.0



# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

@dataclass
class Body:
    """One candidate vertebral body found in the prediction."""
    voxels: np.ndarray            # (N, 3) integer voxel indices
    votes: Counter                # original label -> voxel count
    centroid_mm: np.ndarray = field(default=None)
    axis_pos: float = 0.0         # projection onto the spine axis, mm
    axis_lo: float = 0.0          # extent along that axis, mm
    axis_hi: float = 0.0
    rank: int = -1                # position in the inferior -> superior order
    final_label: int = 0

    @property
    def extent(self) -> float:
        return max(self.axis_hi - self.axis_lo, 1e-6)

    @property
    def size(self) -> int:
        return len(self.voxels)

    @property
    def majority_label(self) -> int:
        return self.votes.most_common(1)[0][0] if self.votes else 0


def voxel_to_world(idx: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map integer voxel indices (N, 3) to world millimetres (N, 3)."""
    homo = np.concatenate([idx.astype(np.float64), np.ones((len(idx), 1))], axis=1)
    return (affine @ homo.T).T[:, :3]


def superior_direction(affine: np.ndarray) -> np.ndarray:
    """Unit world vector pointing toward the head.

    NIfTI world space is RAS+, so +z is superior regardless of how the voxel
    axes happen to be ordered on disk. Reading it from the affine rather than
    assuming an array axis is what makes this work on both the axial and the
    reformatted scans in the demo set.
    """
    return np.array([0.0, 0.0, 1.0])


def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected components, cc3d when available, scipy otherwise."""
    if cc3d is not None:
        lab = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
        return lab, int(lab.max())
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return lab, int(n)


def components_of_labelmap(seg: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of a *label map*, split by label as well as by touching.

    cc3d does this in one pass over the volume, treating a change of value as a
    boundary. The scipy fallback has to loop, because ndimage.label would fuse
    two touching vertebrae into a single object.
    """
    if cc3d is not None:
        lab = cc3d.connected_components(seg.astype(np.uint8), connectivity=26)
        return lab, int(lab.max())
    out = np.zeros(seg.shape, dtype=np.int32)
    total = 0
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    for value in range(1, N_LABELS + 1):
        mask = seg == value
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=struct)
        out[mask] = lab[mask] + total
        total += n
    return out, total


# --------------------------------------------------------------------------
# Stage 0: intensity sanity check against the CT
# --------------------------------------------------------------------------

def trim_non_bone(seg: np.ndarray, ct: np.ndarray, stats: dict) -> np.ndarray:
    """Drop labelled voxels that are plainly not bone.

    Only material connected to the outside of the mask through other non-bone
    voxels is eligible. An enclosed voxel below the threshold is marrow or a
    trabecular gap, and punching holes through the middle of a vertebra would be
    a worse artefact in ITK-SNAP than the leak being removed.
    """
    if ct.shape != seg.shape:
        stats["warnings"].append(
            f"CT shape {ct.shape} != mask shape {seg.shape}, intensity check skipped")
        return seg

    seg = seg.copy()
    fg = seg > 0
    if not fg.any():
        return seg

    # Peeling one shell at a time gives the right answer but costs an erosion of
    # the whole volume per layer, and these volumes reach 365 million voxels.
    # The fixed point of that peel has a closed form: a non-bone voxel is
    # removed exactly when it can be reached from outside the mask through other
    # non-bone voxels. So label the non-bone regions once and drop the ones that
    # touch the mask surface. Marrow enclosed by bone is not reachable and stays,
    # which is the same protection the peel gave, in a single pass.
    struct = np.ones((3, 3, 3), bool)
    soft_inside = fg & (ct < BONE_HU_MIN)
    if not soft_inside.any():
        return seg

    comps, ncomp = connected_components(soft_inside)
    if ncomp == 0:
        return seg

    # Voxels adjacent to anything outside the mask, the volume border included.
    exposed = ndimage.binary_dilation(~fg, structure=struct, border_value=1) & soft_inside
    open_labels = np.unique(comps[exposed])
    open_labels = open_labels[open_labels > 0]
    if not len(open_labels):
        return seg

    drop = np.isin(comps, open_labels)
    total = int(drop.sum())
    if total:
        seg[drop] = 0
        stats["non_bone_voxels_trimmed"] = total
    return seg


# --------------------------------------------------------------------------
# Stage 1: split the prediction into candidate bodies
# --------------------------------------------------------------------------

def extract_bodies(seg: np.ndarray, vox_mm3: float, stats: dict) -> list[Body]:
    """Break every predicted label into connected components and drop fragments.

    One pass labels the whole volume, then the voxels are grouped by component
    with a sort. The obvious loop, running a full-volume comparison per
    component, is what makes this unusable at scale: a large scan has of the
    order of a hundred components, and a hundred passes over a hundred million
    voxels is tens of minutes of pure scanning.

    A component is kept if it is the largest for its label, or if it is large
    enough to be a vertebra in its own right. The second case matters: when the
    model splits one physical vertebra across two labels, the smaller piece is
    real anatomy wearing the wrong name, and deleting it would throw away a body
    that the renumbering could have corrected.
    """
    cc, ncc = components_of_labelmap(seg)
    if ncc == 0:
        return []

    flat = np.flatnonzero(cc)
    if not len(flat):
        return []
    comp = cc.ravel()[flat]
    labs = seg.ravel()[flat]

    order = np.argsort(comp, kind="stable")
    comp_s, flat_s, labs_s = comp[order], flat[order], labs[order]
    ids = np.arange(1, ncc + 1)
    starts = np.searchsorted(comp_s, ids, side="left")
    ends = np.searchsorted(comp_s, ids, side="right")
    sizes = ends - starts

    comp_label = np.zeros(ncc + 1, dtype=np.int64)
    nonempty = np.flatnonzero(sizes)
    comp_label[nonempty + 1] = labs_s[starts[nonempty]]

    # Largest component per label, and how many labels are split at all.
    largest_size: dict[int, int] = {}
    largest_id: dict[int, int] = {}
    per_label_counts: dict[int, int] = {}
    for c in nonempty:
        label = int(comp_label[c + 1])
        per_label_counts[label] = per_label_counts.get(label, 0) + 1
        if sizes[c] > largest_size.get(label, -1):
            largest_size[label] = int(sizes[c])
            largest_id[label] = int(c)
    stats["labels_fragmented"] += sum(1 for n in per_label_counts.values() if n > 1)

    bodies: list[Body] = []
    for c in nonempty:
        size = int(sizes[c])
        label = int(comp_label[c + 1])
        is_largest = largest_id[label] == int(c)
        big_enough = (size * vox_mm3 >= MIN_VERTEBRA_VOLUME_MM3
                      and size >= FRAGMENT_VOLUME_FRACTION * largest_size[label])
        if not is_largest and not big_enough:
            stats["islands_removed"] += 1
            stats["island_voxels_removed"] += size
            continue
        vox = np.stack(np.unravel_index(flat_s[starts[c]:ends[c]], seg.shape), axis=1)
        bodies.append(Body(voxels=vox, votes=Counter({label: size})))
    return bodies


# --------------------------------------------------------------------------
# Stage 2: order the bodies along the spine
# --------------------------------------------------------------------------

def spine_axis(bodies: list[Body]) -> np.ndarray:
    """Principal axis of the centroid cloud, oriented toward the head.

    PCA rather than the world z axis alone, so a tilted or scoliotic spine is
    still ordered correctly. The sign is fixed against the superior direction
    so 'first' always means 'most inferior'.
    """
    if len(bodies) < 3:
        return superior_direction(None)
    pts = np.stack([b.centroid_mm for b in bodies])
    weights = np.array([b.size for b in bodies], dtype=np.float64)
    mean = np.average(pts, axis=0, weights=weights)
    centred = (pts - mean) * np.sqrt(weights)[:, None]
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return superior_direction(None)
    axis = vt[0]
    if np.dot(axis, superior_direction(None)) < 0:
        axis = -axis
    return axis / (np.linalg.norm(axis) + 1e-12)


def merge_duplicates(bodies: list[Body], stats: dict) -> list[Body]:
    """Fuse bodies that are two names for one physical vertebra.

    The test is centre-to-centre spacing measured against the *local* pitch of
    the spine, not against a single global number and not against axial overlap.

    Both of the obvious alternatives are wrong on real data. Overlap fails
    because a vertebra mask includes the spinous and transverse processes, which
    reach alongside the neighbouring level: on the demo scans genuinely adjacent
    vertebrae overlap along the spine axis by 0.46 to 0.98, so an overlap test
    fuses the entire spine. A global median spacing fails the other way, because
    the pitch tapers from about 30 mm in the lumbar spine to about 10 mm in the
    cervical, so one threshold either misses lumbar duplicates or destroys the
    cervical levels.

    The pitch changes smoothly, so the median of the nearby gaps is a good local
    scale, and a duplicate stands out as a gap far smaller than its neighbours.
    """
    if len(bodies) < 3:
        return bodies
    bodies = sorted(bodies, key=lambda b: b.axis_pos)
    gaps = [bodies[i + 1].axis_pos - bodies[i].axis_pos for i in range(len(bodies) - 1)]

    # Decide every merge from the original spacings before applying any, so one
    # merge cannot cascade into the next.
    w = LOCAL_SPACING_WINDOW
    drop = []
    for i, g in enumerate(gaps):
        local = float(np.median(gaps[max(0, i - w): i + w + 1]))
        drop.append(local > 0 and g < LOCAL_SPACING_FRACTION * local)

    merged: list[Body] = [bodies[0]]
    for i, body in enumerate(bodies[1:]):
        if drop[i]:
            prev = merged[-1]
            prev.voxels = np.concatenate([prev.voxels, body.voxels])
            prev.votes.update(body.votes)
            prev.axis_lo = min(prev.axis_lo, body.axis_lo)
            prev.axis_hi = max(prev.axis_hi, body.axis_hi)
            stats["duplicates_merged"] += 1
        else:
            merged.append(body)
    return merged


# --------------------------------------------------------------------------
# Stage 3: re-derive the numbering
# --------------------------------------------------------------------------

def absorb_slivers(bodies: list[Body], stats: dict) -> list[Body]:
    """Fold a body far too small to be a vertebra into its nearest neighbour.

    This is the other half of the duplicate problem, and spacing cannot see it.
    When the model gives a thin slab of one vertebra the name of the level above,
    that slab is a legitimate largest-component for its own label, so the
    fragment filter keeps it, and its centroid sits about as far from its parent
    as a real neighbour would. What gives it away is volume: it is a fraction of
    every genuine body around it.

    The comparison is against the median body in this scan, not an absolute size,
    so it holds for a cervical spine and a lumbar one alike.
    """
    if len(bodies) < 3:
        return bodies
    sizes = np.array([b.size for b in bodies], dtype=np.float64)
    floor = SLIVER_VOLUME_FRACTION * float(np.median(sizes))
    keep = [b for b in bodies if b.size >= floor]
    if len(keep) == len(bodies) or not keep:
        return bodies
    for b in bodies:
        if b.size >= floor:
            continue
        host = min(keep, key=lambda k: abs(k.axis_pos - b.axis_pos))
        host.voxels = np.concatenate([host.voxels, b.voxels])
        host.votes.update(b.votes)
        host.axis_lo = min(host.axis_lo, b.axis_lo)
        host.axis_hi = max(host.axis_hi, b.axis_hi)
        stats["slivers_absorbed"] += 1
    return sorted(keep, key=lambda b: b.axis_pos)


def assign_ranks(bodies: list[Body], stats: dict) -> int:
    """Give each body a rank, leaving holes where a level was missed.

    Ranks are what the numbering is built on, so a missed vertebra has to
    consume a rank. Otherwise one miss shifts every label above it by one, and
    a single dropped level turns into twenty wrong names.
    """
    if len(bodies) < 2:
        for i, b in enumerate(bodies):
            b.rank = i
        return len(bodies)

    gaps = np.array([bodies[i + 1].axis_pos - bodies[i].axis_pos
                     for i in range(len(bodies) - 1)])

    bodies[0].rank = 0
    cursor = 0
    w = LOCAL_SPACING_WINDOW
    for i, gap in enumerate(gaps):
        # Local pitch again, for the same reason as in merge_duplicates: the
        # spine tapers, so a normal lumbar gap is nearly twice a normal cervical
        # one. Measured against a single global median, healthy lumbar spacing
        # looks like a missed level and every rank above it is pushed up.
        window = gaps[max(0, i - w): i + w + 1]
        plausible = window[(window > MIN_PLAUSIBLE_SPACING_MM)
                           & (window < MAX_PLAUSIBLE_SPACING_MM)]
        local = float(np.median(plausible)) if len(plausible) else float(np.median(window))
        steps = 1
        if np.isfinite(local) and local > 0:
            ratio = gap / local
            nearest = int(round(ratio))
            if nearest >= 2 and abs(ratio - nearest) <= GAP_ROUNDING_TOLERANCE * nearest:
                steps = nearest
                stats["missing_levels_inferred"] += steps - 1
        cursor += steps
        bodies[i + 1].rank = cursor
    return cursor + 1


def choose_offset(bodies: list[Body], span: int, stats: dict) -> int:
    """Pick the numbering that disagrees with the model as little as possible.

    label(body) = offset + rank. The offset is a single integer for the whole
    spine, so the sequence stays internally consistent whatever it is. Scoring
    it by voxel-weighted agreement with the original prediction means a spine
    the model already named correctly keeps its names, and only a spine whose
    geometry contradicts its labels gets moved.
    """
    lo = 1 - min(b.rank for b in bodies)
    hi = N_LABELS - max(b.rank for b in bodies)
    if hi < lo:  # more bodies than there are vertebrae, clamp and accept clipping
        stats["warnings"].append(
            f"{span} ranks span more than {N_LABELS} levels, numbering was clipped")
        return max(1 - min(b.rank for b in bodies), min(lo, hi))

    best_offset, best_score = lo, -1.0
    for offset in range(lo, hi + 1):
        score = 0.0
        for b in bodies:
            proposed = offset + b.rank
            score += b.votes.get(proposed, 0)
        # Nudge toward keeping the model's own most confident anchor.
        if score > best_score:
            best_offset, best_score = offset, score
    total = sum(b.size for b in bodies)
    stats["label_agreement_before_offset"] = round(best_score / max(total, 1), 4)
    return best_offset


# --------------------------------------------------------------------------
# Stage 5: anchors and slabs
# --------------------------------------------------------------------------

def _robust_polyfit(x, y, deg, thr_fn, iters=4):
    """Polynomial fit with iterative outlier rejection. Returns coef, inlier mask."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inl = np.ones(len(x), dtype=bool)
    coef = np.polyfit(x, y, min(deg, max(len(x) - 1, 1)))
    for _ in range(iters):
        if inl.sum() <= deg + 1:
            break
        coef = np.polyfit(x[inl], y[inl], min(deg, inl.sum() - 1))
        resid = y - np.polyval(coef, x)
        thr = thr_fn(coef, x, resid[inl])
        new = np.abs(resid) < thr
        if new.sum() < deg + 2 or np.array_equal(new, inl):
            inl = new if new.sum() >= deg + 2 else inl
            break
        inl = new
    return coef, inl


def _monotone(coef, lo=1, hi=N_LABELS):
    ks = np.arange(lo, hi + 1, dtype=np.float64)
    d = np.polyval(np.polyder(coef), ks)
    return bool(np.all(d > 0))


def _reattach_fragments(out: np.ndarray, max_rounds: int = 3) -> np.ndarray:
    """Give every non-largest component of a label to the label it touches most."""
    struct = np.ones((3, 3, 3), dtype=bool)
    for _ in range(max_rounds):
        moved = 0
        for k in range(1, N_LABELS + 1):
            m = out == k
            if not m.any():
                continue
            cc, n = connected_components(m)
            if n <= 1:
                continue
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            keep = int(sizes.argmax())
            for comp in range(1, n + 1):
                if comp == keep or sizes[comp] == 0:
                    continue
                piece = cc == comp
                ring = ndimage.binary_dilation(piece, structure=struct) & ~piece & (out > 0)
                if not ring.any():
                    # Touches nothing. A small free-floating piece is an island by
                    # the same rule used at the start; a large one is left alone.
                    if sizes[comp] < FRAGMENT_VOLUME_FRACTION * sizes[keep]:
                        out[piece] = 0
                        moved += 1
                    continue
                votes = np.bincount(out[ring].ravel(), minlength=N_LABELS + 1)
                votes[k] = 0
                if votes.max() == 0:
                    continue
                out[piece] = int(votes.argmax())
                moved += 1
        if moved == 0:
            break
    return out


def _refine_processes(out: np.ndarray, mask: np.ndarray, affine: np.ndarray,
                      axis: np.ndarray, rebuilt: set[int]) -> np.ndarray:
    """Let each thoracic level reclaim the posterior elements attached to it.

    The planar cuts get the bodies right and hand part of each spinous process
    to the neighbour below. A marker-based watershed can move those voxels
    back, and it failed earlier only because the markers were unbalanced (whole
    anchor bodies against single points). Here every level, anchor or rebuilt,
    is seeded the same way, with the central 40 per cent of its own voxels
    along the axis, so the flood only has to settle the contested band around
    each cut, and no level can swallow a neighbour it did not already border.
    The result is applied to thoracic levels only, for the reason given at the
    point of application below.
    """
    from skimage.segmentation import watershed
    idx = np.argwhere(mask)
    if not len(idx):
        return out
    s = voxel_to_world(idx, affine) @ axis
    labs = out[idx[:, 0], idx[:, 1], idx[:, 2]]
    markers = np.zeros(out.shape, dtype=np.int32)
    lo_f, hi_f = 0.5 - CORE_FRACTION / 2, 0.5 + CORE_FRACTION / 2
    for k in np.unique(labs):
        if k == 0:
            continue
        m = labs == k
        q_lo, q_hi = np.quantile(s[m], [lo_f, hi_f])
        core = m & (s >= q_lo) & (s <= q_hi)
        sel = idx[core]
        markers[sel[:, 0], sel[:, 1], sel[:, 2]] = int(k)
    dist = ndimage.distance_transform_edt(mask)
    grown = watershed(-dist, markers=markers, mask=mask).astype(out.dtype)

    # The flood is allowed to overrule the cut only where the anatomy says the
    # cut is the wrong instrument. A thoracic spinous process slopes steeply
    # downward and lies beside the body of the level below, so a planar cut by
    # height hands part of it to the neighbour, and a flood along the bone
    # gives it back. Lumbar and cervical processes project nearly straight
    # back, so the cut already has them right, and measured on the demo scan
    # the flood there cost 3 to 6 points at L1 and T12 while gaining 2 to 8 at
    # T11 to T6. So: a voxel changes label only if the flood assigns it to a
    # thoracic level and the cut had it on a thoracic level too.
    #
    # And only a rebuilt level may take voxels. An anchor's boundary is the
    # model's own, which on the demo scan scores about 90 against an
    # independent segmenter, and letting the flood redraw it cost T4 to T1
    # four to seven points. A rebuilt level's boundary is a planar cut, which
    # is what the flood is there to improve. So an anchor can lose a process
    # that the cut wrongly gave it, and never gain one.
    thoracic = np.zeros(N_LABELS + 1, dtype=bool)
    for k in REFINE_LEVELS:
        thoracic[k] = True
    takes = np.zeros(N_LABELS + 1, dtype=bool)
    for k in rebuilt:
        if thoracic[k]:
            takes[k] = True
    labs_grown = grown[idx[:, 0], idx[:, 1], idx[:, 2]]
    move = thoracic[labs] & takes[labs_grown]
    result = out.copy()
    sel = idx[move]
    result[sel[:, 0], sel[:, 1], sel[:, 2]] = labs_grown[move]
    unreached = mask & (result == 0)
    result[unreached] = out[unreached]
    return result


def smooth_staircase(seg: np.ndarray, sigma: float = SMOOTH_SIGMA_VOX) -> np.ndarray:
    """Remove the resampling staircase from a label map.

    The network predicted on a 1.5 mm grid and the result was carried back to
    the scan's own grid by nearest neighbour, which on a 0.7 mm scan means every
    boundary is a two-voxel step. A label map a person has revised at native
    resolution has no such steps. Smoothing each label's indicator and taking
    the argmax, background included, moves only the voxels on those steps and
    leaves every interior voxel alone. Measured against an independent
    reference it is worth a few tenths of a point on every level, never more,
    and never meaningfully less.
    """
    labels = [int(k) for k in np.unique(seg) if k > 0]
    if not labels:
        return seg
    best = ndimage.gaussian_filter((seg == 0).astype(np.float32), sigma=sigma)
    out = np.zeros(seg.shape, dtype=seg.dtype)
    for k in labels:
        sm = ndimage.gaussian_filter((seg == k).astype(np.float32), sigma=sigma)
        better = sm > best
        best[better] = sm[better]
        out[better] = k
    return out


def resolve_by_slabs(bodies: list[Body], seg: np.ndarray, affine: np.ndarray,
                     axis: np.ndarray, stats: dict) -> tuple[np.ndarray, dict] | None:
    """Re-derive every level from reliable anchors and a smooth position model.

    The rank-based renumbering assumes the model's components are, one for one,
    the vertebrae. On a scan where the network both splits vertebrae into
    several labels and merges neighbours under one label, that assumption is
    false and the components carry no usable count. What survives on such a
    scan is that *some* levels are still right: the lumbar and the upper spine
    are typically clean, and the mess sits in between.

    So the method is: find the levels the model got right, fit a smooth curve
    of position against level index through them, read off where every other
    level must be, and cut the spine mask into slabs at the midpoints. Anchor
    bodies keep their own voxels and their own boundary, because where the
    model is right its boundary beats a planar cut. Everything else is assigned
    by slab. A merged pair sitting across two expected positions is cut in
    half, which is the only correct thing that can be done to it without a
    vertebra-level segmenter.

    A planar cut misassigns the posterior elements, since a thoracic spinous
    process slopes down beside the body below. Growing labels through the mask
    from seeds instead (a marker-based watershed on the distance transform)
    was tried and was worse: the model's mask is connected across the facet
    joints, so the flood leaks, and levels seeded by a whole anchor body
    swallow their neighbours while a point-seeded level can be starved to a
    few voxels. On the demo scan it turned one level into a 137 mm, 65 cm3
    region and another into 0.1 cm3. The slab is cruder and correct.

    Returns None when there are too few reliable levels to fit, in which case
    the caller falls back to the model's labels.
    """
    if not bodies:
        return None

    # Candidate per level: the largest body carrying that label. A label the
    # model used for more than one body is direct evidence that the model was
    # confused about that level, so such labels are never anchors, whatever
    # their largest body looks like. They are still assigned by slab.
    cand: dict[int, Body] = {}
    for b in bodies:
        lab = b.majority_label
        if lab not in cand or b.size > cand[lab].size:
            cand[lab] = b
    # A label is "single" only when the model used it for exactly one body,
    # however small the others are. This was relaxed once, to stop a 3.7 cm3
    # stray named T5 from demoting a correct T5, and the relaxation let a
    # 52 cm3 blob named T12 (in truth L1, with an 8 cm3 second piece) become
    # an anchor, the curve bent to it, and every level from L1 to T9 came out
    # one off. A second body under the same name is evidence of confusion at
    # that level, and it stays disqualifying. The cost is one demoted T5.
    counts = Counter(b.majority_label for b in bodies)
    single = {k for k, n in counts.items() if n == 1}
    levels = np.array(sorted(k for k in cand if k in single), dtype=np.float64)
    pos = np.array([cand[int(k)].axis_pos for k in levels])
    if len(levels) < ANCHOR_MIN_COUNT:
        return None

    # Position model: cubic if it stays monotone, else quadratic, else linear.
    def thr_fn(coef, x, r_in):
        pitch = np.abs(np.polyval(np.polyder(coef), x))
        mad = 1.4826 * float(np.median(np.abs(r_in))) if len(r_in) else 0.0
        return np.maximum(ANCHOR_RESID_FRACTION * pitch, 3.0 * mad)

    coef = inl = None
    for deg in (3, 2, 1):
        if len(levels) <= deg + 2:
            continue
        c, m = _robust_polyfit(levels, pos, deg, thr_fn)
        if _monotone(c):
            coef, inl = c, m
            break
    if coef is None:
        return None

    # Volume trend through the positional inliers, so that a merged pair (about
    # twice the expected volume) or a fragment is not used as an anchor even if
    # it happens to sit at the right height.
    vol = np.array([cand[int(k)].size for k in levels], dtype=np.float64)
    if inl.sum() >= 4:
        vcoef = np.polyfit(levels[inl], np.log(vol[inl] + 1.0), 2)
        vtrend = np.exp(np.polyval(vcoef, levels))
    else:
        vtrend = np.full(len(levels), float(np.median(vol[inl] if inl.any() else vol)))
    ratio = vol / np.maximum(vtrend, 1.0)
    # C2 is the one level a smooth volume trend cannot fit: the axis carries
    # the dens and is genuinely much larger than C3. Holding it to the trend
    # rejected a correct C2 and cut into C1. It gets its own allowance.
    hi_lim = np.where(levels == 23, ANCHOR_VOLUME_HI_C2, ANCHOR_VOLUME_HI)
    # A body taller along the spine than a couple of pitches is two vertebrae
    # under one name. Clean vertebrae, processes included, run 1.3 to 2.4
    # pitches tall on the demo scans, and merged pairs run about 3.
    pitch_here = np.abs(np.polyval(np.polyder(coef), levels))
    ext = np.array([cand[int(k)].extent for k in levels])
    not_merged = ext < ANCHOR_EXTENT_MAX * np.maximum(pitch_here, 1.0)
    anchor_mask = (inl & not_merged
                   & (ratio < hi_lim) & (ratio > ANCHOR_VOLUME_LO))
    anchors = {int(k) for k, ok in zip(levels, anchor_mask) if ok}
    if len(anchors) < ANCHOR_MIN_COUNT:
        return None

    # Expected position of every level, and which levels are inside the scan.
    ks = np.arange(1, N_LABELS + 1, dtype=np.float64)
    expected = np.polyval(coef, ks)
    pitch_at = np.abs(np.polyval(np.polyder(coef), ks))
    fg_idx = np.argwhere(seg > 0)
    s_all = voxel_to_world(fg_idx, affine) @ axis
    lo_s, hi_s = float(s_all.min()), float(s_all.max())
    present = (expected > lo_s - 0.5 * pitch_at) & (expected < hi_s + 0.5 * pitch_at)
    present_levels = ks[present].astype(int)
    if len(present_levels) < 2:
        return None
    present_pos = expected[present]

    # Assign. Anchors keep their own voxels, every other foreground voxel goes
    # to the nearest expected position among the levels present.
    out = np.zeros_like(seg)
    for k in anchors:
        b = cand[k]
        out[b.voxels[:, 0], b.voxels[:, 1], b.voxels[:, 2]] = k
    remaining = (seg > 0) & (out == 0)
    rem_idx = np.argwhere(remaining)
    order = np.argsort(present_pos)
    pp = present_pos[order]
    pl = present_levels[order]

    # Boundaries between consecutive levels. The midpoint is the fallback. The
    # better cut is at the waist of the mask: the network labels bone and not
    # disc, so the union of all vertebra labels thins out at every disc, and
    # the cross-sectional area along the spine axis dips there. Cutting at the
    # dip follows the real boundary. Cutting at the midpoint assumes the two
    # bodies are the same height, which they are not.
    bounds = 0.5 * (pp[1:] + pp[:-1])

    def snap_to_waists(bounds):
        # Move each cut to the nearest clear waist of the mask, if one exists
        # within the search window. A dip has to be real (15 per cent below the
        # local plateau) or the cut stays where the prior put it.
        lo_b, hi_b = float(s_all.min()) - 1.0, float(s_all.max()) + 1.0
        edges = np.arange(lo_b, hi_b + 1.0, 1.0)
        area, _ = np.histogram(s_all, bins=edges)
        area = ndimage.gaussian_filter1d(area.astype(np.float64), sigma=2.0)
        centres = 0.5 * (edges[1:] + edges[:-1])
        out_b = bounds.copy()
        for i in range(len(out_b)):
            pitch_i = pp[i + 1] - pp[i]
            w = CUT_SEARCH_FRACTION * pitch_i
            sel = (centres >= out_b[i] - w) & (centres <= out_b[i] + w)
            if sel.sum() >= 3:
                seg_area = area[sel]
                if seg_area.min() < 0.85 * seg_area.max():
                    out_b[i] = float(centres[sel][int(np.argmin(seg_area))])
        return out_b

    if CUT_MODE == "disc" and len(pp) > 1:
        bounds = snap_to_waists(bounds)

    if CUT_MODE == "volume" and len(rem_idx) and len(pp) > 1:
        # Between two anchors, place the cuts so that each rebuilt level gets
        # its expected share of the volume that lies between them. Position
        # interpolation assumes the levels are evenly spaced, and on the demo
        # scan that put every rebuilt level about 10 mm too low, because this
        # patient's upper thoracic vertebrae are larger than a smooth curve
        # through the ends expects. Volume per level is the steadier
        # invariant: a tall vertebra takes a longer stretch of the axis, and
        # the cumulative volume along the axis says exactly where it ends.
        s_rem = voxel_to_world(rem_idx, affine) @ axis
        order_rem = np.argsort(s_rem)
        s_sorted = s_rem[order_rem]
        vt_all = np.exp(np.polyval(vcoef, ks)) if inl.sum() >= 4 else np.full(N_LABELS, 1.0)
        anchor_sorted = sorted(anchors)
        # Runs of consecutive non-anchor levels bounded by anchors on both sides.
        for a, b in zip(anchor_sorted[:-1], anchor_sorted[1:]):
            gap_levels = [k for k in range(a + 1, b) if k in set(present_levels.tolist())]
            if not gap_levels:
                continue
            lo_s = cand[a].axis_pos
            hi_s = cand[b].axis_pos
            in_gap = (s_sorted > lo_s) & (s_sorted < hi_s)
            n_gap = int(in_gap.sum())
            if n_gap < 10:
                continue
            # The anchors take nothing from the pool by quota. They already
            # own their own voxels, and an expected-volume quota gave T4 five
            # cubic centimetres of T5's body because the trend over-predicted
            # T4. The pool between two anchors is therefore shared among the
            # rebuilt levels only, and the boundary with each anchor is the
            # midpoint between the anchor's centroid and the neighbouring
            # rebuilt level's expected position.
            first_k, last_k = gap_levels[0], gap_levels[-1]
            lo_edge = 0.5 * (lo_s + float(expected[first_k - 1]))
            hi_edge = 0.5 * (hi_s + float(expected[last_k - 1]))
            in_pool = (s_sorted > lo_edge) & (s_sorted < hi_edge)
            if int(in_pool.sum()) < 10:
                continue
            weights = vt_all[np.array(gap_levels) - 1]
            if weights.sum() <= 0:
                continue
            cum = np.cumsum(weights) / weights.sum()
            inner = np.quantile(s_sorted[in_pool], cum[:-1]) if len(gap_levels) > 1 else np.array([])
            cut_s = np.concatenate([[lo_edge], inner, [hi_edge]])
            for i, k in enumerate(gap_levels):
                lo_c, hi_c = cut_s[i], cut_s[i + 1]
                # index of the boundary in 'bounds' that precedes level k
                ki = int(np.searchsorted(pl, k))
                if 0 < ki < len(pl):
                    bounds[ki - 1] = lo_c
                if 0 <= ki < len(pl) - 1:
                    bounds[ki] = hi_c
        bounds = np.sort(bounds)
        # The volume prior says roughly where each cut belongs. Where the mask
        # shows a real waist near that estimate, the waist is the disc and the
        # cut goes there: the prior handles the merged blob that has no waist,
        # the waist handles the levels where the prior drifts.
        bounds = np.sort(snap_to_waists(bounds))

    if len(rem_idx):
        s = voxel_to_world(rem_idx, affine) @ axis
        j = np.searchsorted(bounds, s)          # 0..len(pp)-1
        chosen = pl[np.clip(j, 0, len(pl) - 1)]
        out[rem_idx[:, 0], rem_idx[:, 1], rem_idx[:, 2]] = chosen

    # A planar cut through a merged body can strand a sliver of one level on
    # the far side of the cut, disconnected from the rest of that level. Hand
    # every such piece to whichever neighbouring label touches it most, so the
    # output keeps one component per level, which is what the audit checks and
    # what a reader would expect to see.
    out = _reattach_fragments(out)

    if REFINE_PROCESSES:
        rebuilt = {int(k) for k in present_levels if int(k) not in anchors}
        out = _refine_processes(out, seg > 0, affine, axis, rebuilt)
        out = _reattach_fragments(out)

    info = {
        "anchor_levels": [CLASS_MAP[k] for k in sorted(anchors)],
        "n_anchors": len(anchors),
        "levels_present": [CLASS_MAP[int(k)] for k in present_levels],
        "position_model_degree": int(len(coef) - 1),
        "rejected_levels": sorted(
            [CLASS_MAP[int(k)] for k, ok in zip(levels, anchor_mask) if not ok]
            + [CLASS_MAP[k] for k in cand if k not in single],
            key=lambda n: [v for v, nm in CLASS_MAP.items() if nm == n][0]),
        "slab_assigned_voxels": int(len(rem_idx)),
    }
    return out, info


# --------------------------------------------------------------------------
# Stage 6 (optional): an absolute anchor from neighbouring anatomy
# --------------------------------------------------------------------------

# Vertebral levels of two landmarks that AbdomenAtlas already annotates. Both
# are textbook averages with real population spread, which is exactly why they
# are used below to flag disagreement rather than to overrule the model.
#   coeliac trunk origin   ~ T12   (label index 6)
#   aortic bifurcation     ~ L4    (label index 2)
LANDMARK_LEVELS = {"celiac_trunk": 6, "aorta_bifurcation": 2}


def dominant_axis(affine: np.ndarray) -> int:
    """Array axis most aligned with the world superior direction."""
    return int(np.argmax(np.abs(affine[2, :3])))


def aortic_bifurcation_point(aorta: np.ndarray, affine: np.ndarray,
                             min_run: int = 4) -> np.ndarray | None:
    """World point where the two common iliac arteries become one aorta.

    Below the bifurcation an axial slice cuts two iliac arteries, and above it
    one aorta. The search runs from the pelvis upward rather than from the top down,
    which matters: the aorta class also covers the thoracic aorta, so an axial
    slice through the chest cuts the ascending and descending aorta and shows
    two components as well. Coming down from the head, that thoracic pair is
    found first and mistaken for the bifurcation, several levels too high.
    Coming up from the pelvis, the first sustained single-component run is the
    bifurcation and nothing above it can be confused for it.

    Returns None when the field of view stops before the iliacs, which is the
    correct answer rather than a guess.
    """
    ax = dominant_axis(affine)
    n = aorta.shape[ax]
    superior_is_higher_index = affine[2, ax] > 0

    order = range(n) if superior_is_higher_index else range(n - 1, -1, -1)
    split_run, run, bif_idx = 0, 0, None
    seen_split = False
    for idx in order:
        sl = np.take(aorta, idx, axis=ax)
        if not sl.any():
            run = 0
            continue
        _, ncomp = ndimage.label(sl)
        if ncomp >= 2:
            # The two-vessel stretch has to persist too. One ragged slice at the
            # very bottom of the mask is not a bifurcation, and accepting it
            # would place the landmark at the edge of the field of view.
            split_run += 1
            if split_run >= min_run:
                seen_split = True
            run = 0
        elif ncomp == 1 and seen_split:
            run += 1
            if run >= min_run:
                # Step back to where the merge actually happened.
                bif_idx = idx - (min_run - 1) if superior_is_higher_index \
                    else idx + (min_run - 1)
                break
    if bif_idx is None:
        return None
    bif_idx = int(np.clip(bif_idx, 0, n - 1))

    # Take the vessel's own centroid within that slice, not the image centre, so
    # the point can be projected onto the spine axis like any other landmark.
    sl = np.take(aorta, bif_idx, axis=ax)
    coords2d = np.argwhere(sl)
    if not len(coords2d):
        return None
    full = np.zeros((len(coords2d), 3), dtype=np.int64)
    other = [a for a in range(3) if a != ax]
    full[:, ax] = bif_idx
    full[:, other[0]] = coords2d[:, 0]
    full[:, other[1]] = coords2d[:, 1]
    return voxel_to_world(full, affine).mean(axis=0)


def load_landmark_masks(organ_dir: str) -> dict:
    """Read the landmark organs from a folder of per-organ NIfTI masks.

    Returns {organ: (bool array, affine)}. Each mask keeps its own affine,
    because organ masks are usually stored on the uncropped grid while the
    bodies were found in a cropped one. Both end up in the same world
    millimetres. Used by the standalone path; the ShapeKit adapter passes the
    masks it already holds in memory instead.
    """
    masks = {}
    if not organ_dir or not os.path.isdir(organ_dir):
        return masks
    for organ in ("celiac_trunk", "aorta"):
        path = os.path.join(organ_dir, organ + ".nii.gz")
        if os.path.isfile(path):
            img = nib.load(path)
            masks[organ] = (np.asarray(img.dataobj) > 0, img.affine)
    return masks


def landmark_offsets(organ_masks: dict, bodies: list[Body], axis: np.ndarray,
                     stats: dict) -> dict:
    """Offsets implied by organ landmarks, as label_index - rank.

    `organ_masks` maps "celiac_trunk" and/or "aorta" to (mask, affine). Each
    mask is mapped with its own affine on purpose: using the cropped
    prediction affine here displaces the landmark by the crop offset, which
    silently turns a correctly named spine into a six-level disagreement.
    """
    found = {}
    if not organ_masks:
        return found

    positions = {}
    if "celiac_trunk" in organ_masks:
        arr, caff = organ_masks["celiac_trunk"]
        arr = np.asarray(arr) > 0
        if arr.any():
            pts = voxel_to_world(np.argwhere(arr), caff)
            positions["celiac_trunk"] = float(np.dot(pts.mean(axis=0), axis))

    if "aorta" in organ_masks:
        arr, aaff0 = organ_masks["aorta"]
        arr = np.asarray(arr) > 0
        if arr.any():
            # Crop to the vessel before scanning slice by slice. The aorta fills
            # a thin column of the scan, so labelling full 512 x 512 slices is
            # almost all empty background.
            aidx = np.argwhere(arr)
            alo = aidx.min(axis=0)
            ahi = aidx.max(axis=0) + 1
            acrop = tuple(slice(int(x), int(y)) for x, y in zip(alo, ahi))
            aaff = np.asarray(aaff0, dtype=np.float64).copy()
            aaff[:3, 3] = aaff[:3, 3] + aaff[:3, :3] @ alo.astype(np.float64)
            point = aortic_bifurcation_point(arr[acrop], aaff)
            if point is not None:
                positions["aorta_bifurcation"] = float(np.dot(point, axis))

    for name, pos in positions.items():
        nearest = min(bodies, key=lambda b: abs(b.axis_pos - pos))
        found[name] = {
            "expected_label": LANDMARK_LEVELS[name],
            "nearest_rank": nearest.rank,
            "implied_offset": LANDMARK_LEVELS[name] - nearest.rank,
        }
    if found:
        stats["landmarks"] = found
    return found


# --------------------------------------------------------------------------
# Per-case driver
# --------------------------------------------------------------------------

def refine_case(seg: np.ndarray, affine: np.ndarray,
                ct: np.ndarray | None = None,
                organ_masks: dict | None = None,
                trust_anchor: bool = False,
                prompt_model: str = "none",
                prompt_device: str = "cuda:0",
                log=print) -> tuple[np.ndarray, dict]:
    """Run the whole refinement on one combined label volume.

    seg          uint8 array, 0 background, 1 = L5 ... 24 = C1
    affine       4x4 voxel-to-world matrix of `seg`
    ct           CT on the same grid as `seg`, or None. Enables the bone
                 check and the prompted boundaries.
    organ_masks  {"celiac_trunk": (mask, affine), "aorta": (mask, affine)}
                 or None. Used to flag a spine that is named consistently one
                 level off; only acted on when trust_anchor is True.
    prompt_model "none" or "nninteractive"
    Returns (refined seg on the full grid, stats dict).
    """
    stats = {
        "islands_removed": 0, "island_voxels_removed": 0, "labels_fragmented": 0,
        "duplicates_merged": 0, "missing_levels_inferred": 0,
        "order_violations_fixed": 0, "labels_renumbered": 0,
        "voxels_relabelled": 0, "bodies_found": 0,
        "non_bone_voxels_trimmed": 0, "slivers_absorbed": 0, "warnings": [],
    }
    vox_mm3 = voxel_volume_mm3(affine)
    stats["input_voxels"] = int((seg > 0).sum())

    # Work inside the spine's bounding box. The connected-component pass runs
    # once per label, and on a 512 x 512 x 1394 scan the spine occupies a few
    # percent of the volume, so scanning the whole array 24 times spends almost
    # all of its time on air. Cropping keeps the result identical: the affine is
    # translated to match, so every world coordinate is unchanged.
    full_shape = seg.shape
    ct_full = ct
    affine_full = affine
    fg_idx = np.argwhere(seg > 0)
    if not len(fg_idx):
        stats["warnings"].append("no vertebrae found, volume returned unchanged")
        return seg.copy(), stats
    lo = np.maximum(fg_idx.min(axis=0) - 2, 0)
    hi = np.minimum(fg_idx.max(axis=0) + 3, np.array(full_shape))
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg[crop]
    affine = affine.copy()
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ lo.astype(np.float64)

    # The bone check runs after the crop, not before. It only ever removes
    # voxels, so the bounding box computed from the untrimmed mask still
    # contains everything, and every operation inside it, in particular the
    # dilation of the mask complement, then costs a fraction of a full-volume
    # pass.
    if ct is not None:
        seg = trim_non_bone(seg, ct[crop], stats)

    bodies = extract_bodies(seg, vox_mm3, stats)
    stats["bodies_found"] = len(bodies)
    if not bodies:
        stats["warnings"].append("no vertebrae found, volume returned unchanged")
        full = np.zeros(full_shape, dtype=seg.dtype)
        full[crop] = seg
        return full, stats

    world = {}
    for b in bodies:
        world[id(b)] = voxel_to_world(b.voxels, affine)
        b.centroid_mm = world[id(b)].mean(axis=0)

    axis = spine_axis(bodies)
    for b in bodies:
        proj = world[id(b)] @ axis
        b.axis_pos = float(proj.mean())
        b.axis_lo, b.axis_hi = float(proj.min()), float(proj.max())
    bodies.sort(key=lambda b: b.axis_pos)

    # Snapshot the cleaned components with their original names before any
    # merging. Merging is only meaningful as a step toward re-deriving the
    # numbering: it folds two names into one, so if the renumbering is later
    # refused, replaying the merged list would delete labels and leave holes in
    # the run. The conservative path needs this untouched copy.
    raw_output = [(b.voxels, int(b.majority_label)) for b in bodies]
    raw_bodies = [Body(voxels=b.voxels, votes=Counter(b.votes), centroid_mm=b.centroid_mm,
                       axis_pos=b.axis_pos, axis_lo=b.axis_lo, axis_hi=b.axis_hi)
                  for b in bodies]

    bodies = merge_duplicates(bodies, stats)
    bodies = absorb_slivers(bodies, stats)
    bodies.sort(key=lambda b: b.axis_pos)

    # Count how badly the original naming disagreed with the geometry, before
    # we fix it. This is the number worth reporting: it is the error the
    # network makes that no amount of per-voxel accuracy would reveal.
    original_order = [b.majority_label for b in bodies]
    stats["order_violations_fixed"] = sum(
        1 for i in range(len(original_order) - 1)
        if original_order[i] >= original_order[i + 1])

    span = assign_ranks(bodies, stats)

    # Decide how to resolve the naming. When every level is a single body in
    # a consistent order the rank renumbering is exact and cheap. When the
    # model has split and merged vertebrae, the components carry no usable
    # count, and the only thing that can be trusted is the clean part of the
    # spine. That case goes to the anchor-and-slab resolver.
    # A swap or a dropped level with one body per label is exact under rank
    # renumbering. Only a spine where labels repeat, or where more bodies than
    # levels exist, needs the resolver.
    one_body_per_label = len(raw_bodies) == len({b.majority_label for b in raw_bodies})
    clean = (span <= N_LABELS and stats["duplicates_merged"] == 0
             and one_body_per_label)
    renumber = clean
    offset = None
    slab_result = None
    if not clean:
        slab_result = resolve_by_slabs(raw_bodies, seg, affine, axis, stats)
        if slab_result is None:
            stats["warnings"].append(
                f"{span} levels implied but a spine has {N_LABELS}, and too few "
                f"reliable levels to fit a position model, kept the model's labels")
            stats["needs_review"] = True
        else:
            stats["resolver"] = "anchors_and_slabs"
            stats.update({f"slab_{k}": v for k, v in slab_result[1].items()})
    if renumber:
        offset = choose_offset(bodies, span, stats)

    # Absolute anchoring. The offset chosen above is the one the model's own
    # votes support, which cannot detect a spine that is consistently named one
    # level off. Organ landmarks can, but they carry population spread of about
    # a level themselves, so by default a disagreement is reported rather than
    # acted on. Flagging a case for a human is the honest output when two weak
    # sources of truth conflict. Silently picking one is not.
    if organ_masks and renumber and offset is not None:
        marks = landmark_offsets(organ_masks, bodies, axis, stats)
        implied = [m["implied_offset"] for m in marks.values()]
        if implied:
            consensus = int(round(float(np.median(implied))))
            stats["anchor_offset"] = consensus
            stats["model_offset"] = offset
            stats["anchor_disagreement"] = consensus - offset
            stats["needs_review"] = bool(consensus != offset)
            if trust_anchor and consensus != offset:
                lo = 1 - min(b.rank for b in bodies)
                hi = N_LABELS - max(b.rank for b in bodies)
                if lo <= consensus <= hi:
                    offset = consensus
                    stats["anchor_applied"] = True
                else:
                    stats["warnings"].append(
                        "landmark offset would push labels out of range, kept model offset")

    out = np.zeros_like(seg)
    if renumber:
        for b in bodies:
            label = int(np.clip(offset + b.rank, 1, N_LABELS))
            b.final_label = label
            if label != b.majority_label:
                stats["labels_renumbered"] += 1
                stats["voxels_relabelled"] += b.size
            out[b.voxels[:, 0], b.voxels[:, 1], b.voxels[:, 2]] = label
        present = sorted({b.final_label for b in bodies})
    elif slab_result is not None:
        out = slab_result[0]
        changed = (out != seg) & (seg > 0)
        stats["voxels_relabelled"] = int(changed.sum())
        stats["labels_renumbered"] = int(len(slab_result[1]["rejected_levels"]))
        present = sorted(int(v) for v in np.unique(out) if v > 0)
    else:
        # Islands and soft-tissue leakage are still removed, and only the naming is
        # left exactly as the model produced it.
        for voxels, label in raw_output:
            out[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = label
        present = sorted({label for _, label in raw_output})

    # Learned boundaries for the rebuilt levels. The resolver has named them
    # and placed them; a promptable model draws them. Anchors keep the
    # network's own boundary, which scores in the nineties against
    # human-revised truth, so the learned boundary goes only where the
    # network's was lost. See prompted_refine.py. This runs on the cropped
    # arrays: the envelope the masks are held to lies inside the crop, and
    # the fragment pass afterwards is a per-label component scan that costs
    # minutes on the full volume and seconds on the crop.
    if (slab_result is not None and prompt_model == "nninteractive"
            and ct_full is not None):
        try:
            from . import vertebrae_anchor_prompt as prompted_refine
            name_to_k = {v: k for k, v in CLASS_MAP.items()}
            rebuilt_levels = [name_to_k[n] for n in slab_result[1]["rejected_levels"]
                              if name_to_k[n] in present]
            axis_fn = lambda idx: voxel_to_world(idx, affine) @ axis
            out, prec = prompted_refine.refine_levels(
                np.asarray(ct_full[crop]), out, rebuilt_levels, axis_fn, log=log,
                device=prompt_device)
            summ = prec.pop("_summary", {})
            stats["prompted_levels_accepted"] = [CLASS_MAP[k] for k in summ.get("accepted", [])]
            stats["prompted_levels_rejected"] = [CLASS_MAP[k] for k in summ.get("rejected", [])]
            stats["prompt_record"] = {CLASS_MAP[k]: v for k, v in prec.items()}
            stats["prompt_seconds"] = summ.get("seconds")
            out = _reattach_fragments(out)
            present = sorted(int(v) for v in np.unique(out) if v > 0)
        except Exception as exc:  # the script must still finish without a GPU
            stats["warnings"].append(f"prompted refinement skipped: {exc}")

    # Only on a scan the resolver rebuilt. A scan the network already had right
    # is returned exactly as it came, cleaned of islands and leakage and nothing
    # else, and the staircase pass would gain a tenth of a point there at the
    # cost of that guarantee.
    if SMOOTH_STAIRCASE and slab_result is not None:
        out = smooth_staircase(out)
        # Smoothing can detach the tip of a thin process; hand any such piece
        # back to the label it touches, as after every other step.
        out = _reattach_fragments(out)
        present = sorted(int(v) for v in np.unique(out) if v > 0)

    full = np.zeros(full_shape, dtype=out.dtype)
    full[crop] = out
    out = full

    stats["levels_present"] = [CLASS_MAP[l] for l in present]
    stats["contiguous"] = bool(present == list(range(present[0], present[-1] + 1)))
    stats["kept_voxels"] = int((out > 0).sum())
    return out, stats
