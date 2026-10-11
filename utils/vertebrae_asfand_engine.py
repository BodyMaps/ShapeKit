"""ShapeKit-Asfand vertebrae engine: level counting from L5 with drift repair.

Mask-only (no CT), CPU-only. Operates on a combined label array with engine
ids 1 = L5 ... 24 = C1 (SuPreM convention) and the superior-inferior axis LAST.

Error this engine targets
-------------------------
Label drift: the model spreads N labels over N + 1 vertebral bodies (or the
reverse), so a run of bodies carries two labels each and every level in the
run is shifted by one. Example from AbdomenAtlas BDMAP_00000031: 12 thoracic
labels on 13 bodies; the L1 body was labelled T12 while its posterior
elements were labelled L1.

Method
------
1. Fragment removal: whole-spine components smaller than min_frag_ml.
2. Body centre line: per axial slice, the centroid of the deepest part of the
   mask (distance transform). The vertebral body is the thickest bone; the
   centroid of the whole slice is avoided because body + pedicles + lamina
   form a ring whose centre lies in the spinal canal. Median-filtered and
   smoothed along the axis.
3. Discs: the model leaves intervertebral discs unlabelled, so along the
   centre line the unlabelled fraction in a small disk peaks at each disc.
   Peaks split the column into one segment per vertebral body.
4. Counting: from the lowest body upward, each body gets the level it must
   have (anchor = the model's label of the lowest body). A body is trusted
   when the model's majority label equals the counted level with share >=
   trust_frac. Climbing stops after sync_run trusted bodies following a
   mismatch, or when a segment is implausibly long (disc detection no longer
   reliable). Nothing above that point is touched.
5. Relabel: bodies in the drifted range are seeded with their counted level
   and grown through the bone only (geodesic), so posterior elements follow
   their own body via the pedicles. Only voxels whose model label lies in the
   drifted range are rewritten.
6. Detached pieces: every label keeps its largest piece; other pieces of that
   label (<= orphan_max_ml) join the vertebra they touch most.

Related work: ShapeKit (Liu et al.) and Meng et al. (2022) for mask-based
vertebra post-processing; this engine adds disc-based level counting.
"""

from dataclasses import dataclass

import cc3d
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

LEVELS = [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
VAL = {name: 24 - i for i, name in enumerate(LEVELS)}  # engine ids: C1=24 ... L5=1
NAMES_BOTTOM_UP = [LEVELS[24 - v] for v in range(1, 25)]  # L5, L4, ..., C1


@dataclass
class Config:
    min_frag_ml: float = 2.0      # delete floating pieces smaller than this
    smooth_mm: float = 8.0        # smoothing of the body centre line
    radius_mm: float = 6.0        # disk radius around the centre line for the gap profile
    gap_height: float = 0.5       # min unlabelled fraction that marks a disc
    min_level_mm: float = 8.0     # minimum distance between discs
    trust_frac: float = 0.9       # body trusted if model majority share >= this
    sync_run: int = 3             # trusted bodies in a row that end the drifted range
    max_len_ratio: float = 1.7    # stop climbing if a segment is this x longer than the previous
    seed_mm: float = 12.0         # radius of the body column used as seeds
    disc_gap_mm: float = 1.5      # slab at each disc excluded from seeds
    max_iter: int = 400           # max geodesic growth steps
    orphan_max_ml: float = 10.0   # max size of a detached piece that is moved
    margin: int = 5               # crop margin in voxels


def lvl_name(v):
    return LEVELS[24 - int(v)] if 1 <= v <= 24 else "-"


def remove_fragments(comb, vox_ml, min_ml):
    union = comb > 0
    cc = cc3d.connected_components(union, connectivity=26)
    counts = np.bincount(cc.ravel())
    keep = counts * vox_ml >= min_ml
    keep[0] = False
    n_rm = int((~keep[1:]).sum())
    ml_rm = float(counts[1:][~keep[1:]].sum() * vox_ml)
    union = keep[cc]
    del cc
    return np.where(union, comb, 0).astype(np.uint8), union, n_rm, ml_rm


def body_centerline(u, spacing, smooth_mm):
    Z = u.shape[2]
    cx = np.full(Z, np.nan)
    cy = np.full(Z, np.nan)
    for z in range(Z):
        sl = u[:, :, z]
        if not sl.any():
            continue
        d = ndimage.distance_transform_edt(sl, sampling=spacing[:2])
        deep = d >= 0.8 * d.max()
        lab, n = ndimage.label(deep)
        if n > 1:
            best = int(np.argmax(ndimage.maximum(d, lab, index=np.arange(1, n + 1)))) + 1
            deep = lab == best
        cx[z], cy[z] = np.argwhere(deep).mean(0)
    ok = ~np.isnan(cx)
    zz = np.arange(Z)
    w = max(1, int(round(10.0 / spacing[2])))

    def robust(x):
        x = x.copy()
        x[~ok] = np.interp(zz[~ok], zz[ok], x[ok])
        x = ndimage.median_filter(x, size=2 * w + 1, mode="nearest")
        return ndimage.gaussian_filter1d(x, smooth_mm / spacing[2])

    return robust(cx), robust(cy), ok


def geodesic_grow(seeds, region, max_iter):
    lab = seeds.copy()
    free = region & (lab == 0)
    fp = np.ones((3, 3, 3), bool)
    for _ in range(max_iter):
        if not free.any():
            break
        grown = ndimage.grey_dilation(lab, footprint=fp)
        new = free & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
        free &= ~new
    return lab, free


def contact_counts(cc):
    """Face-contact counts between different components: dict (a, b) -> count, both directions."""
    out = {}
    for ax in range(3):
        a = np.moveaxis(cc, ax, 0)
        x, y = a[:-1], a[1:]
        m = (x != y) & (x > 0) & (y > 0)
        if not m.any():
            continue
        keys = x[m].astype(np.int64) * (1 << 32) + y[m].astype(np.int64)
        u, n = np.unique(keys, return_counts=True)
        for k, cnt in zip(u, n):
            p, q = int(k >> 32), int(k & 0xFFFFFFFF)
            out[(p, q)] = out.get((p, q), 0) + int(cnt)
            out[(q, p)] = out.get((q, p), 0) + int(cnt)
    return out


def reassign_orphans(lab, vox_ml, max_ml):
    """Every label keeps its largest piece; any other piece (<= max_ml) goes to the label it touches most."""
    cc = cc3d.connected_components(lab, connectivity=26)
    n = int(cc.max()) + 1
    size = np.bincount(cc.ravel(), minlength=n)
    owner = np.zeros(n, np.int64)
    owner[cc.ravel()] = lab.ravel()
    largest = {}
    for i in range(1, n):
        v = owner[i]
        if v not in largest or size[i] > size[largest[v]]:
            largest[v] = i
    orphans = [i for i in range(1, n) if largest[owner[i]] != i and size[i] * vox_ml <= max_ml]
    if not orphans:
        return lab, 0, 0.0
    cont = contact_counts(cc)
    lut = owner.copy()
    moved_ml = 0.0
    for i in orphans:
        score = {}
        for (p, q), k in cont.items():
            if p == i and owner[q] != owner[i]:
                score[owner[q]] = score.get(owner[q], 0) + k
        if score:
            lut[i] = max(score, key=score.get)
            moved_ml += size[i] * vox_ml
    return lut[cc].astype(np.uint8), len(orphans), moved_ml


def refine(comb, spacing, superior_up=True, cfg=None, log=print):
    """Refine a combined vertebra label array.

    comb: uint8 array, engine ids 1 (L5) ... 24 (C1), 0 background; the
        superior-inferior axis must be the LAST axis.
    spacing: voxel size in mm, same axis order as comb.
    superior_up: True if a higher index on the last axis is more superior.
    Returns (refined uint8 array of the same shape, report dict).
    """
    cfg = cfg or Config()
    spacing = tuple(float(s) for s in spacing)
    vox_ml = float(np.prod(spacing)) / 1000.0
    comb = np.asarray(comb).astype(np.uint8)
    report = dict(fragments_removed=0, fragments_ml=0.0, discs=0, bodies=[],
                  relabelled_bodies=0, rewritten_voxels=0, orphans_moved=0, orphans_ml=0.0)

    # 1. fragments
    comb, union, n_rm, ml_rm = remove_fragments(comb, vox_ml, cfg.min_frag_ml)
    report.update(fragments_removed=n_rm, fragments_ml=round(ml_rm, 3))
    log(f"[1] removed {n_rm} floating fragments ({ml_rm:.2f} mL)")
    out = comb.copy()
    if not union.any():
        return out, report

    idx = [np.where(union.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
    m = cfg.margin
    crop = tuple(slice(max(int(a[0]) - m, 0), min(int(a[-1]) + 1 + m, dim)) for a, dim in zip(idx, comb.shape))
    c = comb[crop].copy()
    u = union[crop]
    Z = c.shape[2]
    zz = np.arange(Z)

    # 2. centre line
    cx, cy, ok = body_centerline(u, spacing, cfg.smooth_mm)
    xx, yy = np.meshgrid(np.arange(c.shape[0]) * spacing[0], np.arange(c.shape[1]) * spacing[1], indexing="ij")

    def disk(z, r):
        return (xx - cx[z] * spacing[0]) ** 2 + (yy - cy[z] * spacing[1]) ** 2 <= r * r

    # 3. discs from unlabelled gaps along the centre line
    gap = np.zeros(Z)
    lab_c = np.zeros(Z, int)
    for z in zz[ok]:
        dk = disk(z, cfg.radius_mm)
        gap[z] = 1.0 - (c[:, :, z][dk] > 0).mean()
        lab_c[z] = c[int(round(cx[z])), int(round(cy[z])), z]
    lz = np.where(lab_c > 0)[0]
    info = []
    discs = np.array([], int)
    if len(lz):
        lo, hi = int(lz.min()), int(lz.max())
        pk, _ = find_peaks(gap[lo:hi + 1], height=cfg.gap_height,
                           distance=max(1, int(round(cfg.min_level_mm / spacing[2]))))
        discs = np.sort(pk + lo)
        bounds = np.r_[lo, discs, hi + 1]
        segs = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
        seg_bt = segs if superior_up else segs[::-1]  # bottom -> top anatomically
        for a, b in seg_bt:
            v = lab_c[a:b]
            v = v[v > 0]
            if len(v) < 2:
                continue
            h = np.bincount(v, minlength=25)
            maj = int(h.argmax())
            info.append(dict(z0=a, z1=b, maj=maj, share=h[maj] / len(v), h=h, len_mm=(b - a) * spacing[2]))
    report["discs"] = int(len(discs))
    log(f"[2-3] {len(discs)} discs from mask gaps along the body centre line")

    # 4. count from the bottom (anchor = model label of the lowest body)
    if info:
        anchor = info[0]["maj"]
        mism_seen = False
        run = 0
        plan = []
        prev_len = None
        for i, s in enumerate(info):
            exp = anchor + i  # label value increases upwards (L5=1 ... C1=24)
            if exp > 24:
                break
            if prev_len is not None and s["len_mm"] > cfg.max_len_ratio * prev_len:
                log(f"    stop: segment {s['len_mm']:.0f} mm is too long (disc detection unreliable above)")
                break
            trusted = s["maj"] == exp and s["share"] >= cfg.trust_frac
            plan.append((s, exp, trusted))
            prev_len = s["len_mm"]
            if not trusted:
                mism_seen = True
                run = 0
            else:
                run += 1
                if mism_seen and run >= cfg.sync_run:
                    break
        for s, exp, tr in plan:
            top = [int(k) for k in np.argsort(-s["h"])[:2] if s["h"][k] > 0]
            labels = {lvl_name(k): round(float(s["h"][k] / s["h"].sum()), 3) for k in top}
            report["bodies"].append(dict(length_mm=round(float(s["len_mm"]), 1), model=labels,
                                         counted=lvl_name(exp), status="ok" if tr else "relabel"))
            log(f"    body {s['len_mm']:5.0f} mm  model {labels}  counted {lvl_name(exp)}  "
                f"{'ok' if tr else 'RELABEL'}")

        bad_idx = [i for i, (s, exp, tr) in enumerate(plan) if not tr]
        if bad_idx:
            # 5. relabel the drifted range through the bone; labels that may be
            # rewritten: everything seen in the drifted range, widened by one
            # trusted body on each side
            i0, i1 = max(bad_idx[0] - 1, 0), min(bad_idx[-1] + 1, len(plan) - 1)
            drift_labels = set()
            for s, exp, tr in plan[i0:i1 + 1]:
                drift_labels.add(exp)
                drift_labels.update(int(k) for k in np.nonzero(s["h"])[0] if k > 0)
            region = np.isin(c, sorted(drift_labels)) & u
            seeds = np.zeros(c.shape, np.int32)
            gapz = max(1, int(round(cfg.disc_gap_mm / spacing[2])))
            for s, exp, tr in plan:
                for z in range(s["z0"] + gapz, s["z1"] - gapz):
                    if not ok[z]:
                        continue
                    col = disk(z, cfg.seed_mm) & u[:, :, z]
                    seeds[:, :, z][col] = exp
            grown, left = geodesic_grow(seeds, region | (seeds > 0), cfg.max_iter)
            new_c = c.copy()
            assign = region & (grown > 0)
            new_c[assign] = grown[assign]
            changed = int((new_c != c).sum())
            report.update(relabelled_bodies=len(bad_idx), rewritten_voxels=changed)
            log(f"[4-5] relabelled {len(bad_idx)} drifted bodies; rewritten {changed * vox_ml:.1f} mL")
            out[crop] = new_c
        else:
            log("[4-5] counted levels agree with the model: no relabelling needed")

    # 6. detached pieces: each label keeps one piece; detached bits join the vertebra they touch most
    cleaned, n_orph, ml_orph = reassign_orphans(out[crop], vox_ml, cfg.orphan_max_ml)
    for _ in range(2):  # a moved piece can itself become detached; repeat until stable
        cleaned, n2, ml2 = reassign_orphans(cleaned, vox_ml, cfg.orphan_max_ml)
        n_orph, ml_orph = n_orph + n2, ml_orph + ml2
        if n2 == 0:
            break
    out[crop] = cleaned
    report.update(orphans_moved=int(n_orph), orphans_ml=round(float(ml_orph), 3))
    log(f"[6] {n_orph} detached pieces moved to the vertebra they touch most ({ml_orph:.1f} mL)")
    return out, report
