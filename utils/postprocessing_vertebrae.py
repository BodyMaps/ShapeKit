"""
Postprocessing of predicted vertebrae masks (labels 1 = L5 ... 24 = C1).

Steps: remove stray blobs, smooth label boundaries, merge label fragments,
repair vertebrae whose body is split between two labels, check the order.

python postprocessing_vertebrae.py --pred_dir ./AbdomenAtlasDemoPredict --save_dir ./AbdomenAtlasDemoPostprocessed
"""
import argparse
import os

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

NUM_VERTEBRAE = 24
VERTEBRAE = (
    ["vertebrae_L%d" % i for i in range(5, 0, -1)]
    + ["vertebrae_T%d" % i for i in range(12, 0, -1)]
    + ["vertebrae_C%d" % i for i in range(7, 0, -1)]
)


def foreground_bbox(mask, margin, shape):
    slices = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.flatnonzero(mask.any(axis=other))
        slices.append(slice(max(int(idx[0]) - margin, 0), min(int(idx[-1]) + margin + 1, shape[ax])))
    return tuple(slices)


def remove_stray_components(lab, spacing, min_component_ml, max_gap_mm):
    # keep the largest component (the spine) and big components close to it
    vox_ml = float(np.prod(spacing)) / 1000.0
    cc, n = cc3d.connected_components((lab > 0).astype(np.uint8), connectivity=26, return_N=True)
    if n <= 1:
        return 0.0
    counts = cc3d.statistics(cc)["voxel_counts"]
    sp = np.asarray(spacing, dtype=np.float64)

    def points_mm(ids, max_points):
        pts = np.argwhere(np.isin(cc, ids))
        return pts[:: max(1, len(pts) // max_points)] * sp

    keep = [int(np.argmax(counts[1:])) + 1]
    candidates = [c for c in range(1, n + 1) if c != keep[0] and counts[c] * vox_ml >= min_component_ml]
    grew = True
    while grew and candidates:
        grew = False
        tree = cKDTree(points_mm(keep, 200000))
        for c in list(candidates):
            dist, _ = tree.query(points_mm([c], 2000))
            if dist.min() <= max_gap_mm:
                keep.append(c)
                candidates.remove(c)
                grew = True

    keep_lut = np.zeros(n + 1, dtype=bool)
    keep_lut[keep] = True
    stray = (cc > 0) & ~keep_lut[cc]
    removed_ml = float(stray.sum()) * vox_ml
    lab[stray] = 0
    return removed_ml


def smooth_label_boundaries(lab, spacing, window_mm):
    # majority vote of the labels in a small window
    size = []
    for s in spacing:
        n = max(1, int(round(window_mm / s)))
        size.append(n if n % 2 == 1 else n + 1)
    pad = [n // 2 for n in size]

    best = np.zeros(lab.shape, dtype=np.float32)
    out = np.zeros(lab.shape, dtype=np.uint8)
    objects = ndimage.find_objects(lab, max_label=NUM_VERTEBRAE)
    for k, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        sl = tuple(slice(max(o.start - p, 0), min(o.stop + p, d)) for o, p, d in zip(obj, pad, lab.shape))
        votes = ndimage.uniform_filter((lab[sl] == k).astype(np.float32), size=size, mode="constant")
        best_view = best[sl]
        better = votes > best_view
        best_view[better] = votes[better]
        out[sl][better] = k
    out[lab == 0] = 0
    return out


def merge_fragments(lab, spacing, min_island_ml, max_passes=3):
    # each label keeps its largest piece, the rest goes to the label it touches most
    vox_ml = float(np.prod(spacing)) / 1000.0
    struct = np.ones((3, 3, 3), dtype=bool)
    n_changed = 0
    for _ in range(max_passes):
        cc, n = cc3d.connected_components(lab, connectivity=26, return_N=True)
        stats = cc3d.statistics(cc)
        counts, boxes = stats["voxel_counts"], stats["bounding_boxes"]
        comp_label = np.zeros(n + 1, dtype=np.uint8)
        comp_label[cc.ravel()] = lab.ravel()

        main = {}
        for cid in range(1, n + 1):
            k = int(comp_label[cid])
            if k not in main or counts[cid] > counts[main[k]]:
                main[k] = cid
        fragments = [cid for cid in range(1, n + 1) if main[int(comp_label[cid])] != cid]
        fragments.sort(key=lambda cid: counts[cid], reverse=True)

        changed = 0
        for cid in fragments:
            sl = tuple(slice(max(b.start - 1, 0), min(b.stop + 1, d)) for b, d in zip(boxes[cid], lab.shape))
            piece = cc[sl] == cid
            ring = ndimage.binary_dilation(piece, structure=struct) & ~piece
            touching = lab[sl][ring]
            touching = touching[touching > 0]
            if touching.size:
                new_label = int(np.bincount(touching, minlength=NUM_VERTEBRAE + 1).argmax())
            elif counts[cid] * vox_ml < min_island_ml:
                new_label = 0
            else:
                continue
            if new_label != comp_label[cid]:
                lab[sl][piece] = new_label
                changed += 1
        n_changed += changed
        if changed == 0:
            break
    return n_changed


def grow_seeds(seeds, region, spacing, max_iter=1000):
    # region growing from the seeds, one voxel layer per iteration
    cur = seeds.copy()
    todo = region & (cur == 0)
    # coarse axes advance less often (anisotropic spacing)
    steps = [max(1, int(round(s / min(spacing)))) for s in spacing]
    stalled = 0
    for it in range(1, max_iter + 1):
        if not todo.any():
            break
        grown = cur
        for ax in range(3):
            if it % steps[ax] == 0:
                grown = ndimage.maximum_filter1d(grown, size=3, axis=ax, mode="constant")
        new = todo & (grown > 0)
        if not new.any():
            stalled += 1
            if stalled >= max(steps):
                break
            continue
        stalled = 0
        cur[new] = grown[new]
        todo &= ~new
    return cur


def repair_split_vertebrae(lab, spacing, affine, origin, core_fraction, core_window_mm, purity):
    # vertebral bodies = thickest parts of the mask (distance transform, local threshold)
    dist = ndimage.distance_transform_edt(lab > 0, sampling=spacing).astype(np.float32)
    sa = int(np.argmax(np.abs(affine[2, :3])))  # inferior-superior axis
    profile = dist.max(axis=tuple(a for a in range(3) if a != sa))
    half = max(1, int(round(core_window_mm / spacing[sa])))
    envelope = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(profile, size=2 * half + 1), sigma=half / 2.0)
    shape = [1, 1, 1]
    shape[sa] = -1
    threshold = np.maximum(core_fraction * envelope, 2.0).reshape(shape)
    cores, n = cc3d.connected_components((dist >= threshold).astype(np.uint8), connectivity=26, return_N=True)
    del dist
    if n < 3:
        print("  body repair: fewer than 3 vertebral bodies found, skipped")
        return 0

    stats = cc3d.statistics(cores)
    vox_ml = float(np.prod(spacing)) / 1000.0
    volume = stats["voxel_counts"] * vox_ml
    height = np.array([affine[2, :3] @ (np.asarray(c, dtype=np.float64) + np.asarray(origin)) + affine[2, 3]
                       for c in stats["centroids"]])
    ids = sorted(range(1, n + 1), key=lambda c: height[c])
    # ignore tiny cores
    kept = []
    for i, c in enumerate(ids):
        around = [volume[j] for j in ids[max(0, i - 3):i] + ids[i + 1:i + 4]]
        if not around or volume[c] >= 0.08 * float(np.median(around)):
            kept.append(c)
    ids = kept

    inside = cores > 0
    hist = np.bincount(cores[inside].astype(np.int64) * (NUM_VERTEBRAE + 1) + lab[inside],
                       minlength=(n + 1) * (NUM_VERTEBRAE + 1)).reshape(n + 1, NUM_VERTEBRAE + 1)
    per_label = hist[ids].sum(axis=0)
    dominant, pure, exclusive = [], [], []
    for c in ids:
        d = int(hist[c, 1:].argmax()) + 1
        dominant.append(d)
        pure.append(hist[c, d] / max(1, hist[c].sum()))
        exclusive.append(hist[c, d] / max(1, per_label[d]))
    clean = [p >= purity and e >= purity for p, e in zip(pure, exclusive)]

    # body rank + offset = label, offset voted by the cleanly labelled bodies
    offsets = [d - r for r, (d, ok) in enumerate(zip(dominant, clean)) if ok]
    if len(offsets) < 3:
        print("  body repair: fewer than 3 cleanly labelled bodies, skipped")
        return 0
    values, votes = np.unique(offsets, return_counts=True)
    offset = int(values[votes.argmax()])
    anchor = [ok and d - r == offset for r, (d, ok) in enumerate(zip(dominant, clean))]

    print("  body repair: %d bodies found, %d anchors" % (len(ids), sum(anchor)))
    print("    %4s %9s %7s %6s %7s %7s %9s" % ("body", "height_mm", "ml", "label", "purity", "exclus.", "expected"))
    for r, c in enumerate(ids):
        expected = r + offset
        name = VERTEBRAE[expected - 1][10:] if 1 <= expected <= NUM_VERTEBRAE else "?"
        print("    %4d %9.0f %7.1f %6s %7.2f %7.2f %9s %s" % (r, height[c], volume[c], VERTEBRAE[dominant[r] - 1][10:],
                                                           pure[r], exclusive[r], name, "" if anchor[r] else "<- not an anchor"))

    anchors = [r for r, a in enumerate(anchor) if a]
    n_repaired = 0
    for p, q in zip(anchors[:-1], anchors[1:]):
        if q - p < 2:
            continue
        a, b = p + offset, q + offset
        seed_lut = np.zeros(n + 1, dtype=np.uint8)
        for r in range(p, q + 1):
            seed_lut[ids[r]] = r + offset
        seeds = seed_lut[cores]
        region = ((lab >= a) & (lab <= b)) | (seeds > 0)
        box = foreground_bbox(region, 1, lab.shape)
        grown = grow_seeds(seeds[box], region[box], spacing)
        reached = region[box] & (grown > 0)
        lab[box][reached] = grown[reached]
        n_repaired += q - p - 1
        print("    regrown %s .. %s from their bodies" % (VERTEBRAE[a - 1][10:], VERTEBRAE[b - 1][10:]))
    outside = [r for r, a in enumerate(anchor) if not a and (r < anchors[0] or r > anchors[-1])]
    if outside:
        print("    left untouched (no anchor on one side): bodies %s" % ", ".join(str(r) for r in outside))
    return n_repaired


def label_positions(lab, affine, origin):
    # voxel count and height (world z of the centroid, mm) of each label
    stats = cc3d.statistics(lab)
    counts = np.zeros(NUM_VERTEBRAE + 1, dtype=np.int64)
    height = np.full(NUM_VERTEBRAE + 1, np.nan)
    n = min(len(stats["voxel_counts"]), NUM_VERTEBRAE + 1)
    counts[:n] = stats["voxel_counts"][:n]
    counts[0] = 0
    for k in range(1, n):
        if counts[k] > 0:
            centroid = np.asarray(stats["centroids"][k], dtype=np.float64) + np.asarray(origin)
            height[k] = float(affine[2, :3] @ centroid + affine[2, 3])
    return counts, height


def fix_adjacent_swaps(lab, affine, origin):
    # labels must increase from feet to head
    counts, height = label_positions(lab, affine, origin)
    present = [k for k in range(1, NUM_VERTEBRAE + 1) if counts[k] > 0]
    by_height = sorted(present, key=lambda k: height[k])
    if by_height == present:
        return "ok"
    if any(abs(present.index(k) - i) > 1 for i, k in enumerate(by_height)):
        return "NOT FIXED: labels are out of anatomical order by more than one position"
    lut = np.arange(NUM_VERTEBRAE + 1, dtype=np.uint8)
    for target, source in zip(present, by_height):
        lut[source] = target
    lab[...] = lut[lab]
    swapped = [VERTEBRAE[k - 1][10:] for k, t in zip(by_height, present) if k != t]
    return "swapped " + ", ".join(swapped)


def summarize(lab, affine, origin, spacing):
    vox_ml = float(np.prod(spacing)) / 1000.0
    cc, n = cc3d.connected_components(lab, connectivity=26, return_N=True)
    comp_label = np.zeros(n + 1, dtype=np.uint8)
    comp_label[cc.ravel()] = lab.ravel()
    pieces = np.bincount(comp_label[1:], minlength=NUM_VERTEBRAE + 1)
    counts, height = label_positions(lab, affine, origin)

    # touching labels that are not consecutive
    wrong_contacts = 0
    for ax in range(3):
        moved = np.moveaxis(lab, ax, 0)
        a, b = moved[:-1], moved[1:]
        touching = (a > 0) & (b > 0) & (a != b)
        wrong_contacts += int((np.abs(a[touching].astype(np.int16) - b[touching].astype(np.int16)) > 1).sum())

    present = [k for k in range(1, NUM_VERTEBRAE + 1) if counts[k] > 0]
    inversions = sum(1 for a, b in zip(present[:-1], present[1:]) if height[a] >= height[b])
    return {"ml": counts * vox_ml, "pieces": pieces, "wrong_contacts": wrong_contacts,
            "inversions": inversions, "present": present}


def size_flag(ml, k):
    neighbours = [ml[j] for j in (k - 2, k - 1, k + 1, k + 2) if 1 <= j <= NUM_VERTEBRAE and ml[j] > 0]
    if ml[k] <= 0 or len(neighbours) < 2:
        return ""
    ratio = ml[k] / float(np.median(neighbours))
    if ratio < 0.5:
        return "<- much smaller than neighbours"
    if ratio > 1.7:
        return "<- much larger than neighbours"
    return ""


def print_report(before, after, removed_ml, n_merged, order_msg):
    print("  stray volume removed            : %.2f ml" % removed_ml)
    print("  fragments relabelled or dropped : %d" % n_merged)
    print("  order check                     : %s" % order_msg)
    print("  %-34s %10s %10s" % ("", "before", "after"))
    print("  %-34s %10d %10d" % ("labels present", len(before["present"]), len(after["present"])))
    print("  %-34s %10d %10d" % ("extra pieces (should be 0)",
                                 int(sum(before["pieces"][1:]) - len(before["present"])),
                                 int(sum(after["pieces"][1:]) - len(after["present"]))))
    print("  %-34s %10d %10d" % ("contacts between non-neighbours", before["wrong_contacts"], after["wrong_contacts"]))
    print("  %-34s %10d %10d" % ("order inversions", before["inversions"], after["inversions"]))
    print("  %-6s %10s %10s %8s %8s" % ("label", "ml before", "ml after", "pieces", "pieces"))
    for k in range(NUM_VERTEBRAE, 0, -1):
        if before["ml"][k] == 0 and after["ml"][k] == 0:
            print("  %-6s %10s %10s" % (VERTEBRAE[k - 1][10:], "-", "-"))
            continue
        print("  %-6s %10.1f %10.1f %8d %8d  %s" % (VERTEBRAE[k - 1][10:], before["ml"][k], after["ml"][k],
                                                    before["pieces"][k], after["pieces"][k], size_flag(after["ml"], k)))
    missing = [VERTEBRAE[k - 1][10:] for k in range(after["present"][0], after["present"][-1]) if after["ml"][k] == 0] \
        if after["present"] else []
    if missing:
        print("  WARNING: gap in the label sequence, missing %s" % ", ".join(missing))


def postprocess_case(pred_path, save_dir, case, args):
    img = nib.load(pred_path)
    full = np.asanyarray(img.dataobj).astype(np.uint8)
    full[full > NUM_VERTEBRAE] = 0
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    affine = img.affine
    result = np.zeros_like(full)

    if full.any():
        box = foreground_bbox(full > 0, 2, full.shape)
        origin = [s.start for s in box]
        lab = full[box].copy()
        before = summarize(lab, affine, origin, spacing)
        print("=" * 78)
        print(case)

        removed_ml = remove_stray_components(lab, spacing, args.min_component_ml, args.max_gap_mm)
        lab = smooth_label_boundaries(lab, spacing, args.smooth_mm)
        n_merged = merge_fragments(lab, spacing, args.min_island_ml)
        if repair_split_vertebrae(lab, spacing, affine, origin, args.core_fraction, args.core_window_mm, args.purity):
            n_merged += merge_fragments(lab, spacing, args.min_island_ml)
        order_msg = fix_adjacent_swaps(lab, affine, origin)

        after = summarize(lab, affine, origin, spacing)
        print_report(before, after, removed_ml, n_merged, order_msg)
        result[box] = lab
    else:
        print("%s: empty prediction, nothing to do" % case)

    case_dir = os.path.join(save_dir, case)
    seg_dir = os.path.join(case_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(result, affine), os.path.join(case_dir, "combined_labels.nii.gz"))
    for k in range(1, NUM_VERTEBRAE + 1):
        nib.save(nib.Nifti1Image((result == k).astype(np.uint8), affine),
                 os.path.join(seg_dir, VERTEBRAE[k - 1] + ".nii.gz"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default='./AbdomenAtlasDemoPredict', help='predicted masks')
    parser.add_argument('--save_dir', default='./AbdomenAtlasDemoPostprocessed', help='output folder')
    parser.add_argument('--min_component_ml', default=3.0, type=float, help='min size of a detached component to keep')
    parser.add_argument('--max_gap_mm', default=20.0, type=float, help='max distance of a detached component from the spine')
    parser.add_argument('--smooth_mm', default=5.0, type=float, help='majority vote window')
    parser.add_argument('--min_island_ml', default=0.5, type=float, help='isolated label pieces below this are removed')
    parser.add_argument('--core_fraction', default=0.6, type=float, help='distance threshold for body cores, relative to local max')
    parser.add_argument('--core_window_mm', default=25.0, type=float, help='window for the local max')
    parser.add_argument('--purity', default=0.8, type=float, help='min label purity of an anchor body')
    args = parser.parse_args()

    cases = sorted(c for c in os.listdir(args.pred_dir)
                   if os.path.isfile(os.path.join(args.pred_dir, c, 'combined_labels.nii.gz')))
    print('cases: {}'.format(len(cases)))
    for case in cases:
        postprocess_case(os.path.join(args.pred_dir, case, 'combined_labels.nii.gz'), args.save_dir, case, args)


if __name__ == "__main__":
    main()
