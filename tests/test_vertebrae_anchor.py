"""
tests/test_vertebrae_anchor.py
==============================

Synthetic validation for the ShapeKit-Anchor vertebrae engine
(utils/vertebrae_anchor_engine.py) and its ShapeKit adapter
(utils/vertebrae_anchor.py).

Real cases have no ground-truth labels at hand, so measuring the engine on
them can only ever be a visual impression. This harness builds phantom spines
where the correct answer is known by construction, corrupts them with each
error mode the network actually makes, and reports voxel label accuracy
before and after refinement. The phantoms carry posterior processes, because
a body-only phantom hides the axial overlap between real neighbours and let
an earlier merge rule collapse 24 vertebrae into 4.

Error modes covered:
    islands        spurious blobs attached to a label
    duplicate      one physical vertebra carrying two different names
    swap           two adjacent levels with their names exchanged
    dropout        a level the model failed to name, leaving a hole in the run
    shuffle        several names permuted within a neighbourhood
    global_shift   the whole spine named one level off (a control: NOT fixable
                   without a landmark, and the engine must leave it alone
                   rather than make it worse)
    split_merge    the thoracolumbar failure seen on a real scan: two
                   neighbours merged under one name, another level cut into
                   fragments carrying repeated names, labels running backwards
    tapered        lumbar-to-cervical taper, nothing may be merged
    partial FOV    a spine starting at T12 must not be renumbered to L5
    adapter        the ShapeKit dict round trip reproduces the engine output

Run from the repository root:
    python tests/test_vertebrae_anchor.py
    python -m pytest tests/test_vertebrae_anchor.py
"""

from __future__ import annotations

import numpy as np

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vertebrae_anchor_engine import refine_case, N_LABELS, CLASS_MAP  # noqa: E402
from utils.vertebrae_anchor import postprocessing_vertebrae_anchor  # noqa: E402

RNG = np.random.default_rng(0)

# Phantom geometry, in millimetres.
SPACING_MM = (1.0, 1.0, 1.0)
BODY_MM = (26, 22, 16)      # roughly a lumbar body
PITCH_MM = 26               # centre-to-centre spacing along the spine


def make_affine() -> np.ndarray:
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = SPACING_MM
    return a


def make_spine(n_levels: int = 17, first_label: int = 1, jitter_mm: float = 1.5):
    """Build a phantom spine and its ground-truth label volume.

    Returns (volume, truth) where both carry labels first_label .. first_label
    + n_levels - 1 stacked along +z, i.e. toward the head, matching the SuPreM
    convention that a higher index is more superior.
    """
    depth = PITCH_MM * (n_levels + 1)
    shape = (64, 64, depth)
    vol = np.zeros(shape, dtype=np.uint8)
    cx, cy = shape[0] // 2, shape[1] // 2
    hx, hy, hz = BODY_MM[0] // 2, BODY_MM[1] // 2, BODY_MM[2] // 2

    for i in range(n_levels):
        label = first_label + i
        cz = int(PITCH_MM * (i + 1))
        dx = int(RNG.normal(0, jitter_mm))
        dy = int(RNG.normal(0, jitter_mm))
        vol[cx + dx - hx: cx + dx + hx,
            cy + dy - hy: cy + dy + hy,
            cz - hz: cz + hz] = label
        # Posterior elements. A real vertebra mask is not a compact block: the
        # spinous and transverse processes reach past the disc space and lie
        # alongside the neighbouring level, so consecutive vertebrae overlap
        # heavily when projected onto the spine axis. A phantom without them
        # will happily validate a merge rule that destroys a real spine, which
        # is exactly what happened before this was added.
        pz = hz + PITCH_MM // 2
        vol[cx + dx - 6: cx + dx + 6,
            cy + dy + hy - 1: cy + dy + hy + 8,
            max(cz - pz, 0): cz + pz] = label
    return vol, vol.copy()


def add_islands(vol, n=6, size=3):
    """Scatter small blobs carrying real labels but sitting off the spine."""
    out = vol.copy()
    labels = [l for l in np.unique(vol) if l > 0]
    for _ in range(n):
        label = int(RNG.choice(labels))
        x = int(RNG.integers(4, vol.shape[0] - size - 4))
        y = int(RNG.integers(4, vol.shape[1] - size - 4))
        z = int(RNG.integers(4, vol.shape[2] - size - 4))
        out[x:x + size, y:y + size, z:z + size] = label
    return out


def add_duplicate(vol):
    """Give the upper half of one body a neighbour's name."""
    out = vol.copy()
    labels = sorted(l for l in np.unique(vol) if l > 0)
    target = labels[len(labels) // 2]
    idx = np.argwhere(out == target)
    zs = idx[:, 2]
    # A thin slab off the top, not a clean half. This is what the network
    # actually does: it gives the upper rim of one vertebra the name of the
    # level above. A 50/50 split is not the realistic case, and it is not
    # separable from two real vertebrae by geometry alone.
    cut = np.quantile(zs, 0.80)
    upper = idx[zs > cut]
    out[upper[:, 0], upper[:, 1], upper[:, 2]] = target + 1
    return out


def swap_adjacent(vol):
    """Exchange the names of two neighbouring levels."""
    out = vol.copy()
    labels = sorted(l for l in np.unique(vol) if l > 0)
    a = labels[len(labels) // 2]
    b = a + 1
    out[vol == a] = b
    out[vol == b] = a
    return out


def dropout_level(vol):
    """Rename one interior level to a value that leaves a hole in the run."""
    out = vol.copy()
    labels = sorted(l for l in np.unique(vol) if l > 0)
    target = labels[len(labels) // 3]
    out[vol == target] = min(labels[-1] + 3, N_LABELS)
    return out


def shuffle_local(vol, k=4):
    """Permute the names of k consecutive levels."""
    out = vol.copy()
    labels = sorted(l for l in np.unique(vol) if l > 0)
    start = len(labels) // 3
    block = labels[start:start + k]
    perm = list(block)
    RNG.shuffle(perm)
    for src, dst in zip(block, perm):
        out[vol == src] = dst
    return out


def global_shift(vol, by=1):
    out = np.zeros_like(vol)
    for l in np.unique(vol):
        if l > 0:
            out[vol == l] = np.clip(l + by, 1, N_LABELS)
    return out


def make_ct(truth, leak_mask=None):
    """Phantom CT: bone where a vertebra is, soft tissue everywhere else.

    Values are ordinary Hounsfield units, so the same threshold the script uses
    on real scans is the one under test here.
    """
    ct = np.full(truth.shape, -50.0, dtype=np.float32)   # fat / soft tissue
    ct[truth > 0] = 320.0                                 # cancellous bone
    if leak_mask is not None:
        ct[leak_mask] = 40.0                              # muscle
    return ct


def add_soft_tissue_leak(vol, truth):
    """Grow one label sideways into tissue that is not bone."""
    out = vol.copy()
    labels = sorted(l for l in np.unique(vol) if l > 0)
    target = labels[len(labels) // 2]
    idx = np.argwhere(vol == target)
    x1 = idx[:, 0].max()
    sl = (slice(x1 + 1, x1 + 7),
          slice(idx[:, 1].min(), idx[:, 1].max() + 1),
          slice(idx[:, 2].min(), idx[:, 2].max() + 1))
    leak = np.zeros_like(vol, dtype=bool)
    leak[sl] = True
    out[leak] = target
    return out, leak


def make_tapered_spine(n_levels=17, first_label=1):
    """A spine whose spacing and body size shrink toward the head.

    Real spines are not evenly pitched: lumbar bodies are roughly twice as tall
    and twice as far apart as cervical ones. This is the case that caught a bug.
    An earlier duplicate test compared centroid distance against the median gap
    over the whole spine, which is dominated by the lumbar end, so the genuine
    cervical vertebrae were fused into each other and the labels above them all
    shifted.
    """
    pitches, sizes = [], []
    for i in range(n_levels):
        f = i / max(n_levels - 1, 1)          # 0 at L5, 1 at the top
        pitches.append(34.0 - 20.0 * f)        # 34 mm lumbar -> 14 mm cervical
        sizes.append(26.0 - 14.0 * f)          # 26 mm tall -> 12 mm tall
    depth = int(sum(pitches) + 2 * pitches[0]) + 20
    shape = (64, 64, depth)
    vol = np.zeros(shape, dtype=np.uint8)
    cx, cy = shape[0] // 2, shape[1] // 2
    z = pitches[0]
    for i in range(n_levels):
        hz = max(int(sizes[i] // 2), 2)
        hx = max(int((sizes[i] * 1.1) // 2), 3)
        cz = int(z)
        vol[cx - hx: cx + hx, cy - hx: cy + hx, cz - hz: cz + hz] = first_label + i
        pz = hz + int(pitches[i] // 2)
        vol[cx - 4: cx + 4, cy + hx - 1: cy + hx + 6,
            max(cz - pz, 0): cz + pz] = first_label + i
        z += pitches[min(i + 1, n_levels - 1)]
    return vol, vol.copy()


def split_and_merge(vol):
    """Reproduce the mid-spine failure from the demo scan.

    Levels a..a+4 in the middle are corrupted so that no component count is
    trustworthy there: a and a+1 are merged under one name, a+2 is cut into
    three fragments that carry two different names, and a+3 takes the name of
    a+1, so names run backwards. The ends stay clean, which is what the
    resolver anchors on.
    """
    out = vol.copy()
    labels = sorted(l for l in np.unique(vol) if l > 0)
    a = labels[len(labels) // 3]
    out[vol == a + 1] = a                      # merge a and a+1 under a
    idx = np.argwhere(vol == a + 2)             # split a+2 into three pieces
    zs = idx[:, 2]
    q1, q2 = np.quantile(zs, [0.35, 0.7])
    lowp = idx[zs <= q1]
    midp = idx[(zs > q1) & (zs <= q2)]
    out[lowp[:, 0], lowp[:, 1], lowp[:, 2]] = a + 1
    out[midp[:, 0], midp[:, 1], midp[:, 2]] = a + 3
    out[vol == a + 3] = a + 1                   # a+3 takes a+1's name (backwards)
    return out


def accuracy(pred, truth):
    """Voxel label agreement over the union of the two foregrounds."""
    fg = (pred > 0) | (truth > 0)
    if not fg.any():
        return 1.0
    return float((pred[fg] == truth[fg]).mean())


def run():
    affine = make_affine()
    clean, truth = make_spine(n_levels=17, first_label=1)

    cases = [
        ("clean (must not be damaged)", clean, truth),
        ("islands", add_islands(clean), truth),
        ("duplicate name on one body", add_duplicate(clean), truth),
        ("adjacent levels swapped", swap_adjacent(clean), truth),
        ("interior level dropped out", dropout_level(clean), truth),
        ("four levels shuffled", shuffle_local(clean), truth),
    ]

    print(f"{'case':34s} {'before':>8s} {'after':>8s} {'delta':>8s}  verdict")
    print("-" * 78)
    failures = 0
    for name, corrupted, gt in cases:
        refined, stats = refine_case(corrupted, affine)
        before, after = accuracy(corrupted, gt), accuracy(refined, gt)
        ok = after >= before - 1e-6
        if name.startswith("clean"):
            ok = after > 0.999
        if not ok:
            failures += 1
        print(f"{name:34s} {before:8.3f} {after:8.3f} {after - before:+8.3f}  "
              f"{'ok' if ok else 'REGRESSED'}")

    # Control: a spine named consistently but one level off cannot be corrected
    # from geometry alone, because no landmark is present. The requirement is
    # that the script leaves it as it found it rather than inventing a fix.
    shifted = global_shift(clean, by=1)
    refined, _ = refine_case(shifted, affine)
    kept = accuracy(refined, shifted)
    print("-" * 78)
    print(f"{'global shift (control, unfixable)':34s} "
          f"{accuracy(shifted, truth):8.3f} {accuracy(refined, truth):8.3f} "
          f"{'':8s}  agrees with input {kept:.3f}")
    if kept < 0.999:
        print("  NOTE: the script altered a self-consistent spine, and that is wrong.")
        failures += 1

    # The bone check: a mask that has spilled into muscle should lose the spill
    # without the vertebra itself being eaten away.
    leaked, leak = add_soft_tissue_leak(clean, truth)
    ct = make_ct(truth, leak_mask=leak)
    refined, stats = refine_case(leaked, affine, ct)
    leak_before = int(((leaked > 0) & leak).sum())
    leak_after = int(((refined > 0) & leak).sum())
    body_kept = int(((refined > 0) & (truth > 0)).sum()) / max(int((truth > 0).sum()), 1)
    ok = leak_after < 0.35 * leak_before and body_kept > 0.97
    print(f"{'soft-tissue leak removed':34s} {leak_before:8d} {leak_after:8d} "
          f"{'':8s}  body kept {body_kept:.3f} {'ok' if ok else 'FAILED'}")
    if not ok:
        failures += 1

    # Split-and-merge: the component count is meaningless in the middle, so
    # the resolver must rebuild those levels from the clean ends. A planar cut
    # cannot reproduce the phantom's posterior elements exactly, so the bar is
    # a large improvement and every level present, not 1.000.
    sm = split_and_merge(clean)
    refined, stats = refine_case(sm, affine)
    before, after = accuracy(sm, truth), accuracy(refined, truth)
    n_out = len([l for l in np.unique(refined) if l > 0])
    n_true = len([l for l in np.unique(truth) if l > 0])
    ok = (after - before) > 0.08 and n_out == n_true and stats.get("resolver") == "anchors_and_slabs"
    print(f"{'split and merge (demo failure)':34s} {before:8.3f} {after:8.3f} {after - before:+8.3f}  "
          f"{n_out}/{n_true} levels, {stats.get('resolver', 'rank')} {'ok' if ok else 'FAILED'}")
    if not ok:
        failures += 1

    # Tapered spine: nothing may be merged, and every name must survive.
    tapered, tap_truth = make_tapered_spine(17, first_label=1)
    refined, stats = refine_case(tapered, affine)
    acc = accuracy(refined, tap_truth)
    n_in = len([l for l in np.unique(tapered) if l > 0])
    n_out = len([l for l in np.unique(refined) if l > 0])
    ok = acc > 0.999 and n_out == n_in and stats["duplicates_merged"] == 0
    print(f"{'tapered spine, lumbar to cervical':34s} {1.0:8.3f} {acc:8.3f} "
          f"{'':8s}  {n_in}->{n_out} levels, {stats['duplicates_merged']} merged "
          f"{'ok' if ok else 'FAILED'}")
    if not ok:
        failures += 1

    # A partial field of view must not be renumbered to start at L5.
    partial, partial_truth = make_spine(n_levels=8, first_label=6)
    refined, _ = refine_case(partial, affine)
    acc = accuracy(refined, partial_truth)
    print(f"{'partial FOV, T12 upward':34s} {1.0:8.3f} {acc:8.3f} "
          f"{'':8s}  {'ok' if acc > 0.999 else 'REGRESSED'}")
    if acc <= 0.999:
        failures += 1

    # Adapter: the ShapeKit dict round trip must reproduce the engine output
    # label for label, with no CT and no landmarks present.
    import nibabel as nib
    seg_dict = {name: (sm == k).astype(np.uint8) for k, name in CLASS_MAP.items()}
    seg_dict["liver"] = np.zeros(sm.shape, dtype=np.uint8)
    ref_img = nib.Nifti1Image(np.zeros(sm.shape, dtype=np.uint8), affine)
    out_dict = postprocessing_vertebrae_anchor(
        "phantom", seg_dict, ref_img, ct_path=None, logger=logging.getLogger("test"))
    engine_out, _ = refine_case(sm, affine)
    rebuilt = np.zeros(sm.shape, dtype=np.uint8)
    for k, name in CLASS_MAP.items():
        rebuilt[out_dict[name] > 0] = k
    same = bool(np.array_equal(rebuilt, engine_out))
    liver_kept = bool(np.array_equal(out_dict["liver"], seg_dict["liver"]))
    ok = same and liver_kept
    print(f"{'ShapeKit adapter round trip':34s} {'':8s} {'':8s} "
          f"{'':8s}  identical to engine {same}, other organs untouched {liver_kept} "
          f"{'ok' if ok else 'FAILED'}")
    if not ok:
        failures += 1

    print("-" * 78)
    print("FAILURES:", failures)
    return failures


def test_vertebrae_anchor():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
