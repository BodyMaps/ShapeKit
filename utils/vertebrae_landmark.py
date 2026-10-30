"""ShapeKit adapter for the landmark-numbering vertebrae engine.

Engine: utils/vertebrae_landmark_engine.py (the evaluated warm-up solution,
callable as refine()). This adapter only handles ShapeKit input/output:

  * reads the 24 binary vertebra masks from <case>/<subfolder>/ (missing
    masks are treated as empty and listed in the report);
  * requires the matching calibrated CT (the rib check and the arch step
    use it) on exactly the same grid;
  * runs the engine on the native grid (no resampling);
  * writes all 24 masks, copies every other mask byte for byte, writes
    combined_labels.nii.gz with ShapeKit's class_map IDs and a JSON report;
  * builds the case in a staging folder and publishes it by renaming, so a
    failed case never leaves a partial output folder.
"""
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np

from . import vertebrae_landmark_engine as engine

NAMES = tuple(f'vertebrae_{level}' for level in engine.NAMES)     # L5 ... C1, engine labels 1..24
REPORT_NAME = 'vertebrae_landmark_report.json'


def check_grid(image, reference, name):
    if (len(image.shape) != 3 or image.shape != reference.shape
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5)):
        raise ValueError(f'CT/mask geometry mismatch: {name}')


def check_orientation(image):
    """Any axis-aligned orientation is accepted (the engine reorients to RAS
    internally and back); oblique or sheared affines are rejected."""
    spacing = np.asarray(image.header.get_zooms()[:3], float)
    if not np.all(np.isfinite(image.affine)) or np.any(spacing <= 0):
        raise ValueError('Finite affine and positive spacing required')
    directions = image.affine[:3, :3] / spacing
    if not np.allclose(np.abs(directions), np.round(np.abs(directions)), atol=1e-4):
        raise ValueError('Oblique or sheared orientation is unsupported; no automatic resampling')


def binary(image, name):
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f'Finite binary 3D mask required: {name}')
    return data.astype(bool)


def load_masks(source, subfolder):
    reference, raw, missing = None, None, []
    for label, name in enumerate(NAMES, 1):
        path = source / subfolder / f'{name}.nii.gz'
        if not path.is_file():
            missing.append(name)
            continue
        image = nib.load(path)
        mask = binary(image, name)
        if reference is None:
            reference = image
            check_orientation(image)
            raw = np.zeros(image.shape, np.int16)
        check_grid(image, reference, name)
        if np.any(mask & (raw != 0)):
            raise ValueError(f'Overlapping vertebra masks: {name}')
        raw[mask] = label
    if reference is None:
        raise ValueError('No vertebra masks found')
    return reference, raw, missing


def save(data, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    out = nib.Nifti1Image(data, reference.affine, header)
    out.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    out.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(out, path)


CROP_MM = 30.0


def spine_box(labels, zooms, margin_mm=CROP_MM):
    """Box around the MAIN piece (largest 26-connected component) of every
    vertebra label, grown by margin_mm on every side.

    Stray pieces far from the spine (e.g. pelvic bone labelled C1) are left
    out, so the box stays small on large scans. Anything outside the box is
    more than margin_mm from every vertebra's main piece and touches none of
    them; the engine's Step 1 removes such pieces (far from every vertebra,
    and far from its own), so the adapter clears them directly.
    """
    lo = np.full(3, np.iinfo(np.int64).max)
    hi = np.zeros(3, np.int64)
    struct = np.ones((3, 3, 3), bool)
    for v, box in enumerate(engine.ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        comp, n = engine.ndimage.label(labels[box] == v, structure=struct)
        if n == 0:
            continue
        main = int(np.argmax(np.bincount(comp.ravel())[1:])) + 1
        sub = engine.ndimage.find_objects((comp == main).astype(np.uint8))[0]
        del comp
        starts = [b.start + s.start for b, s in zip(box, sub)]
        stops = [b.start + s.stop for b, s in zip(box, sub)]
        lo, hi = np.minimum(lo, starts), np.maximum(hi, stops)
    out = []
    for axis, size in enumerate(labels.shape):
        pad = int(np.ceil(margin_mm / zooms[axis]))
        out.append(slice(max(int(lo[axis]) - pad, 0), min(int(hi[axis]) + pad, size)))
    return tuple(out)


def disjoint(a, b):
    return a != b and a not in b.parents and b not in a.parents


def process_case(input_path, output_path, ct_path, class_map, subfolder_name='segmentations'):
    started = time.monotonic()
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if not disjoint(source, output):
        raise ValueError('Input and output must be disjoint')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if ct_path is None or not Path(ct_path).is_file():
        raise FileNotFoundError(f'Matching calibrated CT required: {ct_path}')
    ct_path = Path(ct_path).resolve()
    if not disjoint(output, ct_path):
        raise ValueError('Output must not contain the CT input')
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ('', '.', '..'):
        raise ValueError('Invalid segmentation subfolder')
    if (any(not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 65535 for k in class_map)
            or len(set(class_map.values())) != len(class_map)):
        raise ValueError('class_map needs unique names and positive uint16 IDs')
    ids = {v: k for k, v in class_map.items()}
    if any(name not in ids for name in NAMES):
        raise ValueError('class_map must include all 24 vertebrae')

    reference, raw, missing = load_masks(source, subfolder_name)
    ct_image = nib.load(ct_path)
    check_grid(ct_image, reference, str(ct_path))
    extras = []
    for path in sorted((source / subfolder_name).glob('*.nii.gz')):
        if path.name[:-7] not in NAMES:
            other = nib.load(path)
            check_grid(other, reference, path.name)
            binary(other, path.name)
            extras.append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / '.incomplete'
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=source.name + '-', dir=staging_root))
    try:
        zooms = reference.header.get_zooms()[:3]
        refined = np.zeros(raw.shape, raw.dtype)
        outside = 0
        if raw.any():
            # work on the spine plus a CROP_MM margin (see spine_box): the full
            # grid of a fine-sliced scan would need several times the memory
            box = spine_box(raw, zooms)
            outside = int(np.count_nonzero(raw)) - int(np.count_nonzero(raw[box]))
            ct = np.asarray(ct_image.dataobj[box]).astype(np.float32)
            affine = reference.affine.copy()
            affine[:3, 3] = reference.affine[:3, :3] @ [s.start for s in box] + reference.affine[:3, 3]
            refined[box], steps = engine.refine(raw[box], ct, affine, zooms)
            del ct
        else:
            steps = {'actions': [], 'step2': {'reason': 'no vertebra voxels'}, 'step3': {}}
        del raw
        segdir = stage / subfolder_name
        segdir.mkdir()
        for path in extras:
            shutil.copy2(path, segdir / path.name)
        combined = np.zeros(refined.shape, np.uint8 if max(class_map) <= 255 else np.uint16)
        for label_id, name in sorted(class_map.items()):
            if name in NAMES:
                mask = refined == NAMES.index(name) + 1
                save(mask.astype(np.uint8), reference, segdir / f'{name}.nii.gz')
            else:
                path = segdir / f'{name}.nii.gz'
                if not path.exists():
                    continue
                mask = binary(nib.load(path), name)
            combined[mask] = label_id
        save(combined, reference, stage / 'combined_labels.nii.gz')
        report = {
            'engine': 'shapekit_landmark',
            'raw': str(source / subfolder_name), 'ct': str(ct_path),
            'label_map': {str(i): ids[n] for i, n in enumerate(NAMES, 1)},
            'missing_input_masks': missing,
            'far_stray_voxels_removed': outside,
            'step1_summary': {k: sum(a['action'] == k for a in steps['actions'])
                              for k in ('remove', 'relabel', 'keep')},
            'step2': steps['step2'], 'step3': steps['step3'],
            'orientation': list(nib.aff2axcodes(reference.affine)),
            'runtime_seconds': round(time.monotonic() - started, 1),
            'engine_sha256': hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
            'dsc': None,
        }
        (stage / REPORT_NAME).write_text(json.dumps(report, indent=1, default=str) + '\n')
        stage.rename(output)
    except Exception:
        logging.exception('Incomplete case retained at %s', stage)
        raise
    return report
