"""Adapter for the opt-in ``shapekit_driftfix`` vertebrae engine.

The engine itself is ``vertebrae_driftfix_engine.py`` (a vendored, unchanged script that works on one
combined label map with values 1 (L5) ... 24 (C1)). This file only connects it to ShapeKit:
read the 24 per-level masks, check them, build the 1..24 map, run the engine in this process,
and write the 24 masks, ``combined_labels.nii.gz`` (ShapeKit ids) and a JSON report.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
import nibabel as nib
import numpy as np
from . import vertebrae_driftfix_engine as engine

NAMES = tuple(f'vertebrae_{engine.LABEL_NAMES[i]}' for i in range(1, engine.NUM_LABELS + 1))
CASE = 'case'


def shapekit_to_engine(label_id):
    """ShapeKit id (26..49) -> engine label (1..24)."""
    return int(label_id) - 25


def engine_to_shapekit(label):
    """Engine label (1..24) -> ShapeKit id (26..49)."""
    return int(label) + 25


def relabel_from_config(path='config.yaml'):
    """Read the optional config key ``vertebrae_driftfix_relabel``. Default False (the safe choice)."""
    try:
        import yaml
        with open(path, 'r') as f:
            return bool((yaml.safe_load(f) or {}).get('vertebrae_driftfix_relabel', False))
    except OSError:
        return False


def check_grid(image, reference, name):
    if len(image.shape) != 3 or image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-05):
        raise ValueError(f'CT/mask geometry mismatch: {name}')


def binary(image, name):
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f'Finite binary 3D mask required: {name}')
    return data.astype(bool)


def load_masks(source, subfolder):
    """Return (reference image, uint8 map with 1..24, names of missing masks)."""
    reference = None
    raw = None
    missing = []
    for label, name in enumerate(NAMES, 1):
        path = source / subfolder / f'{name}.nii.gz'
        if not path.is_file():
            missing.append(name)
            continue
        image = nib.load(path)
        mask = binary(image, name)
        if reference is None:
            reference = image
            raw = np.zeros(image.shape, np.uint8)
        check_grid(image, reference, name)
        if np.any(mask & (raw != 0)):
            raise ValueError(f'Overlapping vertebra masks: {name}')
        raw[mask] = label
    if reference is None:
        raise ValueError('No vertebra masks found')
    return reference, raw, missing


def save(data, reference, path):
    """Write ``data`` with the header, qform and sform of ``reference``."""
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    out = nib.Nifti1Image(data, reference.affine, header)
    out.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    out.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(out, path)


def disjoint(a, b):
    return a != b and a not in b.parents and b not in a.parents


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def process_case(input_path, output_path, ct_path, class_map, subfolder_name='segmentations', relabel=None):
    """Run the drift-fix engine on one case folder and write a new output folder.

    relabel: True runs all five stages. False skips the two stages that can change which level a voxel
    belongs to (label drift and lost level), like the script's ``--no_relabel``. None (default) reads the
    config key ``vertebrae_driftfix_relabel``, which is False when absent.
    A missing CT is allowed: the engine then runs its mask-only path and the report says so.
    """
    started = time.monotonic()
    if relabel is None:
        relabel = relabel_from_config()
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if not disjoint(source, output):
        raise ValueError('Input and output must be disjoint')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    ct = None
    if ct_path is not None and Path(ct_path).is_file():
        ct = Path(ct_path).resolve()
        if not disjoint(output, ct):
            raise ValueError('Output must not contain the CT input')
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ('', '.', '..'):
        raise ValueError('Invalid segmentation subfolder')
    if any(not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 65535 for k in class_map) or len(set(class_map.values())) != len(class_map):
        raise ValueError('class_map needs unique names and positive uint16 IDs')
    ids = {v: k for k, v in class_map.items()}
    if any(name not in ids for name in NAMES):
        raise ValueError('class_map must include all 24 vertebrae')
    if any(ids[name] != engine_to_shapekit(i) for i, name in enumerate(NAMES, 1)):
        raise ValueError('class_map ids for the vertebrae must be 26 (L5) to 49 (C1)')
    reference, raw, missing = load_masks(source, subfolder_name)
    if ct is not None:
        ct_image = nib.load(ct)
        if len(ct_image.shape) == 3:
            check_grid(ct_image, reference, str(ct))
        else:
            raise ValueError(f'CT must be 3D: {ct}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / '.incomplete'
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=source.name + '-', dir=staging_root))
    try:
        with tempfile.TemporaryDirectory(prefix='driftfix-', dir=staging_root) as temp:
            temp = Path(temp)
            (temp / 'in' / CASE).mkdir(parents=True)
            save(raw, reference, temp / 'in' / CASE / 'combined_labels.nii.gz')
            del raw
            ct_root = None
            if ct is not None:
                (temp / 'ct' / CASE).mkdir(parents=True)
                link_or_copy(ct, temp / 'ct' / CASE / 'ct.nii.gz')
                ct_root = str(temp / 'ct')
            report = engine.process_case(CASE, str(temp / 'in'), ct_root, str(temp / 'out'), relabel=relabel)
            internal = np.asanyarray(nib.load(temp / 'out' / CASE / 'combined_labels.nii.gz').dataobj)
        segdir = stage / subfolder_name
        segdir.mkdir()
        combined = np.zeros(internal.shape, np.uint8 if max(class_map) <= 255 else np.uint16)
        for i, name in enumerate(NAMES, 1):
            mask = internal == i
            save(mask.astype(np.uint8), reference, segdir / f'{name}.nii.gz')
            combined[mask] = ids[name]
        save(combined, reference, stage / 'combined_labels.nii.gz')
        engine_path = Path(engine.__file__)
        report.update(
            raw=str(source / subfolder_name),
            ct=str(ct) if ct is not None else None,
            ct_used=report['inputs']['ct_status'] == 'used',
            label_map={str(i): ids[n] for i, n in enumerate(NAMES, 1)},
            missing_input_masks=missing,
            relabel=bool(relabel),
            review_flags=[c for c in report['changes'] if c.get('action') == 'flagged'] + list(report['order_flags']),
            runtime_seconds=time.monotonic() - started,
            orientation=list(nib.aff2axcodes(reference.affine)),
            engine_sha256_lf=hashlib.sha256(engine_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            integration_status='complete')
        (stage / 'vertebrae_driftfix_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        stage.rename(output)
    except Exception:
        logging.exception('Incomplete case retained at %s', stage)
        raise
    return report
