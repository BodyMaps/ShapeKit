"""
ShapeKit adapter for the axis-based vertebra relabelling engine (`shapekit_axis_relabel`).

Reads the 24 vertebra masks of a case, relabels them against the case CT with
vertebrae_axis_relabel_engine, and writes a new case folder:

    <output>/segmentations/vertebrae_*.nii.gz   relabelled vertebra masks
    <output>/segmentations/<other>.nii.gz       all other masks, copied unchanged
    <output>/combined_labels.nii.gz             combined map using the configured class_map IDs

The case is written to a staging folder first and only renamed to <output> once everything
has been saved, so a folder under <output> is always complete. That makes it safe to resume
a large batch with --continue_prediction.
"""
import logging
import shutil
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from . import vertebrae_axis_relabel_engine as engine

# engine label 1 = L5 ... 24 = C1
LEVELS = ['L5', 'L4', 'L3', 'L2', 'L1',
          'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2', 'T1',
          'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']
NAMES = tuple(f'vertebrae_{level}' for level in LEVELS)


def same_grid(image, reference):
    return (tuple(image.shape[:3]) == tuple(reference.shape[:3])
            and np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5))


def read_binary(image, name):
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f'Binary 3D mask required: {name}')
    return data.astype(bool)


def load_vertebrae(seg_dir):
    """Combine the vertebra masks into one engine label map (1 = L5 ... 24 = C1)."""
    reference, labels, missing = None, None, []
    for label, name in enumerate(NAMES, start=1):
        path = seg_dir / f'{name}.nii.gz'
        if not path.is_file():
            missing.append(name)
            continue
        image = nib.load(path)
        mask = read_binary(image, name)
        if reference is None:
            reference = image
            labels = np.zeros(image.shape, np.uint8)
        elif not same_grid(image, reference):
            raise ValueError(f'Mask geometry mismatch: {name}')
        if np.any(mask & (labels != 0)):
            raise ValueError(f'Overlapping vertebra masks: {name}')
        labels[mask] = label
    if reference is None:
        raise ValueError('No vertebra masks found')
    return reference, labels, missing


def save(data, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    image = nib.Nifti1Image(data, reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(image, path)


def separate(a, b):
    return a != b and a not in b.parents and b not in a.parents


def process_case(input_path, output_path, ct_path, class_map, subfolder_name='segmentations'):
    started = time.monotonic()
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    seg_dir = source / subfolder_name

    if not separate(source, output):
        raise ValueError('Input and output must be separate folders')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if ct_path is None or not Path(ct_path).is_file():
        raise FileNotFoundError(f'CT not found: {ct_path}')
    ct_path = Path(ct_path).resolve()
    if not separate(output, ct_path):
        raise ValueError('Output must not contain the CT')
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ('', '.', '..'):
        raise ValueError('Invalid segmentation subfolder')
    ids = {name: label_id for label_id, name in class_map.items()}
    if len(ids) != len(class_map) or any(name not in ids for name in NAMES):
        raise ValueError('class_map needs unique names and all 24 vertebrae')

    reference, labels, missing = load_vertebrae(seg_dir)
    ct_image = nib.load(ct_path)
    if not same_grid(ct_image, reference):
        raise ValueError(f'CT/mask geometry mismatch: {ct_path}')

    others = sorted(p for p in seg_dir.glob('*.nii.gz') if p.name[:-7] not in NAMES)
    for path in others:
        if not same_grid(nib.load(path), reference):
            raise ValueError(f'Mask geometry mismatch: {path.name}')

    result, status = engine.relabel_volume(labels, ct_image.dataobj, reference.affine,
                                           reference.header.get_zooms())
    del labels

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / '.incomplete'
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=source.name + '-', dir=staging_root))
    try:
        out_seg = stage / subfolder_name
        out_seg.mkdir()
        for path in others:
            shutil.copy2(path, out_seg / path.name)
        combined = np.zeros(result.shape, np.uint8 if max(class_map) <= 255 else np.uint16)
        for label_id, name in sorted(class_map.items()):
            if name in NAMES:
                mask = result == NAMES.index(name) + 1
                save(mask.astype(np.uint8), reference, out_seg / f'{name}.nii.gz')
            elif (out_seg / f'{name}.nii.gz').is_file():
                mask = np.asanyarray(nib.load(out_seg / f'{name}.nii.gz').dataobj) > 0
            else:
                continue
            combined[mask] = label_id
        save(combined, reference, stage / 'combined_labels.nii.gz')
        stage.rename(output)
    except Exception:
        logging.exception('Incomplete case left at %s', stage)
        raise

    message = f'[axis_relabel] {source.name}: {status}, {time.monotonic() - started:.0f} s'
    if missing:
        message += f', missing input masks: {", ".join(missing)}'
    if status == 'spine not found':
        logging.warning(message + ' (input labels kept)')
    else:
        logging.info(message)
    return status
