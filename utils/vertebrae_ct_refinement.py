import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
import nibabel as nib
import numpy as np
from . import vertebrae_ct_refinement_engine as engine
NAMES = tuple((f'vertebrae_{level}' for level in engine.LEVELS[1:]))

def check_grid(image, reference, name):
    if len(image.shape) != 3 or image.shape != reference.shape or (not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-05)):
        raise ValueError(f'CT/mask geometry mismatch: {name}')

def check_orientation(image):
    codes = nib.aff2axcodes(image.affine)
    if codes not in [('R', 'A', 'S'), ('L', 'A', 'S')]:
        raise ValueError(f'Expected RAS or LAS orientation, got {codes}; no automatic resampling')
    linear = image.affine[:3, :3]
    spacing = np.asarray(image.header.get_zooms()[:3], float)
    if not np.all(np.isfinite(image.affine)) or np.any(spacing <= 0):
        raise ValueError('Finite affine and positive spacing required')
    directions = linear / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=0.0001):
        raise ValueError('Sheared affine is unsupported')

def binary(image, name):
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f'Finite binary 3D mask required: {name}')
    return data.astype(bool)

def load_masks(source, subfolder):
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
            check_orientation(image)
            raw = np.zeros(image.shape, np.uint8)
        check_grid(image, reference, name)
        if np.any(mask & (raw != 0)):
            raise ValueError(f'Overlapping vertebra masks: {name}')
        raw[mask] = label
    if reference is None:
        raise ValueError('No vertebra masks found')
    return (reference, raw, missing)

def save(data, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    out = nib.Nifti1Image(data, reference.affine, header)
    out.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    out.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(out, path)

def disjoint(a, b):
    return a != b and a not in b.parents and (b not in a.parents)

def process_case(input_path, output_path, ct_path, class_map, subfolder_name='segmentations'):
    started = time.monotonic()
    source, output = (Path(input_path).resolve(), Path(output_path).resolve())
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
    if any((not isinstance(k, int) or isinstance(k, bool) or (not 1 <= k <= 65535) for k in class_map)) or len(set(class_map.values())) != len(class_map):
        raise ValueError('class_map needs unique names and positive uint16 IDs')
    ids = {v: k for k, v in class_map.items()}
    if any((name not in ids for name in NAMES)):
        raise ValueError('class_map must include all 24 vertebrae')
    reference, raw, missing = load_masks(source, subfolder_name)
    check_grid(nib.load(ct_path), reference, str(ct_path))
    extras = []
    for path in (source / subfolder_name).glob('*.nii.gz'):
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
        with tempfile.TemporaryDirectory(prefix='ct_refinement-input-') as temp:
            raw_path = Path(temp) / 'combined_labels.nii.gz'
            save(raw, reference, raw_path)
            del raw
            report = engine.build_case(raw_path, ct_path, stage, 0.8, 250.0)
        internal = np.asanyarray(nib.load(stage / 'combined_labels.nii.gz').dataobj)
        segdir = stage / 'segmentations'
        if subfolder_name != 'segmentations':
            segdir.rename(stage / subfolder_name)
            segdir = stage / subfolder_name
        for path in extras:
            shutil.copy2(path, segdir / path.name)
        combined = np.zeros(internal.shape, np.uint8 if max(class_map) <= 255 else np.uint16)
        for label_id, name in sorted(class_map.items()):
            if name in NAMES:
                mask = internal == NAMES.index(name) + 1
                save(mask.astype(np.uint8), reference, segdir / f'{name}.nii.gz')
            else:
                path = segdir / f'{name}.nii.gz'
                if not path.exists():
                    continue
                mask = binary(nib.load(path), name)
            combined[mask] = label_id
        save(combined, reference, stage / 'combined_labels.nii.gz')
        report.update(raw=str(source / subfolder_name), ct=str(ct_path), label_map={str(i): ids[n] for i, n in enumerate(NAMES, 1)}, missing_input_masks=missing, runtime_seconds=time.monotonic() - started, orientation=list(nib.aff2axcodes(reference.affine)), engine_sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(), geometry_policy='Native grid, no resampling; LAS or RAS; oblique scans need separate validation', integration_status='complete', dsc=None)
        (stage / 'vertebrae_ct_refinement_report.json').write_text(json.dumps(report, indent=2) + '\n')
        stage.rename(output)
    except Exception:
        logging.exception('Incomplete case retained at %s', stage)
        raise
    return report
