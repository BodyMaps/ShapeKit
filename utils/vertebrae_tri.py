"""ShapeKit adapter for the mask-only anatomical vertebra refinement backend.

Inputs: 24 binary NIfTI masks under CASE/segmentations/vertebrae_*.nii.gz.
Engine labels: 1=L5 through 24=C1 (background 0).
Output combined labels: IDs from ShapeKit class_map (usually 26 through 49).
No CT data are required. Original input folders are never modified.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np

from .vertebrae_tri_engine import refine_vertebrae


NAMES = (
    [f'vertebrae_{name}' for name in ('L5', 'L4', 'L3', 'L2', 'L1')]
    + [f'vertebrae_T{i}' for i in range(12, 0, -1)]
    + [f'vertebrae_C{i}' for i in range(7, 0, -1)]
)


def _check_grid(image, reference, filename):
    if len(image.shape) != 3 or image.shape != reference.shape:
        raise ValueError(f'{filename}: mask shape does not match the case reference')
    if not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
        raise ValueError(f'{filename}: mask affine does not match the case reference')
    if not np.allclose(image.header.get_zooms()[:3], reference.header.get_zooms()[:3], rtol=0, atol=1e-4):
        raise ValueError(f'{filename}: voxel spacing differs from the case reference')


def _load_binary(image, filename):
    arr = np.asarray(image.dataobj)
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f'{filename}: expected a binary mask with values 0 or 1')
    return arr.astype(bool, copy=False)


def _save_like(array, reference, filename):
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    result = nib.Nifti1Image(array, reference.affine, header=header)
    result.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    result.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(result, str(filename))


def process_case(input_path, output_path, ct_path, class_map, subfolder_name='segmentations'):
    """Run one patient independently using the ShapeKit backend interface.

    ct_path is accepted to match the common interface but not used. This
    algorithm is fully mask-based. Existing outputs are never overwritten.
    """
    del ct_path
    started = time.monotonic()
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    seg_folder = source / subfolder_name
    if not seg_folder.is_dir():
        raise FileNotFoundError(f'Segmentation directory not found: {seg_folder}')
    if not source.is_dir():
        raise FileNotFoundError(f'Case directory not found: {source}')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Input and output case directories must be disjoint')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing output: {output}')
    if any(not isinstance(k, int) or k < 1 or k > 65535 for k in class_map):
        raise ValueError('class_map must use positive integer IDs')
    if len(set(class_map.values())) != len(class_map):
        raise ValueError('class_map has duplicate anatomical names')
    ids_by_name = {name: label_id for label_id, name in class_map.items()}
    if any(name not in ids_by_name for name in NAMES):
        raise ValueError('class_map must include all 24 vertebral classes')

    reference = None
    originals = {}
    missing = []
    for name in NAMES:
        path = seg_folder / f'{name}.nii.gz'
        if not path.exists():
            missing.append(name)
            continue
        image = nib.load(str(path))
        if reference is None:
            reference = image
        else:
            _check_grid(image, reference, str(path))
        originals[name] = _load_binary(image, str(path))
    if reference is None:
        raise ValueError(f'No vertebral masks found in {seg_folder}')
    if not np.isfinite(reference.affine).all():
        raise ValueError('NIfTI affine must contain finite values')

    raw = np.zeros(reference.shape, dtype=np.uint8)
    for internal_id, name in enumerate(NAMES, start=1):
        binary = originals.get(name)
        if binary is None:
            continue
        if np.any((raw != 0) & binary):
            raise ValueError(f'Overlapping vertebra masks detected at {name}')
        raw[binary] = internal_id
    # Other organ masks can be copied, but must share the same voxel grid.
    for path in seg_folder.glob('*.nii.gz'):
        if path.name not in {f'{name}.nii.gz' for name in NAMES}:
            _check_grid(nib.load(str(path)), reference, str(path))

    refined = refine_vertebrae(raw, reference.affine)
    if refined.shape != raw.shape or not np.issubdtype(refined.dtype, np.integer):
        raise RuntimeError('Invalid output shape or dtype from vertebra engine')
    if np.any((refined < 0) | (refined > 24)):
        raise RuntimeError('Vertebra engine returned labels outside 0..24')
    # The existing algorithm can remove false positives but never add new bone.
    if np.any((refined != 0) & (raw == 0)):
        raise RuntimeError('Vertebra engine introduced foreground outside the prediction')

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.anatomical_stage_', dir=output.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        stage_seg = staging / subfolder_name
        dtype = np.uint8 if max(class_map) <= 255 else np.uint16
        # Rebuild the full combined map, retaining other organs when present.
        combined = np.zeros(reference.shape, dtype=dtype)
        for class_id, name in sorted(class_map.items()):
            if name in NAMES:
                continue
            path = seg_folder / f'{name}.nii.gz'
            if path.exists():
                combined[_load_binary(nib.load(str(path)), str(path))] = class_id
        for internal_id, name in enumerate(NAMES, start=1):
            binary = refined == internal_id
            _save_like(binary.astype(np.uint8), reference, stage_seg / f'{name}.nii.gz')
            combined[binary] = ids_by_name[name]
        _save_like(combined, reference, staging / 'combined_labels.nii.gz')
        report = {
            'engine': 'shapekit_tri',
            'input_case': source.name,
            'internal_labels': '1=L5 through 24=C1',
            'shape': list(reference.shape),
            'orientation': list(nib.aff2axcodes(reference.affine)),
            'missing_masks': missing,
            'original_foreground_voxels': int(np.count_nonzero(raw)),
            'refined_foreground_voxels': int(np.count_nonzero(refined)),
            'modified_voxels': int(np.count_nonzero(raw != refined)),
            'label_mapping': {str(i): ids_by_name[name] for i, name in enumerate(NAMES, 1)},
            'runtime_seconds': round(time.monotonic() - started, 2),
        }
        (staging / 'vertebrae_tri_report.json').write_text(json.dumps(report, indent=2) + '\n')
        # Atomic publication on the same filesystem (output must not exist).
        if output.exists():
            raise FileExistsError(f'Output created concurrently: {output}')
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
