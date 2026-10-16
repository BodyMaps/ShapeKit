"""File adapter for reviewed component refinement; no experiment-folder dependencies."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np

from . import vertebrae_components_engine as engine

NAMES = tuple('vertebrae_' + name for name in engine.LABEL_NAMES[1:])


def check_grid(image, reference):
    if (len(image.shape) != 3 or image.shape != reference.shape
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5)
            or not np.allclose(image.header.get_zooms()[:3], reference.header.get_zooms()[:3], rtol=0, atol=1e-5)):
        raise ValueError('CT and segmentation geometry differ; resampling is not performed')


def check_ct(ct):
    if len(ct.shape) != 3 or ct.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('A 3D CT with millimetre spatial units is required')
    linear = ct.affine[:3, :3]
    if (not np.all(np.isfinite(ct.affine))
            or not np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=1e-5)
            or nib.aff2axcodes(ct.affine) != ('L', 'A', 'S')):
        raise ValueError('This evaluated implementation requires axis-aligned LAS geometry')
    spacing = np.asarray(ct.header.get_zooms()[:3], dtype=float)
    if np.any(spacing <= 0) or not np.allclose(spacing, np.abs(np.diag(linear)), rtol=0, atol=1e-5):
        raise ValueError('Invalid or inconsistent voxel spacing')
    return spacing


def read_binary(path, ct):
    image = nib.load(str(path))
    check_grid(image, ct)
    data = np.asanyarray(image.dataobj)
    if not np.all((data == 0) | (data == 1)):
        raise ValueError('Expected a binary mask: ' + str(path))
    return image, data.astype(bool)


def load_labels(source, ct, subfolder='segmentations'):
    """Use mask names, not the potentially different combined-map label scheme."""
    raw = np.zeros(ct.shape, dtype=np.uint8)
    for label, name in enumerate(NAMES, 1):
        _, mask = read_binary(Path(source) / subfolder / (name + '.nii.gz'), ct)
        if np.any(mask & (raw != 0)):
            raise ValueError('Overlapping vertebra masks: ' + name)
        raw[mask] = label
    return raw


def fingerprint(labels, affine):
    """Hash geometry and canonical voxel values, independently of gzip metadata."""
    digest = hashlib.sha256()
    digest.update(np.asarray(labels.shape, dtype='<i8').tobytes())
    digest.update(np.asarray(affine, dtype='<f8').tobytes())
    for slab in labels:
        digest.update(np.ascontiguousarray(slab, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def validate_plan(plan, raw, ct):
    if plan.get('schema_version') != 1 or plan.get('mode') not in ('cleanup_only', 'reviewed'):
        raise ValueError('Unsupported component review plan')
    if plan.get('source_sha256') != fingerprint(raw, ct.affine):
        raise ValueError('Review plan does not match the input labels and geometry')
    if plan.get('mode') == 'reviewed':
        if not isinstance(plan.get('seeds'), list) or len(plan['seeds']) != 17:
            raise ValueError('Reviewed mode requires 17 reviewed thoracolumbar seed entries')
        if not isinstance(plan.get('boundary_splits', []), list):
            raise ValueError('boundary_splits must be a list')
    elif plan.get('seeds') or plan.get('boundary_splits'):
        raise ValueError('Cleanup-only plans cannot contain corrections')


def save_verified(data, source, ct, destination):
    header = source.header.copy()
    header.set_data_dtype(data.dtype)
    header.set_xyzt_units(xyz='mm', t=source.header.get_xyzt_units()[1])
    image = nib.Nifti1Image(data, source.affine, header)
    for form in ('qform', 'sform'):
        getattr(image, 'set_' + form)(getattr(source, 'get_' + form)(), code=int(source.header[form + '_code']))
    image.header.set_slope_inter(1.0, 0.0)
    nib.save(image, str(destination))
    actual = nib.load(str(destination))
    check_grid(actual, ct)
    if actual.header.get_xyzt_units()[0] != 'mm' or not np.array_equal(np.asanyarray(actual.dataobj), data):
        raise RuntimeError('Saved prediction verification failed: ' + str(destination))


def disjoint(first, second):
    return first != second and first not in second.parents and second not in first.parents


def process_case(input_path, output_path, ct_path, class_map,
                 subfolder_name='segmentations', plan_path=None):
    """Process a case atomically; a missing or mismatched review plan is an error."""
    started = time.monotonic()
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if not disjoint(source, output):
        raise ValueError('Input and output directories must be disjoint')
    if output.exists():
        raise FileExistsError('Use a new output directory: ' + str(output))
    if plan_path is None or not Path(plan_path).is_file():
        raise FileNotFoundError('An explicit per-case review plan is required')
    if ct_path is None or not Path(ct_path).is_file():
        raise FileNotFoundError('Matching CT required')
    if not disjoint(output, Path(ct_path).resolve()) or not disjoint(output, Path(plan_path).resolve()):
        raise ValueError('Output must not contain CT or review-plan inputs')
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ('', '.', '..'):
        raise ValueError('Invalid segmentation subfolder')
    if (not class_map or any(not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 65535 for k in class_map)
            or len(set(class_map.values())) != len(class_map)):
        raise ValueError('class_map requires unique names and positive uint16 IDs')
    ids = {name: value for value, name in class_map.items()}
    if any(name not in ids for name in NAMES):
        raise ValueError('class_map must contain all 24 vertebrae')
    ct = nib.load(str(ct_path))
    spacing = check_ct(ct)
    raw = load_labels(source, ct, subfolder_name)
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan, raw, ct)
    # Validate additional masks before doing expensive work.
    extras = []
    for path in (source / subfolder_name).glob('*.nii.gz'):
        if path.name[:-7] not in NAMES:
            read_binary(path, ct)
            extras.append(path)
    candidate, report = engine.refine(raw, spacing, abs(float(np.linalg.det(ct.affine[:3, :3]))), plan)
    del raw
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '-components-', dir=str(output.parent)))
    try:
        segdir = stage / subfolder_name
        segdir.mkdir()
        for label, name in enumerate(NAMES, 1):
            reference = nib.load(str(source / subfolder_name / (name + '.nii.gz')))
            save_verified((candidate == label).astype(np.uint8), reference, ct, segdir / (name + '.nii.gz'))
        for path in extras:
            shutil.copy2(str(path), str(segdir / path.name))
        combined = np.zeros(candidate.shape, dtype=np.uint8 if max(class_map) <= 255 else np.uint16)
        for label_id, name in sorted(class_map.items()):
            if name in NAMES:
                mask = candidate == NAMES.index(name) + 1
            else:
                path = segdir / (name + '.nii.gz')
                if not path.exists():
                    continue
                _, mask = read_binary(path, ct)
            combined[mask] = label_id
        reference_path = source / 'combined_labels.nii.gz'
        reference = nib.load(str(reference_path)) if reference_path.is_file() else ct
        check_grid(reference, ct)
        save_verified(combined, reference, ct, stage / 'combined_labels.nii.gz')
        report.update(engine='shapekit_components', review_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                      source_sha256=plan['source_sha256'],
                      engine_sha256=hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
                      runtime_seconds=time.monotonic() - started,
                      label_map={str(i): ids[name] for i, name in enumerate(NAMES, 1)},
                      accuracy_evaluation='Not measured by this run',
                      scope='Reviewed corrections; no automatic vertebral identification')
        (stage / 'vertebrae_components_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        stage.rename(output)
    except Exception:
        # Only the unique directory created by this function is removed.
        shutil.rmtree(str(stage))
        raise
    return report
