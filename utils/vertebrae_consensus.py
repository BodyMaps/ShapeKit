"""Opt-in integration of Hongkang Chu's submitted multi-model warmup.

Inputs are predictions, never reference annotations. Raw mode runs the frozen
CPU postprocessing chain; prepared mode replays its final boundary stage.
Neural inference is deliberately external to ShapeKit.
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

from . import vertebrae_hao_engine as hao
from ._vertebrae_consensus import (
    native_anchor_pipeline as anchor_stage,
    body_root_guided_consistency as identity_stage,
    global_body_identity_graph as global_stage,
    global_body_offset_gate as offset_stage,
    consensus_boundary as boundary_stage,
)

NAMES = tuple(hao.VERTEBRA_NAMES)
LOGGER = logging.getLogger(__name__)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def disjoint(a, b):
    return a != b and a not in b.parents and b not in a.parents


def check_grid(image, reference, name, metadata=False):
    if (len(image.shape) != 3 or image.shape != reference.shape
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5)
            or not np.allclose(image.header.get_zooms()[:3],
                               reference.header.get_zooms()[:3], rtol=0, atol=1e-5)):
        raise ValueError(f'Native-grid mismatch: {name}; no resampling is performed')
    if metadata:
        if image.header.get_xyzt_units() != reference.header.get_xyzt_units():
            raise ValueError(f'Inconsistent mask units: {name}')
        for kind in ('qform', 'sform'):
            a, ac = getattr(image, 'get_' + kind)(coded=True)
            b, bc = getattr(reference, 'get_' + kind)(coded=True)
            if (ac != bc or (a is None) != (b is None)
                    or (a is not None and not np.allclose(a, b, rtol=0, atol=1e-5))):
                raise ValueError(f'Inconsistent mask {kind}: {name}')


def check_physical_grid(image):
    affine = image.affine
    spacing = np.asarray(image.header.get_zooms()[:3], float)
    if len(image.shape) != 3 or not np.isfinite(affine).all() or np.any(spacing <= 0):
        raise ValueError('Finite 3D affine and positive spacing required')
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('Sheared grids or inconsistent header spacing are unsupported')


def binary(image, name):
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f'Finite binary 3D mask required: {name}')
    return data.astype(bool, copy=False)


def load_masks(case, subfolder='segmentations'):
    """Use named masks, avoiding ambiguous combined-map label conventions."""
    folder = Path(case) / subfolder
    unknown = [p.name for p in folder.glob('vertebrae_*.nii*')
               if p.name not in {n + '.nii.gz' for n in NAMES}]
    if unknown:
        raise ValueError(f'Unsupported vertebra files (24-level contract): {unknown}')
    reference, labels, missing, hashes = None, None, [], {}
    for i, name in enumerate(NAMES, 1):
        path = folder / (name + '.nii.gz')
        if not path.is_file():
            missing.append(name)
            continue
        image = nib.load(path)
        check_physical_grid(image)
        mask = binary(image, name)
        if reference is None:
            reference = image
            # NIfTI voxels arrive in Fortran order. Assemble in that order to
            # avoid a strided whole-volume boolean assignment for every mask.
            labels = np.zeros(image.shape, np.uint8, order='F')
        check_grid(image, reference, name, metadata=True)
        if np.any(mask & (labels > 0)):
            raise ValueError(f'Overlapping vertebra masks: {name}')
        np.copyto(labels, np.uint8(i), where=mask)
        hashes[path.name] = digest(path)
    if reference is None:
        raise ValueError(f'No named vertebra masks in {folder}')
    # Numerical stages receive the same C-contiguous labels as the original
    # adapter; this conversion occurs once per model, rather than per mask.
    return reference, np.ascontiguousarray(labels), missing, hashes


def save_volume(array, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    image = nib.Nifti1Image(array, reference.affine, header)
    image.set_qform(*reference.get_qform(coded=True))
    image.set_sform(*reference.get_sform(coded=True))
    nib.save(image, path)


def run_pipeline(primary, suprem, v3, ct, reference, input_stage='raw', observer=None):
    """Run unchanged numerical functions; raw v3 bypasses Hao by design."""
    if input_stage not in ('raw', 'prepared'):
        raise ValueError('input_stage must be raw or prepared')
    stages = []

    def record(name, output, report, started):
        stages.append(dict(stage=name, elapsed_seconds=time.monotonic()-started, **report))
        LOGGER.info('[ShapeKit-Consensus] %s: %s, %s changed voxels',
                    name, report.get('status'), report.get('changed_voxels'))
        if observer is not None:
            observer(name, output)
        return output

    if input_stage == 'raw':
        direction = reference.affine[:3, :3] / nib.affines.voxel_sizes(reference.affine)
        if not (np.allclose(np.abs(direction), np.eye(3), atol=1e-5)
                and direction[1, 1] > 0 and direction[2, 2] > 0):
            raise ValueError('Raw pipeline requires native axis-aligned LAS/RAS; no orientation adapter')
        spacing = np.asarray(reference.header.get_zooms()[:3], float)

        def run_hao(raw, name):
            started = time.monotonic()
            baseline = hao.run_v2(raw, spacing)
            final, report, debug = hao.refine(raw, baseline, ct, reference.affine,
                                            hao.Config(), progress=LOGGER.info)
            del baseline, debug
            return record(name, final, report, started)

        suprem_hao = run_hao(suprem, 'suprem_hao')
        started = time.monotonic()
        anchors, report = anchor_stage.refine(suprem, suprem_hao, ct, reference.affine)
        anchors = record('suprem_anchors', anchors, report, started)
        del suprem_hao
        primary_hao = run_hao(primary, 'primary_hao')
        started = time.monotonic()
        incumbent, report = identity_stage.refine(primary_hao, ct, reference.affine, anchors)
        incumbent = record('incumbent', incumbent, report, started)
        del primary_hao
        started = time.monotonic()
        proposal, report = global_stage.refine(v3, anchors, ct, reference.affine)
        proposal = record('v3_proposal', proposal, report, started)
        started = time.monotonic()
        aligned, report = offset_stage.filter_proposal(v3, anchors, proposal, reference.affine)
        aligned = record('aligned_v3', aligned, report, started)
        del proposal
    else:
        incumbent, anchors, aligned = primary, suprem, v3

    started = time.monotonic()
    final, report = boundary_stage.refine(incumbent, anchors, aligned, ct,
                                          reference.affine, ct_guard=False)
    final = record('boundary_agreement', final, report, started)
    for label in (23, 24):
        if not np.array_equal(final == label, incumbent == label):
            raise AssertionError('Boundary stage changed C1/C2')
    if np.any((incumbent > 0) & (final > 0) & (final != incumbent)):
        raise AssertionError('Boundary stage changed a retained foreground identity')
    return final, stages


def process_case(input_path, output_path, ct_path, class_map,
                 subfolder_name='segmentations', *, suprem_path, v3_path,
                 input_stage='raw', save_stages=False):
    """Publish one case atomically after strict IO checks and full processing.

    Other organ masks are copied unchanged. Combined IDs follow class_map;
    the numerical engine always uses 1=L5 through 24=C1. Full-resolution arrays
    are written one at a time. A failed case never appears as completed output.
    """
    started = time.monotonic()
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    roots = {'primary': source, 'suprem': Path(suprem_path).resolve(),
             'v3': Path(v3_path).resolve()}
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if input_stage not in ('raw', 'prepared'):
        raise ValueError('input_stage must be raw or prepared')
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ('', '.', '..'):
        raise ValueError('subfolder_name must be one directory name')
    if ct_path is None or not Path(ct_path).is_file():
        raise FileNotFoundError(f'Matching calibrated CT required: {ct_path}')
    ct_path = Path(ct_path).resolve()
    for path in [*roots.values(), ct_path]:
        if not disjoint(output, path):
            raise ValueError('Output must be disjoint from every source')
    if (not class_map or any(not isinstance(k, int) or isinstance(k, bool)
                            or not 1 <= k <= 65535 for k in class_map)
            or any(not isinstance(n, str) for n in class_map.values())
            or len(set(class_map.values())) != len(class_map)):
        raise ValueError('class_map requires unique names and positive uint16 IDs')
    name_to_id = {name: i for i, name in class_map.items()}
    if not set(NAMES) <= name_to_id.keys():
        raise ValueError('class_map must include all 24 vertebra names')
    images, arrays, missing, hashes = {}, {}, {}, {}
    for role, path in roots.items():
        images[role], arrays[role], missing[role], hashes[role] = load_masks(path, subfolder_name)
    reference = images['primary']
    for role, image in images.items():
        check_grid(image, reference, role)
        if image.header.get_xyzt_units()[0] not in ('mm', 'unknown'):
            raise ValueError(f'Prediction grid must use millimeter coordinates: {role}')
    ct_image = nib.load(ct_path)
    check_physical_grid(ct_image)
    check_grid(ct_image, reference, 'CT')
    if ct_image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('CT must explicitly declare millimeter units')
    # NIfTI scaling is applied before HU-dependent stages. Preserve mask header
    # unit flags, including legacy unknown, only with a matching mm CT grid.
    ct = ct_image.get_fdata(dtype=np.float32)
    if not np.isfinite(ct).all():
        raise ValueError('Finite CT intensities required')
    hashes['CT'] = digest(ct_path)
    extras = []
    for path in sorted((source / subfolder_name).glob('*.nii.gz')):
        if path.name[:-7] not in NAMES:
            image = nib.load(path)
            check_grid(image, reference, path.name, metadata=True)
            binary(image, path.name)
            extras.append(path)
            hashes['primary'][path.name] = digest(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / '.incomplete'
    staging_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'-', dir=staging_root))

    def observer(name, array):
        if save_stages:
            folder = staging / 'stages'
            folder.mkdir(exist_ok=True)
            save_volume(array, reference, folder / (name+'.nii.gz'))

    try:
        final, stages = run_pipeline(arrays['primary'], arrays['suprem'], arrays['v3'],
                                     ct, reference, input_stage, observer)
        del ct, arrays
        segdir = staging / subfolder_name
        segdir.mkdir()
        for path in extras:
            shutil.copy2(path, segdir / path.name)
        dtype = np.uint8 if max(class_map) <= 255 else np.uint16
        combined = np.zeros(final.shape, dtype=dtype)
        for label_id, name in sorted(class_map.items()):
            if name in NAMES:
                mask = final == NAMES.index(name)+1
                save_volume(mask.astype(np.uint8), reference, segdir / (name+'.nii.gz'))
            else:
                path = segdir / (name+'.nii.gz')
                if not path.is_file():
                    continue
                mask = binary(nib.load(path), name)
            combined[mask] = label_id
        save_volume(combined, reference, staging / 'combined_labels.nii.gz')
        source_files = [Path(__file__), Path(hao.__file__),
                        *sorted(Path(__file__).with_name('_vertebrae_consensus').glob('*.py'))]
        report = dict(
            integration_status='complete', backend='shapekit_consensus',
            case=source.name, input_stage=input_stage, ct_guard=False,
            scope='CPU postprocessing only; upstream neural predictions are supplied',
            missing_input_masks=missing, input_sha256=hashes,
            numerical_source_sha256={p.name:digest(p) for p in source_files},
            label_map={str(i):name_to_id[n] for i,n in enumerate(NAMES,1)},
            prediction_units={r:list(im.header.get_xyzt_units()) for r,im in images.items()},
            physical_units_basis='Matching CT explicitly declares mm; prediction unit flags preserved',
            stage_reports=stages, reference_labels_used=False, dsc=None,
            limitations='Two-case development gain did not generalize to the additional three-case cohort',
            runtime_seconds=time.monotonic()-started)
        (staging / 'vertebrae_consensus_report.json').write_text(json.dumps(report,indent=2)+'\n')
        if output.exists():
            raise FileExistsError(f'Output appeared during processing: {output}')
        staging.rename(output)
    except Exception:
        LOGGER.exception('Incomplete case retained at %s', staging)
        raise
    return report
