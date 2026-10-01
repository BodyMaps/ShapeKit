"""Opt-in, native-grid adapter for Hao Yu's two-stage vertebrae refinement.

The engine is preserved byte-for-byte from the evaluated standalone script.
This adapter changes IO, not the algorithm. It deliberately bypasses the
legacy mask reorientation, organ cleanup, and skip-empty-mask writer.
"""

import json
import logging
from pathlib import Path
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np

from . import vertebrae_hao_engine as engine

VERTEBRA_NAMES = engine.VERTEBRA_NAMES


def _check_grid(image, reference, name):
    if (len(image.shape) != 3 or image.shape != reference.shape
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4)
            or not np.allclose(image.header.get_zooms()[:3],
                               reference.header.get_zooms()[:3], rtol=0, atol=1e-5)):
        raise ValueError(f"Geometry mismatch: {name}; resampling is not performed")


def _binary(image, name):
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f"Finite binary 3D mask required: {name}")
    return data.astype(bool, copy=False)


def load_vertebrae(case_dir, subfolder_name='segmentations'):
    """Read named binary masks without reorientation; reject grid/overlap errors."""
    folder = Path(case_dir) / subfolder_name
    reference = None
    raw = None
    missing = []
    for label, name in enumerate(VERTEBRA_NAMES, 1):
        path = folder / f'{name}.nii.gz'
        if not path.is_file():
            missing.append(name)
            continue
        image = nib.load(str(path))
        mask = _binary(image, name)
        if reference is None:
            reference = image
            raw = np.zeros(image.shape, dtype=np.uint8)
        _check_grid(image, reference, name)
        if np.any(mask & (raw > 0)):
            raise ValueError(f'Overlapping vertebra masks: {name}')
        raw[mask] = label
    if reference is None:
        raise ValueError(f'No vertebra masks found in {folder}')
    return reference, raw, missing


def refine_labels(raw, reference_img, ct_path, logger=None):
    """Run exactly the submitted v2 baseline followed by anatomical refinement."""
    logger = logger or logging.getLogger(__name__)
    if ct_path is None or not Path(ct_path).is_file():
        raise FileNotFoundError(f'Matching calibrated CT required: {ct_path}')
    ct_image = nib.load(str(ct_path))
    _check_grid(ct_image, reference_img, str(ct_path))
    # get_fdata applies NIfTI scaling; do not clip or cast HU to int16.
    ct = ct_image.get_fdata(dtype=np.float32)
    baseline = engine.run_v2(raw, np.asarray(reference_img.header.get_zooms()[:3], float))
    cfg = engine.Config()
    final, report, debug = engine.refine(
        raw, baseline, ct, reference_img.affine, cfg,
        progress=lambda message: logger.info('[ShapeKit-Hao] %s', message))
    del ct, debug
    checks = {str(label): bool(np.array_equal(final == label, baseline == label))
              for label in range(1, 25) if not cfg.first <= label <= cfg.last}
    if not all(checks.values()):
        raise AssertionError('A protected binary mask changed')
    before = np.bincount(baseline.ravel(), minlength=25)
    after = np.bincount(final.ravel(), minlength=25)
    if np.any((before[1:] > 0) & (after[1:] == 0)):
        raise AssertionError('A previously present label disappeared')
    report.update(protected_binary_masks_exact=checks,
                  label_counts_v2=before.tolist(), label_counts_final=after.tolist())
    return final, report


def _disjoint(output, source):
    return output != source and output not in source.parents and source not in output.parents


def process_case(input_path, output_path, ct_path, class_map,
                 subfolder_name='segmentations'):
    """Process one case into a NEW directory; publish only after all writes succeed.

    Per-vertebra files stay binary. The combined map uses ShapeKit's class_map.
    Unrelated segmentation files are copied unchanged, never post-processed.
    Missing CT, inconsistent grids or conflicting masks fail instead of silently
    falling back to another algorithm. Partial runs remain under .incomplete.
    """
    started = time.monotonic()
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if not _disjoint(output, source):
        raise ValueError('Input and output directories must be disjoint')
    if ct_path is not None and not _disjoint(output, Path(ct_path).resolve()):
        raise ValueError('Output must not contain or replace the source CT')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ('', '.', '..'):
        raise ValueError('subfolder_name must be a single directory name')
    if (any(not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= 65535
            for i in class_map)
            or len(set(class_map.values())) != len(class_map)):
        raise ValueError('class_map requires unique names and positive uint16 IDs')
    name_to_id = {name: int(i) for i, name in class_map.items()}
    if any(name not in name_to_id for name in VERTEBRA_NAMES):
        raise ValueError('class_map must contain all 24 vertebra names')

    logger = logging.getLogger(__name__)
    logger.info('[ShapeKit-Hao] %s: loading native-grid masks', source.name)
    reference, raw, missing = load_vertebrae(source, subfolder_name)
    # Verify all other masks before expensive work; no legacy reorientation.
    extras = []
    for path in sorted((source / subfolder_name).glob('*.nii.gz')):
        name = path.name[:-7]
        if name not in VERTEBRA_NAMES:
            image = nib.load(str(path))
            _check_grid(image, reference, name)
            _binary(image, name)
            extras.append((name, path))
    final, report = refine_labels(raw, reference, ct_path, logger)
    del raw

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / '.incomplete'
    staging_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '-', dir=staging_root))
    segdir = staging / subfolder_name
    segdir.mkdir()
    dtype = np.uint8 if max(class_map) <= 255 else np.uint16
    combined = np.zeros(final.shape, dtype=dtype)
    try:
        for name, path in extras:
            shutil.copy2(path, segdir / path.name)
        # Reproduce ShapeKit's ascending-ID overlap priority for combined output.
        for label_id, name in sorted(class_map.items()):
            if name in VERTEBRA_NAMES:
                internal_id = VERTEBRA_NAMES.index(name) + 1
                mask = final == internal_id
                engine.save_volume(mask.astype(np.uint8), reference,
                                   segdir / f'{name}.nii.gz')
            else:
                path = segdir / f'{name}.nii.gz'
                if not path.is_file():
                    continue
                mask = _binary(nib.load(str(path)), name)
            combined[mask] = label_id
        engine.save_volume(combined, reference, staging / 'combined_labels.nii.gz')
        report.update(
            integration_status='complete', case=source.name,
            missing_input_masks=missing,
            label_map={str(i): name_to_id[n] for i, n in enumerate(VERTEBRA_NAMES, 1)},
            engine_sha256=engine.sha256(Path(engine.__file__)),
            adapter_sha256=engine.sha256(Path(__file__)),
            runtime_seconds=round(time.monotonic() - started, 2),
            geometry_policy='Native grid, no resampling or orientation change',
            dsc=None)
        (staging / 'vertebrae_hao_report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        if output.exists():
            raise FileExistsError(f'Output appeared during processing: {output}')
        staging.rename(output)
    except Exception:
        logger.exception('Incomplete output retained for diagnosis: %s', staging)
        raise
    logger.info('[ShapeKit-Hao] %s: %s; %s', source.name,
                report['status'], report['reason'])
    return report
