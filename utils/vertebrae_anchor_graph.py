"""ShapeKit adapter for the anchor-graph vertebrae engine (`shapekit_anchor_graph`).

The engine (`vertebrae_anchor_graph_engine.py`) is kept byte-identical to the evaluated
standalone script; this adapter only converts between ShapeKit's per-organ mask dict and the
engine's 1..24 label map, attaches the case CT when it is on the same grid, and writes a report.
"""

import hashlib
import json
import logging
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.arrayproxy import ArrayProxy

from . import vertebrae_anchor_graph_engine as engine

VERTEBRA_NAMES = dict(engine.VERTEBRAE)  # engine label (1 = L5 ... 24 = C1) -> ShapeKit name
REPORT_NAME = 'vertebrae_anchor_graph_report.json'
ENGINE_SHA256 = hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()


def _load_ct(ct_path: str | None, reference_img: nib.Nifti1Image, shape: tuple[int, ...],
             logger: logging.Logger, patient_id: str) -> np.ndarray | ArrayProxy | None:
    """CT on the mask grid (lazy dataobj), or None (mask-only mode) when missing, unreadable or misaligned."""
    if not ct_path or not Path(ct_path).exists():
        logger.info(f"[ShapeKit-AnchorGraph] {patient_id}: no CT at {ct_path}; running mask-only")
        return None
    try:
        ct_img = nib.load(ct_path)
    except Exception as e:  # an unreadable CT must not cost the mask-only cleanup
        logger.warning(f"[ShapeKit-AnchorGraph] {patient_id}: cannot read CT {ct_path} "
                       f"({type(e).__name__}: {e}); running mask-only")
        return None
    ct_shape = tuple(ct_img.shape)
    if (ct_shape[:3] != tuple(shape) or any(n != 1 for n in ct_shape[3:])
            or not np.allclose(ct_img.affine, reference_img.affine, atol=1e-3)):
        logger.warning(f"[ShapeKit-AnchorGraph] {patient_id}: CT grid {ct_img.shape} does not match masks "
                       f"{tuple(shape)}; running mask-only (no resampling is done)")
        return None
    if len(ct_shape) > 3:
        return np.asanyarray(ct_img.dataobj).reshape(shape)
    return ct_img.dataobj


def postprocessing_vertebrae_anchor_graph(patient_id: str, segmentation_dict: dict[str, np.ndarray],
                                          reference_img: nib.Nifti1Image, ct_path: str | None,
                                          logger: logging.Logger,
                                          report_dir: str | None = None) -> dict[str, np.ndarray]:
    """Refine the 24 vertebra masks in `segmentation_dict` in place and return the dict.

    All 24 vertebra entries are written back (empty ones as zero masks) so labels the engine
    removes or merges stay removed. If the engine fails, the input vertebrae are returned unchanged.
    """
    started = time.time()
    present = [k for k, n in VERTEBRA_NAMES.items()
               if segmentation_dict.get(n) is not None and np.any(segmentation_dict[n])]
    if not present:
        logger.info(f"[ShapeKit-AnchorGraph] {patient_id}: no vertebra masks, nothing to do")
        return segmentation_dict

    shape = segmentation_dict[VERTEBRA_NAMES[present[0]]].shape
    arr = np.zeros(shape, np.uint8)
    for k in present:
        arr[segmentation_dict[VERTEBRA_NAMES[k]] > 0] = k

    summary: dict[str, object] = {'case': patient_id, 'engine': 'shapekit_anchor_graph',
                                  'engine_sha256': ENGINE_SHA256}
    try:
        ct = _load_ct(ct_path, reference_img, shape, logger, patient_id)
        arr, report = engine.postprocess_array(arr, reference_img.affine, reference_img.header.get_zooms()[:3], ct=ct)
        summary['status'] = 'ok'
    except Exception as e:
        logger.error(f"[ShapeKit-AnchorGraph] {patient_id}: engine failed ({type(e).__name__}: {e}); "
                     f"vertebrae left unchanged\n{traceback.format_exc()}")
        summary.update(status='engine_failed', error=f'{type(e).__name__}: {e}')
        report = {}
    else:
        for k, name in VERTEBRA_NAMES.items():
            segmentation_dict[name] = (arr == k).astype(np.uint8)
        logger.info(f"[ShapeKit-AnchorGraph] {patient_id}: {len(report['changes'])} changes, "
                    f"{len(report['flags'])} flags, ct_used={report['ct_used']}")

    summary['runtime_s'] = round(time.time() - started, 1)
    if report_dir:
        report_path = Path(report_dir) / REPORT_NAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(dict(summary, **report), indent=1, default=engine._jsonable)
        tmp = report_path.with_name(report_path.name + '.tmp')
        tmp.write_text(text)
        tmp.replace(report_path)
    return segmentation_dict
