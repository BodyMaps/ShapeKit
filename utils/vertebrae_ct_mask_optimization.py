"""ShapeKit adapter for the shapekit_ct_mask_optimization vertebrae engine.

The engine (vertebrae_ct_mask_optimization_engine.py, an unchanged copy of the reference implementation) corrects
the 24 vertebra masks C1-L5 of one case against the case's CT: it renumbers the vertebral bodies, moves arches and
spinous processes to the vertebra they belong to, and cleans up stray and detached pieces. This adapter connects it
to ShapeKit's case layout:

    <input_path>/<subfolder_name>/vertebrae_L5.nii.gz ... vertebrae_C1.nii.gz   one binary mask per vertebra
    <input_path>/<subfolder_name>/<other organ>.nii.gz                         copied unchanged
    ct_path                                                                    the CT the masks were predicted on

It writes <output_path>/<subfolder_name>/ with every input mask (vertebrae corrected, other organs unchanged),
<output_path>/combined_labels.nii.gz with the ShapeKit label IDs from class_map, and a JSON report
(vertebrae_ct_mask_optimization_report.json) with the engine's log, its status and the run details.

Checks before any work (a failing case raises an error; ShapeKit reports it and continues with the other cases):
  - input, output and CT locations are separate, and the output case folder does not exist yet;
  - class_map holds all 24 vertebrae with unique IDs;
  - every mask is a binary 3D volume on one voxel grid, vertebra masks do not overlap, and the CT is on the same
    grid. Any orientation is accepted (the engine reorients internally); sheared affines are refused;
  - a missing vertebra mask is treated as empty and listed in the report.
Missing CT: the case fails by default (missing_ct="fail"). With missing_ct="shapekit_songlin" the vertebrae are
processed by ShapeKit's mask-only shapekit_songlin engine instead, and the report says so.

The output is written to a staging folder next to the output and renamed only when complete, so a failed case
never leaves a partial output folder. The engine runs single-threaded (no thread settings are needed per worker)
and is deterministic: the same input gives byte-identical output, also inside a multiprocessing pool.
"""
import hashlib
import json
import logging
import platform
import shutil
import tempfile
import time
from importlib import metadata
from pathlib import Path

import nibabel as nib
import numpy as np

from . import vertebrae_ct_mask_optimization_engine as engine

ENGINE_NAME = "shapekit_ct_mask_optimization"
VERTEBRA_NAMES = tuple(f"vertebrae_{level}" for level in engine.LEVELS)   # internal label i + 1 = VERTEBRA_NAMES[i]
REPORT_FILE = "vertebrae_ct_mask_optimization_report.json"
MISSING_CT_POLICIES = ("fail", "shapekit_songlin")
PACKAGES = ("numpy", "scipy", "nibabel", "connected-components-3d", "scikit-image")


def package_versions():
    """Python and the engine's package versions, for the report."""
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def check_affine(image, name):
    """Refuse non-finite affines, non-positive voxel sizes and sheared (non-orthogonal) voxel axes."""
    spacing = np.asarray(image.header.get_zooms()[:3], float)
    if not np.all(np.isfinite(image.affine)) or np.any(spacing <= 0):
        raise ValueError(f"{name}: a finite affine and positive voxel sizes are required")
    directions = image.affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ValueError(f"{name}: sheared voxel axes are not supported")


def check_grid(image, reference, name):
    """Refuse a volume whose shape or affine differs from the reference mask's."""
    if (len(image.shape) != 3 or image.shape != reference.shape
            or not np.allclose(image.affine, reference.affine, rtol=0, atol=engine.GRID_TOLERANCE_MM)):
        raise ValueError(f"{name}: not on the same voxel grid as the vertebra masks")


def read_binary(image, name):
    """The mask as a boolean array; refuse anything that is not a 3D volume of 0s and 1s."""
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.all((data == 0) | (data == 1)):
        raise ValueError(f"{name}: a binary 3D mask (values 0 and 1 only) is required")
    return data.astype(bool)


def load_vertebrae(segmentation_dir):
    """Read the 24 vertebra masks into one internal label map (1 = L5 ... 24 = C1).

    Returns:
        (reference, labels, missing): the first mask's image (the voxel grid of the case), the uint8 label map,
        and the names of the vertebra masks that were not found (treated as empty).
    """
    reference, labels, missing = None, None, []
    for value, name in enumerate(VERTEBRA_NAMES, start=1):
        path = segmentation_dir / f"{name}.nii.gz"
        if not path.is_file():
            missing.append(name)
            continue
        image = nib.load(path)
        mask = read_binary(image, name)
        if reference is None:
            check_affine(image, name)
            reference, labels = image, np.zeros(image.shape, np.uint8)
        check_grid(image, reference, name)
        if np.any(mask & (labels != 0)):
            raise ValueError(f"{name}: overlaps another vertebra mask")
        labels[mask] = value
    if reference is None:
        raise ValueError(f"no vertebra masks found in {segmentation_dir}")
    return reference, labels, missing


def save_like(data, reference, path):
    """Save `data` on the reference's grid, keeping its header, qform and sform."""
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    image = nib.Nifti1Image(data, reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, path)


def separate(a, b):
    """True when neither path is the other or lies inside it."""
    return a != b and a not in b.parents and b not in a.parents


def run_songlin(labels, case_name):
    """The mask-only fallback: ShapeKit's shapekit_songlin engine, on ShapeKit's own vertebra IDs."""
    from .vertebrae_iterative import postprocessing_vertebrae as songlin
    masks = {name: (labels == value).astype(np.uint8) for value, name in enumerate(VERTEBRA_NAMES, start=1)}
    processed = songlin(case_name, masks, logger=logging)
    refined = np.zeros(labels.shape, np.uint8)
    for value, name in enumerate(VERTEBRA_NAMES, start=1):
        refined[processed[name] > 0] = value
    return refined


def process_case(input_path, output_path, ct_path, class_map, subfolder_name="segmentations", missing_ct="fail"):
    """Correct one case's vertebra masks and write its output folder (see the module docstring).

    Args:
        input_path: the case folder (holding <subfolder_name>/).
        output_path: the output case folder to create; it must not exist yet.
        ct_path: the case's CT, on the masks' voxel grid.
        class_map: ShapeKit's {label ID: mask name}, holding all 24 vertebrae.
        subfolder_name: the masks' folder name inside the case folder.
        missing_ct: "fail" (default) or "shapekit_songlin" (mask-only fallback when the CT is missing).

    Returns:
        The report written to the output folder, as a dict.
    """
    started = time.monotonic()
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if missing_ct not in MISSING_CT_POLICIES:
        raise ValueError(f"missing_ct must be one of {MISSING_CT_POLICIES}, got {missing_ct!r}")
    if not separate(source, output):
        raise ValueError("input and output case folders must be separate")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ("", ".", ".."):
        raise ValueError(f"invalid segmentation subfolder name: {subfolder_name!r}")
    if (any(not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 65535 for k in class_map)
            or len(set(class_map.values())) != len(class_map)):
        raise ValueError("class_map needs unique mask names and integer IDs from 1 to 65535")
    ids = {name: label_id for label_id, name in class_map.items()}
    if any(name not in ids for name in VERTEBRA_NAMES):
        raise ValueError("class_map must include all 24 vertebrae (vertebrae_L5 ... vertebrae_C1)")
    ct_found = ct_path is not None and Path(ct_path).is_file()
    if not ct_found and missing_ct == "fail":
        raise FileNotFoundError(f"CT not found: {ct_path} (the engine needs the case's CT)")
    if ct_found:
        ct_path = Path(ct_path).resolve()
        if not separate(output, ct_path):
            raise ValueError("the output folder must not contain the CT")

    segmentation_dir = source / subfolder_name
    reference, labels, missing = load_vertebrae(segmentation_dir)
    if ct_found:
        ct_image = nib.load(ct_path)
        check_grid(ct_image, reference, f"CT {ct_path}")
        del ct_image
    others = []                                 # other organs' masks, copied unchanged
    for path in sorted(segmentation_dir.glob("*.nii.gz")):
        if path.name[:-len(".nii.gz")] not in VERTEBRA_NAMES:
            image = nib.load(path)
            check_grid(image, reference, path.name)
            read_binary(image, path.name)
            others.append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / ".incomplete"
    staging_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=source.name + "-", dir=staging_root))
    try:
        if ct_found:
            with tempfile.TemporaryDirectory(prefix="ct_mask_optimization-") as temp:
                labels_path = Path(temp) / engine.COMBINED_FILE
                save_like(labels, reference, labels_path)
                refined, log, status = engine.refine_labels(str(labels_path), str(ct_path))
        else:
            refined = run_songlin(labels, source.name)
            log, status = [f"CT not found ({ct_path}); the vertebrae were processed by the mask-only "
                           "shapekit_songlin engine instead"], "CT missing: shapekit_songlin applied"
        for line in log:                        # debug level: the full log is also in the case's report
            logging.debug(f"[{ENGINE_NAME}] {source.name}: {line}")

        output_dir = stage / subfolder_name
        output_dir.mkdir()
        for path in others:
            shutil.copy2(path, output_dir / path.name)
        combined = np.zeros(refined.shape, np.uint8 if max(class_map) <= 255 else np.uint16)
        for label_id, name in sorted(class_map.items()):
            if name in VERTEBRA_NAMES:
                mask = refined == VERTEBRA_NAMES.index(name) + 1
                save_like(mask.astype(np.uint8), reference, output_dir / f"{name}.nii.gz")
            elif (output_dir / f"{name}.nii.gz").exists():
                mask = read_binary(nib.load(output_dir / f"{name}.nii.gz"), name)
            else:
                continue
            combined[mask] = label_id
        save_like(combined, reference, stage / engine.COMBINED_FILE)

        voxel_cm3 = float(np.prod(reference.header.get_zooms()[:3])) / 1000
        report = {
            "case": source.name,
            "engine": ENGINE_NAME,
            "status": status,
            "log": log,
            "input": str(segmentation_dir),
            "ct": str(ct_path) if ct_found else None,
            "missing_input_masks": missing,
            "changed_cm3": round(float((labels != refined).sum()) * voxel_cm3, 3),
            "labelled_cm3_before": round(float((labels > 0).sum()) * voxel_cm3, 3),
            "labelled_cm3_after": round(float((refined > 0).sum()) * voxel_cm3, 3),
            "label_map": {str(value): ids[name] for value, name in enumerate(VERTEBRA_NAMES, start=1)},
            "orientation": "".join(nib.aff2axcodes(reference.affine)),
            "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
            "versions": package_versions(),
            "runtime_seconds": round(time.monotonic() - started, 1),
        }
        (stage / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    except Exception:
        logging.exception("incomplete case left at %s", stage)
        raise
    return report
