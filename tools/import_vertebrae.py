"""Import standalone vertebra predictions into ShapeKit without refining them."""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

NAMES = tuple(
    [f"vertebrae_L{i}" for i in range(5, 0, -1)]
    + [f"vertebrae_T{i}" for i in range(12, 0, -1)]
    + [f"vertebrae_C{i}" for i in range(7, 0, -1)]
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_grid(image, reference, name):
    """Reject ambiguous geometry, including conflicting coded NIfTI transforms."""
    spacing = np.asarray(image.header.get_zooms()[:3])
    if (
        len(image.shape) != 3
        or image.shape != reference.shape
        or not np.all(np.isfinite(image.affine))
        or np.any(~np.isfinite(spacing))
        or np.any(spacing <= 0)
        or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10
        or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4)
        or not np.allclose(spacing, reference.header.get_zooms()[:3], rtol=0, atol=1e-5)
    ):
        raise ValueError(f"E_GEOMETRY: {name}; no resampling is performed")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if qcode and scode and not np.allclose(qform, sform, rtol=0, atol=1e-4):
        raise ValueError(f"E_TRANSFORMS: conflicting qform/sform in {name}")


def save_like(array, reference, path):
    header = reference.header.copy()
    header.set_data_dtype(array.dtype)
    image = nib.Nifti1Image(array, reference.affine, header)
    for name in ("qform", "sform"):
        matrix, code = getattr(reference, "get_" + name)(coded=True)
        getattr(image, "set_" + name)(matrix, int(code))
    nib.save(image, path)


def disjoint(a, b):
    return a != b and a not in b.parents and b not in a.parents


def validate_class_map(class_map):
    if (
        not class_map
        or any(type(i) is not int or not 1 <= i <= 65535 for i in class_map)
        or any(not isinstance(name, str) for name in class_map.values())
        or len(set(class_map.values())) != len(class_map)
    ):
        raise ValueError(
            "E_MAPPING: unique names and positive uint16 class IDs required"
        )
    target_ids = {name: i for i, name in class_map.items()}
    if not set(NAMES) <= set(target_ids):
        raise ValueError("E_MAPPING: class_map must include all 24 vertebra names")
    return target_ids


def validate_source(source, mask_paths, target_ids, label_offset, ct_path):
    """Read and validate source masks. Return the remapped data and empty levels."""
    combined_path = source / "combined_labels.nii.gz"
    reference = nib.load(combined_path)
    check_grid(reference, reference, combined_path.name)
    raw = np.asanyarray(reference.dataobj)
    if raw.dtype.kind not in "ui" or not set(np.unique(raw)) <= {
        0,
        *range(1 + label_offset, 25 + label_offset),
    }:
        raise ValueError(
            "E_LABELS: expected integer vertebra-only source IDs for the declared offset"
        )
    if ct_path is not None:
        check_grid(nib.load(ct_path), reference, str(ct_path))
    lookup = np.zeros(50, np.uint16)
    for i, name in enumerate(NAMES, 1 + label_offset):
        lookup[i] = target_ids[name]
    combined = lookup[raw]
    del raw
    missing = []
    for name in NAMES:
        if not (source / "segmentations" / f"{name}.nii.gz").is_file():
            if np.any(combined == target_ids[name]):
                raise ValueError(
                    f"E_MISSING: populated class has no binary mask: {name}"
                )
            missing.append(name)
    for path in mask_paths:
        name = path.name[:-7]
        if name not in target_ids:
            raise ValueError(f"E_MAPPING: unknown mask {name}")
        image = nib.load(path)
        check_grid(image, reference, path.name)
        data = np.asanyarray(image.dataobj)
        if not np.all((data == 0) | (data == 1)):
            raise ValueError(f"E_BINARY: {name}")
        mask = data > 0
        if name in NAMES:
            if not np.array_equal(mask, combined == target_ids[name]):
                raise ValueError(
                    f"E_CONSISTENCY: binary/combined mismatch or overlap: {name}"
                )
        else:
            if np.any(mask & (combined != 0)):
                raise ValueError(
                    f"E_OVERLAP: unrelated mask overlaps another class: {name}"
                )
            combined[mask] = target_ids[name]

    return reference, combined, missing


def import_case(
    source, output, class_map, *, label_offset=0, ct_path=None, resume=False
):
    """Validate every input, then atomically publish a geometry-preserving export.

    No component filtering, interpolation, watershed or label-identity inference is
    performed. Missing empty classes get explicit zero masks. Input named binary
    files are copied byte-for-byte. Unrelated binary masks must not overlap.
    """
    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    if not disjoint(source, output):
        raise ValueError("E_PATH: input and output must be disjoint")
    if label_offset not in (0, 25):
        raise ValueError("E_MAPPING: label_offset must be 0 (SuPreM) or 25 (ShapeKit)")
    target_ids = validate_class_map(class_map)
    if ct_path is not None:
        ct_path = Path(ct_path).resolve()
        if not disjoint(output, ct_path):
            raise ValueError("E_PATH: output must not contain the source CT")
    combined_path = source / "combined_labels.nii.gz"
    mask_paths = sorted((source / "segmentations").glob("*.nii.gz"))
    if not combined_path.is_file() or not mask_paths:
        raise ValueError(
            "E_LAYOUT: combined_labels.nii.gz and segmentations/*.nii.gz required"
        )
    fingerprint = {
        "inputs": {
            str(p.relative_to(source)): sha256(p) for p in [combined_path, *mask_paths]
        },
        "class_map": {str(i): name for i, name in sorted(class_map.items())},
        "label_offset": label_offset,
        "ct_sha256": sha256(ct_path) if ct_path is not None else None,
        "importer_sha256": sha256(__file__),
    }
    if output.exists():
        if not resume:
            raise FileExistsError(f"E_EXISTS: refusing to overwrite {output}")
        report_path = output / "import_report.json"
        if not report_path.is_file():
            raise ValueError("E_RESUME: existing case has no completed report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("fingerprint") != fingerprint
            or report.get("status") != "complete"
        ):
            raise ValueError("E_RESUME: input, configuration or importer changed")
        actual_files = {
            str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()
        }
        if actual_files != set(report["output_sha256"]) | {"import_report.json"}:
            raise ValueError("E_RESUME: output file inventory changed")
        for name, digest in report["output_sha256"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"E_RESUME: corrupt output {name}")
        return {
            **report,
            "resumed": True,
            "runtime_seconds": time.perf_counter() - started,
        }

    reference, combined, missing = validate_source(
        source, mask_paths, target_ids, label_offset, ct_path
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete = output.parent / ".incomplete"
    incomplete.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + "-", dir=incomplete))
    # A failed write leaves a diagnostic staging directory, never a complete case.
    segdir = stage / "segmentations"
    segdir.mkdir()
    for path in mask_paths:
        shutil.copy2(path, segdir / path.name)
    for name in missing:
        save_like(
            np.zeros(reference.shape, np.uint8), reference, segdir / f"{name}.nii.gz"
        )
    dtype = np.uint8 if max(class_map) <= 255 else np.uint16
    save_like(combined.astype(dtype), reference, stage / "combined_labels.nii.gz")
    report = {
        "status": "complete",
        "case": source.name,
        "fingerprint": fingerprint,
        "missing_empty_classes": missing,
        "geometry_policy": "native grid; no resampling",
        "runtime_seconds": time.perf_counter() - started,
        "dsc": None,
        "output_sha256": {
            str(p.relative_to(stage)): sha256(p)
            for p in sorted(stage.rglob("*.nii.gz"))
        },
    }
    (stage / "import_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if output.exists():
        raise FileExistsError("E_EXISTS: output appeared while importing")
    stage.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument("--ct-root", type=Path)
    parser.add_argument("--label-offset", type=int, choices=(0, 25), default=0)
    parser.add_argument("--case", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    source, output = args.input_root.resolve(), args.output_root.resolve()
    if not disjoint(source, output) or (
        args.ct_root is not None and not disjoint(output, args.ct_root.resolve())
    ):
        parser.error("E_PATH: output root must be disjoint from input and CT roots")
    if not source.is_dir():
        parser.error("E_LAYOUT: input root not found")
    cases = sorted(
        p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if args.case:
        if not set(args.case) <= set(cases):
            parser.error("E_LAYOUT: requested case directory not found")
        cases = sorted(set(args.case))
    if not cases:
        parser.error("E_LAYOUT: no cases selected")
    class_map = yaml.safe_load(args.config.read_text(encoding="utf-8"))["class_map"]
    failed = False
    for case in cases:
        try:
            report = import_case(
                source / case,
                output / case,
                class_map,
                label_offset=args.label_offset,
                resume=args.resume,
                ct_path=args.ct_root / case / "ct.nii.gz" if args.ct_root else None,
            )
            print(
                json.dumps(
                    {
                        k: report.get(k)
                        for k in ("case", "status", "runtime_seconds", "resumed")
                    }
                )
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            nib.filebasedimages.ImageFileError,
        ) as error:
            failed = True
            print(json.dumps({"case": case, "status": "failed", "error": str(error)}))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
