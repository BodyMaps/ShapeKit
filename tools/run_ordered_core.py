"""Run an explicitly supplied independent ordered-core source in ShapeKit."""

import json
import logging
import platform
import runpy
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

from tools import import_vertebrae
from tools.import_vertebrae import (
    NAMES,
    check_grid,
    disjoint,
    import_case,
    sha256,
    validate_class_map,
    validate_source,
)

SOURCE_IDS = {name: i for i, name in enumerate(NAMES, 1)}


def fingerprint(source, mask_paths, implementation, class_map, ct):
    return {
        "raw_sha256": {
            p.relative_to(source).as_posix(): sha256(p)
            for p in [source / "combined_labels.nii.gz", *mask_paths]
        },
        "implementation_sha256": sha256(implementation),
        "adapter_sha256": sha256(__file__),
        "importer_sha256": sha256(import_vertebrae.__file__),
        "ct_sha256": sha256(ct) if ct is not None else None,
        "class_map": {str(i): name for i, name in sorted(class_map.items())},
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            **{
                name: version(name)
                for name in ("numpy", "scipy", "scikit-image", "nibabel")
            },
        },
    }


def verify_resume(output, current):
    manifest = output / "workflow_report.json"
    if not manifest.is_file():
        raise ValueError("E_RESUME: no completed ordered-core workflow report")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("fingerprint") != current
    ):
        raise ValueError("E_RESUME: raw inputs, configuration or source changed")
    digests = report.get("output_sha256")
    identity = report.get("identity")
    if (
        not isinstance(digests, dict)
        or not isinstance(identity, dict)
        or type(identity.get("applied")) is not bool
    ):
        raise ValueError("E_RESUME: malformed workflow report")
    actual = {
        p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
    }
    if actual != set(digests) | {manifest.name}:
        raise ValueError("E_RESUME: output inventory changed")
    if any(sha256(output / name) != digest for name, digest in digests.items()):
        raise ValueError("E_RESUME: output checksum mismatch")
    return {**report, "resumed": True}


def run_case(source, output, implementation, class_map, *, ct=None, resume=False):
    """Run the original numerical stages and publish a validated ShapeKit case."""
    source, output = Path(source).resolve(), Path(output).resolve()
    implementation = Path(implementation).resolve()
    if not disjoint(source, output) or not disjoint(implementation, output):
        raise ValueError("E_PATH: output must be disjoint from raw masks and source")
    if ct is not None:
        ct = Path(ct).resolve()
        if not disjoint(ct, output):
            raise ValueError("E_PATH: output must be disjoint from CT")
    if not implementation.is_file():
        raise ValueError("E_IMPLEMENTATION: explicit local Python source is required")
    validate_class_map(class_map)
    combined_path = source / "combined_labels.nii.gz"
    mask_paths = sorted((source / "segmentations").glob("*.nii.gz"))
    if not combined_path.is_file() or not mask_paths:
        raise ValueError("E_LAYOUT: expected combined labels and vertebra masks")
    current = fingerprint(source, mask_paths, implementation, class_map, ct)
    if output.exists():
        if resume:
            return verify_resume(output, current)
        raise FileExistsError(f"E_EXISTS: refusing to overwrite {output}")

    reference, validated, _ = validate_source(source, mask_paths, SOURCE_IDS, 0, ct)
    del validated
    algorithm = runpy.run_path(str(implementation))
    needed = ("get_array", "refine_ordered_cores", "process_case_cleanup", "save_like")
    if any(not callable(algorithm.get(name)) for name in needed):
        raise ValueError("E_IMPLEMENTATION: source must expose " + ", ".join(needed))
    mapping = {name.removeprefix("vertebrae_"): i for name, i in SOURCE_IDS.items()}

    incomplete = output.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{output.name}-", dir=incomplete
    ) as folder:
        work = Path(folder)
        raw = algorithm["get_array"](reference)
        refined, details = algorithm["refine_ordered_cores"](reference, raw, mapping)
        cleanup = algorithm["process_case_cleanup"](source, work / "refined", mapping)
        predictions = work / "refined" / source.name
        if details["applied"]:
            cleaned = algorithm["get_array"](nib.load(predictions / combined_path.name))
            refined[(raw > 0) & (cleaned == 0)] = 0
            added = (raw == 0) & (cleaned > 0)
            refined[added] = cleaned[added]
            details["label_to_label_voxels"] = int(
                np.count_nonzero((raw > 0) & (refined > 0) & (raw != refined))
            )
            details["changed_voxels"] = int(np.count_nonzero(raw != refined))
            del cleaned, added
            save = algorithm["save_like"]
            save(
                reference,
                refined,
                predictions / combined_path.name,
                reference.get_data_dtype(),
            )
            for path in mask_paths:
                mask_reference = nib.load(path)
                save(
                    mask_reference,
                    refined == SOURCE_IDS[path.name[:-7]],
                    predictions / "segmentations" / path.name,
                    mask_reference.get_data_dtype(),
                )
        del refined, raw
        result_grid = nib.load(predictions / combined_path.name)
        check_grid(result_grid, reference, "refined combined map")
        for transform in ("qform", "sform"):
            before, code = getattr(reference, f"get_{transform}")(coded=True)
            after, out_code = getattr(result_grid, f"get_{transform}")(coded=True)
            if code != out_code or (
                code and not np.allclose(before, after, rtol=0, atol=1e-4)
            ):
                raise ValueError(
                    f"E_GEOMETRY: refinement changed the native {transform}"
                )
        imported = work / "export"
        import_case(predictions, imported, class_map, ct_path=ct)
        latest_masks = sorted((source / "segmentations").glob("*.nii.gz"))
        if fingerprint(source, latest_masks, implementation, class_map, ct) != current:
            raise ValueError("E_INPUT_CHANGED: source changed while processing")
        report = {
            "status": "complete",
            "case": source.name,
            "fingerprint": current,
            "identity": details,
            "cleanup": cleanup,
            "output_sha256": {
                p.relative_to(imported).as_posix(): sha256(p)
                for p in sorted(imported.rglob("*"))
                if p.is_file()
            },
        }
        (imported / "workflow_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError("E_EXISTS: output appeared while processing")
        imported.rename(output)
    return report


def _case_summary(
    case, *, source, output, implementation, class_map, ct_root, ct_filename, resume
):
    """Keep external implementation exceptions local to the affected case."""
    started = perf_counter()
    ct = source / case / ct_filename
    if not ct.is_file():
        ct = Path(ct_root) / case / ct_filename if ct_root else None
    try:
        report = run_case(
            source / case,
            output / case,
            implementation,
            class_map,
            ct=ct,
            resume=resume,
        )
        result = {
            "status": "complete",
            "resumed": report.get("resumed", False),
            "identity_applied": report["identity"]["applied"],
        }
    except (Exception, SystemExit) as error:
        logging.getLogger(__name__).exception("Ordered-core case failed: %s", case)
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "case": case,
        **result,
        "elapsed_seconds": round(perf_counter() - started, 6),
    }


def run_batch(
    cases,
    source,
    output,
    implementation,
    class_map,
    *,
    ct_root=None,
    ct_filename="ct.nii.gz",
    resume=False,
    workers=1,
    event_log=None,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not disjoint(source, output) or (
        ct_root is not None and not disjoint(Path(ct_root).resolve(), output)
    ):
        raise ValueError("E_PATH: output root must be disjoint from input and CT roots")
    if workers < 1:
        raise ValueError("E_WORKERS: worker count must be at least 1")
    validate_class_map(class_map)
    cases = sorted(cases)
    workers = min(workers, len(cases))
    if sys.platform == "win32":
        workers = min(workers, 61)
    if not workers:
        raise ValueError("E_LAYOUT: no cases selected")
    process = partial(
        _case_summary,
        source=source,
        output=output,
        implementation=Path(implementation).resolve(),
        class_map=class_map,
        ct_root=ct_root,
        ct_filename=ct_filename,
        resume=resume,
    )
    failed = []

    def record(event):
        line = json.dumps(event)
        if event_log is not None:
            with Path(event_log).open("a", encoding="utf-8") as log:
                log.write(line + "\n")
        print(line, flush=True)
        if event["status"] in {"failed", "batch_failed"}:
            failed.append(event["case"])

    record({"case": None, "status": "started", "cases": len(cases), "workers": workers})
    if workers == 1:
        for case in cases:
            record(process(case))
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            ) as pool:
                pending = {pool.submit(process, case) for case in cases}
                for future in as_completed(pending):
                    record(future.result())
        except BrokenProcessPool as error:
            record({"case": None, "status": "batch_failed", "error": str(error)})
    return failed
