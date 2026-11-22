"""Run an explicitly supplied independent ordered-core source in ShapeKit."""

import json
import runpy
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from tools import import_vertebrae
from tools.import_vertebrae import NAMES, disjoint, import_case, sha256, validate_source

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
    }


def verify_resume(output, current):
    manifest = output / "workflow_report.json"
    if not manifest.is_file():
        raise ValueError("E_RESUME: no completed ordered-core workflow report")
    report = json.loads(manifest.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("fingerprint") != current:
        raise ValueError("E_RESUME: raw inputs, configuration or source changed")
    digests = report["output_sha256"]
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
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not disjoint(source, output) or (
        ct_root is not None and not disjoint(Path(ct_root).resolve(), output)
    ):
        raise ValueError("E_PATH: output root must be disjoint from input and CT roots")
    failed = []
    for case in sorted(cases):
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
            print(
                json.dumps(
                    {
                        "case": case,
                        "status": "complete",
                        "resumed": report.get("resumed", False),
                        "identity_applied": report["identity"]["applied"],
                    }
                ),
                flush=True,
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            ImportError,
            SyntaxError,
            MemoryError,
            nib.filebasedimages.ImageFileError,
        ) as error:
            failed.append(case)
            print(
                json.dumps({"case": case, "status": "failed", "error": str(error)}),
                flush=True,
            )
    return failed
