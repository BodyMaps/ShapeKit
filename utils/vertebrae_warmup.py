"""Expose the warm-up vertebrae methods through ShapeKit."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np

from . import vertebrae_warmup_engine as engine
from .vertebrae_hao import _binary, _check_grid, _disjoint, load_vertebrae
from .vertebrae_hao_engine import save_volume


def run_rooted(source, output, workspace, rooted):
    cache_path, ct_path, suprem_repo, python = rooted
    cache, ct, repo = (Path(path).resolve() for path in (cache_path, ct_path, suprem_repo))
    for path in (cache, ct, repo / "dataset" / "dataloader_test.py", repo / "utils" / "utils.py"):
        if not path.is_file():
            raise FileNotFoundError(f"Required rooted input missing: {path}")
    if any(not _disjoint(output, path) for path in (cache, ct)):
        raise ValueError("Output case must be disjoint from the cache and CT")

    interpreter = str(Path(python).resolve()) if python else sys.executable
    stage_root = workspace / "rooted"
    # The submitted inverse transform imports SuPreM's own utils package.
    # A subprocess gives it a separate import namespace from ShapeKit.
    subprocess.run([
        interpreter, "-I", "-c",
        "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
        str(Path(engine.__file__).resolve()), "rooted",
        "--cache", str(cache), "--ct", str(ct), "--suprem-repo", str(repo),
        "--output-root", str(stage_root),
    ], cwd=repo, check=True)
    native = stage_root / source.name
    reference = nib.load(ct)
    _check_grid(nib.load(native / "combined_labels.nii.gz"), reference, "rooted output")
    stage = workspace / "publish"
    stage.mkdir()
    for name in ("combined_labels.nii.gz", "segmentations"):
        (native / name).rename(stage / name)
    summary = json.loads((stage_root / "workflow_report.json").read_text(encoding="utf-8"))
    summary["native_segmentation"] = str(output / "combined_labels.nii.gz")
    (stage / "workflow_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return reference, stage


def process_case(input_path, output_path, class_map, subfolder_name="segmentations", *, rooted=None):
    """Run a submitted warm-up method and write a ShapeKit case."""
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if not _disjoint(source, output):
        raise ValueError("Input and output case directories must be disjoint")
    if output.exists():
        raise FileExistsError(f"Output case already exists: {output}")
    if Path(subfolder_name).name != subfolder_name or subfolder_name in ("", ".", ".."):
        raise ValueError("Invalid segmentation subfolder")
    if any(type(label) is not int or not 1 <= label <= 65535 for label in class_map):
        raise ValueError("class_map must use positive uint16 label IDs")
    internal_ids = {name: label for label, name in enumerate(engine.NAMES[1:], 1)}
    names = set(class_map.values())
    if len(names) != len(class_map) or not internal_ids.keys() <= names:
        raise ValueError("class_map must contain unique names and all vertebrae")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="warmup-", dir=output.parent) as temp:
        workspace = Path(temp)
        if rooted is None:
            reference, raw, _ = load_vertebrae(source, subfolder_name)
            raw_path = workspace / "input" / source.name / "combined_labels.nii.gz"
            raw_path.parent.mkdir(parents=True)
            save_volume(raw, reference, raw_path)
            del raw

            stage_root = workspace / "result"
            engine.component_filter(raw_path, stage_root, [internal_ids["vertebrae_C1"]])
            stage = stage_root / source.name
        else:
            reference, stage = run_rooted(source, output, workspace, rooted)
        combined_path = stage / "combined_labels.nii.gz"
        labels = np.asarray(nib.load(combined_path).dataobj)
        segdir = stage / subfolder_name
        if subfolder_name != "segmentations":
            (stage / "segmentations").rename(segdir)

        for path in (source / subfolder_name).glob("*.nii.gz"):
            if path.name[:-7] not in internal_ids:
                image = nib.load(path)
                _check_grid(image, reference, path.name)
                shutil.copy2(path, segdir / path.name)

        dtype = np.uint8 if max(class_map) <= 255 else np.uint16
        combined = np.zeros(labels.shape, dtype=dtype)
        for label, name in sorted(class_map.items()):
            if name in internal_ids:
                mask = labels == internal_ids[name]
            else:
                path = segdir / f"{name}.nii.gz"
                if not path.is_file():
                    continue
                mask = _binary(nib.load(path), name)
            combined[mask] = label
        save_volume(combined, reference, combined_path)
        stage.rename(output)
