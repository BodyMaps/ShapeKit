"""Bind previously reviewed seed metadata to a specific prediction volume."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
from utils.vertebrae_components import check_ct, fingerprint, load_labels
from utils.vertebrae_components_engine import LABEL_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-case', required=True, type=Path)
    parser.add_argument('--ct', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['cleanup_only', 'reviewed'])
    parser.add_argument('--reviewed-seeds', type=Path, help='Existing visually reviewed proposed_seeds.csv; this tool does not validate anatomy')
    parser.add_argument('--erosion-mm', type=float, default=5.0)
    parser.add_argument('--boundary-splits', type=Path, help='JSON list of reviewed {original_label, rank, voxels, targets} selections; IDs 1=L5 to 17=T1')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new plan file')
    if args.mode == 'reviewed' and args.reviewed_seeds is None:
        parser.error('Reviewed mode requires --reviewed-seeds')
    if args.mode == 'cleanup_only' and (args.reviewed_seeds or args.boundary_splits):
        parser.error('Cleanup-only mode cannot include corrections')
    ct = nib.load(str(args.ct))
    check_ct(ct)
    raw = load_labels(args.input_case, ct)
    plan = {'schema_version': 1, 'mode': args.mode, 'source_sha256': fingerprint(raw, ct.affine)}
    if args.mode == 'reviewed':
        with args.reviewed_seeds.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if [row['proposed_label'] for row in rows] != list(LABEL_NAMES[1:18]):
            parser.error('Reviewed seed CSV must contain L5 through T1 in order')
        plan.update(seeds=[{key: row[key] for key in ('component_id', 'proposed_label', 'dominant_original_label', 'core_volume_mm3')} for row in rows],
                    erosion_mm=args.erosion_mm,
                    boundary_splits=json.loads(args.boundary_splits.read_text(encoding='utf-8')) if args.boundary_splits else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(plan, handle, indent=2)
        handle.write('\n')
    print('Saved review plan:', args.output)
    print('This binds metadata to an input; it does not establish anatomical correctness.')


if __name__ == '__main__':
    main()
