"""Compare integrated vertebra predictions with a standalone warmup reference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from utils.vertebrae_components import NAMES, check_grid


def compare_case(actual, expected):
    report = {'case': actual.name, 'masks': []}
    for name in NAMES:
        relative = Path('segmentations') / (name + '.nii.gz')
        first, second = nib.load(actual / relative), nib.load(expected / relative)
        check_grid(first, second)
        if first.header.get_xyzt_units() != second.header.get_xyzt_units():
            raise ValueError('Units differ: ' + name)
        for form in ('qform', 'sform'):
            if int(first.header[form + '_code']) != int(second.header[form + '_code']):
                raise ValueError('Form code differs: ' + name)
            if not np.allclose(getattr(first, 'get_' + form)(), getattr(second, 'get_' + form)(), rtol=0, atol=1e-5):
                raise ValueError('Form matrix differs: ' + name)
        different = int(np.count_nonzero(np.asanyarray(first.dataobj) != np.asanyarray(second.dataobj)))
        report['masks'].append({'name': name, 'different_voxels': different})
        if different:
            raise ValueError(f'{actual.name}/{name}: {different} differing voxels')
    combined = nib.load(actual / 'combined_labels.nii.gz')
    reference = nib.load(expected / 'combined_labels.nii.gz')
    check_grid(combined, reference)
    manifest = json.loads((actual / 'vertebrae_components_report.json').read_text(encoding='utf-8'))
    first = np.asanyarray(combined.dataobj)
    second = np.asanyarray(reference.dataobj)
    if not np.all(np.isin(second, np.arange(25))):
        raise ValueError('Expected combined map must use warmup labels 0..24')
    lookup = np.array([0] + [int(manifest['label_map'][str(i)]) for i in range(1, 25)], dtype=first.dtype)
    if not np.array_equal(first, lookup[second.astype(np.uint8)]):
        raise ValueError('Combined labels differ after label-map conversion')
    report['combined_match_after_mapping'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', required=True, type=Path, help='Integrated prediction root')
    parser.add_argument('--expected', required=True, type=Path, help='Standalone warmup prediction root')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report must be a new file')
    cases = sorted(p for p in args.expected.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not cases:
        parser.error('No expected cases found')
    reports = []
    for case in cases:
        reports.append(compare_case(args.actual / case.name, case))
        print(case.name + ': all 24 masks and combined-map conversion match', flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as handle:
        json.dump({'cases': reports, 'scope': 'Reproduction, not DSC evaluation'}, handle, indent=2)


if __name__ == '__main__':
    main()
