"""Compare ShapeKit-Hao exports with a trusted standalone output, not GT."""

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.vertebrae_hao_engine import VERTEBRA_NAMES


def geometry(a, b):
    result = {
        'shape': a.shape == b.shape,
        'affine': bool(np.allclose(a.affine, b.affine, rtol=0, atol=1e-5)),
        'spacing': bool(np.allclose(a.header.get_zooms(), b.header.get_zooms(),
                                   rtol=0, atol=1e-5)),
    }
    for name in ('qform', 'sform'):
        x, xc = getattr(a, 'get_' + name)(coded=True)
        y, yc = getattr(b, 'get_' + name)(coded=True)
        result[name + '_code'] = int(xc) == int(yc)
        result[name + '_matrix'] = (x is None and y is None) or (
            x is not None and y is not None
            and bool(np.allclose(x, y, rtol=0, atol=1e-5)))
    return result


def compare_case(actual, reference, case, actual_subfolder='segmentations'):
    aroot, rroot = actual / case, reference / case
    run = json.loads((aroot / 'vertebrae_hao_report.json').read_text(encoding='utf-8'))
    if run.get('integration_status') != 'complete':
        raise ValueError(f'{case}: missing completed integration report')
    label_map = run['label_map']
    ai = nib.load(str(aroot / 'combined_labels.nii.gz'))
    ri = nib.load(str(rroot / 'combined_labels.nii.gz'))
    geom = geometry(ai, ri)
    if ai.shape != ri.shape:
        return dict(case=case, passed=False, combined_geometry=geom)
    a = np.asanyarray(ai.dataobj)
    r = np.asanyarray(ri.dataobj)
    if a.dtype.kind not in 'ui' or a.min() < 0 or a.max() > 65535:
        raise ValueError('Actual combined map must contain uint16-range integer IDs')
    # One lookup pass avoids 24 strided writes across a full CT-sized volume.
    # Unrelated ShapeKit organs intentionally map to background for comparison.
    lookup = np.zeros(65536, dtype=np.uint8)
    for label in range(1, 25):
        mapped_id = int(label_map[str(label)])
        if not 1 <= mapped_id <= 65535:
            raise ValueError('Invalid ShapeKit vertebra class ID in report')
        lookup[mapped_id] = label
    normalized = lookup[a]
    changed = int(np.count_nonzero(normalized != r))
    del normalized
    rows = []
    for label, name in enumerate(VERTEBRA_NAMES, 1):
        am = nib.load(str(aroot / actual_subfolder / (name + '.nii.gz')))
        rm = nib.load(str(rroot / 'segmentations' / (name + '.nii.gz')))
        g = geometry(am, rm)
        ad, rd = np.asanyarray(am.dataobj), np.asanyarray(rm.dataobj)
        same_shape = ad.shape == rd.shape == a.shape
        mismatch = int(np.count_nonzero(ad != rd)) if same_shape else None
        binary = bool(np.all((ad == 0) | (ad == 1))
                      and np.all((rd == 0) | (rd == 1)))
        ac = same_shape and bool(np.array_equal(ad > 0, a == int(label_map[str(label)])))
        rc = same_shape and bool(np.array_equal(rd > 0, r == label))
        passed = (all(g.values()) and mismatch == 0 and binary and ac and rc)
        rows.append(dict(name=name, passed=passed, geometry=g,
                         changed_voxels=mismatch, binary=binary,
                         actual_combined_consistent=ac, reference_combined_consistent=rc))
        del ad, rd
        print(f'  {case}/{name}: {"PASS" if passed else "FAIL"}', flush=True)
    return dict(case=case, passed=all(geom.values()) and changed == 0
                and all(row['passed'] for row in rows),
                combined_geometry=geom, changed_combined_voxels=changed,
                stage2_status=run.get('status'), stage2_reason=run.get('reason'), masks=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--actual-subfolder', default='segmentations',
                        help='Actual mask subfolder, matching config.yaml subfolder_name')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = dict(dsc=None, purpose='Regression equivalence, NOT ground-truth accuracy',
                  actual=str(args.actual.resolve()), reference=str(args.reference.resolve()),
                  actual_subfolder=args.actual_subfolder,
                  cases=[])
    for case in args.case:
        try:
            report['cases'].append(compare_case(args.actual, args.reference, case,
                                                args.actual_subfolder))
        except Exception as exc:
            report['cases'].append(dict(case=case, passed=False, error=str(exc)))
            print(f'{case}: ERROR {exc}', flush=True)
    report['passed'] = all(c['passed'] for c in report['cases'])
    if args.report:
        if args.report.exists():
            parser.error('Report already exists; choose a new path')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PASS: identical masks and geometry' if report['passed'] else 'FAIL: inspect report')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
