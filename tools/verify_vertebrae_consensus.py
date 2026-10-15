"""Compare the integration with submitted predictions, not ground truth."""
import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.vertebrae_consensus import NAMES


def geometry(a, b):
    result = dict(shape=a.shape==b.shape,
        affine=bool(np.allclose(a.affine,b.affine,rtol=0,atol=1e-5)),
        spacing=bool(np.allclose(a.header.get_zooms(),b.header.get_zooms(),rtol=0,atol=1e-5)),
        units=a.header.get_xyzt_units()==b.header.get_xyzt_units())
    for key in ('qform','sform'):
        x,xc=getattr(a,'get_'+key)(coded=True);y,yc=getattr(b,'get_'+key)(coded=True)
        result[key]=bool(xc==yc and ((x is None and y is None) or
            (x is not None and y is not None and np.allclose(x,y,rtol=0,atol=1e-5))))
    return result


def compare_case(actual, reference, case, prepared_inputs=None):
    aroot,rroot=Path(actual)/case,Path(reference)/case
    report=json.loads((aroot/'vertebrae_consensus_report.json').read_text())
    if report.get('integration_status')!='complete':
        raise ValueError('Completed consensus report required')
    ids=[report['label_map'][str(i)] for i in range(1,25)]
    if (any(type(i) is not int or not 1<=i<=65535 for i in ids)
            or len(set(ids))!=24):
        raise ValueError('Invalid or ambiguous label map')
    ai=nib.load(aroot/'combined_labels.nii.gz');ri=nib.load(rroot/'combined_labels.nii.gz')
    grid=geometry(ai,ri)
    if ai.shape!=ri.shape:
        return dict(case=case,passed=False,combined_geometry=grid)
    a,r=np.asanyarray(ai.dataobj),np.asanyarray(ri.dataobj)
    if a.dtype.kind not in 'ui' or a.min()<0 or a.max()>65535:
        raise ValueError('Actual combined IDs must be nonnegative uint16-range integers')
    if r.dtype.kind not in 'ui' or r.min()<0 or r.max()>24:
        raise ValueError('Standalone reference combined map must use IDs 0..24')
    lookup=np.zeros(65536,np.uint8)
    for i,value in enumerate(ids,1):lookup[value]=i
    changed=int(np.count_nonzero(lookup[a]!=r))
    rows=[]
    for i,name in enumerate(NAMES,1):
        am=nib.load(aroot/'segmentations'/(name+'.nii.gz'))
        rm=nib.load(rroot/'segmentations'/(name+'.nii.gz'))
        g=geometry(am,rm);ad,rd=np.asanyarray(am.dataobj),np.asanyarray(rm.dataobj)
        shape=ad.shape==rd.shape==a.shape
        binary=bool(np.all((ad==0)|(ad==1)) and np.all((rd==0)|(rd==1)))
        mismatch=int(np.count_nonzero(ad!=rd)) if shape else None
        consistent=shape and bool(np.array_equal(ad>0,a==ids[i-1]) and np.array_equal(rd>0,r==i))
        rows.append(dict(name=name,passed=all(g.values()) and mismatch==0 and binary and consistent,
                         geometry=g,changed_voxels=mismatch,binary=binary,combined_consistent=consistent))
        del ad,rd
        print(f'{case}/{name}: {"PASS" if rows[-1]["passed"] else "FAIL"}',flush=True)
    del a,r
    stage_rows=[]
    if prepared_inputs is not None:
        for stage,role in [('suprem_anchors','SuPreM_anchors'),('incumbent','incumbent'),('aligned_v3','aligned_v3')]:
            x=nib.load(aroot/'stages'/(stage+'.nii.gz'))
            y=nib.load(Path(prepared_inputs)/role/case/'combined_labels.nii.gz')
            g=geometry(x,y)
            n=int(np.count_nonzero(np.asanyarray(x.dataobj)!=np.asanyarray(y.dataobj))) if x.shape==y.shape else None
            stage_rows.append(dict(stage=stage,passed=all(g.values()) and n==0,geometry=g,changed_voxels=n))
    return dict(case=case,passed=all(grid.values()) and changed==0 and all(x['passed'] for x in rows+stage_rows),
        combined_geometry=grid,changed_combined_voxels=changed,masks=rows,intermediate_stages=stage_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--case',action='append',required=True)
    p.add_argument('--prepared-inputs',type=Path)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():p.error('Report already exists; choose a new path')
    if not args.case:p.error('At least one case is required')
    rows=[]
    for case in args.case:
        try:rows.append(compare_case(args.actual,args.reference,case,args.prepared_inputs))
        except Exception as exc:rows.append(dict(case=case,passed=False,error=str(exc)))
    result=dict(purpose='Output equivalence, not ground-truth accuracy',dsc=None,cases=rows,
                passed=bool(rows) and all(r['passed'] for r in rows))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS' if result['passed'] else 'FAIL',flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
