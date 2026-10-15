"""Contracts for the multi-model integration, with real numerical stages."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from utils import vertebrae_consensus as adapter


def save(path, data, units='unknown', affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units(units)
    image.header['descrip'] = b'consensus contract fixture'
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 2)
    nib.save(image, path)


def masks(root, labels):
    for i, name in enumerate(adapter.NAMES, 1):
        # Present empty masks are distinct from missing masks in reports.
        save(root/'segmentations'/(name+'.nii.gz'), (labels==i).astype(np.uint8))


@pytest.fixture
def case(tmp_path):
    primary = np.zeros((20, 20, 24), np.uint8)
    primary[4:10, 4:10, 4:10] = 1
    primary[2, 2, 2] = 23
    primary[2, 2, 3] = 24
    auxiliary = primary.copy()
    auxiliary[10, 6, 6] = 1
    auxiliary[4, 6, 6] = 0
    # Unanimity alone cannot expand beyond 2 mm or change the nearest ID.
    auxiliary[16, 6, 6] = 1
    auxiliary[10, 7, 6] = 2
    paths = {r:tmp_path/r/'case' for r in ('primary','suprem','v3')}
    for role, root in paths.items():
        masks(root, primary if role=='primary' else auxiliary)
    ct = tmp_path/'ct/case/ct.nii.gz'
    save(ct, np.full(primary.shape, 300, np.float32), units='mm')
    mapping = {i+26:n for i,n in enumerate(adapter.NAMES)}
    return dict(paths=paths,ct=ct,mapping=mapping,primary=primary,auxiliary=auxiliary,
                output=tmp_path/'output/case')


def run(case, **kwargs):
    return adapter.process_case(case['paths']['primary'],case['output'],case['ct'],
                                case['mapping'],suprem_path=case['paths']['suprem'],
                                v3_path=case['paths']['v3'],input_stage='prepared',**kwargs)


def test_numerical_function_bodies_match_submitted_sources():
    folder=Path(adapter.__file__).with_name('_vertebrae_consensus')
    provenance=json.loads((folder/'SOURCE_PROVENANCE.json').read_text())
    for module, row in provenance['modules'].items():
        tree=ast.parse((folder/(module+'.py')).read_text())
        found={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
               for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        assert found==row['function_ast_sha256']


def test_loader_preserves_values_and_c_order_at_numerical_boundary(case):
    reference,labels,missing,hashes=adapter.load_masks(case['paths']['suprem'])
    np.testing.assert_array_equal(labels,case['auxiliary'])
    assert labels.flags.c_contiguous
    assert not missing
    assert len(hashes)==24


def test_real_boundary_geometry_protected_masks_and_source_immutability(case):
    files=[p for root in case['paths'].values() for p in root.rglob('*.nii.gz')]+[case['ct']]
    hashes={p:adapter.digest(p) for p in files}
    report=run(case,save_stages=True)
    expected=case['primary'].copy();expected[10,6,6]=1;expected[4,6,6]=0
    assert report['reference_labels_used'] is False
    assert report['dsc'] is None
    assert report['input_stage']=='prepared'
    assert report['stage_reports'][0]['changed_voxels']==2
    combined=nib.load(case['output']/'combined_labels.nii.gz')
    want=np.where(expected>0,expected+25,0)
    np.testing.assert_array_equal(np.asarray(combined.dataobj),want)
    assert combined.header.get_xyzt_units()[0]=='unknown'
    assert int(combined.header['qform_code'])==1
    assert int(combined.header['sform_code'])==2
    for i,name in enumerate(adapter.NAMES,1):
        image=nib.load(case['output']/'segmentations'/(name+'.nii.gz'))
        np.testing.assert_array_equal(np.asarray(image.dataobj),expected==i)
        assert image.header['descrip']==combined.header['descrip']
    assert len(list((case['output']/'segmentations').glob('*.nii.gz')))==24
    assert (case['output']/'stages/boundary_agreement.nii.gz').is_file()
    assert hashes=={p:adapter.digest(p) for p in files}
    assert json.loads((case['output']/'vertebrae_consensus_report.json').read_text())==report


def test_copy_other_organ_and_nonstandard_combined_ids(case):
    other=np.zeros_like(case['primary']);other[0,0,0]=1
    path=case['paths']['primary']/'segmentations/liver.nii.gz';save(path,other)
    case['mapping']={i+301:n for i,n in enumerate(adapter.NAMES)}
    case['mapping'][5]='liver'
    run(case)
    assert path.read_bytes()==(case['output']/'segmentations/liver.nii.gz').read_bytes()
    result=nib.load(case['output']/'combined_labels.nii.gz')
    assert result.get_data_dtype()==np.dtype(np.uint16)
    assert result.dataobj[6,6,6]==301
    assert result.dataobj[0,0,0]==5


def test_raw_pipeline_runs_all_stages_without_cached_case_data(case):
    empty=np.zeros_like(case['primary'])
    final, stages=adapter.run_pipeline(empty,empty,empty,np.zeros_like(empty,dtype=np.float32),
        nib.load(case['ct']),input_stage='raw')
    np.testing.assert_array_equal(final,empty)
    assert [s['stage'] for s in stages]==['suprem_hao','suprem_anchors','primary_hao',
        'incumbent','v3_proposal','aligned_v3','boundary_agreement']
    assert all(s.get('changed_voxels',0)==0 for s in stages)


@pytest.mark.parametrize('role',['primary','suprem','v3'])
def test_nonbinary_or_overlapping_masks_rejected(case,role):
    bad=np.zeros_like(case['primary']);bad[0,0,0]=2
    save(case['paths'][role]/'segmentations/vertebrae_L5.nii.gz',bad)
    with pytest.raises(ValueError,match='binary'):run(case)
    assert not case['output'].exists()


def test_overlapping_classes_rejected(case):
    save(case['paths']['suprem']/'segmentations/vertebrae_L4.nii.gz',
         (case['auxiliary']==1).astype(np.uint8))
    with pytest.raises(ValueError,match='Overlapping'):run(case)


def test_missing_auxiliary_is_not_silently_dropped(case):
    case['paths']['suprem']=case['paths']['suprem']/'missing'
    with pytest.raises(ValueError,match='No named vertebra'):run(case)
    assert not case['output'].exists()


def test_missing_single_mask_is_reported_and_empty_output_is_written(case):
    (case['paths']['primary']/'segmentations/vertebrae_C7.nii.gz').unlink()
    report=run(case)
    assert report['missing_input_masks']['primary']==['vertebrae_C7']
    assert not np.any(nib.load(case['output']/'segmentations/vertebrae_C7.nii.gz').dataobj)


@pytest.mark.parametrize('units',['unknown','meter','micron'])
def test_ct_must_declare_mm(case,units):
    save(case['ct'],np.zeros_like(case['primary'],dtype=np.float32),units=units)
    with pytest.raises(ValueError,match='millimeter'):run(case)


def test_scaled_ct_is_loaded_as_hu(case,monkeypatch):
    image=nib.Nifti1Image(np.ones_like(case['primary'],dtype=np.int16),np.eye(4))
    image.header.set_xyzt_units('mm');image.header.set_slope_inter(2,-1024);nib.save(image,case['ct'])
    real=adapter.run_pipeline
    def check(primary,suprem,v3,ct,*args,**kwargs):
        assert ct.dtype==np.float32
        assert np.all(ct==-1022)
        return real(primary,suprem,v3,ct,*args,**kwargs)
    monkeypatch.setattr(adapter,'run_pipeline',check)
    run(case)


def test_ct_nan_rejected(case):
    data=np.zeros_like(case['primary'],dtype=np.float32);data[0,0,0]=np.nan
    save(case['ct'],data,units='mm')
    with pytest.raises(ValueError,match='Finite CT'):run(case)


def test_auxiliary_grid_mismatch_rejected(case):
    affine=np.eye(4);affine[0,3]=2
    masks_root=case['paths']['v3']/'segmentations'
    for p in masks_root.glob('*.nii.gz'):
        data=np.asarray(nib.load(p).dataobj);save(p,data,affine=affine)
    with pytest.raises(ValueError,match='Native-grid'):run(case)


def test_shear_rejected(case):
    affine=np.eye(4);affine[0,1]=.1
    save(case['ct'],np.zeros_like(case['primary'],dtype=np.float32),units='mm',affine=affine)
    with pytest.raises(ValueError,match='Sheared'):run(case)


def test_extra_vertebra_level_rejected(case):
    save(case['paths']['v3']/'segmentations/vertebrae_L6.nii.gz',np.zeros_like(case['primary']))
    with pytest.raises(ValueError,match='24-level'):run(case)


def test_output_overlap_and_overwrite_rejected(case):
    case['output']=case['paths']['suprem']/'new-output'
    with pytest.raises(ValueError,match='disjoint'):run(case)
    case['output']=case['paths']['v3']
    with pytest.raises(FileExistsError):run(case)


def test_partial_write_is_not_published(case,monkeypatch):
    def fail(*args,**kwargs):raise OSError('injected write failure')
    monkeypatch.setattr(adapter,'save_volume',fail)
    with pytest.raises(OSError,match='injected'):run(case)
    assert not case['output'].exists()
    assert len(list((case['output'].parent/'.incomplete').iterdir()))==1


def test_cli_isolates_failed_case_with_nonzero_batch_exit(case,tmp_path):
    repo=Path(__file__).resolve().parents[1]
    for root in case['paths'].values():
        (root.parent/'bad'/'segmentations').mkdir(parents=True)
    command=[sys.executable,'main.py','--vertebrae_engine','shapekit_consensus',
        '--vertebrae_only','--consensus_input_stage','prepared',
        '--input_folder',str(case['paths']['primary'].parent),
        '--suprem_root',str(case['paths']['suprem'].parent),
        '--totalseg_v3_root',str(case['paths']['v3'].parent),
        '--ct_root',str(case['ct'].parents[1]),'--output_folder',str(case['output'].parent),
        '--cpu_count','2','--log_folder',str(tmp_path/'logs')]
    result=subprocess.run(command,cwd=repo,capture_output=True,text=True,timeout=90)
    assert result.returncode==1,result.stdout+result.stderr
    assert 'Failed cases: bad' in result.stdout
    assert (case['output']/'vertebrae_consensus_report.json').is_file()
    assert not (case['output'].parent/'bad').exists()


def make_reference(case):
    expected=case['primary'].copy();expected[10,6,6]=1;expected[4,6,6]=0
    root=case['output'].parent.parent/'reference/case'
    masks(root,expected)
    save(root/'combined_labels.nii.gz',expected)
    return root


def test_equivalence_checker_accepts_label_conversion_and_checks_every_mask(case):
    from tools.verify_vertebrae_consensus import compare_case
    run(case);reference=make_reference(case)
    report=compare_case(case['output'].parent,reference.parent,'case')
    assert report['passed']
    assert len(report['masks'])==24
    assert report['changed_combined_voxels']==0


@pytest.mark.parametrize('corruption',['mask','units','label_map'])
def test_equivalence_checker_detects_corruption(case,corruption):
    from tools.verify_vertebrae_consensus import compare_case
    run(case);reference=make_reference(case)
    target=case['output']/'segmentations/vertebrae_L5.nii.gz'
    data=np.asarray(nib.load(target).dataobj).copy()
    if corruption=='mask':
        data[0,0,0]=1;save(target,data)
    elif corruption=='units':save(target,data,units='mm')
    else:
        path=case['output']/'vertebrae_consensus_report.json'
        report=json.loads(path.read_text());report['label_map']['2']=report['label_map']['1']
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError,match='label map'):
            compare_case(case['output'].parent,reference.parent,'case')
        return
    result=compare_case(case['output'].parent,reference.parent,'case')
    assert not result['passed']
    assert not result['masks'][0]['passed']
