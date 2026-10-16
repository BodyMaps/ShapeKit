"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
from __future__ import annotations

import nibabel as nib

import numpy as np

from scipy import ndimage as ndi

from skimage.segmentation import watershed

from .postprocessing_causal import NAMES

def body_anchors(base, affine):
    spacing = nib.affines.voxel_sizes(affine)
    centers = []
    for label in range(16, 24):
        depth = ndi.distance_transform_edt(base == label, sampling=spacing)
        cc, count = ndi.label(depth >= 3., np.ones((3,3,3)))
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        if count == 0 or sizes.max()*np.prod(spacing) < 60:
            raise ValueError(f'insufficient_cervical_body_seed_{label}')
        xyz = ndi.center_of_mass(cc == sizes.argmax())
        centers.append(nib.affines.apply_affine(affine, xyz))
    centers = np.array(centers)
    gaps = np.diff(centers[:,2])
    if np.any(gaps<5) or np.any(gaps>30):
        raise ValueError('cervical_anchors_not_ordered')
    return centers

def sweep(base, ct, affine, centers, width, root_depth):
    spacing = nib.affines.voxel_sizes(affine)
    direction = affine[:3,:3]/spacing
    if not (np.allclose(np.abs(direction),np.eye(3),atol=1e-5) and direction[2,2]>0 and direction[1,1]>0):
        raise ValueError('Native superior-positive axial LAS/RAS required')
    x = affine[0,3]+affine[0,0]*np.arange(base.shape[0])[:,None]
    y = affine[1,3]+affine[1,1]*np.arange(base.shape[1])[None,:]
    z = affine[2,3]+affine[2,2]*np.arange(base.shape[2])
    boundaries = (centers[:-1,2]+centers[1:,2])/2
    previous = np.zeros(base.shape[:2],np.uint8)
    out = base.copy()
    root_counts = {str(i):0 for i in range(16,24)}
    for k in range(base.shape[2]-1,-1,-1):
        cx = np.interp(z[k],centers[:,2],centers[:,0]); cy=np.interp(z[k],centers[:,2],centers[:,1])
        expected = 16+int(np.clip(np.searchsorted(boundaries,z[k]),0,7))
        field = base[:,:,k]
        domain = ((np.abs(x-cx)<=width)&(y<cy-16)&(y>=cy-65)
                  &(field>=16)&(field<=23)&(ct[:,:,k]>=100))
        own_bone = (field==expected)&(ct[:,:,k]>=100)
        cc,_=ndi.label(own_bone,np.ones((3,3)))
        body_region=(np.abs(x-cx)<=18)&(y>=cy-12)&(y<=cy+16)
        ids=np.unique(cc[own_bone&body_region]);ids=ids[ids>0]
        connected=np.isin(cc,ids) if len(ids) else np.zeros(field.shape,bool)
        roots=domain&connected&(y>=cy-root_depth)
        root_counts[str(expected)]+=int(roots.sum())
        # Permit only actual 26-neighbor continuity across the native slices.
        # Dilation is a marker transfer, not foreground growth.
        continued=ndi.grey_dilation(previous,size=(3,3))
        markers=np.where(domain,continued,0).astype(np.uint8)
        markers[roots]=expected
        if expected==23:
            markers[domain&(field==23)]=23
        if markers.any():
            depth=ndi.distance_transform_edt(domain,sampling=spacing[:2])
            owner=watershed(-depth,markers=markers,mask=domain,connectivity=np.ones((3,3))).astype(np.uint8)
        else:owner=np.zeros(field.shape,np.uint8)
        editable=(field>=17)&(field<=22)&domain&(owner>=17)&(owner<=23)
        out[:,:,k][editable]=owner[editable]
        previous=owner
    return out,root_counts

def refine(base,ct,affine,centers):
    settings=[(6.,26.),(8.,30.),(10.,34.)]
    votes=[];root_rows=[]
    for width,depth in settings:
        print('  cervical root corridor',width,'mm; depth',depth,'mm',flush=True)
        v,roots=sweep(base,ct,affine,centers,width,depth);votes.append(v);root_rows.append(roots)
    proposed=votes[0]
    change=(proposed==votes[1])&(proposed==votes[2])&(proposed.astype(int)-base.astype(int)==1)
    out=base.copy();out[change]=proposed[change]
    capped=[]
    per_label=[]
    for label in range(17,24):
        n=int(np.count_nonzero(base==label));m=int(np.count_nonzero(out==label))
        affected=int(np.count_nonzero((out!=base)&((out==label)|(base==label))))
        row={'label':label,'before_voxels':n,'proposed_voxels':m,'changed_fraction':affected/max(n,1),'net_volume_change_fraction':(m-n)/max(n,1)}
        per_label.append(row)
        if affected/max(n,1)>.50 or abs(m-n)/max(n,1)>.25:capped.append(label)
    revert=np.isin(base,capped)|np.isin(out,capped);out[revert]=base[revert]
    assert np.array_equal(out>0,base>0)
    assert np.all(out[base==23]==23)  # C2's existing identity is frozen.
    for label in list(range(1,17))+[24]:assert np.array_equal(out==label,base==label)
    return out,{'method':'original cervical/upper-thoracic body-root-supported caudal ownership',
                'reference_labels_used':False,'settings_width_root_depth_mm':settings,
                'body_seed_radius_mm':3.,'minimum_body_seed_mm3':60.,
                'capped_labels':capped,'per_label_proposed_changes':per_label,'root_voxels_by_scale':root_rows,
                'changed_voxels':int(np.count_nonzero(out!=base)),'foreground_exact':True,
                'C1_exact':True,'existing_C2_voxels_frozen':True,
                'status':'applied' if np.any(out!=base) else 'abstained','dsc':None,
                'limitation':'C1/atlas ownership unresolved; this anatomical exception is included in overall evaluation'}
