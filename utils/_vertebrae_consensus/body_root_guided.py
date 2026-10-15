"""Numerical routines from the submitted warmup; no cohort paths or CLI.

Author: Hongkang Chu. Algorithm bodies are unchanged. See SOURCE_PROVENANCE.json.
"""
from dataclasses import asdict,dataclass

import nibabel as nib

import numpy as np

from scipy import ndimage as ndi

from skimage.graph import MCP_Geometric

from .postprocessing_causal import NAMES

@dataclass(frozen=True)
class Region:
    name:str
    first_anchor:int
    last_anchor:int
    first_edit:int
    last_edit:int
    body_core_radius_mm:float
    minimum_core_mm3:float
    posterior_offset_mm:float
    atlas_marker:bool=False

REGIONS=[Region('thoracolumbar',3,14,4,13,5.,300.,22.),
         Region('upper_thoracic_and_cervical',14,23,15,22,3.,60.,12.,True)]

def centers_and_seeds(base,affine,region):
    spacing=nib.affines.voxel_sizes(affine);centers=[];cores={}
    for label in range(region.first_anchor,region.last_anchor+1):
        depth=ndi.distance_transform_edt(base==label,sampling=spacing)
        cc,count=ndi.label(depth>=region.body_core_radius_mm,np.ones((3,3,3)))
        sizes=np.bincount(cc.ravel());sizes[0]=0
        if not count or sizes.max()*np.prod(spacing)<region.minimum_core_mm3:
            raise ValueError(f'insufficient_body_core_{label}')
        mask=cc==sizes.argmax();cores[label]=mask
        center=ndi.center_of_mass(mask);centers.append(nib.affines.apply_affine(affine,center))
    centers=np.round(np.array(centers),8)
    gaps=np.diff(centers[:,2])
    if np.any(gaps<5) or np.any(gaps>60):raise ValueError('body_cores_not_anatomically_ordered')
    origin=centers.mean(0);_,_,vh=np.linalg.svd(centers-origin,full_matrices=False)
    t=vh[0].copy()
    if t[2]<0:t=-t
    anterior=np.array([0.,1.,0.])-t[1]*t
    if np.linalg.norm(anterior)<.2:raise ValueError('ambiguous_anterior_direction')
    anterior/=np.linalg.norm(anterior);right=np.cross(anterior,t);right/=np.linalg.norm(right)
    basis=np.column_stack([right,anterior,t]);local_centers=(centers-origin)@basis
    if np.any(np.diff(local_centers[:,2])<5):raise ValueError('intrinsic_core_order_invalid')
    return centers,cores,origin,basis,local_centers

def refine_region(base,ct,affine,region,anchors):
    spacing=nib.affines.voxel_sizes(affine);direction=affine[:3,:3]/spacing
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5):raise ValueError('Sheared grids unsupported')
    upper=24 if region.atlas_marker else region.last_anchor
    pos=np.where((base>=region.first_anchor)&(base<=upper))
    if not len(pos[0]):raise ValueError('empty_region')
    padding=np.ceil(5/spacing).astype(int)
    roi=tuple(slice(max(0,int(p.min())-int(q)),min(base.shape[i],int(p.max())+int(q)+1))
              for i,(p,q) in enumerate(zip(pos,padding)))
    b=base[roi];hu=ct[roi]
    a=affine.copy();a[:3,3]=nib.affines.apply_affine(affine,[s.start for s in roi])
    centers,cores,origin,basis,local_centers=centers_and_seeds(anchors[roi],a,region)
    domain=(b>=region.first_anchor)&(b<=upper)
    outside=ndi.distance_transform_edt(~domain,sampling=spacing)
    shell=(outside>0)&(outside<=2.5)
    contrast=float(np.median(hu[domain])-np.median(hu[shell])) if shell.any() else 0.
    del outside
    if contrast<80.:raise ValueError('insufficient_CT_bone_to_surroundings_contrast')
    protected=np.zeros(b.shape,bool)
    for label,core in cores.items():protected|=core&(b==label)
    # Exclude thick posterior pieces from the identity seeds. Some baseline
    # IDs incorrectly include a neighbor's process; only anterior body cores
    # are trusted, even if those pieces join the largest eroded component.
    markers={}
    for label,core in cores.items():
        coords=np.array(np.where(core)).T
        local_core=(nib.affines.apply_affine(a,coords)-origin)@basis
        center=local_centers[label-region.first_anchor]
        selected=local_core[:,1]>=center[1]-region.body_core_radius_mm
        selected &= domain[tuple(coords.T)]
        seeds=coords[selected]
        if len(seeds)*np.prod(spacing)<region.minimum_core_mm3/3:
            raise ValueError(f'insufficient_trusted_anterior_seeds_{label}')
        markers[label]=[tuple(int(v) for v in q) for q in seeds]
    if region.atlas_marker:
        # Atlas has no body: use its existing CT-supported largest component
        # as a distinct ring anchor, without imposing a body-center ordering.
        atlas=(b==24)&(hu>=100)
        cc,count=ndi.label(atlas,np.ones((3,3,3)));sizes=np.bincount(cc.ravel());sizes[0]=0
        if count and sizes.max()*np.prod(spacing)>=300:
            markers[24]=[tuple(int(v) for v in q) for q in np.array(np.where(cc==sizes.argmax())).T]
    depth=ndi.distance_transform_edt(domain,sampling=spacing).astype(np.float32)
    CT_penalty=1+np.maximum(0,150-hu)/150
    votes=[];margins=[]
    for power in [1.,1.5,2.]:
        print(region.name,'3D geodesic thickness power',power,flush=True)
        costs=np.full(b.shape,np.inf,np.float32)
        costs[domain]=CT_penalty[domain]/(depth[domain]+.5)**power
        best=np.full(b.shape,np.inf,np.float64);second=best.copy();owner=np.zeros(b.shape,np.uint8)
        for label,seeds in markers.items():
            solver=MCP_Geometric(costs,sampling=tuple(spacing))
            distance,_=solver.find_costs(seeds)
            better=distance<best
            second[better]=best[better];best[better]=distance[better];owner[better]=label
            alternate=(~better)&(distance<second);second[alternate]=distance[alternate]
            del solver,distance
        margin=np.zeros(b.shape,np.float32)
        unique=np.isfinite(best)&~np.isfinite(second);margin[unique]=1
        ambiguous=np.isfinite(best)&np.isfinite(second)
        margin[ambiguous]=(second[ambiguous]-best[ambiguous])/np.maximum(second[ambiguous],1e-6)
        votes.append(owner);margins.append(margin)
    consensus=(votes[0]==votes[1])&(votes[0]==votes[2])
    confidence=(margins[0]>=.10)&(margins[1]>=.10)&(margins[2]>=.10)
    target=votes[0]
    editable=((b>=region.first_edit)&(b<=region.last_edit)&(~protected)&(hu>=100)
              &(target>=region.first_anchor)&(target<=upper)
              &(np.abs(target.astype(int)-b.astype(int))==1)&consensus&confidence)
    result=b.copy();result[editable]=target[editable]
    # The volume limits are evaluated on the actual native proposal, then
    # rechecked after every rollback so neighboring rollbacks cannot bypass it.
    capped=[]
    while True:
        reject=[]
        for label in range(region.first_anchor,upper+1):
            n=int(np.count_nonzero(b==label));m=int(np.count_nonzero(result==label))
            affected=int(np.count_nonzero((b!=result)&((b==label)|(result==label))))
            if affected/max(n,1)>.50 or abs(m-n)/max(n,1)>.25:reject.append(label)
        if not reject:break
        restore=np.isin(b,reject)|np.isin(result,reject)
        if not np.any(result[restore]!=b[restore]):raise ValueError('unresolved_volume_gate')
        result[restore]=b[restore];capped.extend(reject)
    assert np.array_equal(result>0,b>0)
    assert not np.any((result!=b)&protected)
    assert np.array_equal(result==24,b==24)
    if region.atlas_marker:assert np.all(result[b==23]==23)
    assert np.all(result[b==24]==24)
    out=base.copy();out[roi]=result
    return out,{'region':asdict(region),'reference_labels_used':False,
        'body_centers_RAS_mm':centers.tolist(),'axis_basis_world_columns':basis.tolist(),
        'physical_cost_powers':[1.,1.5,2.],'required_each_scale_runner_up_margin':.10,
        'marker_voxels':{str(k):len(v) for k,v in markers.items()},
        'native_ROI_shape':list(b.shape),'capped_labels':sorted(set(capped)),
        'median_CT_foreground_to_surrounding_shell_contrast_HU':contrast,
        'foreground_exact':True,'concordant_source_body_core_voxels_exact':True,'posterior_gate_used':False,
        'C1_mask_exact':True,'existing_C2_voxels_frozen':True if region.atlas_marker else None,'changed_voxels':int(np.count_nonzero(out!=base)),
        'status':'applied' if np.any(out!=base) else 'abstained','DSC':None}

def refine(base,ct,affine,anchors):
    if base.shape!=ct.shape or base.ndim!=3 or not np.isfinite(ct).all():raise ValueError('Matching finite 3D CT required')
    if not np.issubdtype(base.dtype,np.integer) or base.min()<0 or base.max()>24:raise ValueError('Integer labels0..24 required')
    if anchors.shape!=base.shape or not np.issubdtype(anchors.dtype,np.integer) or anchors.min()<0 or anchors.max()>24:raise ValueError('Matching integer SuPreM anchor labels0..24 required')
    out=base.copy();reports=[]
    for region in REGIONS:
        try:out,r=refine_region(out,ct,affine,region,anchors)
        except ValueError as e:r={'region':asdict(region),'status':'abstained','reason':str(e),'changed_voxels':0,'reference_labels_used':False,'DSC':None}
        reports.append(r)
    assert np.array_equal(out>0,base>0)
    return out,{'method':'Independent SuPreM body anchors classify CT neural foreground with 3D geodesic consensus',
        'reference_labels_used':False,'foreground_exact':True,'regions':reports,
        'changed_voxels':int(np.count_nonzero(out!=base)),
        'status':'applied' if np.any(out!=base) else 'abstained','DSC':None,
        'credit':'Foreground: Jakob Wasserthal et al. / TotalSegmentator and Hao Yu / unchanged ShapeKit-Hao. Identity anchors: authentic SuPreM plus credited Hao and original fixed roots. Guided graph: original experimental research code.'}
