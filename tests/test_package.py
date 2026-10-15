import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from nicheir import load_pretrained,validate_inputs,load_inputs,save_inputs,Cells,HalfPlane,Corner,prepare_cells,read_cells_csv
from nicheir.example import synthetic_cells,run_example
from nicheir.preprocessing import _corner_coverage

@pytest.fixture(scope='module')
def predictor():
    torch.set_num_threads(2)
    return load_pretrained()

@pytest.fixture
def batch():
    c,q=synthetic_cells();return prepare_cells(c,q,HalfPlane(0))


def test_seed_and_parameters(predictor):
    assert sum(p.numel() for p in predictor.model.parameters())==2917
    torch.manual_seed(123);before=torch.random.get_rng_state().clone();load_pretrained()
    assert torch.equal(before,torch.random.get_rng_state())


def test_public_example_roundtrip(tmp_path,predictor):
    report=run_example(tmp_path,'corner');assert report['synthetic'] and report['queries']==8
    cells=read_cells_csv(tmp_path/'cells.csv')
    with (tmp_path/'queries.csv').open() as f:ids=[r['cell_id'] for r in csv.DictReader(f)]
    fresh=prepare_cells(cells,ids,Corner(0,0));saved=load_inputs(tmp_path/'inputs.npz')
    for k in saved:np.testing.assert_array_equal(fresh.inputs[k],saved[k])
    result=predictor.predict_npz(tmp_path/'inputs.npz',output=tmp_path/'again.npz')
    with np.load(tmp_path/'predictions.npz') as old:np.testing.assert_array_equal(result.composition,old['composition'])


def test_simplex_nonnegative_and_full_identity(batch,predictor):
    p=predictor.predict(batch.inputs)
    assert np.isfinite(p.composition).all() and (p.composition>=0).all() and (p.missing_counts>=0).all()
    np.testing.assert_allclose(p.composition[p.valid].sum(-1),1,atol=2e-7)
    c,q=synthetic_cells();full=prepare_cells(c,q,HalfPlane(1000)).inputs
    observed=full['local_observed_counts'][full['valid']];result=predictor.predict(full)
    np.testing.assert_allclose(result.composition[result.valid],observed/observed.sum(-1,keepdims=True),rtol=0,atol=0)
    assert not result.missing_counts.any()


def test_count_bins_against_direct_coordinates():
    cells,ids=synthetic_cells('corner');a=prepare_cells(cells,ids,Corner(0,0)).inputs
    for j,name in enumerate(ids):
        index=np.flatnonzero(cells.cell_id==name)[0];v=(cells.xy-cells.xy[index])/72;distance=np.linalg.norm(v,axis=1)
        angle=np.mod(np.arctan2(v[:,1],v[:,0]),2*np.pi);sector=np.floor(angle*16/(2*np.pi)).astype(int)
        rings=np.select([distance<=.5,distance<=1,distance<=2,distance<=3],[0,1,2,3],default=-1)
        keep=(cells.class_id>=0)&(rings>=0);keep[index]=False
        counts=np.zeros((4,16,4),dtype=int)
        np.add.at(counts,(rings[keep],sector[keep],cells.class_id[keep]),1)
        np.testing.assert_array_equal(counts,a['polar_counts'][0,j])


def test_corner_area_independent_quadrature():
    rings=[0,.5,1,2,3];width=np.pi/8
    for dx,dy in [(.2,.6),(.9,.1),(0,0),(.5,.5)]:
        area=_corner_coverage(np.array([[dx,dy]]))[0]
        for r,(lo,hi) in enumerate(zip(rings[:-1],rings[1:])):
            breaks=[]
            for d in [dx,dy]:
                for radius in [lo,hi]:
                    if radius and d<=radius:
                        alpha=np.arccos(d/radius)
                        breaks.extend([alpha,2*np.pi-alpha,(np.pi/2-alpha)%(2*np.pi),(np.pi/2+alpha)%(2*np.pi)])
            breaks += [np.arctan2(dy,dx),0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]
            def integrand(t):
                cap=hi
                if np.cos(t)>1e-14:cap=min(cap,dx/np.cos(t))
                if np.sin(t)>1e-14:cap=min(cap,dy/np.sin(t))
                return max(cap*cap-lo*lo,0)/(hi*hi-lo*lo)
            for s in range(16):
                left=s*width;right=(s+1)*width
                points=sorted(set(v for v in breaks if left<v<right))
                truth=quad(integrand,left,right,points=points,epsabs=1e-10)[0]/width
                assert abs(area[r,s]-truth)<1e-7


def test_isolated_queries_and_reflection(batch,predictor):
    original=batch.inputs;p=predictor.predict(original).composition
    changed={k:v.copy() for k,v in original.items()};changed['polar_counts'][0,0,0,8,0]+=3
    changed['local_observed_counts']=changed['polar_counts'][:,:,:2].sum((2,3))
    y=predictor.predict(changed).composition
    np.testing.assert_array_equal(y[:,1:],p[:,1:])
    reflected={k:v.copy() for k,v in original.items()}
    reflected['polar_counts']=reflected['polar_counts'][:,:,:,::-1].copy()
    for k in ['crop_coverage','visible_hull_support']:reflected[k]=reflected[k][:,:,:,::-1].copy()
    reflected['query_features'][...,2]*=-1
    np.testing.assert_allclose(predictor.predict(reflected).composition,p,atol=3e-7,rtol=2e-6)


def test_energy_descent(batch,predictor):
    x={k:torch.from_numpy(v) for k,v in batch.inputs.items()}
    with torch.no_grad():trace=predictor.model.operator_diagnostics(x)['energy_trace'].numpy()
    assert np.isfinite(trace).all()
    assert np.max(np.diff(trace,axis=0))<2e-5


@pytest.mark.parametrize('failure',['extra','shape','nan','count','binary','support','identity','zero_exposure','centre'])
def test_invalid_inputs_fail_closed(batch,failure):
    a={k:v.copy() for k,v in batch.inputs.items()}
    if failure=='extra':a['hidden_target']=np.zeros(1)
    elif failure=='shape':a['query_features']=a['query_features'][...,:6]
    elif failure=='nan':a['polar_counts'][0,0,0,0,0]=np.nan
    elif failure=='count':a['polar_counts'][0,0,0,0,0]=.5
    elif failure=='binary':a['valid']=a['valid'].astype(int);a['valid'][0,0]=2
    elif failure=='support':a['visible_hull_support'][0,0,0,0]=1;a['crop_coverage'][0,0,0,0]=0
    elif failure=='identity':a['local_observed_counts'][0,0,0]+=1
    elif failure=='zero_exposure':a['crop_coverage'][0,0,0,8]=0;a['visible_hull_support'][0,0,0,8]=0;a['polar_counts'][0,0,0,8,0]+=1;a['local_observed_counts']=a['polar_counts'][:,:,:2].sum((2,3))
    else:a['query_features'][0,0,3:]=[1,0,0,0]
    with pytest.raises(ValueError):validate_inputs(a)


def test_boundary_tie_rejected():
    c=Cells(np.array(['centre','ray','a','b']),np.array([[0,-10],[0,-20],[-15,-15],[-30,-30]],float),np.array([3,0,3,0]))
    with pytest.raises(ValueError,match='zero-exposure'):prepare_cells(c,['centre'],Corner(0,0))


def test_hidden_cells_and_mixed_centres_rejected():
    c,q=synthetic_cells();xy=c.xy.copy();xy[0,0]=1
    with pytest.raises(ValueError,match='outside'):prepare_cells(Cells(c.cell_id,xy,c.class_id),q,HalfPlane(0))
    labels=c.class_id.copy();labels[-1]=0
    with pytest.raises(ValueError,match='epithelial'):prepare_cells(Cells(c.cell_id,c.xy,labels),q,HalfPlane(0))


def test_query_chunking(predictor):
    c,_=synthetic_cells();ids=c.cell_id[c.class_id==3][:40];batch=prepare_cells(c,ids,HalfPlane(0))
    assert batch.inputs['valid'].shape==(2,32)
    whole=predictor.predict(batch.inputs,batch_windows=2).composition
    split=predictor.predict(batch.inputs,batch_windows=1).composition
    np.testing.assert_allclose(whole,split,atol=2e-7,rtol=1e-6)
