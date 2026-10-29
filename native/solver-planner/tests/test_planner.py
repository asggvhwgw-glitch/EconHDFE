import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from econhdfe_planner import PlannedAbsorber, PlanningError, ProjectionDisagreement
from econhdfe_planner.structure import compact_codes, tuple_sketch

ROUTES = ['within', 'compressed_within', 'fereg', 'auto']


def draw(seed=91, n=720, k=3):
    r=np.random.default_rng(seed)
    g=np.column_stack([r.integers(l,size=n) for l in (31,17,7)]).astype(np.uint32)
    y=r.normal(size=(n,k)); w=np.exp(r.normal(scale=.7,size=n))
    return g,y,w


def oracle(g,y,w):
    d=np.column_stack([np.eye(int(c.max())+1)[c] for c in g.T])
    root=np.sqrt(w)
    coef=np.linalg.lstsq(root[:,None]*d,root[:,None]*y,rcond=1e-13)[0]
    return y-d@coef


@pytest.mark.parametrize('backend',ROUTES)
@pytest.mark.parametrize('k',[1,2,8])
def test_dense_svd_weight_update_and_layout(backend,k):
    g,y,w=draw(k=k)
    y=np.asfortranarray(y); original=y.copy()
    s=PlannedAbsorber(g,weights=w,backend=backend,allow_fallback=False)
    for step in range(2):
        ww=w*np.exp(.2*np.sin(np.arange(len(w))*(.1+step)))
        s.update_weights(ww)
        out,info=s.residualize(y,tol=1e-10,return_info=True)
        assert info['accepted']
        assert_allclose(out,oracle(g,y,ww),atol=2e-8,rtol=2e-8)
        assert_allclose(y,original,atol=0,rtol=0)
    assert s.weight_generation==2
    json.dumps(s.diagnostics)


@pytest.mark.parametrize('backend',ROUTES)
def test_invalid_update_and_input(backend):
    g,y,w=draw(n=120)
    s=PlannedAbsorber(g,weights=w,backend=backend)
    before=s.residualize(y)
    for bad in [w*0, -w, np.full(len(w),np.nan), w[:-1]]:
        with pytest.raises(ValueError): s.update_weights(bad)
    assert s.weight_generation==0
    assert_allclose(s.residualize(y),before,atol=1e-12)
    with pytest.raises(ValueError): s.residualize(y,tol=0)
    with pytest.raises(ValueError): s.residualize(y,max_iter=True)
    with pytest.raises(ValueError): s.residualize(np.full_like(y,np.inf))


@pytest.mark.parametrize('backend',ROUTES)
def test_zero_constant_empty_vector_and_sparse_labels(backend):
    g,y,w=draw(n=220)
    g=g*1000000+99999
    s=PlannedAbsorber(g,weights=w,backend=backend)
    assert s.residualize(np.empty((len(g),0))).shape==(len(g),0)
    z=s.residualize(np.column_stack([np.zeros(len(g)),np.ones(len(g))]))
    assert_allclose(z,0,atol=2e-10)
    assert s.residualize(y[:,0]).shape==(len(g),)


def test_sketch_sees_shuffled_pairs_when_small_sample_does_not():
    r=np.random.default_rng(10)
    n=120000
    unique=np.column_stack([np.arange(n//2),r.integers(100,size=n//2)]).astype(np.uint32)
    a=np.tile(unique,(2,1)); r.shuffle(a)
    x=tuple_sketch(a); y=tuple_sketch(a[r.permutation(n)],chunk_rows=3333)
    assert abs(x.distinct_ratio-.5)<.025
    assert x==y
    assert x.scanned_rows==n and x.ratio_upper<.7
    distinct=np.column_stack([np.arange(n),r.integers(100,size=n)]).astype(np.uint32)
    x=tuple_sketch(distinct)
    assert x.distinct_ratio>.96 and x.ratio_upper>.98


@pytest.mark.parametrize('copies',[1,2,16])
def test_sketch_ratio_varies_without_order_bias(copies):
    base=np.column_stack([np.arange(10000),np.arange(10000)%3]).astype(np.uint32)
    a=np.repeat(base,copies,axis=0)
    s=tuple_sketch(a)
    assert abs(s.distinct_estimate/10000-1)<.04
    assert abs(s.distinct_ratio-1/copies)<.04


def test_auto_candidates_accounting_and_reuse():
    g,y,w=draw(n=2000)
    s=PlannedAbsorber(g,weights=w)
    _,i=s.residualize(y,return_info=True)
    assert i['calibrated_this_call'] and i['plan_id']==1
    trials=[x for x in i['plan']['trials'] if x['status']=='accepted']
    assert s.backend==min(trials,key=lambda x:x['score_seconds'])['backend']
    assert i['plan']['calibration_seconds']>=sum(x['cold_seconds']+sum(t['total_seconds'] for t in x['warm']) for x in trials)
    name=s.backend
    _,j=s.residualize(y[:,0],return_info=True)
    assert j['plan_id']==1 and not j['calibrated_this_call'] and j['plan_rhs_changed']
    assert s.backend==name
    s.retune(); assert s.backend=='uninitialized'
    _,j=s.residualize(y,return_info=True); assert j['plan_id']==2


def test_route_admission_not_size_veto():
    g,y,w=draw(n=240)
    s=PlannedAbsorber(g,weights=w)
    s._levels=(50000,45000,40000)
    s.residualize(y)
    assert {'within','fereg'}<={x['backend'] for x in s.diagnostics['plan']['trials']}


def test_soft_budget_is_honest():
    g,y,w=draw(n=250)
    s=PlannedAbsorber(g,weights=w,probe_budget_seconds=1e-9)
    s.residualize(y)
    statuses=[v['status'] for v in s.diagnostics['plan']['trials']]
    assert statuses[0]=='accepted' and 'untested_budget' in statuses


def test_failed_candidate_is_not_ranked_and_failure_rolls_back(monkeypatch):
    g,y,w=draw(n=240); s=PlannedAbsorber(g,weights=w)
    new=s._new_route
    def build(name):
        if name=='fereg': raise PlanningError('injected construction failure')
        return new(name)
    monkeypatch.setattr(s,'_new_route',build)
    s.residualize(y)
    assert s.backend!='fereg'
    assert any(t['status']=='failed' for t in s.diagnostics['plan']['trials'])
    def fail(_): raise RuntimeError('injected numeric refresh failure')
    monkeypatch.setattr(s._route,'update_weights',fail)
    old=s._w.copy()
    with pytest.raises(RuntimeError):s.update_weights(w*2)
    assert_allclose(s._w,old,atol=0,rtol=0); assert s.weight_generation==0
    assert_allclose(s.residualize(y),oracle(g,y,w),atol=2e-8)


def test_disagreement_fails_closed(monkeypatch):
    g,y,w=draw(n=240); s=PlannedAbsorber(g,weights=w)
    new=s._new_route
    class Corrupt:
        name='fereg'
        def project(self,data,tol,max_iter): return data+100,{'accepted':True}
        def update_weights(self,w): pass
    monkeypatch.setattr(s,'_new_route',lambda name:Corrupt() if name=='fereg' else new(name))
    with pytest.raises(ProjectionDisagreement):s.residualize(y)
    assert s.backend=='uninitialized'


def test_solver_failure_uses_certified_alternative(monkeypatch):
    g,y,w=draw(n=240); s=PlannedAbsorber(g,weights=w,backend='within')
    s.residualize(y)
    monkeypatch.setattr(s._route,'project',lambda *args: (_ for _ in ()).throw(PlanningError('injected')))
    out=s.residualize(y)
    assert s.backend!='within'
    assert_allclose(out,oracle(g,y,w),atol=2e-8)
    assert s.diagnostics['plan']['reason']=='certification_failover'


def test_reweight_probe_measures_update():
    g,y,w=draw(n=240); s=PlannedAbsorber(g,weights=w,workload='reweight')
    s.residualize(y)
    assert s.weight_generation==0
    for t in s.diagnostics['plan']['trials']:
        if t['status']=='accepted': assert t['warm'][0]['update_seconds']>0


@pytest.mark.parametrize('bad',[np.array([1,2]),np.ones((10,2)),np.array([[-1],[0]]),np.empty((0,2),dtype=int)])
def test_bad_codes(bad):
    with pytest.raises(ValueError):PlannedAbsorber(bad)


def test_nested_disconnected_singletons_not_removed():
    rng=np.random.default_rng(13); n=201
    a=np.arange(n)//5; b=a%4; comp=np.arange(n)//100
    g=np.column_stack([a,b,comp,a]).astype(np.uint32)
    y=rng.normal(size=(n,2));w=np.exp(rng.normal(size=n))
    for route in ROUTES:
        z=PlannedAbsorber(g,weights=w,backend=route).residualize(y)
        assert z.shape==y.shape
        assert_allclose(z,oracle(g,y,w),atol=2e-8)


@pytest.mark.parametrize('backend',ROUTES)
def test_common_ppml_outer_loop_dense_parity(backend):
    r=np.random.default_rng(2831);n=600
    g=r.integers(0,10,size=(n,3),dtype=np.uint32)
    x=r.normal(size=(n,2));d=np.column_stack([x]+[np.eye(10)[c] for c in g.T])
    y=np.exp(.3+x@np.array([.13,-.11])+.08*r.normal(size=n))
    def fit(projector):
        eta=np.full(n,np.log(y.mean()))
        for step in range(40):
            mu=np.exp(eta);z=eta+(y-mu)/mu
            if projector is None:
                coef=np.linalg.lstsq(np.sqrt(mu)[:,None]*d,np.sqrt(mu)*z,rcond=1e-13)[0]
                new=d@coef;beta=coef[:2]
            else:
                projector.update_weights(mu)
                q=projector.residualize(np.column_stack([z,x]),tol=1e-11)
                beta=np.linalg.solve(q[:,1:].T@(mu[:,None]*q[:,1:]),q[:,1:].T@(mu*q[:,0]))
                new=z-q[:,0]+q[:,1:]@beta
            if np.max(np.abs(new-eta))<1e-10:return beta,np.exp(new)
            eta=new
        raise AssertionError('outer loop failed')
    b,mu=fit(PlannedAbsorber(g,backend=backend,workload='reweight'))
    br,mr=fit(None)
    assert_allclose(b,br,atol=2e-8);assert_allclose(mu,mr,atol=2e-8)
    assert np.max(np.abs(d.T@(y-mu)))/n<1e-8


def test_unique_cycle_is_an_independent_oracle():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'fixtures.py'
    spec=importlib.util.spec_from_file_location('fixtures',path);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    for case in ['unique_three','unique_one','weak_cycle','paired_three','repeated']:
        g,y,w,truth,u=f.corpus(case,640,2,223)
        for step in [0,1]:
            ww=f.updated_weights(w,u,step)
            assert_allclose(oracle(g,y,ww),truth,atol=2e-9)
