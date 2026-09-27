"""Resource-bounded and mathematical edge guards for the next local release.

Small exact oracles and explicit projections are deliberately independent of
production elimination and absorption. Fault injection is not a claim that a
large real-world workload naturally exhausts the same resource.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import product
import importlib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from econhdfe.compute.encoding import factorize_1d
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.compute.stable_linalg import equilibrated_lstsq
from econhdfe.iv.solve import weighted_2sls_block
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.plan import FEPlan
from econhdfe import PPMLConfig
from econhdfe.models.ppml.heterogeneous import fit_block_arrays


def block(a):
    a=np.asarray(a,dtype=float); n,k=a.shape
    return BlockDesign(n,k,(DenseDesignBlock(0,np.arange(n),np.arange(k),a.copy()),))


def fraction_rank(A):
    rows=[[Fraction(int(v)) for v in row] for row in A]; r=0
    for j in range(len(rows[0])):
        p=next((i for i in range(r,len(rows)) if rows[i][j]),None)
        if p is None: continue
        rows[r],rows[p]=rows[p],rows[r]
        c=rows[r][j]; rows[r]=[v/c for v in rows[r]]
        for i in range(r+1,len(rows)):
            c=rows[i][j]
            rows[i]=[v-c*t for v,t in zip(rows[i],rows[r])]
        r+=1
        if r==len(rows): break
    return r


def parity_edges():
    return np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=np.int32)


@pytest.mark.parametrize('budget',[0,1,2])
def test_actual_gf2_budget_shortfall_is_not_an_exact_rank(monkeypatch,budget):
    mod=importlib.import_module('econhdfe.hdfe.rank')
    edges=parity_edges(); global_edges=edges+np.array([0,2,4])
    rows,ncols=mod._reduced_component_rows(global_edges)
    lower=mod._gf2_bitset_rank(rows,4,max_xors=budget)
    assert lower < 4
    monkeypatch.setattr(mod,'_GF2_MIN_XOR_BUDGET',budget)
    monkeypatch.setattr(mod,'_GF2_MAX_XOR_BUDGET',budget)
    # The same bad prime forces actual integer, rather than modular, fallback.
    monkeypatch.setattr(mod,'_EXACT_PRIME',2)
    calls=[]; original=mod._rational_sparse_rank
    def exact(a, **kwargs):
        calls.append(True); return original(a, **kwargs)
    monkeypatch.setattr(mod,'_rational_sparse_rank',exact)
    D=np.column_stack([np.eye(2,dtype=int)[g] for g in edges.T])
    assert mod.categorical_rank(list(edges.T),backend='native')==fraction_rank(D)==4
    assert calls


def test_sympy_bad_prime_uses_characteristic_zero_result(monkeypatch):
    pytest.importorskip('sympy',reason='optional exact backend unavailable')
    mod=importlib.import_module('econhdfe.hdfe.rank')
    edges=parity_edges()+np.array([0,2,4]); rows,ncols=mod._reduced_component_rows(edges)
    monkeypatch.setattr(mod,'_EXACT_PRIME',2)
    assert mod._sympy_sparse_rank(rows,ncols,4)==4
    assert mod.categorical_rank(list(parity_edges().T),backend='sympy')==4


@pytest.mark.parametrize('error',[MemoryError,RuntimeError])
def test_exact_backend_failure_never_returns_uncertified_rank(monkeypatch,error):
    mod=importlib.import_module('econhdfe.hdfe.rank')
    monkeypatch.setattr(mod,'_EXACT_PRIME',2)
    def failed(*args,**kwargs): raise error('injected exact arithmetic failure')
    monkeypatch.setattr(mod,'_rational_sparse_rank',failed)
    with pytest.raises(error,match='injected'):
        mod.categorical_rank(list(parity_edges().T),backend='native')


def test_flint_dense_budget_checked_before_dependency_import():
    mod=importlib.import_module('econhdfe.hdfe.rank')
    with pytest.raises(ValueError,match='limit'):
        mod._flint_dense_rank([(0,),(1,)],2,max_cells=0)


@pytest.mark.parametrize('labels',[[0,10**9,0,10**9],[0,100,3,100],[0,1,2,1]])
def test_sparse_integer_labels_do_not_allocate_by_largest_label(monkeypatch,labels):
    mod=importlib.import_module('econhdfe.compute.encoding')
    original=mod.np.bincount
    def bounded(x,**kwargs):
        assert kwargs.get('minlength',0)<=len(labels), 'unbounded label-based allocation'
        return original(x,**kwargs)
    monkeypatch.setattr(mod.np,'bincount',bounded)
    codes,k=factorize_1d(np.array(labels))
    assert k==len(set(labels))
    assert np.array_equal(codes[:,None]==codes[None,:],np.array(labels)[:,None]==np.array(labels)[None,:])


@pytest.mark.parametrize('method,acceleration',[('map','none'),('map','cg'),('lsmr','none')])
@pytest.mark.parametrize('core',['off','on'])
def test_weighted_projection_orthogonality_idempotence_and_update(method,acceleration,core):
    rng=np.random.default_rng(62314)
    rows=np.array(list(product(range(3),range(2),range(2))),dtype=int)
    rows=np.r_[rows,rows,[[3,2,2],[4,3,3]]]
    gs=list(rows.T); n=len(rows)
    D=np.column_stack([np.eye(g.max()+1)[g] for g in gs])
    Y=rng.normal(size=(n,3)); w=np.geomspace(.02,20,n)
    a=HDFEAbsorber(gs,weights=w,tol=1e-12,method=method,acceleration=acceleration,
                   core_reduction=core,absorb_threads=1,max_iter=10000)
    for wi in [w,w[::-1].copy()]:
        a.update_weights(wi)
        got=a.residualize(Y)
        sw=np.sqrt(wi)
        oracle=Y-D@np.linalg.lstsq(sw[:,None]*D,sw[:,None]*Y,rcond=None)[0]
        assert_allclose(got,oracle,atol=2e-8,rtol=2e-8)
        assert_allclose(D.T@(wi[:,None]*got),0.,atol=2e-8)
        assert_allclose(a.residualize(got),got,atol=2e-8,rtol=2e-8)
        assert_allclose(a.residualize(D@rng.normal(size=(D.shape[1],2))),0.,atol=2e-8)


@pytest.mark.parametrize('shape',[(80,4),(4,8),(2,2),(0,3),(6,0)])
@pytest.mark.parametrize('chunk',[1,17,32768])
def test_equilibrated_qr_matches_explicit_scaled_reference(shape,chunk):
    rng=np.random.default_rng(7360); n,k=shape
    X=rng.normal(size=shape); y=rng.normal(size=n)
    if k: X*=np.geomspace(1e-8,1e8,k)
    b,V,r=equilibrated_lstsq(X,y,chunk_rows=chunk)
    if not n or not k:
        assert r==0 and b.shape==(k,) and V.shape==(k,k); return
    scale=np.max(abs(X),axis=0); scale[scale==0]=1
    P=np.linalg.pinv(X/scale)
    oracle=P@y/scale; inverse=P/scale[:,None]
    assert_allclose(b,oracle,rtol=2e-10,atol=1e-8)
    assert_allclose(V,inverse@inverse.T,rtol=2e-10,atol=1e-8)
    assert r==np.linalg.matrix_rank(X/scale)


@pytest.mark.parametrize('scale',[1e-8,1.,4e7])
def test_block_iv_rank_and_bread_use_the_same_equilibrated_coordinates(scale):
    rng=np.random.default_rng(7026); n=240
    Z=rng.normal(size=(n,3)); X=Z[:,:2]+rng.normal(scale=.15,size=(n,2)); y=X@np.array([.1,.5])+rng.normal(size=n)
    ref=weighted_2sls_block(y,block(X),block(Z))
    units=np.array([1.,scale]); got=weighted_2sls_block(y,block(X*units),block(Z))
    assert got.rank==ref.rank==2
    assert_allclose(got.beta*units,ref.beta,rtol=2e-9,atol=1e-10)
    assert_allclose(got.bread*units[:,None]*units[None,:],ref.bread,rtol=2e-9,atol=1e-10)


@pytest.mark.parametrize('kind',['model','robust','cluster'])
@pytest.mark.parametrize('scale',[1.,4e7])
def test_ppml_block_covariance_preserves_units(kind,scale):
    rng=np.random.default_rng(61526);n=400;g=np.arange(n)%8
    X=rng.normal(size=(n,2)); y=rng.poisson(np.exp(.5+X@np.array([.1,.2])+.04*g))
    cfg=PPMLConfig(standardize=False,separation=(),tolerance=1e-11,target_inner_tol=1e-12)
    def fit(X):
        return fit_block_arrays(y,block(X),FEPlan.from_arrays([g]),offset=np.zeros(n),true_w=np.ones(n),
              vce=kind,clusters=[g] if kind=='cluster' else None,names=('x','z'),config=cfg)
    ref=fit(X); got=fit(X*np.array([1.,scale]));units=np.array([1.,scale])
    assert_allclose(got.coef*units,ref.coef,rtol=2e-6,atol=2e-8)
    assert_allclose(got.vcov*units[:,None]*units[None,:],ref.vcov,rtol=2e-5,atol=1e-10)


@pytest.mark.parametrize('chunk',[1,9,32768])
@pytest.mark.parametrize('zero',[False,True])
def test_qr_rank_deficiency_retains_scaled_minimum_norm_and_input(chunk,zero):
    rng=np.random.default_rng(501); X=rng.normal(size=(70,3)); X[:,2]=2*X[:,0]
    if zero: X[:,1]=0
    X*=np.array([1e-6,3.,1e6]); y=rng.normal(size=70);before=X.copy()
    scale=np.max(abs(X),axis=0);scale[scale==0]=1
    P=np.linalg.pinv(X/scale,rcond=np.finfo(float).eps*70);oracle=P@y/scale
    b,V,rank=equilibrated_lstsq(X,y,chunk_rows=chunk)
    assert rank==(1 if zero else 2)
    assert_allclose(b,oracle,rtol=2e-10,atol=1e-8)
    assert_allclose(X@b,X@oracle,rtol=2e-10,atol=1e-10)
    assert_allclose(V,(P/scale[:,None])@(P/scale[:,None]).T,rtol=2e-10,atol=1e-8)
    assert np.array_equal(X,before)


@pytest.mark.parametrize('labels',['string','shifted','fractional'])
def test_ppml_block_cluster_relabeling(labels):
    rng=np.random.default_rng(919);n=240;g=np.arange(n)%8
    X=rng.normal(size=(n,2));y=rng.poisson(np.exp(.6+X@np.array([.1,.2])))
    other=np.array([f'group-{a}' for a in g]) if labels=='string' else g+(.25 if labels=='fractional' else 100)
    cfg=PPMLConfig(separation=())
    def fit(c):
        return fit_block_arrays(y,block(X),FEPlan.from_arrays([g]),offset=np.zeros(n),true_w=np.ones(n),
                               vce='cluster',clusters=[c],names=('x','z'),config=cfg)
    r=fit(g);s=fit(other)
    assert r.cluster_counts==s.cluster_counts==(8,)
    assert_allclose(r.vcov,s.vcov,rtol=1e-10,atol=1e-12)
