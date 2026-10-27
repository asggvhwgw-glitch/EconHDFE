import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import pytest
from feselection import *
from feselection.allocation import (AllocationLedger,inspect,prepare_system,guarded_star,product_row_counts,
                                    group_union_counts,elimination_candidates)
from feselection.partitioning import candidates
from feselection.coarse import GuardedCoarseRight
from budget_preconditioner import ComponentRight,CoarseRight,solve_rect
from scalable_hyperfe import dummy


def tiny(seed=731,q=3):
    rng=np.random.default_rng(seed);n=240
    c=np.column_stack([rng.permutation(n)%x for x in [40,8,6][:q]]).astype(np.int32)
    w=np.exp(rng.normal(size=n));Y=rng.normal(size=(n,3))
    return c,w,Y

def ref(c,w,Y):
    D=dummy(c,[int(c[:,j].max())+1 for j in range(c.shape[1])]).toarray()
    A=np.sqrt(w)[:,None]*D
    return Y-D@np.linalg.lstsq(A,np.sqrt(w)[:,None]*Y,rcond=None)[0]

@pytest.mark.parametrize('mode',['native','observe','auto','candidate'])
@pytest.mark.parametrize('seed',[18,72,311])
def test_independent_weighted_dummy(mode,seed):
    c,w,Y=tiny(seed)
    p=FESelector(SelectionConfig(mode=mode)).prepare(c,w)
    r=p.residualize(Y)
    np.testing.assert_allclose(r.residuals,ref(c,w,Y),rtol=2e-9,atol=2e-10)
    assert all(x<1e-8 for x in r.diagnostics['normal_residuals'])

@pytest.mark.parametrize('bad',[0.,-1.,np.inf,np.nan])
def test_invalid_weights_no_fallback(bad):
    c,w,Y=tiny();w[0]=bad
    with pytest.raises(ValueError,match='weights'):FESelector().prepare(c,w)

@pytest.mark.parametrize('kind',['float','negative','holes','nonmatrix'])
def test_invalid_codes_no_fallback(kind):
    c,w,Y=tiny()
    if kind=='float':c=c.astype(float)
    elif kind=='negative':c[0,0]=-1
    elif kind=='holes':c[:,1]*=2
    elif kind=='nonmatrix':c=c[:,0]
    with pytest.raises(ValueError):FESelector().prepare(c,w)

def test_uniform_not_a_qualification_rejection():
    c,w,Y=tiny();p=FESelector(SelectionConfig(mode='candidate')).prepare(c,np.ones(len(c)))
    r=p.residualize(Y)
    assert p.profile['equal_weights']
    assert not any(e.get('reason')=='uniform_weight_bypass' for e in r.diagnostics['events'])
    assert any(e['stage']=='candidate_built' for e in r.diagnostics['events'])

@pytest.mark.parametrize('q',[0,1])
def test_exact_paths(q):
    c,w,Y=tiny(q=1);c=c[:,:q]
    r=FESelector().prepare(c,w).residualize(Y)
    expected=Y if q==0 else ref(c,w,Y)
    np.testing.assert_allclose(r.residuals,expected,atol=2e-12)

def test_input_snapshot_and_weight_update():
    c,w,Y=tiny();p=FESelector(SelectionConfig(mode='candidate')).prepare(c,w)
    original_c=c.copy();original_w=w.copy()
    r=p.residualize(Y);c[:]=0;w[:]=5
    np.testing.assert_allclose(p.residualize(Y).residuals,r.residuals,rtol=1e-9,atol=1e-10)
    u=p.with_weights(original_w[::-1])
    assert u.fingerprint!=p.fingerprint and u.cached_route is None
    np.testing.assert_allclose(u.residualize(Y).residuals,ref(original_c,original_w[::-1],Y),atol=1e-9)

def test_shape_errors_do_not_fallback():
    c,w,Y=tiny();p=FESelector().prepare(c,w,expected_rhs=3)
    with pytest.raises(ValueError):p.residualize(Y[:,:2])
    Y[0,0]=np.nan
    with pytest.raises(ValueError):p.residualize(Y)

def test_symbolic_product_matches_boolean():
    rng=np.random.default_rng(112)
    for m,k,n in [(31,12,22),(10,1,5),(3,17,4)]:
        A=sp.csr_matrix((rng.random((m,k))>.7).astype(float));B=sp.csr_matrix((rng.random((k,n))>.7).astype(float))
        counts,total,work,status=product_row_counts(A.indptr,A.indices,B.indptr,B.indices,n,100000,100000)
        assert status==0
        np.testing.assert_array_equal(counts,np.diff((A@B).tocsr().indptr))
        assert total==(A@B).nnz

def test_symbolic_reject_before_numeric_gram(monkeypatch):
    c,w,Y=tiny();L=AllocationLedger(ResourceBudget(max_gram_nnz=5))
    s,_=prepare_system(c,w,0,L)
    with pytest.raises(CandidateRejected) as e:guarded_star(s,L)
    assert e.value.reason=='star_gram_nnz_budget'
    assert not any(x['stage']=='star_gram_before_multiply' for x in L.events)

def test_resource_gate_before_profile(monkeypatch):
    from feselection import allocation
    c,w,Y=tiny()
    def no_build(*args,**kwargs):raise AssertionError('must not allocate')
    monkeypatch.setattr(allocation.ProfiledSystem,'build',no_build)
    L=AllocationLedger(ResourceBudget(workspace_mib=.001))
    with pytest.raises(CandidateRejected):prepare_system(c,w,0,L)

def test_partition_relabel_ties():
    sizes=[8,11,7];T=sp.block_diag([np.eye(k)+.25*np.ones((k,k)) for k in sizes],format='csc')
    T=(T+.002*sp.csc_matrix(np.ones(T.shape))).tocsc()
    def choose(M):return candidates(M,AllocationLedger(ResourceBudget(max_block=16)))[0]
    a=choose(T);p=np.random.default_rng(8).permutation(T.shape[0]);ip=np.argsort(p);b=choose(T[p,:][:,p])
    assert a['tau']==b['tau']
    np.testing.assert_array_equal(a['labels'][:,None]==a['labels'][None,:],
                b['labels'][ip][:,None]==b['labels'][ip][None,:])

def test_partition_does_not_modify_T():
    T=sp.csc_matrix(np.eye(6)+.2*np.ones((6,6)));old=T.copy()
    candidates(T,AllocationLedger(ResourceBudget(max_block=3)))
    assert (T!=old).nnz==0

def test_guarded_coarse_matches_frozen():
    c,w,Y=tiny();L=AllocationLedger(ResourceBudget())
    s,_=prepare_system(c,w,0,L);T,scale,_=guarded_star(s,L)
    base=ComponentRight(T,scale,tau=0,max_block=100,eig_cut=.5,max_coarse=100)
    a=CoarseRight(s,base,balanced=True);b=GuardedCoarseRight(s,base,L)
    rng=np.random.default_rng(98);x=rng.normal(size=b.m+b.k);y=rng.normal(size=b.m)
    np.testing.assert_allclose(a.forward(x),b.forward(x),rtol=1e-10,atol=1e-10)
    assert abs(b.forward(x)@y-x@b.adjoint(y))<1e-10

def test_local_pilot_failure_finishes_residual():
    c,w,Y=tiny(885);cfg=SelectionConfig(mode='candidate',pilot_iterations=1)
    result=FESelector(cfg).prepare(c,w).residualize(Y)
    assert any(e['stage']=='native_finish_partial' for e in result.diagnostics['events'])
    np.testing.assert_allclose(result.residuals,ref(c,w,Y),atol=2e-9)

def test_zero_rhs():
    c,w,Y=tiny();Y*=0
    r=FESelector(SelectionConfig(mode='auto')).prepare(c,w).residualize(Y)
    np.testing.assert_array_equal(r.residuals,Y)

@pytest.mark.parametrize('scale',[1e-120,1e120])
def test_weight_global_scaling(scale):
    c,w,Y=tiny();cfg=SelectionConfig(mode='candidate')
    a=FESelector(cfg).prepare(c,w).residualize(Y);b=FESelector(cfg).prepare(c,w*scale).residualize(Y)
    np.testing.assert_allclose(a.residuals,b.residuals,atol=5e-10)

@pytest.mark.parametrize('operation',['rows','fe','labels'])
def test_permutations(operation):
    c,w,Y=tiny();rng=np.random.default_rng(88)
    if operation=='rows':p=rng.permutation(len(c));c,w,Y=c[p],w[p],Y[p]
    elif operation=='fe':c=c[:,[2,0,1]]
    else:
        c=c.copy()
        for j in range(c.shape[1]):c[:,j]=rng.permutation(int(c[:,j].max())+1)[c[:,j]]
    a=FESelector(SelectionConfig(mode='candidate')).prepare(c,w).residualize(Y)
    np.testing.assert_allclose(a.residuals,ref(c,w,Y),atol=3e-9)

def test_not_every_native_error_is_swallowed(monkeypatch):
    c,w,Y=tiny();p=FESelector().prepare(c,w)
    def bug(*a,**kw):raise RuntimeError('test internal failure')
    monkeypatch.setattr(p,'_native_solve',bug)
    with pytest.raises(RuntimeError,match='test internal failure'):p.residualize(Y)

@pytest.mark.parametrize('scale',[1e-100,1e100])
def test_rhs_units(scale):
    c,w,Y=tiny(37);Y*=scale
    r=FESelector(SelectionConfig(mode='candidate')).prepare(c,w).residualize(Y)
    np.testing.assert_allclose(r.residuals/scale,ref(c,w,Y/scale),rtol=1e-8,atol=1e-9)

def test_empty_coarse_is_allowed():
    c,w,Y=tiny(75)
    r=FESelector(SelectionConfig(mode='candidate',eig_cut=0)).prepare(c,w).residualize(Y)
    built=[e for e in r.diagnostics['events'] if e['stage']=='candidate_built']
    assert built and built[0]['kind']=='local_only'
    np.testing.assert_allclose(r.residuals,ref(c,w,Y),atol=1e-9)

def test_coarse_admission_precedes_HZ_product():
    c,w,Y=tiny();ledger=AllocationLedger(ResourceBudget())
    s,_=prepare_system(c,w,0,ledger);T,scale,_=guarded_star(s,ledger)
    b=ComponentRight(T,scale,tau=0,max_block=100,eig_cut=.5,max_coarse=100)
    tiny_ledger=AllocationLedger(ResourceBudget(workspace_mib=.001))
    with pytest.raises(CandidateRejected):GuardedCoarseRight(s,b,tiny_ledger)
    assert len(tiny_ledger.events)==1 and tiny_ledger.events[0]['stage']=='coarse_symbolic_before_allocation'

def test_cost_configuration_invalid():
    from feselection.cost_policy import CostPolicy
    for kw in [dict(gain_margin=float('nan')),dict(exploration_fraction=0),dict(planning_seconds_cap=-1)]:
        with pytest.raises(ValueError):CostPolicy(**kw)

@pytest.mark.parametrize('scale',[1e-200,1e200])
def test_extreme_rhs_units_are_normalized_before_solve(scale):
    c,w,Y=tiny(137);scaled=Y*scale
    r=FESelector(SelectionConfig(mode='candidate')).prepare(c,w).residualize(scaled)
    np.testing.assert_allclose(r.residuals/scale,ref(c,w,Y),rtol=2e-9,atol=2e-9)

def test_single_rhs_does_not_explore():
    c,w,Y=tiny();r=FESelector(SelectionConfig(mode='auto')).prepare(c,w).residualize(Y[:,0])
    assert r.residuals.shape==(len(c),)
    assert any(e.get('reason')=='insufficient_known_rhs' for e in r.diagnostics['events'])
    assert not any(e['stage']=='elimination_shortlist' for e in r.diagnostics['events'])

def test_sparse_symbolic_work_bound():
    A=sp.csr_matrix(np.ones((30,10)));B=sp.csr_matrix(np.ones((10,30)))
    _,_,work,status=product_row_counts(A.indptr,A.indices,B.indptr,B.indices,30,100000,100)
    assert status==2 and work==101

def test_weights_are_not_reused_across_update():
    c,w,Y=tiny();p=FESelector(SelectionConfig(mode='candidate')).prepare(c,w);p.residualize(Y)
    q=p.with_weights(np.ones(len(w)))
    assert q.root is None and q.native is None and q.cache_builds==0
    assert p.fingerprint!=q.fingerprint
    np.testing.assert_allclose(q.residualize(Y).residuals,ref(c,np.ones(len(w)),Y),atol=1e-9)
