from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import scipy.sparse as sp
import pytest
from feselection import FESelector, SelectionConfig, ResourceBudget
from feselection.cost_policy import CostPolicy
from feselection.numerical_screen import NumericalScreenPolicy, preview_partition
from feselection.allocation import AllocationLedger, CandidateRejected
from test_selector import tiny, ref


def example(weak=False,cross=.3):
    a=.999 if weak else .2
    A=np.array([[1.,a],[a,1.]])
    C=np.full((2,2),cross)
    T=sp.csc_matrix(np.block([[A,C],[C,A]]))
    lab=np.array([0,0,1,1])
    upper=sp.triu(T,1).tocoo();en=upper.data**2
    cut=float(en[lab[upper.row]!=lab[upper.col]].sum()/en.sum())
    return T,{'labels':lab,'cut_energy_fraction':cut}


def screen(T,p,policy=None,budget=None,eig_cut=.005):
    return preview_partition(T,p,AllocationLedger(budget or ResourceBudget()),
                eig_cut=eig_cut,policy=policy or NumericalScreenPolicy())


def test_high_cut_without_weak_local_direction_rejected():
    T,p=example()
    out=screen(T,p)
    assert not out['accepted']
    assert out['reason']=='large_cut_without_sampled_spectral_evidence'
    assert len(out['inspected_blocks'])==2
    assert min(x['relative_minimum'] for x in out['inspected_blocks'])>.5


def test_high_cut_with_weak_direction_has_evidence_not_speed_proof():
    T,p=example(weak=True,cross=.4)
    assert np.linalg.eigvalsh(T.toarray()).min()>0
    out=screen(T,p)
    assert out['accepted'] and out['reason']=='sampled_local_low_energy_direction'
    assert len(out['inspected_blocks'])==1


def test_no_coarse_requirement_for_small_cut_blocks():
    T,p=example(cross=.001)
    out=screen(T,p,eig_cut=0.)
    assert out['accepted'] and out['inspected_blocks']==[]


def test_explicit_ablation_disables_only_routing_screen():
    T,p=example();out=screen(T,p,NumericalScreenPolicy(enabled=False))
    assert out['accepted'] and out['reason']=='screen_disabled_for_ablation'


def test_high_cut_with_coarse_disabled_is_economic_rejection():
    T,p=example();out=screen(T,p,eig_cut=0.)
    assert not out['accepted'] and out['reason']=='large_cut_coarse_disabled'


def test_preview_workspace_is_checked_before_dense_allocation(monkeypatch):
    T,p=example()
    def forbidden(*a,**kw):raise AssertionError('dense allocation reached')
    monkeypatch.setattr(type(T),'toarray',forbidden)
    with pytest.raises(CandidateRejected,match='candidate_workspace_budget'):
        screen(T,p,budget=ResourceBudget(workspace_mib=1e-6))


def test_preview_work_and_block_count_are_bounded():
    T,p=example()
    out=screen(T,p,NumericalScreenPolicy(max_preview_cubic_work=1))
    assert not out['accepted'] and not out['inspected_blocks']
    assert out['preview_budget_exhausted']
    out=screen(T,p,NumericalScreenPolicy(max_preview_blocks=1))
    assert len(out['inspected_blocks'])==1


@pytest.mark.parametrize('perm',[[1,0,3,2],[3,1,2,0]])
def test_numeric_evidence_not_based_on_original_level_labels(perm):
    T,p=example(weak=True,cross=.4)
    ix=np.array(perm)
    got=screen(T[ix,:][:,ix],dict(labels=p['labels'][ix],cut_energy_fraction=p['cut_energy_fraction']))
    assert got['accepted']
    assert got['inspected_blocks'][0]['relative_minimum']==pytest.approx(screen(T,p)['inspected_blocks'][0]['relative_minimum'])


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,1.1])
def test_invalid_cut_policy(bad):
    with pytest.raises(ValueError):NumericalScreenPolicy(max_cut_without_spectral_evidence=bad)


@pytest.mark.parametrize('key,value',[('max_preview_blocks',0),('max_preview_blocks',True),
    ('max_preview_cubic_work',-1),('max_preview_cubic_work',1.5),('enabled',1)])
def test_invalid_screen_policy(key,value):
    with pytest.raises(ValueError):NumericalScreenPolicy(**{key:value})


@pytest.mark.parametrize('bad',[True,0,2,3.5])
def test_invalid_warm_confirmation_width(bad):
    with pytest.raises(ValueError):CostPolicy(warm_confirmation_min_rhs=bad)


def test_warm_confirmation_is_actual_retained_work(monkeypatch):
    c,w,y=tiny(722);Y=np.tile(y,(1,4))
    cfg=SelectionConfig(mode='auto',cost=CostPolicy(setup_iteration_equivalents=0,planning_seconds_cap=100.))
    plan=FESelector(cfg).prepare(c,w);plan.inspection_seconds=0.
    original=plan._native_solve;count=[0]
    def measured(y,events,**kw):
        r,t,it=original(y,events,**kw);count[0]+=1
        it=80 if count[0]==1 else 25
        events[-1].update(iterations=it,escalated=True)
        plan._last_warm_seconds=1.
        return r,1.,it
    def forbidden(*args,**kwargs):raise AssertionError('weak warm evidence should bypass build')
    monkeypatch.setattr(plan,'_native_solve',measured)
    monkeypatch.setattr(plan,'_build_candidate',forbidden)
    got=plan.residualize(Y)
    assert count[0]==2
    assert any(e['stage']=='useful_warm_confirmation' and e['warm_iterations']==25 for e in got.diagnostics['events'])
    assert any(e['stage']=='native_batch' and e['columns']==10 for e in got.diagnostics['events'])
    np.testing.assert_allclose(got.residuals,ref(c,w,Y),atol=2e-10)


def test_local_pilot_uses_next_unsolved_column_after_confirmation(monkeypatch):
    c,w,y=tiny(723);rng=np.random.default_rng(84);Y=rng.normal(size=(len(c),8))
    cfg=SelectionConfig(mode='auto',cost=CostPolicy(setup_iteration_equivalents=0,planning_seconds_cap=100.))
    plan=FESelector(cfg).prepare(c,w);plan.inspection_seconds=0.
    original=plan._native_solve;native_columns=[];local_columns=[]
    def measured(y,events,**kw):
        native_columns.append(y.copy());r,_,_=original(y,events,**kw)
        events[-1].update(iterations=80,escalated=True);plan._last_warm_seconds=1.
        return r,1.,80
    def build(*args,**kwargs):plan.root=object();plan.system=object()
    def local(y,events,maxiter):
        local_columns.append(y.copy());r=ref(c,w,y[:,None])[:,0]
        plan._last_normals=np.array([0.]);return r,.1,True
    monkeypatch.setattr(plan,'_native_solve',measured);monkeypatch.setattr(plan,'_build_candidate',build)
    monkeypatch.setattr(plan,'_local_column',local)
    got=plan.residualize(Y)
    assert len(native_columns)==2 and len(local_columns)==6
    np.testing.assert_array_equal(local_columns[0],Y[:,2])
    np.testing.assert_allclose(got.residuals,ref(c,w,Y),atol=2e-10)


def test_prefactor_rejection_precedes_component_factorization(monkeypatch):
    import feselection.selector as module
    c,w,y=tiny(835);p=FESelector(SelectionConfig(mode='auto')).prepare(c,w)
    def reject(*a,**kw):return {'accepted':False,'reason':'no_numeric_evidence','stage':'prefactor_numerical_screen'}
    def forbidden(*a,**kw):raise AssertionError('full block factorization reached')
    monkeypatch.setattr(module,'preview_partition',reject)
    monkeypatch.setattr(module,'ComponentRight',forbidden)
    events=[]
    with pytest.raises(CandidateRejected,match='no_numeric_evidence'):
        p._build_candidate(y,float('inf'),events)
    assert p.root is None and p.cache_builds==0
    assert any(e['stage']=='prefactor_numerical_screen' for e in events)
