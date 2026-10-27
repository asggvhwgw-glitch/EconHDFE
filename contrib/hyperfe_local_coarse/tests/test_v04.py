from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
import numpy as np
import pytest
from feselection import FESelector,SelectionConfig,ResourceBudget
from feselection.early_screen import EarlyScreenPolicy,complete_group_rows,scout
from feselection.numerical_screen import NumericalScreenPolicy
from feselection.cost_policy import CostPolicy
from feselection.allocation import AllocationLedger,inspect
from test_selector import tiny,ref


def sample_fixture(seed=16):
    rng=np.random.default_rng(seed)
    g=np.repeat(np.arange(200),6)
    c=np.column_stack([g,rng.integers(0,12,len(g)),rng.integers(0,9,len(g))]).astype(np.int32)
    w=np.exp(rng.normal(size=len(c)))
    return c,w


def run_scout(c,w,policy=None,budget=None):
    return scout(c,w,inspect(c,w),AllocationLedger(budget or ResourceBudget()),
        policy=policy or EarlyScreenPolicy(max_rows=360,max_groups=60,minimum_groups=20),
        screen_policy=NumericalScreenPolicy(),eig_cut=.005)


def test_sample_is_complete_groups_and_bounded():
    c,w=sample_fixture();p=EarlyScreenPolicy(max_rows=365,max_groups=80,minimum_groups=1)
    rows,groups=complete_group_rows(c,0,200,p)
    assert len(rows)==360 and groups==60
    chosen=np.unique(c[rows,0]);assert len(chosen)==60
    np.testing.assert_array_equal(rows,np.flatnonzero(np.isin(c[:,0],chosen)))


def test_sample_row_permutation_preserves_selected_groups():
    c,w=sample_fixture();p=EarlyScreenPolicy(max_rows=360,max_groups=80,minimum_groups=1)
    rows,_=complete_group_rows(c,0,200,p)
    perm=np.random.default_rng(192).permutation(len(c));other,_=complete_group_rows(c[perm],0,200,p)
    np.testing.assert_array_equal(np.sort(perm[other]),rows)


def test_oversized_first_group_is_not_truncated_or_skipped():
    p=EarlyScreenPolicy(max_rows=3,max_groups=5,minimum_groups=1)
    first=np.random.default_rng(p.seed).permutation(5)[0]
    g=np.concatenate([np.repeat(i,10 if i==first else 1) for i in range(5)])
    c=np.column_stack([g,g])
    rows,k=complete_group_rows(c,0,5,p)
    assert not len(rows) and k==0


def test_small_input_returns_unknown_without_numeric_build(monkeypatch):
    import feselection.early_screen as m
    c,w=sample_fixture()
    monkeypatch.setattr(m,'prepare_system',lambda *a,**kw:pytest.fail('sample build reached'))
    out=run_scout(c,w,EarlyScreenPolicy())
    assert out['verdict']=='unknown' and out['reason']=='small_input_no_scout'


def test_unavailable_workspace_does_not_allocate_sample(monkeypatch):
    import feselection.early_screen as m
    c,w=sample_fixture()
    monkeypatch.setattr(m,'complete_group_rows',lambda *a,**kw:pytest.fail('sample allocated'))
    out=run_scout(c,w,EarlyScreenPolicy(max_rows=360,max_groups=60,minimum_groups=1,workspace_mib=.0001))
    assert out['verdict']=='unknown' and out['reason']=='scout_workspace_unavailable'


def test_low_level_coverage_is_unknown_not_rejection():
    c,w=sample_fixture();c[:,1]=c[:,0]
    out=run_scout(c,w)
    assert out['verdict']=='unknown' and out['reason']=='sample_level_coverage_insufficient'


@pytest.mark.parametrize('accepted',[False,True])
def test_proxy_evidence_tristate_and_does_not_modify_input(monkeypatch,accepted):
    import feselection.early_screen as m
    c,w=sample_fixture();cb=c.copy();wb=w.copy()
    def mock(T,p,ledger,**kw):
        return {'accepted':accepted,'reason':'sampled_local_low_energy_direction' if accepted else 'large_cut_without_sampled_spectral_evidence',
                'inspected_blocks':[{'size':5}],'cut_energy_fraction':.3}
    monkeypatch.setattr(m,'preview_partition',mock)
    out=run_scout(c,w)
    assert out['verdict']==('promising' if accepted else 'reject')
    assert out['sample_rows']<len(c) and out['target_rows_unchanged']
    np.testing.assert_array_equal(c,cb);np.testing.assert_array_equal(w,wb)


def test_proxy_failure_cannot_certify_global_singularity(monkeypatch):
    import feselection.early_screen as m
    c,w=sample_fixture()
    def fail(*args,**kwargs):raise ValueError('sample has extra zero direction')
    monkeypatch.setattr(m,'prepare_system',fail)
    out=run_scout(c,w)
    assert out['verdict']=='unknown' and out['reason']=='proxy_not_certifiable'
    assert 'zero direction' in out['proxy_issue']


def test_weight_rescaling_does_not_change_proxy_decision():
    c,w=sample_fixture();a=run_scout(c,w);b=run_scout(c,w*37.)
    assert a['verdict']==b['verdict'] and a['reason']==b['reason']
    assert a['remaining_level_coverage']==b['remaining_level_coverage']


@pytest.mark.parametrize('key,value',[
    ('enabled',1),('max_rows',0),('max_groups',False),('minimum_groups',-1),
    ('seed',0),('workspace_mib',float('nan')),('workspace_mib',0),
    ('minimum_level_coverage',0),('minimum_level_coverage',1.1),
    ('warm_confirmation_max_iteration_ratio',.5),
    ('warm_confirmation_max_iteration_ratio',float('inf'))])
def test_policy_values_are_validated(key,value):
    with pytest.raises(ValueError):EarlyScreenPolicy(**{key:value})


def mocked_plan(monkeypatch,verdict,iterations=80,scout_seconds=.01):
    import feselection.selector as m
    c,w,y=tiny(725);Y=np.tile(y,(1,4))
    cfg=SelectionConfig(mode='auto',cost=CostPolicy(setup_iteration_equivalents=0,planning_seconds_cap=100.))
    p=FESelector(cfg).prepare(c,w);p.inspection_seconds=0.;calls=[]
    original=p._native_solve
    def measured(y,events,**kw):
        calls.append(y.copy());r,_,_=original(y,events,**kw)
        events[-1].update(iterations=iterations,escalated=True);p._last_warm_seconds=1.
        return r,1.,iterations
    monkeypatch.setattr(p,'_native_solve',measured)
    monkeypatch.setattr(m,'scout',lambda *args,**kwargs:dict(stage='early_group_scout',verdict=verdict,seconds=scout_seconds,reason='controlled_test'))
    return p,c,w,Y,calls


def test_negative_scout_no_longer_bypasses_full_data_checks(monkeypatch):
    """0.5 intentionally replaces the unsafe 0.4 sampled-veto contract."""
    from feselection.allocation import CandidateRejected
    p,c,w,Y,calls=mocked_plan(monkeypatch,'reject');builds=[]
    def reject(*args,**kw):
        builds.append(True)
        raise CandidateRejected('controlled_full_data_rejection')
    monkeypatch.setattr(p,'_build_candidate',reject)
    result=p.residualize(Y)
    assert len(calls)==2 and builds==[True]
    ev=result.diagnostics['events']
    assert any(e['stage']=='sample_evidence_downgraded' for e in ev)
    assert not any(e.get('reason')=='early_scout_rejected_before_full_profile' for e in ev)
    assert any(e.get('reason')=='controlled_full_data_rejection' for e in ev)
    np.testing.assert_allclose(result.residuals,ref(c,w,Y),atol=2e-10)


@pytest.mark.parametrize('verdict,it,expected_native', [('promising',80,2),('promising',160,1),('unknown',160,1),('unknown',80,2)])
def test_positive_proxy_does_not_cancel_marginal_warm_confirmation(monkeypatch,verdict,it,expected_native):
    p,c,w,Y,calls=mocked_plan(monkeypatch,verdict,iterations=it);local_columns=[]
    def build(*a,**kw):p.root=object();p.system=object()
    def local(y,events,maxiter):
        local_columns.append(y.copy());r=ref(c,w,y[:,None])[:,0];p._last_normals=np.array([0.]);return r,.1,True
    monkeypatch.setattr(p,'_build_candidate',build);monkeypatch.setattr(p,'_local_column',local)
    result=p.residualize(Y)
    assert len(calls)==expected_native and len(local_columns)==12-expected_native
    np.testing.assert_array_equal(local_columns[0],Y[:,expected_native])
    np.testing.assert_allclose(result.residuals,ref(c,w,Y),atol=2e-10)


def test_scout_time_is_charged_to_planning_budget(monkeypatch):
    p,c,w,Y,calls=mocked_plan(monkeypatch,'promising',scout_seconds=100.)
    monkeypatch.setattr(p,'_build_candidate',lambda *a,**kw:pytest.fail('spent budget reused'))
    result=p.residualize(Y)
    assert any(e.get('reason')=='scout_spent_construction_budget' for e in result.diagnostics['events'])
    np.testing.assert_allclose(result.residuals,ref(c,w,Y),atol=2e-10)


def test_cached_native_rejection_does_not_reprobe(monkeypatch):
    import feselection.selector as m
    p,c,w,Y,calls=mocked_plan(monkeypatch,'reject')
    from feselection.allocation import CandidateRejected
    def reject(*args,**kw):raise CandidateRejected('controlled_full_data_rejection')
    monkeypatch.setattr(p,'_build_candidate',reject)
    p.residualize(Y)
    monkeypatch.setattr(m,'scout',lambda *a,**kw:pytest.fail('cached route was reprobed'))
    got=p.residualize(Y*3.)
    assert any(e['stage']=='cache_hit' for e in got.diagnostics['events'])
    np.testing.assert_allclose(got.residuals,ref(c,w,Y*3),atol=6e-10)


def test_candidate_mode_bypasses_only_economic_scout(monkeypatch):
    import feselection.selector as m
    c,w,y=tiny(735)
    monkeypatch.setattr(m,'scout',lambda *a,**kw:pytest.fail('candidate scout reached'))
    p=FESelector(SelectionConfig(mode='candidate',eig_cut=.5)).prepare(c,w)
    got=p.residualize(y)
    np.testing.assert_allclose(got.residuals,ref(c,w,y),atol=2e-10)
