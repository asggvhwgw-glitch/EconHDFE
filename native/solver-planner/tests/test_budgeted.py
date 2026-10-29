import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import pytest
from numpy.testing import assert_allclose
from econhdfe_planner import BudgetedAbsorber, PlanningError, ProjectionDisagreement
from econhdfe_planner.budgeted import payback_decision
from econhdfe_planner._bounded_process import TrialWorker, TrialTimeout
from test_planner import draw, oracle


@pytest.mark.parametrize('backend',['auto','within','fereg','compressed_within','legacy_auto'])
@pytest.mark.parametrize('k',[1,2,8])
def test_dense_oracle_and_weights(backend,k):
    g,y,w=draw(k=k)
    with BudgetedAbsorber(g,weights=w,backend=backend,allow_fallback=False) as s:
        old=y.copy()
        for j in range(2):
            ww=w*np.exp(.2*np.sin(np.arange(len(w))*(.1+j)))
            s.update_weights(ww)
            out,info=s.residualize(y,return_info=True)
            assert info['accepted']
            assert_allclose(out,oracle(g,y,ww),atol=2e-8,rtol=2e-8)
            assert_allclose(y,old,atol=0,rtol=0)
        assert s.weight_generation==2
        assert info['plan']['optional_seconds']==0
        json.dumps(info,allow_nan=False)


@pytest.mark.parametrize('h',[1,2,3])
def test_short_horizon_no_worker(monkeypatch,h):
    g,y,w=draw(n=120)
    with BudgetedAbsorber(g,weights=w,expected_calls=h) as s:
        monkeypatch.setattr(s,'_worker',lambda *a:pytest.fail('short horizon started a trial'))
        out,info=s.residualize(y,return_info=True)
        assert info['plan']['reason']=='short_horizon_no_calibration'
        assert info['plan']['trials']==[]
        assert_allclose(out,oracle(g,y,w),atol=2e-8)
        s.update_weights(w*2)
        s.residualize(y)
        assert s.diagnostics['plan']['plan_id']==1


def test_tiny_budget_never_builds(monkeypatch):
    g,y,w=draw(n=120)
    with BudgetedAbsorber(g,weights=w,expected_calls=100,max_probe_seconds=1e-9) as s:
        monkeypatch.setattr(s,'_worker',lambda *a:pytest.fail('tiny budget started a trial'))
        s.residualize(y)
        assert s.diagnostics['plan']['reason']=='deferred_until_actual_reuse'
        s.residualize(y)
        assert s.diagnostics['plan']['reason']=='budget_too_small_no_calibration'


def test_actual_worker_probe_oracle_and_cleanup():
    g,y,w=draw(n=900)
    p=TrialWorker('fereg',g,w,2,512<<20,time.perf_counter()+10)
    root=p._root
    try:
        z,info=p.project(y,1e-10,32,deadline=time.perf_counter()+10)
        assert info['accepted']
        assert_allclose(z,oracle(g,y,w),atol=2e-8)
        p.update_weights(w*2)
        z,_=p.project(y,1e-10,32)
        assert_allclose(z,oracle(g,y,w),atol=2e-8)
    finally:p.close()
    assert not p.alive and not root.exists()


def test_real_unresponsive_process_is_killed(monkeypatch):
    # Substitute child startup only: emulate a Rust call that never checks its
    # Python deadline. No signal is sent to the test runner or incumbent.
    import econhdfe_planner._bounded_process as mod
    original=mod.subprocess.Popen
    def slow(args,**kw):return original([sys.executable,'-c','import time;time.sleep(30)'],**kw)
    monkeypatch.setattr(mod.subprocess,'Popen',slow)
    g,y,w=draw(n=40)
    begin=time.perf_counter()
    p=TrialWorker('fereg',g,w,1,512<<20,begin+3)
    root=p._root
    with pytest.raises(TrialTimeout):p.project(y,1e-10,32,deadline=begin+.2)
    assert time.perf_counter()-begin<2
    assert not p.alive and not root.exists()


def test_worker_gc_has_no_thread_reference_cycle():
    g,y,w=draw(n=40)
    p=TrialWorker('fereg',g,w,1,512<<20,time.perf_counter()+3)
    process=p._process;root=p._root
    del p;gc.collect()
    assert process.poll() is not None and not root.exists()


def make_fast_fake(s,y,w):
    class Fake:
        name='fereg'
        def __init__(self):self.closed=False;self.calls=[]
        def project(self,data,tol,max_iter,**kwargs):
            self.calls.append((data.shape,max_iter))
            return oracle(s._codes,data,w),{'accepted':True,'worker_timing':{'solve_seconds':1e-9}}
        def close(self):self.closed=True
        def update_weights(self,w):pass
    return Fake()


def test_full_rhs_required_before_promotion(monkeypatch):
    g,y,w=draw(n=80,k=8)
    with BudgetedAbsorber(g,weights=w,expected_calls=20,max_probe_seconds=2,
                          max_overhead_fraction=1,exploration_fraction=1) as s:
        oldout,info,_=s._solve_incumbent(y,1e-10,6000)
        candidate=make_fast_fake(s,y,w)
        monkeypatch.setattr(s,'_worker',lambda *a:candidate)
        out,_=s._explore_routes(y,oldout,info,2.,1e-10,6000)
        assert s.backend=='fereg'
        assert candidate.calls==[((80,1),32),((80,8),128)]
        assert not candidate.closed
        assert_allclose(out,oracle(g,y,w),atol=2e-8)
        assert s._plan['trials'][0]['status']=='promoted'


def test_failed_pilot_never_ranked(monkeypatch):
    g,y,w=draw(n=80)
    with BudgetedAbsorber(g,weights=w,expected_calls=20,max_candidates=1) as s:
        oldout,info,_=s._solve_incumbent(y,1e-10,6000)
        candidate=make_fast_fake(s,y,w)
        def reject(*args,**kw):raise PlanningError('iteration cap')
        candidate.project=reject
        monkeypatch.setattr(s,'_worker',lambda *a:candidate)
        out,_=s._explore_routes(y,oldout,info,2.,1e-10,6000)
        assert s.backend=='legacy_auto'
        assert s._plan['trials'][0]['status']=='failed_or_iteration_limited_unranked'
        assert candidate.closed
        assert_allclose(out,oldout,atol=0,rtol=0)


def test_timeout_keeps_certified_incumbent(monkeypatch):
    g,y,w=draw(n=80)
    with BudgetedAbsorber(g,weights=w,expected_calls=20,max_candidates=1) as s:
        oldout,info,_=s._solve_incumbent(y,1e-10,6000)
        monkeypatch.setattr(s,'_worker',lambda *a:(_ for _ in ()).throw(TrialTimeout('injected timeout')))
        out,_=s._explore_routes(y,oldout,info,2.,1e-10,6000)
        assert s.backend=='legacy_auto'
        assert s._plan['trials'][0]['status']=='timeout_unranked'
        assert_allclose(out,oldout,atol=0,rtol=0)


def test_disagreement_is_fatal(monkeypatch):
    g,y,w=draw(n=80)
    with BudgetedAbsorber(g,weights=w,expected_calls=20,max_candidates=1) as s:
        oldout,info,_=s._solve_incumbent(y,1e-10,6000)
        candidate=make_fast_fake(s,y,w)
        candidate.project=lambda data,*a,**kw:(data+100,{'accepted':True})
        monkeypatch.setattr(s,'_worker',lambda *a:candidate)
        with pytest.raises(ProjectionDisagreement):s._explore_routes(y,oldout,info,2.,1e-10,6000)
        assert candidate.closed


def test_payback_includes_all_paid_exploration():
    assert not payback_decision(1,.8,2,3)['accept']
    assert payback_decision(1,.5,.2,3)['accept']
    assert not payback_decision(1,1.1,0,100)['accept']
    assert not payback_decision(1,.1,1,0)['accept']
    assert payback_decision(1,.5,2,6)['estimated_remaining_calls_to_pay_back']==5


@pytest.mark.parametrize('kw',[{'expected_calls':True},{'expected_calls':0},{'max_probe_seconds':float('inf')},
                               {'probe_max_iter':0},{'probe_rhs':0},{'max_candidates':-1},
                               {'minimum_probe_seconds':-1},{'payback_margin':.9}])
def test_invalid_controls(kw):
    g,y,w=draw(n=40)
    with pytest.raises(ValueError):BudgetedAbsorber(g,**kw)


def test_invalid_weight_rollback_and_internal_failure(monkeypatch):
    g,y,w=draw(n=120)
    with BudgetedAbsorber(g,weights=w) as s:
        before=s.residualize(y)
        for bad in [-w,w*0,w[:-1],np.full(len(w),np.nan)]:
            with pytest.raises(ValueError):s.update_weights(bad)
        assert s.weight_generation==0
        monkeypatch.setattr(s._route,'update_weights',lambda w:(_ for _ in ()).throw(PlanningError('injected')))
        with pytest.raises(PlanningError):s.update_weights(w*2)
        assert s._route is None and s.weight_generation==0
        assert_allclose(s.residualize(y),before,atol=2e-10)


def test_empty_zero_constant_sparse_nested_and_retune():
    g,y,w=draw(n=120)
    g=np.column_stack([g,g[:,0]])*1000000+99999
    with BudgetedAbsorber(g,weights=w) as s:
        assert s.residualize(np.empty((len(g),0))).shape==(len(g),0)
        assert s._calls==0
        assert_allclose(s.residualize(np.ones((len(g),2))),0,atol=2e-10)
        s.retune(expected_calls=2)
        before=s._route
        z=s.residualize(y[:,0])
        assert s._route is before
        assert s.diagnostics['plan']['plan_id']==2
        assert_allclose(z,oracle(g//1000000,y,w)[:,0],atol=2e-8)
    with pytest.raises(PlanningError):s.residualize(y)


def test_arbitrary_changing_weights_remote():
    g,y,w=draw(n=600,k=2)
    p=TrialWorker('fereg',g,w,2,512<<20,time.perf_counter()+5)
    try:
        p.project(y,1e-11,32,deadline=time.perf_counter()+5)
        rng=np.random.default_rng(909)
        for _ in range(3):
            ww=np.exp(rng.normal(scale=1.5,size=len(w)))
            yy=rng.normal(size=y.shape)
            p.update_weights(ww)
            z,info=p.project(yy,1e-10,6000)
            assert_allclose(z,oracle(g,yy,ww),atol=2e-8)
    finally:p.close()


def test_default_deferral_uses_real_second_call(monkeypatch):
    g,y,w=draw(n=80)
    with BudgetedAbsorber(g,weights=w,expected_calls=8) as s:
        monkeypatch.setattr(s,'_worker',lambda *a:pytest.fail('no first-call speculation'))
        s.residualize(y)
        assert s._calls==1 and s._tune_pending
        assert s.diagnostics['plan']['reason']=='deferred_until_actual_reuse'
        assert s.diagnostics['plan']['optional_seconds']==0


def test_second_call_still_obeys_finite_horizon(monkeypatch):
    g,y,w=draw(n=80)
    with BudgetedAbsorber(g,weights=w,expected_calls=4) as s:
        monkeypatch.setattr(s,'_worker',lambda *a:pytest.fail('remaining horizon too short'))
        s.residualize(y);s.residualize(y)
        assert s.diagnostics['plan']['reason']=='short_horizon_no_calibration'


def test_numeric_cap_retry_keeps_one_overall_deadline(monkeypatch):
    g,y,w=draw(n=80,k=2)
    with BudgetedAbsorber(g,weights=w,expected_calls=20,max_candidates=1) as s:
        oldout,info,_=s._solve_incumbent(y,1e-10,6000)
        candidate=make_fast_fake(s,y,w)
        base=candidate.project
        deadlines=[]
        def project(data,tol,cap,**kw):
            deadlines.append(kw.get('deadline'))
            if cap==32:raise PlanningError('convergence not achieved in 32 iterations')
            return base(data,tol,cap,**kw)
        candidate.project=project
        monkeypatch.setattr(s,'_worker',lambda *a:candidate)
        s._explore_routes(y,oldout,info,2.,1e-10,6000)
        assert s.backend=='fereg'
        assert len(set(deadlines))==1
        assert [x['status'] for x in s._plan['trials'][0]['pilot_stages']]==['unranked','certified']
