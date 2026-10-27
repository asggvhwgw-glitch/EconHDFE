from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
import numpy as np
import pytest
from scipy.sparse.linalg import lsmr, aslinearoperator
from feselection.monitored_lsmr import monitored_lsmr
from feselection.pilot_policy import PilotPolicy, PilotController
from feselection import FESelector,SelectionConfig
from test_selector import tiny,ref


@pytest.mark.parametrize('shape',[(60,20),(20,60),(40,40)])
@pytest.mark.parametrize('rank_loss',[False,True])
def test_observer_preserves_scipy_recurrence_exactly(shape,rank_loss):
    rng=np.random.default_rng(701);A=rng.normal(size=shape);b=rng.normal(size=shape[0])
    if rank_loss:A[:,-1]=A[:,0]+A[:,1]
    calls=[]
    out=monitored_lsmr(A,b,atol=1e-13,btol=1e-13,maxiter=300,
                      monitor=lambda x:calls.append(x),check_every=5)
    expected=lsmr(A,b,atol=1e-13,btol=1e-13,maxiter=300)
    np.testing.assert_array_equal(out[0],expected[0])
    np.testing.assert_array_equal(out[1:],expected[1:])
    assert calls and calls[-1]['stop_code']==out[1]


@pytest.mark.parametrize('damp,x0',[(.1,None),(0.,np.ones(25)),(2.,np.zeros(25))])
def test_preserves_optional_upstream_arguments(damp,x0):
    rng=np.random.default_rng(22);A=rng.normal(size=(80,25));b=rng.normal(size=80)
    kw=dict(atol=1e-13,btol=1e-13,maxiter=120,damp=damp,x0=x0)
    got=monitored_lsmr(A,b,monitor=lambda d:True,**kw);want=lsmr(A,b,**kw)
    np.testing.assert_array_equal(got[0],want[0]);assert got[1:]==want[1:]


def test_zero_rhs_and_orthogonal_rhs():
    A=np.vstack([np.eye(6),np.zeros((3,6))])
    for b in (np.zeros(9),np.r_[np.zeros(6),[1.,2.,3.]]):
        got=monitored_lsmr(A,b,monitor=lambda d:pytest.fail('nonexistent iteration'))
        assert got[1]==0 and got[2]==0
        np.testing.assert_array_equal(got[0],np.zeros(6))


def test_controller_stop_not_convergence_and_same_partial_iterate():
    rng=np.random.default_rng(20);A=rng.normal(size=(90,40));b=rng.normal(size=90)
    got=monitored_lsmr(A,b,atol=1e-14,btol=1e-14,maxiter=200,
        monitor=lambda s:s['iteration']<8,check_every=4)
    want=lsmr(A,b,atol=1e-14,btol=1e-14,maxiter=8)
    assert got[1]==8 and got[2]==8
    np.testing.assert_array_equal(got[0],want[0])


def state(it,elapsed,gap,stop=0):
    return dict(iteration=it,elapsed=elapsed,normal_backward_error=gap*1e-13,
        compatible_relative_residual=1.,compatible_tolerance=1e-13,atol=1e-13,
        normar=gap*1e-13,normr=1.,norma=1.,condition=1.,stop_code=stop)


def test_progress_can_continue_past_old_24_limit_without_restart():
    c=PilotController(PilotPolicy(),baseline_seconds=3.,remaining_rhs=2,initial_steps=24)
    assert c(state(8,.10,1e9));assert c(state(16,.20,1e6));assert c(state(24,.30,1e3))
    assert c.records[-1]['reason']=='progress_and_cost_support_continuation'
    assert c(state(32,.40,.5,2))
    assert c.records[-1]['reason']=='numerical_stop'


def test_stagnation_stops_instead_of_blindly_doubling_limit():
    c=PilotController(PilotPolicy(),baseline_seconds=3.,remaining_rhs=12,initial_steps=24)
    for it in (8,16):assert c(state(it,it*.01,1e5))
    assert not c(state(24,.24,1e5))
    assert c.records[-1]['reason']=='insufficient_progress'


def test_time_budget_applies_before_initial_grace_and_checks_not_preemptive():
    c=PilotController(PilotPolicy(),baseline_seconds=.1,remaining_rhs=12,initial_steps=24)
    assert not c(state(8,.11,1e4))
    assert c.records[-1]['reason']=='pilot_time_budget'


def test_slow_forecast_stops_even_with_progress():
    c=PilotController(PilotPolicy(),baseline_seconds=1.,remaining_rhs=2,initial_steps=24)
    assert c(state(8,.1,1e10));assert c(state(16,.2,1e9))
    assert not c(state(24,.3,1e8))
    assert c.records[-1]['reason']=='forecast_does_not_pay'


def test_numerical_convergence_takes_precedence_over_cost_stop():
    c=PilotController(PilotPolicy(),baseline_seconds=.1,remaining_rhs=2,initial_steps=24)
    assert c(state(8,5.,.5,2))
    assert c.records[-1]['convergence']


@pytest.mark.parametrize('key,value',[
    ('enabled',1),('checkpoint_steps',0),('checkpoint_steps',True),
    ('forecast_safety',.5),('seconds_cap',0),('seconds_cap',float('inf')),
    ('maximum_native_fraction',0),('maximum_native_fraction',1.1),
    ('minimum_contraction',1),('history_windows',0),
    ('exploratory_time_fraction',-1),('exploratory_time_fraction',1.01)])
def test_controller_policy_validation(key,value):
    with pytest.raises(ValueError):PilotPolicy(**{key:value})


def test_incomplete_controlled_projection_finished_on_original_target():
    from feselection.pilot_policy import solve_profiled_pilot
    from scalable_hyperfe import ProfiledSystem
    from budget_preconditioner import build_star,ComponentRight
    c,w,Y=tiny(998);system=ProfiledSystem.build(c,w);T,scale,_=build_star(system)
    base=ComponentRight(T,scale,tau=.999,max_block=256,eig_cut=0)
    ctrl=PilotController(PilotPolicy(checkpoint_steps=1),baseline_seconds=1e-12,
                         remaining_rhs=2,initial_steps=24)
    partial,info=solve_profiled_pilot(system,Y[:,0],base,tol=1e-13,maxiter=500,controller=ctrl)
    assert info['status']==8 and info['state_restarts']==0
    final=FESelector(SelectionConfig(mode='native')).prepare(c,w).residualize(partial).residuals
    np.testing.assert_allclose(final,ref(c,w,Y[:,0,None])[:,0],atol=2e-10)


def test_progress_records_cannot_mutate_solution():
    rng=np.random.default_rng(4);A=rng.normal(size=(45,30));b=rng.normal(size=45)
    def observer(s):
        assert 'x' not in s
        s['iteration']=-9
    got=monitored_lsmr(A,b,monitor=observer,atol=1e-13,btol=1e-13)
    want=lsmr(A,b,atol=1e-13,btol=1e-13)
    np.testing.assert_array_equal(got[0],want[0]);assert got[1:]==want[1:]


def test_bounded_extra_window_when_progress_forecast_is_too_pessimistic():
    c=PilotController(PilotPolicy(),baseline_seconds=4.,remaining_rhs=2,initial_steps=24)
    assert c(state(8,.2,4e8));assert c(state(16,.4,7e7));assert c(state(24,.6,2e7))
    assert c.records[-1]['reason']=='bounded_progress_extension_forecast_uncertain'
    assert c(state(32,.8,5e2))
    assert c(state(36,.9,.5,2))
    assert c.records[-1]['reason']=='numerical_stop'


def test_progress_only_extension_cannot_exceed_exploratory_cap():
    c=PilotController(PilotPolicy(),baseline_seconds=1.,remaining_rhs=2,initial_steps=24)
    assert c(state(8,.12,1e14));assert c(state(16,.24,1e13))
    assert not c(state(24,.36,1e12))
    assert c.records[-1]['reason']=='forecast_does_not_pay'


def test_zero_grace_is_supported_for_ablation():
    c=PilotController(PilotPolicy(exploratory_time_fraction=0.),baseline_seconds=4.,remaining_rhs=2,initial_steps=24)
    assert c(state(8,.2,4e8));assert c(state(16,.4,7e7))
    assert not c(state(24,.6,2e7))


def test_monitor_adds_no_extra_matrix_products_or_restarts():
    from scipy.sparse.linalg import LinearOperator
    rng=np.random.default_rng(58);a=rng.normal(size=(70,35));b=rng.normal(size=70)
    counts=[0,0]
    def mv(x):counts[0]+=1;return a@x
    def rmv(x):counts[1]+=1;return a.T@x
    op=LinearOperator(a.shape,matvec=mv,rmatvec=rmv,dtype=float)
    checks=[]
    got=monitored_lsmr(op,b,atol=1e-13,btol=1e-13,maxiter=120,
        monitor=lambda s:checks.append(s),check_every=4)
    before=counts.copy();counts[:]=[0,0]
    expected=lsmr(op,b,atol=1e-13,btol=1e-13,maxiter=120)
    assert counts==before
    assert len(checks)>1
    np.testing.assert_array_equal(got[0],expected[0])


def test_nonfinite_observer_progress_cannot_be_accepted_as_convergence():
    c=PilotController(PilotPolicy(),baseline_seconds=3.,remaining_rhs=2,initial_steps=24)
    s=state(8,.1,1e8);s['normar']=float('nan')
    assert not c(s)
    assert c.records[-1]['reason']=='nonfinite_progress'
