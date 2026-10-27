from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from feselection import FESelector,SelectionConfig,ResourceBudget
from feselection.selector import _normal
from feselection.auditing import audit_columns,column_scales
from feselection.cost_policy import CostPolicy
from test_selector import tiny,ref

@pytest.mark.parametrize('scale',[1.,1e-200,1e200])
def test_fused_audit_matches_original_metric(scale):
    c,w,y=tiny(123);r=ref(c,w,y);r[0,0]+=.001;y*=scale;r*=scale
    p=FESelector().prepare(c,w)
    vals,feas=audit_columns(p.codes,p.w,y,r,p.offsets,p.masses)
    expected=[_normal(p.codes,p.w,y[:,j],r[:,j]) for j in range(3)]
    np.testing.assert_allclose(vals,expected,rtol=5e-13,atol=3e-16)
    assert vals[0]>1e-8


def test_fit_audit_detects_corruption_even_when_normal_passes():
    import within
    c,w,y=tiny(839);p=FESelector().prepare(c,w)
    scales=column_scales(y);solver=p._native()
    sol=solver.solve_batch(np.asfortranarray(y/scales),options=within.LsmrOptions(tol=1e-13))
    r=np.asarray(sol.demeaned)*scales;x=np.array(sol.x,copy=True)
    n,f=audit_columns(p.codes,p.w,y,r,p.offsets,p.masses,coef=x,scales=scales)
    assert max(n)<1e-10 and max(f)<1e-10
    x[0,1]+=.1
    n,f=audit_columns(p.codes,p.w,y,r,p.offsets,p.masses,coef=x,scales=scales)
    assert max(n)<1e-10 and f[1]>1e-4


def test_native_mode_batches_and_audits_once():
    c,w,y=tiny(124);p=FESelector(SelectionConfig(mode='native')).prepare(c,w)
    result=p.residualize(y)
    calls=[e for e in result.diagnostics['events'] if e['stage']=='native_batch']
    assert len(calls)==1 and calls[0]['columns']==3 and calls[0]['calls']==1
    np.testing.assert_allclose(result.residuals,ref(c,w,y),atol=2e-10)
    np.testing.assert_equal(result.diagnostics['normal_residuals'],calls[0]['normal'])


def test_cached_native_uses_batch_on_new_rhs():
    c,w,y=tiny(319);p=FESelector(SelectionConfig(mode='native')).prepare(c,w)
    p.residualize(y);y=y.copy();y[:,0]+=y[:,1]
    r=p.residualize(y)
    assert any(e['stage']=='cache_hit' for e in r.diagnostics['events'])
    assert len([e for e in r.diagnostics['events'] if e['stage']=='native_batch'])==1
    np.testing.assert_allclose(r.residuals,ref(c,w,y),atol=1e-10)


def test_inadmissible_dimension_bypasses_pilot_and_build(monkeypatch):
    c,w,y=tiny(44);p=FESelector(SelectionConfig(mode='auto',budget=ResourceBudget(max_reduced_dimension=1))).prepare(c,w)
    def forbidden(*a,**kw):raise AssertionError('costly build was reached')
    monkeypatch.setattr(p,'_build_candidate',forbidden)
    monkeypatch.setattr(p,'_native_solve',forbidden)
    r=p.residualize(y)
    assert r.diagnostics['events'][0]['reason']=='no_admitted_elimination'
    np.testing.assert_allclose(r.residuals,ref(c,w,y),atol=1e-10)


def test_iteration_work_gate_is_conservative_and_width_aware():
    p=CostPolicy()
    assert not p.exploration_admitted(49,2)
    assert not p.exploration_admitted(51,2)
    assert p.exploration_admitted(172,2)
    assert p.exploration_admitted(49,11)
    for bad in [-1,True,.5]:
        with pytest.raises(ValueError):CostPolicy(setup_iteration_equivalents=bad)


def test_immutable_F_snapshot_and_mass_reuse():
    c,w,y=tiny(125);p=FESelector(SelectionConfig(mode='native')).prepare(c,w)
    assert p.codes.flags.f_contiguous and not p.codes.flags.writeable
    assert np.shares_memory(p.codes,p.codes.view(np.uint32))
    masses=p.masses.copy();offsets=p.offsets.copy()
    p.residualize(y);p.residualize(y*2)
    np.testing.assert_array_equal(p.masses,masses);np.testing.assert_array_equal(p.offsets,offsets)
    assert not p.masses.flags.writeable


def test_complex_weights_are_rejected_not_silently_cast():
    c,w,y=tiny()
    with pytest.raises(ValueError,match='real'):FESelector().prepare(c,w.astype(complex)+1j)

@pytest.mark.parametrize('cols',[1,2,12])
def test_native_batch_widths_units_and_shape(cols):
    c,w,y=tiny(271);rng=np.random.default_rng(82);y=rng.normal(size=(len(c),cols))
    y[:,0]*=1e-200
    if cols>1:y[:,1]*=1e200
    r=FESelector(SelectionConfig(mode='native')).prepare(c,w).residualize(y)
    scales=column_scales(y)
    np.testing.assert_allclose(r.residuals/scales,ref(c,w,y/scales),atol=2e-10)
    assert r.residuals.shape==y.shape


def test_nan_result_cannot_hide_behind_completed_audits(monkeypatch):
    c,w,y=tiny(339);p=FESelector(SelectionConfig(mode='native')).prepare(c,w)
    from feselection.auditing import audit_columns
    bad=y.copy();bad[0,1]=np.nan
    with pytest.raises(FloatingPointError):audit_columns(p.codes,p.w,y,bad,p.offsets,p.masses)


def test_completed_root_is_not_discarded_after_soft_deadline(monkeypatch):
    import feselection.selector as module
    c,w,y=tiny(745);p=FESelector(SelectionConfig(mode='candidate',eig_cut=.5)).prepare(c,w)
    now=[0.];real=module.GuardedCoarseRight
    monkeypatch.setattr(module,'perf_counter',lambda:now[0])
    def delayed(*args,**kwargs):
        root=real(*args,**kwargs);now[0]=10.;return root
    monkeypatch.setattr(module,'GuardedCoarseRight',delayed)
    events=[];p._build_candidate(y,1.,events)
    assert p.root is not None and getattr(p.root,'k',0)>0
    assert any(e['stage']=='ready_candidate_over_soft_budget' and e['available']=='local_coarse' for e in events)
    r,_,ok=p._local_column(y[:,0],[],2000)
    assert ok
    np.testing.assert_allclose(r,ref(c,w,y)[:,0],atol=2e-10)


def test_ready_local_root_skips_further_coarse_build_on_budget(monkeypatch):
    import feselection.selector as module
    c,w,y=tiny(746);p=FESelector(SelectionConfig(mode='candidate',eig_cut=.5)).prepare(c,w)
    now=[0.];real=module.ComponentRight
    monkeypatch.setattr(module,'perf_counter',lambda:now[0])
    def delayed(*args,**kwargs):
        root=real(*args,**kwargs);now[0]=10.;return root
    def forbidden(*args,**kwargs):raise AssertionError('must skip further construction')
    monkeypatch.setattr(module,'ComponentRight',delayed)
    monkeypatch.setattr(module,'GuardedCoarseRight',forbidden)
    events=[];p._build_candidate(y,1.,events)
    assert p.root is not None and not hasattr(p.root,'k')
    assert any(e['stage']=='ready_candidate_over_soft_budget' and e['available']=='local_only' for e in events)
    r,_,ok=p._local_column(y[:,0],[],2000)
    assert ok
    np.testing.assert_allclose(r,ref(c,w,y)[:,0],atol=2e-10)
