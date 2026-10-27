"""Structural eligibility, transactional fallback and nonlinear integration."""
from dataclasses import replace
import importlib

import numpy as np
import pytest

from econhdfe import ExecutionConfig, PPMLConfig, IVPPMLConfig, ppmlhdfe, ivppmlhdfe
from econhdfe.errors import ConvergenceError, SpecificationError
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.plan import FEPlan
from econhdfe.hdfe.weighted_projection import make_weighted_projector
from econhdfe.hdfe.adaptive_weighted import AdaptiveWeightedFEProjector

routing = importlib.import_module('econhdfe.hdfe.adaptive_weighted')


@pytest.fixture
def small_policy(monkeypatch):
    # Eligibility semantics are tested at small scale; real benchmarks keep
    # the production cutoff and run complete million-row public fits.
    monkeypatch.setattr(routing, '_MIN_ROWS', 100)


def data(n=2400, seed=781):
    rng = np.random.default_rng(seed)
    groups = FEPlan.from_arrays([rng.integers(L, size=n) for L in (30, 12, 7)]).groups
    return groups, rng.normal(size=(n, 3)), np.exp(.4*rng.normal(size=n))


def projector(groups, **kw):
    return make_weighted_projector(groups, engine='adaptive', absorb_threads=1,
                                   projection_memory_budget_mb=192, **kw)


def test_small_and_two_way_do_not_load_native(monkeypatch):
    def forbidden():
        raise AssertionError('native availability check was unnecessary')
    monkeypatch.setattr(routing, '_native_available', forbidden)
    groups, A, w = data()
    p = projector(groups)
    p.residualize(A, w, tol=1e-9)
    assert p.resource_info['selection']['reason'] == 'outside_validated_size_or_dimension'
    monkeypatch.setattr(routing, '_MIN_ROWS', 100)
    p2 = projector(groups[:2])
    p2.residualize(A, w, tol=1e-9)
    assert p2.resource_info['backend'] == 'twoway'


def test_missing_native_is_a_supported_adaptive_fallback(small_policy, monkeypatch):
    monkeypatch.setattr(routing, '_native_available', lambda: False)
    groups, A, w = data()
    p = projector(groups)
    out, info = p.residualize(A, w, tol=1e-10, return_info=True)
    assert info.converged and np.isfinite(out).all()
    assert p.resource_info['selection']['reason'] == 'native_unavailable_or_incompatible'
    assert p.resource_info['backend'] == 'generic'


@pytest.mark.parametrize('case', ['repeated', 'crossed'])
def test_exact_repetition_gate_is_row_order_invariant(small_policy, monkeypatch, case):
    monkeypatch.setattr(routing, '_native_available', lambda: True)
    rng = np.random.default_rng(932)
    n = 4000
    if case == 'repeated':
        gs = [np.repeat(rng.integers(L, size=n//8), 8) for L in (150, 120, 80)]
    else:
        gs = [rng.integers(L, size=n) for L in (150, 120, 80)]
    groups = FEPlan.from_arrays(gs).groups
    outcomes = []
    for order in (np.arange(n), rng.permutation(n)):
        p = projector([g[order] for g in groups]); p._select()
        outcomes.append(p.resource_info['selection'])
    assert outcomes[0]['candidate'] == outcomes[1]['candidate']
    if case == 'repeated':
        assert outcomes[0]['candidate'] == 'compressed'
        assert outcomes[0]['exact_cells'] == outcomes[1]['exact_cells'] <= n//8
    else:
        assert outcomes[0]['reason'] == outcomes[1]['reason'] == 'insufficient_repetition'


def test_key_overflow_declines_without_integer_wrap(small_policy, monkeypatch):
    monkeypatch.setattr(routing, '_native_available', lambda: True)
    g = np.tile(np.array([0, np.iinfo(np.int32).max], dtype=np.int32), 100)
    p = projector((g, g, g)); p._select()
    assert p.resource_info['selection']['reason'] == 'tuple_key_overflow'


def test_probe_memory_gate_precedes_allocation(small_policy, monkeypatch):
    monkeypatch.setattr(routing, '_native_available', lambda: True)
    groups = tuple(np.arange(4000, dtype=np.int32) % L for L in (150, 120, 80))
    p = projector(groups)
    monkeypatch.setattr(p, '_capacity', lambda: 100)
    p._select()
    assert p.resource_info['selection']['reason'] == 'tuple_probe_budget'


def test_native_memory_gate_precedes_construction(small_policy, monkeypatch):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    def forbidden(*args, **kw):
        raise AssertionError('must not construct inadmissible native engine')
    monkeypatch.setattr(WithinFEAbsorber, '__init__', forbidden)
    groups, A, w = data()
    p = projector(groups)
    monkeypatch.setattr(p, '_capacity', lambda: 100)
    out = p.residualize(A, w, tol=1e-10)
    assert np.isfinite(out).all()
    assert p.resource_info['selection']['fallback_events'] == ('native_workspace_admission',)


@pytest.mark.parametrize('copy', [True, False])
def test_failed_native_trial_rolls_back_and_is_not_retried(small_policy, monkeypatch, copy):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    from econhdfe.hdfe.absorber import AbsorbInfo
    groups, A, w = data()
    reference = HDFEAbsorber(groups, weights=w, method='map', acceleration='cg', tol=1e-10,
                            absorb_threads=1).residualize(A)
    original = A.copy()
    calls = []
    def fail(self, values, **kw):
        calls.append(self.max_iter)
        values[:] = 123456789.0  # Deliberately destroy the native input buffer.
        return values, AbsorbInfo(False, self.max_iter, 1., 'injected')
    monkeypatch.setattr(WithinFEAbsorber, 'residualize', fail)
    p = projector(groups, max_iter=400)
    result, info = p.residualize(A, w, tol=1e-10, copy=copy, return_info=True)
    np.testing.assert_allclose(result, reference, atol=1e-8, rtol=1e-8)
    if copy:
        np.testing.assert_array_equal(A, original)
    else:
        assert result is A
    assert info.converged and info.iterations <= 400
    assert calls == [100]
    assert p._absorber.max_iter == 300
    p.residualize(original, w*1.1, tol=1e-10)
    assert calls == [100]  # Sticky demotion; new calls regain full fallback budget.
    assert p._absorber.max_iter == 400
    assert p.resource_info['selection']['fallback_events'] == ('native_certificate_failed',)


def test_failed_fallback_still_raises(small_policy, monkeypatch):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    from econhdfe.hdfe.absorber import AbsorbInfo
    def fail(self, values, **kw):
        return values, AbsorbInfo(False, self.max_iter, 1., 'injected')
    monkeypatch.setattr(WithinFEAbsorber, 'residualize', fail)
    monkeypatch.setattr(HDFEAbsorber, 'residualize', fail)
    groups, A, w = data()
    p = projector(groups, max_iter=20)
    with pytest.raises(ConvergenceError) as exc:
        p.residualize(A, w, tol=1e-10, return_info=False)
    assert exc.value.code == 'convergence.weighted_fe'
    assert p.max_iter == 20


def test_unexpected_programming_exception_is_not_hidden(small_policy, monkeypatch):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    def fail(*args, **kw):
        raise TypeError('programming bug')
    monkeypatch.setattr(WithinFEAbsorber, 'residualize', fail)
    groups, A, w = data()
    with pytest.raises(TypeError, match='programming bug'):
        projector(groups).residualize(A, w, tol=1e-10)


def test_schur_rejection_is_safe_and_sticky(small_policy):
    pytest.importorskip('_econhdfe_within')
    groups, A, w = data()
    # Exact redundancy: the narrow Schur factor has extra nullity.
    p = projector((groups[0], groups[1], groups[1]))
    out, info = p.residualize(A, w, tol=1e-10, return_info=True)
    assert info.converged
    assert p.resource_info['selection']['fallback_events'] == ('schur_numeric_gate',)
    ref = HDFEAbsorber((groups[0], groups[1]), weights=w, tol=1e-10, absorb_threads=1).residualize(A)
    np.testing.assert_allclose(out, ref, atol=1e-8, rtol=1e-8)


def test_zero_support_switches_to_lsmr_then_back_to_map(small_policy):
    pytest.importorskip('_econhdfe_within')
    groups, A, w = data()
    p = projector(groups)
    p.residualize(A, w, tol=1e-10)
    assert p.resource_info['backend'] == 'within'
    w[groups[0] == 0] = 0
    p.residualize(A, w, tol=1e-9)
    assert p._absorber.method == 'lsmr'
    w[:] = 1
    p.residualize(A, w, tol=1e-9)
    assert p._absorber.method == 'map'
    assert p.resource_info['selection']['native_disabled']


@pytest.mark.parametrize('which', ['weight', 'rhs', 'tol', 'readonly'])
def test_invalid_input_does_not_trigger_fallback(small_policy, which):
    groups, A, w = data()
    tol = 1e-10
    if which == 'weight': w[0] = -1
    if which == 'rhs': A[0,0] = np.inf
    if which == 'tol': tol = np.nan
    if which == 'readonly': A.flags.writeable = False
    p = projector(groups)
    with pytest.raises(SpecificationError):
        p.residualize(A, w, tol=tol, copy=False)
    assert not p.resource_info['selection']['fallback_events']


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
@pytest.mark.parametrize('fast_partial', [True, False])
def test_public_estimator_native_dynamic_weights_and_parity(small_policy, model, fast_partial):
    pytest.importorskip('_econhdfe_within')
    groups, A, w = data()
    c, z, noise = A.T
    e = .8*z + .2*c + .3*noise
    off = .1*np.sin(np.arange(len(c)))
    rng = np.random.default_rng(37)
    y = rng.poisson(np.exp(.5+.12*c+.18*e+off)).astype(float)
    common = dict(y=y, absorb=groups, weights=w, offset=off,
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=192))
    def fit(engine):
        cfg = dict(engine=engine, fast_partial=fast_partial, separation=('fe',),
                   tolerance=1e-9, target_inner_tol=1e-10)
        if model == 'ppml':
            return ppmlhdfe(x=np.column_stack([c,e]), config=PPMLConfig(**cfg), **common)
        return ivppmlhdfe(exog=c,endog=e,instruments=z,weight_type='pweight',
                          config=IVPPMLConfig(**cfg), **common)
    b, r = fit('optimized'), fit('adaptive')
    assert r.converged and r.diagnostics['projection_resources']['backend'] == 'within'
    assert r.diagnostics['projection_resources']['native']['weight_generation'] > 1
    np.testing.assert_array_equal(r.sample_mask, b.sample_mask)
    np.testing.assert_allclose(r.params, b.params, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(r.mu, b.mu, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(r.vcov, b.vcov, atol=1e-9, rtol=1e-6)


def test_config_defaults_and_dataclass_fields_remain_unchanged():
    assert PPMLConfig().engine == IVPPMLConfig().engine == 'replica'
    for cls in (PPMLConfig, IVPPMLConfig):
        cls(engine='adaptive').validate()


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
@pytest.mark.parametrize('phase', ['rank', 'irls', 'final'])
def test_public_fallback_from_unmodified_input_in_each_phase(small_policy, monkeypatch, model, phase):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    module = importlib.import_module('econhdfe.models.' + ('ppml' if model == 'ppml' else 'ppml_iv') + '.estimator')
    fn = 'fit_irls' if model == 'ppml' else 'fit_iv_irls'
    original_irls = getattr(module, fn)
    current_phase = ['rank']
    def track(*args, **kwargs):
        current_phase[0] = 'irls'
        state = original_irls(*args, **kwargs)
        current_phase[0] = 'final'
        return state
    monkeypatch.setattr(module, fn, track)
    original = WithinFEAbsorber.residualize
    def corrupt(self, values, **kwargs):
        out, info = original(self, values, **kwargs)
        if current_phase[0] == phase:
            out[:] = 9876543.
            info.converged = False
        return out, info
    monkeypatch.setattr(WithinFEAbsorber, 'residualize', corrupt)
    groups, A, w = data()
    c,z,noise = A.T
    e = z+.2*noise
    y = np.random.default_rng(84).poisson(np.exp(.3+.1*c+.15*e)).astype(float)
    common = dict(y=y,absorb=groups,weights=w,execution_config=ExecutionConfig(threads=1,memory_budget_mb=192))
    def fit(engine):
        cfg = dict(engine=engine,separation=(),tolerance=1e-9,target_inner_tol=1e-10)
        current_phase[0] = 'rank'
        if model == 'ppml':
            return ppmlhdfe(x=np.column_stack([c,e]),config=PPMLConfig(**cfg),**common)
        return ivppmlhdfe(exog=c,endog=e,instruments=z,weight_type='pweight',config=IVPPMLConfig(**cfg),**common)
    baseline = fit('optimized')
    result = fit('adaptive')
    assert result.converged
    assert result.diagnostics['projection_resources']['selection']['fallback_events'] == ('native_certificate_failed',)
    np.testing.assert_allclose(result.params,baseline.params,atol=2e-7,rtol=2e-7)
    np.testing.assert_allclose(result.mu,baseline.mu,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
def test_separation_uses_optimized_operators_not_adaptive_trials(small_policy, monkeypatch, model):
    separation = importlib.import_module('econhdfe.models.ppml.separation')
    original = separation.get_projector
    seen = []
    def observe(*args, **kwargs):
        seen.append((kwargs['engine'],kwargs['method']))
        return original(*args, **kwargs)
    monkeypatch.setattr(separation,'get_projector',observe)
    groups,A,w = data()
    c,z,e = A.T
    y = np.random.default_rng(95).poisson(np.exp(.5+.1*c+.1*e)).astype(float)
    if model == 'ivppml': e = z+.2*e
    common=dict(y=y,absorb=groups,weights=w,execution_config=ExecutionConfig(threads=1,memory_budget_mb=192))
    if model == 'ppml':
        r=ppmlhdfe(x=np.column_stack([c,e]),config=PPMLConfig(engine='adaptive'),**common)
    else:
        r=ivppmlhdfe(exog=c,endog=e,instruments=z,weight_type='pweight',config=IVPPMLConfig(engine='adaptive'),**common)
    assert r.converged and seen
    assert all(engine=='optimized' for engine,_ in seen)
    assert r.diagnostics['separation_solvers']['simplex']=='optimized:map'


@pytest.mark.parametrize('trim', [False, True])
def test_block_projectors_select_per_component_and_rebuild_after_trim(small_policy, trim):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
    from econhdfe.hdfe.block_projection import BlockWeightedFEProjector
    rng=np.random.default_rng(712)
    n=2400
    row_block=np.arange(n)%2
    groups=[row_block*L+rng.integers(L,size=n) for L in (30,12,7)]
    plan=FEPlan.from_arrays(groups)
    values=rng.normal(size=(n,2))
    design=BlockDesign(n,2,tuple(DenseDesignBlock(b,np.flatnonzero(row_block==b),np.arange(2),values[row_block==b]) for b in (0,1)))
    weights=np.exp(.2*rng.normal(size=n))
    if trim:
        keep=np.arange(n)%5!=0
        design=design.subset_rows(keep);plan=plan.subset(keep);weights=weights[keep]
    p=BlockWeightedFEProjector.from_design(design,plan,engine='adaptive',absorb_threads=1,projection_memory_budget_mb=192)
    baseline=BlockWeightedFEProjector.from_design(design,plan,engine='optimized',absorb_threads=1,projection_memory_budget_mb=192)
    response=np.arange(design.nobs,dtype=float)/design.nobs
    result=p.residualize_response_design(response,design,weights,tol=1e-10)
    ref=baseline.residualize_response_design(response,design,weights,tol=1e-10)
    np.testing.assert_allclose(result.response,ref.response,atol=1e-8,rtol=1e-8)
    for a,b in zip(result.design.blocks,ref.design.blocks):
        np.testing.assert_allclose(a.values,b.values,atol=1e-8,rtol=1e-8)
    assert all(b['projector']['backend']=='within' for b in p.resource_info['blocks'])


def test_reusable_model_retains_one_selection_and_invalidates_mutated_topology(small_policy):
    pytest.importorskip('_econhdfe_within')
    import pandas as pd
    from econhdfe import PPMLHDFE
    groups,A,w=data();c,e,z=A.T
    df=pd.DataFrame(dict(c=c,e=e,w=w,y=np.random.default_rng(17).poisson(np.exp(.2+.1*c+.15*e)),**{f'g{i}':g for i,g in enumerate(groups)}))
    m=PPMLHDFE(df,absorb=('g0','g1','g2'),config=PPMLConfig(engine='adaptive',separation=()),execution_config=ExecutionConfig(threads=1,memory_budget_mb=192,cache_validation='signature'))
    first=m.fit('y',('c','e'),weights='w')
    old=[v for k,v in m.context.cache.items() if k.startswith('ppml.projector:')]
    assert old and isinstance(old[0],AdaptiveWeightedFEProjector)
    probe=old[0]._probe_seconds
    second=m.fit('y',('c','e'),weights='w')
    assert old[0]._probe_seconds==probe
    np.testing.assert_allclose(first.params,second.params,atol=1e-9,rtol=1e-9)
    df.loc[:20,'g0']=df['g0'].max()+1
    m.fit('y',('c','e'),weights='w')
    new=[v for k,v in m.context.cache.items() if k.startswith('ppml.projector:')]
    assert all(v is not old[0] for v in new)
