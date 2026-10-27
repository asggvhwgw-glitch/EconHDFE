"""Regression tests for selectable weighted FE backends in Poisson-family IRLS."""
from dataclasses import replace

import numpy as np
import pytest

from econhdfe import (
    ExecutionConfig, PPMLConfig, IVPPMLConfig, ppmlhdfe, ivppmlhdfe,
)
from econhdfe.errors import SpecificationError
from econhdfe.hdfe.weighted_projection import WeightedFEProjector


def _fixture(n=1800, seed=26027):
    rng = np.random.default_rng(seed)
    groups = [rng.integers(0, L, size=n) for L in (45, 25, 10)]
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    e = 0.75 * z + 0.4 * c + rng.normal(scale=0.6, size=n)
    off = 0.08 * np.sin(np.arange(n) * 0.09)
    mu = np.exp(0.2 + 0.11 * e + 0.07 * c + off)
    y = rng.poisson(mu).astype(float)
    weights = np.exp(rng.normal(scale=0.3, size=n))
    return y, c, e, z, off, groups, weights


@pytest.mark.parametrize("config_type", [PPMLConfig, IVPPMLConfig])
def test_invalid_engine_rejected(config_type):
    with pytest.raises(SpecificationError, match="engine"):
        config_type(engine="unknown").validate()


@pytest.mark.parametrize("kind", ["ppml", "ivppml"])
def test_default_map_selector_unchanged(kind):
    y, c, e, z, off, groups, weights = _fixture(n=550)
    execution = ExecutionConfig(threads=1, memory_budget_mb=192)
    if kind == "ppml":
        result = ppmlhdfe(
            y=y, x=np.column_stack([e, c]), absorb=groups, weights=weights,
            offset=off, config=PPMLConfig(separation=()), execution_config=execution,
        )
    else:
        result = ivppmlhdfe(
            y=y, exog=c, endog=e, instruments=z, absorb=groups,
            offset=off, weights=weights, weight_type="pweight",
            config=IVPPMLConfig(separation=()), execution_config=execution,
        )
    assert result.converged
    resources = result.diagnostics["projection_resources"]
    assert resources["method"] == "map"


@pytest.mark.parametrize("kind", ["ppml", "ivppml"])
def test_native_within_matches_map_with_updated_irls_weights(kind, monkeypatch):
    pytest.importorskip("_econhdfe_within")
    from econhdfe.hdfe.within_backend import WithinFEAbsorber

    y, c, e, z, off, groups, weights = _fixture()
    execution = ExecutionConfig(threads=1, memory_budget_mb=192)
    updates = []
    original = WithinFEAbsorber.update_weights

    def capture(self, w, *, tol=None):
        updates.append(np.asarray(w, dtype=np.float64).copy())
        return original(self, w, tol=tol)

    monkeypatch.setattr(WithinFEAbsorber, "update_weights", capture)
    if kind == "ppml":
        cfg = PPMLConfig(
            engine="optimized", standardize=False, separation=(),
            tolerance=1e-8, target_inner_tol=1e-10, fast_partial=False,
        )
        def fit(config):
            return ppmlhdfe(
                y=y, x=np.column_stack([e, c]), absorb=groups, weights=weights,
                offset=off, config=config, execution_config=execution,
            )
    else:
        cfg = IVPPMLConfig(
            engine="optimized", separation=(), tolerance=1e-8,
            target_inner_tol=1e-10, max_iter=200,
        )
        def fit(config):
            return ivppmlhdfe(
                y=y, exog=c, endog=e, instruments=z, absorb=groups,
                offset=off, weights=weights, weight_type="pweight",
                config=config, execution_config=execution,
            )

    baseline = fit(cfg)
    candidate = fit(replace(cfg, engine="within"))
    assert baseline.converged and candidate.converged
    resources = candidate.diagnostics["projection_resources"]
    assert resources["method"] == "within" and resources["backend"] == "within"
    assert len(updates) >= 2
    assert any(not np.array_equal(updates[i], updates[i + 1]) for i in range(len(updates) - 1))
    np.testing.assert_allclose(candidate.params, baseline.params, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(candidate.mu, baseline.mu, atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(candidate.vcov, baseline.vcov, atol=7e-5, rtol=7e-5)


def test_native_selector_falls_back_on_zero_weight_support():
    g = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    projector = WeightedFEProjector((g,), engine="replica", method="within")
    assert projector._kind(np.array([1., 2., 0., 1., 1., 1.])) == "generic"
    assert projector._kind(np.ones(6)) == "within"


@pytest.mark.parametrize('kind', ['ppml', 'ivppml'])
def test_native_matches_independent_dense_poisson_moment_root(kind):
    """Exactly identified full-dummy score oracle, independent of FWL/IRLS."""
    pytest.importorskip('_econhdfe_within')
    from scipy.optimize import root
    rng = np.random.default_rng(260281)
    n = 1600
    groups = [rng.integers(L, size=n) for L in (12, 7, 4)]
    c = rng.normal(size=n)
    z = rng.normal(size=n)
    e = .9*z + .2*c + rng.normal(scale=.4, size=n)
    weights = np.exp(.5*rng.normal(size=n))
    off = .15*rng.normal(size=n)
    blocks = [np.eye(int(g.max())+1)[g] for g in groups]
    D = np.column_stack([blocks[0], *(b[:, 1:] for b in blocks[1:])])
    A = np.column_stack([c, e, D])
    Q = A if kind == 'ppml' else np.column_stack([c, z, D])
    truth = np.r_[.16, -.12, rng.normal(scale=.1, size=D.shape[1])]
    y = rng.poisson(np.exp(off + A @ truth)).astype(float)
    def fun(b):
        mu = np.exp(off + A @ b)
        return Q.T @ (weights * (y-mu)) / n
    def jac(b):
        mu = np.exp(off + A @ b)
        return -(Q.T @ ((weights*mu)[:, None]*A)) / n
    oracle = root(fun, np.zeros(A.shape[1]), jac=jac, tol=1e-11)
    assert oracle.success and np.max(np.abs(fun(oracle.x))) < 1e-10
    common = dict(y=y, absorb=groups, weights=weights, offset=off,
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=192))
    cfg = dict(engine='within', separation=(), tolerance=1e-10, target_inner_tol=1e-11)
    if kind == 'ppml':
        result = ppmlhdfe(x=np.column_stack([c,e]), config=PPMLConfig(**cfg), **common)
    else:
        result = ivppmlhdfe(exog=c, endog=e, instruments=z, weight_type='pweight',
                           config=IVPPMLConfig(**cfg), **common)
    np.testing.assert_allclose(result.params[:2], oracle.x[:2], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(result.mu, np.exp(off+A@oracle.x), atol=5e-8, rtol=5e-8)


@pytest.mark.parametrize('kind', ['ppml', 'ivppml'])
@pytest.mark.parametrize('phase', ['rank', 'irls', 'final'])
def test_failed_native_certificate_never_reaches_nonlinear_result(kind, phase, monkeypatch):
    pytest.importorskip('_econhdfe_within')
    import importlib
    from econhdfe.errors import ConvergenceError
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    module = importlib.import_module('econhdfe.models.' + ('ppml' if kind=='ppml' else 'ppml_iv') + '.estimator')
    state = ['rank']
    irls_name = 'fit_irls' if kind=='ppml' else 'fit_iv_irls'
    old_irls = getattr(module, irls_name)
    def track(*args, **kw):
        state[0] = 'irls'
        result = old_irls(*args, **kw)
        state[0] = 'final'
        return result
    monkeypatch.setattr(module, irls_name, track)
    old = WithinFEAbsorber.residualize
    def fail_certificate(self, *args, **kw):
        out = old(self, *args, **kw)
        if state[0] == phase and kw.get('return_info'):
            out[1].converged = False
        return out
    monkeypatch.setattr(WithinFEAbsorber, 'residualize', fail_certificate)
    y, c, e, z, off, groups, weights = _fixture()
    common = dict(y=y, absorb=groups, weights=weights, offset=off,
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=192))
    with pytest.raises(ConvergenceError) as err:
        if kind == 'ppml':
            ppmlhdfe(x=np.column_stack([c,e]), config=PPMLConfig(engine='within',separation=()), **common)
        else:
            ivppmlhdfe(exog=c, endog=e, instruments=z, weight_type='pweight',
                       config=IVPPMLConfig(engine='within',separation=()), **common)
    assert err.value.code == 'convergence.weighted_fe'
    assert err.value.details['backend'] == 'within'


@pytest.mark.parametrize('bad_error', [np.nan, np.inf])
def test_nonfinite_certificate_is_rejected_even_with_true_flag(bad_error, monkeypatch):
    pytest.importorskip('_econhdfe_within')
    from econhdfe.errors import ConvergenceError
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    old = WithinFEAbsorber.residualize
    def corrupt(self, *args, **kw):
        out = old(self, *args, **kw)
        out[1].max_update = bad_error
        return out
    monkeypatch.setattr(WithinFEAbsorber, 'residualize', corrupt)
    g = np.arange(100, dtype=np.int32) % 10
    p = WeightedFEProjector((g,), engine='within', method='within', absorb_threads=1, require_convergence=True)
    with pytest.raises(ConvergenceError, match='certificate'):
        p.residualize(np.arange(100.), np.ones(100), tol=1e-9)


def test_native_projector_zero_support_and_return_to_positive_weights():
    pytest.importorskip('_econhdfe_within')
    rng = np.random.default_rng(451)
    groups = (np.arange(180, dtype=np.int32)%9, np.arange(180, dtype=np.int32)%5)
    D = np.column_stack([np.eye(int(g.max())+1)[g] for g in groups])
    X = rng.normal(size=(180,3))
    w = np.exp(rng.normal(scale=.4,size=180))
    p = WeightedFEProjector(groups, engine='within', method='within', absorb_threads=1)
    p.residualize(X, w, tol=1e-11)
    native = p._absorber
    # Mutate the same caller-owned array; never use identity as a cache key.
    w[groups[0]==0] = 0
    r = p.residualize(X, w, tol=1e-11)
    assert p._active_kind == 'generic'
    sw = np.sqrt(w)
    oracle = X-D@np.linalg.lstsq(D*sw[:,None], X*sw[:,None], rcond=1e-12)[0]
    np.testing.assert_allclose(r[w>0],oracle[w>0],atol=2e-9,rtol=2e-9)
    w[:] = np.exp(rng.normal(scale=.4,size=180))
    r = p.residualize(X, w, tol=1e-11)
    assert p._absorber is native and p._active_kind == 'within'
    sw = np.sqrt(w)
    oracle = X-D@np.linalg.lstsq(D*sw[:,None], X*sw[:,None], rcond=1e-12)[0]
    np.testing.assert_allclose(r,oracle,atol=2e-9,rtol=2e-9)


def test_dynamic_weight_benchmark_recodes_missing_levels():
    import runpy
    from pathlib import Path
    mod = runpy.run_path(str(Path(__file__).resolve().parents[2]/'benchmarks/hdfe/ppml_dynamic_weights.py'))
    groups, _, _ = mod['sample'](96, 2, 20261027)
    for g in groups:
        np.testing.assert_array_equal(np.unique(g), np.arange(int(g.max())+1))


def test_actual_exhausted_native_iteration_budget_is_rejected():
    pytest.importorskip('_econhdfe_within')
    from econhdfe.errors import ConvergenceError
    rng = np.random.default_rng(1027)
    groups = tuple(np.unique(rng.integers(L,size=4000), return_inverse=True)[1].astype(np.int32)
                   for L in (400,150,35))
    p = WeightedFEProjector(groups, engine='within', method='within', max_iter=1,
                           absorb_threads=1, require_convergence=True)
    with pytest.raises(ConvergenceError) as err:
        p.residualize(rng.normal(size=(4000,2)), np.exp(rng.normal(size=4000)), tol=1e-12)
    assert err.value.code == 'convergence.weighted_fe'


@pytest.mark.parametrize('kind', ['ppml','ivppml'])
def test_requested_native_engine_reports_missing_extension(kind, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, '_econhdfe_within', None)
    y, c, e, z, off, groups, weights = _fixture(n=550)
    common = dict(y=y, absorb=groups, weights=weights, offset=off,
                  execution_config=ExecutionConfig(threads=1,memory_budget_mb=192))
    with pytest.raises(SpecificationError, match='requires the optional native extension'):
        if kind == 'ppml':
            ppmlhdfe(x=e, config=PPMLConfig(engine='within',separation=()), **common)
        else:
            ivppmlhdfe(exog=c, endog=e, instruments=z, weight_type='pweight',
                       config=IVPPMLConfig(engine='within',separation=()), **common)
