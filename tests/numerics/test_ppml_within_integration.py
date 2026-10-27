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
