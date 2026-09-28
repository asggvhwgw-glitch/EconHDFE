import numpy as np
import pandas as pd

from econhdfe import (
    ExecutionConfig, PPMLConfig, IVPPMLConfig,
    ppmlhdfe, ivppmlhdfe, PPMLHDFE,
)
from econhdfe.hdfe.weighted_projection import WeightedFEProjector


def _ppml_data(seed=945, n=4000):
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, 120, n)
    g2 = rng.integers(0, 70, n)
    g3 = rng.integers(0, 25, n)
    x = rng.normal(size=n)
    a = rng.normal(scale=.15, size=120)
    b = rng.normal(scale=.10, size=70)
    c = rng.normal(scale=.08, size=25)
    y = rng.poisson(np.exp(a[g1] + b[g2] + c[g3] + .12 * x))
    return y, x, g1, g2, g3


def test_ppml_direct_execution_config_reaches_weighted_projector():
    y, x, g1, g2, g3 = _ppml_data()
    r = ppmlhdfe(
        y, x[:, None], absorb=[g1, g2, g3], vce="model",
        config=PPMLConfig(engine="optimized", separation=(), tolerance=1e-8),
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=96),
    )
    ex = r.diagnostics["execution"]
    pr = r.diagnostics["projection_resources"]
    assert ex == {"threads": 2, "memory_budget_mb": 96, "cache": "auto"}
    assert pr["requested_threads"] == 2
    assert pr["memory_budget_mb"] == 96
    assert pr["actual_threads"] == 2


def test_reusable_ppml_caches_main_projector_when_enabled():
    y, x, g1, g2, g3 = _ppml_data(n=2500)
    df = pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2, "g3": g3})
    m = PPMLHDFE(
        df, absorb=["g1", "g2", "g3"],
        config=PPMLConfig(engine="optimized", separation=()),
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=128, cache="on"),
    )
    r1 = m.fit("y", ["x"], vce="model")
    keys1 = {k for k in m.context.cache if str(k).startswith("ppml.projector:")}
    r2 = m.fit("y", ["x"], vce="model", warm_start=r1)
    keys2 = {k for k in m.context.cache if str(k).startswith("ppml.projector:")}
    assert keys1 and keys1 == keys2
    assert np.allclose(r1.coef, r2.coef, atol=2e-8, rtol=2e-8)


def test_optimized_separation_uses_optimized_plan_and_reports_stage_costs():
    y = np.array([0., 1., 0., 0., 1.])
    g1 = np.array([1, 1, 2, 2, 2])
    g2 = np.array([1, 1, 1, 2, 2])
    cfg = dict(tolerance=1e-8, target_inner_tol=1e-10)
    replica = ppmlhdfe(
        y, None, absorb=[g1, g2], vce="model",
        config=PPMLConfig(engine="replica", **cfg),
    )
    optimized = ppmlhdfe(
        y, None, absorb=[g1, g2], vce="model",
        config=PPMLConfig(engine="optimized", **cfg),
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=96),
    )
    assert np.array_equal(replica.separation_mask, optimized.separation_mask)
    assert optimized.diagnostics["separation_solvers"]["simplex"] == "optimized:map"
    assert optimized.diagnostics["separation_solvers"]["relu"] == "optimized:map"
    for method in ("fe", "simplex", "relu"):
        assert method in optimized.diagnostics["separation_seconds"]
        assert optimized.diagnostics["separation_seconds"][method] >= 0.0
        assert method in optimized.diagnostics["separation_iterations"]


def test_optimized_projector_falls_back_safely_for_zero_weight_simplex_state():
    g1 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    g2 = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    A = np.arange(12, dtype=float).reshape(6, 2)
    p = WeightedFEProjector(
        [g1, g2], engine="optimized", method="map",
        absorb_threads=2, projection_memory_budget_mb=96,
    )
    p.residualize(A, np.ones(6), tol=1e-9)
    assert p.resource_info["backend"] == "twoway"
    zero_w = np.array([1., 1., 1., 1., 0., 0.])
    out = p.residualize(A, zero_w, tol=1e-9)
    assert np.all(np.isfinite(out))
    assert p.resource_info["backend"] == "generic"
    p.residualize(A, np.ones(6), tol=1e-9)
    assert p.resource_info["backend"] == "twoway"


def test_ivppml_execution_config_uses_same_resource_policy():
    rng = np.random.default_rng(946)
    n = 2200
    g1 = rng.integers(0, 70, n)
    g2 = rng.integers(0, 40, n)
    g3 = rng.integers(0, 18, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    x = .7 * z + .25 * c + rng.normal(scale=.7, size=n)
    y = rng.poisson(np.exp(.12 * x + .08 * c + .02 * (g1 % 5)))
    r = ivppmlhdfe(
        y, exog=c[:, None], endog=x[:, None], instruments=z[:, None],
        absorb=[g1, g2, g3], vce="model",
        config=IVPPMLConfig(engine="optimized", separation=(), max_iter=300),
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=96),
    )
    assert r.diagnostics["execution"]["threads"] == 2
    assert r.diagnostics["projection_resources"]["requested_threads"] == 2
    assert r.diagnostics["projection_resources"]["memory_budget_mb"] == 96


def test_optimized_default_separation_matches_replica_on_induced_fe_separation():
    rng = np.random.default_rng(947)
    n = 5000
    g1 = rng.integers(0, 80, n)
    g2 = rng.integers(0, 45, n)
    x = rng.normal(size=(n, 1))
    y = rng.poisson(np.exp(-.3 + .1 * x[:, 0]))
    # Force two FE levels to have only zeros so FE separation definitely trims.
    y[np.isin(g1, [0, 1])] = 0
    common = dict(tolerance=1e-8, target_inner_tol=1e-9)
    rr = ppmlhdfe(y, x, absorb=[g1, g2], vce="model", config=PPMLConfig(engine="replica", **common))
    oo = ppmlhdfe(
        y, x, absorb=[g1, g2], vce="model", config=PPMLConfig(engine="optimized", **common),
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
    )
    assert np.array_equal(rr.sample_mask, oo.sample_mask)
    assert np.array_equal(rr.separation_mask, oo.separation_mask)
    assert np.allclose(rr.coef, oo.coef, atol=2e-7, rtol=2e-7)


def test_reusable_ppml_cache_off_keeps_projector_cache_empty():
    y, x, g1, g2, g3 = _ppml_data(n=1800)
    df = pd.DataFrame({"y": y, "x": x, "g1": g1, "g2": g2, "g3": g3})
    m = PPMLHDFE(
        df, absorb=["g1", "g2", "g3"],
        config=PPMLConfig(engine="optimized", separation=()),
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=128, cache="off"),
    )
    m.fit("y", ["x"], vce="model")
    assert not {k for k in m.context.cache if str(k).startswith("ppml.projector:")}


def test_default_separation_optimized_replica_randomized_parity():
    # Regression matrix for the optimized ReLU/simplex routing change. These
    # small designs mix structural zero-only FE cells with ordinary zeros and
    # 2/3-way FE topologies; the optimized path must preserve the replica
    # sample and coefficients, not merely converge.
    for seed in range(950, 956):
        rng = np.random.default_rng(seed)
        n = 1400
        g1 = rng.integers(0, 35, n)
        g2 = rng.integers(0, 22, n)
        groups = [g1, g2]
        if seed % 2:
            g3 = rng.integers(0, 11, n)
            groups.append(g3)
        x = rng.normal(size=(n, 2))
        eta = -0.8 + 0.10 * x[:, 0] - 0.06 * x[:, 1] + 0.03 * (g1 % 5)
        y = rng.poisson(np.exp(eta))
        # Create guaranteed FE-separated observations plus many ordinary zeros.
        y[np.isin(g1, [0, 1])] = 0
        common = dict(tolerance=1e-8, target_inner_tol=2e-9, max_iter=150)
        rr = ppmlhdfe(y, x, absorb=groups, vce="model", config=PPMLConfig(engine="replica", **common))
        oo = ppmlhdfe(
            y, x, absorb=groups, vce="model",
            config=PPMLConfig(engine="optimized", **common),
            execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
        )
        assert np.array_equal(rr.sample_mask, oo.sample_mask), seed
        assert np.array_equal(rr.separation_mask, oo.separation_mask), seed
        assert np.allclose(rr.coef, oo.coef, atol=3e-7, rtol=3e-7), seed


def test_two_way_projector_honors_requested_threads_and_matches_single_thread():
    rng = np.random.default_rng(9451)
    n = 12000
    g1 = rng.integers(0, 500, n)
    g2 = rng.integers(0, 180, n)
    A = rng.normal(size=(n, 7))
    w = np.exp(rng.normal(scale=.2, size=n))
    p1 = WeightedFEProjector([g1, g2], engine="optimized", absorb_threads=1)
    p3 = WeightedFEProjector([g1, g2], engine="optimized", absorb_threads=3)
    r1, i1 = p1.residualize(A, w, tol=1e-10, return_info=True)
    r3, i3 = p3.residualize(A, w, tol=1e-10, return_info=True)
    assert i1.converged and i3.converged
    np.testing.assert_allclose(r3, r1, atol=3e-9, rtol=3e-9)
    assert p1.resource_info["backend"] == p3.resource_info["backend"] == "twoway"
    assert p1.resource_info["actual_threads"] == 1
    assert p3.resource_info["actual_threads"] == 3
    assert p3.resource_info["index_bytes"] > p1.resource_info["index_bytes"]


def test_public_two_way_ppml_reports_and_uses_requested_projection_threads():
    y, x, g1, g2, _ = _ppml_data(seed=9452, n=6000)
    r = ppmlhdfe(
        y, x[:, None], absorb=[g1, g2], vce="model",
        config=PPMLConfig(engine="optimized", separation=(), tolerance=1e-8),
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=96),
    )
    pr = r.diagnostics["projection_resources"]
    assert pr["backend"] == "twoway"
    assert pr["requested_threads"] == 2
    assert pr["actual_threads"] == 2
    assert pr["index_bytes"] > 0
