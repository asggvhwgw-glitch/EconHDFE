"""Independent scale-invariance contracts against dense FWL."""

import numpy as np
import econhdfe.hdfe.absorber as module
from econhdfe import ExecutionConfig, HDFEConfig, olshdfe
from econhdfe.hdfe.absorber import HDFEAbsorber


def test_small_units_weighted_ols_matches_dense_fwl():
    scale = 1e-8
    rng = np.random.default_rng(861)
    n = 480
    groups = [rng.integers(0, size, n) for size in (29, 31, 13)]
    x = rng.normal(size=(n, 4))
    d = np.column_stack([np.eye(g.max() + 1)[g] for g in groups])
    y = (x @ np.array([0.3, -0.6, 0.8, 1.2])
         + d @ rng.normal(size=d.shape[1]) + rng.normal(size=n))
    w = np.exp(rng.normal(size=n))
    sw = np.sqrt(w)
    z = np.column_stack([y, x])
    residual = z - d @ np.linalg.lstsq(d * sw[:, None], z * sw[:, None], rcond=None)[0]
    oracle = np.linalg.lstsq(residual[:, 1:] * sw[:, None], residual[:, 0] * sw, rcond=None)[0]
    result = olshdfe(
        None, y=y * scale, x=x * scale, absorb=groups, weights=w,
        weight_type="aweight", vce="robust", pool_size=5,
        execution_config=ExecutionConfig(threads=4, cache="off"),
        hdfe_config=HDFEConfig(tolerance=1e-10),
    )
    assert result.nobs == n and result.rank == 4 and result.converged
    np.testing.assert_allclose(result.params[:4], oracle, atol=1e-8, rtol=1e-8)


def test_auto_with_mixed_units_matches_dense_oracle(monkeypatch):
    acceleration = "auto"
    monkeypatch.setattr(module, "_CG_PARALLEL_MIN_ROWS", 1)
    scale = 1e-8
    rng = np.random.default_rng(815)
    n = 240
    groups = [rng.integers(0, size, n) for size in (17, 19, 13)]
    x = rng.normal(size=(n, 4))
    x[:, 0] = 0
    w = np.exp(rng.normal(size=n))
    w[::17] = 0
    d = np.column_stack([np.eye(g.max() + 1)[g] for g in groups])
    sw = np.sqrt(w)
    oracle = x - d @ np.linalg.lstsq(d * sw[:, None], x * sw[:, None], rcond=None)[0]
    scales = np.array([1.0, scale, 1.0, 1 / scale])
    solver = HDFEAbsorber(
        groups, weights=w, tol=1e-10, acceleration=acceleration,
        projection_backend="indexed", absorb_threads=4, core_reduction="off",
    )
    a = x * scales
    result, info = solver.residualize(a, copy=False, return_info=True)
    assert info.converged and np.shares_memory(a, result)
    np.testing.assert_allclose(result / scales, oracle, atol=2e-9, rtol=2e-9)


def test_unrepresentable_energy_is_not_convergence():
    scale = 1e200
    groups = [np.arange(60) % 7, np.arange(60) % 11, np.arange(60) % 13]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        _, info = HDFEAbsorber(groups, core_reduction="off").residualize(
            np.full((60, 2), scale), return_info=True,
        )
    assert not info.converged and info.criterion == "cg_breakdown"


def test_pooled_plain_map_preserves_units_in_later_pools(monkeypatch):
    import econhdfe.pipeline as pipeline

    monkeypatch.setattr(pipeline, "should_probe_plain_map", lambda **kw: True)
    rng = np.random.default_rng(863)
    n = 600
    groups = [rng.integers(0, size, n) for size in (7, 9, 11)]
    weights = np.exp(rng.normal(scale=0.2, size=n))
    rhs = rng.normal(size=(n, 5))
    scales = np.array([1.0, 1.0, 1e-8, 1e8, 1e-8])
    d = np.column_stack([np.eye(g.max() + 1)[g] for g in groups])
    sw = np.ones(n) if weights is None else np.sqrt(weights)
    oracle = rhs - d @ np.linalg.lstsq(d * sw[:, None], rhs * sw[:, None], rcond=None)[0]
    absorber = HDFEAbsorber(
        groups, weights=weights, acceleration="cg", tol=1e-10,
        projection_backend="indexed", core_reduction="off", absorb_threads=4,
    )
    result, info = pipeline.residualize_blocks(
        absorber, rhs * scales, 2, adaptive_auto=True, profile=True,
    )
    assert info.converged and absorber.acceleration == "none"
    pools = absorber.execution_profile["pools"]
    assert pools[0]["criterion"] == "auto_plain"
    assert all("cg_iterations" not in pool["stages"] for pool in pools)
    np.testing.assert_allclose(result / scales, oracle, atol=2e-9, rtol=2e-9)
