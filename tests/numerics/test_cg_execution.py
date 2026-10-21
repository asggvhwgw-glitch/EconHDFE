"""Numerical and execution regressions for pooled warm-start CG."""
import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, olshdfe
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.pipeline import _residualize_blocks


def cycle(m=80):
    i = np.arange(m, dtype=np.int32)
    rows = np.vstack([np.column_stack([i, i, i]),
                      np.column_stack([(i + 1) % m, i, i])])
    return [rows[:, j] for j in range(3)]


def test_pooled_hard_problem_keeps_probe_and_profiles_all_work(monkeypatch):
    import econhdfe.pipeline as pipeline
    monkeypatch.setattr(pipeline, "should_probe_plain_map", lambda **kw: True)
    groups = cycle()
    original = np.random.default_rng(1843).normal(size=(160, 7))
    absorber = HDFEAbsorber(
        groups, tol=1e-10, acceleration="cg", core_reduction="off",
        absorb_threads=2,
    )
    reference = absorber.residualize(original)
    cg_inputs = []
    cg = absorber._cg

    def observe(a, **kwargs):
        cg_inputs.append(a.copy())
        return cg(a, **kwargs)

    monkeypatch.setattr(absorber, "_cg", observe)
    got, info = _residualize_blocks(
        absorber, original.copy(), 3, adaptive_auto=True, profile=True,
    )
    probe = original[:, :3].copy()
    probe /= np.max(np.abs(probe), axis=0)
    absorber._auto_warmup(probe)
    np.testing.assert_allclose(cg_inputs[0], probe, atol=0, rtol=0)
    assert not np.allclose(cg_inputs[0], original[:, :3])
    np.testing.assert_allclose(got, reference, atol=2e-8, rtol=2e-8)
    assert info.converged
    pools = absorber.execution_profile["pools"]
    assert [p["columns"] for p in pools] == [3, 3, 1]
    assert pools[0]["stages"]["probe_iterations"] == 2
    assert pools[0]["iterations"] == 2 + pools[0]["stages"]["cg_iterations"]
    assert "probe_iterations" not in pools[1]["stages"]
    for p in pools:
        stages = p["stages"]
        assert 0 <= stages["cg_projection_seconds"] <= stages["cg_seconds"] <= p["seconds"]
    assert absorber.auto_plain_limit == 8


def test_auto_respects_total_iteration_budget():
    groups = cycle()
    rhs = np.random.default_rng(1844).normal(size=(160, 3))
    absorber = HDFEAbsorber(
        groups, tol=1e-14, max_iter=3, acceleration="auto",
        core_reduction="off", absorb_threads=2,
    )
    absorber._profile_enabled = True
    _, info = absorber.residualize(rhs, return_info=True)
    assert not info.converged
    assert info.iterations == 3
    assert absorber._iteration_profile["cg_iterations"] == 1


def test_ols_profile_preserves_estimation_and_accounts_for_stages():
    fe_count = 3
    rng = np.random.default_rng(1845)
    n = 2000
    df = pd.DataFrame({f"g{j}": rng.integers(0, 25 + j, n) for j in range(fe_count)})
    for j in range(4):
        df[f"x{j}"] = rng.normal(size=n)
    df["y"] = df.x0 + df.g0 / 10 + rng.normal(size=n)
    kwargs = dict(y="y", x=[f"x{j}" for j in range(4)],
                  absorb=[f"g{j}" for j in range(fe_count)],
                  cluster="g0", pool_size=2)
    plain = olshdfe(df, **kwargs, execution_config=ExecutionConfig(threads=2))
    timed = olshdfe(df, **kwargs, execution_config=ExecutionConfig(threads=2, profile="full"))
    np.testing.assert_array_equal(plain.params, timed.params)
    np.testing.assert_array_equal(plain.vcov, timed.vcov)
    assert plain.df_absorbed == timed.df_absorbed
    assert plain.profile is None
    stages = timed.profile["stages"]
    assert all(t >= 0 for t in stages.values())
    assert sum(stages.values()) == pytest.approx(timed.profile["total_seconds"], abs=.002)
    assert sum(p["seconds"] for p in timed.profile["absorption"]["pools"]) <= stages["absorption_seconds"]
