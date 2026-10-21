import numpy as np
import pandas as pd
from econhdfe import olshdfe, HDFEConfig, InferenceConfig, ExecutionConfig


def _linear_df(seed=101, n=1200):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 80, n)
    year = rng.integers(0, 12, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    x = 0.8 * z + 0.2 * c + rng.normal(size=n)
    y = 1.3 * x + 0.4 * c + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "c": c, "z": z, "firm": firm, "year": year})


def test_common_configs_control_strategy_reporting_and_profile():
    df = _linear_df(n=800)
    r = olshdfe(
        df, y="y", x=["x"], absorb=["firm", "year"],
        hdfe_config=HDFEConfig(solver="map", tolerance=1e-9, max_iter=4000, dof_method="firstpair"),
        inference_config=InferenceConfig(vce="robust", confidence_level=0.90, diagnostics="off"),
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=128, profile="summary"),
    )
    assert r.vce == "robust"
    assert r.confidence_level == 0.90
    assert r.profile is not None and r.profile["total_seconds"] >= 0
    assert r.absorb_info["solver_selection"]["requested"] == "map"
    ci90 = r.conf_int()
    ci95 = r.conf_int(0.95)
    assert np.all((ci90[:, 1] - ci90[:, 0]) < (ci95[:, 1] - ci95[:, 0]))


def test_reusable_ppml_model_invalidates_fe_plan_when_source_fe_changes():
    from econhdfe import PPMLHDFE
    rng = np.random.default_rng(404)
    n = 500
    df = pd.DataFrame({
        "y": rng.poisson(1.2, n),
        "x": rng.normal(size=n),
        "firm": rng.integers(0, 30, n),
        "year": rng.integers(0, 6, n),
    })
    m = PPMLHDFE(df, absorb=["firm", "year"], execution_config=ExecutionConfig(cache_validation="signature"))
    sig0 = m.plan.fingerprint
    _ = m.fit("y", ["x"])
    df.loc[:20, "firm"] = 999
    _ = m.fit("y", ["x"])
    assert m.plan.fingerprint != sig0
