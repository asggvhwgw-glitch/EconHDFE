import numpy as np
import pandas as pd
from econhdfe import PPMLConfig, ivreghdfe, ppmlhdfe, reghdfe
from econhdfe.hdfe.dof import absorbed_dof


def _counterexample_groups(repeat=1):
    edges = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=np.int32)
    edges = np.repeat(edges, repeat, axis=0)
    return [edges[:, j] for j in range(3)]


def test_exact_dof_finds_collective_three_way_dependency_missed_by_pairwise():
    groups = _counterexample_groups()
    pair = absorbed_dof(groups, method="pairwise", adjust_nested=False)
    exact = absorbed_dof(groups, method="exact", adjust_nested=False)
    assert pair.df_absorbed == 5
    assert exact.df_absorbed == 4
    assert exact.m_by_component == (0, 1, 2)
    assert exact.exact_by_component == (True, True, True)


def _nested_cluster_fixture(seed=9302):
    rng = np.random.default_rng(seed)
    city = np.repeat(np.arange(8), 5 * 6)
    year = np.tile(np.repeat(np.arange(6), 5), 8)
    n = len(city)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    endog = 0.8 * z + 0.3 * x + v
    cy = city * 6 + year
    cell = rng.normal(size=48)
    y = 0.65 * x + 1.4 * endog + cell[cy] + 0.4 * v + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x": x, "endog": endog, "z": z, "city": city, "year": year})


def test_solver_canonicalization_does_not_change_ols_requested_fe_inference():
    df = _nested_cluster_fixture()
    base = dict(
        data=df, y="y", x=["x"], absorb=["year", ("city", "year")],
        cluster="city", vce="cluster", drop_singletons=False, tol=1e-10,
        dof_method="exact",
    )
    a = reghdfe(**base, canonicalize_fe=True)
    b = reghdfe(**base, canonicalize_fe=False)
    assert a.df_absorbed == b.df_absorbed
    assert a.dof_info.nested == b.dof_info.nested
    np.testing.assert_allclose(a.params, b.params, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=2e-9, rtol=2e-9)


def test_solver_canonicalization_does_not_change_iv_requested_fe_inference():
    df = _nested_cluster_fixture(seed=9303)
    base = dict(
        data=df, y="y", exog=["x"], endog=["endog"], instruments=["z"],
        absorb=["year", ("city", "year")], cluster="city", vce="cluster",
        drop_singletons=False, tol=1e-10, dof_method="exact",
    )
    a = ivreghdfe(**base, canonicalize_fe=True)
    b = ivreghdfe(**base, canonicalize_fe=False)
    assert a.df_absorbed == b.df_absorbed
    assert a.dof_info.nested == b.dof_info.nested
    np.testing.assert_allclose(a.params, b.params, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=2e-9, rtol=2e-9)


def test_ppml_can_request_exact_three_way_dof():
    groups = _counterexample_groups(repeat=3)
    n = len(groups[0])
    y = np.linspace(1.0, 3.0, n)
    X = np.linspace(-1.0, 1.0, n)[:, None]
    pair = ppmlhdfe(y, X, absorb=groups, names=("x",), config=PPMLConfig(separation=("fe",), dof_method="pairwise"))
    exact = ppmlhdfe(y, X, absorb=groups, names=("x",), config=PPMLConfig(separation=("fe",), dof_method="exact"))
    assert pair.df_absorbed == 5
    assert exact.df_absorbed == 4
    np.testing.assert_allclose(pair.coef, exact.coef, atol=1e-9, rtol=1e-9)


def test_ivppml_can_request_exact_three_way_dof():
    from econhdfe import IVPPMLConfig, ivppmlhdfe
    rng = np.random.default_rng(9304)
    base = np.column_stack(_counterexample_groups(repeat=40))
    n = len(base)
    g1, g2, g3 = (base[:, j] for j in range(3))
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    e = 0.8 * z + 0.2 * c + rng.normal(scale=0.5, size=n)
    a1 = np.array([0.1, -0.1])
    a2 = np.array([0.08, -0.04])
    a3 = np.array([0.03, -0.02, 0.01])
    y = rng.poisson(np.exp(a1[g1] + a2[g2] + a3[g3] + 0.1*c + 0.25*e))
    common = dict(
        y=y, exog=c, endog=e, instruments=z, absorb=[g1, g2, g3], vce="robust",
        exog_names=["c"], endog_names=["e"], instrument_names=["z"],
    )
    base_cfg = dict(separation=(), standardize=False, tolerance=1e-9, target_inner_tol=1e-10, max_iter=300)
    pair = ivppmlhdfe(**common, config=IVPPMLConfig(**base_cfg, dof_method="pairwise"))
    exact = ivppmlhdfe(**common, config=IVPPMLConfig(**base_cfg, dof_method="exact"))
    assert pair.df_absorbed == 5
    assert exact.df_absorbed == 4
    np.testing.assert_allclose(pair.coef, exact.coef, atol=2e-8, rtol=2e-8)
