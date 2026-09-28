import numpy as np
import pytest
from pyreghdfe import reghdfe, ivreghdfe, stock_yogo_critical_values
from pyreghdfe.encoding import factorize_interaction
from pyreghdfe.kernels import hac_meat, driscoll_kraay_meat
from pyreghdfe.vcov import score_covariance


def _meat(scores, codes):
    g = int(np.max(codes)) + 1
    sums = np.column_stack([
        np.bincount(codes, weights=scores[:, j], minlength=g)
        for j in range(scores.shape[1])
    ])
    return sums.T @ sums


def test_three_way_cluster_meat_matches_inclusion_exclusion():
    rng = np.random.default_rng(301)
    n = 3000
    scores = rng.normal(size=(n, 3))
    cs = [rng.integers(0, 40, n).astype(np.int32),
          rng.integers(0, 25, n).astype(np.int32),
          rng.integers(0, 15, n).astype(np.int32)]
    got = score_covariance(scores, kind="cluster", clusters=cs, small_sample=False)
    expected = np.zeros((3, 3))
    for i in range(3):
        expected += _meat(scores, cs[i])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        code, _ = factorize_interaction([cs[i], cs[j]])
        expected -= _meat(scores, code)
    code, _ = factorize_interaction(cs)
    expected += _meat(scores, code)
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-9)


def test_hac_bartlett_matches_direct_lag_formula():
    rng = np.random.default_rng(302)
    s = rng.normal(size=(500, 2))
    got, bw = hac_meat(s, bandwidth=3, kernel="bartlett")
    expected = s.T @ s
    for lag in (1, 2):
        cross = s[lag:].T @ s[:-lag]
        expected += (1 - lag / 3) * (cross + cross.T)
    assert bw == 3
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_dkraay_aggregates_scores_by_time():
    rng = np.random.default_rng(303)
    units, periods = 30, 20
    time = np.tile(np.arange(periods), units)
    s = rng.normal(size=(units * periods, 2))
    got, bw, t = driscoll_kraay_meat(s, time, bandwidth=3)
    sums = np.zeros((periods, 2))
    for tt in range(periods):
        sums[tt] = s[time == tt].sum(axis=0)
    expected = sums.T @ sums
    for lag in (1, 2):
        cross = sums[lag:].T @ sums[:-lag]
        expected += (1 - lag / 3) * (cross + cross.T)
    assert (bw, t) == (3, periods)
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_single_endog_kp_cluster_f_equals_robust_first_stage_f():
    rng = np.random.default_rng(304)
    n = 7000
    z = rng.normal(size=(n, 2)); c = rng.integers(0, 100, n)
    w = rng.normal(size=n); v = rng.normal(size=n)
    x = .5*z[:, 0] + .25*z[:, 1] + .2*w + v
    y = 1.4*x + .3*w + .5*v + rng.normal(size=n)
    fe = rng.integers(0, 150, n)
    r = ivreghdfe(None, y=y, exog=w, endog=x, instruments=z, absorb=fe,
                  cluster=c, vce="cluster")
    fs = r.first_stage["diagnostics"][0]["f_robust"]
    kp = r.diagnostics["kleibergen_paap"]["rk_wald_f"]
    np.testing.assert_allclose(kp, fs, atol=1e-8, rtol=1e-8)


def test_sw_diagnostics_multiple_endogenous_are_finite():
    rng = np.random.default_rng(305)
    n = 6000
    z = rng.normal(size=(n, 3)); w = rng.normal(size=n)
    v1, v2 = rng.normal(size=(2, n))
    x1 = .7*z[:, 0] + .3*z[:, 1] + .1*w + v1
    x2 = .2*z[:, 0] + .6*z[:, 2] - .1*w + v2
    y = 1.2*x1 - .7*x2 + .2*w + .4*v1 + .3*v2 + rng.normal(size=n)
    fe = rng.integers(0, 100, n)
    r = ivreghdfe(None, y=y, exog=w, endog=np.c_[x1, x2], instruments=z,
                  absorb=fe, vce="robust")
    sw = r.diagnostics["sanderson_windmeijer"]
    assert len(sw) == 2
    assert all(item["df1"] == 2 for item in sw)
    assert all(np.isfinite(item["sw"]["f_robust"]) for item in sw)
    assert np.isfinite(r.diagnostics["kleibergen_paap"]["rk_wald_f"])


def test_stock_yogo_common_values():
    a = stock_yogo_critical_values(2, 1)
    assert a["maximal_iv_size"]["10%"] == 19.93
    b = stock_yogo_critical_values(3, 2)
    assert b["maximal_iv_size"]["10%"] == 13.43
    c = stock_yogo_critical_values(3, 1)
    assert c["maximal_relative_bias"]["5%"] == 13.91
    assert stock_yogo_critical_values(11, 1)["available"] is False


def test_api_dkraay_requires_time_and_runs_with_time():
    rng = np.random.default_rng(306)
    units, periods = 50, 15
    fe = np.repeat(np.arange(units), periods)
    time = np.tile(np.arange(periods), units)
    X = rng.normal(size=(len(fe), 2))
    y = X @ np.array([.5, -.3]) + rng.normal(size=units)[fe] + rng.normal(size=len(fe))
    with pytest.raises(ValueError):
        reghdfe(None, y=y, x=X, absorb=fe, vce="dkraay")
    r = reghdfe(None, y=y, x=X, absorb=fe, vce="dkraay", time=time, bandwidth=4)
    assert np.all(np.isfinite(r.stderr))


def test_dkraay_bandwidth_one_equals_time_cluster_vcov():
    rng = np.random.default_rng(919)
    n_t, n_i = 18, 25
    n = n_t * n_i
    t = np.repeat(np.arange(n_t), n_i)
    X = rng.normal(size=(n, 3))
    e = rng.normal(size=n)
    bread = np.linalg.pinv(X.T @ X, hermitian=True)
    tcodes = t.astype(np.int32)
    from pyreghdfe.vcov import ols_vcov
    v_cluster = ols_vcov(X, e, bread, kind="cluster", clusters=[tcodes], k_total=3)
    v_dk = ols_vcov(X, e, bread, kind="dkraay", time=t, bandwidth=1, k_total=3)
    np.testing.assert_allclose(v_dk, v_cluster, rtol=1e-12, atol=1e-12)


def test_cluster_dimension_cap_matches_reghdfe():
    rng = np.random.default_rng(120)
    n = 100
    df = {"y": rng.normal(size=n), "x": rng.normal(size=n), "fe": np.arange(n) % 10}
    for j in range(11):
        df[f"c{j}"] = np.arange(n) % (j + 2)
    import pandas as pd
    df = pd.DataFrame(df)
    import pytest
    with pytest.raises(ValueError, match="at most 10 cluster"):
        reghdfe(df, y="y", x=["x"], absorb=["fe"], cluster=[f"c{j}" for j in range(11)])


def test_kp_supports_hac_and_dkraay_covariance_engines():
    rng = np.random.default_rng(121)
    units, periods = 80, 25
    n = units * periods
    firm = np.repeat(np.arange(units), periods)
    time = np.tile(np.arange(periods), units)
    z = rng.normal(size=(n, 2)); w = rng.normal(size=n); v = rng.normal(size=n)
    x = .5*z[:, 0] + .2*z[:, 1] + .1*w + v
    y = 1.3*x + .2*w + .4*v + rng.normal(size=n)
    hac = ivreghdfe(None, y=y, exog=w, endog=x, instruments=z, absorb=firm,
                    vce="hac", time=time, panel=firm, bandwidth=3)
    dk = ivreghdfe(None, y=y, exog=w, endog=x, instruments=z, absorb=firm,
                   vce="dkraay", time=time, bandwidth=3)
    assert np.isfinite(hac.diagnostics["kleibergen_paap"]["rk_wald_f"])
    assert np.isfinite(dk.diagnostics["kleibergen_paap"]["rk_wald_f"])


def test_kp_hac_uses_bartlett_for_ivreg2_compatibility():
    rng = np.random.default_rng(122)
    n = 2500
    t = np.arange(n)
    z = rng.normal(size=(n, 2)); w = rng.normal(size=n); v = rng.normal(size=n)
    x = .45*z[:, 0] + .25*z[:, 1] + .15*w + v
    y = 1.2*x + .2*w + .3*v + rng.normal(size=n)
    fe = np.arange(n) % 50
    a = ivreghdfe(None, y=y, exog=w, endog=x, instruments=z, absorb=fe,
                  vce="hac", time=t, bandwidth=5, kernel="bartlett")
    b = ivreghdfe(None, y=y, exog=w, endog=x, instruments=z, absorb=fe,
                  vce="hac", time=t, bandwidth=5, kernel="parzen")
    np.testing.assert_allclose(
        a.diagnostics["kleibergen_paap"]["rk_wald_f"],
        b.diagnostics["kleibergen_paap"]["rk_wald_f"],
        rtol=1e-12, atol=1e-12,
    )
    # Coefficient VCE still respects the user kernel.
    assert not np.allclose(a.stderr, b.stderr)


def test_all_ivreg2_kernel_weights_and_spectral_horizon():
    from pyreghdfe.kernels import kernel_weight, hac_meat
    assert kernel_weight(1, 4, "thann") == pytest.approx(0.5 + 0.5*np.cos(np.pi/4))
    assert kernel_weight(1, 4, "thamm") == pytest.approx(0.54 + 0.46*np.cos(np.pi/4))
    assert np.isfinite(kernel_weight(1, 4, "qs"))
    assert kernel_weight(1, 4, "daniell") == pytest.approx(np.sin(np.pi/4)/(np.pi/4))
    assert np.isfinite(kernel_weight(1, 4, "tent"))
    # Spectral kernels retain nonzero lags beyond bandwidth; lag-window
    # Bartlett does not.  A score process engineered to correlate only at
    # lag 5 therefore distinguishes the horizons when bw=3.
    s = np.zeros((12, 1))
    s[[0, 5], 0] = 1.0
    bart, _ = hac_meat(s, bandwidth=3, kernel="bartlett")
    qs, _ = hac_meat(s, bandwidth=3, kernel="qs")
    assert not np.allclose(bart, qs)


def test_truncated_includes_bandwidth_boundary_lag():
    from pyreghdfe.kernels import hac_meat
    s = np.zeros((8, 1))
    s[[0, 3], 0] = 1.0
    got, _ = hac_meat(s, bandwidth=3, kernel="truncated")
    # Diagonal 2 plus twice the lag-3 cross-product of 1.
    np.testing.assert_allclose(got, np.array([[4.0]]), atol=1e-12)


def test_panel_hac_matches_manual_within_panel_lags_and_rejects_duplicate_time_without_panel():
    from pyreghdfe.kernels import hac_meat
    # Two panels, three periods each. Cross-panel adjacency must never enter.
    panel = np.repeat([10, 20], 3)
    time = np.tile(np.arange(3), 2)
    s = np.array([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]])
    got, _ = hac_meat(s, time=time, panel=panel, bandwidth=2, kernel="bartlett")
    diag = float(np.sum(s[:, 0] ** 2))
    lag1 = 2*1 + 3*2 + 20*10 + 30*20
    expected = np.array([[diag + 2 * 0.5 * lag1]])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    with pytest.raises(ValueError, match="provide panel"):
        hac_meat(s, time=time, bandwidth=2, kernel="bartlett")
