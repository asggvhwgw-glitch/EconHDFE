import numpy as np
import pandas as pd

from econhdfe import olshdfe, ivhdfe, FixedEffect
from econhdfe.reporting import reghdfe_r2_statistics
from econhdfe.sessions import OLSHDFESession


def _df(seed=4701, n=1200):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(60), n // 60)
    if len(firm) < n:
        firm = np.r_[firm, np.arange(n - len(firm))]
    year = np.arange(n) % 10
    cluster = np.arange(n) % 12
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    firm_fe = rng.normal(scale=0.7, size=int(firm.max()) + 1)
    year_fe = rng.normal(scale=0.3, size=10)
    y = 0.8 * x + firm_fe[firm] + year_fe[year] + rng.normal(scale=0.8, size=n)
    endog = 0.7 * z + 0.2 * x + rng.normal(size=n)
    yiv = 1.1 * endog + 0.3 * x + firm_fe[firm] + year_fe[year] + rng.normal(size=n)
    return pd.DataFrame({
        "y": y, "yiv": yiv, "x": x, "endog": endog, "z": z,
        "firm": firm, "year": year, "cluster": cluster,
    })


def _oracle(result, y, yw, weights, *, has_intercept=True):
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    rss = float(np.sum(w * result.residuals ** 2))
    if has_intercept:
        mean = float(np.sum(w * y) / np.sum(w))
        tss = float(np.sum(w * (y - mean) ** 2))
    else:
        tss = float(np.sum(w * y ** 2))
    tssw = float(np.sum(w * yw ** 2))
    n = float(result.nobs)
    fit_df = n - result.df_absorbed - result.dof_info.nested - result.rank
    r2 = 1.0 - rss / tss
    r2w = 1.0 - rss / tssw
    r2a = 1.0 - (rss / fit_df) / (tss / (n - int(has_intercept)))
    r2aw = 1.0 - (rss / fit_df) / (tssw / (fit_df + result.rank))
    return r2, r2w, r2a, r2aw, fit_df


def test_cluster_df_does_not_enter_adjusted_r2():
    df = _df()
    iid = olshdfe(df, y="y", x=["x"], absorb=["firm", "year"], vce="iid", keep_state=True)
    clu = olshdfe(
        df, y="y", x=["x"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", keep_state=True,
    )
    # Point estimates/residuals and fit statistics are VCE-invariant.
    np.testing.assert_allclose(clu.params, iid.params, rtol=0, atol=2e-12)
    np.testing.assert_allclose(clu.residuals, iid.residuals, rtol=0, atol=2e-11)
    assert clu.df_resid == 11  # G - 1, inference df
    assert iid.df_resid > clu.df_resid
    assert clu.r2_adjusted == iid.r2_adjusted
    assert clu.r2_adjusted_within == iid.r2_adjusted_within


def test_nested_cluster_fe_is_charged_to_fit_df_but_not_crv1_df_a():
    df = _df()
    r = olshdfe(
        df, y="y", x=["x"], absorb=["firm"],
        cluster="firm", vce="cluster", keep_state=True,
    )
    assert r.dof_info.nested > 0
    # Nested FE coefficients are excluded from df_absorbed for CRV1 small-sample
    # scaling, but reghdfe adds df_a_nested back when forming fit residual df.
    assert r.df_absorbed == 0
    oracle = _oracle(r, df["y"].to_numpy(), r.state.y_within, None)
    np.testing.assert_allclose(
        [r.r2, r.r2_within, r.r2_adjusted, r.r2_adjusted_within], oracle[:4],
        rtol=2e-12, atol=2e-12,
    )
    assert oracle[4] == r.nobs - r.rank - r.dof_info.nested


def test_adjusted_within_matches_reghdfe_formula():
    df = _df()
    r = olshdfe(df, y="y", x=["x"], absorb=["firm", "year"], keep_state=True)
    oracle = _oracle(r, df["y"].to_numpy(), r.state.y_within, None)
    np.testing.assert_allclose(r.r2_adjusted_within, oracle[3], rtol=2e-12, atol=2e-12)
    assert "r2_adjusted_within" in r.model_stats()


def test_slope_only_absorb_uses_uncentered_total_tss():
    rng = np.random.default_rng(4702)
    n = 500
    g = np.arange(n) % 25
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    y = 2.5 + 0.4 * x + (0.15 * (g % 5)) * z + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "g": g})
    r = olshdfe(
        df, y="y", x=["x"],
        absorb=[FixedEffect("g", slopes=("z",), intercept=False)],
        keep_state=True,
    )
    oracle = _oracle(r, y, r.state.y_within, None, has_intercept=False)
    np.testing.assert_allclose(
        [r.r2, r.r2_adjusted], [oracle[0], oracle[2]], rtol=2e-11, atol=2e-11
    )


def test_linear_iv_adjusted_r2_is_not_cluster_df_driven():
    df = _df()
    iid = ivhdfe(
        df, y="yiv", exog=["x"], endog=["endog"], instruments=["z"],
        absorb=["firm", "year"], vce="iid",
    )
    clu = ivhdfe(
        df, y="yiv", exog=["x"], endog=["endog"], instruments=["z"],
        absorb=["firm", "year"], cluster="cluster", vce="cluster",
    )
    assert clu.df_resid == 11
    np.testing.assert_allclose(
        [clu.r2, clu.r2_within, clu.r2_adjusted, clu.r2_adjusted_within],
        [iid.r2, iid.r2_within, iid.r2_adjusted, iid.r2_adjusted_within],
        rtol=5e-11, atol=5e-11,
    )


def test_reporting_helper_reproduces_reghdfe_used_df_r_algebra():
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    yw = np.array([-2.0, -1.0, 1.0, 2.0, 1.5, -1.5])
    e = np.array([0.2, -0.1, 0.3, -0.2, 0.1, -0.3])
    out = reghdfe_r2_statistics(
        y, yw, e, effective_n=100, rank=3, df_absorbed=20, df_nested=10,
        has_intercept=True,
    )
    assert out["df_resid_fit"] == 67
    rss = float(e @ e)
    tss = float(np.sum((y - y.mean()) ** 2))
    tssw = float(yw @ yw)
    assert out["r2_adjusted"] == 1 - (rss / 67) / (tss / 99)
    assert out["r2_adjusted_within"] == 1 - (rss / 67) / (tssw / 70)


def test_nested_cluster_adjusted_r2_is_same_as_unclustered_fit():
    df = _df()
    iid = olshdfe(df, y="y", x=["x"], absorb=["firm"], vce="iid")
    clu = olshdfe(df, y="y", x=["x"], absorb=["firm"], cluster="firm", vce="cluster")
    assert iid.dof_info.nested == 0
    assert clu.dof_info.nested > 0
    np.testing.assert_allclose(
        [clu.r2, clu.r2_within, clu.r2_adjusted, clu.r2_adjusted_within],
        [iid.r2, iid.r2_within, iid.r2_adjusted, iid.r2_adjusted_within],
        rtol=2e-12, atol=2e-12,
    )
