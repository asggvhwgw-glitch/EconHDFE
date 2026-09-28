import math

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import f as f_dist, chi2 as chi2_dist

from econhdfe import olshdfe, ivhdfe, ppmlhdfe, PPMLConfig
from econhdfe.sessions import OLSHDFESession, IVHDFESession


def _linear_df(seed=4801, n=1600):
    rng = np.random.default_rng(seed)
    firm = np.arange(n) % 80
    year = np.arange(n) % 8
    cluster = rng.integers(0, 20, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    endog = 0.75 * z + 0.25 * x1 + v
    af = rng.normal(scale=0.7, size=80)
    at = rng.normal(scale=0.2, size=8)
    y = 0.6 * x1 - 0.3 * x2 + af[firm] + at[year] + rng.normal(size=n)
    yiv = 1.1 * endog + 0.2 * x1 + af[firm] + at[year] + 0.5 * v + rng.normal(size=n)
    return pd.DataFrame({
        "y": y, "yiv": yiv, "x1": x1, "x2": x2, "z": z, "endog": endog,
        "firm": firm, "year": year, "cluster": cluster,
    })


def test_ols_main_fit_statistics_match_reghdfe_algebra():
    df = _linear_df()
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", keep_state=True,
    )
    y = df["y"].to_numpy()
    yw = r.state.y_within
    e = r.residuals
    rss = float(e @ e)
    tss = float(np.sum((y - y.mean()) ** 2))
    tssw = float(yw @ yw)
    fit_df = r.nobs - r.df_absorbed - r.dof_info.nested - r.rank
    rmse = math.sqrt(rss / fit_df)
    ll = -0.5 * r.nobs * (1 + math.log(2 * math.pi) + math.log(rss / r.nobs))
    ll0 = -0.5 * r.nobs * (1 + math.log(2 * math.pi) + math.log(tssw / r.nobs))
    wald = float(r.params @ np.linalg.solve(r.vcov, r.params))
    F = wald / r.rank
    Fp = float(f_dist.sf(F, r.rank, r.df_resid))

    np.testing.assert_allclose(
        [r.rss, r.tss, r.tss_within, r.mss, r.rmse, r.loglike, r.loglike_null,
         r.f_statistic, r.f_pvalue, r.df_model, r.df_resid_fit],
        [rss, tss, tssw, tss-rss, rmse, ll, ll0, F, Fp, r.rank, fit_df],
        rtol=2e-12, atol=2e-12,
    )
    assert r.vcov_rank == np.linalg.matrix_rank(r.vcov)
    stats = r.model_stats()
    for key in ("rss", "tss", "tss_within", "mss", "rmse", "loglike", "loglike_null",
                "f_statistic", "f_pvalue", "df_model", "df_resid_fit", "vcov_rank"):
        assert key in stats


def test_ols_secondary_stats_are_vce_invariant_except_model_f():
    df = _linear_df()
    iid = olshdfe(df, y="y", x=["x1", "x2"], absorb=["firm", "year"], vce="iid")
    clu = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster",
    )
    np.testing.assert_allclose(
        [clu.rss, clu.tss, clu.tss_within, clu.mss, clu.rmse, clu.loglike, clu.loglike_null,
         clu.df_model, clu.df_resid_fit],
        [iid.rss, iid.tss, iid.tss_within, iid.mss, iid.rmse, iid.loglike, iid.loglike_null,
         iid.df_model, iid.df_resid_fit], rtol=2e-12, atol=2e-12,
    )
    assert clu.df_resid == 19
    assert iid.df_resid > clu.df_resid


def test_iv_unambiguous_small_sample_fit_statistics_match_oracle():
    df = _linear_df()
    r = ivhdfe(
        df, y="yiv", exog=["x1"], endog=["endog"], instruments=["z"],
        absorb=["firm", "year"], cluster="cluster", vce="cluster", keep_state=True,
    )
    rss = float(r.residuals @ r.residuals)
    tssw = float(r.state.y_within @ r.state.y_within)
    fit_df = r.nobs - r.df_absorbed - r.dof_info.nested - r.rank
    wald = float(r.params @ np.linalg.solve(r.vcov, r.params))
    F = wald / r.rank
    Fp = float(f_dist.sf(F, r.rank, r.df_resid))
    np.testing.assert_allclose(
        [r.rss, r.tss_within, r.rmse, r.f_statistic, r.f_pvalue, r.df_model, r.df_resid_fit],
        [rss, tssw, math.sqrt(rss / fit_df), F, Fp, r.rank, fit_df],
        rtol=5e-11, atol=5e-11,
    )
    # Overall IV R2 remains econhdfe's extended reconstruction; do not relabel
    # the partialled-out ivreghdfe convention as overall TSS/MSS.
    assert r.tss is None and r.mss is None and r.loglike is None


def _ppml_data(seed=4802, n=1800):
    rng = np.random.default_rng(seed)
    firm = np.arange(n) % 60
    year = np.arange(n) % 6
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    af = rng.normal(scale=0.25, size=60)
    at = rng.normal(scale=0.1, size=6)
    eta = 0.35 * x1 - 0.2 * x2 + af[firm] + at[year] + 0.7
    y = rng.poisson(np.exp(eta))
    return y.astype(float), np.column_stack([x1, x2]), [firm, year]


def test_ppml_deviance_is_invariant_to_internal_standardization():
    y, X, fe = _ppml_data()
    common = dict(separation=(), tolerance=1e-9, target_inner_tol=1e-10, max_iter=300)
    a = ppmlhdfe(y, X, absorb=fe, vce="model", config=PPMLConfig(standardize=True, **common))
    b = ppmlhdfe(y, X, absorb=fe, vce="model", config=PPMLConfig(standardize=False, **common))
    np.testing.assert_allclose(a.coef, b.coef, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(a.loglike, b.loglike, rtol=2e-9, atol=2e-7)
    np.testing.assert_allclose(a.deviance, b.deviance, rtol=2e-8, atol=2e-7)


def test_ppml_main_scalars_match_ppmlhdfe_definitions():
    y, X, fe = _ppml_data(seed=4803)
    r = ppmlhdfe(
        y, X, absorb=fe, vce="model",
        config=PPMLConfig(standardize=True, separation=(), tolerance=1e-9, target_inner_tol=1e-10),
    )
    keep = r.sample_mask
    ys = y[keep]
    mu = r.mu[keep]
    ll = float(np.sum(ys * np.log(mu) - mu - gammaln(ys + 1)))
    term = np.empty_like(ys, dtype=float)
    pos = ys > 0
    term[pos] = ys[pos] * np.log(ys[pos] / mu[pos]) - (ys[pos] - mu[pos])
    term[~pos] = mu[~pos]
    dev = float(2 * np.sum(term))
    mu0 = float(np.mean(ys))
    ll0 = float(np.sum(ys * math.log(mu0) - mu0 - gammaln(ys + 1)))
    idx = [i for i, name in enumerate(r.names) if name != "_cons"]
    V = r.vcov[np.ix_(idx, idx)]
    beta = r.coef[idx]
    chi2 = float(beta @ np.linalg.solve(V, beta))
    p = float(chi2_dist.sf(chi2, len(idx)))
    np.testing.assert_allclose(
        [r.loglike, r.deviance, r.loglike_null, r.pseudo_r2, r.chi2, r.chi2_pvalue],
        [ll, dev, ll0, 1 - ll / ll0, chi2, p], rtol=2e-8, atol=2e-7,
    )
    assert r.df_model == len(idx)
    assert r.nobs_full == len(y)
    assert r.vcov_rank == np.linalg.matrix_rank(r.vcov)
    stats = r.model_stats()
    for key in ("loglike_null", "pseudo_r2", "chi2", "chi2_pvalue", "df_model", "nobs_full"):
        assert key in stats
