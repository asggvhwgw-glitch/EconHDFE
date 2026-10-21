import numpy as np
import pandas as pd
from econhdfe import olshdfe, FixedEffect


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
