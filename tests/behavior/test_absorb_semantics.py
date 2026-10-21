import numpy as np
import pandas as pd
import pytest
from pyreghdfe import FixedEffect, HDFEConvergenceError, interaction, reghdfe


def _interaction_fixture(seed=701, n=24000):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 1800, n)
    year = rng.integers(0, 16, n)
    city = rng.integers(0, 120, n)
    city_year = city * 16 + year
    x = rng.normal(size=(n, 3))
    beta = np.array([0.8, -0.45, 0.25])
    y = (
        x @ beta
        + rng.normal(size=1800)[firm]
        + rng.normal(size=120 * 16)[city_year]
        + rng.normal(scale=0.2, size=n)
    )
    df = pd.DataFrame({
        "firm": firm, "year": year, "city": city, "city_year": city_year,
        "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "y": y,
    })
    return df


def test_slope_and_intercept_term_is_not_eliminated_when_intercept_is_nested():
    rng = np.random.default_rng(704)
    n = 16000
    state = rng.integers(0, 20, n)
    zipcode = state * 50 + rng.integers(0, 50, n)
    t = rng.normal(size=n)
    x = rng.normal(size=n)
    y = 0.6*x + rng.normal(size=1000)[zipcode] + rng.normal(size=20)[state]*t + rng.normal(size=n)
    df = pd.DataFrame({"state": state, "zip": zipcode, "t": t, "x": x, "y": y})
    r = reghdfe(df, y="y", x=["x"],
                absorb=[FixedEffect("state", slopes=("t",), intercept=True), "zip"],
                tol=1e-8)
    assert r.absorb_info["canonicalization"]["effective"] == ("state", "zip")


def test_public_api_raises_on_nonconvergence_unless_explicitly_allowed():
    rng = np.random.default_rng(705)
    n = 12000
    # Nearly nested partitions make one unaccelerated MAP iteration deliberately insufficient.
    a = rng.integers(0, 800, n)
    b = a.copy()
    swap = rng.choice(n, size=250, replace=False)
    b[swap] = rng.integers(0, 800, len(swap))
    x = rng.normal(size=n)
    y = 0.9*x + rng.normal(size=800)[a] + rng.normal(size=800)[b] + rng.normal(size=n)
    with pytest.raises(HDFEConvergenceError):
        reghdfe(None, y=y, x=x, absorb=[a, b], acceleration="none", max_iter=1, tol=1e-12)
    r = reghdfe(None, y=y, x=x, absorb=[a, b], acceleration="none", max_iter=1,
                tol=1e-12, allow_nonconverged=True)
    assert not r.converged


def test_twoway_schur_solver_matches_weighted_map_cg():
    df = _interaction_fixture(seed=709, n=15000)
    rng = np.random.default_rng(709)
    df["w"] = 0.2 + rng.random(len(df))
    common = dict(data=df, y="y", x=["x1", "x2"], absorb=["firm", ("city", "year")],
                  weights="w", weight_type="aweight", tol=1e-10)
    a = reghdfe(**common, method="map", acceleration="cg")
    b = reghdfe(**common, method="twoway")
    np.testing.assert_allclose(a.params, b.params, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(a.stderr, b.stderr, atol=3e-8, rtol=3e-8)


def test_twoway_schur_iv_matches_map_cg():
    rng = np.random.default_rng(711)
    n = 18000
    firm = rng.integers(0, 1400, n)
    year = rng.integers(0, 12, n)
    city = rng.integers(0, 90, n)
    cy = city * 12 + year
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    endog = 0.8*z + 0.25*w + rng.normal(size=n)
    y = 1.3*endog + 0.4*w + rng.normal(size=1400)[firm] + rng.normal(size=1080)[cy] + rng.normal(size=n)
    df = pd.DataFrame({"firm":firm,"year":year,"city":city,"z":z,"w":w,"x":endog,"y":y})
    kw = dict(data=df, y="y", exog=["w"], endog=["x"], instruments=["z"],
              absorb=["firm", "year", interaction("city", "year")], tol=1e-10)
    from pyreghdfe import ivreghdfe
    a = ivreghdfe(**kw, method="map", acceleration="cg")
    b = ivreghdfe(**kw, method="twoway")
    np.testing.assert_allclose(a.params, b.params, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(a.stderr, b.stderr, atol=3e-8, rtol=3e-8)
