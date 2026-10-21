import numpy as np
import pandas as pd
import pytest
from pyreghdfe import reghdfe, ivreghdfe, interaction, factor


def _df(seed=1, n=6000):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 300, n)
    year = rng.integers(0, 8, n)
    city = rng.integers(0, 40, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.normal(size=n)
    fe = rng.normal(size=300)[firm] + rng.normal(size=(40, 8))[city, year]
    endog = .7*z + .4*x1 + rng.normal(size=n)
    y = 1.2*x1 - .5*x2 + 1.5*endog + fe + rng.normal(scale=.4, size=n)
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2, "xsum": x1+x2,
        "xdup": 2*x1, "z": z, "zdup": -3*z, "endog": endog,
        "firm": firm, "year": year, "city": city,
    })


def test_ols_order_preserving_composite_collinearity():
    df = _df(4)
    r = reghdfe(df, y="y", x=["x1", "x2", "xsum", "xdup"], absorb=["firm", "year"])
    assert r.names == ("x1", "x2")
    omitted = r.collinearity_info["omitted"]
    by_name = {o["name"]: o for o in omitted}
    assert set(by_name) == {"xsum", "xdup"}
    assert by_name["xsum"]["reason"] == "linear_combination"
    assert by_name["xdup"]["reason"] == "duplicate_or_scaled"
    assert by_name["xdup"]["dependent_on"] == ("x1",)


def test_iv_drops_absorbed_and_duplicate_excluded_instruments():
    df = _df(7)
    r = ivreghdfe(
        df, y="y", exog=["x1", "x2"], endog=["endog"],
        instruments=["z", "zdup", factor("year")],
        absorb=["firm", interaction("city", "year")], vce="robust",
    )
    info = r.collinearity_info["excluded_instruments"]
    assert info["active"] == ("z",)
    omitted = {o["name"]: o["reason"] for o in info["omitted"]}
    assert omitted["zdup"] == "duplicate_or_scaled"
    assert any(name.startswith("year[") and reason == "absorbed_or_zero"
               for name, reason in omitted.items())
    assert r.params.shape[0] == 3


def test_collinearity_raise_mode_reports_composite_failure():
    df = _df(8)
    with pytest.raises(np.linalg.LinAlgError, match="xsum"):
        reghdfe(df, y="y", x=["x1", "x2", "xsum"], absorb=["firm", "year"],
                collinearity="raise")
