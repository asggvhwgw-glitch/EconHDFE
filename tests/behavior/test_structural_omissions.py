import numpy as np
import pandas as pd
from pyreghdfe import reghdfe, FixedEffect, factor


def _df(seed=730, n=8000):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 250, n)
    year = rng.integers(0, 8, n)
    city = rng.integers(0, 40, n)
    province = city // 10
    q = rng.integers(0, 4, n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.2*x + rng.normal(size=250)[firm] + rng.normal(size=8)[year] + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, x=x, z=z, firm=firm, year=year, city=city, province=province, q=q))


def test_slope_only_absorb_does_not_structurally_absorb_group_dummies():
    df = _df(7)
    r = reghdfe(
        df, y="y", x=[factor("firm")],
        absorb=[FixedEffect("firm", slopes=("x",), intercept=False), "year"],
        method="map",
    )
    # A slope-only firm FE does not contain firm intercept dummies.
    assert len(r.names) > 0
    structural = r.collinearity_info.get("structural")
    assert structural is not None
    assert not any(o.get("structural_reason") == "absorbed_by_fe" for o in r.collinearity_info["omitted"])


def test_public_structural_collinearity_toggle_preserves_estimates():
    df = _df(11)
    specs = ["x", factor("year")]
    absorb = ["firm", ("city", "year")]
    a = reghdfe(df, y="y", x=specs, absorb=absorb, structural_collinearity=True)
    b = reghdfe(df, y="y", x=specs, absorb=absorb, structural_collinearity=False)
    np.testing.assert_allclose(a.params, b.params, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(a.stderr, b.stderr, atol=1e-10, rtol=1e-10)
    assert a.collinearity_info.get("structural") is not None
    assert b.collinearity_info.get("structural") is None
