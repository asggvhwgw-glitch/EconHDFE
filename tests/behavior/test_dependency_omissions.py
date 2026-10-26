import numpy as np
import pandas as pd
import pytest
from econhdfe import reghdfe, factor, OmittedVariableWarning


def _hierarchy_df(seed=740, n=12000):
    rng = np.random.default_rng(seed)
    city = rng.integers(0, 300, n)
    province = city // 10
    region = province // 6
    year = rng.integers(0, 8, n)
    firm = rng.integers(0, 700, n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 0.8 * z + 0.3 * u + rng.normal(size=n)
    y = 1.1 * x + 1.5 * endog + rng.normal(size=700)[firm] + rng.normal(size=n)
    return pd.DataFrame(dict(
        y=y, x=x, z=z, endog=endog, firm=firm, year=year,
        city=city, province=province, region=region,
    ))


def test_default_collinearity_warns_instead_of_silent_drop():
    df = _hierarchy_df(3)
    with pytest.warns(OmittedVariableWarning, match="omitted"):
        r = reghdfe(
            df, y="y", x=["x", factor("year")],
            absorb=["firm", ("city", "year")],
        )
    assert r.has_omitted_variables
    assert any(v["name"].startswith("year[") for v in r.omitted_variables)
