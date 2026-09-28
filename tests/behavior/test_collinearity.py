import numpy as np
import pandas as pd
import pytest

from pyreghdfe import reghdfe, ivreghdfe, interaction, factor, reg_interaction


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


def test_auto_solver_uses_canonical_two_way_system():
    df = _df(2)
    r = reghdfe(df, y="y", x=["x1", "x2"],
                absorb=["firm", "year", interaction("city", "year")])
    assert r.absorb_info["canonicalization"]["effective"] == ("firm", "city#year")
    assert r.absorb_info["solver_selection"]["resolved"] == "twoway"
    assert r.absorb_info["method"] == "twoway"


def test_auto_solver_falls_back_to_map_for_three_effective_fes():
    df = _df(3)
    # city is not nested in firm or year in this random design.
    r = reghdfe(df, y="y", x=["x1", "x2"], absorb=["firm", "city", "year"])
    assert r.absorb_info["solver_selection"]["resolved"] == "map"
    assert r.absorb_info["acceleration"] == "cg"


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


def test_factor_term_fully_absorbed_by_interaction_fe_is_reported():
    df = _df(5)
    r = reghdfe(
        df, y="y", x=["x1", factor("year")],
        absorb=["firm", interaction("city", "year")],
    )
    assert r.names == ("x1",)
    omitted = r.collinearity_info["omitted"]
    year_terms = [o for o in omitted if o["name"].startswith("year[")]
    assert year_terms
    assert all(o["reason"] == "absorbed_or_zero" for o in year_terms)


def test_composite_factor_continuous_interaction_keeps_provenance():
    df = _df(6)
    term = reg_interaction(factor("city"), "x1", name="city_x1")
    r = reghdfe(df, y="y", x=["x2", term], absorb=["firm", "year"])
    assert r.names[0] == "x2"
    assert any("#x1" in n for n in r.names[1:])
    assert len(r.names) == len(r.params)


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


def test_nested_factor_frontend_collinearity_is_resolved_after_expansion():
    df = _df(9)
    df["province"] = (df["city"] // 10).astype(int)
    r = reghdfe(
        df, y="y", x=[factor("city"), factor("province")], absorb=["year"],
    )
    info = r.collinearity_info
    omitted = info["omitted"]
    province = [o for o in omitted if o["name"].startswith("province[")]
    assert province
    assert all(o["reason"] == "linear_combination" for o in province)


def test_continuous_interaction_duplicate_is_traced_to_composite_term():
    df = _df(10)
    df["prod"] = df["x1"] * df["x2"]
    r = reghdfe(
        df, y="y", x=[reg_interaction("x1", "x2", name="x1_x2"), "prod"],
        absorb=["firm", "year"],
    )
    assert len(r.names) == 1
    omitted = r.collinearity_info["omitted"]
    assert r.collinearity_info["columns"][0]["term"] == "x1_x2"
    assert omitted[0]["name"] == "prod"
    assert omitted[0]["reason"] == "duplicate_or_scaled"


def test_near_collinearity_tolerance_is_explicit_and_user_controllable():
    df = _df(11)
    rng = np.random.default_rng(11)
    df["xnear"] = df["x1"] + 1e-5*rng.normal(size=len(df))
    r = reghdfe(df, y="y", x=["x1", "xnear"], absorb=["firm", "year"], collinear_tol=1e-4)
    assert r.names == ("x1",)
    r2 = reghdfe(df, y="y", x=["x1", "xnear"], absorb=["firm", "year"], collinear_tol=1e-7)
    assert len(r2.names) == 2


def test_iv_factor_interaction_joint_collinearity_after_absorbing_component_fe():
    df = _df(12, n=9000)
    rng = np.random.default_rng(1200)
    df["qob"] = rng.integers(1, 5, len(df))
    # Full qob x year cell indicators: after absorbing year FE, each year's
    # four cell columns have one exact linear relation. The resolver must drop
    # the joint redundancy rather than relying only on zero-column checks.
    inst = reg_interaction(
        factor("qob", drop_base=False), factor("year", drop_base=False),
        name="qob_year",
    )
    r = ivreghdfe(
        df, y="y", exog=["x1", "x2"], endog=["endog"], instruments=[inst],
        absorb=["firm", "year"], vce="robust",
    )
    zi = r.collinearity_info["excluded_instruments"]
    assert len(zi["requested"]) == 32
    assert len(zi["active"]) == 24
    assert len(zi["omitted"]) == 8
    assert all(o["reason"] in {"linear_combination", "duplicate_or_scaled"}
               for o in zi["omitted"])
    assert np.isfinite(r.diagnostics["cragg_donald_f"])
