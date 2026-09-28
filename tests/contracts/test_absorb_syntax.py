import numpy as np
import pandas as pd
import pytest

from econhdfe import fv, olshdfe
from econhdfe.errors import SpecificationError
from econhdfe.hdfe.specs import fe, interaction
from econhdfe.hdfe.factorvars import compile_absorb_factor_variable


def _df(seed=4101, n=3500):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 140, n)
    year = rng.integers(0, 6, n)
    trend = rng.normal(size=n)
    x = rng.normal(size=n)
    firm_int = rng.normal(size=140)[firm]
    firm_slope = rng.normal(scale=0.25, size=140)[firm]
    year_fe = rng.normal(size=6)[year]
    y = 0.7 * x + firm_int + year_fe + firm_slope * trend + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x": x, "firm": firm, "year": year, "trend": trend})


def _assert_result_same(a, b, atol=2e-11):
    assert a.names == b.names
    np.testing.assert_allclose(a.params, b.params, rtol=0, atol=atol)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=0, atol=atol)
    assert a.df_absorbed == b.df_absorbed
    assert a.nobs == b.nobs


def test_fv_absorb_single_factor_matches_plain_absorb():
    df = _df()
    a = olshdfe(df, y="y", x=["x"], absorb=["firm", "year"], vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=[fv("i(firm)"), fv("i(year)")], vce="robust")
    _assert_result_same(a, b)


def test_fv_absorb_categorical_interaction_matches_interaction_helper():
    df = _df(n=2500)
    a = olshdfe(df, y="y", x=["x"], absorb=[interaction("firm", "year")], vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=[fv("i(firm)#i(year)")], vce="robust")
    _assert_result_same(a, b)


def test_fv_absorb_categorical_full_factorial_canonicalizes_to_joint_partition():
    df = _df(n=2500)
    a = olshdfe(df, y="y", x=["x"], absorb=[interaction("firm", "year")], vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=[fv("i(firm)##i(year)")], vce="robust")
    _assert_result_same(a, b)


def test_fv_absorb_slope_only_matches_fe_helper():
    df = _df(n=2800)
    a = olshdfe(df, y="y", x=["x"], absorb=[fe("firm", "trend", intercept=False), "year"], vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=[fv("i(firm)#c(trend)"), "year"], vce="robust")
    _assert_result_same(a, b, atol=5e-11)


def test_fv_absorb_intercept_and_slope_matches_fe_helper():
    df = _df(n=2800)
    a = olshdfe(df, y="y", x=["x"], absorb=[fe("firm", "trend"), "year"], vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=[fv("i(firm)##c(trend)"), "year"], vce="robust")
    _assert_result_same(a, b, atol=5e-11)


def test_fv_absorb_three_way_full_factorial_reduces_to_joint_intercept_slope():
    df = _df(n=2600)
    manual = [fe(interaction("firm", "year"), "trend")]
    a = olshdfe(df, y="y", x=["x"], absorb=manual, vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=[fv("i(firm)##i(year)##c(trend)")], vce="robust")
    _assert_result_same(a, b, atol=8e-11)


def test_fv_absorb_compiler_merges_multiple_slopes_on_same_partition():
    expr = fv("i(firm)##(c(trend)+c(x))")
    specs = compile_absorb_factor_variable(expr)
    assert len(specs) == 1
    assert specs[0].group == "firm"
    assert specs[0].intercept is True
    assert specs[0].slopes == ("trend", "x")


def test_fv_absorb_rejects_pure_continuous_and_continuous_products():
    with pytest.raises(SpecificationError) as e1:
        compile_absorb_factor_variable(fv("c(trend)"))
    assert e1.value.code == "absorb.factorvars.pure_continuous"

    with pytest.raises(SpecificationError) as e2:
        compile_absorb_factor_variable(fv("i(firm)#c(trend)#c(x)"))
    assert e2.value.code == "absorb.factorvars.continuous_product"


def test_fv_absorb_rejects_nondefault_base_selector():
    with pytest.raises(SpecificationError) as e:
        compile_absorb_factor_variable(fv("i(firm,base=last)"))
    assert e.value.code == "absorb.factorvars.base_selector"


def test_fv_absorb_scalar_expression_is_accepted():
    df = _df(n=1800)
    a = olshdfe(df, y="y", x=["x"], absorb="firm", vce="robust")
    b = olshdfe(df, y="y", x=["x"], absorb=fv("i(firm)"), vce="robust")
    _assert_result_same(a, b)


def test_repeated_session_accepts_categorical_fv_absorb_and_rejects_slope_fv():
    from econhdfe import OLSHDFESession
    df = _df(n=1800)
    session = OLSHDFESession(df)
    a = session.fit(y="y", x=["x"], absorb=[("firm", "year")])
    b = session.fit(y="y", x=["x"], absorb=fv("i(firm)#i(year)"))
    _assert_result_same(a, b)
    with pytest.raises(SpecificationError) as e:
        session.fit(y="y", x=["x"], absorb=fv("i(firm)##c(trend)"))
    assert e.value.code == "specification.session_absorb_slope"
