import numpy as np
import pandas as pd

from pyreghdfe import reghdfe, FixedEffect, factor, reg_interaction
from pyreghdfe.design import build_design


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


def test_structural_factor_block_absorbed_before_materialization():
    df = _df(1)
    r = reghdfe(df, y="y", x=["x", factor("year")], absorb=["firm", ("city", "year")])
    s = r.collinearity_info["structural"]
    assert s["requested_columns"] == 1 + 7
    assert s["materialized_columns"] == 1
    assert all(o["structural"] for o in r.collinearity_info["omitted"] if o["name"].startswith("year["))
    assert all(o["structural_reason"] == "absorbed_by_fe" for o in r.collinearity_info["omitted"] if o["name"].startswith("year["))


def test_continuous_main_effect_reduces_later_full_factor_slope_block():
    df = _df(2)
    term = reg_interaction(factor("q", drop_base=False), "x", name="q_x")
    d = build_design(df, ["x", term], len(df), structural=True)
    # x + four q-specific slopes has one exact relation: x=sum_q 1[q]*x.
    assert len(d.requested_names) == 5
    assert d.values.shape[1] == 4
    omitted = d.structural_plan.omissions
    assert len(omitted) == 1
    assert omitted[0].reason == "structural_linear_combination"
    assert omitted[0].dependent_on == ("x",)


def test_full_factor_slope_first_structurally_spans_later_main_effect():
    df = _df(3)
    term = reg_interaction(factor("q", drop_base=False), "x", name="q_x")
    d = build_design(df, [term, "x"], len(df), structural=True)
    assert len(d.requested_names) == 5
    assert d.values.shape[1] == 4
    omitted = d.structural_plan.omissions
    assert len(omitted) == 1
    assert omitted[0].name == "x"
    assert omitted[0].reason == "structurally_spanned"
    assert omitted[0].dependent_on == ("q_x",)


def test_reference_coded_factor_slope_does_not_span_main_effect():
    df = _df(4)
    term = reg_interaction(factor("q"), "x", name="q_x")
    d = build_design(df, ["x", term], len(df), structural=True)
    # q has one omitted base, so the three interaction slopes do not sum to x.
    assert len(d.requested_names) == 4
    assert d.values.shape[1] == 4
    assert not d.structural_plan.omissions


def test_full_factorial_reduces_against_earlier_main_effect_block():
    df = _df(5)
    main = factor("year", drop_base=False)
    cells = reg_interaction(
        factor("q", drop_base=False), factor("year", drop_base=False), name="q_year"
    )
    d = build_design(df, [main, cells], len(df), structural=True)
    # 8 year dummies + 32 full cells contain 8 exact relations.
    assert len(d.requested_names) == 40
    assert d.values.shape[1] == 32
    assert len(d.structural_plan.omissions) == 8
    assert all(o.dependent_on == ("year",) for o in d.structural_plan.omissions)


def test_nested_factor_order_preserves_earlier_coarse_basis():
    df = _df(6)
    d = build_design(df, [factor("province"), factor("city")], len(df), structural=True)
    # province contributes 3 columns. City contributes 39 reference-coded
    # columns, but one city column in each non-base province is redundant.
    assert len(d.requested_names) == 42
    assert d.values.shape[1] == 39
    omitted = d.structural_plan.omissions
    assert len(omitted) == 3
    assert all(o.dependent_on == ("province",) for o in omitted)


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


def test_structural_and_unpruned_design_have_same_column_space():
    df = _df(8)
    main = factor("year", drop_base=False)
    cells = reg_interaction(
        factor("q", drop_base=False), factor("year", drop_base=False), name="q_year"
    )
    ds = build_design(df, [main, cells], len(df), structural=True)
    dfull = build_design(df, [main, cells], len(df), structural=False)
    rng = np.random.default_rng(8)
    target = rng.normal(size=len(df))
    fs = ds.values @ np.linalg.lstsq(ds.values, target, rcond=None)[0]
    ff = dfull.values @ np.linalg.lstsq(dfull.values, target, rcond=None)[0]
    np.testing.assert_allclose(fs, ff, atol=2e-10, rtol=2e-10)


def test_collinearity_raise_also_catches_structural_omissions():
    df = _df(9)
    import pytest
    with pytest.raises(np.linalg.LinAlgError, match="year"):
        reghdfe(
            df, y="y", x=["x", factor("year")],
            absorb=["firm", ("city", "year")], collinearity="raise",
        )


def test_iv_style_protected_exog_prunes_redundant_factor_slope_column():
    df = _df(10)
    c = build_design(df, ["x"], len(df), structural=True)
    zterm = reg_interaction(factor("q", drop_base=False), "x", name="q_x")
    z = build_design(
        df, [zterm], len(df), structural=True, protected_terms=c.structural_terms,
    )
    assert len(z.requested_names) == 4
    assert z.values.shape[1] == 3
    assert len(z.structural_plan.omissions) == 1
    assert z.structural_plan.omissions[0].dependent_on == ("x",)


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
