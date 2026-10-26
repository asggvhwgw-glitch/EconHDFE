import numpy as np
import pandas as pd
import pytest
from econhdfe import fv, factor, reg_interaction, ivhdfe
from econhdfe.design import build_design
from econhdfe.errors import SpecificationError


def _df(seed=4901, n=2500):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 180, n)
    year = rng.integers(0, 6, n)
    x = rng.integers(0, 3, n)
    ycat = rng.integers(0, 2, n)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    inst = rng.normal(size=n)
    endog = 0.8 * inst + 0.4 * z + rng.normal(size=n)
    fe = rng.normal(size=180)[firm] + rng.normal(size=6)[year]
    outcome = (
        0.5 * z - 0.2 * w + 0.35 * (x == 1) * z - 0.25 * (x == 2) * z
        + 0.15 * (ycat == 1) * z + 1.1 * endog + fe + rng.normal(scale=0.4, size=n)
    )
    return pd.DataFrame({
        "outcome": outcome, "firm": firm, "year": year, "x": x, "ycat": ycat,
        "z": z, "w": w, "inst": inst, "endog": endog,
    })


def _manual_full():
    return [
        factor("x"), factor("ycat"), "z",
        reg_interaction(factor("x"), factor("ycat")),
        reg_interaction(factor("x"), "z"),
        reg_interaction(factor("ycat"), "z"),
        reg_interaction(factor("x"), factor("ycat"), "z"),
    ]


def test_fv_three_way_full_factorial_matches_manual_design_exactly():
    df = _df(n=500)
    a = build_design(df, _manual_full(), len(df), structural=False)
    b = build_design(df, [fv("i(x)##i(ycat)##c(z)")], len(df), structural=False)
    assert a.names == b.names
    np.testing.assert_array_equal(a.values, b.values)


def test_fv_base_first_last_freq_none_and_exact():
    df = pd.DataFrame({"g": [1, 1, 2, 2, 2, 3], "z": np.arange(6.0)})
    first = build_design(df, fv("i(g)"), len(df), structural=False)
    last = build_design(df, fv("i(g,base=last)"), len(df), structural=False)
    freq = build_design(df, fv("i(g,base=freq)"), len(df), structural=False)
    none = build_design(df, fv("i(g,base=none)"), len(df), structural=False)
    exact = build_design(df, fv("i(g,base=2)"), len(df), structural=False)
    assert first.names == ("g[2]", "g[3]")
    assert last.names == ("g[1]", "g[2]")
    assert freq.names == ("g[1]", "g[3]")
    assert none.names == ("g[1]", "g[2]", "g[3]")
    assert exact.names == ("g[1]", "g[3]")


def test_fv_rejects_dot_style_and_bad_syntax_with_structured_error():
    with pytest.raises(SpecificationError) as e:
        fv("i.x##c(z)")
    assert e.value.code == "design.factorvars.syntax"
    assert "i(x)" in e.value.suggestion
    with pytest.raises(SpecificationError):
        fv("i(x)##")
    with pytest.raises(SpecificationError):
        fv("q(x)")


def test_fv_linear_iv_exog_and_instrument_design_matches_manual():
    df = _df(n=2300)
    exog_manual = [factor("x"), "z", reg_interaction(factor("x"), "z"), "w"]
    exog_fv = [fv("i(x)##c(z)"), "w"]
    inst_manual = ["inst", reg_interaction(factor("ycat"), "inst")]
    inst_fv = ["inst", fv("i(ycat)#c(inst)")]
    a = ivhdfe(df, y="outcome", exog=exog_manual, endog=["endog"], instruments=inst_manual,
               absorb=["firm", "year"], vce="robust", collinearity="drop")
    b = ivhdfe(df, y="outcome", exog=exog_fv, endog=["endog"], instruments=inst_fv,
               absorb=["firm", "year"], vce="robust", collinearity="drop")
    assert a.names == b.names
    np.testing.assert_allclose(a.params, b.params, rtol=0, atol=1e-11)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=0, atol=1e-10)
