import numpy as np
import pandas as pd
import pytest

from econhdfe import ivhdfe, factor, reg_interaction
from econhdfe.design import build_design


def _iv_data(seed=993, groups=6, per_group=360):
    rng = np.random.default_rng(seed)
    n = groups * per_group
    g = np.repeat(np.arange(groups), per_group)
    q = rng.normal(size=n)
    z = rng.normal(size=n)
    shock = rng.normal(scale=0.4, size=groups)[g]
    x = 0.7 * z + 0.25 * q + shock + rng.normal(size=n)
    y = np.linspace(0.15, 0.65, groups)[g] * x + 0.25 * q + shock + rng.normal(size=n)
    cluster = np.arange(n) // 45
    return pd.DataFrame({"y": y, "g": g, "q": q, "x": x, "z": z, "cluster": cluster})


def _dense_iv_reference_frame(df):
    n = len(df)
    end = [reg_interaction(factor("g", drop_base=False), "x", name="gx")]
    inst = [reg_interaction(factor("g", drop_base=False), "z", name="gz")]
    e = build_design(df, end, n, structural=False)
    i = build_design(df, inst, n, structural=False)
    out = df.copy()
    en = []
    zn = []
    for j in range(e.values.shape[1]):
        name = f"E{j}"
        out[name] = e.values[:, j]
        en.append(name)
    for j in range(i.values.shape[1]):
        name = f"Z{j}"
        out[name] = i.values[:, j]
        zn.append(name)
    return out, end, inst, en, zn


@pytest.mark.parametrize("estimator", ["2sls", "liml"])
@pytest.mark.parametrize("vce", ["robust", "cluster"])
def test_linear_iv_heterogeneous_spec_matches_dense(estimator, vce):
    df = _iv_data()
    dense, end, inst, dense_end, dense_inst = _dense_iv_reference_frame(df)
    extra = {"cluster": "cluster"} if vce == "cluster" else {}
    got = ivhdfe(
        df, y="y", exog=["q"], endog=end, instruments=inst, absorb=["g"],
        vce=vce, estimator=estimator, collinearity="drop", **extra,
    )
    ref = ivhdfe(
        dense, y="y", exog=["q"], endog=dense_end, instruments=dense_inst,
        absorb=["g"], vce=vce, estimator=estimator, collinearity="drop", **extra,
    )
    assert got.diagnostics.get("heterogeneous_spec_path") is True
    np.testing.assert_allclose(got.params, ref.params, rtol=0, atol=5e-12)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=0, atol=5e-11)

from econhdfe import IVPPMLConfig, IVPPMLHDFE


def _ivppml_data(seed=9202, groups=5, per_group=360):
    rng = np.random.default_rng(seed)
    n = groups * per_group
    g = np.repeat(np.arange(groups), per_group)
    pos = np.tile(np.arange(per_group), groups)
    c = rng.normal(size=n)
    z = rng.normal(size=n)
    e = 0.8 * z + 0.2 * c + rng.normal(scale=0.5, size=n)
    mu = np.exp(
        rng.normal(scale=0.15, size=groups)[g]
        + 0.06 * c + np.linspace(0.12, 0.32, groups)[g] * e
    )
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "g": g, "c": c, "e": e, "z": z, "cluster": pos % 37})


@pytest.mark.parametrize("standardize", [False, True])
def test_ivppml_heterogeneous_spec_matches_dense_clustered(standardize):
    df = _ivppml_data()
    n = len(df)
    end = [reg_interaction(factor("g", drop_base=False), "e", name="ge")]
    inst = [reg_interaction(factor("g", drop_base=False), "z", name="gz")]
    ed = build_design(df, end, n, structural=False)
    zd = build_design(df, inst, n, structural=False)
    dense = df.copy()
    en = []
    zn = []
    for j in range(ed.values.shape[1]):
        name = f"E{j}"
        dense[name] = ed.values[:, j]
        en.append(name)
    for j in range(zd.values.shape[1]):
        name = f"Z{j}"
        dense[name] = zd.values[:, j]
        zn.append(name)
    cfg = IVPPMLConfig(
        separation=(), standardize=standardize, tolerance=1e-9,
        target_inner_tol=1e-10, max_iter=300, engine="optimized",
    )
    got = IVPPMLHDFE(df, absorb=["g"], config=cfg).fit(
        "y", exog=["c"], endog=end, instruments=inst,
        vce="cluster", clusters="cluster",
    )
    ref = IVPPMLHDFE(dense, absorb=["g"], config=cfg).fit(
        "y", exog=["c"], endog=en, instruments=zn,
        vce="cluster", clusters="cluster",
    )
    assert got.diagnostics.get("heterogeneous_spec_path") is True
    assert got.iterations == ref.iterations
    np.testing.assert_allclose(got.coef, ref.coef, rtol=0, atol=5e-11)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=0, atol=5e-10)
    np.testing.assert_allclose(got.moments, ref.moments, rtol=0, atol=5e-10)

from econhdfe import olshdfe, PPMLConfig, PPMLHDFE


def test_ols_heterogeneous_spec_matches_dense():
    rng = np.random.default_rng(9301)
    groups, per_group = 6, 280
    n = groups * per_group
    g = np.repeat(np.arange(groups), per_group)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.linspace(-0.3, 0.5, groups)[g] * x + 0.2 * z + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "g": g, "x": x, "z": z})
    spec = [reg_interaction(factor("g", drop_base=False), "x", name="gx"), "z"]
    dense_design = build_design(df, spec, n, structural=False)
    dense = df.copy()
    cols = []
    for j in range(dense_design.values.shape[1]):
        name = f"X{j}"
        dense[name] = dense_design.values[:, j]
        cols.append(name)
    got = olshdfe(df, y="y", x=spec, absorb=["g"], vce="robust", collinearity="drop")
    ref = olshdfe(dense, y="y", x=cols, absorb=["g"], vce="robust", collinearity="drop")
    assert got.absorb_info.get("heterogeneous_spec") is not None
    np.testing.assert_allclose(got.params, ref.params, rtol=0, atol=2e-12)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=0, atol=2e-11)


def test_ppml_heterogeneous_spec_matches_dense():
    rng = np.random.default_rng(9302)
    groups, per_group = 6, 320
    n = groups * per_group
    g = np.repeat(np.arange(groups), per_group)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    mu = np.exp(rng.normal(scale=0.18, size=groups)[g] + np.linspace(-0.15, 0.2, groups)[g] * x + 0.08 * z)
    y = rng.poisson(mu)
    df = pd.DataFrame({"y": y, "g": g, "x": x, "z": z})
    spec = [reg_interaction(factor("g", drop_base=False), "x", name="gx"), "z"]
    dense_design = build_design(df, spec, n, structural=False)
    dense = df.copy()
    cols = []
    for j in range(dense_design.values.shape[1]):
        name = f"X{j}"
        dense[name] = dense_design.values[:, j]
        cols.append(name)
    cfg = PPMLConfig(separation=(), standardize=False, tolerance=1e-9, target_inner_tol=1e-10)
    got = PPMLHDFE(df, absorb=["g"], config=cfg).fit("y", X=spec, vce="robust")
    ref = PPMLHDFE(dense, absorb=["g"], config=cfg).fit("y", X=cols, vce="robust")
    assert got.diagnostics.get("heterogeneous_spec_path") is True
    np.testing.assert_allclose(got.coef, ref.coef, rtol=0, atol=5e-11)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=0, atol=5e-10)
