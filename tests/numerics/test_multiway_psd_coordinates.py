"""Multiway-cluster PSD repair in reghdfe's original-regressor stdev coordinates.

The reference covariances were produced with Stata 18 MP, reghdfe 6.12.5:
``reghdfe y x1 x2 x3, absorb(fe) vce(cluster c1 c2)`` on the data generated
by ``_data(seed)`` (n=240, 5x4 clusters, regressor scales 1, 1e3, 1e-2). All
three seeds have an indefinite CGM sandwich, so every reference is repaired.
reghdfe reports ``_cons`` and replaces its slope block by the separately
clipped slope block, which is what econhdfe (no ``_cons``) must reproduce.
"""
import numpy as np
import pandas as pd
import pytest

import econhdfe
from econhdfe import HDFEConfig, IVPPMLConfig, PPMLConfig
from econhdfe.compute.vcov import _fix_psd, finalize_cluster_vcov, regressor_scale

REGHDFE_SLOPE_V = {
    7: [[0.0574802636181467, -3.18432109641e-05, -0.763156692137708],
        [-3.18432109641e-05, 3.14145973574e-08, -0.0002301980084104],
        [-0.763156692137708, -0.0002301980084104, 41.08766961756148]],
    11: [[0.0362312355093565, 1.46818522784e-05, -0.8366327560120621],
         [1.46818522784e-05, 5.94947379778e-09, -0.000339025660106],
         [-0.8366327560120621, -0.000339025660106, 19.3190863792536]],
    23: [[0.1376302691631043, -3.80580295992e-05, -3.558729794342499],
         [-3.80580295992e-05, 2.1653615891e-08, 0.0012144201635576],
         [-3.558729794342499, 0.0012144201635576, 96.78611205337224]],
}


def _data(seed):
    rng = np.random.default_rng(seed)
    n = 240
    c1 = rng.integers(0, 5, n); c2 = rng.integers(0, 4, n); fe = rng.integers(0, 12, n)
    x1 = rng.normal(size=n) + 0.8 * (c1 == 0)
    x2 = 1000.0 * (rng.normal(size=n) + 0.5 * x1)
    x3 = 0.01 * (rng.normal(size=n) - 0.7 * (c2 == 1))
    u = rng.normal(size=n) + 1.5 * rng.normal(size=5)[c1] - 1.2 * rng.normal(size=4)[c2]
    y = 0.3 * x1 + 0.0004 * x2 - 20 * x3 + 0.2 * rng.normal(size=12)[fe] + u
    cnt = rng.poisson(np.exp(0.2 + 0.25 * x1 + 0.0002 * x2 - 10 * x3 + 0.3 * rng.normal(size=12)[fe]))
    z = x1 + 0.5 * rng.normal(size=n)
    return pd.DataFrame(dict(y=y, cnt=cnt, x1=x1, x2=x2, x3=x3, z=z, fe=fe, c1=c1, c2=c2))


def _repaired(result, X):
    """True when the slope VCE is clipped (singular) in the stdev coordinates of X."""
    X = np.asarray(X, dtype=float)
    mask = getattr(result, "sample_mask", None)
    s = regressor_scale(X if mask is None else X[np.asarray(mask, bool)])
    k = len(s)
    eig = np.linalg.eigvalsh(np.asarray(result.vcov)[:k, :k] * np.outer(s, s))
    return eig[0] <= 1e-12 * eig[-1]


@pytest.mark.parametrize("seed", sorted(REGHDFE_SLOPE_V))
def test_ols_multiway_repair_matches_reghdfe(seed):
    df = _data(seed)
    r = econhdfe.olshdfe(df, y="y", x=["x1", "x2", "x3"], absorb=["fe"], cluster=["c1", "c2"], vce="cluster")
    ref = np.asarray(REGHDFE_SLOPE_V[seed])
    assert _repaired(r, df[["x1", "x2", "x3"]])
    assert np.max(np.abs(r.vcov - ref)) / np.max(np.abs(ref)) <= 1e-6
    np.testing.assert_allclose(r.stderr, np.sqrt(np.diag(ref)), rtol=1e-6)


def test_finalize_without_rescaling_is_the_cgm_clip_and_constant_branch_follows_reghdfe():
    rng = np.random.default_rng(5)
    for _ in range(25):
        A = rng.normal(size=(4, 4))
        V = A @ A.T - 1.5 * np.outer(A[0], A[0])
        clusters = [np.zeros(3), np.zeros(3)]
        np.testing.assert_array_equal(
            finalize_cluster_vcov(V, "cluster", clusters, lambda: np.ones(4)), _fix_psd(V))
        assert finalize_cluster_vcov(V, "cluster", clusters[:1], lambda: 1 / 0) is V
        assert finalize_cluster_vcov(V, "robust", clusters, lambda: 1 / 0) is V
        s = rng.uniform(0.1, 10.0, 4)
        got = finalize_cluster_vcov(V, "cluster", clusters, lambda: s, constant_index=3)
        W = V * np.outer(s[:3].tolist() + [1.0], s[:3].tolist() + [1.0])
        expected = _fix_psd(W)
        if np.linalg.eigvalsh(W)[0] < 0:
            expected[:3, :3] = _fix_psd(W[:3, :3])
        np.testing.assert_allclose(got * np.outer(np.r_[s[:3], 1.0], np.r_[s[:3], 1.0]), expected,
                                   rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def _scaled_pair(fit, column, factor=1000.0):
    df = _data(11)
    base = fit(df)
    scaled_df = df.copy()
    scaled_df[column] = scaled_df[column] * factor
    return base, fit(scaled_df)


def _ols(df, mode="stata_compat", **kw):
    return econhdfe.olshdfe(df, y="y", x=["x1", "x2", "x3"], absorb=["fe"], cluster=["c1", "c2"], vce="cluster",
                            hdfe_config=HDFEConfig(singleton_mode=mode), **kw)


def _iv(df, mode="stata_compat", **kw):
    return econhdfe.ivhdfe(df, y="y", exog=["x2", "x3"], endog=["x1"], instruments=["z"], absorb=["fe"],
                           cluster=["c1", "c2"], vce="cluster", hdfe_config=HDFEConfig(singleton_mode=mode), **kw)


def _ppml(df, mode="stata_compat", **kw):
    return econhdfe.ppmlhdfe(df, y="cnt", x=["x1", "x2", "x3"], absorb=["fe"], cluster=["c1", "c2"], vce="cluster",
                             config=PPMLConfig(singleton_mode=mode, tolerance=1e-12, target_inner_tol=1e-13), **kw)


def _ivppml(df, mode="stata_compat", **kw):
    return econhdfe.ivppmlhdfe(df, y="cnt", exog=["x2", "x3"], endog=["x1"], instruments=["z"], absorb=["fe"],
                               cluster=["c1", "c2"], vce="cluster",
                               config=IVPPMLConfig(singleton_mode=mode, tolerance=1e-12, target_inner_tol=1e-13),
                               **kw)


@pytest.mark.parametrize("fit,columns", [
    (_ols, ["x1", "x2", "x3"]), (_iv, ["x2", "x3", "x1"]),
    (_ppml, ["x1", "x2", "x3"]), (_ivppml, ["x2", "x3", "x1"]),
], ids=["ols", "iv", "ppml", "ivppml"])
def test_multiway_repair_is_invariant_to_regressor_units(fit, columns):
    base, scaled = _scaled_pair(fit, "x2")
    k = len(columns)
    V0, V1 = np.asarray(base.vcov)[:k, :k], np.asarray(scaled.vcov)[:k, :k]
    assert _repaired(base, _data(11)[columns])
    unit = np.array([1000.0 if c == "x2" else 1.0 for c in columns])
    np.testing.assert_allclose(V1 * np.outer(unit, unit), V0, rtol=1e-10, atol=0)


@pytest.mark.parametrize("fit,columns", [
    (_ols, ["x1", "x2", "x3"]), (_iv, ["x2", "x3", "x1"]),
    (_ppml, ["x1", "x2", "x3"]), (_ivppml, ["x2", "x3", "x1"]),
], ids=["ols", "iv", "ppml", "ivppml"])
def test_unified_frequency_weights_repair_like_expanded_rows(fit, columns):
    """Unified fweight fits clip in the coordinates of the expanded data."""
    df = _data(11)
    df["fw"] = np.random.default_rng(3).integers(1, 4, len(df)).astype(float)
    expanded = df.loc[df.index.repeat(df.fw.astype(int))].reset_index(drop=True)
    compact = fit(df, "unified", weights="fw", weight_type="fw")
    full = fit(expanded, "unified")
    k = len(columns)
    assert _repaired(full, expanded[columns])
    np.testing.assert_allclose(np.asarray(compact.vcov)[:k, :k], np.asarray(full.vcov)[:k, :k], rtol=1e-10, atol=0)
