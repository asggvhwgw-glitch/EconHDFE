from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from econhdfe import (
    ExecutionConfig,
    IVHDFESession,
    IVSpec,
    OLSHDFESession,
    OLSSpec,
    ivhdfe,
    olshdfe,
)


def _data(seed=123, n=6000):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "firm": rng.integers(0, 120, n),
        "year": rng.integers(0, 20, n),
        "industry": rng.integers(0, 25, n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "z1": rng.normal(size=n),
        "z2": rng.normal(size=n),
    })
    df["endog"] = 0.75 * df.z1 + 0.25 * df.z2 + 0.15 * df.x1 + rng.normal(size=n)
    firm_fe = rng.normal(scale=0.5, size=120)[df.firm.to_numpy()]
    year_fe = rng.normal(scale=0.3, size=20)[df.year.to_numpy()]
    df["y1"] = 0.7 * df.x1 - 0.2 * df.x2 + 0.55 * df.endog + firm_fe + year_fe + rng.normal(size=n)
    df["y2"] = -0.3 * df.x1 + 0.4 * df.x3 + firm_fe + rng.normal(size=n)
    return df


def _assert_same(a, b, atol=2e-10):
    assert a.names == b.names
    np.testing.assert_allclose(a.params, b.params, atol=atol, rtol=atol)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=atol, rtol=atol)
    assert a.nobs == b.nobs
    assert a.df_absorbed == b.df_absorbed
    assert a.cluster_counts == b.cluster_counts
    for field in ("r2", "r2_within", "r2_adjusted", "r2_adjusted_within", "rss",
                  "tss", "tss_within", "mss", "rmse", "loglike", "loglike_null",
                  "f_statistic", "f_pvalue", "df_model", "df_resid_fit", "vcov_rank"):
        left, right = getattr(a, field), getattr(b, field)
        if left is None or right is None:
            assert left is right, field
        else:
            np.testing.assert_allclose(left, right, atol=atol, rtol=atol, err_msg=field)


def test_ols_many_y_uses_one_within_batch_and_matches_standalone():
    df = _data()
    s = OLSHDFESession(df, cluster=["firm", "industry"], vce="cluster", execution_config=ExecutionConfig(cache_validation="none"))
    out = s.fit_many_y(y=["y1", "y2"], x=["x1", "x2"], absorb=["firm", "year"])
    for y, got in zip(("y1", "y2"), out, strict=True):
        ref = olshdfe(df, y=y, x=["x1", "x2"], absorb=["firm", "year"], cluster=["firm", "industry"], vce="cluster")
        _assert_same(got, ref)
    info = s.cache_info()
    assert info["fe_entries"] == 1
    assert info["residualize_calls"] == 1
    assert info["residualized_columns"] == 4


def test_ols_add_drop_x_reuses_cached_within_columns():
    df = _data()
    s = OLSHDFESession(df, cluster="firm", vce="cluster", execution_config=ExecutionConfig(cache_validation="none"))
    specs = [
        OLSSpec("y1", ("x1",), ("firm", "year")),
        OLSSpec("y1", ("x1", "x2"), ("firm", "year")),
        OLSSpec("y1", ("x1", "x2", "x3"), ("firm", "year")),
        OLSSpec("y1", ("x1", "x3"), ("firm", "year")),
    ]
    out = s.fit_many(specs)
    for spec, got in zip(specs, out, strict=True):
        ref = olshdfe(df, y=spec.y, x=list(spec.x), absorb=list(spec.absorb), cluster="firm", vce="cluster")
        _assert_same(got, ref)
    info = s.cache_info()
    # y1/x1/x2/x3 are partialled out in a single multi-RHS pass.
    assert info["residualize_calls"] == 1
    assert info["residualized_columns"] == 4


def test_ols_add_drop_fe_keeps_separate_exact_caches_and_reuses_returned_spec():
    df = _data()
    s = OLSHDFESession(df, cluster="firm", vce="cluster", execution_config=ExecutionConfig(cache_validation="none"))
    fe_path = [("firm",), ("firm", "year"), ("firm", "year", "industry"), ("firm", "year")]
    calls_before_return = None
    for j, absorb in enumerate(fe_path):
        got = s.fit(y="y1", x=["x1", "x2"], absorb=absorb)
        ref = olshdfe(df, y="y1", x=["x1", "x2"], absorb=list(absorb), cluster="firm", vce="cluster")
        _assert_same(got, ref)
        if j == 2:
            calls_before_return = s.cache_info()["residualize_calls"]
    info = s.cache_info()
    assert info["fe_entries"] == 3
    # Returning to firm+year is a pure cache hit: no new within transform.
    assert info["residualize_calls"] == calls_before_return


def test_fe_specific_singleton_sample_matches_standalone():
    df = _data(n=1200)
    # Create a level that is singleton only when this extra FE is included.
    df["rare"] = 0
    df.loc[0, "rare"] = 1
    s = OLSHDFESession(df, execution_config=ExecutionConfig(cache_validation="none"))
    a = s.fit(y="y1", x=["x1", "x2"], absorb=["firm"])
    b = s.fit(y="y1", x=["x1", "x2"], absorb=["firm", "rare"])
    ar = olshdfe(df, y="y1", x=["x1", "x2"], absorb=["firm"])
    br = olshdfe(df, y="y1", x=["x1", "x2"], absorb=["firm", "rare"])
    _assert_same(a, ar); _assert_same(b, br)
    assert b.nobs == br.nobs
    assert b.dropped_singletons == br.dropped_singletons


def test_iv_session_matches_standalone_and_reuses_fe_bank():
    df = _data()
    s = IVHDFESession(df, cluster="firm", vce="cluster", execution_config=ExecutionConfig(cache_validation="none"))
    specs = [
        IVSpec("y1", ("x1",), ("endog",), ("z1", "z2"), ("firm", "year")),
        IVSpec("y1", ("x1", "x2"), ("endog",), ("z1", "z2"), ("firm", "year")),
    ]
    out = s.fit_many(specs)
    for spec, got in zip(specs, out, strict=True):
        ref = ivhdfe(
            df, y=spec.y, exog=list(spec.exog), endog=list(spec.endog), instruments=list(spec.instruments),
            absorb=list(spec.absorb), cluster="firm", vce="cluster",
        )
        _assert_same(got, ref, atol=5e-10)
    info = s.cache_info()
    assert info["fe_entries"] == 1
    assert info["residualize_calls"] == 1


def test_signature_validation_invalidates_mutated_fe_source():
    df = _data(n=1500)
    s = OLSHDFESession(df, execution_config=ExecutionConfig(cache_validation="signature"))
    r1 = s.fit(y="y1", x=["x1"], absorb=["firm", "year"])
    misses1 = s.cache_info()["misses"]
    df.loc[:100, "firm"] = 999
    r2 = s.fit(y="y1", x=["x1"], absorb=["firm", "year"])
    ref = olshdfe(df, y="y1", x=["x1"], absorb=["firm", "year"])
    _assert_same(r2, ref)
    assert s.cache_info()["misses"] > misses1
    assert not np.allclose(r1.params, r2.params, atol=0, rtol=0)
