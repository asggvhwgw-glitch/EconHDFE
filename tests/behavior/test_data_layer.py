from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from econhdfe.data import DataFrameSource, CSVSource, materialize_required_data
from econhdfe import factor


def test_materialization_encodes_identifier_only_columns_without_touching_factor_columns():
    from econhdfe.frontend.columns import compile_data_requirements
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "firm": ["b", "a", "b", "c"],
        "industry": ["z", "y", "z", "x"],
    })
    req = compile_data_requirements(y="y", x=factor("industry"), absorb=["firm", "industry"])
    out = materialize_required_data(
        DataFrameSource(df), req.columns, memory_budget_mb=64,
        identifier_columns=req.identifier_only_columns,
    )
    assert req.identifier_only_columns == ("firm",)
    assert out.frame["firm"].dtype == np.int32
    assert out.identifier_levels["firm"] == ("b", "a", "c")
    assert out.frame["industry"].tolist() == df["industry"].tolist()
    # Explicit materialization must not mutate the caller's DataFrame.
    assert df["firm"].tolist() == ["b", "a", "b", "c"]


def test_csv_source_string_fe_is_preencoded_with_ols_parity(tmp_path: Path):
    from econhdfe import olshdfe, ExecutionConfig
    rng = np.random.default_rng(123)
    n = 900
    firm = np.array([f"f{i % 90}" for i in range(n)], dtype=object)
    year = np.array([f"y{i % 6}" for i in range(n)], dtype=object)
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "firm": firm, "year": year, "junk": "payload"})
    path = tmp_path / "strings.csv"
    df.to_csv(path, index=False)
    ref = olshdfe(df, y="y", x=["x"], absorb=["firm", "year"], drop_singletons=False)
    got = olshdfe(
        CSVSource(path), y="y", x=["x"], absorb=["firm", "year"], drop_singletons=False,
        execution_config=ExecutionConfig(profile="summary", memory_budget_mb=64),
    )
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-10, atol=1e-10)
    assert got.profile["data_ingestion"]["encoded_identifier_count"] == 2


def test_persistent_session_completed_ols_result_resumes_without_residualization(tmp_path, monkeypatch):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(8802)
    n = 1600
    df = pd.DataFrame({
        "y": rng.normal(size=n), "x": rng.normal(size=n),
        "firm": np.arange(n) % 160, "year": np.arange(n) % 10,
    })
    path = tmp_path / "resume.csv"; df.to_csv(path, index=False)
    cache = tmp_path / "cache"
    a = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    ref = a.fit(y="y", x=["x"], absorb=["firm", "year"])
    b = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    got = b.fit(y="y", x=["x"], absorb=["firm", "year"])
    np.testing.assert_allclose(got.params, ref.params, rtol=0, atol=0)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=0, atol=0)
    assert got.prediction_state.coefficient_names == ref.prediction_state.coefficient_names
    assert got.prediction_state.design("regressor").active_names == ref.names
    np.testing.assert_array_equal(
        got.prediction_state.sample.mask(), ref.prediction_state.sample.mask()
    )
    assert b.persistent_cache_info()["result_hits"] == 1
    assert b.cache_info()["residualize_calls"] == 0

    # Pre-0.8.1 caches must not bypass the scale-aware absorption correction.
    from econhdfe.data import persistent
    legacy_cache = tmp_path / "legacy-cache"
    with monkeypatch.context() as patch:
        patch.setattr(persistent, "PERSISTENT_NUMERICAL_ABI", "linear-session-1")
        legacy = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(legacy_cache)
        legacy.fit(y="y", x=["x"], absorb=["firm", "year"])
    fresh = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(legacy_cache)
    recomputed = fresh.fit(y="y", x=["x"], absorb=["firm", "year"])
    info = fresh.persistent_cache_info()
    assert info["column_hits"] == 4
    assert info["result_hits"] == info["within_hits"] == 0
    assert fresh.cache_info()["residualized_columns"] == 2
    np.testing.assert_allclose(recomputed.params, ref.params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(recomputed.vcov, ref.vcov, rtol=1e-10, atol=1e-11)


def test_persistent_session_source_content_change_invalidates_cache(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(8803)
    n = 1200
    df = pd.DataFrame({
        "y": rng.normal(size=n), "x": rng.normal(size=n),
        "firm": np.arange(n) % 120, "year": np.arange(n) % 10,
    })
    path = tmp_path / "changing.csv"; df.to_csv(path, index=False)
    cache = tmp_path / "cache"
    old = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    old_result = old.fit(y="y", x=["x"], absorb=["firm", "year"])

    changed = df.copy(); changed["y"] = changed["y"] + 0.75 * changed["x"]
    changed.to_csv(path, index=False)
    fresh = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    got = fresh.fit(y="y", x=["x"], absorb=["firm", "year"])
    ref = OLSHDFESession(changed, drop_singletons=False).fit(y="y", x=["x"], absorb=["firm", "year"])
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-11, atol=1e-11)
    assert fresh.persistent_cache_info()["result_hits"] == 0
    assert fresh.persistent_cache_info()["column_hits"] == 0
    assert not np.allclose(got.params, old_result.params)


def test_persistent_result_roundtrip_preserves_packed_prediction_sample(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource

    rng = np.random.default_rng(8807)
    firm = np.r_[np.repeat(np.arange(40), 4), 40]
    n = len(firm)
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=41)[firm] + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"y": y, "x": x, "firm": firm})
    path = tmp_path / "packed.csv"
    df.to_csv(path, index=False)
    cache = tmp_path / "cache"

    first = OLSHDFESession(CSVSource(path)).enable_persistent_cache(cache)
    ref = first.fit(y="y", x=["x"], absorb=["firm"])
    assert ref.prediction_state.sample.packed_mask is not None

    second = OLSHDFESession(CSVSource(path)).enable_persistent_cache(cache)
    got = second.fit(y="y", x=["x"], absorb=["firm"])
    assert second.persistent_cache_info()["result_hits"] == 1
    assert got.prediction_state.sample.packed_mask == ref.prediction_state.sample.packed_mask
    np.testing.assert_array_equal(
        got.prediction_state.sample.mask(), ref.prediction_state.sample.mask()
    )
