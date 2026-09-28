from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from econhdfe.data import (
    DataFrameSource, CSVSource, StataSource, StableCategoricalEncoder,
    EstimationSampleState, materialize_required_data, plan_ingestion,
)
from econhdfe.frontend.columns import column_names_in_spec, required_columns
from econhdfe import factor, reg_interaction, fv, FixedEffect, interaction, reghdfe


def test_required_columns_understands_econometric_specs_without_expansion():
    x = [
        "control",
        factor("industry"),
        reg_interaction(factor("year"), "tariff"),
        fv("i(cohort)##i(event_time) + c(size)"),
    ]
    absorb = [
        "firm",
        FixedEffect(interaction("city", "year"), slopes=("trend",)),
    ]
    got = required_columns(
        y="outcome", x=x, absorb=absorb, weights="w", cluster=["state", "year"],
    )
    assert got == (
        "outcome", "control", "industry", "year", "tariff", "cohort",
        "event_time", "size", "firm", "city", "trend", "w", "state",
    )


def test_column_names_ignore_resident_arrays():
    a = np.arange(5)
    assert column_names_in_spec(["x", a, reg_interaction("z", a)]) == ("x", "z")


def test_dataframe_source_projects_and_batches_without_unused_columns():
    df = pd.DataFrame({
        "y": np.arange(11.0), "x": np.arange(11.0) * 2,
        "unused": ["large" * 10] * 11,
    })
    src = DataFrameSource(df)
    chunks = list(src.scan(["y", "x"], batch_rows=4))
    assert [len(c) for c in chunks] == [4, 4, 3]
    assert all(tuple(c.columns) == ("y", "x") for c in chunks)
    out = src.materialize(["x", "y"])
    assert tuple(out.columns) == ("x", "y")
    np.testing.assert_array_equal(out["y"].to_numpy(), df["y"].to_numpy())


def test_csv_source_pushes_column_projection_into_parser(tmp_path: Path):
    df = pd.DataFrame({
        "y": np.arange(21.0), "x": np.arange(21.0) * 2,
        "unused_a": np.arange(21), "unused_b": ["payload"] * 21,
    })
    path = tmp_path / "wide.csv"
    df.to_csv(path, index=False)
    src = CSVSource(path)
    out = src.materialize(["y", "x"], batch_rows=7)
    assert tuple(out.columns) == ("y", "x")
    assert len(out) == 21
    np.testing.assert_allclose(out["x"], df["x"])


def test_stata_source_projection_is_stable_across_chunks(tmp_path: Path):
    df = pd.DataFrame({"y": np.arange(17.0), "firm": np.arange(17) % 4, "unused": np.arange(17)})
    path = tmp_path / "panel.dta"
    df.to_stata(path, write_index=False)
    src = StataSource(path, convert_categoricals=False)
    chunks = list(src.scan(["firm", "y"], batch_rows=5))
    assert sum(map(len, chunks)) == 17
    assert all(tuple(c.columns) == ("firm", "y") for c in chunks)


def test_stable_categorical_encoder_preserves_codes_across_batches():
    enc = StableCategoricalEncoder()
    a = enc.transform(np.array(["a", "b", "a"], dtype=object))
    b = enc.transform(np.array(["b", "c", "a"], dtype=object))
    np.testing.assert_array_equal(a.codes, [0, 1, 0])
    np.testing.assert_array_equal(b.codes, [1, 2, 0])
    assert enc.levels == ("a", "b", "c")
    assert b.new_levels == ("c",)


def test_categorical_encoder_rejects_missing_ids():
    enc = StableCategoricalEncoder()
    with pytest.raises(Exception, match="missing"):
        enc.transform(np.array(["a", None], dtype=object))


def test_estimation_sample_state_is_monotone_and_auditable():
    s = EstimationSampleState(8)
    e1 = s.exclude([1, 4], stage="data", reason="missing regressor")
    e2 = s.keep(np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool), stage="hdfe", reason="singleton")
    assert e1.dropped == 2 and e1.remaining == 6
    assert e2.dropped == 1 and e2.remaining == 5
    assert s.nobs == 5
    assert s.summary()["dropped_total"] == 3
    assert s.mask.flags.writeable is False


def test_ingestion_planner_uses_projected_width_not_source_width():
    n = 1000
    df = pd.DataFrame({f"x{j}": np.arange(n, dtype=np.float64) for j in range(40)})
    src = DataFrameSource(df)
    narrow = plan_ingestion(src, ["x0", "x1"], memory_budget_mb=64, min_batch_rows=1, max_batch_rows=10_000_000)
    wide = plan_ingestion(src, list(df.columns), memory_budget_mb=64, min_batch_rows=1, max_batch_rows=10_000_000)
    assert narrow.estimated_bytes_per_row < wide.estimated_bytes_per_row / 10
    assert narrow.batch_rows > wide.batch_rows


def test_materialize_required_data_returns_plan_and_only_required_columns():
    df = pd.DataFrame({"y": np.arange(10.0), "x": np.arange(10.0), "junk": "x"})
    out = materialize_required_data(DataFrameSource(df), ["y", "x"], memory_budget_mb=64)
    assert out.columns == ("y", "x")
    assert out.plan.required_columns == ("y", "x")
    assert out.nobs == 10


def _small_panel(seed=812, n=1200):
    rng = np.random.default_rng(seed)
    firm = np.arange(n) % 120
    year = np.arange(n) % 8
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    x = 0.75 * z + 0.2 * c + rng.normal(scale=0.7, size=n)
    y = 1.1 * x - 0.25 * c + rng.normal(size=n)
    junk = rng.normal(size=(n, 12))
    data = {"y": y, "x": x, "z": z, "c": c, "firm": firm, "year": year}
    data.update({f"junk{j}": junk[:, j] for j in range(junk.shape[1])})
    return pd.DataFrame(data)


def test_olshdfe_can_project_directly_from_csv_source(tmp_path: Path):
    from econhdfe import olshdfe, ExecutionConfig
    df = _small_panel()
    path = tmp_path / "wide_panel.csv"
    df.to_csv(path, index=False)
    ref = olshdfe(df, y="y", x=["x", "c"], absorb=["firm", "year"], drop_singletons=False)
    got = olshdfe(
        CSVSource(path), y="y", x=["x", "c"], absorb=["firm", "year"],
        drop_singletons=False, execution_config=ExecutionConfig(profile="summary", memory_budget_mb=64),
    )
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-10, atol=1e-10)
    assert got.profile["data_ingestion"]["required_column_count"] == 5
    assert "required_columns" not in got.profile["data_ingestion"]
    assert "source_label" not in got.profile["data_ingestion"]


def test_ivhdfe_can_project_directly_from_stata_source(tmp_path: Path):
    from econhdfe import ivhdfe, ExecutionConfig
    df = _small_panel()
    path = tmp_path / "wide_panel.dta"
    df.to_stata(path, write_index=False)
    ref = ivhdfe(
        df, y="y", exog=["c"], endog=["x"], instruments=["z"],
        absorb=["firm", "year"], drop_singletons=False,
    )
    got = ivhdfe(
        StataSource(path, convert_categoricals=False), y="y", exog=["c"], endog=["x"], instruments=["z"],
        absorb=["firm", "year"], drop_singletons=False,
        execution_config=ExecutionConfig(profile="summary", memory_budget_mb=64),
    )
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-9, atol=1e-9)
    assert got.profile["data_ingestion"]["required_column_count"] == 6
    assert "required_columns" not in got.profile["data_ingestion"]
    assert "source_label" not in got.profile["data_ingestion"]


def test_source_constructor_reserves_projection_and_batch_controls(tmp_path: Path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"x": [1]}).to_csv(path, index=False)
    with pytest.raises(Exception, match="controlled by econhdfe"):
        CSVSource(path, usecols=["x"])


def test_stata_auto_backend_preserves_pandas_options(tmp_path: Path):
    path = tmp_path / "x.dta"
    pd.DataFrame({"x": [1.0, 2.0]}).to_stata(path, write_index=False)
    src = StataSource(path, convert_categoricals=False)
    assert src.resolved_backend == "pandas"


def test_stata_explicit_pyreadstat_missing_dependency_is_structured(tmp_path: Path, monkeypatch):
    path = tmp_path / "x.dta"
    pd.DataFrame({"x": [1.0]}).to_stata(path, write_index=False)
    src = StataSource(path, backend="pyreadstat")
    monkeypatch.setattr(src, "_pyreadstat", lambda: None)
    with pytest.raises(Exception, match="requires pyreadstat"):
        src.materialize(["x"])


def test_role_aware_requirements_protect_factor_semantics():
    from econhdfe.frontend.columns import compile_data_requirements
    req = compile_data_requirements(
        y="y",
        x=[factor("industry"), "size"],
        absorb=["firm", FixedEffect("industry")],
        cluster="firm",
    )
    assert req.roles_for("firm") == ("fixed_effect", "cluster")
    assert req.roles_for("industry") == ("factor", "fixed_effect")
    assert req.roles_for("size") == ("regressor",)
    assert req.identifier_only_columns == ("firm",)


def test_factor_variable_role_map_distinguishes_factor_and_continuous_atoms():
    from econhdfe.frontend.columns import compile_data_requirements
    req = compile_data_requirements(x=fv("i(cohort)##c(treatment)"), absorb="firm")
    assert req.roles_for("cohort") == ("factor",)
    assert req.roles_for("treatment") == ("regressor",)
    assert req.roles_for("firm") == ("fixed_effect",)


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


def test_encoded_dataset_snapshots_columns_and_reuses_numeric_conversion():
    from econhdfe.data import EncodedEconometricDataset
    df = pd.DataFrame({
        "y": np.array([1.0, 2.0, 3.0]),
        "x": np.array([4, 5, 6], dtype=np.int64),
        "firm": ["a", "b", "a"],
    })
    ds = EncodedEconometricDataset.from_frame(df, identifier_columns=["firm"])
    assert ds.column("firm").dtype == np.int32
    assert ds.identifier_levels["firm"] == ("a", "b")
    x1 = ds.numeric_column("x")
    x2 = ds.numeric_column("x")
    assert x1 is x2
    assert x1.dtype == np.float64
    sig = ds.column_signature("x")
    df.loc[0, "x"] = 999
    assert ds.column_signature("x") == sig
    np.testing.assert_array_equal(ds.column("x"), np.array([4, 5, 6]))


def test_encoded_dataset_validity_and_sample_mask_cache():
    from econhdfe.data import EncodedEconometricDataset
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    ds = EncodedEconometricDataset.from_frame(df)
    np.testing.assert_array_equal(ds.valid_mask("a"), [True, False, True])
    m1 = ds.sample_mask(["a", "b"])
    m2 = ds.sample_mask(["b", "a"])
    assert m1 is m2
    np.testing.assert_array_equal(m1, [True, False, True])


def test_repeated_ols_session_accepts_encoded_dataset_with_parity():
    from econhdfe import OLSHDFESession
    from econhdfe.config import ExecutionConfig
    from econhdfe.data import EncodedEconometricDataset
    rng = np.random.default_rng(7771)
    n = 1200
    firm = np.repeat(np.arange(120), 10)
    year = np.tile(np.arange(10), 120)
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=120)[firm] + rng.normal(size=10)[year] + rng.normal(scale=.2, size=n)
    df = pd.DataFrame({"y": y, "x": x, "firm": firm, "year": year})
    cfg = ExecutionConfig(cache_validation="signature")
    base = OLSHDFESession(df, execution_config=cfg).fit(y="y", x=["x"], absorb=["firm", "year"])
    ds = EncodedEconometricDataset.from_frame(df, identifier_columns=["firm", "year"])
    sess = OLSHDFESession(ds, execution_config=cfg)
    got = sess.fit(y="y", x=["x"], absorb=["firm", "year"])
    np.testing.assert_allclose(got.params, base.params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(got.vcov, base.vcov, rtol=1e-10, atol=1e-11)
    info = sess.cache_info()
    assert info["encoded_dataset"]["column_count"] == 4
    assert info["encoded_dataset"]["numeric_cache_columns"] >= 2


def test_prepare_repeated_dataset_unions_roles_before_materialization(tmp_path):
    from econhdfe.data import CSVSource, prepare_repeated_dataset
    df = pd.DataFrame({
        "y1": [1.0, 2.0, 3.0, 4.0], "y2": [2.0, 1.0, 4.0, 3.0],
        "x1": [1.0, 0.0, 1.0, 0.0], "x2": [0.0, 1.0, 0.0, 1.0],
        "firm": ["a", "a", "b", "b"], "year": [1, 2, 1, 2],
        "unused": [9, 9, 9, 9],
    })
    path = tmp_path / "wide.csv"; df.to_csv(path, index=False)
    ds = prepare_repeated_dataset(
        CSVSource(path),
        [
            {"y": "y1", "x": ["x1"], "absorb": ["firm", "year"]},
            {"y": "y2", "x": ["x1", "x2"], "absorb": ["firm", "year"]},
        ],
    )
    assert set(ds.columns) == {"y1", "y2", "x1", "x2", "firm", "year"}
    assert "unused" not in ds.columns
    assert ds.column("firm").dtype == np.int32
    assert ds.column("year").dtype == np.int32


def test_repeated_role_union_does_not_encode_column_used_as_explicit_regressor(tmp_path):
    from econhdfe.data import CSVSource, prepare_repeated_dataset
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 1.5, 2.5],
        "industry": ["a", "b", "a", "b"], "firm": ["f1", "f1", "f2", "f2"],
    })
    path = tmp_path / "roles.csv"; df.to_csv(path, index=False)
    ds = prepare_repeated_dataset(
        CSVSource(path),
        [
            {"y": "y", "x": ["x"], "absorb": ["industry", "firm"]},
            {"y": "y", "x": ["industry", "x"], "absorb": ["firm"]},
        ],
    )
    # industry cannot be identifier-only because another specification uses it
    # as an explicit regressor. firm remains safely encoded.
    assert ds.column("industry").dtype.kind in {"O", "U", "S"}
    assert ds.column("firm").dtype == np.int32


def test_repeated_iv_session_accepts_encoded_dataset_with_parity():
    from econhdfe import IVHDFESession
    from econhdfe.config import ExecutionConfig
    from econhdfe.data import EncodedEconometricDataset
    rng = np.random.default_rng(7772)
    n = 1600
    firm = np.repeat(np.arange(160), 10)
    year = np.tile(np.arange(10), 160)
    z = rng.normal(size=n); c = rng.normal(size=n)
    u = rng.normal(size=n); e = 0.8 * z + 0.4 * u + rng.normal(scale=.4, size=n)
    y = 0.6 * e + 0.2 * c + u + rng.normal(scale=.3, size=n)
    df = pd.DataFrame({"y": y, "e": e, "z": z, "c": c, "firm": firm, "year": year})
    cfg = ExecutionConfig(cache_validation="signature")
    base = IVHDFESession(df, execution_config=cfg).fit(
        y="y", exog=["c"], endog=["e"], instruments=["z"], absorb=["firm", "year"]
    )
    ds = EncodedEconometricDataset.from_frame(df, identifier_columns=["firm", "year"])
    got = IVHDFESession(ds, execution_config=cfg).fit(
        y="y", exog=["c"], endog=["e"], instruments=["z"], absorb=["firm", "year"]
    )
    np.testing.assert_allclose(got.params, base.params, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got.vcov, base.vcov, rtol=1e-9, atol=1e-10)


def test_ols_session_datasource_fit_many_materializes_once_with_projection(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.config import ExecutionConfig
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(7773)
    n = 1000
    firm = np.repeat(np.arange(100), 10); year = np.tile(np.arange(10), 100)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    y1 = .4*x1 + rng.normal(size=n); y2 = .2*x2 + rng.normal(size=n)
    df = pd.DataFrame({"y1":y1,"y2":y2,"x1":x1,"x2":x2,"firm":firm,"year":year,"unused":rng.normal(size=n)})
    path=tmp_path/"session.csv"; df.to_csv(path,index=False)
    specs=[{"y":"y1","x":["x1"],"absorb":["firm","year"]},{"y":"y2","x":["x1","x2"],"absorb":["firm","year"]}]
    cfg=ExecutionConfig(cache_validation="signature")
    ref=OLSHDFESession(df,execution_config=cfg).fit_many(specs)
    sess=OLSHDFESession(CSVSource(path),execution_config=cfg)
    got=sess.fit_many(specs)
    for a,b in zip(got,ref,strict=True):
        np.testing.assert_allclose(a.params,b.params,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(a.vcov,b.vcov,rtol=1e-9,atol=1e-10)
    info=sess.cache_info()
    assert info["data_source"]["materializations"] == 1
    assert info["data_source"]["loaded_column_count"] == 6
    assert "unused" not in sess.dataset.columns


def test_datasource_session_sequential_fit_only_rebuilds_when_new_column_needed(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource
    rng=np.random.default_rng(7774); n=600
    df=pd.DataFrame({"y":rng.normal(size=n),"x1":rng.normal(size=n),"x2":rng.normal(size=n),"firm":np.arange(n)%60,"year":np.arange(n)%10})
    path=tmp_path/"incremental.csv";df.to_csv(path,index=False)
    sess=OLSHDFESession(CSVSource(path),drop_singletons=False)
    sess.fit(y="y",x=["x1"],absorb=["firm","year"])
    assert sess.cache_info()["data_source"]["materializations"] == 1
    sess.fit(y="y",x=["x1"],absorb=["firm","year"])
    assert sess.cache_info()["data_source"]["materializations"] == 1
    sess.fit(y="y",x=["x1","x2"],absorb=["firm","year"])
    assert sess.cache_info()["data_source"]["materializations"] == 2


def test_iv_session_datasource_fit_many_parity(tmp_path):
    from econhdfe import IVHDFESession
    from econhdfe.data import CSVSource
    rng=np.random.default_rng(7775);n=1200
    firm=np.repeat(np.arange(120),10);year=np.tile(np.arange(10),120)
    z=rng.normal(size=n);c=rng.normal(size=n);u=rng.normal(size=n);e=.9*z+.3*u+rng.normal(scale=.4,size=n);y=.6*e+.2*c+u+rng.normal(scale=.3,size=n)
    df=pd.DataFrame({"y":y,"e":e,"z":z,"c":c,"firm":firm,"year":year,"unused":rng.normal(size=n)})
    path=tmp_path/"ivsession.csv";df.to_csv(path,index=False)
    spec={"y":"y","exog":["c"],"endog":["e"],"instruments":["z"],"absorb":["firm","year"]}
    ref=IVHDFESession(df).fit_many([spec])[0]
    sess=IVHDFESession(CSVSource(path));got=sess.fit_many([spec])[0]
    np.testing.assert_allclose(got.params,ref.params,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(got.vcov,ref.vcov,rtol=1e-9,atol=1e-10)
    assert sess.cache_info()["data_source"]["materializations"] == 1


def test_persistent_session_reuses_columns_and_within_across_new_sessions(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(8801)
    n = 2400
    firm = np.repeat(np.arange(240), 10)
    year = np.tile(np.arange(10), 240)
    x = rng.normal(size=n)
    y1 = 0.6 * x + rng.normal(size=240)[firm] + rng.normal(scale=.2, size=n)
    y2 = -0.3 * x + rng.normal(size=240)[firm] + rng.normal(scale=.2, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2, "x": x, "firm": firm, "year": year, "junk": rng.normal(size=n)})
    path = tmp_path / "research.csv"; df.to_csv(path, index=False)
    cache = tmp_path / ".econhdfe-cache"

    first = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    r1 = first.fit(y="y1", x=["x"], absorb=["firm", "year"])
    assert first.persistent_cache_info()["column_misses"] == 4
    assert first.cache_info()["residualized_columns"] == 2

    second = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    got = second.fit(y="y2", x=["x"], absorb=["firm", "year"])
    ref = OLSHDFESession(df, drop_singletons=False).fit(y="y2", x=["x"], absorb=["firm", "year"])
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=1e-10, atol=1e-11)
    info = second.persistent_cache_info()
    assert info["column_hits"] >= 3  # x + firm + year
    assert info["column_misses"] == 1  # new outcome only
    assert info["within_hits"] >= 1  # x does not need another HDFE projection
    assert second.cache_info()["residualized_columns"] == 1
    assert not np.allclose(r1.params, got.params)


def test_persistent_session_completed_ols_result_resumes_without_residualization(tmp_path):
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
    assert b.persistent_cache_info()["result_hits"] == 1
    assert b.cache_info()["residualize_calls"] == 0


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


def test_persistent_iv_result_roundtrip_when_payload_is_supported(tmp_path):
    from econhdfe import IVHDFESession
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(8804)
    n = 1800
    firm = np.repeat(np.arange(180), 10); year = np.tile(np.arange(10), 180)
    z = rng.normal(size=n); c = rng.normal(size=n); u = rng.normal(size=n)
    e = 0.9 * z + 0.4 * u + rng.normal(scale=.4, size=n)
    y = 0.5 * e + 0.2 * c + u + rng.normal(scale=.3, size=n)
    df = pd.DataFrame({"y": y, "e": e, "z": z, "c": c, "firm": firm, "year": year})
    path = tmp_path / "iv.csv"; df.to_csv(path, index=False)
    cache = tmp_path / "cache"
    kwargs = dict(y="y", exog=["c"], endog=["e"], instruments=["z"], absorb=["firm", "year"])
    a = IVHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    ref = a.fit(**kwargs)
    b = IVHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    got = b.fit(**kwargs)
    np.testing.assert_allclose(got.params, ref.params, rtol=0, atol=0)
    np.testing.assert_allclose(got.vcov, ref.vcov, rtol=0, atol=0)
    assert b.persistent_cache_info()["result_hits"] == 1


def test_persistent_fit_many_skips_completed_specifications(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(8805)
    n = 2000
    df = pd.DataFrame({
        "y1": rng.normal(size=n), "y2": rng.normal(size=n),
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "firm": np.arange(n) % 200, "year": np.arange(n) % 10,
    })
    path = tmp_path / "suite.csv"; df.to_csv(path, index=False)
    cache = tmp_path / "cache"
    s1 = {"y": "y1", "x": ["x1"], "absorb": ["firm", "year"]}
    s2 = {"y": "y2", "x": ["x1", "x2"], "absorb": ["firm", "year"]}
    first = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    r1 = first.fit_many([s1])[0]
    second = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(cache)
    got = second.fit_many([s1, s2])
    np.testing.assert_allclose(got[0].params, r1.params, rtol=0, atol=0)
    assert second.persistent_cache_info()["result_hits"] >= 1
    # Only the unfinished second specification should require fresh projection;
    # x1 may be restored from the persistent within-column cache.
    assert second.cache_info()["residualized_columns"] <= 2


def test_persistent_metadata_validation_invalidates_after_normal_file_rewrite(tmp_path):
    from econhdfe import OLSHDFESession
    from econhdfe.data import CSVSource
    rng = np.random.default_rng(8806)
    n = 800
    df = pd.DataFrame({
        "y": rng.normal(size=n), "x": rng.normal(size=n),
        "firm": np.arange(n) % 80, "year": np.arange(n) % 10,
    })
    path = tmp_path / "metadata.csv"; df.to_csv(path, index=False)
    cache = tmp_path / "cache"
    a = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(
        cache, source_validation="metadata"
    )
    a.fit(y="y", x=["x"], absorb=["firm", "year"])
    changed = df.copy(); changed["y"] += changed["x"]
    changed.to_csv(path, index=False)
    b = OLSHDFESession(CSVSource(path), drop_singletons=False).enable_persistent_cache(
        cache, source_validation="metadata"
    )
    got = b.fit(y="y", x=["x"], absorb=["firm", "year"])
    ref = OLSHDFESession(changed, drop_singletons=False).fit(y="y", x=["x"], absorb=["firm", "year"])
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-11, atol=1e-11)
    assert b.persistent_cache_info()["result_hits"] == 0


def test_persistent_cache_rejects_unknown_source_validation(tmp_path):
    from econhdfe import OLSHDFESession
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
    sess = OLSHDFESession(df)
    with pytest.raises(Exception):
        sess.enable_persistent_cache(tmp_path / "cache", source_validation="unsafe")


def test_direct_estimator_mapping_input_remains_backward_compatible():
    rng = np.random.default_rng(91350)
    n = 240
    firm = np.repeat(np.arange(24), 10)
    year = np.tile(np.arange(10), 24)
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=24)[firm] + rng.normal(size=10)[year] + rng.normal(scale=0.2, size=n)
    data = {"y": y, "x": x, "firm": firm, "year": year}
    out = reghdfe(data, y="y", x=["x"], absorb=["firm", "year"], drop_singletons=False)
    assert out.converged
    assert out.params[0] == pytest.approx(0.7, abs=0.08)
