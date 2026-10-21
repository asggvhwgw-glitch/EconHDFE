"""Execution policy and stage snapshots, including actual skipped work."""
import numpy as np
import pandas as pd
import pytest
from econhdfe import ivhdfe, InferenceConfig, ExecutionConfig, IVHDFESession
from econhdfe.errors import EconHDFEError, SpecificationError
from econhdfe.models.linear_iv import estimators as est, diagnostics as diag
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock


def data():
    rng = np.random.default_rng(721)
    n = 400
    df = pd.DataFrame({f"z{i}": rng.normal(size=n) for i in range(4)})
    df["c"] = rng.normal(size=n) * 1e3
    df["d"] = (df.z0 + .3 * df.z1 + rng.normal(size=n)) * 1e-3
    df["d2"] = df.z2 + .4 * df.z3 + rng.normal(size=n)
    df["y"] = .001 * df.c + 900 * df.d + .7 * df.d2 + rng.normal(size=n)
    df["fe"] = np.arange(n) % 20
    df["cluster"] = np.arange(n) % 13
    df["w"] = 1. + np.arange(n) % 3
    return df


SPEC = dict(y="y", exog=["c"], endog=["d", "d2"],
            instruments=["z0", "z1", "z2", "z3"], absorb=["fe"])


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, str) or a is None:
        assert a == b
    else:
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True)


def assert_common(full, basic):
    for key in ("params", "vcov", "stderr", "rank", "df_resid", "nobs"):
        same(getattr(full, key), getattr(basic, key))
    for x, y in zip(full.first_stage["diagnostics"], basic.first_stage["diagnostics"]):
        assert "shea_partial_r2" not in y
        same({k:v for k,v in x.items() if k != "shea_partial_r2"}, y)
    for key in ("kleibergen_paap", "overidentification"):
        same(full.diagnostics[key], basic.diagnostics[key])
    for key in ("sanderson_windmeijer", "cragg_donald_f", "stock_yogo"):
        assert key in full.diagnostics and key not in basic.diagnostics
    assert basic.diagnostics["advanced_statistics"]["status"] == "not_requested"


@pytest.mark.parametrize("model", ["2sls", "liml", "kclass", "gmm2s"])
@pytest.mark.parametrize("vce", ["iid", "robust", "cluster"])
@pytest.mark.parametrize("weight", [None, "aweight", "pweight", "fweight"])
def test_basic_matches_full_and_stages_restore_units(model, vce, weight):
    df = data()
    kw = dict(SPEC, estimator=model, vce=vce,
              cluster="cluster" if vce == "cluster" else None,
              kappa=.7 if model == "kclass" else None)
    if weight:
        kw.update(weights="w", weight_type=weight)
    full = ivhdfe(df, **kw)
    events = []
    basic = ivhdfe(df, **kw, inference_config=InferenceConfig(iv_advanced_statistics=False),
                   on_stage=events.append)
    assert_common(full, basic)
    assert [e["stage"] for e in events] == ["estimation", "first_stage", "diagnostics", "complete"]
    first = events[0]
    same(first["params"], basic.params)
    same(first["vcov"], basic.vcov)
    same(first["stderr"], basic.stderr)
    assert first["names"] == basic.names and first["df_resid"] == basic.df_resid
    assert first["nobs"] == basic.nobs and first["diagnostics_status"] == "pending"
    assert first["df_absorbed"] == basic.df_absorbed
    assert events[-1]["result"]["model"]["nobs"] == basic.nobs
    assert "kleibergen_paap" in basic.publication_output()["identification_tests"]


def forbidden(*args, **kwargs):
    raise AssertionError("advanced diagnostic was evaluated")


@pytest.mark.parametrize("model", ["2sls", "gmm2s"])
def test_disabled_really_skips_advanced(monkeypatch, model):
    for obj, name in ((diag, "_shea_partial_r2"), (diag, "sanderson_windmeijer_diagnostics"),
                      (est, "cragg_donald_stat"), (est, "stock_yogo_critical_values")):
        monkeypatch.setattr(obj, name, forbidden)
    r = ivhdfe(data(), **SPEC, estimator=model,
               inference_config=InferenceConfig(iv_advanced_statistics=False))
    assert "kleibergen_paap" in r.diagnostics and "overidentification" in r.diagnostics


def test_snapshot_mutations_do_not_change_fit():
    df = data()
    reference = ivhdfe(df, **SPEC)
    def mutate(event):
        if event["stage"] == "estimation":
            event["params"][:] = 0
            event["vcov"][:] = 0
        elif event["stage"] == "first_stage":
            event["diagnostics"][0].clear()
        elif event["stage"] == "diagnostics":
            event["diagnostics"]["kleibergen_paap"].clear()
        else:
            event["result"]["first_stage"]["coefficients"][:] = 0
    got = ivhdfe(df, **SPEC, on_stage=mutate)
    same(reference.params, got.params)
    same(reference.vcov, got.vcov)
    same(reference.first_stage, got.first_stage)
    same(reference.diagnostics, got.diagnostics)


def test_diagnostics_failure_occurs_after_estimation_event(monkeypatch):
    events = []
    def fail(*args, **kwargs):
        assert [e["stage"] for e in events] == ["estimation"]
        raise ValueError("diagnostic failure fixture")
    monkeypatch.setattr(est, "_first_stage_suite", fail)
    with pytest.raises(EconHDFEError):
        ivhdfe(data(), **SPEC, on_stage=events.append)
    assert len(events) == 1  # No complete event on failure.


def test_callback_failure_is_not_swallowed():
    def fail(event):
        raise SpecificationError("callback fixture")
    with pytest.raises(SpecificationError, match="callback fixture"):
        ivhdfe(data(), **SPEC, on_stage=fail)


def test_basic_does_not_skip_identification_checks(monkeypatch):
    def fail(*args, **kwargs):
        raise SpecificationError("identification fixture")
    monkeypatch.setattr(est, "_check_iv_moment_rank", fail)
    with pytest.raises(SpecificationError, match="identification fixture"):
        ivhdfe(data(), **SPEC, inference_config=InferenceConfig(iv_advanced_statistics=False))


def test_invalid_options_are_structured():
    with pytest.raises(SpecificationError):
        ivhdfe(data(), **SPEC, on_stage=4)
    with pytest.raises(SpecificationError):
        ivhdfe(data(), **SPEC, inference_config=InferenceConfig(iv_advanced_statistics="false"))


def test_display_mode_is_independent():
    df = data()
    full = ivhdfe(df, **SPEC, inference_config=InferenceConfig(diagnostics="off"))
    basic = ivhdfe(df, **SPEC, inference_config=InferenceConfig(diagnostics="full", iv_advanced_statistics=False))
    assert "stock_yogo" in full.diagnostics
    assert "stock_yogo" not in basic.publication_output()["diagnostics"]


def test_block_core_basic_and_stages(monkeypatch):
    df = data()
    y = df.y.to_numpy()
    def block(cols):
        v = df[cols].to_numpy()
        n,k = v.shape
        return BlockDesign(n, k, (DenseDesignBlock(0, np.arange(n), np.arange(k), v),))
    args = (y, block(["c"]), block(["d","d2"]), block(["z0","z1","z2","z3"]))
    base = est.fit_iv_kclass_block(*args, vce="robust")
    monkeypatch.setattr(diag, "_shea_partial_r2", forbidden)
    monkeypatch.setattr(est, "cragg_donald_stat", forbidden)
    events = []
    got = est.fit_iv_kclass_block(*args, vce="robust", advanced_statistics=False, on_stage=events.append)
    same(base[:5], got[:5])
    same(events[0]["params"], got[0])
    same(events[0]["vcov"], got[1])
    assert [e["stage"] for e in events] == ["estimation","first_stage","diagnostics"]


@pytest.mark.parametrize("advanced", [False, True])
def test_group_public_path(advanced):
    df = data()
    expanded = df.loc[df.index.repeat(2)].reset_index(drop=True)
    events = []
    got = ivhdfe(expanded, **SPEC, group=np.repeat(np.arange(len(df)),2),
                 inference_config=InferenceConfig(iv_advanced_statistics=advanced), on_stage=events.append)
    base = ivhdfe(df, **SPEC, inference_config=InferenceConfig(iv_advanced_statistics=advanced))
    same(base.params, got.params)
    same(events[0]["params"], got.params)
    same(events[-1]["result"]["first_stage"]["diagnostics"], got.first_stage["diagnostics"])
    assert [e["stage"] for e in events] == ["estimation","first_stage","diagnostics","complete"]


def test_session_policy_cache_keys_and_callbacks():
    df = data()
    events = []
    s = IVHDFESession(df, inference_config=InferenceConfig(iv_advanced_statistics=False),
                      execution_config=ExecutionConfig(cache_validation="none"), on_stage=events.append)
    result = s.fit(**SPEC)
    assert events[-1]["stage"] == "complete"
    assert "stock_yogo" not in result.diagnostics
    same(events[0]["params"], result.params)
    # The result-key payload already includes asdict(InferenceConfig).
    from dataclasses import asdict
    assert asdict(s.inference)["iv_advanced_statistics"] is False
    events.clear()
    out = s.fit_many([SPEC, SPEC])
    assert len([e for e in events if e["stage"] == "complete"]) == 2
    same(out[0].params, out[1].params)


def test_persistent_cache_separates_policy_and_emits_complete_only(tmp_path):
    from econhdfe.data import CSVSource
    path = tmp_path / "data.csv"
    data().to_csv(path, index=False)
    cache = tmp_path / "cache"
    events = []
    def session(advanced):
        return IVHDFESession(
            CSVSource(path), inference_config=InferenceConfig(iv_advanced_statistics=advanced),
            on_stage=events.append,
        ).enable_persistent_cache(cache)
    full = session(True).fit(**SPEC)
    events.clear()
    basic_session = session(False)
    basic = basic_session.fit(**SPEC)
    assert_common(full, basic)
    assert basic_session.persistent_cache_info()["result_hits"] == 0
    assert len(events) == 4
    events.clear()
    hit = session(False)
    cached = hit.fit(**SPEC)
    assert hit.persistent_cache_info()["result_hits"] == 1
    same(basic.diagnostics, cached.diagnostics)
    assert len(events) == 1 and events[0]["stage"] == "complete" and events[0]["cached"]
    events.clear()
    hit.fit_many([SPEC, SPEC])
    assert len(events) == 2 and all(e["cached"] for e in events)
    events.clear()
    full_hit = session(True)
    full_again = full_hit.fit(**SPEC)
    same(full.diagnostics, full_again.diagnostics)
    assert full_hit.persistent_cache_info()["result_hits"] == 1


def test_default_matches_explicit_true_and_snapshots_are_small():
    df = data()
    default = ivhdfe(df, **SPEC)
    events = []
    explicit = ivhdfe(df, **SPEC, on_stage=events.append,
                     inference_config=InferenceConfig(iv_advanced_statistics=True))
    same(default.params, explicit.params)
    same(default.vcov, explicit.vcov)
    same(default.first_stage, explicit.first_stage)
    same(default.diagnostics, explicit.diagnostics)
    def check(value):
        if isinstance(value, np.ndarray):
            assert len(df) not in value.shape
        elif isinstance(value, pd.DataFrame):
            assert len(value) == len(explicit.params)
        elif isinstance(value, dict):
            assert not {"fitted_endog", "residuals", "fitted", "state"} & value.keys()
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
    check(events)


@pytest.mark.parametrize("model", ["2sls", "gmm2s"])
@pytest.mark.parametrize("vce", ["hac", "dk", "multiway"])
def test_common_diagnostics_for_other_covariances(model, vce):
    df = data()
    kw = dict(SPEC, estimator=model)
    if vce == "multiway":
        kw.update(vce="cluster", cluster=["cluster", "fe"])
    else:
        kw.update(vce=vce, time=np.arange(len(df)), bandwidth=2)
        if vce == "dk":
            kw.update(time=np.arange(len(df)) % 20, panel=np.arange(len(df)) // 20)
    full = ivhdfe(df, **kw)
    events = []
    basic = ivhdfe(df, **kw, inference_config=InferenceConfig(iv_advanced_statistics=False),
                   on_stage=events.append)
    assert_common(full, basic)
    same(events[0]["vcov"], basic.vcov)
    assert events[0]["df_resid"] == basic.df_resid


@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("model", ["2sls", "gmm2s"])
def test_nested_fe_snapshot_reports_frontend_dof(session, model):
    events = []
    kw = dict(vce="cluster", cluster="fe", estimator=model, on_stage=events.append)
    result = (IVHDFESession(data(), **kw).fit(**SPEC) if session
              else ivhdfe(data(), **SPEC, **kw))
    assert result.df_absorbed == 0  # All requested FE are nested in the cluster.
    assert events[0]["df_absorbed"] == result.df_absorbed
    assert events[0]["df_resid"] == result.df_resid
    same(events[0]["vcov"], result.vcov)


@pytest.mark.parametrize("vce", ["robust", "cluster"])
def test_public_block_route_respects_policy_and_names(vce):
    from econhdfe import factor, reg_interaction
    df = data()
    df["g"] = np.arange(len(df)) // 50
    role = factor("g", drop_base=False)
    kw = dict(y="y", exog=["c"], endog=[reg_interaction(role, "d", name="gd")],
              instruments=[reg_interaction(role, "z0", name="gz0"),
                           reg_interaction(role, "z1", name="gz1")],
              absorb=["g"], vce=vce, cluster="cluster" if vce == "cluster" else None)
    full = ivhdfe(df, **kw)
    events = []
    basic = ivhdfe(df, **kw, inference_config=InferenceConfig(iv_advanced_statistics=False),
                   on_stage=events.append)
    assert basic.diagnostics["heterogeneous_spec_path"]
    assert_common(full, basic)
    assert events[0]["names"] == basic.names
    same(events[0]["params"], basic.params)
    same(events[0]["vcov"], basic.vcov)
