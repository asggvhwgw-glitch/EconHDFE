"""Public regression contracts for IV diagnostic policy and stage snapshots."""

import numpy as np
import pandas as pd
import pytest
from econhdfe import ivhdfe, InferenceConfig, IVHDFESession
from econhdfe.errors import SpecificationError
from econhdfe.models.linear_iv import estimators as est, diagnostics as diag


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


@pytest.mark.parametrize('model,vce,weight', [
    ('2sls', 'cluster', 'aweight'),
    ('kclass', 'robust', 'fweight'),
])
def test_basic_matches_full_and_stages_restore_units(model, vce, weight):
    df = data()
    kw = dict(SPEC, estimator=model, vce=vce,
              cluster="cluster" if vce == "cluster" else None,
              kappa=.7 if model == "kclass" else None)
    if weight:
        kw.update(weights="w", weight_type=weight)
    full = ivhdfe(df, **kw)
    events = []
    basic = ivhdfe(df, **kw, inference_config=InferenceConfig(iv_advanced_statistics=False, diagnostics="full"),
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


def test_disabled_really_skips_advanced(monkeypatch):
    model = 'gmm2s'
    for obj, name in ((diag, "_shea_partial_r2"), (diag, "_shea_from_moments"), (diag, "sanderson_windmeijer_diagnostics"),
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


def test_callback_failure_is_not_swallowed():
    seen = []
    def fail(event):
        seen.append(event["stage"])
        if event["stage"] == "first_stage":
            raise SpecificationError("callback fixture")
    with pytest.raises(SpecificationError, match="callback fixture"):
        ivhdfe(data(), **SPEC, on_stage=fail)
    assert seen == ["estimation", "first_stage"]


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


def test_public_block_route_respects_policy_and_names():
    vce = "cluster"
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
