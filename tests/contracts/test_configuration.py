import numpy as np
import pandas as pd
from econhdfe import olshdfe, HDFEConfig, InferenceConfig, ExecutionConfig


def _linear_df(seed=101, n=1200):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 80, n)
    year = rng.integers(0, 12, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    x = 0.8 * z + 0.2 * c + rng.normal(size=n)
    y = 1.3 * x + 0.4 * c + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "c": c, "z": z, "firm": firm, "year": year})


def test_common_configs_control_strategy_reporting_and_profile():
    df = _linear_df(n=800)
    r = olshdfe(
        df, y="y", x=["x"], absorb=["firm", "year"],
        hdfe_config=HDFEConfig(solver="map", tolerance=1e-9, max_iter=4000, dof_method="firstpair"),
        inference_config=InferenceConfig(vce="robust", confidence_level=0.90, diagnostics="off"),
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=128, profile="summary"),
    )
    assert r.vce == "robust"
    assert r.confidence_level == 0.90
    assert r.profile is not None and r.profile["total_seconds"] >= 0
    assert r.absorb_info["solver_selection"]["requested"] == "map"
    ci90 = r.conf_int()
    ci95 = r.conf_int(0.95)
    assert np.all((ci90[:, 1] - ci90[:, 0]) < (ci95[:, 1] - ci95[:, 0]))


def test_reusable_ppml_model_invalidates_fe_plan_when_source_fe_changes():
    from econhdfe import PPMLHDFE
    rng = np.random.default_rng(404)
    n = 500
    df = pd.DataFrame({
        "y": rng.poisson(1.2, n),
        "x": rng.normal(size=n),
        "firm": rng.integers(0, 30, n),
        "year": rng.integers(0, 6, n),
    })
    m = PPMLHDFE(df, absorb=["firm", "year"], execution_config=ExecutionConfig(cache_validation="signature"))
    sig0 = m.plan.fingerprint
    _ = m.fit("y", ["x"])
    df.loc[:20, "firm"] = 999
    _ = m.fit("y", ["x"])
    assert m.plan.fingerprint != sig0



# --- scalar/config precedence (0.9.0) -------------------------------------
import inspect

import pytest

from econhdfe import ivhdfe, OLSHDFESession, IVHDFESession
from econhdfe.errors import SpecificationError
from econhdfe.models.linear import CONFIG_SCALAR_FIELDS, config_precedence, signature_defaults

# canonicalize/canonicalize_fe cannot conflict: both default to True, so the
# only "set" value on either side is False.
_CONFLICTS = [
    ("tol", dict(tol=1e-10), dict(hdfe_config=HDFEConfig(tolerance=1e-12))),
    ("method", dict(method="lsmr"), dict(hdfe_config=HDFEConfig(solver="map"))),
    ("max_iter", dict(max_iter=50), dict(hdfe_config=HDFEConfig(max_iter=60))),
    ("dof_method", dict(dof_method="none"), dict(hdfe_config=HDFEConfig(dof_method="firstpair"))),
    ("vce", dict(vce="iid"), dict(inference_config=InferenceConfig(vce="robust"))),
    ("memory_budget_mb", dict(memory_budget_mb=256), dict(execution_config=ExecutionConfig(memory_budget_mb=128))),
    ("absorb_threads", dict(absorb_threads=1), dict(execution_config=ExecutionConfig(threads=2))),
]
_IDS = [c[0] for c in _CONFLICTS]


def _fit(iv, df, **kw):
    if iv:
        return ivhdfe(df, y="y", exog=["c"], endog=["x"], instruments=["z"], absorb=["firm", "year"], **kw)
    return olshdfe(df, y="y", x=["x", "c"], absorb=["firm", "year"], **kw)


def _capture(*args, **kwargs):
    return kwargs


_capture.__signature__ = inspect.signature(olshdfe)
_probe = config_precedence(_capture)
_DEFAULTS = signature_defaults(olshdfe)
_EXECUTION_FIELDS = {argument: field for field, argument in CONFIG_SCALAR_FIELDS[2][1]}


def _resolve(**kw):
    """Resolved value of every config-backed scalar after config_precedence."""
    out = _probe(None, y="y", x=["x"], absorb=["firm"], **kw)
    execution = out.get("execution_config")
    return {
        name: (getattr(execution, _EXECUTION_FIELDS[name]) if name in _EXECUTION_FIELDS and execution is not None
               else out.get(name, _DEFAULTS[name]))
        for name in _DEFAULTS
    }


def test_estimator_defaults_come_from_their_signatures():
    assert signature_defaults(olshdfe) == signature_defaults(ivhdfe) == {
        "method": "auto", "tol": 1e-8, "max_iter": 16_000, "dof_method": "pairwise",
        "canonicalize_fe": True, "vce": None, "memory_budget_mb": "auto", "absorb_threads": "auto",
    }


@pytest.mark.parametrize("iv", [False, True])
@pytest.mark.parametrize("name,scalar,config", _CONFLICTS, ids=_IDS)
def test_scalar_and_config_conflict_is_rejected(iv, name, scalar, config):
    with pytest.raises(SpecificationError) as info:
        _fit(iv, _linear_df(n=400), **scalar, **config)
    assert info.value.code == "specification.config_conflict"
    assert info.value.details == {"argument": name}


@pytest.mark.parametrize("name,scalar,config", _CONFLICTS, ids=_IDS)
def test_one_sided_or_equal_settings_resolve_without_error(name, scalar, config):
    (config_kw, cfg), = config.items()
    from_config = _resolve(**config)[name]
    assert _resolve(**scalar, **{config_kw: type(cfg)()})[name] == scalar[name]  # scalar only
    assert from_config != _DEFAULTS[name]                                        # config only
    assert _resolve(**config, **{name: from_config})[name] == from_config         # equal: no error
    assert _resolve(**{config_kw: type(cfg)()})[name] == _DEFAULTS[name]          # neither


def test_execution_config_auto_budget_no_longer_overrides_scalar():
    config = ExecutionConfig(profile="summary")
    out = _probe(None, y="y", x=["x"], absorb=["firm"], memory_budget_mb=256, execution_config=config)
    assert out["execution_config"] == ExecutionConfig(profile="summary", memory_budget_mb=256)
    unchanged = _probe(None, y="y", x=["x"], absorb=["firm"], execution_config=config)
    assert unchanged["execution_config"] is config
    assert _resolve(execution_config=ExecutionConfig(memory_budget_mb=128))["memory_budget_mb"] == 128
    assert _resolve(canonicalize_fe=False, hdfe_config=HDFEConfig())["canonicalize_fe"] is False
    assert _resolve(hdfe_config=HDFEConfig(canonicalize=False))["canonicalize_fe"] is False


@pytest.mark.parametrize("iv", [False, True])
def test_scalars_and_config_fields_combine_end_to_end(iv):
    df = _linear_df(n=600)
    r = _fit(iv, df, method="lsmr", vce="robust",
             hdfe_config=HDFEConfig(dof_method="firstpair"),
             inference_config=InferenceConfig(confidence_level=0.9))
    assert r.absorb_info["solver_selection"]["requested"] == "lsmr"
    assert r.dof_info.method == "firstpair"
    assert r.vce == "robust" and r.confidence_level == 0.9
    same = _fit(iv, df, method="lsmr", hdfe_config=HDFEConfig(solver="lsmr"),
                vce="robust", inference_config=InferenceConfig(vce="robust"))
    assert same.absorb_info["solver_selection"]["requested"] == "lsmr" and same.vce == "robust"
    config_only = _fit(iv, df, hdfe_config=HDFEConfig(solver="lsmr"), inference_config=InferenceConfig(vce="robust"))
    assert config_only.absorb_info["solver_selection"]["requested"] == "lsmr" and config_only.vce == "robust"


@pytest.mark.parametrize("Session", [OLSHDFESession, IVHDFESession])
def test_session_vce_follows_the_same_precedence(Session):
    df = _linear_df(n=600)
    with pytest.raises(SpecificationError) as info:
        Session(df, vce="iid", inference_config=InferenceConfig(vce="robust"))
    assert info.value.code == "specification.config_conflict"
    fit = (lambda s: s.fit(y="y", exog=["c"], endog=["x"], instruments=["z"], absorb=["firm", "year"])) \
        if Session is IVHDFESession else (lambda s: s.fit(y="y", x=["x", "c"], absorb=["firm", "year"]))
    cfg = dict(inference_config=InferenceConfig(confidence_level=0.9, iv_advanced_statistics=False))
    assert fit(Session(df, vce="robust", **cfg)).vce == "robust"
    assert fit(Session(df, inference_config=InferenceConfig(vce="robust"))).vce == "robust"
    assert fit(Session(df, vce="robust", inference_config=InferenceConfig(vce="robust"))).vce == "robust"
    assert fit(Session(df, **cfg)).vce == "iid"


# --- InferenceConfig.diagnostics tiers -------------------------------------
@pytest.mark.parametrize("iv", [False, True])
def test_full_diagnostics_includes_profile_in_publication_output(iv):
    df = _linear_df(n=500)
    profiled = ExecutionConfig(profile="summary")
    full = _fit(iv, df, inference_config=InferenceConfig(diagnostics="full"), execution_config=profiled)
    pub = _fit(iv, df, inference_config=InferenceConfig(diagnostics="publication"), execution_config=profiled)
    assert full.publication_output()["profile"] is full.profile
    assert "profile" not in full.publication_output(include_profile=False)
    assert "profile" not in pub.publication_output()
    assert pub.publication_output(include_profile=True)["profile"] is pub.profile
    unprofiled = _fit(iv, df, inference_config=InferenceConfig(diagnostics="full"))
    assert unprofiled.profile is None and "profile" not in unprofiled.publication_output()
