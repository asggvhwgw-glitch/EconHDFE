"""Opt-in native PPML/IV-PPML: independent moments and lifecycle boundaries."""
from dataclasses import replace
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import root

if os.environ.get("REQUIRE_WITHIN_NATIVE") == "1":
    import _econhdfe_within  # noqa: F401
else:
    pytest.importorskip("_econhdfe_within")

from econhdfe.config import ExecutionConfig
from econhdfe.errors import ConvergenceError, SpecificationError
from econhdfe.hdfe.plan import FEPlan
from econhdfe.hdfe.weighted_projection import WeightedFEProjector
from econhdfe.hdfe.within_backend import WithinFEAbsorber
from econhdfe.models.ppml.api import ppmlhdfe, PPMLHDFE
from econhdfe.models.ppml.config import PPMLConfig
from econhdfe.models.ppml_iv.api import ivppmlhdfe, IVPPMLHDFE
from econhdfe.models.ppml_iv.config import IVPPMLConfig


def fixture(n=1200):
    rng = np.random.default_rng(20261027)
    groups = [rng.integers(L, size=n, dtype=np.int32) for L in (12, 8, 5)]
    C = rng.normal(size=(n, 2))
    I = rng.normal(size=(n, 1))
    E = .7 * I + .2 * C[:, :1] + .3 * rng.normal(size=(n, 1))
    off = .15 * rng.normal(size=n)
    eta = 1 + C @ np.array([.1, -.2]) + .3 * E[:, 0] + off
    for g in groups:
        eta += rng.normal(scale=.15, size=int(g.max()) + 1)[g]
    y = np.exp(eta) * np.exp(.15 * rng.normal(size=n) - .5 * .15**2)
    w = np.exp(.3 * rng.normal(size=n))
    return dict(y=y, C=C, E=E, I=I, off=off, w=w, groups=groups)


def config(kind, engine="within", **kwargs):
    cls = PPMLConfig if kind == "ppml" else IVPPMLConfig
    return cls(engine=engine, tolerance=1e-11, target_inner_tol=1e-12,
               start_inner_tol=1e-5, max_iter=200, min_ok=2, **kwargs)


def fit(kind, d, engine="within", *, cfg=None):
    common = dict(y=d["y"], absorb=d["groups"], weights=d["w"],
                  exposure=np.exp(d["off"]), vce="robust",
                  config=config(kind, engine) if cfg is None else cfg,
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=128))
    if kind == "ppml":
        return ppmlhdfe(x=np.column_stack([d["C"], d["E"]]), **common)
    return ivppmlhdfe(exog=d["C"], endog=d["E"], instruments=d["I"],
                      weight_type="pweight", **common)


def dummy(groups):
    # Independent reference coding: all levels of first FE, omit level 0 of others.
    blocks = [np.eye(int(g.max()) + 1)[g] for g in groups]
    return np.column_stack([blocks[0]] + [b[:, 1:] for b in blocks[1:]])


def reference(kind, d):
    D = dummy(d["groups"])
    X = np.column_stack([d["C"], d["E"]])
    R = np.column_stack([X, D])
    Q = R if kind == "ppml" else np.column_stack([d["C"], d["I"], D])
    # Exactly identified additive IV moments, not the package's IRLS implementation.
    assert R.shape == Q.shape
    assert np.linalg.matrix_rank(R) == R.shape[1]
    def moment(b):
        mu = np.exp(R @ b + d["off"])
        return Q.T @ (d["w"] * (d["y"] - mu)) / len(mu)
    def jac(b):
        mu = np.exp(R @ b + d["off"])
        return -(Q.T * (d["w"] * mu)) @ R / len(mu)
    out = root(moment, np.zeros(R.shape[1]), jac=jac, tol=1e-11)
    assert out.success, out.message
    assert np.max(np.abs(moment(out.x))) < 1e-10
    return out.x[:X.shape[1]], np.exp(R @ out.x + d["off"])


def compare(a, b):
    assert a.converged and b.converged
    assert a.names == b.names
    assert a.nobs == b.nobs
    assert a.df_absorbed == b.df_absorbed
    assert_array_equal(a.sample_mask, b.sample_mask)
    assert_array_equal(a.separation_mask, b.separation_mask)
    assert_allclose(a.params, b.params, atol=5e-8, rtol=2e-7)
    assert_allclose(a.mu, b.mu, atol=3e-7, rtol=2e-7, equal_nan=True)
    assert_allclose(a.vcov, b.vcov, atol=2e-9, rtol=2e-6)
    assert_allclose(a.deviance, b.deviance, atol=2e-7, rtol=2e-8)


@pytest.mark.parametrize("kind,standardize", [("ppml", True), ("ivppml", False)])
def test_estimator_matches_independent_dummy_moments(kind, standardize):
    d = fixture()
    native = fit(kind, d, cfg=config(kind, standardize=standardize))
    beta, mu = reference(kind, d)
    assert_allclose(native.params, beta, atol=5e-8, rtol=2e-7)
    assert_allclose(native.mu, mu, atol=3e-7, rtol=2e-7)
    compare(native, fit(kind, d, cfg=config(kind, "replica", standardize=standardize)))
    resources = native.diagnostics["projection_resources"]
    assert resources["backend"] == resources["method"] == "within"
    assert resources["native"]["weight_generation"] >= native.iterations
    assert resources["native"]["nobs"] == native.nobs


@pytest.mark.parametrize("kind", ["ppml", "ivppml"])
def test_units_and_column_permutation(kind):
    d = fixture()
    base = fit(kind, d)
    changed = dict(d, C=d["C"][:, ::-1] * [100., .01], E=d["E"] * 3,
                   I=d["I"] * .2, groups=d["groups"][::-1])
    result = fit(kind, changed)
    order = [1, 0, 2]
    scale = np.array([.01, 100., 3.])
    assert_allclose(result.params[order] * scale, base.params, atol=5e-8, rtol=2e-7)
    assert_allclose(result.vcov[np.ix_(order, order)] * np.outer(scale, scale),
                    base.vcov, atol=2e-9, rtol=2e-6)
    assert_allclose(result.mu, base.mu, atol=3e-7, rtol=2e-7)


def test_positive_weight_updates_and_zero_support_roundtrip():
    d = fixture(900)
    plan = FEPlan.from_arrays(d["groups"])
    p = WeightedFEProjector(plan.groups, engine="within", absorb_threads=1)
    A = np.column_stack([d["C"], d["E"]])
    D = dummy(plan.groups)
    w = d["w"].copy()
    original_w = w.copy()
    saved = None
    for state in range(4):
        if state == 1:
            w *= np.exp(.3 * d["C"][:, 0])  # same mutable IRLS buffer
        elif state == 2:
            w[plan.groups[0] == 0] = 0  # remove a complete level from support
        elif state == 3:
            w[:] = original_w
        out, info = p.residualize(A, w, tol=1e-12, return_info=True)
        assert info.converged
        sw = np.sqrt(w)
        expected = A - D @ np.linalg.lstsq(D * sw[:, None], A * sw[:, None], rcond=None)[0]
        assert_allclose(out[w > 0], expected[w > 0], atol=2e-8, rtol=2e-8)
        if state == 0:
            saved = p._absorber
            generation = p.resource_info["native"]["weight_generation"]
        elif state in (1, 3):
            assert p._absorber is saved
            assert p.resource_info["native"]["weight_generation"] > generation
        else:
            assert p.resource_info["backend"] == "generic"
            assert p.resource_info["native"] is None
    # One-shot FEPlan entry must honor the same opt-in engine.
    assert_allclose(plan.residualize(A, original_w, tol=1e-12, engine="within")[0],
                    out, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("kind", ["ppml", "ivppml"])
def test_advanced_separation_rebuilds_the_native_sample(kind):
    d = fixture()
    separated = np.arange(len(d["y"])) % 11 == 0
    y = d["y"].copy()
    y[separated] = 0
    d = dict(d, y=y, C=np.column_stack([d["C"], separated.astype(float)]))
    native = fit(kind, d)
    compare(native, fit(kind, d, "replica"))
    assert_array_equal(native.sample_mask, ~separated)
    resources = native.diagnostics["projection_resources"]
    assert resources["backend"] == "within"
    assert resources["native"]["nobs"] == int((~separated).sum())
    # Separation retains the existing replica solvers, not the native positive-weight solver.
    assert all("within" not in s for s in native.diagnostics["separation_solvers"].values())


@pytest.mark.parametrize("kind,stage", [("ppml", "rank"), ("ivppml", "irls"),
                                       ("ppml", "final"), ("ivppml", "final")])
def test_uncertified_inner_solve_never_becomes_a_successful_estimate(monkeypatch, kind, stage):
    old = WithinFEAbsorber.residualize
    calls = 0
    final = False
    if stage == "final":
        name = "ppml" if kind == "ppml" else "ppml_iv"
        module = importlib.import_module(f"econhdfe.models.{name}.estimator")
        attr = "fit_irls" if kind == "ppml" else "fit_iv_irls"
        solve = getattr(module, attr)
        def mark_final(*args, **kwargs):
            nonlocal final
            result = solve(*args, **kwargs)
            final = True
            return result
        monkeypatch.setattr(module, attr, mark_final)
    def fail(self, data, **kwargs):
        nonlocal calls
        out, info = old(self, data, **kwargs)
        calls += 1
        trigger = final if stage == "final" else calls == (1 if stage == "rank" else 2)
        if trigger:
            info.converged = False
            info.max_update = 1.
        return out, info
    monkeypatch.setattr(WithinFEAbsorber, "residualize", fail)
    with pytest.raises(ConvergenceError, match="weighted FE projection"):
        fit(kind, fixture())


@pytest.mark.parametrize("kind", ["ppml", "ivppml"])
def test_missing_native_extension_is_not_a_silent_backend_fallback(monkeypatch, kind):
    monkeypatch.setitem(sys.modules, "_econhdfe_within", None)
    with pytest.raises(SpecificationError, match="optional native extension"):
        fit(kind, fixture())


@pytest.mark.parametrize("kind", ["ppml", "ivppml"])
def test_reusable_dataframe_warm_start_updates_weights(kind):
    d = fixture()
    frame = pd.DataFrame(dict(y=d["y"], c0=d["C"][:, 0], c1=d["C"][:, 1],
                              e=d["E"][:, 0], z=d["I"][:, 0], off=d["off"], w=d["w"],
                              g0=d["groups"][0], g1=d["groups"][1], g2=d["groups"][2]))
    cls = PPMLHDFE if kind == "ppml" else IVPPMLHDFE
    model = cls(frame, absorb=("g0", "g1", "g2"), config=config(kind),
                execution_config=ExecutionConfig(threads=1, memory_budget_mb=128))
    args = dict(offset="off", weights="w", vce="robust")
    if kind == "ppml":
        args["x"] = ("c0", "c1", "e")
    else:
        args.update(exog=("c0", "c1"), endog=("e",), instruments=("z",), weight_type="pweight")
    a = model.fit("y", **args)
    frame["w"] *= np.exp(.2 * frame["c0"])
    b = model.fit("y", warm_start=a, **args)
    fresh = cls(frame, absorb=("g0", "g1", "g2"), config=config(kind),
                execution_config=ExecutionConfig(threads=1, memory_budget_mb=128)).fit("y", **args)
    compare(b, fresh)
    assert b.diagnostics["projection_resources"]["native"]["weight_generation"] > a.diagnostics["projection_resources"]["native"]["weight_generation"]


def test_iv_frequency_weights_match_replicated_observations():
    d = fixture(600)
    fw = 1 + np.arange(600) % 3
    kwargs = dict(y=d["y"], exog=d["C"], endog=d["E"], instruments=d["I"],
                  absorb=d["groups"], offset=d["off"], weights=fw, weight_type="fweight",
                  config=config("ivppml"), execution_config=ExecutionConfig(threads=1))
    a = ivppmlhdfe(**kwargs)
    idx = np.repeat(np.arange(600), fw)
    b = ivppmlhdfe(y=d["y"][idx], exog=d["C"][idx], endog=d["E"][idx],
                   instruments=d["I"][idx], absorb=[g[idx] for g in d["groups"]],
                   offset=d["off"][idx], config=config("ivppml"),
                   execution_config=ExecutionConfig(threads=1))
    assert_allclose(a.params, b.params, atol=5e-8, rtol=2e-7)
    assert_allclose(a.mu[idx], b.mu, atol=3e-7, rtol=2e-7)
    assert_allclose(a.vcov, b.vcov, atol=2e-9, rtol=2e-6)
