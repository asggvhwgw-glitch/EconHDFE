"""Independent regression references for the October 2026 PPML audit."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root

from econhdfe import ExecutionConfig, InferenceConfig, PPMLConfig, PPMLHDFE, ppmlhdfe
from econhdfe.design import factor, reg_interaction
from econhdfe.errors import SpecificationError
from econhdfe.models.ppml.standardize import outcome_scale


EXECUTION = ExecutionConfig(threads=1, memory_budget_mb=128)


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("log_units", [-60.0, -40.0, 30.0])
def test_default_ppml_outcome_units_preserve_solution_and_robust_covariance(standardize, log_units):
    # The old standardize=True path used max(sd(y), 1e-3), so at -40 it
    # returned slope .01168 instead of .60171 and still declared convergence.
    rng = np.random.default_rng(294)
    x = rng.normal(size=200)
    y = rng.poisson(np.exp(.3 + .8*x)).astype(float) + .5
    cfg = PPMLConfig(standardize=standardize)
    base = ppmlhdfe(y=y, x=x[:, None], config=cfg, execution_config=EXECUTION)
    units = np.exp(log_units)
    got = ppmlhdfe(y=y*units, x=x[:, None], config=cfg, execution_config=EXECUTION)
    assert base.converged and got.converged
    assert got.iterations == base.iterations
    np.testing.assert_allclose(got.params - [0, log_units], base.params, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(got.vcov, base.vcov, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(got.mu/units, base.mu, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(got.eta-log_units, base.eta, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(got.sample_mask, base.sample_mask)
    assert base.params[0] == pytest.approx(.6017099328707416, abs=2e-10)


@pytest.mark.parametrize("standardize", [True, False])
def test_ppml_model_covariance_has_inverse_outcome_unit_scaling(standardize):
    rng = np.random.default_rng(385)
    x = rng.normal(size=180)
    y = rng.poisson(np.exp(.2 + .35*x)).astype(float) + .5
    cfg = PPMLConfig(standardize=standardize)
    base = ppmlhdfe(y=y, x=x, vce="model", config=cfg, execution_config=EXECUTION)
    units = np.exp(-60.)
    got = ppmlhdfe(y=y*units, x=x, vce="model", config=cfg, execution_config=EXECUTION)
    # Unlike a sandwich, the Poisson inverse Hessian scales by 1/c.
    np.testing.assert_allclose(got.vcov*units, base.vcov, atol=1e-12, rtol=2e-10)


@pytest.mark.parametrize("block", [False, True], ids=["dense", "block"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("vce", ["robust", "cluster"])
def test_weighted_offset_ppml_unit_invariance_across_storage_paths(block, standardize, vce):
    rng = np.random.default_rng(967)
    n = 240
    component = np.repeat(np.arange(4), 60)
    fe = np.arange(n)//6
    cluster = np.arange(n) % 13  # Crosses the independent FE components.
    x = rng.normal(size=n)
    off = rng.normal(scale=.2, size=n)
    fw = rng.integers(1, 4, size=n)
    y = rng.poisson(np.exp(.2 + .15*x + .1*(fe % 3) + off)).astype(float) + .5
    cfg = PPMLConfig(standardize=standardize, singleton_mode="unified", separation=("fe",),
                     tolerance=1e-11, target_inner_tol=1e-12)
    frame = pd.DataFrame(dict(y=y, x=x, component=component, fe=fe, off=off, fw=fw, cluster=cluster))
    specs = [reg_interaction(factor("component", drop_base=False), "x")]
    inference = InferenceConfig(ppml_small_sample="residual")

    def fit(units):
        if block:
            frame["y"] = y*units
            return PPMLHDFE(frame, absorb=["fe"], config=cfg,
                            inference_config=inference, execution_config=EXECUTION).fit(
                "y", specs, offset="off", weights="fw", weight_type="fw", vce=vce,
                cluster="cluster" if vce == "cluster" else None)
        X = x[:, None] * (component[:, None] == np.arange(4))
        return ppmlhdfe(y=y*units, x=X, absorb=[fe], offset=off, weights=fw, weight_type="fw",
                        vce=vce, cluster=[cluster] if vce == "cluster" else None,
                        config=cfg, inference_config=inference, execution_config=EXECUTION)

    base, got = fit(1.), fit(np.exp(-60.))
    assert base.converged and got.converged
    if block:
        assert got.diagnostics["heterogeneous_spec_path"]
    assert got.diagnostics["finite_sample_correction"]["convention"] == "residual"
    np.testing.assert_allclose(got.params, base.params, atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(got.vcov, base.vcov, atol=2e-11, rtol=2e-8)
    np.testing.assert_allclose(got.mu/np.exp(-60.), base.mu, atol=2e-10, rtol=2e-9)
    np.testing.assert_array_equal(got.sample_mask, base.sample_mask)


@pytest.mark.parametrize("convention", ["asymptotic", "residual"])
@pytest.mark.parametrize("vce,weight_type,nested", [
    ("robust", None, False), ("robust", "fw", False), ("robust", "pw", False),
    ("cluster", None, False), ("cluster", "fw", False), ("cluster", "pw", False),
    ("cluster", None, True),
])
def test_ppml_finite_sample_conventions_match_explicit_dummy_sandwich(convention, vce, weight_type, nested):
    rng = np.random.default_rng(618)
    n = 120
    fe = np.repeat(np.arange(40), 3)
    cluster = fe if nested else np.tile(np.arange(8), 15)
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(.2 + .4*x + rng.normal(size=40)[fe]*.2)).astype(float) + .5
    w = rng.integers(1, 4, n).astype(float) if weight_type else np.ones(n)
    got = ppmlhdfe(y=y, x=x, absorb=[fe], weights=w if weight_type else None, weight_type=weight_type,
                   vce=vce, cluster=[cluster] if vce == "cluster" else None,
                   inference_config=InferenceConfig(ppml_small_sample=convention),
                   config=PPMLConfig(tolerance=1e-11, target_inner_tol=1e-12), execution_config=EXECUTION)
    assert np.all(got.sample_mask)
    # Independent full dummy score/Jacobian: no absorption, VCE or scaling
    # helper from the implementation is used to construct this reference.
    U = np.column_stack([x, np.eye(40)[fe]])
    start = np.r_[0., np.full(40, np.log(np.average(y, weights=w)))]
    def score(beta):
        return U.T @ (w*(y-np.exp(U @ beta)))
    def jac(beta):
        return -U.T @ ((w*np.exp(U @ beta))[:, None]*U)
    solved = root(score, start, jac=jac, tol=1e-11)
    assert np.max(np.abs(score(solved.x))) < 2e-8
    mu = np.exp(U @ solved.x)
    bread = np.linalg.inv(U.T @ ((w*mu)[:, None]*U))
    scores = U * (w*(y-mu))[:, None]
    n_eff = w.sum() if weight_type == "fw" else n
    k = 1 if convention == "asymptotic" else (1 + (0 if nested else 40))
    nested_adjustment = int(nested and convention == "residual")
    if vce == "robust":
        if weight_type == "fw":
            scores /= np.sqrt(w[:, None])
        meat = scores.T @ scores
        correction = n_eff/(n_eff-k)
    else:
        levels = np.unique(cluster)
        sums = np.vstack([scores[cluster == g].sum(axis=0) for g in levels])
        meat = sums.T @ sums
        correction = (n_eff-1)/(n_eff-k-nested_adjustment)*len(levels)/(len(levels)-1)
    expected = correction*(bread @ meat @ bread)
    np.testing.assert_allclose(got.params, solved.x[:1], atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(got.vcov, expected[:1, :1], atol=2e-10, rtol=2e-7)
    diag = got.diagnostics["finite_sample_correction"]
    assert diag["convention"] == convention
    assert diag["factor"] == pytest.approx(correction, rel=1e-14)


def test_outcome_scale_is_finite_without_absolute_floor_or_overflow():
    y = np.array([1., 2., 4.])*1e300
    weights = np.array([1., 2., 3.])*1e300
    assert outcome_scale(y, weights) == pytest.approx((17./6)*1e300)
    assert outcome_scale(y*1e-320, weights) == pytest.approx((17./6)*1e-20, rel=2e-5, abs=0)
    assert outcome_scale(np.ones(10)*1e-30, np.ones(10)) == pytest.approx(1e-30, rel=1e-14, abs=0)


@pytest.mark.parametrize("value", ["hc1", None, []])
def test_invalid_ppml_covariance_convention_is_a_structured_error(value):
    with pytest.raises(SpecificationError):
        InferenceConfig(ppml_small_sample=value).validate()


@pytest.mark.parametrize("convention", ["asymptotic", "residual"])
@pytest.mark.parametrize("kind", ["hac", "newey_west", "neweywest"])
def test_ppml_hac_diagnostic_reports_the_applied_covariance_factor(convention, kind):
    from econhdfe.models.ppml.vce import ppml_covariance_diagnostics, ppml_vcov
    n = 12
    X = np.column_stack([np.linspace(-1., 1., n), np.ones(n)])
    mu = np.exp(.2 + .1*X[:, 0])
    y = mu + .3*np.sin(np.arange(n))
    bread = np.linalg.inv(X.T @ (mu[:, None]*X))
    scores = X*(y-mu)[:, None]
    # Explicit Bartlett lag sum with the documented default bandwidth.
    bandwidth = 1 + int(np.floor(4.*(n/100.)**(2./9.)))
    meat = scores.T @ scores
    for lag in range(1, bandwidth):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1.-lag/bandwidth)*(cross+cross.T)
    k = 1 if convention == "asymptotic" else X.shape[1]+2
    factor_used = n/(n-k)
    actual = ppml_vcov(X, y, mu, np.ones(n), kind=kind, df_absorbed=2,
                       small_sample=convention)
    np.testing.assert_allclose(actual, factor_used*(bread @ meat @ bread), rtol=2e-13, atol=1e-14)
    diagnostic = ppml_covariance_diagnostics(kind=kind, small_sample=convention,
        effective_n=n, k_total=X.shape[1]+2, nested_adj=0, clusters=None)
    assert diagnostic["factor"] == pytest.approx(factor_used, rel=1e-14)


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("design", ["explicit_constant", "full_categories"])
@pytest.mark.parametrize("vce", ["robust", "cluster"])
def test_ppml_preserves_intercept_direction_when_user_columns_span_constant(standardize, design, vce):
    rng = np.random.default_rng(41)
    n = 60
    a = np.repeat(np.arange(6), 10)
    b = np.tile(np.arange(10), 6)
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(.2 + .2*x + rng.normal(size=6)[a]*.5
                          + rng.normal(size=10)[b]*.5)).astype(float) + .5
    if design == "explicit_constant":
        X = np.column_stack([np.ones(n)*3., x])
        names = ("user_constant", "x")
    else:
        category = np.arange(n) % 3
        X = np.column_stack([x, np.eye(3)[category]])
        names = ("x", "level0", "level1", "level2")
    cfg = PPMLConfig(standardize=standardize, tolerance=1e-12, target_inner_tol=1e-12)
    kwargs = dict(x=X, names=names, vce=vce, config=cfg, execution_config=EXECUTION,
                  cluster=[a, b] if vce == "cluster" else None)
    base = ppmlhdfe(y=y, **kwargs)
    tiny = ppmlhdfe(y=y*np.exp(-40.), **kwargs)
    assert base.names[-1] == tiny.names[-1] == "_cons"
    assert len(base.names) == X.shape[1]  # Exactly one redundant user column.
    if design == "explicit_constant":
        assert base.names == ("x", "_cons")
    for result, units in ((base, 0.), (tiny, -40.)):
        columns = {name: X[:, j] for j, name in enumerate(names)}
        columns["_cons"] = np.ones(n)
        kept = np.column_stack([columns[name] for name in result.names])
        assert result.diagnostics["constant_index"] == len(result.names)-1
        assert result.df_model == len(result.names)-1
        np.testing.assert_allclose(kept @ result.params, result.eta, atol=2e-9, rtol=2e-10)
        # The omission chooses a basis; it does not change the model span.
        np.testing.assert_allclose(kept @ np.linalg.lstsq(kept, X, rcond=None)[0], X, atol=2e-12)
        np.testing.assert_allclose(X.T @ (y-result.mu/np.exp(units)), 0., atol=2e-8)
        assert np.linalg.eigvalsh(result.vcov)[0] >= -1e-12
    np.testing.assert_allclose(tiny.params[:-1], base.params[:-1], atol=2e-10, rtol=2e-9)
    assert tiny.params[-1]+40. == pytest.approx(base.params[-1], abs=2e-10)
    np.testing.assert_allclose(tiny.vcov, base.vcov, atol=2e-11, rtol=2e-8)


def test_ppml_wald_uses_actual_intercept_index_not_user_column_label():
    rng = np.random.default_rng(745)
    x = rng.normal(size=60)
    y = rng.poisson(np.exp(.4*x)).astype(float)+.5
    ordinary = ppmlhdfe(y=y, x=x, names=("x",), execution_config=EXECUTION)
    labelled = ppmlhdfe(y=y, x=x, names=("_cons",), execution_config=EXECUTION)
    assert ordinary.df_model == labelled.df_model == 1
    assert labelled.chi2 == pytest.approx(ordinary.chi2, rel=1e-12)
