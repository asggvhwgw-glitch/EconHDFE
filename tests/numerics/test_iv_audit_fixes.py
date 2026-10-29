"""Analytic counterexamples and outcome-unit invariants from the IV audit.

The identification and separation fixtures have closed-form moment systems.
The nonlinear checks retain the reported beta, FE index, means and covariance
at one common state, including the structured path and weighted inference.
"""

import numpy as np
import pytest

from econhdfe import ExecutionConfig, InferenceConfig, IVPPMLConfig, ivppmlhdfe
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.errors import UnderidentifiedError
from econhdfe.hdfe.plan import FEPlan
from econhdfe.iv.solve import weighted_2sls, weighted_2sls_block
from econhdfe.models.ppml_iv.heterogeneous import fit_block_arrays


def _block(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return BlockDesign(
        len(a), a.shape[1],
        (DenseDesignBlock(0, np.arange(len(a)), np.arange(a.shape[1]), a.copy()),),
    )


@pytest.mark.parametrize("path", ["dense", "block"])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_weighted_iv_rejects_zero_cross_rank_in_original_units(path, scale):
    # Instruments and regressors each have rank one; their cross moment is
    # exactly zero. Renormalizing a roundoff-only projection must not create
    # an identified direction or a zero standard error.
    x = scale * np.array([-1.0, -1.0, 1.0, 1.0])[:, None]
    z = np.array([-1.0, 1.0, -1.0, 1.0])[:, None] / scale
    w = np.array([1.0, 2.0, 1.0, 2.0])
    solver = weighted_2sls if path == "dense" else weighted_2sls_block
    if path == "block":
        x, z = _block(x), _block(z)
    with pytest.raises(UnderidentifiedError) as err:
        solver(np.ones(4), x, z, weights=w)
    assert err.value.code == "identification.underidentified"
    assert err.value.details == {"rank": 0, "n_parameters": 1}


@pytest.mark.parametrize("weighted,change_units", [(False, False), (True, True)])
def test_block_iv_self_instrumentation_uses_unsquared_identification(weighted, change_units):
    # X=Z is ordinary WLS, hence identified whenever X has full column rank.
    # Here s_min(X)=O(delta), but s_min(Z'X)=O(delta**2). A rank test of the
    # latter with the former's tolerance falsely rejected the block path;
    # its previous Gram solve also returned materially wrong coefficients.
    n = 100
    delta = 2e-7
    a = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    b = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    base = np.c_[a, a + delta * b]
    y = a + 0.5 * delta * b
    w = 1.0 + np.arange(n) % 4 if weighted else np.ones(n)
    xs = np.array([1e-7, 1e5]) if change_units else np.ones(2)
    zs = np.array([1e3, -1e-2]) if change_units else np.ones(2)
    X, Z = base * xs, base * zs

    def partitioned(values):
        blocks = []
        for index, lo in enumerate(range(0, n, 7)):
            rows = np.arange(lo, min(lo + 7, n))
            blocks.append(DenseDesignBlock(index, rows, np.arange(2), values[rows].copy()))
        return BlockDesign(n, 2, tuple(blocks))

    # Independent observation-space WLS: no production Gram/IV routines.
    u, singular, vt = np.linalg.svd(base * np.sqrt(w)[:, None], full_matrices=False)
    inverse = (vt.T / singular) @ u.T
    expected_bread = inverse @ inverse.T
    for result in (
        weighted_2sls(y, X, Z, weights=w),
        weighted_2sls_block(y, partitioned(X), partitioned(Z), weights=w),
    ):
        assert result.rank == 2
        np.testing.assert_allclose(result.beta * xs, [0.5, 0.5], rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(result.bread * xs[:, None] * xs[None, :], expected_bread, rtol=2e-8)
        np.testing.assert_allclose(result.fitted, y, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("mode", ["stata_compat", "unified"])
def test_ivppml_rejects_continuum_of_exact_moment_roots(mode):
    # For every beta, alpha=-log(cosh(beta)) satisfies both moments. This is
    # exact nonidentification, not merely a weak-instrument standard error.
    x = np.array([-1.0, -1.0, 1.0, 1.0])
    z = np.array([-1.0, 1.0, -1.0, 1.0])
    for b in (-1.0, 0.0, 1.0):
        mu = np.exp(-np.log(np.cosh(b)) + b * x)
        np.testing.assert_allclose(np.c_[np.ones(4), z].T @ (1.0 - mu), 0, atol=1e-14)
    with pytest.raises(UnderidentifiedError):
        ivppmlhdfe(
            y=np.ones(4), endog=x, instruments=z,
            config=IVPPMLConfig(singleton_mode=mode),
            execution_config=ExecutionConfig(threads=1),
        )


@pytest.mark.parametrize("mode", ["stata_compat", "unified"])
@pytest.mark.parametrize("path", ["dense", "block"])
def test_iv_default_retains_ppml_separated_but_iv_identified_rows(mode, path):
    # Additive moments give 2*mu0+2*mu1=4 and 2*mu0+4*mu1=6,
    # hence the unique IV solution is mu0=mu1=1. PPML would delete x=1.
    y = np.array([1.0, 3.0, 0.0, 0.0])
    x = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 2.0, 1.0, 3.0])
    cfg = IVPPMLConfig(singleton_mode=mode, tolerance=1e-11)
    execution = ExecutionConfig(threads=1)
    if path == "dense":
        fit = ivppmlhdfe(y=y, endog=x, instruments=z, config=cfg, execution_config=execution)
    else:
        fit = fit_block_arrays(
            y, _block(np.empty((4, 0))), _block(x), _block(z),
            FEPlan.from_arrays([np.zeros(4, dtype=int)]), offset=np.zeros(4),
            true_w=np.ones(4), endog_names=("x",), instrument_names=("z",),
            config=cfg, execution_config=execution,
        )
    assert fit.converged and fit.sample_mask.all() and not fit.separation_mask.any()
    np.testing.assert_allclose(fit.mu, 1.0, atol=2e-10)
    np.testing.assert_allclose(fit.params, 0.0, atol=2e-10)
    np.testing.assert_allclose(np.c_[np.ones(4), z].T @ (y - fit.mu), 0.0, atol=2e-10)


def test_explicit_legacy_iv_separation_warns_about_estimand():
    with pytest.warns(RuntimeWarning, match="do not certify existence of IV moments"):
        IVPPMLConfig(separation=("fe", "simplex", "relu")).validate()


def _panel():
    rng = np.random.default_rng(1107)
    n = 900
    g = np.arange(n) % 3
    z = rng.normal(size=(n, 2))
    v = rng.normal(size=n)
    x = 0.8 * z[:, 0] + 0.4 * z[:, 1] + 0.7 * v
    off = 0.1 * np.sin(np.arange(n))
    y = np.exp(0.35 * x + 0.24 * z[:, 0] ** 2 + 0.25 * v + np.array([0.1, 0.3, -0.2])[g] + off)
    return y, x, z, g, off


@pytest.mark.parametrize("path,kind,weight_kind,ee", [
    ("dense", "robust", "none", False),
    ("dense", "robust", "fweight", True),
    ("dense", "cluster", "pweight", True),
    ("block", "robust", "fweight", False),
    ("block", "cluster", "pweight", False),
])
def test_iv_final_inference_preserves_outcome_units_and_joint_state(path, kind, weight_kind, ee):
    y, x, z, g, off = _panel()
    n = len(y)
    w = np.ones(n) if weight_kind == "none" else 1.0 + np.arange(n) % 4
    groups = [np.arange(n) % 12] if kind == "cluster" else None
    cfg = IVPPMLConfig(singleton_mode="unified", tolerance=1e-12, standardize=True)
    execution = ExecutionConfig(threads=1)

    def estimate(shift):
        if path == "dense":
            return ivppmlhdfe(
                y=y * np.exp(shift), endog=x, instruments=z, absorb=[g], offset=off,
                weights=None if weight_kind == "none" else w,
                weight_type=None if weight_kind == "none" else weight_kind,
                vce=kind, cluster=groups, config=cfg, execution_config=execution,
                inference_config=InferenceConfig(ivppml_misspecification_robust=ee),
            )
        return fit_block_arrays(
            y * np.exp(shift), _block(np.empty((n, 0))), _block(x), _block(z),
            FEPlan.from_arrays([g]), offset=off, true_w=w, weight_kind=weight_kind,
            vce=kind, clusters=groups, endog_names=("x",), instrument_names=("z0", "z1"),
            config=cfg, execution_config=execution,
        )

    baseline = estimate(0.0)
    for shift in (-60.0, -45.0, 40.0):
        fit = estimate(shift)
        np.testing.assert_allclose(fit.params[:-1], baseline.params[:-1], rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(fit.params[-1] - shift, baseline.params[-1], atol=2e-9)
        np.testing.assert_allclose(fit.vcov, baseline.vcov, rtol=2e-8, atol=2e-12)
        np.testing.assert_allclose(fit.mu / np.exp(shift), baseline.mu, rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(fit.eta - shift, baseline.eta, atol=2e-9)
        np.testing.assert_allclose(np.exp(fit.eta), fit.mu, rtol=2e-14)
        assert fit.iterations == baseline.iterations
        # Removing the reported slope and known offset must leave an index
        # constant within each FE level, including after the final solve.
        nuisance = fit.eta - x * fit.params[0] - off
        assert max(np.ptp(nuisance[g == j]) for j in range(3)) < 2e-9
        if path == "dense":
            np.testing.assert_allclose(
                fit.diagnostics["overidentification"]["stat"],
                baseline.diagnostics["overidentification"]["stat"], rtol=2e-8,
            )

    # Independent full-dummy projection checks the overidentified estimating
    # equation. Individual raw Z moments need not vanish in this fixture.
    D = np.eye(3)[g]
    Q = np.column_stack([D, z])
    mu = baseline.mu
    gamma = np.linalg.solve(Q.T @ ((w * mu)[:, None] * Q), Q.T @ (w * mu * x))
    u = w * (y - mu)
    moment = np.r_[D.T @ u, (Q @ gamma) @ u]
    assert np.max(np.abs(moment)) / np.sum(w * y) < 2e-8

