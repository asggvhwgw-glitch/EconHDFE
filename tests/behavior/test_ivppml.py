from __future__ import annotations
import numpy as np
import pandas as pd

from econhdfe import (
    ivppmlhdfe, IVPPMLConfig, IVPPMLHDFE,
    ppmlhdfe, PPMLConfig,
)
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml_iv.irls import fit_iv_irls
from econhdfe.models.ppml_iv.vce import ivppml_vcov
from econhdfe.iv.solve import weighted_2sls


def ivcfg(**kw):
    d = dict(
        separation=(), standardize=False, tolerance=1e-9,
        target_inner_tol=1e-10, max_iter=300,
    )
    d.update(kw)
    return IVPPMLConfig(**d)


def test_iv_irls_reduces_to_ppml_when_instrument_space_equals_x():
    rng = np.random.default_rng(9001)
    n = 2500
    c = rng.normal(size=(n, 1))
    e = rng.normal(size=(n, 1))
    X = np.column_stack([c, e])
    y = rng.poisson(np.exp(0.25 + 0.18*c[:, 0] - 0.13*e[:, 0]))
    pp = ppmlhdfe(
        y, X, vce="model",
        config=PPMLConfig(
            standardize=False, separation=(), tolerance=1e-9,
            target_inner_tol=1e-10,
        ),
    )
    plan = FEPlan.from_arrays([np.zeros(n, dtype=np.int8)])
    iv = fit_iv_irls(
        y, c, e, e, plan, np.ones(n), np.zeros(n), ivcfg(),
        projector=plan.projector(engine="replica"),
    )
    assert iv.converged
    np.testing.assert_allclose(iv.beta, pp.coef[:2], atol=2e-9, rtol=2e-9)


def test_public_ivppml_recovers_simple_one_fe_dgp_and_moments():
    rng = np.random.default_rng(9002)
    n = 6000
    g = rng.integers(0, 100, n)
    c = rng.normal(size=n)
    z = rng.normal(size=n)
    e = 0.85*z + 0.25*c + rng.normal(scale=.55, size=n)
    a = rng.normal(scale=.2, size=100)
    beta = np.array([.22, .38])
    mu = np.exp(a[g] + beta[0]*c + beta[1]*e)
    y = rng.poisson(mu)
    r = ivppmlhdfe(
        y, exog=c, endog=e, instruments=z, absorb=[g],
        exog_names=["c"], endog_names=["e"], instrument_names=["z"],
        vce="robust", config=ivcfg(),
    )
    assert r.converged and r.nobs == n
    np.testing.assert_allclose(r.coef[:2], beta, atol=.06, rtol=0)
    assert np.max(np.abs(r.moments)) < 1e-7
    assert r.names == ("c", "e", "_cons")
    assert np.all(np.isfinite(r.stderr))


def test_ivppml_replica_and_optimized_engines_agree_two_way():
    rng = np.random.default_rng(9003)
    n = 7000
    g1 = rng.integers(0, 120, n)
    g2 = rng.integers(0, 55, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    e = .75*z + .2*c + rng.normal(scale=.6, size=n)
    a = rng.normal(scale=.15, size=120)
    d = rng.normal(scale=.12, size=55)
    y = rng.poisson(np.exp(a[g1] + d[g2] + .12*c + .28*e))
    args = dict(y=y, exog=c, endog=e, instruments=z, absorb=[g1, g2], vce="robust")
    rr = ivppmlhdfe(**args, config=ivcfg(engine="replica"))
    oo = ivppmlhdfe(**args, config=ivcfg(engine="optimized"))
    np.testing.assert_allclose(rr.coef, oo.coef, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(rr.stderr, oo.stderr, atol=2e-7, rtol=2e-7)


def test_ivppml_multiple_endogenous_overidentified_smoke():
    rng = np.random.default_rng(9004)
    n = 5000
    g = rng.integers(0, 80, n)
    z = rng.normal(size=(n, 3))
    c = rng.normal(size=(n, 1))
    e1 = .8*z[:, 0] - .2*z[:, 2] + rng.normal(scale=.6, size=n)
    e2 = .7*z[:, 1] + .25*z[:, 2] + rng.normal(scale=.6, size=n)
    E = np.column_stack([e1, e2])
    a = rng.normal(scale=.2, size=80)
    y = rng.poisson(np.exp(a[g] + .1*c[:, 0] + .24*e1 - .16*e2))
    r = ivppmlhdfe(
        y, exog=c, endog=E, instruments=z, absorb=[g],
        config=ivcfg(standardize=True), vce="robust",
    )
    assert r.converged
    assert len(r.coef) == 4  # exog + 2 endog + normalized constant
    assert r.diagnostics["rank"] == 3
    assert r.diagnostics["first_stage"].shape == (4, 3)


def test_ivppml_vce_matches_explicit_robust_formula():
    rng = np.random.default_rng(9006)
    n = 1800
    Z = rng.normal(size=(n, 4))
    X = Z @ rng.normal(size=(4, 2)) + rng.normal(scale=.4, size=(n, 2))
    y = rng.normal(size=n)
    w = np.exp(rng.normal(scale=.2, size=n))
    solved = weighted_2sls(y, X, Z, weights=w)
    Xhat = solved.projected_X
    bread = np.linalg.pinv((Xhat.T @ (w[:, None]*X) + (Xhat.T @ (w[:, None]*X)).T)/2, hermitian=True)
    S = Xhat * (w*solved.residuals)[:, None]
    expected = bread @ ((n/(n-1))*S.T@S) @ bread
    got = ivppml_vcov(Xhat, X, solved.residuals, w, kind="robust")
    np.testing.assert_allclose(got, expected, atol=3e-11, rtol=3e-11)


def test_ivppml_class_c_three_fe_smoke():
    rng = np.random.default_rng(9008)
    ne, ni, T = 9, 8, 6
    rows = [(e, i, t) for e in range(ne) for i in range(ni) for t in range(T)]
    exp = np.array([r[0] for r in rows]); imp = np.array([r[1] for r in rows]); year = np.array([r[2] for r in rows])
    pair = exp*ni + imp
    ey = exp*T + year; iy = imp*T + year
    n = len(rows)
    z = rng.normal(size=n)
    evar = .8*z + rng.normal(scale=.6, size=n)
    pair_fe = rng.normal(scale=.12, size=ne*ni)
    ey_fe = rng.normal(scale=.1, size=ne*T)
    iy_fe = rng.normal(scale=.1, size=ni*T)
    y = rng.poisson(np.exp(pair_fe[pair] + ey_fe[ey] + iy_fe[iy] + .2*evar))
    r = ivppmlhdfe(
        y, exog=None, endog=evar, instruments=z, absorb=[pair, ey, iy],
        vce="robust", config=ivcfg(engine="optimized", tolerance=1e-8, target_inner_tol=1e-9),
    )
    assert r.converged
    assert np.isfinite(r.coef[0]) and abs(r.coef[0] - .2) < .15


def test_ivppml_standardization_is_numerically_invariant():
    rng = np.random.default_rng(9009)
    n = 6000
    g1 = rng.integers(0, 100, n)
    g2 = rng.integers(0, 40, n)
    c = 3.0 * rng.normal(size=n)
    z = 0.15 * rng.normal(size=n)
    e = 4.0*z + .08*c + rng.normal(scale=.45, size=n)
    a = rng.normal(scale=.15, size=100)
    b = rng.normal(scale=.12, size=40)
    y = rng.poisson(np.exp(a[g1] + b[g2] + .035*c + .27*e))
    kwargs = dict(
        y=y, exog=c, endog=e, instruments=z, absorb=[g1, g2],
        vce="robust",
    )
    raw = ivppmlhdfe(**kwargs, config=ivcfg(engine="optimized", standardize=False))
    std = ivppmlhdfe(**kwargs, config=ivcfg(engine="optimized", standardize=True))
    np.testing.assert_allclose(raw.coef, std.coef, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(raw.stderr, std.stderr, atol=2e-8, rtol=2e-8)


def test_ivppml_prunes_duplicate_excluded_instrument_and_stays_identified():
    rng = np.random.default_rng(9010)
    n = 4500
    g = rng.integers(0, 75, n)
    z = rng.normal(size=n)
    Z = np.column_stack([z, 2.0*z])
    c = rng.normal(size=n)
    e = .8*z + .15*c + rng.normal(scale=.55, size=n)
    a = rng.normal(scale=.2, size=75)
    y = rng.poisson(np.exp(a[g] + .1*c + .3*e))
    r = ivppmlhdfe(
        y, exog=c, endog=e, instruments=Z, absorb=[g],
        instrument_names=["z", "zdup"], vce="robust", config=ivcfg(),
    )
    assert r.converged
    assert len(r.instrument_names) == 1
    assert r.instrument_names[0] in {"z", "zdup"}
    assert r.diagnostics["first_stage"].shape == (2, 2)


def test_ivppml_rechecks_identification_after_instrument_pruning():
    rng = np.random.default_rng(9011)
    n = 2500
    g = rng.integers(0, 50, n)
    z = rng.normal(size=n)
    Z = np.column_stack([z, -3.0*z])
    e1 = .8*z + rng.normal(scale=.55, size=n)
    e2 = -.5*z + rng.normal(scale=.55, size=n)
    E = np.column_stack([e1, e2])
    a = rng.normal(scale=.15, size=50)
    y = rng.poisson(np.exp(a[g] + .15*e1 - .12*e2))
    with np.testing.assert_raises_regex(ValueError, "not identified"):
        ivppmlhdfe(
            y, exog=None, endog=E, instruments=Z, absorb=[g],
            vce="robust", config=ivcfg(),
        )


def test_ivppml_quadvariance_standardization_matches_stata_formula():
    from econhdfe.models.ppml_iv.standardize import IVPPMLStandardization
    X = np.array([[1.0, 4.0], [3.0, 2.0], [7.0, 9.0], [11.0, -1.0]])
    Z = np.array([[2.0], [5.0], [8.0], [13.0]])
    w = np.array([1.0, 2.0, 4.0, 3.0])
    n = len(w); sw = w.sum()
    def oracle(A):
        m = (w @ A) / sw
        return np.sqrt(n / (sw * (n - 1.0)) * np.sum(w[:, None] * (A - m) ** 2, axis=0))
    std = IVPPMLStandardization.fit(X, Z, w)
    np.testing.assert_allclose(std.x_scale, oracle(X), rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(std.z_scale, oracle(Z), rtol=2e-15, atol=2e-15)


def test_ivppml_fweight_matches_physical_replication_robust_vce():
    rng = np.random.default_rng(9012)
    n = 1200
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    e = .85*z + .15*c + rng.normal(scale=.55, size=n)
    y = rng.poisson(np.exp(.15 + .08*c + .25*e))
    fw = rng.integers(1, 4, size=n)
    cfg = ivcfg(tolerance=1e-8, target_inner_tol=1e-9)
    compact = ivppmlhdfe(
        y, exog=c, endog=e, instruments=z, weights=fw, weight_type="fweight",
        vce="robust", config=cfg,
    )
    idx = np.repeat(np.arange(n), fw)
    expanded = ivppmlhdfe(
        y[idx], exog=c[idx], endog=e[idx], instruments=z[idx],
        vce="robust", config=cfg,
    )
    assert compact.nobs == len(idx)
    assert compact.diagnostics["n_rows"] == n
    np.testing.assert_allclose(compact.coef, expanded.coef, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(compact.vcov, expanded.vcov, rtol=3e-6, atol=3e-8)


def test_ivppml_pweight_is_scale_invariant_for_coef_and_robust_vce():
    rng = np.random.default_rng(9013)
    n = 1800
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    e = .8*z + .1*c + rng.normal(scale=.6, size=n)
    y = rng.poisson(np.exp(.1 + .12*c + .22*e))
    pw = np.exp(rng.normal(scale=.4, size=n))
    cfg = ivcfg(tolerance=1e-8, target_inner_tol=1e-9)
    a = ivppmlhdfe(y, exog=c, endog=e, instruments=z, weights=pw, weight_type="pweight", vce="robust", config=cfg)
    b = ivppmlhdfe(y, exog=c, endog=e, instruments=z, weights=17.0*pw, weight_type="pweight", vce="robust", config=cfg)
    assert a.nobs == b.nobs == n
    np.testing.assert_allclose(a.coef, b.coef, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=2e-8, atol=2e-10)


def test_ivppml_mu_separation_keeps_sample_and_reports_mask():
    rng = np.random.default_rng(9014)
    n = 2200
    z = rng.normal(size=n)
    e = z + rng.normal(scale=.25, size=n)
    y = rng.poisson(np.exp(.1 - .75*e))
    # One estimable zero lies far into the negative-linear-predictor tail.
    z[-1] = 35.0
    e[-1] = 35.0
    y[-1] = 0
    r = ivppmlhdfe(
        y, exog=None, endog=e, instruments=z,
        config=ivcfg(separation=("mu",), tolerance=1e-8, target_inner_tol=1e-9),
        vce="robust",
    )
    assert r.converged
    assert r.diagnostics["num_sep_mu"] >= 1
    assert r.sample_mask[-1]
    assert r.separation_mask[-1]
    assert r.nobs == n


def test_ivppml_nonconvergence_fails_loudly():
    rng = np.random.default_rng(9015)
    n = 1800
    z = rng.normal(size=n)
    e = .8*z + rng.normal(scale=.5, size=n)
    y = rng.poisson(np.exp(.2 + .3*e))
    cfg = ivcfg(max_iter=1, tolerance=1e-12, target_inner_tol=1e-12)
    with np.testing.assert_raises_regex(RuntimeError, "failed to converge"):
        ivppmlhdfe(y, exog=None, endog=e, instruments=z, vce="robust", config=cfg)
