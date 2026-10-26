from __future__ import annotations
import numpy as np
from econhdfe import ivppmlhdfe, IVPPMLConfig


def ivcfg(**kw):
    d = dict(
        separation=(), standardize=False, tolerance=1e-9,
        target_inner_tol=1e-10, max_iter=300,
    )
    d.update(kw)
    return IVPPMLConfig(**d)


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
    np.testing.assert_allclose(raw.params, std.params, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(raw.stderr, std.stderr, atol=2e-8, rtol=2e-8)


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
            y=y, exog=None, endog=E, instruments=Z, absorb=[g],
            vce="robust", config=ivcfg(),
        )


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
        y=y, exog=None, endog=e, instruments=z,
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
        ivppmlhdfe(y=y, exog=None, endog=e, instruments=z, vce="robust", config=cfg)
