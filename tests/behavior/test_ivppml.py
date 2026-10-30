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


def _overid_panel(seed, n=3000, misspec=0.0, strength=1.0):
    """Two FEs, one endogenous and one exogenous regressor, three instruments.

    ``misspec`` lets the last instrument enter the outcome directly, so the
    overidentifying moments fail at the pseudo-true slope.
    """
    rng = np.random.default_rng(seed)
    g1, g2 = rng.integers(0, 6, n), rng.integers(0, 5, n)
    z = rng.normal(size=(n, 3))
    v = rng.normal(size=n)
    c = rng.normal(size=n)
    e = strength * z.sum(1) / np.sqrt(3) + .5 * rng.normal(0, .4, 6)[g1] + v
    eta = .3 + .5 * e - .3 * c + .3 * (.6 * v + .8 * rng.normal(size=n)) + misspec * z[:, 2]
    y = rng.poisson(np.exp(eta + rng.normal(0, .4, 6)[g1] + rng.normal(0, .4, 5)[g2])).astype(float)
    return y, c, e, z, [g1, g2]


def test_ivppml_estimates_are_invariant_to_outcome_units():
    # IRLS safeguards used to be absolute: rescaling y failed to converge,
    # stopped early or reported divergence. Slopes and SEs must not move.
    y, c, e, z, fe = _overid_panel(4, strength=0.15)
    fits = {s: ivppmlhdfe(y=y * np.exp(s), exog=c, endog=e, instruments=z, absorb=fe)
            for s in (-20.0, 0.0, 28.0)}
    for s, r in fits.items():
        np.testing.assert_allclose(r.params[:-1], fits[0.0].params[:-1], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(r.stderr, fits[0.0].stderr, rtol=1e-8)
        np.testing.assert_allclose(r.params[-1], fits[0.0].params[-1] + s, atol=1e-8)


def _stacked_m_vcov(y, X, Z, D, beta, eta):
    """Robust sandwich of (beta, FE, first stage [Z D] -> X) by central differences."""
    n, k = X.shape
    L, p = Z.shape[1], D.shape[1]
    R = np.column_stack([Z, D])
    W = np.exp(eta)
    first = np.linalg.solve(R.T @ (W[:, None] * R), R.T @ (W[:, None] * X))
    phi0 = np.r_[beta, np.linalg.lstsq(D, eta - X @ beta, rcond=None)[0], first.ravel()]

    def scores(phi):
        b, a, P = phi[:k], phi[k:k + p], phi[k + p:].reshape(L + p, k)
        mu = np.exp(X @ b + D @ a)
        r = y - mu
        stage = (R * mu[:, None])[:, :, None] * (X - R @ P)[:, None, :]
        return np.column_stack([(Z @ P[:L]) * r[:, None], D * r[:, None], stage.reshape(n, -1)])

    J = np.empty((len(phi0), len(phi0)))
    for j in range(len(phi0)):
        step = np.zeros(len(phi0))
        step[j] = 1e-6 * max(1.0, abs(phi0[j]))
        J[:, j] = (scores(phi0 + step).sum(0) - scores(phi0 - step).sum(0)) / (2 * step[j])
    g = scores(phi0)
    Jinv = np.linalg.inv(J)
    return (Jinv @ (g.T @ g) @ Jinv.T)[:k, :k]


def test_ivppml_misspecification_robust_vce_and_overid_test():
    from scipy.stats import chi2
    from econhdfe import InferenceConfig
    y, c, e, z, fe = _overid_panel(11, misspec=0.5)
    standard = ivppmlhdfe(y=y, exog=c, endog=e, instruments=z, absorb=fe)
    robust = ivppmlhdfe(y=y, exog=c, endog=e, instruments=z, absorb=fe,
                        inference_config=InferenceConfig(ivppml_misspecification_robust=True))
    np.testing.assert_array_equal(robust.params, standard.params)

    s = standard.sample_mask
    D = np.column_stack([(fe[0][s][:, None] == np.arange(6)), (fe[1][s][:, None] == np.arange(1, 5))]).astype(float)
    X, Z = np.column_stack([c, e])[s], np.column_stack([c, z])[s]
    n = int(s.sum())
    oracle = np.sqrt(np.diag(_stacked_m_vcov(y[s], X, Z, D, standard.params[:-1], standard.eta[s])))
    # ivppmlhdfe's robust convention scales the meat by N/(N-1).
    np.testing.assert_allclose(robust.stderr[:-1], oracle * np.sqrt(n / (n - 1)), rtol=1e-6)
    assert standard.stderr[1] < 0.95 * robust.stderr[1]  # standard SE misses the first-stage terms

    # Hansen J equals the orthogonalized moment Wald statistic N'm (N'SN)^-1 N'm.
    mu = standard.mu[s]
    Wsq = np.sqrt(mu)[:, None]
    resid = lambda A: A - D @ np.linalg.lstsq(D * Wsq, A * Wsq, rcond=None)[0]
    Zt, Xt = resid(Z), resid(X)
    scores = Zt * (y[s] - mu)[:, None]
    N = np.linalg.svd(Zt.T @ (mu[:, None] * Xt))[0][:, 2:]
    m = N.T @ scores.sum(0)
    stat = float(m @ np.linalg.solve(N.T @ (scores.T @ scores) @ N, m))
    over = standard.diagnostics["overidentification"]
    assert over["df"] == 2
    np.testing.assert_allclose([over["stat"], over["pvalue"]], [stat, chi2.sf(stat, 2)], rtol=1e-6)
