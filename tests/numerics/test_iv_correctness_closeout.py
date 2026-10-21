"""Independent matrix references for IV identification and output semantics."""

import numpy as np
import pytest
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.errors import UnderidentifiedError, NumericalError
from econhdfe.models.linear_iv.estimators import (
    fit_iv_kclass, fit_iv_kclass_block, fit_iv_gmm2s,
)


def _block(a):
    n, k = a.shape
    return BlockDesign(n, k, (DenseDesignBlock(0, np.arange(n), np.arange(k), a),))


def _fit(y, c, e, z, path, **kwargs):
    if path == "block":
        return fit_iv_kclass_block(y, *map(_block, (c, e, z)), **kwargs)
    if path == "gmm":
        return fit_iv_gmm2s(y, c, e, z, **kwargs)
    return fit_iv_kclass(y, c, e, z, **kwargs)


def _regular():
    rng = np.random.default_rng(20260930)
    n = 180
    c = np.column_stack((np.ones(n), rng.normal(size=n)))
    z = rng.normal(size=(n, 3))
    e = z[:, :2] + .3 * c[:, 1:] + rng.normal(size=(n, 2))
    y = c @ np.array([1., -.4]) + e @ np.array([.8, 1.3]) + rng.normal(size=n)
    w = rng.uniform(.3, 2.5, n)
    return y, c, e, z, w


@pytest.mark.parametrize('path,kappa,vce', [
    ('dense', 0.0, 'iid'),
    ('block', 0.4, 'robust'),
    ('dense', 1.0, 'cluster'),
])
def test_fixed_kclass_covariance_matches_defining_score(path, kappa, vce):
    y, c, e, z, w = _regular()
    n = len(y)
    x = np.column_stack((c, e)) * np.sqrt(w)[:, None]
    zz = np.column_stack((c, z)) * np.sqrt(w)[:, None]
    yy = y * np.sqrt(w)
    # Independent observation-space definition, deliberately small n.
    pz = zz @ np.linalg.pinv(zz)
    h = (1 - kappa) * x + kappa * (pz @ x)
    bread = np.linalg.inv(h.T @ x)
    beta = bread @ (h.T @ yy)
    u = yy - x @ beta
    groups = np.arange(n) % 12
    if vce == "iid":
        meat = h.T @ h * (u @ u) / (n - x.shape[1])
    elif vce == "robust":
        scores = h * u[:, None]
        meat = scores.T @ scores * n / (n - x.shape[1])
    else:
        scores = np.array([(h * u[:, None])[groups == g].sum(axis=0) for g in range(12)])
        meat = scores.T @ scores * 12 / 11 * (n - 1) / (n - x.shape[1])
    expected = bread @ meat @ bread.T
    got = _fit(y, c, e, z, path, weights=w, vce=vce,
               clusters=[groups] if vce == "cluster" else None,
               estimator="kclass", kappa=kappa)
    np.testing.assert_allclose(got[0], beta, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(got[1], expected, rtol=2e-10, atol=2e-11)


def _identification_problem(delta):
    rng = np.random.default_rng(31093)
    # Orthogonal observation vectors separate full role rank from IV rank.
    q = np.linalg.qr(rng.normal(size=(128, 6)))[0]
    c, z = q[:, :1], q[:, 1:3]
    e = np.column_stack((q[:, 1] + q[:, 3], q[:, 1] + q[:, 4] + delta * q[:, 2]))
    y = e @ np.array([1., 2.]) + q[:, 5]
    return y, c, e, z


@pytest.mark.parametrize('path,delta,error', [
    ('dense', 0., UnderidentifiedError),
    ('block', 1e-10, NumericalError),
    ('gmm', 0., UnderidentifiedError),
])
def test_identification_policy_is_invariant_to_column_units(path, delta, error):
    y, c, e, z = _identification_problem(delta)
    with pytest.raises(error):
        _fit(y, c * 1e8, e * np.array([1e-7, 1e7]),
             z * np.array([1e6, 1e-6]), path)


def test_near_identification_matches_exact_float_input_moments():
    path,delta = ('block', 1e-06)
    from fractions import Fraction as F
    from scipy.linalg import hadamard
    q = np.tile(hadamard(8)[:, 1:6], (250, 1)).astype(float)
    z = q[:, :2]
    e = np.column_stack((q[:, 0] + q[:, 2], q[:, 0] + delta * q[:, 1] + q[:, 3]))
    y = e[:, 0] + 2 * e[:, 1] + .3 * q[:, 4]
    # Exact rational arithmetic on the actual float64 inputs, independent
    # of the package's projection, rank, and least-squares implementations.
    def dot(a, b):
        return sum((F(float(x)) * F(float(v)) for x, v in zip(a, b)), F(0))
    a, b = dot(z[:, 0], e[:, 0]), dot(z[:, 0], e[:, 1])
    c, d = dot(z[:, 1], e[:, 0]), dot(z[:, 1], e[:, 1])
    r, t = dot(z[:, 0], y), dot(z[:, 1], y)
    determinant = a * d - b * c
    expected = [float((r * d - b * t) / determinant), float((a * t - c * r) / determinant)]
    got = _fit(y, np.empty((len(y), 0)), e, z, path)
    assert got[3] == 2
    np.testing.assert_allclose(got[0], expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize('path,vce,weight_kind', [
    ('dense', 'robust', 'fweight'),
    ('block', 'cluster', 'fweight'),
])
def test_overid_uses_efficient_moments_and_iv_small_convention(path, vce, weight_kind):
    from econhdfe.compute.weights import prepare_weights
    from scipy.stats import chi2
    y, c, e, z, w = _regular()
    if weight_kind == "fweight":
        w = np.arange(len(y)) % 3 + 1
    if weight_kind is None:
        w = None
    wi = prepare_weights(w, len(y), weight_kind)
    sw = np.ones(len(y)) if wi.estimation is None else np.sqrt(wi.estimation)
    X = np.column_stack([c, e]) * sw[:, None]
    Z = np.column_stack([c, z]) * sw[:, None]
    yy = y * sw
    # Explicit observation-space projection, independent of the package core.
    xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.lstsq(xhat, yy, rcond=None)[0]
    u = yy - X @ b
    neff = wi.effective_n
    groups = np.arange(len(y)) % 12
    scores = Z * u[:, None]
    if vce == "iid":
        S = (Z.T @ Z) * (u @ u) / neff
    elif vce == "cluster":
        sums = np.array([scores[groups == g].sum(axis=0) for g in range(12)])
        S = sums.T @ sums
    else:
        if weight_kind == "fweight":
            scores = scores / np.sqrt(w)[:, None]
        S = scores.T @ scores
    W = np.linalg.inv(S)
    D = Z.T @ X
    b2 = np.linalg.solve(D.T @ W @ D, D.T @ W @ (Z.T @ yy))
    g2 = Z.T @ (yy - X @ b2)
    expected_j = g2 @ W @ g2
    got = _fit(y, c, e, z, path, weights=wi.estimation, weight_info=wi,
               vce=vce, clusters=[groups] if vce == "cluster" else None,
               df_absorbed=3, nested_adj=1)
    np.testing.assert_allclose(got[0], b, rtol=2e-11, atol=2e-11)
    over = got[6]["overidentification"]
    np.testing.assert_allclose([over["stat"], over["pvalue"]],
                               [expected_j, chi2.sf(expected_j, 1)], rtol=2e-10, atol=2e-11)
    # Nested FE use the IV denominator N-K, without OLS's extra -1.
    bread = np.linalg.inv(xhat.T @ xhat)
    if vce == "iid":
        V = bread * (u @ u) / (neff - X.shape[1] - 3)
    else:
        xscores = xhat * u[:, None]
        if vce == "cluster":
            sums = np.array([xscores[groups == g].sum(axis=0) for g in range(12)])
            meat = sums.T @ sums * 12/11 * (neff-1)/(neff-X.shape[1]-3)
        else:
            if weight_kind == "fweight":
                xscores = xscores / np.sqrt(w)[:, None]
            meat = xscores.T @ xscores * neff/(neff-X.shape[1]-3)
        V = bread @ meat @ bread
    np.testing.assert_allclose(got[1], V, rtol=2e-10, atol=2e-11)
