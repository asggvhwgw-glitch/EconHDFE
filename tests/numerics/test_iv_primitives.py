from __future__ import annotations

from pathlib import Path
import numpy as np

from econhdfe.iv import IVDesign, additive_moments, weighted_2sls
from econhdfe.models.linear_iv.estimators import fit_iv_kclass


def test_iv_design_is_outcome_agnostic_and_builds_full_x_z():
    n = 8
    c = np.arange(n, dtype=float)[:, None]
    e = (np.arange(n, dtype=float) + 1)[:, None]
    z = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    d = IVDesign.from_arrays(c, e, z, n)
    assert d.X.shape == (n, 2)
    assert d.Z.shape == (n, 3)
    np.testing.assert_array_equal(d.X[:, 0], c[:, 0])
    np.testing.assert_array_equal(d.X[:, 1], e[:, 0])
    np.testing.assert_array_equal(d.Z[:, 0], c[:, 0])
    np.testing.assert_array_equal(d.Z[:, 1:], z)


def test_shared_weighted_2sls_matches_linear_iv_2sls_core():
    rng = np.random.default_rng(4201)
    n = 700
    c = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 2))
    e = 0.8 * z[:, :1] - 0.3 * z[:, 1:] + 0.4 * c[:, :1] + rng.normal(size=(n, 1))
    y = 0.5 * c[:, 0] - 0.2 * c[:, 1] + 1.4 * e[:, 0] + rng.normal(size=n)
    w = np.exp(0.15 * rng.normal(size=n))

    d = IVDesign.from_arrays(c, e, z, n)
    shared = weighted_2sls(y, d.X, d.Z, weights=w)
    beta, *_ = fit_iv_kclass(y, c, e, z, weights=w, estimator="2sls")
    np.testing.assert_allclose(shared.beta, beta, rtol=1e-12, atol=1e-12)


def test_additive_moment_primitive_matches_direct_formula():
    rng = np.random.default_rng(4202)
    n = 200
    q = rng.normal(size=(n, 4))
    r = rng.normal(size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    got = additive_moments(r, q, weights=w, normalize=False)
    expected = q.T @ (w * r)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
