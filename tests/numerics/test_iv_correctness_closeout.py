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


@pytest.mark.parametrize("path", ["dense", "block", "gmm"])
def test_weighted_first_stage_fitted_is_in_unweighted_within_coordinates(path):
    y, c, e, z, w = _regular()
    got = _fit(y, c, e, z, path, weights=w, vce="robust")
    zw = np.column_stack((c, z)) * np.sqrt(w)[:, None]
    reference = np.linalg.lstsq(zw, e * np.sqrt(w)[:, None], rcond=None)[0]
    expected = np.column_stack((c, z)) @ reference
    np.testing.assert_allclose(got[5]["fitted_endog"], expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("path", ["dense", "block"])
@pytest.mark.parametrize("kappa", [0., .4, 1.])
@pytest.mark.parametrize("vce", ["iid", "robust", "cluster"])
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


@pytest.mark.parametrize("path", ["dense", "block", "gmm"])
def test_full_role_rank_does_not_imply_iv_identification(path):
    y, c, e, z = _identification_problem(0.)
    assert np.linalg.matrix_rank(np.column_stack((c, e))) == 3
    with pytest.raises(UnderidentifiedError):
        _fit(y, c, e, z, path)


@pytest.mark.parametrize("path", ["dense", "block", "gmm"])
def test_near_unidentified_iv_is_not_silently_reported_as_reliable(path):
    y, c, e, z = _identification_problem(1e-10)
    with pytest.raises(NumericalError):
        _fit(y, c, e, z, path)


@pytest.mark.parametrize("path", ["dense", "block", "gmm"])
@pytest.mark.parametrize("delta,error", [(0., UnderidentifiedError), (1e-10, NumericalError)])
def test_identification_policy_is_invariant_to_column_units(path, delta, error):
    y, c, e, z = _identification_problem(delta)
    with pytest.raises(error):
        _fit(y, c * 1e8, e * np.array([1e-7, 1e7]),
             z * np.array([1e6, 1e-6]), path)


@pytest.mark.parametrize("path", ["dense", "block"])
def test_fixed_zero_kclass_does_not_require_excluded_instrument_identification(path):
    y, c, e, z = _identification_problem(0.)
    x = np.column_stack((c, e))
    expected = np.linalg.lstsq(x, y, rcond=None)[0]
    got = _fit(y, c, e, z, path, estimator="kclass", kappa=0., vce="robust")
    np.testing.assert_allclose(got[0], expected, atol=1e-12)


def test_exact_identification_does_not_materialize_joint_instrument_design(monkeypatch):
    y, c, e, z, w = _regular()
    original = BlockDesign.materialize
    materialized_columns = []
    def record(self, *args, **kwargs):
        materialized_columns.append(self.ncols)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(BlockDesign, "materialize", record)
    got = _fit(y, c, e, z[:, :2], "block", weights=w)
    assert got[6]["overidentification"]["df"] == 0
    assert c.shape[1] + e.shape[1] not in materialized_columns


@pytest.mark.parametrize("path", ["dense", "block"])
@pytest.mark.parametrize("delta", [1e-3, 1e-6])
def test_near_identification_matches_exact_float_input_moments(path, delta):
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
