"""Independent counterexamples for full multiway covariance PSD completion."""

import numpy as np
import pytest

from econhdfe import ExecutionConfig, ppmlhdfe
from econhdfe.compute.vcov import finalize_cluster_vcov
from econhdfe.postestimation import linear_combination


_CLUSTERS = (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))


def _complete(V, scale=None, constant=1):
    return finalize_cluster_vcov(
        V, "cluster", _CLUSTERS,
        lambda: np.ones(len(V)) if scale is None else scale,
        constant_index=constant,
    )


def _assert_psd(V):
    np.testing.assert_array_equal(V, V.T)
    magnitude = np.max(np.abs(V), initial=0.0)
    # Divide before eigenvalue calculation so very small covariance units do
    # not let an absolute tolerance hide an indefinite result.
    if magnitude:
        assert np.linalg.eigvalsh(V / magnitude)[0] >= -2e-13


def test_two_parameter_schur_completion_has_analytical_solution():
    # Full spectral clipping gives 1.5 * ones((2,2)). Restoring the slope
    # variance 1 while leaving c=1.5 makes determinant -0.75. The minimum
    # constant variance for this retained cross-covariance is 1.5**2 / 1.
    got = _complete(np.array([[1., 2.], [2., 1.]]))
    np.testing.assert_allclose(got, [[1., 1.5], [1.5, 2.25]], rtol=5e-15)
    _assert_psd(got)
    contrast = np.array([-1.75, 1.])
    assert contrast @ got @ contrast == pytest.approx(0.0625)


def test_singular_slope_covariance_has_no_constant_covariance_in_null_direction():
    raw = np.array([[-1., 0., 2.], [0., 2., 1.], [2., 1., 1.]])
    got = _complete(raw, constant=2)
    np.testing.assert_allclose(got[:2, :2], [[0., 0.], [0., 2.]])
    assert got[0, 2] == 0.
    # For the nonzero slope coordinate the Schur bound is b**2 / 2;
    # satisfying it and having zero null-space cross terms is sufficient.
    assert got[2, 2] >= got[1, 2] ** 2 / 2.
    _assert_psd(got)


@pytest.mark.parametrize("magnitude", [1e-150, 1., 1e150])
def test_completion_is_equivariant_to_units_signs_and_coefficient_order(magnitude):
    raw = magnitude * np.array([[1., .4, 2.], [.4, .1, -.2], [2., -.2, 1.]])
    scale = np.array([.8, 4., 1.])
    baseline = _complete(raw, scale, constant=2)
    _assert_psd(baseline)
    units = np.array([-1e4, 1e-2, 1.])
    changed = _complete(raw / np.outer(units, units), scale * np.abs(units), constant=2)
    np.testing.assert_allclose(
        changed * np.outer(units, units) / magnitude, baseline / magnitude,
        rtol=2e-13, atol=2e-13,
    )
    for permutation in (np.array([2, 0, 1]), np.array([1, 2, 0])):
        got = _complete(
            raw[np.ix_(permutation, permutation)], scale[permutation],
            constant=int(np.flatnonzero(permutation == 2)[0]),
        )
        np.testing.assert_allclose(
            got / magnitude, baseline[np.ix_(permutation, permutation)] / magnitude,
            rtol=2e-13, atol=2e-13,
        )
        _assert_psd(got)


def test_repair_preserves_psd_inputs_and_does_not_touch_other_vce_kinds():
    V = np.array([[1., .2], [.2, 2.]])
    np.testing.assert_array_equal(_complete(V), V)
    for kind, clusters in (("robust", _CLUSTERS), ("cluster", _CLUSTERS[:1])):
        # The callback must remain lazy: these paths need no regressor scale.
        got = finalize_cluster_vcov(V, kind, clusters, lambda: 1 / 0, constant_index=1)
        assert got is V
    np.testing.assert_array_equal(_complete(np.array([[-1.]]), constant=0), [[0.]])


def test_public_ppml_two_way_covariance_supports_mean_index_contrast():
    rng = np.random.default_rng(41)
    a = np.repeat(np.arange(6), 10)
    b = np.tile(np.arange(10), 6)
    x = rng.normal(size=60)
    y = rng.poisson(np.exp(
        .2 + .2 * x + .5 * rng.normal(size=6)[a] + .5 * rng.normal(size=10)[b]
    ))
    result = ppmlhdfe(
        y=y, x=x[:, None], names=("x",), vce="cluster", cluster=(a, b),
        execution_config=ExecutionConfig(threads=1),
    )
    assert result.converged
    _assert_psd(result.vcov)
    # Independent two-way sandwich: every (a,b) cell is a singleton. The
    # raw slope variance is positive, so reporting a constant must preserve
    # it even though the full CGM matrix needs repair.
    design = np.column_stack((x, np.ones(len(x))))
    bread = np.linalg.inv(design.T @ (result.mu[:, None] * design))
    scores = design * (y - result.mu)[:, None]
    grouped_a = np.vstack([scores[a == group].sum(axis=0) for group in np.unique(a)])
    grouped_b = np.vstack([scores[b == group].sum(axis=0) for group in np.unique(b)])
    meat = grouped_a.T @ grouped_a + grouped_b.T @ grouped_b - scores.T @ scores
    raw = (6. / 5.) * bread @ meat @ bread
    assert raw[0, 0] > 0.
    assert np.linalg.eigvalsh(raw)[0] < 0.
    assert result.vcov[0, 0] == pytest.approx(raw[0, 0], rel=2e-10)
    # This x value is within the observed range. The old full covariance
    # returned variance -0.001418325094 for this meaningful contrast.
    assert np.min(x) < -1.75 < np.max(x)
    contrast = linear_combination(
        result.params, result.vcov, {"x": -1.75, "_cons": 1.}, names=result.names,
    )
    assert contrast.std_error > 0.
    assert np.isfinite(contrast.statistic)
    assert 0. <= contrast.p_value <= 1.
