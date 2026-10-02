"""Small scale-equivalence oracles; no GPU jobs or timing benchmarks."""
import numpy as np
import pytest

from boundedqr import bootstrap, solve_perturbations


@pytest.mark.parametrize('exponent', [-500, -46, 46, 500])
@pytest.mark.parametrize('verify_radius', [False, True])
def test_tiny_and_huge_units_preserve_known_median(exponent, verify_radius):
    scale = np.ldexp(1., exponent)
    x = np.full((3, 1), scale)
    y = np.array([1., 2., 3.])
    beta, diag = solve_perturbations(x, y, .5, np.zeros((1, 1)), [0.],
                                    threads=1, verify_radius=verify_radius)
    # Independent analytic optimum is the median fitted value 2; checking
    # only a raw coefficient tolerance would ignore the measurement units.
    np.testing.assert_allclose(x @ beta[0], np.full(3, 2.), atol=1e-12, rtol=0)
    assert abs(.5 * np.sum(np.abs(y - x @ beta[0])) - 1.) < 1e-12
    assert diag['all_checked']
    assert diag['fits'][0]['stationarity_coordinates'] == 'column-normalized'
    if verify_radius:
        cert = diag['fits'][0]['radius_certificate']
        assert np.isfinite(cert['radius'])
        assert cert['radius_coordinates'] == 'original Euclidean coefficient units'
        assert cert['radius'] * scale < 1e-9


@pytest.mark.parametrize('exponent', [-46, 46])
def test_scale_map_includes_nonzero_gradient_and_active_pseudo_row(exponent):
    scale = np.ldexp(1., exponent)
    x, y = np.array([[scale]]), np.ones(1)
    beta, diag = solve_perturbations(x, y, .5, np.array([[-scale]]),
                                    [1. / scale], threads=1)
    # Original finite objective = .5|1-scale*b|+.5|1+2*scale*b|.
    assert abs(scale * beta[0, 0] + .5) < 1e-12
    assert abs(diag['fits'][0]['pseudo_residual']) < 1e-12
    assert diag['all_checked']


def test_mixed_column_units_preserve_perturbation_family_and_radius_units():
    rng = np.random.default_rng(2201)
    x = np.column_stack((np.ones(41), rng.normal(size=(41, 2))))
    y = x @ np.array([.3, -.7, 1.2]) + rng.normal(size=41)
    w = rng.normal(size=(3, 3)) * .2
    original, first = solve_perturbations(x, y, .3, w, np.zeros(3), threads=1)
    scales = np.ldexp(np.ones(3), [-45, 40, 20])
    converted, second = solve_perturbations(x * scales, y, .3, w * scales[:, None],
                                           np.zeros(3), threads=1)
    np.testing.assert_allclose(converted * scales, original, rtol=0, atol=2e-11)
    for a, b in zip(first['fits'], second['fits']):
        assert b['numerically_checked']
        np.testing.assert_allclose(b['pseudo_residual'], a['pseudo_residual'], rtol=0, atol=1e-11)
        ca, cb = a['radius_certificate'], b['radius_certificate']
        assert np.isfinite(cb['radius'])
        np.testing.assert_allclose(cb['normalized_radius'], ca['normalized_radius'], rtol=1e-12)
        np.testing.assert_allclose(cb['radius'] * scales[0], ca['radius'], rtol=1e-12)


def test_bootstrap_baseline_is_fitted_in_supported_column_coordinates():
    scale = np.ldexp(1., -46)
    y = np.arange(1., 10.)
    result = bootstrap(np.full((9, 1), scale), y, np.arange(9) % 3,
                       reps=3, threads=1, batch_size=2)
    np.testing.assert_allclose(result.coefficients * scale, [[5.]], atol=1e-6, rtol=0)
    assert np.isfinite(result.bootstrap_coefficients).all()
    assert result.diagnostics['quantiles'][0]['base_column_normalization'] is not None


def test_unrepresentable_within_column_dynamic_range_is_rejected():
    # Normalizing the 1e300 entry would erase the nonzero 1e-300 entry.
    with pytest.raises(ValueError, match='dynamic range'):
        solve_perturbations(np.array([[1e300], [1e-300], [1.]]), [1., 2., 3.],
                            .5, np.zeros((1, 1)), [0.], threads=1)


def test_scaling_does_not_broadcast_a_wrong_length_starting_coefficient():
    with pytest.raises(ValueError, match='beta0'):
        solve_perturbations(np.eye(2), [1., 2.], .5, np.zeros((2, 1)), [0.], threads=1)
