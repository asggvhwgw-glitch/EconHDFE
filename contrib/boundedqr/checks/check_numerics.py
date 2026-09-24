"""Independent tiny oracles and adversarial checks, without GPU/timing tests."""
from fractions import Fraction
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.optimize import linprog

from boundedqr import coefficient_radius, solve_perturbations
from boundedqr.inference import summarize_process


def primal_qr_oracle(X, y, tau, W=None):
    """Independent primal LP with positive/negative residuals (production is dual)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if W is not None:
        X = np.vstack((X, np.asarray(W) / tau))
        y = np.r_[y, len(y) * np.max(np.abs(y))]
    n, p = X.shape
    cost = np.r_[np.zeros(p), np.full(n, tau), np.full(n, 1 - tau)]
    matrix = np.column_stack((X, np.eye(n), -np.eye(n)))
    fit = linprog(cost, A_eq=matrix, b_eq=y,
                  bounds=[(None, None)] * p + [(0, None)] * (2 * n),
                  method="highs-ipm", options={"primal_feasibility_tolerance": 1e-9,
                                               "dual_feasibility_tolerance": 1e-9})
    assert fit.success, fit.message
    return fit.x[:p], fit.fun, fit.eqlin.marginals


def check_loss(X, y, tau, beta, W=None):
    r = np.asarray(y) - np.asarray(X) @ beta
    loss = np.maximum(tau * r, (tau - 1) * r).sum()
    if W is not None:
        r0 = len(y) * np.max(np.abs(y)) - (np.asarray(W) / tau) @ beta
        loss += max(tau * r0, (tau - 1) * r0)
    return loss


def test_finite_pseudo_problem_when_ideal_tilt_is_unbounded():
    # rho(.1-beta) is not used here: y=1, X=1, tau=.5, W=-1.
    # Finite loss=.5|1-beta|+.5|1+2beta| has unique minimum beta=-.5.
    # Ideal tilt=.5|1-beta|+beta instead tends to -infinity.
    X, y, W = np.ones((1, 1)), np.ones(1), np.array([[-1.0]])
    beta, diag = solve_perturbations(X, y, .5, W, [1.0], threads=1)
    np.testing.assert_allclose(beta, [[-.5]], atol=1e-10, rtol=0)
    assert diag["all_checked"]
    assert abs(diag["fits"][0]["pseudo_residual"]) < 1e-10
    cert = diag["fits"][0]["radius_certificate"]
    assert cert["radius"] < 1e-7


@pytest.mark.parametrize("tau", [.05, .2, .8, .95])
@pytest.mark.parametrize("w_scale", [.1, 1e4])
def test_tail_quantiles_match_independent_primal_lp(tau, w_scale):
    rng = np.random.default_rng(447)
    X = np.column_stack((np.ones(31), rng.normal(size=(31, 2))))
    y = X @ [.3, -.7, 1.2] + rng.standard_t(4, 31)
    W = rng.normal(size=(3, 2)) * w_scale
    initial, _, _ = primal_qr_oracle(X, y, tau)
    fitted, info = solve_perturbations(X, y, tau, W, initial, threads=1)
    assert info["all_checked"]
    for j, beta in enumerate(fitted):
        exact, objective, _ = primal_qr_oracle(X, y, tau, W[:, j])
        assert abs(check_loss(X, y, tau, beta, W[:, j]) - objective) <= 2e-7 * (1 + abs(objective))
        np.testing.assert_allclose(beta, exact, atol=2e-6, rtol=2e-7)


def test_nonunique_median_radius_covers_whole_optimal_segment():
    # Every beta in [-1,1] minimizes the loss. Full-rank X does not imply uniqueness.
    X, y = np.ones((2, 1)), np.array([-1., 1.])
    cert = coefficient_radius(X, y, .5, [0.], [0.], [0., 0., .5])
    assert np.isfinite(cert.radius)
    assert cert.radius >= 1
    beta, info = solve_perturbations(X, y, .5, np.zeros((1, 1)), [0.], threads=1)
    assert -1 - 1e-9 <= beta[0, 0] <= 1 + 1e-9
    assert info["all_checked"]


def test_rank_deficient_radius_does_not_invent_full_coefficient_control():
    X = np.array([[1., 1.], [1., 1.], [1., 1.]])
    cert = coefficient_radius(X, [-1., 0., 1.], .5, [0., 0.], [0., 0.],
                              [-.5, 0., .5, .5])
    assert np.isinf(cert.radius)
    assert cert.status == "uninformative"
    # The direction (d,-d) leaves every residual unchanged: any finite radius is false.


def test_insufficient_interior_rows_does_not_use_short_svd_as_full_rank():
    X = np.eye(2)
    cert = coefficient_radius(X, [0., 1.], .5, [0., 0.], [0., 1.],
                              [0., .5, .5])
    assert cert.interior_count < 2
    assert np.isinf(cert.radius)


def test_nonstationary_dual_radius_still_encloses_known_unique_median():
    X, y = np.ones((3, 1)), np.array([-2., 0., 3.])
    beta = np.array([.2])
    # a has stationarity error .01; exact stationarity is not assumed by the proof.
    cert = coefficient_radius(X, y, .5, [0.], beta, [-.5, .01, .5, .5])
    assert cert.stationarity_norm_upper >= .01
    assert np.isfinite(cert.radius)
    assert cert.radius >= .2


def test_dual_box_clipping_recomputes_stationarity():
    X, y = np.ones((3, 1)), np.array([-2., 0., 3.])
    cert = coefficient_radius(X, y, .5, [0.], [.1], [-2., 0., 2., 4.])
    assert cert.clipping_adjustment_max >= 3.5
    assert np.isfinite(cert.radius)
    assert cert.radius >= .1


def test_small_design_scale_cannot_turn_absolute_stationarity_into_false_success():
    # HiGHS can discard tiny matrix entries. With X=1e-14, beta=0 has an
    # apparently tiny stationarity residual but loss 3 versus exact minimum 1.
    # Scaling the design or explicitly rejecting it is acceptable; false success is not.
    X, y = np.full((3, 1), 1e-14), np.array([1., 2., 3.])
    try:
        beta, info = solve_perturbations(X, y, .5, np.zeros((1, 1)), [0.], threads=1)
    except (ValueError, RuntimeError):
        return
    assert info["all_checked"]
    assert abs(.5 * np.sum(np.abs(y - X @ beta[0])) - 1.) <= 1e-7


def test_inference_quantile_and_sd_bounds_cover_independent_perturbations():
    rng = np.random.default_rng(931)
    proposed = rng.normal(size=(9, 2, 3))
    radius = rng.uniform(.001, .1, size=(9, 2))
    error = rng.normal(size=proposed.shape)
    error *= (radius / np.linalg.norm(error, axis=2))[:, :, None]
    truth = proposed + error
    out = summarize_process(np.zeros((2, 3)), proposed, level=.8, radii=radius)
    qtrue = np.quantile(truth, [.1, .9], axis=0, method="linear")
    assert np.all(out["bootstrap_quantile_lower"] <= qtrue + 1e-14)
    assert np.all(qtrue <= out["bootstrap_quantile_upper"] + 1e-14)
    sd_error = np.abs(truth.std(axis=0, ddof=1) - out["standard_errors"])
    assert np.all(sd_error <= out["standard_error_numerical_bound"] + 1e-14)
    expected_covariance = np.cov(proposed.reshape(9, -1).T, ddof=1)
    np.testing.assert_allclose(out["covariance"], expected_covariance)


def test_quantile_enclosure_is_outward_rounded_at_adjacent_floats():
    # Exact 25% quantile lies strictly between adjacent binary64 values.
    draws = np.array([1., np.nextafter(1., 2.)]).reshape(2, 1, 1)
    out = summarize_process([[1.]], draws, level=.5, radii=np.zeros((2, 1)))
    exact = Fraction(float(draws[0, 0, 0])) * Fraction(3, 4) + Fraction(float(draws[1, 0, 0])) / 4
    low = Fraction(float(out["bootstrap_quantile_lower"][0, 0, 0]))
    high = Fraction(float(out["bootstrap_quantile_upper"][0, 0, 0]))
    assert low <= exact <= high


def test_uninformative_radii_do_not_drop_draws_or_create_nan_bounds():
    draws = np.arange(12., dtype=float).reshape(3, 2, 2)
    out = summarize_process(np.ones((2, 2)), draws, radii=np.full((3, 2), np.inf))
    assert np.isneginf(out["bootstrap_quantile_lower"]).all()
    assert np.isposinf(out["bootstrap_quantile_upper"]).all()
    assert np.isposinf(out["standard_error_numerical_bound"]).all()


def test_zero_standard_error_is_explicit():
    out = summarize_process([[2.]], np.full((5, 1, 1), 2.))
    assert out["zero_standard_error"].all()
    assert np.isinf(out["uniform_critical_value"])
    np.testing.assert_array_equal(out["uniform_band"], [[[-np.inf, np.inf]]])


def test_summary_overflow_is_explicitly_rejected_or_handled_finitely():
    draws = np.array([1e200, -1e200]).reshape(2, 1, 1)
    try:
        out = summarize_process([[0.]], draws)
    except (ValueError, FloatingPointError, RuntimeError):
        return
    assert not np.isnan(out["uniform_band"]).any()
    assert out["uniform_critical_value"] > 0
    # Infinite covariance is allowed only with conservative, explicit diagnostics.


@pytest.mark.parametrize("bad", [-1., np.nan])
def test_invalid_coefficient_radii_rejected(bad):
    with pytest.raises(ValueError):
        summarize_process([[0.]], np.zeros((3, 1, 1)), radii=np.full((3, 1), bad))


def exact_sd_decimal(values):
    """Exact rational variance followed by 100-digit Decimal sqrt."""
    rational = [Fraction(float(v)) for v in values]
    mean = sum(rational) / len(rational)
    variance = sum((v - mean)**2 for v in rational) / (len(rational) - 1)
    with localcontext() as context:
        context.prec = 100
        return (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()


@pytest.mark.parametrize("values", [
    [1., np.nextafter(1., 2.)],
    [1., 1., 1., np.nextafter(1., 2.)],
    [1e12, 1e12 + .0001220703125, 1e12 + .000244140625, 1e12 - .0001220703125],
    [1e-200, 2e-200, 3e-200, -1e-200],
    [np.nextafter(0., 1.), 2 * np.nextafter(0., 1.), 3 * np.nextafter(0., 1.)],
    [1e100] * 49,
    [1e150, -1e150, 2e150, -2e150],
    [0.] * 5,
])
def test_sd_arithmetic_bound_against_exact_rational_variance(values):
    draws = np.asarray(values).reshape(-1, 1, 1)
    out = summarize_process([[0.]], draws, radii=np.zeros((len(values), 1)))
    truth = exact_sd_decimal(values)
    computed = Decimal.from_float(float(out["standard_errors"][0, 0]))
    arithmetic = Decimal.from_float(float(out["standard_error_arithmetic_bound"][0, 0]))
    assert abs(computed - truth) <= arithmetic
    assert out["standard_error_optimization_bound"][0, 0] == 0
    assert out["standard_error_numerical_bound"][0, 0] >= float(arithmetic)
    assert "FP64" in out["standard_error_bound_scope"]


def test_sd_total_bound_includes_both_optimization_and_roundoff():
    epsilon = np.spacing(1.)
    proposed = np.array([1., 1. + epsilon])
    truth = np.array([1. - epsilon, 1. + 2 * epsilon])
    out = summarize_process([[1.]], proposed.reshape(2, 1, 1),
                            radii=np.full((2, 1), epsilon))
    computed = Decimal.from_float(float(out["standard_errors"][0, 0]))
    bound = Decimal.from_float(float(out["standard_error_numerical_bound"][0, 0]))
    assert abs(computed - exact_sd_decimal(truth)) <= bound
    assert out["standard_error_optimization_bound"][0, 0] > 0
    assert out["standard_error_arithmetic_bound"][0, 0] > 0
