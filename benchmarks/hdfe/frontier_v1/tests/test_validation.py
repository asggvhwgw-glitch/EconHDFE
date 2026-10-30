"""Independent-oracle regression checks for the research frontier benchmark."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.hdfe.frontier_v1.validation import (
    certification,
    normalized_gram_spectrum,
    verify_case,
)


def cube_case(weights=None, rhs=2):
    groups = np.array([(a, b, c) for a in range(2) for b in range(2) for c in range(2)])
    w = np.ones(8) if weights is None else np.asarray(weights, dtype=float)
    h = (-1.0) ** (groups[:, 0] + groups[:, 1])
    truth = h[:, None] * np.arange(1, rhs + 1)[None, :] / w[:, None]
    coeff = tuple((np.arange(2)[:, None] + 0.25 * j) * np.arange(1, rhs + 1)[None, :] for j in range(3))
    fitted = sum(alpha[groups[:, j]] for j, alpha in enumerate(coeff))
    return SimpleNamespace(groups=groups, weights=weights, truth=truth, y=fitted + truth,
                           coefficients=coeff,
                           metadata={"oracle": {"kind": "weighted_cube_contrast", "certification": "algebraic"}})


def test_exact_weighted_oracle_and_output_pass():
    case = cube_case(np.geomspace(0.1, 10, 8))
    verified = verify_case(case)
    assert verified["aggregate_pass"]
    assert len(verified["feasibility"]["witnesses"]) == 2
    result = certification(case, case.truth)
    assert result["aggregate_pass"]
    assert result["max_forward_relative_error"] == 0
    assert result["max_normalized_fe_stationarity"] < 1e-15
    json.dumps(result, allow_nan=False)
    json.dumps(verified, allow_nan=False)


def test_small_group_moments_do_not_certify_forward_accuracy():
    # A tiny weighted bridge gives very small FE moments for a wholly wrong
    # residual.  All three observations are in col(D), so the true residual is 0.
    groups = np.array([[0, 0], [0, 1], [1, 1]])
    y = np.array([[0.0], [1e10], [0.0]])
    case = SimpleNamespace(groups=groups, y=y, truth=np.zeros_like(y),
                           weights=np.array([1.0, 1e-20, 1.0]),
                           coefficients=(np.array([[0.0], [-1e10]]), np.array([[0.0], [1e10]])),
                           metadata={})
    result = certification(case, y)
    assert result["stationarity_pass"]
    assert result["max_normalized_fe_stationarity"] < 2e-8
    assert result["max_forward_relative_error"] == pytest.approx(1.0)
    assert not result["forward_pass"]
    assert not result["aggregate_pass"]


def test_nonfinite_rhs_never_passes_and_is_json_safe():
    case = cube_case()
    output = case.truth.copy()
    output[0, 1] = np.nan
    result = certification(case, output)
    assert result["per_rhs"][0]["pass"]
    assert not result["per_rhs"][1]["pass"]
    assert result["per_rhs"][1]["forward_relative_error"] is None
    assert not result["finite"]
    assert not result["aggregate_pass"]
    json.dumps(result, allow_nan=False)


def test_zero_rhs_has_explicit_zero_over_zero_behavior():
    case = cube_case(rhs=1)
    case.y = np.zeros_like(case.y)
    case.truth = np.zeros_like(case.truth)
    assert certification(case, case.truth)["aggregate_pass"]
    wrong = certification(case, np.ones_like(case.truth))
    assert not wrong["aggregate_pass"]
    assert wrong["max_forward_relative_error"] is None
    json.dumps(wrong, allow_nan=False)


def test_weight_rescaling_does_not_change_certificate_or_spectrum():
    case = cube_case(np.geomspace(0.5, 2, 8))
    error = np.full_like(case.truth, 1e-9)
    before = certification(case, case.truth + error)
    s_before = normalized_gram_spectrum(case.groups, case.weights)
    case.weights = case.weights * 1e120
    after = certification(case, case.truth + error)
    s_after = normalized_gram_spectrum(case.groups, case.weights)
    assert before["max_forward_relative_error"] == pytest.approx(after["max_forward_relative_error"])
    assert before["max_normalized_fe_stationarity"] == pytest.approx(after["max_normalized_fe_stationarity"])
    assert s_before["kappa_plus"] == pytest.approx(s_after["kappa_plus"])


def test_cube_spectrum_reports_only_gauge_nullity():
    case = cube_case()
    result = normalized_gram_spectrum(case.groups)
    assert result["status"] == "ok"
    assert result["numerical_rank"] == 4
    assert result["normal_nullity"] == 2
    assert result["extra_nullity"] == 0
    assert result["lambda_max"] == pytest.approx(3)
    assert result["lambda_min_positive"] == pytest.approx(1)
    assert result["kappa_plus"] == pytest.approx(3)
    assert result["numerical_estimate_only"]
    assert not result["structural_rank_proof"]


def test_extra_nullity_is_explicit():
    # Second and third factors coincide even though the incidence graph connects.
    groups = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    result = normalized_gram_spectrum(groups)
    assert result["components"] == 1
    assert result["numerical_rank"] == 3
    assert result["normal_nullity"] == 2
    assert result["extra_nullity"] == 1
    assert result["has_extra_nullity"]
    assert any("unresolved" in warning for warning in result["warnings"])


def test_disconnected_components_adjust_normal_nullity():
    result = normalized_gram_spectrum(np.array([[0, 0, 0], [1, 1, 1]]))
    assert result["components"] == 2
    assert result["normal_nullity"] == 4
    assert result["numerical_rank"] == 2
    assert result["extra_nullity"] == 0


def test_sparse_and_negative_labels_are_encoded_without_large_allocations():
    base = np.array([[0, 0], [0, 1], [1, 1]])
    sparse = base * np.array([10**10, 10**11]) - np.array([100, 1000])
    expected = normalized_gram_spectrum(base)
    actual = normalized_gram_spectrum(sparse)
    assert actual["levels"] == [2, 2]
    assert actual["positive_eigenvalues"] == pytest.approx(expected["positive_eigenvalues"])


def test_dense_budget_is_checked_before_svd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SVD must not run after the budget is exceeded")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    result = normalized_gram_spectrum(cube_case().groups, max_categories=5)
    assert result["status"] == "skipped_budget"
    assert "max_categories=5" in result["reason"]
    assert "numerical_rank" not in result


def test_large_case_witness_checked_without_dense_rank_claim():
    case = cube_case()
    result = verify_case(case, max_categories=1)
    assert result["oracle_valid"]
    assert len(result["feasibility"]["witnesses"]) == 1
    assert not result["dense_budget"]["allowed"]
    assert not result["structural_rank_proof"]


def test_algebraic_metadata_alone_cannot_pass():
    case = cube_case()
    case.coefficients = None
    result = verify_case(case, max_categories=1)
    assert result["orthogonality"]["pass"]
    assert result["feasibility"]["status"] == "unverified"
    assert not result["oracle_valid"]


def test_corrupted_fe_coefficient_witness_fails_even_if_dense_fit_succeeds():
    case = cube_case()
    case.coefficients[0][0, 0] += 1.0
    result = verify_case(case)
    assert result["feasibility"]["witnesses"][1]["pass"]
    assert not result["feasibility"]["witnesses"][0]["pass"]
    assert not result["aggregate_pass"]


def test_nonorthogonal_truth_fails_oracle_check():
    case = cube_case()
    case.truth = case.truth + 0.1
    result = verify_case(case)
    assert not result["orthogonality"]["pass"]
    assert not result["oracle_valid"]


@pytest.mark.parametrize("weights", [np.zeros(8), -np.ones(8), np.full(8, np.nan)])
def test_invalid_weights_are_rejected(weights):
    case = cube_case()
    case.weights = weights
    with pytest.raises(ValueError, match="strictly positive"):
        certification(case, case.truth)


def test_shape_mismatch_is_rejected():
    case = cube_case()
    with pytest.raises(ValueError, match="same shape"):
        certification(case, case.truth[:, :1])


def test_declared_spectral_bound_rejects_a_small_moment_false_oracle():
    good = cube_case()
    good.metadata["condition_number"] = {"analytic_upper_bound": 3.0}
    check = verify_case(good)
    assert check["aggregate_pass"]
    assert check["oracle_forward_bound_pass"]
    assert check["oracle_forward_error_bound"] < check["oracle_forward_tolerance"]
    assert check["depends_on_declared_analytic_spectral_bound"]
    assert check["computed_in_float64_not_interval_certified"]
    assert not check["oracle_forward_bound_details"]["numerical_spectrum_estimates_used"]

    # This is a *false oracle*, not merely a bad backend result.  It passes
    # small moment and exact feasibility checks but its true forward error is 1.
    groups = np.array([[0, 0], [0, 1], [1, 1]])
    y = np.array([[0.0], [1e12], [0.0]])
    bad = SimpleNamespace(
        groups=groups, y=y, truth=y.copy(), weights=np.array([1.0, 1e-24, 1.0]),
        coefficients=(np.zeros((2, 1)), np.zeros((2, 1))),
        metadata={"condition_number": {"analytic_upper_bound": 1e28}},
    )
    check = verify_case(bad)
    assert check["orthogonality"]["pass"]
    assert check["feasibility"]["pass"]
    assert check["oracle_forward_error_bound"] > 1
    assert not check["oracle_forward_bound_pass"]
    assert not check["aggregate_pass"]
    json.dumps(check, allow_nan=False)


def test_missing_analytic_bound_is_explicit_not_replaced_by_numeric_spectrum():
    case = cube_case()
    case.metadata["condition_number"] = {"numerical_kappa_estimate": 3.0}
    check = verify_case(case)
    assert check["aggregate_pass"]  # Qualified constructive/dense checks only.
    assert check["oracle_forward_error_bound"] is None
    assert check["oracle_forward_bound_pass"] is None
    assert not check["oracle_forward_bound_available"]
    assert not check["depends_on_declared_analytic_spectral_bound"]
    assert "no analytic forward bound" in check["oracle_basis"]
    assert any("do not by themselves" in text for text in check["warnings"])
    json.dumps(check, allow_nan=False)
