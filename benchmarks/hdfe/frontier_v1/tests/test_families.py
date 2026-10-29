"""Independent small-instance checks of the benchmark oracle and promises."""
from __future__ import annotations

from fractions import Fraction
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from benchmarks.hdfe.frontier_v1.families import (
    FAMILY_NAMES,
    RHS_KINDS,
    fitted_values,
    make_case,
)


def _dense_design(case):
    counts = case.metadata["categories_per_dimension"]
    offsets = np.cumsum([0, *counts[:-1]])
    design = np.zeros((len(case.groups), sum(counts)))
    design[np.arange(len(design))[:, None], case.groups + offsets] = 1.0
    return design


def _norm(values, weights):
    return np.sqrt(np.sum(weights[:, None] * values * values, axis=0))


@pytest.mark.parametrize("family", FAMILY_NAMES)
@pytest.mark.parametrize("weight_ratio,repeat", [(1.0, 1), (1e4, 2)])
def test_algebraic_oracle_matches_independent_dense_lstsq(family, weight_ratio, repeat):
    size = 4 if "grid" in family else 9
    case = make_case(family, size, rhs=2, seed=73,
                     weight_ratio=weight_ratio, repeat=repeat)
    D = _dense_design(case)
    w = np.ones(len(D)) if case.weights is None else case.weights
    B = np.sqrt(w)[:, None] * D
    beta, _, rank, _ = np.linalg.lstsq(B, np.sqrt(w)[:, None] * case.y, rcond=None)
    reference = case.y - D @ beta
    assert rank == case.metadata["expected_rank"] == D.shape[1] - 2
    assert np.max(_norm(reference - case.truth, w) / _norm(case.y, w)) < 2e-11
    assert np.max(_norm(fitted_values(case.groups, case.coefficients)
                        - (case.y - case.truth), w) / _norm(case.y, w)) < 2e-14
    normal_residual = D.T @ (w[:, None] * case.truth)
    assert np.linalg.norm(normal_residual) < 2e-12 * np.linalg.norm(w[:, None] * case.truth)
    assert len(np.unique(case.groups, axis=0)) == case.metadata["U"]
    assert case.metadata["N"] == len(D)
    assert case.metadata["G"] == D.shape[1]
    json.dumps(case.metadata)


@pytest.mark.parametrize("rhs_kind", RHS_KINDS)
def test_single_rhs_shape_and_reproducibility(rhs_kind):
    a = make_case("kernel_grid", 5, rhs=1, seed=18, rhs_kind=rhs_kind)
    b = make_case("kernel_grid", 5, rhs=1, seed=18, rhs_kind=rhs_kind)
    assert a.y.shape == a.truth.shape == (400, 1)
    np.testing.assert_array_equal(a.groups, b.groups)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.truth, b.truth)


@pytest.mark.parametrize("rhs_kind,targeted_fraction", [("mixed", 0.9), ("broadband", 0.5)])
def test_targeted_fit_energy_is_not_diluted_by_oracle_or_broad_component(rhs_kind, targeted_fraction):
    common = dict(family="rigidity_grid", size=8, rhs=2, seed=14, weight_ratio=1e3, repeat=2)
    target = make_case(**common, rhs_kind=rhs_kind)
    slow = make_case(**common, rhs_kind="slow")
    w = target.weights
    fitted = target.y - target.truth
    targeted = slow.y - slow.truth
    correlation = np.sum(w[:, None] * fitted * targeted, axis=0) / (
        _norm(fitted, w) * _norm(targeted, w)
    )
    np.testing.assert_allclose(correlation**2, targeted_fraction, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(_norm(fitted, w)**2 / _norm(target.y, w)**2, 0.5, atol=2e-13)
    np.testing.assert_allclose(_norm(target.truth, w)**2 / _norm(target.y, w)**2, 0.5, atol=2e-13)


def test_broadband_fields_excite_all_nonzero_grid_frequencies():
    case = make_case("kernel_grid", 12, rhs=2, seed=9, rhs_kind="broadband")
    # The zero-index representative coefficients recover the generator's
    # fields. Their nonzero Fourier support must grow beyond the 24 targeted
    # frequencies; this check does not call the generator's FFT helper.
    for coefficients in case.coefficients[:2]:
        field = coefficients[::2].reshape(12, 12, 2)
        magnitude = np.abs(np.fft.fftn(field, axes=(0, 1))).reshape(144, 2)
        assert np.all(magnitude[1:] > 1e-10)
    assert case.metadata["rhs"]["broadband_frequency_count"] == 143


def test_rigidity_colors_are_disconnected_but_combined_system_has_normal_rank():
    case = make_case("rigidity_grid", 6, rhs=1, rhs_kind="random")
    m = 36
    connector_modules = case.groups[8 * m:] // 2
    counts = case.metadata["connector_rows_by_kind"]
    expected_components = (6, 6, 2)
    position = 0
    for kind, column, expected in zip(("u", "v", "u_plus_v"), (0, 1, 0), expected_components):
        rows = connector_modules[position:position + counts[kind]]
        first = rows[:, column]
        second = rows[:, 2]
        graph = csr_matrix((np.ones(len(rows)), (first, second)), shape=(m, m))
        assert connected_components(graph, directed=False, return_labels=False) == expected
        position += counts[kind]
    assert np.linalg.matrix_rank(_dense_design(case), tol=1e-10) == 6 * m - 2


@pytest.mark.parametrize("family,size", [
    ("kernel_grid", 5), ("rigidity_grid", 5), ("rigidity_grid_unique", 5),
    ("kernel_cycle", 3), ("kernel_cycle", 9), ("kernel_cycle", 17),
])
def test_analytic_gap_bound_against_independently_assembled_gram(family, size):
    case = make_case(family, size, rhs=1, rhs_kind="random")
    D = _dense_design(case)
    A = D.T @ D
    scales = 1 / np.sqrt(np.diag(A))
    eigenvalues = np.linalg.eigvalsh(scales[:, None] * A * scales[None, :])
    assert np.count_nonzero(np.abs(eigenvalues) <= 1e-10) == 2
    assert abs(eigenvalues[-1] - 3) < 2e-12
    kappa = 3 / eigenvalues[2]
    assert kappa <= case.metadata["condition_number"]["analytic_upper_bound"] * (1 + 1e-10)


def _exact_determinant(matrix):
    a = [[Fraction(int(value)) for value in row] for row in matrix]
    determinant = Fraction(1)
    for k in range(len(a)):
        pivot = a[k][k]
        determinant *= pivot
        for i in range(k + 1, len(a)):
            ratio = a[i][k] / pivot
            for j in range(k + 1, len(a)):
                a[i][j] -= ratio * a[k][j]
    return determinant


def test_cycle_endpoint_certificate_has_exact_positive_principal_minors():
    # Build the local incidence matrix independently of the private template.
    T = np.zeros((8, 6), dtype=np.int64)
    for row in range(8):
        bits = ((row >> 2) & 1, (row >> 1) & 1, row & 1)
        for k, bit in enumerate(bits):
            T[row, 2 * k + bit] = 1
    R4 = T.T @ T
    R4[np.ix_([0, 2, 4], [0, 2, 4])] += np.array([[4, -4, 0], [-4, 4, 0], [0, 0, 4]])
    certificate = 8 * R4 - np.diag([8, 4, 8, 4, 8, 4])
    minors = [_exact_determinant(certificate[:k, :k]) for k in range(1, 7)]
    assert minors == [56, 1568, 66304, 914432, 35020800, 74088448]


@pytest.mark.parametrize("kwargs", [
    {"family": "unknown", "size": 4},
    {"family": "rigidity_grid", "size": 3},
    {"family": "kernel_grid", "size": 4.0},
    {"family": "kernel_grid", "size": 4, "repeat": 0},
    {"family": "kernel_grid", "size": 4, "rhs": 0},
    {"family": "kernel_grid", "size": 4, "weight_ratio": 0.5},
    {"family": "kernel_grid", "size": 4, "weight_ratio": np.inf},
])
def test_reject_invalid_problem_parameters(kwargs):
    with pytest.raises(ValueError):
        make_case(**kwargs)
