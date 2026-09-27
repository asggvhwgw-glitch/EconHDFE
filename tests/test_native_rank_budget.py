"""Native exact-rank resource guards; no finite-field lower bound is a result."""
from fractions import Fraction
import importlib

import numpy as np
import pytest

rank = importlib.import_module("econhdfe.hdfe.rank")


def parity_edges():
    return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])


def fraction_rank(groups):
    # Independent dense rational elimination, not the production sparse code.
    arrays = [np.eye(len(np.unique(g)), dtype=int)[np.unique(g, return_inverse=True)[1]]
              for g in groups]
    matrix = [[Fraction(int(v)) for v in row] for row in np.column_stack(arrays)]
    pivot = 0
    for column in range(len(matrix[0])):
        selected = next((i for i in range(pivot, len(matrix)) if matrix[i][column]), None)
        if selected is None:
            continue
        matrix[pivot], matrix[selected] = matrix[selected], matrix[pivot]
        scale = matrix[pivot][column]
        matrix[pivot] = [v / scale for v in matrix[pivot]]
        for i in range(pivot + 1, len(matrix)):
            factor = matrix[i][column]
            matrix[i] = [v - factor * p for v, p in zip(matrix[i], matrix[pivot])]
        pivot += 1
        if pivot == len(matrix):
            break
    return pivot


@pytest.mark.parametrize("phase", ["_modular_sparse_rank", "_rational_sparse_rank"])
@pytest.mark.parametrize("resource", ["work", "bytes"])
def test_reject_before_materializing_native_rows(monkeypatch, phase, resource):
    edges = parity_edges() + [0, 2, 4]
    monkeypatch.setattr(rank, "_NATIVE_MAX_WORK" if resource == "work" else "_NATIVE_MAX_BYTES", 1)
    def forbidden(*args):
        pytest.fail("row dictionaries allocated before budget admission")
    monkeypatch.setattr(rank, "_rows_as_unit_dicts", forbidden)
    kwargs = {"prime": 2} if phase == "_modular_sparse_rank" else {}
    with pytest.raises(rank._ExactRankResourceError, match="no exact rank returned") as exc:
        getattr(rank, phase)(edges, **kwargs)
    assert exc.value.used > exc.value.limit
    assert exc.value.resource == ("work_units" if resource == "work" else "estimated_bytes")


def test_modular_and_rational_passes_share_work_budget(monkeypatch):
    edges = parity_edges() + [0, 2, 4]
    budget = rank._NativeRankBudget()
    assert rank._modular_sparse_rank(edges, 2, target_rank=4, _budget=budget) == 3
    first_work = budget.work
    monkeypatch.setattr(rank, "_NATIVE_MAX_WORK", first_work)
    with pytest.raises(rank._ExactRankResourceError) as exc:
        rank._rational_sparse_rank(edges, _budget=budget)
    assert exc.value.stage == "native_rational"
    assert exc.value.resource == "work_units"
    assert exc.value.used == first_work + edges.size


def test_integer_growth_rejected_before_cross_products(monkeypatch):
    monkeypatch.setattr(rank, "_NATIVE_MAX_INTEGER_BITS", 2)
    with pytest.raises(rank._ExactRankResourceError) as exc:
        rank._rational_sparse_rank(parity_edges() + [0, 2, 4])
    assert exc.value.resource == "integer_bits"
    assert exc.value.stage == "native_rational"


def test_fill_in_budget_rejects_after_initial_admission(monkeypatch):
    edges = parity_edges() + [0, 2, 4]
    source = len(edges) * rank._row_storage(edges.shape[1])
    monkeypatch.setattr(rank, "_NATIVE_MAX_BYTES", source + 1)
    with pytest.raises(rank._ExactRankResourceError) as exc:
        rank._rational_sparse_rank(edges)
    assert exc.value.resource == "estimated_bytes"
    assert exc.value.used > source


def test_public_native_failure_never_returns_bad_prime_lower_bound(monkeypatch):
    monkeypatch.setattr(rank, "_EXACT_PRIME", 2)
    monkeypatch.setattr(rank, "_NATIVE_MAX_WORK", 1)
    with pytest.raises(RuntimeError, match="no exact rank returned"):
        rank.categorical_rank(list(parity_edges().T), backend="native")


def test_resource_failure_does_not_poison_later_calls(monkeypatch):
    groups = list(parity_edges().T)
    with monkeypatch.context() as context:
        context.setattr(rank, "_NATIVE_MAX_WORK", 1)
        with pytest.raises(rank._ExactRankResourceError):
            rank.categorical_rank(groups, backend="native")
    assert rank.categorical_rank(groups, backend="native") == 4


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("width", [3, 4])
def test_native_permuted_labels_and_bad_prime_match_exact_oracle(monkeypatch, seed, width):
    monkeypatch.setattr(rank, "_EXACT_PRIME", 2)
    rng = np.random.default_rng(seed)
    groups = [rng.integers(0, 3, 16) for _ in range(width)]
    expected = fraction_rank(groups)
    original = [g.copy() for g in groups]
    assert rank.categorical_rank(groups, backend="native") == expected
    order = rng.permutation(16)
    relabeled = [(g[order] * 13 - 10**9) for g in reversed(groups)]
    assert rank.categorical_rank(relabeled, backend="native") == expected
    assert all(np.array_equal(a, b) for a, b in zip(original, groups))


def test_structural_certificate_does_not_require_native_elimination(monkeypatch):
    monkeypatch.setattr(rank, "_NATIVE_MAX_WORK", 0)
    groups = [np.repeat(np.arange(3), 4), np.tile(np.repeat(np.arange(2), 2), 3),
              np.tile([0, 1], 6)]
    assert rank.categorical_rank(groups, backend="native") == fraction_rank(groups)
