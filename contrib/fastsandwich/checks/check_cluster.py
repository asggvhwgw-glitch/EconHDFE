"""Production cluster contract and integration-reference checks."""

import numpy as np
import pytest

from fastsandwich.cluster import cluster_meat


@pytest.mark.parametrize("method", ["auto", "reduceat", "bincount"])
@pytest.mark.parametrize("label_kind", ["random", "sorted", "unique", "one", "string", "sparse"])
def test_released_reference(method, label_kind):
    from linearmodels.shared.covariance import cov_cluster

    rng = np.random.default_rng(718)
    scores = rng.normal(size=(91, 7))
    groups = rng.integers(0, 11, len(scores))
    if label_kind == "sorted":
        groups = np.sort(groups)
    elif label_kind == "unique":
        groups = rng.permutation(len(scores))
    elif label_kind == "one":
        groups[:] = -17
    elif label_kind == "string":
        groups = np.array([f"id-{g}" for g in groups])
    elif label_kind == "sparse":
        groups = groups * np.int64(10**14) - 2**60
    original_scores, original_groups = scores.copy(), groups.copy()
    np.testing.assert_allclose(
        cluster_meat(scores, groups, method), cov_cluster(scores, groups),
        rtol=3e-13, atol=3e-13,
    )
    np.testing.assert_array_equal(scores, original_scores)
    np.testing.assert_array_equal(groups, original_groups)


@pytest.mark.parametrize("method", ["auto", "reduceat", "bincount"])
def test_scalar_and_conversion_contract(method):
    # Expected (1+3)^2 + (2+4)^2, divided by four observations.
    expected = np.array([[13.0]])
    for dtype in (np.float32, np.int64):
        x = np.array([1, 2, 3, 4], dtype=dtype)
        result = cluster_meat(x, [0, 1, 0, 1], method)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(cluster_meat([3.0], ["a"], method), [[9.0]])


@pytest.mark.parametrize("method", ["auto", "reduceat", "bincount"])
def test_two_way_duplicate_cells(method):
    from linearmodels.shared.covariance import cluster_union, cov_cluster

    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]])
    x = np.arange(15, dtype=float).reshape(5, 3) - 4
    intersection = cluster_union(pairs)
    expected = cov_cluster(x, pairs[:, 0]) + cov_cluster(x, pairs[:, 1]) - cov_cluster(x, intersection)
    result = cluster_meat(x, pairs[:, 0], method) + cluster_meat(x, pairs[:, 1], method) - cluster_meat(x, intersection, method)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("groups", [
    [0, np.nan], np.array([0, None], dtype=object),
    np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"),
    np.array([0, np.nan], dtype=object),
    ["a", np.nan],
])
@pytest.mark.parametrize("method", ["auto", "reduceat", "bincount"])
def test_missing_labels_rejected(groups, method):
    with pytest.raises(ValueError, match="missing"):
        cluster_meat([[1.0], [2.0]], groups, method)


@pytest.mark.parametrize("x, groups, message", [
    ([], [], "nonempty"), (np.ones((2, 0)), [0, 1], "nonempty"),
    (np.ones((2, 2, 2)), [0, 1], "nonempty"),
    ([np.inf, 1], [0, 1], "finite"), ([np.nan, 1], [0, 1], "finite"),
    ([1 + 2j, 2], [0, 1], "real numeric"),
    ([1, 2], [[0], [1]], "one-dimensional"),
    ([1, 2], [0], "one-dimensional"),
    ([1, 2], np.array([1, "one"], dtype=object), "orderable"),
    ([1, 2], [1, "one"], "orderable"),
    ([1, 2], [1 + 1j, 2 + 1j], "orderable"),
])
def test_invalid_inputs(x, groups, message):
    with pytest.raises(ValueError, match=message):
        cluster_meat(x, groups)


def test_invalid_method():
    with pytest.raises(ValueError, match="method"):
        cluster_meat([1.0], [0], "unsupported")


@pytest.mark.parametrize("method", ["auto", "reduceat", "bincount"])
@pytest.mark.parametrize("groups", [
    [2**63, 2**63 + 1, -1],
    [2**64 - 1, -2**63, 2**64 - 2],
    [2**53 + 1, float(2**53), 0.5],
    [2**63 + 1, np.float64(2**63), -1],
    [np.uint64(2**63 + 1), np.uint64(2**63), np.int64(-1)],
])
def test_large_python_integer_sequence_keeps_distinct_labels(groups, method):
    # Three singleton groups: (1**2 + (-1)**2 + 2**2) / 3 = 2.
    # A float64 conversion would merge the first two labels in several cases.
    np.testing.assert_array_equal(cluster_meat([1, -1, 2], groups, method), [[2.0]])


@pytest.mark.parametrize("method", ["auto", "reduceat", "bincount"])
def test_large_python_integer_sequence_still_aggregates_and_rejects_missing(method):
    labels = [2**63 + 1, -1, 2**63 + 1, 2**63]
    np.testing.assert_array_equal(cluster_meat([1, 2, 3, 4], labels, method), [[9.0]])
    with pytest.raises(ValueError, match="missing"):
        cluster_meat([1, -1, 2], [2**63 + 1, np.nan, -1], method)
