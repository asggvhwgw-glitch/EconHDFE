"""Independent, deliberately slow mathematical oracles for sandwich meats.

These tests use scalar observation-pair definitions, not lagged matrix products,
cumulative sums, FFTs, group sorting, or grouped reduction implementation code.
"""

import math

import numpy as np
import pytest

from fastsandwich import cluster_meat, hac_meat


HAC_METHODS = ("auto", "direct", "prefix", "blocked", "fft")
CLUSTER_METHODS = ("auto", "reduceat", "bincount")


def pair_hac_oracle(scores, lag):
    """Unnormalized double sum with Bartlett observation-pair weights."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, k = x.shape
    result = np.empty((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            result[a, b] = math.fsum(
                float(x[t, a]) * float(x[u, b]) * (1.0 - abs(t-u)/(lag+1))
                for t in range(n)
                for u in range(n)
                if abs(t-u) <= lag
            )
    return result


def pair_cluster_oracle(scores, groups):
    """Observation-pair indicator definition, with linearmodels /N scale."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    labels = np.asarray(groups)
    n, k = x.shape
    result = np.empty((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            result[a, b] = math.fsum(
                float(x[t, a]) * float(x[u, b])
                for t in range(n)
                for u in range(n)
                if labels[t] == labels[u]
            ) / n
    return result


@pytest.mark.parametrize("method", HAC_METHODS)
@pytest.mark.parametrize("n", [1, 2, 7, 13])
@pytest.mark.parametrize("lag_kind", ["zero", "one", "edge", "beyond"])
def test_hac_against_observation_pair_definition(method, n, lag_kind):
    rng = np.random.default_rng(182+n)
    x = rng.normal(size=(n, 3)) + [1.0, -0.5, 0.0]
    lag = {"zero": 0, "one": 1, "edge": n-1, "beyond": 2*n+3}[lag_kind]
    expected = pair_hac_oracle(x, lag)
    got = hac_meat(x, lag, method=method)
    np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("method", HAC_METHODS)
@pytest.mark.parametrize("lag,expected", [(0, 13.0), (1, 19.0), (2, 21.0)])
def test_hac_hand_calculation_and_no_lag_clipping(method, lag, expected):
    np.testing.assert_allclose(hac_meat([2.0, 3.0], lag, method=method), [[expected]],
                               rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("method", HAC_METHODS)
def test_hac_zero_extension_retains_both_boundaries(method):
    x = np.zeros((11, 2))
    x[0] = [2, -3]
    x[-1] = [-5, 7]
    # These endpoints are too distant to covary at L=3. Neither may be lost.
    expected = np.outer(x[0], x[0]) + np.outer(x[-1], x[-1])
    np.testing.assert_allclose(hac_meat(x, 3, method=method), expected,
                               rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("method", HAC_METHODS)
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_hac_strided_input_dtype_and_nonmutation(method, dtype):
    x = np.arange(72, dtype=dtype).reshape(12, 6)[::2, ::2]
    old = x.copy()
    got = hac_meat(x, 4, method=method)
    np.testing.assert_allclose(got, pair_hac_oracle(x, 4), rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(x, old)
    assert got.dtype == np.float64


@pytest.mark.parametrize("method", HAC_METHODS)
def test_hac_no_demeaning_and_default_bandwidth(method):
    x = np.ones((17, 2))
    lag = int(np.floor(4 * (len(x)/100.0)**(2.0/9.0)))
    expected = pair_hac_oracle(x, lag)
    got = hac_meat(x, method=method)
    np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-12)
    assert got[0, 0] > 17


@pytest.mark.parametrize("method", HAC_METHODS)
def test_hac_time_reversal_psd_and_score_transform(method):
    rng = np.random.default_rng(924)
    x = rng.normal(size=(23, 4))
    x[:, 3] = x[:, 0] - 2*x[:, 1]  # Rank deficiency is legal for a meat matrix.
    transform = rng.normal(size=(4, 2))
    got = hac_meat(x, 8, method=method)
    np.testing.assert_allclose(got, hac_meat(x[::-1], 8, method=method),
                               rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(hac_meat(x @ transform, 8, method=method),
                               transform.T @ got @ transform, rtol=3e-12, atol=3e-12)
    assert np.linalg.eigvalsh(got).min() >= -3e-12 * max(1, np.linalg.norm(got, 2))


@pytest.mark.parametrize("method", ["auto", "direct", "blocked"])
def test_hac_large_early_value_does_not_erase_later_cross_covariance(method):
    # Whole-matrix relative error is misleading: the 1e32 first diagonal can
    # hide complete loss of a scientifically meaningful O(N*L) off-diagonal.
    x = np.ones((257, 2))
    x[0, 0] = 1e16
    x[:8, 1] = 0.0
    expected = pair_hac_oracle(x, 7)
    got = hac_meat(x, 7, method=method)
    np.testing.assert_allclose(got[0, 1], expected[0, 1], rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(got[1, 0], expected[1, 0], rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("n,lag", [(37, 6), (65, 16)])
@pytest.mark.parametrize("target_rows", [1, 7, 15, 32])
def test_blocked_batch_halos_match_pair_oracle(n, lag, target_rows):
    # Force many small batches without making the independent O(N²) oracle
    # large. Public tests with tiny inputs otherwise never cross a batch halo.
    from fastsandwich.hac import _blocked
    rng = np.random.default_rng(334+n)
    x = rng.normal(size=(n, 2))
    got = _blocked(x, lag, target_rows=target_rows)
    np.testing.assert_allclose(got, pair_hac_oracle(x, lag), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("method", HAC_METHODS)
@pytest.mark.parametrize("bad_lag", [-1, 1.5, True, np.bool_(False)])
def test_hac_rejects_invalid_bandwidth(method, bad_lag):
    with pytest.raises((TypeError, ValueError)):
        hac_meat(np.ones((3, 2)), bad_lag, method=method)


@pytest.mark.parametrize("method", HAC_METHODS)
@pytest.mark.parametrize("bad_x", [np.empty((0, 2)), np.empty((3, 0)),
                                    np.ones((2, 2, 2)), [[1, np.nan]],
                                    [[np.inf]], [[1+2j]]])
def test_hac_rejects_invalid_scores(method, bad_x):
    with pytest.raises((TypeError, ValueError)):
        hac_meat(bad_x, 1, method=method)


@pytest.mark.parametrize("method", CLUSTER_METHODS)
@pytest.mark.parametrize("labels", [
    np.array([9, 3, 9, -1, 3, -1, 9]),
    np.array([2**60, -2**60, 2**60, 9, -2**60, 9, 2**60]),
    np.array(["village-z", "county-a", "village-z", "town", "county-a", "town", "village-z"]),
    np.arange(7),
    np.repeat(100, 7),
])
def test_cluster_against_observation_pair_definition(method, labels):
    rng = np.random.default_rng(633)
    x = rng.normal(size=(7, 3)) + [1, 0, -2]
    np.testing.assert_allclose(cluster_meat(x, labels, method=method),
                               pair_cluster_oracle(x, labels), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("method", CLUSTER_METHODS)
def test_cluster_hand_normalization_and_no_centering(method):
    # Cluster sums are 2+4=6 and 3. Meat is (36+9)/3=15.
    np.testing.assert_allclose(cluster_meat([[2], [3], [4]], [10, 30, 10], method=method),
                               [[15.0]], rtol=0, atol=1e-14)


@pytest.mark.parametrize("method", CLUSTER_METHODS)
def test_cluster_singletons_one_group_psd_and_permutation(method):
    rng = np.random.default_rng(359)
    x = rng.normal(size=(31, 4))
    labels = np.arange(31) % 7
    permutation = rng.permutation(len(x))
    got = cluster_meat(x, labels, method=method)
    np.testing.assert_allclose(cluster_meat(x[permutation], labels[permutation], method=method),
                               got, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(cluster_meat(x, np.arange(len(x)), method=method),
                               x.T @ x / len(x), rtol=3e-12, atol=3e-12)
    total = np.array([math.fsum(x[:, j]) for j in range(x.shape[1])])
    np.testing.assert_allclose(cluster_meat(x, np.zeros(len(x)), method=method),
                               np.outer(total, total)/len(x), rtol=3e-12, atol=3e-12)
    assert np.linalg.eigvalsh(got).min() >= -3e-12 * max(1, np.linalg.norm(got, 2))


@pytest.mark.parametrize("method", CLUSTER_METHODS)
@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_cluster_strides_dtype_and_nonmutation(method, dtype):
    x = np.arange(96, dtype=dtype).reshape(16, 6)[::2, ::2]
    groups = np.array([7, -3, 7, 1, 1, 7, -3, -3])
    old_x, old_groups = x.copy(), groups.copy()
    got = cluster_meat(x, groups, method=method)
    np.testing.assert_allclose(got, pair_cluster_oracle(x, groups),
                               rtol=3e-12, atol=3e-12)
    np.testing.assert_array_equal(x, old_x)
    np.testing.assert_array_equal(groups, old_groups)
    assert got.dtype == np.float64


@pytest.mark.parametrize("method", CLUSTER_METHODS)
@pytest.mark.parametrize("bad_groups", [[0, 1], [[0], [1], [0]], [0, np.nan, 1],
                                       np.array(["a", None, "a"], dtype=object)])
def test_cluster_rejects_invalid_or_missing_labels(method, bad_groups):
    with pytest.raises((TypeError, ValueError)):
        cluster_meat(np.ones((3, 2)), bad_groups, method=method)


@pytest.mark.parametrize("method", CLUSTER_METHODS)
@pytest.mark.parametrize("bad_x", [np.empty((0, 2)), np.empty((3, 0)),
                                    np.ones((2, 2, 2)), [[1, np.nan]],
                                    [[np.inf]], [[1+2j]]])
def test_cluster_rejects_invalid_scores(method, bad_x):
    with pytest.raises((TypeError, ValueError)):
        cluster_meat(bad_x, np.zeros(len(bad_x)), method=method)
