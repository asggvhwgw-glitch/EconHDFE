"""Bounded scratch with independent dense-SVD and layout/scale guards."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from econhdfe.compute.stable_linalg import equilibrated_lstsq


def reference(X, y):
    scale = np.max(np.abs(X), axis=0)
    scale[scale == 0] = 1
    design = X / scale
    beta, _, numerical_rank, _ = np.linalg.lstsq(
        design, y, rcond=np.finfo(float).eps * max(X.shape))
    _, s, vh = np.linalg.svd(design, full_matrices=False)
    keep = s > np.finfo(float).eps * max(X.shape) * s[0]
    inverse = (vh[keep].T / s[keep]) / scale[:, None]
    return beta / scale, inverse @ inverse.T, numerical_rank


@pytest.mark.parametrize('shape,layout,deficient', [
    ((17, 5), "C", False), ((17, 5), "F", True), ((17, 5), "sliced", False), ((17, 5), "readonly", True),
    ((9, 14), "C", True), ((9, 14), "F", False), ((9, 14), "sliced", True), ((9, 14), "readonly", False),
    ((123, 6), "C", False), ((123, 6), "F", True), ((123, 6), "sliced", True), ((123, 6), "readonly", False),
])
def test_dense_svd_oracle_layout_rank_and_input_ownership(shape, layout, deficient):
    rng = np.random.default_rng(6403)
    X = rng.normal(size=shape)
    if deficient:
        X[:, -1] = 0
        X[:, -2] = X[:, 0] + X[:, 1]
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "sliced":
        storage = np.zeros((shape[0] * 2, shape[1] * 2))
        storage[::2, ::2] = X
        X = storage[::2, ::2]
    elif layout == "readonly":
        X.setflags(write=False)
    y = rng.normal(size=shape[0])
    before, y_before = X.copy(), y.copy()
    expected = reference(X, y)
    actual = equilibrated_lstsq(X, y, chunk_rows=7)
    assert actual[2] == expected[2]
    assert_allclose(actual[0], expected[0], rtol=2e-10, atol=2e-12)
    assert_allclose(actual[1], expected[1], rtol=2e-10, atol=2e-12)
    assert_array_equal(X, before)
    assert_array_equal(y, y_before)


def test_extreme_units_and_column_permutation():
    rng = np.random.default_rng(643)
    X = rng.normal(size=(81, 6))
    y = rng.normal(size=81)
    units = np.array([1e-120, -1e-80, 1, -1e40, 1e80, 1e120])
    order = np.array([5, 0, 3, 1, 4, 2])
    beta, cov, numerical_rank = equilibrated_lstsq(X, y, chunk_rows=11)
    b, V, r = equilibrated_lstsq((X * units)[:, order], y, chunk_rows=11)
    assert r == numerical_rank == 6
    assert_allclose(b * units[order], beta[order], rtol=1e-10, atol=1e-12)
    assert_allclose(V * units[order, None] * units[None, order], cov[np.ix_(order, order)],
                    rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["X", "y"])
def test_nonfinite_in_last_chunk_rejected_before_qr(monkeypatch, value, target):
    import econhdfe.compute.stable_linalg as mod
    X = np.ones((29, 4)); y = np.ones(29)
    if target == "X":
        X[-1, -1] = value
    else:
        y[-1] = value
    def forbidden(*args, **kwargs):
        pytest.fail("QR executed before all chunks were validated")
    monkeypatch.setattr(mod.la, "qr", forbidden)
    with pytest.raises(ValueError, match="finite"):
        equilibrated_lstsq(X, y, chunk_rows=7)


def test_abs_workspace_bounded_and_division_writes_to_output(monkeypatch):
    import econhdfe.compute.stable_linalg as mod
    X = np.arange(333., dtype=float).reshape(111, 3)
    y = np.arange(111., dtype=float)
    original_abs, original_divide = np.abs, np.divide
    seen_abs, seen_divide = [], []
    def bounded_abs(a, *args, **kwargs):
        if np.ndim(a) == 2:
            seen_abs.append(np.shape(a))
            assert np.shape(a)[0] <= 13
        return original_abs(a, *args, **kwargs)
    def direct_divide(a, b, *args, **kwargs):
        if np.ndim(a) == 2 and np.shape(a)[1] == 3:
            out = kwargs.get("out")
            assert out is not None
            assert not np.shares_memory(out, X)
            seen_divide.append(out.shape)
        return original_divide(a, b, *args, **kwargs)
    monkeypatch.setattr(mod.np, "abs", bounded_abs)
    monkeypatch.setattr(mod.np, "divide", direct_divide)
    equilibrated_lstsq(X, y, chunk_rows=13)
    assert seen_abs and seen_divide


@pytest.mark.parametrize("n,k", [(0, 3), (4, 0), (0, 0)])
def test_empty_inputs_keep_shapes(n, k):
    b, V, r = equilibrated_lstsq(np.empty((n, k)), np.zeros(n))
    assert b.shape == (k,) and V.shape == (k, k) and r == 0


def test_empty_design_still_validates_outcome():
    with pytest.raises(ValueError, match="finite"):
        equilibrated_lstsq(np.empty((2, 0)), np.array([0., np.nan]))
