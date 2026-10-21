"""Scalar kernels, blocked tails and scratch reuse: necessary regressions."""

import numpy as np
import pytest
from econhdfe.hdfe import projection as p


def generic_project(a, index, weights, denom):
    with p.numba_thread_limit(2):
        if weights is None:
            kernel = p._project_contiguous_unweighted if index.contiguous else p._project_indexed_unweighted
            kernel(a, index.order, index.indptr)
        else:
            kernel = p._project_contiguous_weighted if index.contiguous else p._project_indexed_weighted
            kernel(a, index.order, index.indptr, weights, denom)


@pytest.mark.parametrize('width,weighted', [
    (2, False),
    (10, False),
    (11, True),
])
def test_scalar_and_blocked_group_oracle(width, weighted):
    rng = np.random.default_rng(4631)
    codes = np.repeat([7, 1, 9, 0], [37, 1, 61, 14]).astype(np.int32)
    rng.shuffle(codes)
    raw = rng.normal(size=(len(codes), width))
    a = raw.copy()
    w = np.exp(rng.normal(size=len(codes))) if weighted else None
    if weighted:
        w[codes == 9] = 0
        w[::7] = 0
    idx = p.build_group_index(codes, 12)
    # Both compact and wide row-index types must preserve the same arithmetic.
    if width % 2 == 0:
        idx.order = idx.order.astype(np.int64)
    denom = np.bincount(codes, weights=w, minlength=12)
    ref = raw.copy()
    generic_project(ref, idx, w, denom)
    p.project_indexed_inplace(a, idx, weights=w, denom=denom, threads=2)
    np.testing.assert_array_equal(a, ref)
    expected = raw.copy()
    for g in np.unique(codes):
        rows = codes == g
        if denom[g] > 0:
            expected[rows] -= np.average(raw[rows], axis=0, weights=None if w is None else w[rows])
    np.testing.assert_allclose(a, expected, atol=2e-14, rtol=2e-14)
