"""Scalar kernels, blocked tails and scratch reuse: necessary regressions."""
import numpy as np
import pytest
from econhdfe.hdfe import projection as p
from econhdfe.hdfe.absorber import HDFEAbsorber


def generic_project(a, index, weights, denom):
    with p.numba_thread_limit(2):
        if weights is None:
            kernel = p._project_contiguous_unweighted if index.contiguous else p._project_indexed_unweighted
            kernel(a, index.order, index.indptr)
        else:
            kernel = p._project_contiguous_weighted if index.contiguous else p._project_indexed_weighted
            kernel(a, index.order, index.indptr, weights, denom)


@pytest.mark.parametrize('width,weighted', [(2, False), (5, True), (10, False), (11, True), (14, False), (15, True)])
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


def test_float32_projection_keeps_generic(monkeypatch):
    width, dtype = 11, np.float32
    def forbidden(*args):
        pytest.fail('unqualified input entered scalar specialization')
    for k in range(2, 11):
        for weighted in ('weighted', 'unweighted'):
            monkeypatch.setattr(p, f'_project_scalar{k}_{weighted}', forbidden)
    for mode in ('weighted', 'unweighted'):
        monkeypatch.setattr(p, f'_project_blocked5_{mode}', forbidden)
    codes = np.repeat([2, 0, 1], 5).astype(np.int32)
    idx = p.build_group_index(codes, 4)
    raw = np.arange(15 * width, dtype=dtype).reshape(15, width)
    for w in (None, np.linspace(.1, 2, 15)):
        denom = np.bincount(codes, weights=w, minlength=4)
        a, ref = raw.copy(), raw.copy()
        generic_project(ref, idx, w, denom)
        p.project_indexed_inplace(a, idx, weights=w, denom=denom, threads=2)
        np.testing.assert_array_equal(a, ref)


def test_fused_scratch_reused_and_weight_update_reflected():
    rng = np.random.default_rng(929)
    g = np.tile(np.arange(40, dtype=np.int32), 12)
    raw = rng.normal(size=(480, 5))
    w = None
    a = HDFEAbsorber([g], weights=w, projection_backend="fused", absorb_threads=1)
    first = a.residualize(raw)
    scratch = a._projection_scratch
    second = a.residualize(raw[:, :2])
    assert a._projection_scratch is scratch
    np.testing.assert_allclose(second, first[:, :2], atol=1e-14, rtol=1e-14)
    new_weights = .2 + rng.random(480)
    a.update_weights(new_weights)
    got = a.residualize(raw)
    assert a._projection_scratch is scratch
    expected = raw.copy()
    for level in range(40):
        rows = g == level
        expected[rows] -= np.average(raw[rows], axis=0, weights=new_weights[rows])
    np.testing.assert_allclose(got, expected, atol=1e-14, rtol=1e-14)
