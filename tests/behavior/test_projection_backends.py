import numpy as np

from pyreghdfe import HDFEAbsorber, reghdfe
from pyreghdfe.fe_projection import build_group_index, project_indexed_inplace, max_relative_update


def _rng_data(seed=606, n=12000):
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 500, n, dtype=np.int32)
    f2 = rng.integers(0, 350, n, dtype=np.int32)
    x = rng.normal(size=(n, 4))
    y = x @ np.array([0.7, -0.3, 1.1, 0.2]) + rng.normal(size=500)[f1] + rng.normal(size=350)[f2] + rng.normal(scale=.2, size=n)
    return rng, f1, f2, x, y


def test_group_index_is_complete_and_grouped():
    codes = np.array([2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    idx = build_group_index(codes, 3)
    np.testing.assert_array_equal(np.sort(idx.order), np.arange(codes.size))
    for g in range(3):
        rows = idx.order[idx.indptr[g]:idx.indptr[g + 1]]
        assert np.all(codes[rows] == g)


def test_indexed_projection_matches_direct_group_means():
    rng = np.random.default_rng(1)
    n = 5000
    codes = rng.integers(0, 180, n, dtype=np.int32)
    a = rng.normal(size=(n, 5))
    ref = a.copy()
    for g in range(180):
        rows = codes == g
        if np.any(rows):
            ref[rows] -= ref[rows].mean(axis=0)
    got = a.copy()
    project_indexed_inplace(got, build_group_index(codes, 180), threads=2)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_weighted_indexed_projection_matches_direct_weighted_means():
    rng = np.random.default_rng(2)
    n = 4500
    codes = rng.integers(0, 150, n, dtype=np.int32)
    w = .2 + rng.random(n)
    a = rng.normal(size=(n, 3))
    denom = np.bincount(codes, weights=w, minlength=150)
    ref = a.copy()
    for g in range(150):
        rows = codes == g
        if np.any(rows):
            ref[rows] -= np.average(ref[rows], axis=0, weights=w[rows])
    got = a.copy()
    project_indexed_inplace(got, build_group_index(codes, 150), weights=w, denom=denom, threads=2)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_indexed_absorber_matches_fused_absorber():
    _, f1, f2, x, y = _rng_data()
    block = np.column_stack([y, x])
    fused = HDFEAbsorber([f1, f2], projection_backend='fused', absorb_threads=1, tol=1e-10)
    indexed = HDFEAbsorber([f1, f2], projection_backend='indexed', absorb_threads=2, tol=1e-10)
    a = fused.residualize(block)
    b = indexed.residualize(block)
    np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-9)


def test_auto_backend_uses_indexed_when_eligible():
    _, f1, f2, _, _ = _rng_data(n=10050)
    absorber = HDFEAbsorber(
        [f1, f2], projection_backend='auto', absorb_threads=2,
        projection_min_nobs=1000, projection_memory_budget_mb=256,
    )
    assert absorber.projection_backend == 'indexed'
    assert absorber.index_bytes > 0


def test_auto_backend_falls_back_when_index_budget_is_zero():
    _, f1, f2, _, _ = _rng_data(n=10050)
    absorber = HDFEAbsorber(
        [f1, f2], projection_backend='auto', absorb_threads=2,
        projection_min_nobs=1000, projection_memory_budget_mb=0,
    )
    assert absorber.projection_backend == 'fused'
    assert absorber.index_bytes == 0


def test_public_api_indexed_matches_single_thread_fused_and_thread_override_is_local():
    _, f1, f2, x, y = _rng_data(n=15000)
    base = reghdfe(None, y=y, x=x, absorb=[f1, f2], projection_backend='fused', absorb_threads=1, tol=1e-9)
    fast = reghdfe(None, y=y, x=x, absorb=[f1, f2], projection_backend='indexed', absorb_threads=2, tol=1e-9)
    np.testing.assert_allclose(base.params, fast.params, atol=2e-8, rtol=2e-8)
    absorber = HDFEAbsorber([f1, f2], projection_backend='indexed', absorb_threads=2, tol=1e-9)
    before = absorber.absorb_threads
    absorber.residualize(np.column_stack([y, x[:, 0]]), absorb_threads=1)
    assert absorber.absorb_threads == before


def test_parallel_convergence_metric_matches_numpy_reference():
    rng = np.random.default_rng(88)
    prev = rng.normal(size=(5000, 6))
    cur = prev + rng.normal(scale=1e-5, size=prev.shape)
    eps = 1e-12
    ref = np.max(np.abs(cur-prev) / (np.maximum(np.abs(prev), 1.0) + eps))
    got = max_relative_update(cur, prev, eps, threads=2)
    np.testing.assert_allclose(got, ref, atol=0, rtol=1e-15)
