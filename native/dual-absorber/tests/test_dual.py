import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from econhdfe_dual import DualAbsorber


def fixture(n=720, rhs=4, seed=29):
    rng = np.random.default_rng(seed)
    g = np.column_stack([rng.integers(0, l, n) for l in (45, 13, 7)]).astype(np.uint32)
    y = rng.normal(size=(n, rhs))
    for j in range(g.shape[1]):
        y += rng.normal(size=(int(g[:, j].max()) + 1, rhs))[g[:, j]]
    w = np.exp(rng.normal(scale=0.5, size=n))
    return g, y, w


def oracle(g, y, w):
    dummy = np.column_stack([np.eye(int(col.max())+1)[col] for col in g.T])
    root = np.sqrt(w / w.max())
    alpha = np.linalg.lstsq(root[:, None] * dummy, root[:, None] * y, rcond=1e-13)[0]
    return y - dummy @ alpha


def group_certificate(g, original, residual, w):
    scale = np.max(np.abs(original), axis=0)
    worst = 0.0
    for col in g.T:
        mass = np.bincount(col, weights=w)
        for j in range(original.shape[1]):
            if scale[j] == 0:
                assert np.all(residual[:, j] == 0)
                continue
            sums = np.bincount(col, weights=w*residual[:, j])
            active = mass > 0
            worst = max(worst, np.max(np.abs(sums[active]/mass[active]))/scale[j])
    return worst


@pytest.mark.parametrize('backend', ['within', 'fereg', 'auto'])
@pytest.mark.parametrize('rhs', [1, 2, 8])
@pytest.mark.parametrize('weighted', [False, True])
def test_dense_oracle(backend, rhs, weighted):
    g, y, w = fixture(rhs=rhs)
    if not weighted: w[:] = 1
    before = y.copy()
    solver = DualAbsorber(g, weights=w, backend=backend, allow_fallback=False)
    out, info = solver.residualize(y, return_info=True)
    assert info['accepted']
    assert max(info['certificate']) <= info['certificate_gate']
    assert group_certificate(g, y, out, w) < 2e-10
    assert_allclose(out, oracle(g, y, w), atol=2e-8, rtol=2e-8)
    assert_array_equal(y, before)
    assert solver.weight_generation == 0


@pytest.mark.parametrize('backend', ['within', 'fereg', 'auto'])
def test_weight_updates_and_transaction(backend):
    g, y, w = fixture()
    s = DualAbsorber(g, backend=backend)
    for i, wi in enumerate([w, np.ones(len(w)), w[::-1].copy()]):
        s.update_weights(wi)
        out = s.residualize(y)
        assert_allclose(out, oracle(g, y, wi), atol=2e-8, rtol=2e-8)
        assert s.weight_generation == i+1
    last = out.copy()
    for bad in [np.zeros(len(w)), -w, np.full(len(w), np.nan), w[:-1]]:
        with pytest.raises(ValueError): s.update_weights(bad)
        assert s.weight_generation == 3
        assert_allclose(s.residualize(y), last, atol=2e-9, rtol=2e-9)
    assert s.diagnostics['kernel_builds'] == 1


@pytest.mark.parametrize('backend', ['within', 'fereg', 'auto'])
@pytest.mark.parametrize('unit', [1e-100, 1.0, 1e100])
def test_units_and_noncontiguous(backend, unit):
    g, y, w = fixture()
    s = DualAbsorber(g[::-1, ::-1], weights=(w[::-1]*unit), backend=backend)
    out = s.residualize(y[::-1, ::-1]*unit)/unit
    assert_allclose(out, oracle(g, y, w)[::-1, ::-1], atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize('backend', ['within', 'fereg'])
def test_sparse_labels_and_row_order(backend):
    g, y, w = fixture()
    sparse = g.astype(np.uint64)*1234567 + 98765432
    a = DualAbsorber(sparse, weights=w, backend=backend).residualize(y)
    p = np.random.default_rng(73).permutation(len(y))
    b = DualAbsorber(sparse[p], weights=w[p], backend=backend).residualize(y[p])
    assert_allclose(a[p], b, atol=2e-8, rtol=2e-8)
    assert_allclose(a, oracle(g, y, w), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize('backend', ['within', 'fereg'])
def test_duplicate_nested_disconnected_and_singleton(backend):
    # Collinear FE columns, disjoint graph components, and a singleton are valid
    # numerical projections, not a license for statistical sample deletions.
    g, y, w = fixture(n=400, rhs=3)
    g = np.column_stack([g[:, 0], g[:, 0], g[:, 0]//5, g[:, 1]])
    g[200:, 0:3] += 50
    g[200:, 3] += 20
    g[-1, 0] = 200
    out = DualAbsorber(g, weights=w, backend=backend).residualize(y)
    assert out.shape == y.shape
    assert_allclose(out, oracle(g, y, w), atol=4e-8, rtol=3e-8)
    assert np.max(np.abs(out[-1])) < 1e-8


@pytest.mark.parametrize('backend', ['within', 'fereg', 'auto'])
def test_zero_constant_and_empty_columns(backend):
    g, _, w = fixture()
    y = np.column_stack([np.zeros(len(w)), np.ones(len(w)), np.full(len(w), 1e100)])
    s = DualAbsorber(g, weights=w, backend=backend)
    out = s.residualize(y)
    assert np.max(np.abs(out[:, :2])) < 1e-8
    assert np.max(np.abs(out[:, 2]/1e100)) < 1e-8
    assert s.residualize(np.empty((len(w), 0))).shape == (len(w), 0)
    assert s.residualize(y[:, 1]).shape == (len(w),)


@pytest.mark.parametrize('backend', ['within', 'fereg'])
def test_blocking_preserves_result(backend):
    g, y, w = fixture(rhs=8)
    s = DualAbsorber(g, weights=w, backend=backend, scratch_bytes=1024)
    out, info = s.residualize(y, return_info=True)
    assert info['rhs_block_width'] == 1
    assert_allclose(out, oracle(g, y, w), atol=2e-8, rtol=2e-8)


def test_failure_is_not_reported_as_convergence():
    g, y, w = fixture()
    s = DualAbsorber(g, weights=w, backend='within', allow_fallback=False)
    with pytest.raises(RuntimeError): s.residualize(y, max_iter=1)
    assert not s.diagnostics['accepted']
    # The same object can still recover under a proper iteration budget.
    out = s.residualize(y, max_iter=6000)
    assert_allclose(out, oracle(g, y, w), atol=2e-8, rtol=2e-8)


def test_failed_within_can_fall_back_to_exact_reduced_solve():
    g, y, w = fixture()
    s = DualAbsorber(g, weights=w, backend='within', allow_fallback=True)
    out, info = s.residualize(y, max_iter=1, return_info=True)
    assert info['accepted'] and info['fallback_count'] == 1
    assert info['backend'] == 'fereg'
    assert any(not a.get('accepted', False) for a in info['attempts'])
    assert_allclose(out, oracle(g, y, w), atol=2e-8, rtol=2e-8)


def test_validation():
    g, y, w = fixture()
    for bad in [g.astype(float), np.full_like(g, -1, dtype=np.int32), np.array([[2**40]], dtype=np.uint64), np.empty((0, 2), dtype=np.uint32)]:
        with pytest.raises(ValueError): DualAbsorber(bad)
    with pytest.raises(ValueError): DualAbsorber(g, weights=np.zeros(len(y)))
    s = DualAbsorber(g)
    for bad in [np.full_like(y, np.nan), np.full_like(y, np.inf)]:
        with pytest.raises(RuntimeError): s.residualize(bad)
    with pytest.raises(ValueError): s.residualize(y[:-1])
    with pytest.raises(ValueError): s.residualize(y, max_iter=0)
    with pytest.raises(ValueError): s.residualize(y, tol=0)
    out = s.residualize(y)
    assert np.isfinite(out).all()


def test_auto_route_is_explicit_and_stable():
    g, y, _ = fixture()
    s = DualAbsorber(g)
    assert s.backend == 'uninitialized'
    _, d = s.residualize(y, return_info=True)
    assert d['backend'] == 'fereg'
    assert d['decision_reason'] == 'small_reduced_dimension'
    s.residualize(y[:, :1])
    assert s.diagnostics['kernel_builds'] == 1
