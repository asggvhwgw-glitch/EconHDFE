"""PERF-01: independent reduction, layout, weight and stopping guards.

No timing assertions: noisy performance measurements belong to the benchmark.
"""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from econhdfe.hdfe.absorber import HDFEAbsorber


def legacy(self, a, b):
    if self.weights is None:
        return np.sum(a * b, axis=0)
    return np.sum(self.weights[:, None] * a * b, axis=0)


class LegacyAbsorber(HDFEAbsorber):
    _weighted_colsum = legacy


def layout(a, name):
    if name == 'F':
        return np.asfortranarray(a)
    if name == 'slice':
        out = np.empty((a.shape[0] * 2, a.shape[1] * 2), dtype=a.dtype)
        out[::2, ::2] = a
        return out[::2, ::2]
    if name == 'reverse':
        return a[::-1, ::-1]
    return a.copy(order='C')


@pytest.mark.parametrize('order,shape,weighted', [
    (order, shape, (i+j)%2 == 0)
    for i, order in enumerate(("C", "F", "slice", "reverse"))
    for j, shape in enumerate(((37,5), (0,3), (6,0), (1,7)))
])
def test_layout_empty_and_input_preservation(order, shape, weighted):
    rng = np.random.default_rng(6301)
    a = layout(rng.normal(size=shape), order)
    b = layout(rng.normal(size=shape), order)
    w = np.geomspace(1e-6, 1e6, shape[0]) if weighted else None
    obj = SimpleNamespace(backend='numpy', xp=np, weights=w)
    before = (a.copy(), b.copy(), None if w is None else w.copy())
    a.flags.writeable = b.flags.writeable = False
    got = HDFEAbsorber._weighted_colsum(obj, a, b)
    want = np.array([math.fsum((float(a[i,j]) if w is None else float(w[i])*float(a[i,j]))
                               *float(b[i,j]) for i in range(shape[0])) for j in range(shape[1])])
    # Forward error bound, not relative error against a potentially cancelling sum.
    products = a*b if w is None else w[:, None]*a*b
    bound = 4 * (shape[0]+2) * np.finfo(float).eps * np.sum(np.abs(products), axis=0)
    assert got.shape == (shape[1],) and got.dtype == np.float64
    assert np.all(np.abs(got-want) <= bound)
    assert_array_equal(a, before[0]); assert_array_equal(b, before[1])
    if w is not None:
        assert_array_equal(w, before[2])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('weighted', [False, True])
def test_dtype_and_repeated_reduction(dtype, weighted):
    rng = np.random.default_rng(6302)
    a = rng.normal(size=(81, 4)).astype(dtype)
    b = rng.normal(size=(81, 4)).astype(dtype)
    obj = SimpleNamespace(backend='numpy', xp=np, weights=np.linspace(.1, 3, 81) if weighted else None)
    got = HDFEAbsorber._weighted_colsum(obj, a, b)
    ref = legacy(obj, a, b)
    assert got.dtype == ref.dtype
    assert_allclose(got, ref, rtol=2e-6 if dtype==np.float32 else 1e-13, atol=2e-6 if dtype==np.float32 else 1e-13)
    assert_array_equal(got, HDFEAbsorber._weighted_colsum(obj, a, b))


@pytest.mark.parametrize('scale', [1e-140, 1., 1e140])
def test_extreme_finite_products_and_cancellation(scale):
    # Positive weights, mixed signs, including large cancellation and zero terms.
    a = np.array([[1., 2.],[-1., 1.],[1e-10, -2.],[0., 0.]])*scale
    b = np.array([[1., 1.],[1., -1.],[1., 1.],[2., 3.]])/scale
    w = np.array([1e40, 1e40, 1., 0.])
    obj = SimpleNamespace(backend='numpy', xp=np, weights=w)
    got = HDFEAbsorber._weighted_colsum(obj, a, b)
    terms = w[:,None]*a*b
    ref = np.array([math.fsum(terms[:,j]) for j in range(2)])
    assert np.all(np.isfinite(got))
    assert np.all(abs(got-ref) <= 32*np.finfo(float).eps*np.sum(abs(terms),axis=0))


def test_weighted_multiply_order_does_not_overflow_finite_reference():
    a = np.full((3, 2), 1e200); b = np.full((3, 2), 1e200)
    obj = SimpleNamespace(backend='numpy', xp=np, weights=np.full(3, 1e-250))
    with np.errstate(over='raise', invalid='raise'):
        got = HDFEAbsorber._weighted_colsum(obj, a, b)
        ref = legacy(obj, a, b)
    assert np.all(np.isfinite(got))
    assert_allclose(got, ref, rtol=1e-14)


@pytest.mark.parametrize('weighted', [False, True])
def test_non_numpy_branch_not_redirected_to_einsum(monkeypatch, weighted):
    # Backend dispatch contract only; this does not certify real GPU execution.
    obj = SimpleNamespace(backend='cupy', xp=np, _gpu_weights=np.arange(1.,5.) if weighted else None)
    a = np.arange(12.,dtype=float).reshape(4,3)
    ref = np.sum(a*a if not weighted else obj._gpu_weights[:,None]*a*a,axis=0)
    def forbidden(*args, **kwargs):
        raise AssertionError('non-NumPy route changed')
    monkeypatch.setattr(np,'einsum',forbidden)
    assert_array_equal(HDFEAbsorber._weighted_colsum(obj,a,a),ref)


@pytest.mark.parametrize('order', ['C','F','slice'])
@pytest.mark.parametrize('core', ['off','on'])
def test_dynamic_weights_projection_and_zero_weight_invalidation(order, core):
    rng = np.random.default_rng(6303)
    rows = np.array([(i,j,k) for i in range(4) for j in range(3) for k in range(2)]*3+[(4,3,2),(5,4,3)])
    gs = list(rows.T); n=len(rows); y=layout(rng.normal(size=(n,3)),order)
    D = np.column_stack([np.eye(int(g.max())+1)[g] for g in gs])
    w = np.geomspace(.1, 10., n)
    a = HDFEAbsorber(gs,weights=w,tol=1e-12,max_iter=20000,core_reduction=core,absorb_threads=1)
    for wi in [w,w[::-1].copy(),np.r_[0.,w[1:]]]:
        a.update_weights(wi)
        got, info = a.residualize(y,return_info=True)
        sw = np.sqrt(wi)
        oracle = y-D@np.linalg.lstsq(sw[:,None]*D,sw[:,None]*y,rcond=None)[0]
        old, old_info = LegacyAbsorber(gs,weights=wi,tol=1e-12,max_iter=20000,
                          core_reduction=core,absorb_threads=1).residualize(y,return_info=True)
        support=wi>0
        assert info.converged and old_info.converged
        assert_allclose(got[support], oracle[support], rtol=2e-8,atol=2e-8)
        assert_allclose(got,old,rtol=2e-8,atol=2e-8)
        assert_allclose(D.T@(wi[:,None]*got),0.,atol=2e-8)
        if not np.all(support):
            assert a._core_plan is None


@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('side',[1.-1e-6,1.+1e-6])
def test_near_stopping_boundary_keeps_decision_and_budget(weighted,side):
    rng=np.random.default_rng(6304);n=320
    gs=[rng.integers(0,17,n),rng.integers(0,11,n),rng.integers(0,5,n)]
    y=rng.normal(size=(n,3));w=np.geomspace(.2,5,n) if weighted else None
    kw=dict(weights=w,core_reduction='off',absorb_threads=1,max_iter=1,acceleration='cg')
    _, metric=LegacyAbsorber(gs,tol=1e-15,**kw).residualize(y,return_info=True)
    assert np.isfinite(metric.max_update) and metric.max_update>0
    tol=metric.max_update*side
    got,info=HDFEAbsorber(gs,tol=tol,**kw).residualize(y,return_info=True)
    ref,old=LegacyAbsorber(gs,tol=tol,**kw).residualize(y,return_info=True)
    assert info.converged==old.converged==(side>1)
    assert info.iterations==old.iterations==1
    assert info.criterion==old.criterion=='hestenes'
    assert_allclose(got,ref,rtol=1e-12,atol=1e-12)
