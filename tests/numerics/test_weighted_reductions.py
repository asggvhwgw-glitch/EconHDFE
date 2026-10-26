"""PERF-01: independent reduction, layout, weight and stopping guards.

No timing assertions: noisy performance measurements belong to the benchmark.
"""

from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
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


def test_weighted_multiply_order_does_not_overflow_finite_reference():
    a = np.full((3, 2), 1e200); b = np.full((3, 2), 1e200)
    obj = SimpleNamespace(backend='numpy', xp=np, weights=np.full(3, 1e-250))
    with np.errstate(over='raise', invalid='raise'):
        got = HDFEAbsorber._weighted_colsum(obj, a, b)
        ref = legacy(obj, a, b)
    assert np.all(np.isfinite(got))
    assert_allclose(got, ref, rtol=1e-14)


def test_dynamic_weights_projection_and_zero_weight_invalidation():
    order,core = ('slice', 'on')
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
