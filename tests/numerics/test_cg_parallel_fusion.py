"""Parallel CG updates and fused reductions against dense and serial oracles."""
import numpy as np

import econhdfe.hdfe.absorber as module


def test_wide_parallel_cg_matches_serial_and_dense(monkeypatch):
    width = 6
    rng=np.random.default_rng(861)
    n=240;groups=[rng.integers(0,l,n) for l in (29,31,13)]
    raw=rng.normal(size=(n,width));w=np.exp(rng.normal(size=n))
    w[::11]=0
    store=np.empty((n*2,width*2));store[::2,::2]=raw;raw=store[::2,::2]
    original=raw.copy()
    kw=dict(weights=w,projection_backend='indexed',core_reduction='off',tol=1e-10,absorb_threads=4)
    monkeypatch.setattr(module,'_CG_PARALLEL_MIN_ROWS',n+1)
    ref,ri=module.HDFEAbsorber(groups,**kw).residualize(raw,return_info=True)
    monkeypatch.setattr(module,'_CG_PARALLEL_MIN_ROWS',1)
    obj=module.HDFEAbsorber(groups,**kw);obj._profile_enabled=True
    # Exercise writable strided inputs at both thread counts.
    second_store=np.empty((n*2,width*2));second_store[::2,::2]=original
    same_layout=second_store[::2,::2]
    got,info=obj.residualize(raw,copy=False,return_info=True)
    assert ri.converged and info.converged
    assert obj._iteration_profile['cg_parallel_fused']
    np.testing.assert_allclose(got,ref,atol=1e-8,rtol=1e-8)
    D=np.column_stack([np.eye(g.max()+1)[g] for g in groups]);sw=np.sqrt(w)
    b=np.linalg.lstsq(D*sw[:,None],original*sw[:,None],rcond=None)[0]
    np.testing.assert_allclose(got,original-D@b,atol=1e-8,rtol=1e-8)
    assert np.shares_memory(got,raw)
    again=obj.residualize(same_layout,copy=False,absorb_threads=2)
    np.testing.assert_array_equal(got,again)  # 固定chunk数保证mask变化不改归约顺序。

def test_float32_keeps_serial_cg(monkeypatch):
    rows, width, dtype, threads, active = 120, 6, np.float32, 2, False
    monkeypatch.setattr(module,'_CG_PARALLEL_MIN_ROWS',120)
    rng=np.random.default_rng(862)
    groups=[rng.integers(0,l,rows) for l in (13,17,19)]
    obj=module.HDFEAbsorber(groups,projection_backend='indexed',core_reduction='off',dtype=dtype,
                           absorb_threads=threads,tol=1e-5)
    obj._profile_enabled=True
    obj.residualize(rng.normal(size=(rows,width)).astype(dtype))
    assert obj._iteration_profile['cg_parallel_fused']==active
    assert obj._iteration_profile['cg_projection_includes_inner_product']==active


def test_weighted_fused_product_does_not_overflow_finite_reference():
    a=np.full((33,2),1e200);r=a.copy();u=a.copy();v=np.zeros_like(a);w=np.full(33,1e-250)
    norm=module._cg_update_residuals_and_norm(a,r,u,v,np.zeros(2),w)
    dot=module._cg_subtract_and_inner_product(a,v,w)
    for value in (norm,dot):
        assert np.isfinite(value).all();np.testing.assert_allclose(value,33e150,rtol=1e-14)
