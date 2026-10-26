"""Reuse the independent existing weak-design oracle, not the candidate backend."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from test_native import native


@pytest.mark.parametrize('hidden,weighted,pool,threads', [
    (False,False,1,1), (False,True,3,2), (True,False,3,1), (True,True,1,2),
])
def test_public_native_weak_design_keeps_stable_covariance(hidden, weighted, pool, threads, monkeypatch):
    from econhdfe import olshdfe
    root=Path(__file__).resolve().parents[3]
    path=root/'tests/numerics/test_conditioned_absorption.py'
    spec=importlib.util.spec_from_file_location('native_weak_design_oracle',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,'olshdfe',lambda *a,**kw: olshdfe(*a,**dict(kw,method='within')))
    module.test_near_collinear_dense_fwl_and_covariance(hidden,weighted,pool,threads)


def test_nearly_disconnected_small_schur_is_not_silently_solved():
    codes=np.array([[0,0],[0,0],[1,1],[1,1],[0,1],[0,1]],dtype=np.uint32)
    w=np.array([1.,1.,1.,1.,1e-12,1e-12])
    with pytest.raises(ValueError,match='Schur rejected'):
        native.NativeSolver(codes,weights=w,strategy='schur')
    solver=native.NativeSolver(codes,weights=w,strategy='auto')
    assert solver.route != 'schur'
    truth=np.array([1.,-1.,2.,-2.,3.,-3.])[:,None]
    y=truth+2*codes[:,0,None]-0.7*codes[:,1,None]
    r,ok,_,_=solver.solve(y,tol=1e-11)
    assert all(ok)
    np.testing.assert_allclose(r,truth,atol=2e-8,rtol=2e-8)
