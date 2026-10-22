"""Shared CG safeguards ported from validated OLS commit 3369c29.

Only the parallel test thread count changes from four to two to bound local load.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from econhdfe.hdfe.absorber import HDFEAbsorber


@pytest.mark.parametrize("parallel", [False, True])
def test_roundoff_orthogonal_rhs_does_not_grow_in_cg(parallel, monkeypatch):
    import econhdfe.hdfe.absorber as module
    if parallel:
        monkeypatch.setattr(module, "_CG_PARALLEL_MIN_ROWS", 1)
    rng = np.random.default_rng(67)
    n = 400
    groups = [rng.integers(0, m, n) for m in (13, 17, 19)]
    D = np.column_stack([np.eye(g.max()+1)[g] for g in groups])
    raw = rng.normal(size=(n, 2))
    weights = np.exp(rng.normal(scale=.3, size=n)) if parallel else np.ones(n)
    sw = np.sqrt(weights)
    expected = raw-D@np.linalg.lstsq(D*sw[:, None], raw*sw[:, None], rcond=None)[0]
    raw[:, 0] = expected[:, 0]
    solver = HDFEAbsorber(groups, weights=weights, tol=1e-12, core_reduction="off", absorb_threads=2 if parallel else 1)
    result, info = solver.residualize(raw, return_info=True)
    assert info.converged
    assert_allclose(result, expected, rtol=1e-9, atol=1e-9)



def test_nonpositive_curvature_restarts_from_true_residual(monkeypatch):
    rng = np.random.default_rng(89)
    n = 200
    groups = [rng.integers(0, m, n) for m in (13, 17, 19)]
    D = np.column_stack([np.eye(g.max()+1)[g] for g in groups])
    raw = rng.normal(size=(n, 2))
    expected = raw-D@np.linalg.lstsq(D, raw, rcond=None)[0]
    solver = HDFEAbsorber(groups, tol=1e-12, core_reduction="off", absorb_threads=1)
    original = solver._sym_projection
    calls = 0
    def one_bad_curvature(a, out=None):
        nonlocal calls
        calls += 1
        result = original(a, out=out)
        if calls == 2:
            result *= -1
        return result
    monkeypatch.setattr(solver, "_sym_projection", one_bad_curvature)
    solver._profile_enabled = True
    result, info = solver.residualize(raw, return_info=True)
    assert info.converged
    assert solver._iteration_profile["cg_residual_restarts"] == 1
    assert_allclose(result, expected, rtol=1e-9, atol=1e-9)
