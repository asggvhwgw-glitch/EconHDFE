import numpy as np
import pytest
from test_native import native, fixture, oracle


def test_total_iteration_budget_is_respected():
    codes, y, w = fixture(n=4000, k=2)
    solver = native.NativeSolver(codes, weights=w, strategy="plain")
    out, ok, it, err = solver.solve(y, tol=1e-12, max_iter=1)
    assert all(i <= 1 for i in it)
    assert not all(ok)
    assert np.isfinite(out).all()


def test_wide_rhs_and_copy_contract():
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    codes, y, w = fixture(k=16)
    solver = WithinFEAbsorber(list(codes.T), weights=w, tol=1e-11, absorb_threads=2)
    before = y.copy()
    out, info = solver.residualize(y, return_info=True, absorb_threads=1)
    assert info.converged
    np.testing.assert_array_equal(y, before)
    np.testing.assert_allclose(out, oracle(codes, y, w), atol=2e-9, rtol=2e-8)
    assert solver.resource_info["threads"] == 2
    target = y.copy(); out = solver.residualize(target, copy=False)
    assert np.shares_memory(target, out)
    np.testing.assert_allclose(target, oracle(codes, y, w), atol=2e-9, rtol=2e-8)
    target.flags.writeable = False
    with pytest.raises(ValueError):
        solver.residualize(target, copy=False)


def test_zero_weight_projection_uses_existing_fallback():
    from econhdfe.hdfe.weighted_projection import WeightedFEProjector
    codes, y, w = fixture()
    projector = WeightedFEProjector(list(codes.T), engine="optimized", method="within", absorb_threads=2)
    out, info = projector.residualize(y, w, tol=1e-11, return_info=True)
    assert info.converged
    w[::5] = 0.0
    out, info = projector.residualize(y, w, tol=1e-11, return_info=True)
    assert info.converged and projector.resource_info["backend"] == "generic"
    keep = w > 0
    # Zero-weight rows have no unique weighted-projection target; certify active rows.
    np.testing.assert_allclose(out[keep], oracle(codes, y, w)[keep], atol=2e-8, rtol=2e-8)
    projector.residualize(y, np.ones(len(w)), tol=1e-11)
    assert projector.resource_info["backend"] == "within"
