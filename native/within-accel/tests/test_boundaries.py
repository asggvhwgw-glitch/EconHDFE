import math

import numpy as np
import pytest
from test_native import native, fixture, oracle


def certification_case():
    """Exact dyadic projection with unequal masses and a weak FE direction.

    Each equal-weight tuple occurs twice with opposite residuals. All RHS/FE
    terms are dyadic, so the stored arrays have the stated projection exactly;
    no iterative or truncated-SVD reference is used for the weak direction.
    """
    rng = np.random.default_rng(1)
    n = 2000
    groups = [rng.integers(l, size=n, dtype=np.uint32) for l in (200, 70, 25)]
    codes = np.column_stack([
        np.unique(c, return_inverse=True)[1] for c in groups
    ]).astype(np.uint32)
    fitted = sum(
        rng.integers(-20, 21, size=int(c.max()) + 1)[c] for c in codes.T
    ) / 16.0
    residual = rng.integers(-20, 21, size=n) / 16.0
    weights = np.repeat(np.ldexp(np.ones(n), rng.integers(-48, 1, size=n)), 2)
    codes = np.repeat(codes, 2, axis=0)
    residual = np.column_stack([residual, -residual]).ravel()
    fitted = np.repeat(fitted, 2)
    truth = np.column_stack([residual, 3.0 * residual])
    y = np.column_stack([fitted + residual, -2.0 * fitted + 3.0 * residual])
    return codes, y, weights, truth


def original_row_metric(codes, weights, residual, y):
    """Independent original-row check with extended-precision group reductions."""
    w = np.asarray(weights / weights.max(), dtype=np.longdouble)
    r = np.asarray(residual, dtype=np.longdouble)
    scale = np.max(np.abs(y), axis=0)
    worst = np.zeros(y.shape[1], dtype=np.longdouble)
    for c in codes.T:
        mass = np.zeros(int(c.max()) + 1, dtype=np.longdouble)
        sums = np.zeros((len(mass), y.shape[1]), dtype=np.longdouble)
        np.add.at(mass, c, w)
        np.add.at(sums, c, w[:, None] * r)
        worst = np.maximum(worst, np.max(np.abs(sums / mass[:, None]), axis=0) / scale)
    return worst


@pytest.mark.parametrize("weight_exp", ["numpy", "scalar"])
def test_progressing_continuation_uses_remaining_budget_before_fallback(weight_exp):
    rng = np.random.default_rng(0)
    n = 4000
    codes = np.column_stack([
        rng.integers(l, size=n, dtype=np.uint32) for l in (400, 150, 35)
    ])
    y = rng.normal(size=(n, 1))
    logw = np.clip(10 * np.random.default_rng(26).normal(size=n), -50, 50)
    # NumPy's vectorized exp differs from scalar libm by 1 ULP on some CPUs.
    # Both are legitimate inputs: retain the seed, target and accuracy gates.
    w = np.exp(logw) if weight_exp == "numpy" else np.fromiter(
        (math.exp(float(v)) for v in logw), dtype=float, count=n,
    )
    solver = native.NativeSolver(codes, weights=w, strategy="plain")
    out, ok, iterations, metrics = solver.solve(y, tol=1e-12)
    assert all(ok) and max(metrics) <= 1e-12
    assert max(iterations) <= 16_000
    assert solver.fallback_count == 0
    assert np.max(original_row_metric(codes, w, out, y)) <= 1e-12
    # Drop only the two known dummy gauges; verify the independent dense solve
    # retains every other direction despite the strongly unequal weights.
    d = np.column_stack([
        np.eye(int(c.max()) + 1)[c][:, 0 if j == 0 else 1:]
        for j, c in enumerate(codes.T)
    ])
    sw = np.sqrt(w / w.max())
    b, _, rank, _ = np.linalg.lstsq(d * sw[:, None], y * sw[:, None], rcond=1e-13)
    assert rank == d.shape[1]
    np.testing.assert_allclose(out, y - d @ b, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("unit,permuted", [(1.0, False), (1e14, True)])
def test_certification_polish_preserves_projection_and_units(unit, permuted):
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    codes, y, w, truth = certification_case()
    if permuted:
        order = np.random.default_rng(92).permutation(len(y))
        codes, y, w, truth = codes[order, ::-1], y[order, ::-1], w[order], truth[order, ::-1]
        w = w * 1e180
    solver = WithinFEAbsorber(
        list(codes.T), weights=w, tol=1e-12, max_iter=16_000,
        strategy="plain", absorb_threads=1, projection_memory_budget_mb=1.0,
    )
    out, info = solver.residualize(y * unit, return_info=True)
    assert info.converged
    assert info.max_update <= 1e-12
    assert info.iterations <= 16_000
    np.testing.assert_allclose(out / unit, truth, atol=2e-9, rtol=2e-8)
    assert np.max(original_row_metric(codes, w, out / unit, y)) <= 1e-12
    assert solver.resource_info["certification_fallbacks"] == 0
    assert solver.resource_info["schur_fallbacks"] == 0


@pytest.mark.parametrize("max_iter,workspace", [
    (1, 512 * 1024**2), (100, 512 * 1024**2), (16_000, 600_000),
])
def test_certification_polish_respects_iteration_and_workspace_limits(max_iter, workspace):
    codes, y, w, truth = certification_case()
    solver = native.NativeSolver(codes, weights=w, strategy="plain", workspace_bytes=workspace)
    out, ok, iterations, errors = solver.solve(y[:, :1], tol=1e-12, max_iter=max_iter)
    assert all(i <= max_iter for i in iterations)
    assert np.isfinite(out).all()
    assert solver.fallback_count == 0
    if max_iter == 1:
        assert not all(ok)
        assert errors[0] > 1e-12
    else:
        # One bounded original-row sweep now certifies this case within the
        # primary scratch, including when a second solver would not fit.
        assert all(ok) and max(errors) <= 1e-12
        np.testing.assert_allclose(out, truth[:, :1], atol=2e-9, rtol=2e-8)
        assert np.max(original_row_metric(codes, w, out, y[:, :1])) <= 1e-12


def test_primary_only_budget_allows_certified_polish():
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    codes, y, w = fixture(k=1)
    solver = WithinFEAbsorber(
        list(codes.T), weights=w, tol=1e-12, strategy="plain",
        absorb_threads=1, projection_memory_budget_mb=100_000 / 1024**2,
    )
    out, info = solver.residualize(y, return_info=True)
    assert info.converged
    np.testing.assert_allclose(out, oracle(codes, y, w), atol=2e-9, rtol=2e-8)
    assert solver.resource_info["certification_fallbacks"] == 0

    codes, y, w, truth = certification_case()
    solver = WithinFEAbsorber(
        list(codes.T), weights=w, tol=1e-12, strategy="plain",
        absorb_threads=1, projection_memory_budget_mb=600_000 / 1024**2,
    )
    out, info = solver.residualize(y[:, :1], return_info=True)
    assert info.converged and info.max_update <= 1e-12
    np.testing.assert_allclose(out, truth[:, :1], atol=2e-9, rtol=2e-8)
    assert np.max(original_row_metric(codes, w, out, y[:, :1])) <= 1e-12
    assert solver.resource_info["certification_fallbacks"] == 0


def test_native_polish_fits_budget_with_other_buffered_rhs():
    codes, y, w, truth = certification_case()
    solver = native.NativeSolver(codes, weights=w, strategy="plain", workspace_bytes=980_000)
    out, ok, iterations, metrics = solver.solve(y, tol=1e-12)
    assert all(ok) and solver.fallback_count == 0
    assert max(iterations) <= 16_000 and max(metrics) <= 1e-12
    np.testing.assert_allclose(out, truth, atol=2e-9, rtol=2e-8)
    assert np.max(original_row_metric(codes, w, out, y)) <= 1e-12
    # This budget admits a second engine for only one bridge column, but the
    # bounded polish now needs neither. Rust admission tests check allocation
    # refusal directly, independently of the route taken by this response.
    solver = native.NativeSolver(codes, weights=w, strategy="plain", workspace_bytes=980_000)
    out, ok, iterations, metrics = solver.solve(y[:, :1], tol=1e-12)
    assert all(ok) and solver.fallback_count == 0
    assert max(iterations) <= 16_000 and max(metrics) <= 1e-12
    np.testing.assert_allclose(out, truth[:, :1], atol=2e-9, rtol=2e-8)
    assert np.max(original_row_metric(codes, w, out, y[:, :1])) <= 1e-12


def test_schur_retained_cross_is_charged_to_workspace():
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    a = np.repeat(np.arange(1000, dtype=np.uint32), 2)
    b = np.column_stack([np.arange(1000) % 33, (np.arange(1000) + 1) % 33]).ravel()
    solver = WithinFEAbsorber(
        [a, b], strategy="schur", tol=1e-12, absorb_threads=1,
        projection_memory_budget_mb=800_000 / 1024**2,
    )
    resident, per_column = solver.workspace_cost()
    assert solver._native.extra_resident_bytes >= 8 * 1000 * 32
    assert resident >= solver._native.extra_resident_bytes
    # Primary-only operation fits, but a second retained solver would not.
    assert resident + per_column <= 800_000 < resident + 2 * per_column
    y = ((a % 17) / 8.0 + b / 16.0)[:, None]
    out, info = solver.residualize(y, return_info=True)
    assert info.converged
    np.testing.assert_allclose(out, 0.0, atol=1e-10)
    assert solver.resource_info["schur_fallbacks"] == 0


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
