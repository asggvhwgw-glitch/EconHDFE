"""Independent dense FWL and public econhdfe parity gates."""
import importlib
import os
import numpy as np
import pytest

if os.getenv("REQUIRE_WITHIN_NATIVE") == "1":
    native = importlib.import_module("_econhdfe_within")
else:
    native = pytest.importorskip("_econhdfe_within")


def fixture(seed=81, n=720, k=3):
    rng = np.random.default_rng(seed)
    codes = np.column_stack([rng.integers(l, size=n, dtype=np.uint32) for l in (17, 7, 3)])
    return codes, rng.normal(size=(n, k)), np.exp(rng.normal(size=n))


def oracle(codes, y, weights):
    d = np.column_stack([np.eye(int(g.max())+1)[g] for g in codes.T])
    s = np.sqrt(weights / weights.max())
    b = np.linalg.lstsq(d*s[:, None], y*s[:, None], rcond=1e-13)[0]
    return y - d @ b


@pytest.mark.parametrize("strategy", ["plain", "compressed", "schur", "auto"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("k", [1, 4])
def test_dense_oracle(strategy, weighted, k):
    codes, y, w = fixture(k=k)
    w = w if weighted else np.ones(len(y))
    codes = np.tile(codes, (2, 1)); y = np.vstack((y, y+0.7)); w = np.tile(w, 2)
    solver = native.NativeSolver(codes, weights=w, strategy=strategy, threads=2)
    out, ok, _, error = solver.solve(y, tol=1e-11)
    assert all(ok), error
    np.testing.assert_allclose(out, oracle(codes, y, w), rtol=2e-8, atol=2e-9)
    assert out.shape == y.shape


@pytest.mark.parametrize("unit", [1e-14, 1.0, 1e14])
@pytest.mark.parametrize("weight_unit", [1e-180, 1e180])
def test_scale_and_strides(unit, weight_unit):
    codes, y, w = fixture()
    y0 = y.copy(); codes0 = codes.copy(); w0 = w.copy()
    solver = native.NativeSolver(codes[:, ::-1], weights=w*weight_unit, strategy="compressed")
    out, ok, _, _ = solver.solve((y*unit)[:, ::-1], tol=1e-11)
    assert all(ok)
    np.testing.assert_allclose(out/unit, oracle(codes, y, w)[:, ::-1], atol=3e-9, rtol=3e-8)
    np.testing.assert_array_equal(y, y0); np.testing.assert_array_equal(codes, codes0); np.testing.assert_array_equal(w, w0)


def test_reweight_is_transactional():
    codes, y, w = fixture()
    solver = native.NativeSolver(codes, strategy="compressed")
    for new in (w, w[::-1].copy()):
        solver.update_weights(new)
        out, ok, _, _ = solver.solve(y, tol=1e-11)
        assert all(ok)
        np.testing.assert_allclose(out, oracle(codes, y, new), atol=2e-9, rtol=2e-8)
    before = solver.solve(y, tol=1e-11)[0].copy(); version = solver.weight_generation
    with pytest.raises(ValueError):
        solver.update_weights(np.zeros(len(w)))
    assert solver.weight_generation == version
    np.testing.assert_allclose(solver.solve(y, tol=1e-11)[0], before, atol=1e-10)


def test_singular_schur_declines_and_auto_is_exact():
    codes, y, w = fixture(); codes[:, 1] = codes[:, 0]
    with pytest.raises(ValueError, match="Schur rejected"):
        native.NativeSolver(codes, weights=w, strategy="schur")
    solver = native.NativeSolver(codes, weights=w, strategy="auto")
    assert solver.route != "schur"
    out, ok, _, _ = solver.solve(y, tol=1e-11)
    assert all(ok)
    np.testing.assert_allclose(out, oracle(codes, y, w), atol=2e-9, rtol=2e-8)


def test_invalid_input_and_empty_rhs():
    codes, y, w = fixture()
    with pytest.raises(ValueError):
        native.NativeSolver(codes, weights=np.zeros(len(w)))
    with pytest.raises(ValueError):
        native.NativeSolver(codes, strategy="compressed", workspace_bytes=1)
    solver = native.NativeSolver(codes)
    with pytest.raises(ValueError):
        solver.solve(y[:-1])
    with pytest.raises((ValueError, RuntimeError)):
        solver.solve(np.full(y.shape, np.nan))
    out, ok, _, _ = solver.solve(np.empty((len(y), 0)))
    assert out.shape == (len(y), 0) and ok == []


def test_numerical_singletons_are_not_statistical_deletions():
    codes = np.array([[0, 0], [0, 1], [1, 1], [2, 2]], dtype=np.uint32)
    y = np.array([[1.], [3.], [7.], [10.]])
    out, ok, _, _ = native.NativeSolver(codes, strategy="compressed").solve(y, tol=1e-11)
    assert all(ok) and len(out) == 4
    np.testing.assert_allclose(out, oracle(codes, y, np.ones(4)), atol=2e-9)


def regression_data():
    import pandas as pd
    rng = np.random.default_rng(197); n = 3600
    a = rng.integers(60, size=n); b = rng.integers(9, size=n)
    x = rng.normal(size=n); z = rng.normal(size=n); u = rng.normal(size=n)
    d = z + 0.4*u + rng.normal(size=n)
    y = 0.7*x + 0.5*d + 0.1*a - 0.2*b + u
    return pd.DataFrame(dict(y=y, x=x, d=d, z=z, a=a, b=b, w=rng.integers(1, 4, size=n)))


@pytest.mark.parametrize("weight_type", [None, "aweight", "fweight"])
def test_public_ols(weight_type):
    from econhdfe import olshdfe, HDFEConfig, ExecutionConfig
    df = regression_data()
    kw = dict(y="y", x=["x", "d"], absorb=["a", "b"], cluster=["a", "b"],
              execution_config=ExecutionConfig(threads=2, memory_budget_mb=512))
    if weight_type:
        kw.update(weights="w", weight_type=weight_type)
    ref = olshdfe(df, method="map", tol=1e-11, **kw)
    out = olshdfe(df, hdfe_config=HDFEConfig(solver="within", tolerance=1e-11), **kw)
    for name in ("nobs", "nobs_raw", "df_absorbed", "dropped_singletons"):
        assert getattr(out, name) == getattr(ref, name)
    np.testing.assert_allclose(out.params, ref.params, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(out.vcov, ref.vcov, atol=2e-9, rtol=2e-6)
    assert out.absorb_info["method"] == "within"


def test_public_iv():
    from econhdfe import ivhdfe, ExecutionConfig
    df = regression_data()
    kw = dict(y="y", exog=["x"], endog=["d"], instruments=["z"], absorb=["a", "b"],
              cluster=["a"], tol=1e-11, execution_config=ExecutionConfig(threads=2, memory_budget_mb=512))
    ref = ivhdfe(df, method="map", **kw); out = ivhdfe(df, method="within", **kw)
    assert out.nobs == ref.nobs and out.df_absorbed == ref.df_absorbed
    np.testing.assert_allclose(out.params, ref.params, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(out.vcov, ref.vcov, atol=2e-9, rtol=2e-6)


def test_weighted_projector_and_plan():
    from econhdfe.hdfe.plan import FEPlan
    codes, y, w = fixture(); plan = FEPlan.from_arrays(list(codes.T))
    projector = plan.projector(engine="optimized", method="within")
    for weights in (w, np.ones(len(w))):
        out, info = projector.residualize(y, weights, tol=1e-11, return_info=True)
        assert info.converged
        np.testing.assert_allclose(out, oracle(codes, y, weights), atol=2e-9, rtol=2e-8)
    assert plan.absorber(w, tol=1e-11, method="within").method == "within"
