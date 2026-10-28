"""Near-collinear FWL against dense dummies, including covariance and state."""
import numpy as np
import pytest
import pandas as pd
from numpy.testing import assert_allclose

from econhdfe import olshdfe, OLSHDFESession, ExecutionConfig, HDFEConfig
from econhdfe.compute.stable_linalg import _absorption_basis
from econhdfe.collinearity import _weighted_gram
from econhdfe.errors import EconHDFEError


@pytest.mark.parametrize("hidden,weighted,pool,threads", [
    (False, False, 1, 1), (False, True, 3, 4),
    (True, False, 3, 1), (True, True, 1, 4),
])
def test_near_collinear_dense_fwl_and_covariance(hidden, weighted, pool, threads):
    rng = np.random.default_rng(8041)
    n = 600
    groups = [rng.integers(0, m, n) for m in (23, 19, 17)]
    D = np.column_stack([np.eye(g.max()+1)[g] for g in groups])
    loading = D @ rng.normal(size=D.shape[1])
    x = rng.normal(size=(n, 4))
    x[:, 1] = x[:, 0] + 1e-5 * x[:, 1] + (loading if hidden else 0)
    y = .4*x[:, 0] + .7*x[:, 2] + loading + rng.normal(size=n)
    w = np.exp(rng.normal(scale=.3, size=n)) if weighted else np.ones(n)
    order = np.array([3, 1, 0, 2]) if weighted else np.arange(4)
    units = np.array([1e-4, 1e3, 1, 1e2])
    x = (x*units)[:, order]
    sw = np.sqrt(w)
    rhs = np.column_stack([y, x])
    exact = rhs - D @ np.linalg.lstsq(D*sw[:, None], rhs*sw[:, None], rcond=None)[0]
    X = exact[:, 1:]*sw[:, None]
    Y = exact[:, 0]*sw
    scale = np.linalg.norm(X, axis=0)
    u, s, vh = np.linalg.svd(X/scale, full_matrices=False)
    inverse = (vh.T/s)/scale[:, None]
    beta = inverse @ (u.T @ Y)
    error = Y-X@beta
    # HC0 computed through the independent orthogonal influence matrix.
    influence = (u @ inverse.T) * error[:, None]
    V0 = influence.T @ influence
    r = olshdfe(None, y=y, x=x, absorb=groups, weights=w if weighted else None,
        weight_type="aweight" if weighted else None, vce="robust", pool_size=pool,
        keep_state=True, execution_config=ExecutionConfig(threads=threads, cache="off"))
    assert r.rank == 4 and r.converged and r.nobs == n
    assert r.absorb_info["conditioning"]["stage"] == ("after_absorption" if hidden else "before_absorption")
    assert_allclose(r.params, beta, rtol=3e-7, atol=2e-5)
    correction = n/(n-r.rank-r.df_absorbed)
    reference = V0*correction
    se = np.sqrt(np.diag(reference))
    assert np.max(np.abs(r.vcov-reference)/(se[:, None]*se[None, :])) < 1e-7
    assert_allclose(r.state.X_within, exact[:, 1:], atol=2e-6, rtol=2e-6)
    assert_allclose(r.state.y_within, exact[:, 0], atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("hidden,weighted", [(False, False), (False, True), (True, False), (True, True)])
def test_conditioned_sessions_batch_prewarm_mutation_and_persistence(hidden, weighted, tmp_path):
    rng = np.random.default_rng(8204)
    n = 800
    df = pd.DataFrame({f"g{i}": rng.integers(m, size=n) for i, m in enumerate((23, 19, 17))})
    groups = ["g0", "g1", "g2"]
    loading = sum(rng.normal(size=df[g].max()+1)[df[g]] for g in groups)
    df["x0"] = rng.normal(size=n)
    df["x1"] = df.x0 + 1e-5*rng.normal(size=n) + (loading if hidden else 0)
    df["x2"] = rng.normal(size=n)
    df["y"] = .7*df.x0 + .4*df.x2 + loading + rng.normal(size=n)
    df["y2"] = df.y + rng.normal(size=n)
    df["w"] = np.exp(rng.normal(scale=.3, size=n))
    options = dict(cluster="g0", weights="w" if weighted else None,
                   weight_type="aweight" if weighted else None,
                   execution_config=ExecutionConfig(threads=1))
    spec = dict(y="y", x=["x0", "x1", "x2"], absorb=groups)
    session = OLSHDFESession(df, **options)
    # Prewarmed column caches stay ordinary even after a conditioned fit.
    session.prewarm(absorb=groups, columns=["y", "y2", "x0", "x1", "x2"])
    entry = next(iter(session._entries.values()))
    before = {k: v.copy() for k, v in entry.within.items()}
    got = session.fit(**spec)
    reference = olshdfe(df, **spec, **options)
    def check(a, b):
        assert a.names == b.names and a.rank == b.rank and a.df_absorbed == b.df_absorbed
        assert_allclose(a.params, b.params, atol=2e-5, rtol=1e-7)
        se = b.stderr
        assert np.max(np.abs(a.vcov-b.vcov)/(se[:, None]*se[None, :])) < 1e-7
        assert_allclose(a.residuals, b.residuals, atol=2e-7, rtol=2e-7)
        assert a.absorb_info["conditioning"]["stage"] == ("after_absorption" if hidden else "before_absorption")
    check(got, reference)
    assert entry.absorber.tol == session.hdfe.tolerance
    for k, v in before.items():
        assert_allclose(entry.within[k], v, atol=0, rtol=0)
    batch = OLSHDFESession(df, **options).fit_many([
        spec, dict(spec, y="y2"), dict(spec, x=["x2"]), dict(spec, x=["x2", "x1", "x0"]),
    ])
    check(batch[0], reference)
    check(batch[1], olshdfe(df, **dict(spec, y="y2"), **options))
    check(batch[3], olshdfe(df, **dict(spec, x=["x2", "x1", "x0"]), **options))
    assert "conditioning" not in batch[2].absorb_info
    df["x1"] += 2e-6*rng.normal(size=n)
    check(session.fit(**spec), olshdfe(df, **spec, **options))
    # A completed persistent result retains the refined numerical result.
    persistent = OLSHDFESession(df, **options).enable_persistent_cache(tmp_path / "cache")
    saved = persistent.fit(**spec)
    restored = OLSHDFESession(df, **options).enable_persistent_cache(tmp_path / "cache")
    check(restored.fit(**spec), saved)
    assert restored.cache_info()["residualize_calls"] == 0


def test_ordinary_and_exactly_singular_design_do_not_whiten():
    rng = np.random.default_rng(39)
    x = rng.normal(size=(300, 4))
    assert _absorption_basis(x) is None
    x[:, 3] = x[:, 0]+x[:, 1]
    assert _absorption_basis(x) is None


@pytest.mark.parametrize("gap", [0, 1e-3, 1e-5])
@pytest.mark.parametrize("weighted", [False, True])
def test_reused_gram_preserves_screening_and_basis(gap, weighted):
    rng = np.random.default_rng(735)
    x = rng.normal(size=(700, 4))
    x[:, 1] = x[:, 0] + gap*x[:, 1]
    x *= [1e-50, 1e60, 1e-4, 1e4]
    w = np.exp(rng.normal(size=len(x))) if weighted else None
    original = _absorption_basis(x, w)
    reused = _absorption_basis(x, w, gram=_weighted_gram(x, w))
    if original is None:
        assert reused is None
    else:
        for a, b in zip(original, reused, strict=True):
            assert_allclose(a, b, atol=0, rtol=0)


def test_session_restores_shared_tolerance_after_failed_conditioned_projection():
    rng = np.random.default_rng(735)
    n = 500
    df = pd.DataFrame({"a": rng.integers(30, size=n), "b": rng.integers(35, size=n),
                       "c": rng.integers(40, size=n), "y": rng.normal(size=n),
                       "x": rng.normal(size=n)})
    df["z"] = df.x + 1e-5*rng.normal(size=n)
    session = OLSHDFESession(df, hdfe_config=HDFEConfig(max_iter=1),
                             execution_config=ExecutionConfig(threads=1))
    with pytest.raises(EconHDFEError):
        session.fit(y="y", x=["x", "z"], absorb=["a", "b", "c"])
    entry = next(iter(session._entries.values()))
    assert entry.absorber.tol == session.hdfe.tolerance
    assert not entry.within
