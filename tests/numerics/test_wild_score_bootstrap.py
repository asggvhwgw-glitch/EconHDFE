"""Independent FE projection oracles for coefficient-only wild draws."""
import math

import numpy as np
import pytest

from econhdfe import ExecutionConfig
from econhdfe.hdfe.absorber import HDFEAbsorber, HDFEConvergenceError
from econhdfe.models.ols_bootstrap import wild_cluster_bootstrap_ols
from econhdfe.resampling.sampling import wild_weights


def _fixture(weighted, seed=825):
    rng = np.random.default_rng(seed)
    n = 330
    levels = (23, 11, 7)
    groups = [rng.integers(L, size=n, dtype=np.int32) for L in levels]
    D = np.column_stack([np.eye(L)[g] for L, g in zip(levels, groups)])
    X = rng.normal(size=(n, 3)) + D @ rng.normal(size=(sum(levels), 3))
    u = rng.normal(size=n)
    w = (rng.integers(1, 5, size=n).astype(float) if weighted == "frequency"
         else np.exp(rng.normal(scale=.5, size=n)) if weighted else None)
    # First appearances are ordered, matching the public cluster normalizer.
    cluster = np.arange(n, dtype=np.int32) % 15
    return D, X, u, w, cluster, groups


def _dense_draws(D, X, beta, u, cluster, w, *, projection_weights=None,
                 reps=17, batch_size=5, seed=197):
    n = len(X)
    sw = np.ones(n) if w is None else np.sqrt(w)
    sp = sw if projection_weights is None else np.sqrt(projection_weights)
    Xw = X * sw[:, None]
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)
    # One independent dense dummy SVD; no econhdfe projector is used.
    U, s, _ = np.linalg.svd(D * sp[:, None], full_matrices=False)
    U = U[:, s > np.finfo(float).eps * max(D.shape) * s[0]]
    G = len(X) if cluster is None else int(cluster.max()) + 1
    codes = np.arange(n) if cluster is None else cluster
    out = []
    seeds = np.random.SeedSequence(seed).spawn(math.ceil(reps / batch_size))
    for start, ss in zip(range(0, reps, batch_size), seeds):
        b = min(batch_size, reps-start)
        v = wild_weights(np.random.default_rng(ss), (G, b))
        ys = (X @ beta)[:, None] + u[:, None] * v[codes]
        yp = ys - (U @ (U.T @ (ys * sp[:, None]))) / sp[:, None]
        out.append((bread @ (Xw.T @ (yp * sw[:, None]))).T)
    return np.vstack(out)


def _draws(absorber, X, beta, u, cluster, w, **kwargs):
    return wild_cluster_bootstrap_ols(
        absorber=absorber, y_within=np.zeros(len(X)), X_within=X,
        beta=beta, residuals=u, clusters=cluster, weights=w,
        reps=17, batch_size=5, seed=197, n_jobs=1,
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=128),
        **kwargs,
    )


@pytest.mark.parametrize("weighted", [False, True, "frequency"])
def test_score_draws_match_dense_dummy_with_raw_design_and_units(weighted):
    D, X, u, w, cluster, groups = _fixture(weighted)
    absorber = HDFEAbsorber(groups, weights=w, tol=1e-13, absorb_threads=1)
    beta = np.array([.7, -.4, .2])
    before = X.copy()
    expected = _dense_draws(D, X, beta, u, cluster, w)
    got = _draws(absorber, X, beta, u, cluster, w)
    np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(X, before)
    order = np.array([2, 0, 1])
    scales = np.array([.2, 4., 2.])
    changed = _draws(absorber, X[:, order]*scales, beta[order]/scales,
                     u, cluster, w)
    np.testing.assert_allclose(changed*scales, got[:, order], rtol=2e-10, atol=2e-11)


def test_rank_deficient_nonwithin_draws_keep_general_pinv_center():
    D, X, u, w, cluster, groups = _fixture(True)
    X = np.column_stack([X[:, 0], X[:, 0], X[:, 2]])
    beta = np.array([1., 0., -.5])  # Not in the minimum-norm coefficient range.
    absorber = HDFEAbsorber(groups, weights=w, tol=1e-13, absorb_threads=1)
    expected = _dense_draws(D, X, beta, u, cluster, w)
    got = _draws(absorber, X, beta, u, cluster, w)
    np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(got[:, 0], got[:, 1], atol=2e-12)
    assert np.linalg.norm(got.mean(axis=0)-beta) > .5


def test_unclustered_near_absorbed_design_matches_dense_projection():
    D, X, u, w, _, groups = _fixture(True)
    U, s, _ = np.linalg.svd(D*np.sqrt(w[:, None]), full_matrices=False)
    U = U[:, s > np.finfo(float).eps*max(D.shape)*s[0]]
    within = X-(U@(U.T@(X*np.sqrt(w[:, None]))))/np.sqrt(w[:, None])
    X = (X-within) + 1e-5*within
    absorber = HDFEAbsorber(groups, weights=w, tol=1e-13, absorb_threads=1)
    beta = np.array([.7, -.4, .2])
    expected = _dense_draws(D, X, beta, u, None, w)
    got = _draws(absorber, X, beta, u, None, w)
    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=2e-11)


def test_mismatched_absorber_metric_keeps_original_draws():
    D, X, u, w, cluster, groups = _fixture(True)
    projection_w = np.linspace(.3, 2., len(X))
    absorber = HDFEAbsorber(groups, weights=projection_w, tol=1e-13, absorb_threads=1)
    beta = np.array([.7, -.4, .2])
    expected = _dense_draws(D, X, beta, u, cluster, w, projection_weights=projection_w)
    got = _draws(absorber, X, beta, u, cluster, w)
    np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-11)


def test_failed_design_projection_does_not_return_bootstrap_draws():
    _, X, u, w, cluster, groups = _fixture(True)
    absorber = HDFEAbsorber(groups, weights=w, tol=1e-14, max_iter=1,
                            acceleration="none", absorb_threads=1)
    with pytest.raises(HDFEConvergenceError) as raised:
        _draws(absorber, X, np.ones(3), u, cluster, w)
    assert raised.value.code == "convergence.hdfe"


def test_empty_design_keeps_empty_coefficient_draws():
    _, X, u, w, cluster, groups = _fixture(False)
    absorber = HDFEAbsorber(groups, tol=1e-13, absorb_threads=1)
    got = _draws(absorber, X[:, :0], np.empty(0), u, cluster, w)
    assert got.shape == (17, 0)


def test_float32_projector_keeps_legacy_rounding():
    _, X, u, _, cluster, groups = _fixture(False)
    absorber = HDFEAbsorber(groups[:1], dtype=np.float32, absorb_threads=1)
    beta = np.array([1e8, -1e8, .2])
    bread = np.linalg.pinv(X.T @ X, hermitian=True)
    expected = []
    for start, ss in zip(range(0, 17, 5), np.random.SeedSequence(197).spawn(4)):
        b = min(5, 17-start)
        v = wild_weights(np.random.default_rng(ss), (15, b))
        ys = (X @ beta)[:, None] + u[:, None]*v[cluster]
        ys = absorber.residualize(ys, absorb_threads=1)
        expected.append((bread @ (X.T @ ys)).T)
    got = _draws(absorber, X, beta, u, cluster, None)
    np.testing.assert_allclose(got, np.vstack(expected), rtol=0, atol=1e-7)


def test_near_collinear_design_keeps_legacy_coefficient_arithmetic():
    _, X, u, w, cluster, groups = _fixture(True)
    absorber = HDFEAbsorber(groups, tol=1e-13, weights=w, absorb_threads=1)
    X = absorber.residualize(X)
    X = np.column_stack([X[:, 0], X[:, 0] + 1e-6*X[:, 1]])
    beta = np.array([.7, -.4])
    u *= 1e-8
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)
    expected = []
    for start, ss in zip(range(0, 17, 5), np.random.SeedSequence(197).spawn(4)):
        b = min(5, 17-start)
        v = wild_weights(np.random.default_rng(ss), (15, b))
        ys = (X @ beta)[:, None] + u[:, None]*v[cluster]
        ys = absorber.residualize(ys, absorb_threads=1)
        expected.append((bread @ (Xw.T @ (ys*sw[:, None]))).T)
    got = _draws(absorber, X, beta, u, cluster, w)
    np.testing.assert_array_equal(got, np.vstack(expected))


def test_native_score_draws_match_dense_projection_when_available():
    pytest.importorskip("_econhdfe_within")
    from econhdfe.hdfe.within_backend import WithinFEAbsorber
    D, X, u, w, cluster, groups = _fixture(True)
    absorber = WithinFEAbsorber(groups, weights=w, tol=1e-12, absorb_threads=1)
    beta = np.array([.7, -.4, .2])
    for design in (X, absorber.residualize(X)):
        expected = _dense_draws(D, design, beta, u, cluster, w)
        got = _draws(absorber, design, beta, u, cluster, w)
        np.testing.assert_allclose(got, expected, rtol=2e-10, atol=2e-11)
