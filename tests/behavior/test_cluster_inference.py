from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from econhdfe import olshdfe


def _fixture(seed=6201, n=720):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "firm": np.arange(n) % 60,
        "year": (np.arange(n) // 60) % 12,
        "cluster": np.arange(n) % 18,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    af = rng.normal(scale=0.7, size=60)
    at = rng.normal(scale=0.3, size=12)
    shock = rng.normal(scale=0.5, size=18)
    df["y"] = (
        0.8 * df.x1.to_numpy() - 0.35 * df.x2.to_numpy()
        + af[df.firm.to_numpy()] + at[df.year.to_numpy()]
        + shock[df.cluster.to_numpy()] + rng.normal(scale=0.6, size=n)
    )
    return df


def test_weighted_legacy_wild_bootstrap_matches_manual_wls_metric():
    from econhdfe import wild_bootstrap
    df = _fixture(seed=6210, n=420)
    df["w"] = 0.5 + (np.arange(len(df)) % 7) / 5.0
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", weights="w", weight_type="aweight",
        keep_state=True, drop_singletons=False,
    )
    got = wild_bootstrap(r, reps=11, seed=99, n_jobs=1, batch_size=11)
    state = r.state
    X = np.asarray(state.X_within, dtype=float)
    y = np.asarray(state.y_within, dtype=float)
    u = np.asarray(r.residuals, dtype=float)
    w = np.asarray(state.weights, dtype=float)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)
    cluster = np.asarray(state.clusters[0])
    G = int(cluster.max()) + 1
    rng = np.random.default_rng(np.random.SeedSequence(99).spawn(1)[0])
    vg = rng.integers(0, 2, size=(G, 11), dtype=np.int8).astype(float) * 2 - 1
    ystar = (X @ r.params)[:, None] + u[:, None] * vg[cluster]
    ystar = state.absorber.residualize(ystar, absorb_threads=getattr(state.absorber, "absorb_threads", 1))
    expected = (bread @ (Xw.T @ (ystar * sw[:, None]))).T
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-12)


def test_weighted_wild_bootstrap_parallel_reproducibility_and_distributions():
    from econhdfe import wild_bootstrap
    df = _fixture(seed=6211, n=540)
    df["w"] = 0.5 + (np.arange(len(df)) % 7) / 5.0
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", weights="w", weight_type="aweight",
        keep_state=True, drop_singletons=False,
    )
    a = wild_bootstrap(r, reps=23, seed=99, n_jobs=1, batch_size=5)
    b = wild_bootstrap(r, reps=23, seed=99, n_jobs=2, batch_size=5)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    for dist in ("mammen", "webb", "normal"):
        out = wild_bootstrap(r, reps=11, seed=91, n_jobs=1, batch_size=4, weight_distribution=dist)
        assert out.shape == (11, 2)
        assert np.isfinite(out).all()


def test_wcr_rejects_multiway_cluster_and_non_ols():
    from econhdfe import wild_cluster_test_ols
    df = _fixture(seed=6223, n=540)
    df["coarse"] = df["cluster"] // 3
    r = olshdfe(
        df, y="y", x=["x1"], absorb=["firm", "year"],
        cluster=["cluster", "coarse"], vce="cluster", keep_state=True,
        drop_singletons=False,
    )
    with pytest.raises(Exception) as exc:
        wild_cluster_test_ols(r, param="x1", reps=99)
    assert getattr(exc.value, "code", None) == "resampling.multiway_cluster"


def test_wcr11_enumeration_matches_explicit_intercept_ols():
    from econhdfe import wild_cluster_test_ols
    rng = np.random.default_rng(6224)
    G, m = 7, 24
    n = G * m
    cluster = np.repeat(np.arange(G), m)
    x = rng.normal(size=n)
    y = 0.25 * x + rng.normal(scale=0.6, size=G)[cluster] + rng.normal(size=n)
    df = pd.DataFrame({"intercept_fe": 0, "cluster": cluster, "x": x, "y": y})
    fit = olshdfe(
        df, y="y", x=["x"], absorb=["intercept_fe"], cluster="cluster",
        vce="cluster", keep_state=True, drop_singletons=False, tol=1e-12,
    )
    got = wild_cluster_test_ols(fit, param="x", r=0.0, reps=999, batch_size=13)
    X = np.column_stack([np.ones(n), x])
    XXi = np.linalg.inv(X.T @ X)
    beta = XXi @ X.T @ y
    beta_r = np.array([y.mean(), 0.0])
    ur = y - X @ beta_r
    scale = ((n - 1) / (n - 2)) * (G / (G - 1))
    direct = []
    for bits in range(2**G):
        v = np.array([1.0 if (bits >> g) & 1 else -1.0 for g in range(G)])
        ys = X @ beta + ur * v[cluster]
        bs = XXi @ X.T @ ys
        es = ys - X @ bs
        score = np.zeros((G, 2))
        for g in range(G):
            ix = cluster == g
            score[g] = X[ix].T @ es[ix]
        V = scale * XXi @ (score.T @ score) @ XXi
        direct.append((bs[1] - beta[1]) / np.sqrt(V[1, 1]))
    np.testing.assert_allclose(np.sort(got.bootstrap_statistics), np.sort(np.asarray(direct)), rtol=3e-9, atol=3e-9)
