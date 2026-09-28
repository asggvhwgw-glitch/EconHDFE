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


def test_cluster_diagnostics_balance_intersections_and_scores():
    from econhdfe import cluster_diagnostics
    df = _fixture()
    df["coarse"] = df["cluster"] // 3
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster=["cluster", "coarse"], vce="cluster", keep_state=True,
        drop_singletons=False,
    )
    d = cluster_diagnostics(r)
    assert d.cluster_counts == (18, 6)
    assert d.pairwise_intersections[(0, 1)] == 18
    assert d.dimensions[0].min_size == d.dimensions[0].max_size == 40
    assert d.dimensions[0].effective_clusters_size == pytest.approx(18.0)
    assert d.dimensions[0].score_share_max is not None
    assert d.dimensions[0].score_effective_clusters is not None
    assert "few_clusters" in d.flags


def test_cluster_diagnostics_requires_state_and_clusters():
    from econhdfe import cluster_diagnostics
    df = _fixture(n=360)
    no_state = olshdfe(df, y="y", x=["x1"], absorb=["firm", "year"], cluster="cluster", vce="cluster")
    with pytest.raises(Exception) as exc:
        cluster_diagnostics(no_state)
    assert getattr(exc.value, "code", None) == "inference.missing_state"
    no_cluster = olshdfe(df, y="y", x=["x1"], absorb=["firm", "year"], keep_state=True)
    with pytest.raises(Exception) as exc2:
        cluster_diagnostics(no_cluster)
    assert getattr(exc2.value, "code", None) == "inference.cluster_required"


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


def test_wcr11_observed_stat_matches_crv1_t_and_parallel_reproducible():
    from econhdfe import wild_cluster_test_ols
    df = _fixture(seed=6220, n=720)
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", keep_state=True,
        drop_singletons=False, method="map", tol=1e-10,
    )
    a = wild_cluster_test_ols(r, param="x1", reps=199, seed=77, batch_size=17, n_jobs=1)
    b = wild_cluster_test_ols(r, param="x1", reps=199, seed=77, batch_size=17, n_jobs=2)
    idx = r.names.index("x1")
    assert a.statistic == pytest.approx(r.params[idx] / r.stderr[idx], rel=2e-9, abs=2e-9)
    assert a.bootstrap_type == "WCR11"
    np.testing.assert_allclose(a.bootstrap_statistics, b.bootstrap_statistics, rtol=0, atol=0)
    assert a.pvalue == b.pvalue
    assert 0.0 < a.pvalue <= 1.0


def test_wild_cluster_supports_wcu_webb_mammen_and_joint_restrictions():
    from econhdfe import wild_cluster_test_ols
    df = _fixture(seed=6221, n=540)
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", keep_state=True, drop_singletons=False,
    )
    for dist in ("mammen", "webb"):
        out = wild_cluster_test_ols(r, param="x2", reps=149, seed=81, weight_distribution=dist)
        assert out.completed_reps == 149
        assert np.isfinite(out.bootstrap_statistics).all()
        assert 0.0 < out.pvalue <= 1.0
    joint = wild_cluster_test_ols(r, R=np.eye(2), r=np.zeros(2), reps=149, seed=82)
    assert joint.statistic >= 0
    assert joint.restriction_matrix.shape == (2, 2)
    wcr = wild_cluster_test_ols(r, param="x1", reps=149, seed=83, impose_null=True)
    wcu = wild_cluster_test_ols(r, param="x1", reps=149, seed=83, impose_null=False)
    assert wcr.bootstrap_type == "WCR11" and wcu.bootstrap_type == "WCU11"
    assert not np.allclose(wcr.bootstrap_statistics, wcu.bootstrap_statistics)


def test_wcr11_full_rademacher_enumeration_for_very_few_clusters():
    from econhdfe import wild_cluster_test_ols
    rng = np.random.default_rng(6222)
    n = 320
    df = pd.DataFrame({"fe": np.arange(n) % 20, "cluster": np.arange(n) % 8, "x": rng.normal(size=n)})
    df["y"] = 0.15 * df.x + rng.normal(size=20)[df.fe] + rng.normal(size=n)
    r = olshdfe(
        df, y="y", x=["x"], absorb=["fe"], cluster="cluster",
        vce="cluster", keep_state=True, drop_singletons=False,
    )
    out = wild_cluster_test_ols(r, param="x", reps=999, seed=1, batch_size=31)
    out2 = wild_cluster_test_ols(r, param="x", reps=999, seed=999, batch_size=19)
    assert out.full_enumeration and out.reps == 256 and out.requested_reps == 999
    assert out2.pvalue == out.pvalue
    np.testing.assert_allclose(np.sort(out2.bootstrap_statistics), np.sort(out.bootstrap_statistics), rtol=0, atol=5e-15)


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


@pytest.mark.parametrize("weight_type", ["aweight", "pweight", "fweight"])
def test_weighted_wcr_observed_stat_matches_fitted_crv1(weight_type):
    from econhdfe import wild_cluster_test_ols
    df = _fixture(seed=6230, n=540)
    if weight_type == "fweight":
        df["w"] = 1 + (np.arange(len(df)) % 3)
    else:
        df["w"] = 0.5 + (np.arange(len(df)) % 7) / 5.0
    r = olshdfe(
        df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
        cluster="cluster", vce="cluster", weights="w", weight_type=weight_type,
        keep_state=True, drop_singletons=False,
    )
    out = wild_cluster_test_ols(r, param="x1", reps=99, seed=91, batch_size=11)
    idx = r.names.index("x1")
    assert out.statistic == pytest.approx(r.params[idx] / r.stderr[idx], rel=5e-9, abs=5e-9)
    assert np.isfinite(out.bootstrap_statistics).all()


def test_crv3_not_promoted_to_public_api():
    import econhdfe
    assert not hasattr(econhdfe, "cluster_jackknife_ols")
