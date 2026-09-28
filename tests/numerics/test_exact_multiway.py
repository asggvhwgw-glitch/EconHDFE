import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from econhdfe import HDFEConfig, PPMLConfig, categorical_rank, ivreghdfe, ppmlhdfe, reghdfe
from econhdfe.hdfe.dof import absorbed_dof
from econhdfe.hdfe.rank import available_rank_backends, categorical_prefix_ranks


def _dense_dummy(groups):
    n = len(groups[0]) if groups else 0
    blocks = []
    for raw in groups:
        _, g = np.unique(np.asarray(raw), return_inverse=True)
        L = int(g.max()) + 1 if len(g) else 0
        D = np.zeros((n, L), dtype=np.int64)
        if n:
            D[np.arange(n), g] = 1
        blocks.append(D)
    return np.column_stack(blocks) if blocks else np.empty((n, 0), dtype=np.int64)


def _fraction_rank(A):
    A = [[Fraction(int(v)) for v in row] for row in np.asarray(A)]
    if not A:
        return 0
    m, n = len(A), len(A[0])
    r = 0
    for c in range(n):
        pivot = next((i for i in range(r, m) if A[i][c]), None)
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        p = A[r][c]
        A[r] = [v / p for v in A[r]]
        for i in range(m):
            if i == r or not A[i][c]:
                continue
            a = A[i][c]
            A[i] = [x - a * y for x, y in zip(A[i], A[r], strict=False)]
        r += 1
        if r == m:
            break
    return r


def _counterexample_groups(repeat=1):
    edges = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=np.int32)
    edges = np.repeat(edges, repeat, axis=0)
    return [edges[:, j] for j in range(3)]


def test_exact_dof_finds_collective_three_way_dependency_missed_by_pairwise():
    groups = _counterexample_groups()
    pair = absorbed_dof(groups, method="pairwise", adjust_nested=False)
    exact = absorbed_dof(groups, method="exact", adjust_nested=False)
    assert pair.df_absorbed == 5
    assert exact.df_absorbed == 4
    assert exact.m_by_component == (0, 1, 2)
    assert exact.exact_by_component == (True, True, True)


def test_exact_rank_exhaustive_two_by_two_by_two_hypergraphs():
    edges = np.array(list(itertools.product(range(2), repeat=3)), dtype=np.int32)
    for mask in range(1, 1 << len(edges)):
        take = np.fromiter((bool(mask & (1 << i)) for i in range(len(edges))), bool)
        e = edges[take]
        groups = [e[:, j] for j in range(3)]
        ref = _fraction_rank(_dense_dummy(groups))
        assert categorical_rank(groups, backend="native") == ref
        assert categorical_rank(groups, backend="auto") == ref


def test_exact_rank_random_three_and_four_way_designs():
    rng = np.random.default_rng(9301)
    for G in (3, 4):
        for _ in range(60):
            n = int(rng.integers(3, 20))
            groups = [rng.integers(0, int(rng.integers(1, 6)), n) for _ in range(G)]
            ref = _fraction_rank(_dense_dummy(groups))
            assert categorical_rank(groups, backend="native") == ref


def test_sympy_backend_if_available_and_prefix_ranks():
    groups = _counterexample_groups(repeat=2)
    expected = tuple(_fraction_rank(_dense_dummy(groups[:j])) for j in range(1, 4))
    assert categorical_prefix_ranks(groups, backend="native") == expected
    if "sympy" in available_rank_backends():
        assert categorical_prefix_ranks(groups, backend="sympy") == expected


def _nested_cluster_fixture(seed=9302):
    rng = np.random.default_rng(seed)
    city = np.repeat(np.arange(8), 5 * 6)
    year = np.tile(np.repeat(np.arange(6), 5), 8)
    n = len(city)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    endog = 0.8 * z + 0.3 * x + v
    cy = city * 6 + year
    cell = rng.normal(size=48)
    y = 0.65 * x + 1.4 * endog + cell[cy] + 0.4 * v + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x": x, "endog": endog, "z": z, "city": city, "year": year})


def test_solver_canonicalization_does_not_change_ols_requested_fe_inference():
    df = _nested_cluster_fixture()
    base = dict(
        data=df, y="y", x=["x"], absorb=["year", ("city", "year")],
        cluster="city", vce="cluster", drop_singletons=False, tol=1e-10,
        dof_method="exact",
    )
    a = reghdfe(**base, canonicalize_fe=True)
    b = reghdfe(**base, canonicalize_fe=False)
    assert a.df_absorbed == b.df_absorbed
    assert a.dof_info.nested == b.dof_info.nested
    np.testing.assert_allclose(a.params, b.params, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=2e-9, rtol=2e-9)


def test_solver_canonicalization_does_not_change_iv_requested_fe_inference():
    df = _nested_cluster_fixture(seed=9303)
    base = dict(
        data=df, y="y", exog=["x"], endog=["endog"], instruments=["z"],
        absorb=["year", ("city", "year")], cluster="city", vce="cluster",
        drop_singletons=False, tol=1e-10, dof_method="exact",
    )
    a = ivreghdfe(**base, canonicalize_fe=True)
    b = ivreghdfe(**base, canonicalize_fe=False)
    assert a.df_absorbed == b.df_absorbed
    assert a.dof_info.nested == b.dof_info.nested
    np.testing.assert_allclose(a.params, b.params, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=2e-9, rtol=2e-9)


def test_ppml_can_request_exact_three_way_dof():
    groups = _counterexample_groups(repeat=3)
    n = len(groups[0])
    y = np.linspace(1.0, 3.0, n)
    X = np.linspace(-1.0, 1.0, n)[:, None]
    pair = ppmlhdfe(y, X, absorb=groups, names=("x",), config=PPMLConfig(separation=("fe",), dof_method="pairwise"))
    exact = ppmlhdfe(y, X, absorb=groups, names=("x",), config=PPMLConfig(separation=("fe",), dof_method="exact"))
    assert pair.df_absorbed == 5
    assert exact.df_absorbed == 4
    np.testing.assert_allclose(pair.coef, exact.coef, atol=1e-9, rtol=1e-9)


def test_hdfe_config_accepts_exact():
    cfg = HDFEConfig(dof_method="exact")
    cfg.validate()


def test_ivppml_can_request_exact_three_way_dof():
    from econhdfe import IVPPMLConfig, ivppmlhdfe
    rng = np.random.default_rng(9304)
    base = np.column_stack(_counterexample_groups(repeat=40))
    n = len(base)
    g1, g2, g3 = (base[:, j] for j in range(3))
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    e = 0.8 * z + 0.2 * c + rng.normal(scale=0.5, size=n)
    a1 = np.array([0.1, -0.1])
    a2 = np.array([0.08, -0.04])
    a3 = np.array([0.03, -0.02, 0.01])
    y = rng.poisson(np.exp(a1[g1] + a2[g2] + a3[g3] + 0.1*c + 0.25*e))
    common = dict(
        y=y, exog=c, endog=e, instruments=z, absorb=[g1, g2, g3], vce="robust",
        exog_names=["c"], endog_names=["e"], instrument_names=["z"],
    )
    base_cfg = dict(separation=(), standardize=False, tolerance=1e-9, target_inner_tol=1e-10, max_iter=300)
    pair = ivppmlhdfe(**common, config=IVPPMLConfig(**base_cfg, dof_method="pairwise"))
    exact = ivppmlhdfe(**common, config=IVPPMLConfig(**base_cfg, dof_method="exact"))
    assert pair.df_absorbed == 5
    assert exact.df_absorbed == 4
    np.testing.assert_allclose(pair.coef, exact.coef, atol=2e-8, rtol=2e-8)
