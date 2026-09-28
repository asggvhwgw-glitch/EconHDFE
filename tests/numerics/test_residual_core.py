import numpy as np

from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.numerical_core import build_numerical_core
from econhdfe.hdfe.plan import FEPlan


def _leaf_fixture(seed=1201, n=4000, k=4):
    rng = np.random.default_rng(seed)
    core = n // 3
    leaf = n - core
    g0 = np.empty(n, dtype=np.int32)
    g0[:leaf] = np.arange(leaf, dtype=np.int32)
    g0[leaf:] = leaf + rng.integers(0, max(10, core // 20), core, dtype=np.int32)
    groups = [
        g0,
        rng.integers(0, max(20, n // 20), n, dtype=np.int32),
        rng.integers(0, 40, n, dtype=np.int32),
    ]
    X = rng.normal(size=(n, k))
    return groups, X


def test_numerical_core_matches_full_projection_unweighted():
    groups, X = _leaf_fixture()
    full = HDFEAbsorber(
        groups, tol=1e-10, acceleration="cg", core_reduction="off",
        projection_backend="fused",
    ).residualize(X)
    fast_abs = HDFEAbsorber(
        groups, tol=1e-10, acceleration="auto", core_reduction="auto",
        projection_backend="fused",
    )
    fast = fast_abs.residualize(X)
    assert fast_abs._core_plan is not None
    assert fast_abs._core_plan.peeled > 0
    np.testing.assert_allclose(fast, full, atol=2e-8, rtol=2e-8)


def test_numerical_core_matches_dense_weighted_projection_oracle():
    groups, X = _leaf_fixture(seed=1202, n=240, k=3)
    rng = np.random.default_rng(1203)
    w = np.exp(rng.normal(scale=.4, size=len(X)))
    fast_abs = HDFEAbsorber(
        groups, weights=w, tol=1e-11, acceleration="auto", core_reduction="auto",
        projection_backend="fused", max_iter=20000,
    )
    fast = fast_abs.residualize(X)
    assert fast_abs._core_plan is not None

    blocks = []
    for g in groups:
        L = int(np.max(g)) + 1
        D = np.zeros((len(g), L), dtype=np.float64)
        D[np.arange(len(g)), g] = 1.0
        blocks.append(D)
    D = np.column_stack(blocks)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(sw[:, None] * D, sw[:, None] * X, rcond=None)
    oracle = X - D @ coef
    np.testing.assert_allclose(fast, oracle, atol=2e-10, rtol=2e-10)


def test_core_plan_peeling_rows_are_exactly_zero_residuals():
    groups, X = _leaf_fixture(seed=1204)
    plan = build_numerical_core(groups)
    out = HDFEAbsorber(groups, tol=1e-10, core_reduction="auto").residualize(X)
    peeled = np.ones(len(X), dtype=bool)
    peeled[plan.core_index] = False
    assert np.max(np.abs(out[peeled])) == 0.0


def test_auto_acceleration_uses_plain_for_easy_and_cg_for_hard_cycle():
    rng = np.random.default_rng(1205)
    n = 6000
    easy = [rng.integers(0, 100, n), rng.integers(0, 80, n), rng.integers(0, 30, n)]
    X = rng.normal(size=(n, 2))
    a = HDFEAbsorber(easy, tol=1e-8, acceleration="auto", core_reduction="off")
    _, info = a.residualize(X, return_info=True)
    assert info.converged
    assert info.criterion == "auto_plain"

    m = 300
    i = np.arange(m, dtype=np.int32)
    edges = np.vstack([
        np.column_stack([i, i, i]),
        np.column_stack([(i + 1) % m, i, i]),
    ])
    hard = [edges[:, j] for j in range(3)]
    Xh = rng.normal(size=(len(edges), 2))
    h = HDFEAbsorber(hard, tol=1e-8, acceleration="auto", core_reduction="off", max_iter=5000)
    out, info = h.residualize(Xh, return_info=True)
    assert info.converged
    assert info.criterion == "auto_cg"
    assert h.fe_orthogonality_error(out) < 1e-7


def test_feplan_generic_multiway_uses_auto_not_unconditional_cg():
    groups, X = _leaf_fixture(seed=1206)
    plan = FEPlan.from_arrays(groups)
    absorber = plan.absorber(np.ones(len(X)), tol=1e-8, engine="optimized")
    assert absorber.acceleration == "auto"


def test_many_rhs_fused_dispatch_matches_indexed_projection():
    rng = np.random.default_rng(1207)
    n = 12000
    groups = [rng.integers(0, 180, n), rng.integers(0, 100, n), rng.integers(0, 40, n)]
    X = rng.normal(size=(n, 12))
    w = np.exp(rng.normal(scale=.2, size=n))
    indexed = HDFEAbsorber(
        groups, weights=w, tol=1e-9, acceleration="none", core_reduction="off",
        projection_backend="indexed", absorb_threads=2,
    ).residualize(X)
    fused = HDFEAbsorber(
        groups, weights=w, tol=1e-9, acceleration="none", core_reduction="off",
        projection_backend="fused", absorb_threads=2,
    ).residualize(X)
    np.testing.assert_allclose(fused, indexed, atol=2e-9, rtol=2e-9)


def test_core_topology_reused_across_positive_weight_updates():
    groups, X = _leaf_fixture(seed=1208, n=3000, k=3)
    rng = np.random.default_rng(1209)
    w1 = np.exp(rng.normal(scale=.2, size=len(X)))
    w2 = np.exp(rng.normal(scale=.3, size=len(X)))
    a = HDFEAbsorber(groups, weights=w1, tol=1e-9, acceleration="auto", core_reduction="auto")
    _ = a.residualize(X)
    plan_id = id(a._core_plan)
    a.update_weights(w2)
    got = a.residualize(X)
    assert id(a._core_plan) == plan_id
    ref = HDFEAbsorber(
        groups, weights=w2, tol=1e-9, acceleration="auto", core_reduction="off"
    ).residualize(X)
    np.testing.assert_allclose(got, ref, atol=2e-7, rtol=2e-7)


def test_zero_weight_update_disables_structural_core_fast_path():
    groups, X = _leaf_fixture(seed=1210, n=2000, k=2)
    w = np.ones(len(X))
    a = HDFEAbsorber(groups, weights=w, tol=1e-8, core_reduction="auto")
    assert a._core_plan is not None
    w[0] = 0.0
    a.update_weights(w)
    assert a._core_plan is None
    out = a.residualize(X)
    assert np.all(np.isfinite(out))


def test_core_residualize_reuses_inplace_buffer_and_preserves_copy_semantics():
    groups, X = _leaf_fixture(n=20_000, k=3)
    absorber = HDFEAbsorber(groups, tol=1e-10, acceleration="auto", core_reduction="on")
    assert absorber._core_plan is not None

    original = X.copy()
    work = X.copy()
    out = absorber.residualize(work, copy=False)
    assert np.shares_memory(out, work)
    assert np.max(np.abs(work - original)) > 0

    source = original.copy()
    copied = absorber.residualize(source, copy=True)
    assert not np.shares_memory(copied, source)
    np.testing.assert_allclose(source, original, rtol=0, atol=0)
    np.testing.assert_allclose(copied, out, rtol=1e-10, atol=1e-10)
