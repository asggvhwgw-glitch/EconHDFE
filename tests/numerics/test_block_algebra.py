import numpy as np
import pytest

from econhdfe.compute.block_design import BlockDesign, compile_block_design
from econhdfe.compute.design_ops import column_moments, gram_matrix
from econhdfe.compute.design_plan import analyze_design_structure
from econhdfe.compute.linalg import independent_columns
from econhdfe.models.ppml.standardize import Standardization


def _block_fixture(seed=1301, blocks=5, rows_per=90, width=4):
    rng = np.random.default_rng(seed)
    n = blocks * rows_per
    k = blocks * width
    X = np.zeros((n, k), dtype=np.float64)
    block = np.repeat(np.arange(blocks), rows_per)
    for b in range(blocks):
        ix = block == b
        X[ix, b * width:(b + 1) * width] = rng.normal(size=(ix.sum(), width))
    plan = analyze_design_structure(X)
    assert plan.certified_block_separable
    return rng, X, plan


def test_block_column_moments_equal_dense_full_sample_semantics():
    _, X, plan = _block_fixture()
    dense = column_moments(X)
    blocked = column_moments(X, structure=plan)
    np.testing.assert_allclose(blocked.sum, dense.sum, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(blocked.sum_squares, dense.sum_squares, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(blocked.sample_scale(), np.std(X, axis=0, ddof=1), rtol=3e-14, atol=3e-14)


def test_block_gram_equal_dense_weighted_and_unweighted():
    rng, X, plan = _block_fixture(seed=1302)
    w = np.exp(rng.normal(scale=0.2, size=len(X)))
    np.testing.assert_allclose(gram_matrix(X, structure=plan), X.T @ X, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        gram_matrix(X, weights=w, structure=plan),
        X.T @ (X * w[:, None]), rtol=2e-13, atol=2e-13,
    )


def test_block_design_exact_operator_parity_and_payload_reduction():
    rng, X, plan = _block_fixture(seed=1303, blocks=6, rows_per=70, width=3)
    B = BlockDesign.from_dense(X, plan)
    np.testing.assert_array_equal(B.materialize(), X)
    beta = rng.normal(size=X.shape[1])
    v = rng.normal(size=X.shape[0])
    w = np.exp(rng.normal(scale=0.1, size=X.shape[0]))
    np.testing.assert_allclose(B.matvec(beta), X @ beta, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(B.t_matvec(v), X.T @ v, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(B.gram(), X.T @ X, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(B.gram(weights=w), X.T @ (X * w[:, None]), rtol=2e-13, atol=2e-13)
    assert B.payload_bytes == plan.block_dense_bytes
    assert B.payload_bytes == X.nbytes // 6
    assert B.nbytes < X.nbytes // 4  # row-index metadata is small relative to zero payload saved.


def test_standardization_can_consume_plan_or_block_without_semantic_change():
    rng, X, plan = _block_fixture(seed=1304, blocks=4, rows_per=100, width=5)
    y = np.exp(rng.normal(scale=0.4, size=len(X)))
    dense = Standardization.fit(y, X)
    planned = Standardization.fit(y, X, structure=plan)
    B = BlockDesign.from_dense(X, plan)
    native = Standardization.fit_block(y, B)
    np.testing.assert_allclose(planned.x_scale, dense.x_scale, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(native.x_scale, dense.x_scale, rtol=3e-14, atol=3e-14)
    assert planned.y_scale == pytest.approx(dense.y_scale, rel=1e-15)
    ys_dense, Xs_dense = dense.transform(y, X)
    ys_block, Xs_block = native.transform_block(y, B)
    np.testing.assert_allclose(ys_block, ys_dense, rtol=0, atol=0)
    np.testing.assert_allclose(Xs_block.materialize(), Xs_dense, rtol=2e-14, atol=2e-14)


def test_rank_certificate_can_use_structure_without_changing_fallback_semantics():
    rng, X, plan = _block_fixture(seed=1305, blocks=4, rows_per=130, width=4)
    dense = independent_columns(X, tol=1e-12)
    blocked = independent_columns(X, tol=1e-12, structure=plan)
    assert np.array_equal(blocked[0], dense[0])
    assert np.array_equal(blocked[1], dense[1])

    # Make one block exactly rank deficient.  The structured certificate must
    # decline the easy full-rank path and the established global QR decides the
    # omitted column exactly as before.
    X2 = X.copy()
    X2[:, 3] = X2[:, 0] + 2.0 * X2[:, 1]
    p2 = analyze_design_structure(X2)
    expected = independent_columns(X2, tol=1e-10)
    got = independent_columns(X2, tol=1e-10, structure=p2)
    assert np.array_equal(got[0], expected[0])
    assert np.array_equal(got[1], expected[1])


def test_stale_structure_plan_is_rejected_by_algebra_operators():
    _, X, plan = _block_fixture(seed=1306)
    with pytest.raises(ValueError, match="does not match X shape"):
        column_moments(X[:-1], structure=plan)
    with pytest.raises(ValueError, match="does not match X shape"):
        gram_matrix(X[:, :-1], structure=plan)


def test_block_design_rejects_same_shape_stale_plan_that_would_discard_nonzero():
    _, X, plan = _block_fixture(seed=1307, blocks=3, rows_per=40, width=2)
    bad = X.copy()
    bad[0, -1] = 1.0  # cross-component nonzero invalidates the old certificate
    with pytest.raises(ValueError, match="stale"):
        BlockDesign.from_dense(bad, plan)


def test_compile_block_design_owns_plan_and_supports_row_column_restriction():
    rng, X, _ = _block_fixture(seed=1308, blocks=4, rows_per=50, width=3)
    B, plan = compile_block_design(X)
    assert plan.certified_block_separable
    keep_cols = np.array([0, 2, 3, 5, 7, 11])
    Bs = B.select_columns(keep_cols)
    np.testing.assert_array_equal(Bs.materialize(), X[:, keep_cols])
    mask = rng.random(len(X)) > 0.3
    Br = Bs.subset_rows(mask)
    np.testing.assert_array_equal(Br.materialize(), X[mask][:, keep_cols])


def test_block_design_rank_fast_path_and_qr_fallback_match_dense():
    _, X, plan = _block_fixture(seed=1309, blocks=4, rows_per=80, width=3)
    B = BlockDesign.from_dense(X, plan, verify=False)
    kd, dd = independent_columns(X, tol=1e-12)
    kb, db = independent_columns(B, tol=1e-12)
    assert np.array_equal(kb, kd)
    assert np.array_equal(db, dd)

    X2 = X.copy()
    X2[:, 2] = X2[:, 0] + X2[:, 1]
    B2, _ = compile_block_design(X2)
    kd, dd = independent_columns(X2, tol=1e-10)
    kb, db = independent_columns(B2, tol=1e-10)
    assert np.array_equal(kb, kd)
    assert np.array_equal(db, dd)


def test_storage_planner_distinguishes_one_shot_reuse_and_memory_pressure():
    from econhdfe.compute.execution import plan_design_storage
    _, X, plan = _block_fixture(seed=1310, blocks=6, rows_per=80, width=3)
    one = plan_design_storage(plan, expected_passes=1)
    repeated = plan_design_storage(plan, expected_passes=2)
    constrained = plan_design_storage(
        plan, expected_passes=1,
        memory_budget_mb=(plan.block_dense_bytes + 8 * plan.nobs + 8 * plan.ncols) / 1024**2 * 1.05,
    )
    assert one.representation == "dense"
    assert repeated.representation == "block_dense"
    assert repeated.savings_fraction > 0.7
    assert constrained.representation == "block_dense"
    assert constrained.reason == "dense_exceeds_memory_budget"


def test_select_columns_preserves_response_only_component_rows():
    X = np.zeros((8, 2))
    X[:4, 0] = np.arange(1., 5.)
    X[4:, 1] = np.arange(1., 5.)
    g = np.r_[np.zeros(4, dtype=int), np.ones(4, dtype=int)]
    from econhdfe.compute.design_plan import analyze_design_structure
    from econhdfe.compute.block_design import compile_block_design
    from econhdfe.hdfe.plan import FEPlan
    from econhdfe.hdfe.block_projection import BlockWeightedFEProjector
    plan = FEPlan.from_arrays([g])
    structure = analyze_design_structure(X, groups=plan.groups)
    B, _ = compile_block_design(X, structure=structure)
    one = B.select_columns([0])
    assert len(one.blocks) == 2
    assert one.blocks[1].values.shape == (4, 0)
    projector = BlockWeightedFEProjector.from_design(one, plan, engine="replica")
    out = projector.residualize_response_design(np.arange(8., dtype=float), one, np.ones(8), tol=1e-12)
    assert out.response.shape == (8,)
    assert len(out.design.blocks) == 2
    assert out.design.blocks[1].values.shape == (4, 0)


def test_partitioned_qr_wls_matches_dense_with_shared_columns_and_small_chunks():
    from econhdfe.compute.block_design import DenseDesignBlock
    from econhdfe.compute.partitioned_lstsq import partitioned_weighted_lstsq
    from econhdfe.compute.wls import weighted_lstsq

    rng = np.random.default_rng(9051)
    B, m, local_p = 5, 73, 3
    n = B * m
    k = B * local_p + 2
    blocks = []
    dense = np.zeros((n, k), dtype=np.float64)
    for b in range(B):
        rows = np.arange(b*m, (b+1)*m, dtype=np.int64)
        local_cols = np.arange(b*local_p, (b+1)*local_p, dtype=np.int64)
        shared = np.array([B*local_p, B*local_p + 1], dtype=np.int64)
        cols = np.r_[local_cols, shared]
        vals = rng.normal(size=(m, len(cols)))
        # Mild local ill-conditioning exercises the QR rather than normal eqs.
        vals[:, 1] = vals[:, 0] + 1e-5 * vals[:, 1]
        dense[np.ix_(rows, cols)] = vals
        blocks.append(DenseDesignBlock(b, rows, cols, vals.copy()))
    design = BlockDesign(n, k, tuple(blocks))
    beta0 = rng.normal(scale=.2, size=k)
    y = dense @ beta0 + rng.normal(scale=.4, size=n)
    w = np.exp(rng.normal(scale=.3, size=n))

    beta_dense, resid_dense = weighted_lstsq(dense, y, w, fast=False)
    beta, resid, info = partitioned_weighted_lstsq(
        design, y, w, chunk_rows=17,
    )
    np.testing.assert_allclose(beta, beta_dense, rtol=3e-9, atol=3e-9)
    np.testing.assert_allclose(resid, resid_dense, rtol=2e-9, atol=2e-9)
    assert info.block_count == B
    assert info.compressed_rows <= B * (local_p + 3)
    assert info.compressed_rows < n
    assert info.rank == k
    assert info.method == "block_angular_qr"


def test_partitioned_qr_wls_handles_locally_underdetermined_but_globally_identified_design():
    from econhdfe.compute.block_design import DenseDesignBlock
    from econhdfe.compute.partitioned_lstsq import partitioned_weighted_lstsq

    # Each component has only two rows and three active columns, so no local
    # coefficient vector is identified. Shared columns plus different local
    # supports identify the global system after stacking all compressed blocks.
    rng = np.random.default_rng(9052)
    B, m = 8, 2
    shared = 2
    k = B + shared
    n = B * m
    blocks = []
    dense = np.zeros((n, k))
    for b in range(B):
        rows = np.arange(b*m, (b+1)*m, dtype=np.int64)
        cols = np.array([b, B, B+1], dtype=np.int64)
        vals = rng.normal(size=(m, 3))
        dense[np.ix_(rows, cols)] = vals
        blocks.append(DenseDesignBlock(b, rows, cols, vals.copy()))
    design = BlockDesign(n, k, tuple(blocks))
    y = rng.normal(size=n)
    w = np.exp(rng.normal(scale=.2, size=n))

    sw = np.sqrt(w)
    ref, *_ = np.linalg.lstsq(dense * sw[:, None], y * sw, rcond=None)
    got, resid, info = partitioned_weighted_lstsq(design, y, w, chunk_rows=1)
    np.testing.assert_allclose(got, ref, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(resid, y - dense @ got, rtol=0, atol=2e-12)
    assert info.rank == np.linalg.matrix_rank(dense * sw[:, None])
