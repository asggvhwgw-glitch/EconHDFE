from types import SimpleNamespace

import numpy as np
import pytest

from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.compute.design_plan import analyze_design_structure, analyze_structural_terms
from econhdfe.compute.partitioned_lstsq import (
    partitioned_weighted_lstsq,
    plan_partitioned_wls,
)


def _dense_wls(X, y, w):
    sw = np.sqrt(w)
    beta, _, rank, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta, y - X @ beta, int(rank)


def test_block_design_rejects_overlapping_rows_before_operator_use():
    blocks = (
        DenseDesignBlock(0, np.array([0, 1]), np.array([0]), np.ones((2, 1))),
        DenseDesignBlock(1, np.array([1, 2]), np.array([1]), np.ones((2, 1))),
    )
    design = BlockDesign(3, 2, blocks)
    with pytest.raises(ValueError, match="disjoint"):
        design.matvec(np.ones(2))


def test_block_design_rejects_missing_rows_duplicate_columns_and_bad_values_shape():
    missing = BlockDesign(
        3, 1,
        (DenseDesignBlock(0, np.array([0, 1]), np.array([0]), np.ones((2, 1))),),
    )
    with pytest.raises(ValueError, match="cover every observation"):
        missing.column_moments()

    dupcols = BlockDesign(
        2, 2,
        (DenseDesignBlock(0, np.array([0, 1]), np.array([0, 0]), np.ones((2, 2))),),
    )
    with pytest.raises(ValueError, match="unique within each block"):
        dupcols.gram()

    badshape = BlockDesign(
        2, 1,
        (DenseDesignBlock(0, np.array([0, 1]), np.array([0]), np.ones((1, 1))),),
    )
    with pytest.raises(ValueError, match="match row/column"):
        badshape.t_matvec(np.ones(2))


def test_block_design_scale_rejects_nonfinite_and_zero_inverse_scale():
    d = BlockDesign(
        2, 1,
        (DenseDesignBlock(0, np.array([0, 1]), np.array([0]), np.ones((2, 1))),),
    )
    with pytest.raises(ValueError, match="nonzero"):
        d.scale_columns(np.array([0.0]))
    with pytest.raises(ValueError, match="finite"):
        d.scale_columns(np.array([np.nan]))


def test_structural_analyzer_rejects_bad_tolerance_and_raw_sparse_fe_codes():
    X = np.ones((4, 1))
    with pytest.raises(ValueError, match="zero_tol"):
        analyze_design_structure(X, zero_tol=-1.0)
    with pytest.raises(ValueError, match="densely factorized"):
        analyze_design_structure(X, groups=[np.array([0, 1, 2, 100])])


def test_symbolic_analyzer_rejects_noninteger_negative_and_out_of_range_codes():
    def term(codes):
        return SimpleNamespace(codes=np.asarray(codes), active=np.array([True, True]))

    with pytest.raises(TypeError, match="integer"):
        analyze_structural_terms([term([0.0, 1.0])])
    with pytest.raises(ValueError, match="outside"):
        analyze_structural_terms([term([0, -1])])
    with pytest.raises(ValueError, match="outside"):
        analyze_structural_terms([term([0, 2])])


def test_partitioned_solver_plan_reports_shared_border_and_fallback_geometry():
    rng = np.random.default_rng(4101)
    blocks = []
    nblocks, per, shared = 5, 20, 2
    ncols = nblocks + shared
    for b in range(nblocks):
        rows = np.arange(b * per, (b + 1) * per)
        cols = np.array([b, nblocks, nblocks + 1])
        vals = rng.normal(size=(per, 3))
        blocks.append(DenseDesignBlock(b, rows, cols, vals))
    d = BlockDesign(nblocks * per, ncols, tuple(blocks))
    p = plan_partitioned_wls(d)
    assert p.shared_width == shared
    assert p.exclusive_width == nblocks
    assert p.recommended_method == "block_angular_qr"
    assert p.border_workspace_bytes > 0
    assert p.compact_fallback_bytes > p.border_workspace_bytes


def test_partitioned_disjoint_local_qr_matches_dense_and_does_not_use_global_fallback():
    rng = np.random.default_rng(4102)
    blocks = []
    B, per, local = 4, 35, 3
    for b in range(B):
        rows = np.arange(b * per, (b + 1) * per)
        cols = np.arange(b * local, (b + 1) * local)
        vals = rng.normal(size=(per, local))
        blocks.append(DenseDesignBlock(b, rows, cols, vals))
    d = BlockDesign(B * per, B * local, tuple(blocks))
    y = rng.normal(size=d.nobs)
    w = np.exp(rng.normal(scale=.2, size=d.nobs))
    got, resid, info = partitioned_weighted_lstsq(d, y, w, chunk_rows=7)
    X = d.materialize()
    ref, ref_resid, rank = _dense_wls(X, y, w)
    np.testing.assert_allclose(got, ref, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(resid, ref_resid, rtol=2e-12, atol=2e-12)
    assert info.rank == rank
    assert info.method == "disjoint_local_qr"


def test_partitioned_streamed_border_qr_matches_dense_with_many_shared_patterns():
    rng = np.random.default_rng(4103)
    B, per, S = 18, 24, 7
    blocks = []
    ncols = B + S
    for b in range(B):
        rows = np.arange(b * per, (b + 1) * per)
        shared = np.array([B + (b % S), B + ((b + 2) % S), B + ((b + 4) % S)])
        cols = np.r_[b, shared]
        vals = rng.normal(size=(per, len(cols)))
        blocks.append(DenseDesignBlock(b, rows, cols, vals))
    d = BlockDesign(B * per, ncols, tuple(blocks))
    y = rng.normal(size=d.nobs)
    w = np.exp(rng.normal(scale=.4, size=d.nobs))
    got, resid, info = partitioned_weighted_lstsq(d, y, w, chunk_rows=5)
    X = d.materialize()
    ref, ref_resid, rank = _dense_wls(X, y, w)
    np.testing.assert_allclose(got, ref, rtol=5e-11, atol=5e-11)
    np.testing.assert_allclose(resid, ref_resid, rtol=5e-11, atol=5e-11)
    assert info.rank == rank
    assert info.method == "block_angular_qr"


def test_partitioned_rank_ambiguity_fallback_honors_memory_budget():
    rng = np.random.default_rng(4104)
    per = 20
    # col 0/1 are exclusive but identically zero => local Rll is singular.
    b0 = DenseDesignBlock(0, np.arange(per), np.array([0, 2]), np.c_[np.zeros(per), rng.normal(size=per)])
    b1 = DenseDesignBlock(1, np.arange(per, 2 * per), np.array([1, 2]), np.c_[np.zeros(per), rng.normal(size=per)])
    d = BlockDesign(2 * per, 3, (b0, b1))
    y = rng.normal(size=d.nobs)
    w = np.ones(d.nobs)
    with pytest.raises(MemoryError, match="fallback exceeds"):
        partitioned_weighted_lstsq(d, y, w, fallback_memory_budget_mb=1e-9)

    got, resid, info = partitioned_weighted_lstsq(d, y, w)
    ref, ref_resid, rank = _dense_wls(d.materialize(), y, w)
    np.testing.assert_allclose(d.materialize() @ got, d.materialize() @ ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(resid, ref_resid, rtol=1e-12, atol=1e-12)
    assert info.rank == rank
    assert info.method == "compact_qr_fallback"


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_partitioned_wls_randomized_full_rank_parity(seed):
    rng = np.random.default_rng(4200 + seed)
    B = int(rng.integers(2, 7))
    per = int(rng.integers(12, 35))
    local = int(rng.integers(1, 4))
    S = int(rng.integers(1, 4))
    ncols = B * local + S
    blocks = []
    for b in range(B):
        rows = np.arange(b * per, (b + 1) * per)
        local_cols = np.arange(b * local, (b + 1) * local)
        shared_cols = np.arange(B * local, ncols)
        cols = np.r_[local_cols, shared_cols]
        vals = rng.normal(size=(per, len(cols)))
        blocks.append(DenseDesignBlock(b, rows, cols, vals))
    d = BlockDesign(B * per, ncols, tuple(blocks))
    y = rng.normal(size=d.nobs)
    w = np.exp(rng.normal(scale=.3, size=d.nobs))
    # Include zero-weight observations without making any whole component vanish.
    if per > 4:
        w[np.arange(0, d.nobs, per) + 1] = 0.0
    got, resid, info = partitioned_weighted_lstsq(d, y, w, chunk_rows=int(rng.integers(1, 9)))
    X = d.materialize()
    ref, ref_resid, rank = _dense_wls(X, y, w)
    np.testing.assert_allclose(got, ref, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(resid, ref_resid, rtol=2e-10, atol=2e-10)
    assert info.rank == rank == ncols


def test_partitioned_solver_rejects_numerically_unstable_local_elimination():
    rng = np.random.default_rng(4301)
    B, per, local, S = 4, 32, 2, 1
    K = B * local + S
    blocks = []
    for b in range(B):
        rows = np.arange(b * per, (b + 1) * per)
        cols = np.r_[np.arange(b * local, (b + 1) * local), K - 1]
        base = rng.normal(size=per)
        vals = np.c_[base, base + 1e-10 * rng.normal(size=per), rng.normal(size=per)]
        blocks.append(DenseDesignBlock(b, rows, cols, vals))
    d = BlockDesign(B * per, K, tuple(blocks))
    y = rng.normal(size=d.nobs)
    w = np.exp(rng.normal(scale=1.0, size=d.nobs))
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        partitioned_weighted_lstsq(d, y, w, chunk_rows=7)


def test_partitioned_forced_streaming_border_matches_collected_border_and_dense():
    rng = np.random.default_rng(4302)
    B, per, local, S = 10, 28, 2, 5
    K = B * local + S
    blocks = []
    for b in range(B):
        rows = np.arange(b * per, (b + 1) * per)
        cols = np.r_[np.arange(b * local, (b + 1) * local), np.arange(B * local, K)]
        blocks.append(DenseDesignBlock(b, rows, cols, rng.normal(size=(per, len(cols)))))
    d = BlockDesign(B * per, K, tuple(blocks))
    y = rng.normal(size=d.nobs)
    w = np.exp(rng.normal(scale=.25, size=d.nobs))
    b_collect, r_collect, i_collect = partitioned_weighted_lstsq(
        d, y, w, chunk_rows=6, border_collect_budget_mb=64.0
    )
    b_stream, r_stream, i_stream = partitioned_weighted_lstsq(
        d, y, w, chunk_rows=6, border_collect_budget_mb=0.0
    )
    ref, ref_resid, _ = _dense_wls(d.materialize(), y, w)
    np.testing.assert_allclose(b_collect, ref, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(b_stream, ref, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(r_collect, ref_resid, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(r_stream, ref_resid, rtol=2e-11, atol=2e-11)
    assert i_collect.method == i_stream.method == "block_angular_qr"


def test_external_block_layout_metadata_is_detached_after_validation():
    rows = np.array([0, 1], dtype=np.int64)
    cols = np.array([0], dtype=np.int64)
    d = BlockDesign(2, 1, (DenseDesignBlock(0, rows, cols, np.ones((2, 1))),))
    np.testing.assert_allclose(d.matvec(np.array([2.0])), np.array([2.0, 2.0]))
    rows[:] = [1, 1]
    cols[:] = [99]
    # The certified layout owns read-only metadata and is unaffected by caller mutation.
    np.testing.assert_allclose(d.matvec(np.array([3.0])), np.array([3.0, 3.0]))
    assert not d.blocks[0].rows.flags.writeable
    assert not d.blocks[0].columns.flags.writeable


def test_partitioned_wls_rejects_nonfinite_block_values():
    d = BlockDesign(
        2, 1,
        (DenseDesignBlock(0, np.array([0, 1]), np.array([0]), np.array([[1.0], [np.nan]])),),
    )
    with pytest.raises(ValueError, match="must be finite"):
        partitioned_weighted_lstsq(d, np.ones(2), np.ones(2))


def test_storage_planner_does_not_equate_partition_certificate_with_block_execution():
    from econhdfe.compute.design_plan import PartitionedDesignPlan, DesignBlock
    from econhdfe.compute.execution import plan_design_storage

    n, k, B = 1000, 8, 20
    row_blocks = np.repeat(np.arange(B, dtype=np.int32), n // B)
    blocks = tuple(
        DesignBlock(b, tuple(range(k)), n // B, (n // B) * k * 8)
        for b in range(B)
    )
    structure = PartitionedDesignPlan(
        nobs=n,
        ncols=k,
        column_nnz=np.full(k, n, dtype=np.int64),
        blocks=blocks,
        row_blocks=row_blocks,
        column_block_counts=np.full(k, B, dtype=np.int64),
        dense_bytes=n * k * 8,
        block_dense_bytes=n * k * 8,
        structural_zero_fraction=0.0,
        reason="exact_fe_partition_shared_columns",
    )
    assert structure.certified_partitioned
    storage = plan_design_storage(structure, expected_passes=20)
    assert storage.representation == "dense"
    assert storage.reason == "insufficient_storage_savings"
