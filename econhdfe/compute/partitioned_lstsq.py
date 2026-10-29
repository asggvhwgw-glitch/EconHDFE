from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np

from .block_design import BlockDesign
from ..planner import plan_memory


@dataclass(frozen=True, slots=True)
class PartitionedLeastSquaresInfo:
    """Diagnostics for exact row-partitioned weighted least squares.

    The numerical reduction is a two-stage QR/TSQR construction.  Each row
    component is first compressed to at most ``local_width + 1`` rows using an
    augmented ``[sqrt(W) X, sqrt(W) y]`` factorization.  Those compact factors
    are then embedded in global coefficient coordinates and solved together.
    """

    block_count: int
    compressed_rows: int
    original_rows: int
    ncols: int
    rank: int
    max_local_width: int
    chunk_rows: int
    method: str = "block_angular_qr"

    @property
    def row_compression_ratio(self) -> float:
        if self.original_rows <= 0:
            return 1.0
        return float(self.compressed_rows) / float(self.original_rows)


@dataclass(frozen=True, slots=True)
class PartitionedSolvePlan:
    """Static memory/geometry estimate for row-partitioned WLS.

    The plan is deliberately independent of numerical rank.  It describes the
    largest workspaces implied by the certified row/column topology so callers
    can reject pathological shared-border or fallback geometries before a large
    allocation occurs.
    """

    block_count: int
    ncols: int
    shared_width: int
    exclusive_width: int
    max_local_width: int
    estimated_border_input_rows: int
    border_collect_bytes: int
    border_workspace_bytes: int
    local_factor_bytes: int
    compact_fallback_rows: int
    compact_fallback_bytes: int
    dense_bytes: int
    estimated_dense_qr_work: float
    estimated_partitioned_qr_work: float
    shared_fraction: float
    recommended_method: str
    reason: str

    @property
    def estimated_work_ratio(self) -> float:
        if self.estimated_dense_qr_work <= 0.0:
            return 1.0
        return float(self.estimated_partitioned_qr_work / self.estimated_dense_qr_work)

    def as_dict(self) -> dict:
        return {
            "block_count": int(self.block_count),
            "ncols": int(self.ncols),
            "shared_width": int(self.shared_width),
            "exclusive_width": int(self.exclusive_width),
            "max_local_width": int(self.max_local_width),
            "estimated_border_input_rows": int(self.estimated_border_input_rows),
            "border_collect_bytes": int(self.border_collect_bytes),
            "border_workspace_bytes": int(self.border_workspace_bytes),
            "local_factor_bytes": int(self.local_factor_bytes),
            "compact_fallback_rows": int(self.compact_fallback_rows),
            "compact_fallback_bytes": int(self.compact_fallback_bytes),
            "dense_bytes": int(self.dense_bytes),
            "estimated_dense_qr_work": float(self.estimated_dense_qr_work),
            "estimated_partitioned_qr_work": float(self.estimated_partitioned_qr_work),
            "estimated_work_ratio": float(self.estimated_work_ratio),
            "shared_fraction": float(self.shared_fraction),
            "recommended_method": self.recommended_method,
            "reason": self.reason,
        }


def plan_partitioned_wls(design: BlockDesign) -> PartitionedSolvePlan:
    """Estimate block-angular and generic compact-QR workspace from topology."""
    if not isinstance(design, BlockDesign):
        raise TypeError("design must be a BlockDesign")
    design._ensure_layout()
    counts = design.column_block_counts
    shared = counts > 1
    shared_width = int(np.sum(shared))
    exclusive_width = int(np.sum(counts == 1))
    max_local_width = 0
    border_rows = 0
    max_border_rows = 0
    local_factor_bytes = 0
    compact_rows = 0
    local_qr_work = 0.0
    item = np.dtype(np.float64).itemsize

    for block in design.blocks:
        cols = np.asarray(block.columns, dtype=np.int64)
        kb = int(len(cols))
        max_local_width = max(max_local_width, kb)
        if kb == 0:
            continue
        rrows = min(int(len(block.rows)), kb + 1)
        compact_rows += rrows
        local_qr_work += float(len(block.rows)) * float((kb + 1) ** 2)
        local = int(np.sum(counts[cols] == 1))
        shared_b = kb - local
        if local:
            # Rll + Rls + rly retained for back-substitution.
            local_factor_bytes += item * (local * local + local * shared_b + local)
        if shared_b:
            br = max(0, rrows - local)
            border_rows += br
            max_border_rows = max(max_border_rows, br)

    # Streaming second-stage QR retains at most (S+1) rows plus one incoming
    # block.  This is intentionally an upper bound; the actual R may be smaller.
    border_cols = shared_width + 1
    retained_rows = min(border_rows, border_cols) if border_cols else 0
    border_collect_bytes = item * border_rows * border_cols
    border_workspace = item * border_cols * (retained_rows + max_border_rows)
    compact_fallback_bytes = item * compact_rows * (int(design.ncols) + 1)
    dense_bytes = int(design.nobs) * int(design.ncols) * item
    dense_qr_work = float(design.nobs) * float((int(design.ncols) + 1) ** 2)
    border_qr_work = float(border_rows) * float((shared_width + 1) ** 2)
    partitioned_qr_work = local_qr_work + border_qr_work
    shared_fraction = 0.0 if design.ncols == 0 else float(shared_width) / float(design.ncols)

    if design.ncols == 0:
        method = "empty_design"; reason = "no_regressor_columns"
    elif shared_width == 0:
        method = "disjoint_local_qr"; reason = "columns_are_component_local"
    else:
        method = "block_angular_qr"; reason = "shared_border_with_local_elimination"
    return PartitionedSolvePlan(
        block_count=len(design.blocks), ncols=int(design.ncols),
        shared_width=shared_width, exclusive_width=exclusive_width,
        max_local_width=max_local_width,
        estimated_border_input_rows=int(border_rows),
        border_collect_bytes=int(border_collect_bytes),
        border_workspace_bytes=int(border_workspace),
        local_factor_bytes=int(local_factor_bytes),
        compact_fallback_rows=int(compact_rows),
        compact_fallback_bytes=int(compact_fallback_bytes),
        dense_bytes=int(dense_bytes),
        estimated_dense_qr_work=float(dense_qr_work),
        estimated_partitioned_qr_work=float(partitioned_qr_work),
        shared_fraction=float(shared_fraction),
        recommended_method=method, reason=reason,
    )


def _validate_wls_inputs(design: BlockDesign, y, weights):
    y0 = np.asarray(y, dtype=np.float64)
    w0 = np.asarray(weights, dtype=np.float64)
    if y0.ndim != 1 or len(y0) != design.nobs:
        raise ValueError("y must have one value per BlockDesign observation")
    if w0.ndim != 1 or len(w0) != design.nobs:
        raise ValueError("weights must have one value per BlockDesign observation")
    if not np.all(np.isfinite(y0)) or not np.all(np.isfinite(w0)):
        raise ValueError("y and weights must be finite")
    if np.any(w0 < 0.0):
        raise ValueError("weights must be nonnegative")
    for block in design.blocks:
        if not np.all(np.isfinite(block.values)):
            raise ValueError("BlockDesign values must be finite for weighted least squares")
    return y0, w0


def _compress_weighted_block(values, y, weights, *, chunk_rows: int) -> np.ndarray:
    """Compress one weighted local least-squares block to an augmented R.

    The weighted matrix is constructed chunk by chunk so a very tall local
    component never requires an additional full ``n_b x (k_b+1)`` workspace.
    """
    X = np.asarray(values, dtype=np.float64)
    y0 = np.asarray(y, dtype=np.float64)
    w0 = np.asarray(weights, dtype=np.float64)
    if X.ndim != 2 or y0.shape != (X.shape[0],) or w0.shape != y0.shape:
        raise ValueError("local block dimensions do not agree")
    p = X.shape[1] + 1
    chunk = max(1, int(chunk_rows))
    R = None
    for lo in range(0, len(y0), chunk):
        hi = min(len(y0), lo + chunk)
        sw = np.sqrt(w0[lo:hi])
        aug = np.empty((hi - lo, p), dtype=np.float64)
        if X.shape[1]:
            aug[:, :-1] = X[lo:hi] * sw[:, None]
        aug[:, -1] = y0[lo:hi] * sw
        Rc = np.linalg.qr(aug, mode="r")
        if R is None:
            R = np.asarray(Rc, dtype=np.float64)
        else:
            R = np.linalg.qr(np.vstack((R, Rc)), mode="r")
    if R is None:
        return np.empty((0, p), dtype=np.float64)
    return np.asarray(R, dtype=np.float64)



def _merge_augmented_qr(current: np.ndarray | None, incoming: np.ndarray) -> np.ndarray | None:
    """Streaming TSQR merge for a small augmented least-squares factor."""
    A = np.asarray(incoming, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("incoming QR block must be two-dimensional")
    if A.shape[0] == 0:
        return current
    Rc = np.asarray(np.linalg.qr(A, mode="r"), dtype=np.float64)
    if current is None or current.shape[0] == 0:
        return Rc
    if current.shape[1] != Rc.shape[1]:
        raise ValueError("streamed QR factors have incompatible widths")
    return np.asarray(np.linalg.qr(np.vstack((current, Rc)), mode="r"), dtype=np.float64)


def _augmented_qr_workspace(nobs, width, max_rows, *, label="augmented_qr_solve"):
    """Admit compact QR factors and size row scratch to the active allocation.

    Compact factors, SVD/bread work and three observation vectors are retained
    independently of row scratch. This conservative bound is O(width**2 + N),
    not a promise of width-independent memory. Resource policy stays in the
    compute layer; callers need only supply their numerical matrix geometry.
    """
    from ..planner.scheduler import current_allocation, reserve_memory

    chunk = max(1, min(32768, 2_000_000 // width, max_rows))
    fixed = 8 * (12 * width * width + 3 * nobs + 4 * width)
    per_row = 8 * 4 * width
    allocation = current_allocation()
    if allocation is None:
        return chunk, nullcontext()
    available = allocation.unreserved_memory_bytes()
    chunk = max(1, min(chunk, (available - fixed) // per_row))
    return chunk, reserve_memory(fixed + chunk * per_row, label=label)


def _disjoint_weighted_lstsq(
    design: BlockDesign, y0: np.ndarray, w0: np.ndarray, *, resid_out, chunk_rows: int,
):
    """Exact local-QR WLS when every coefficient belongs to one row component."""
    beta = np.zeros(design.ncols, dtype=np.float64)
    resid = np.empty(design.nobs, dtype=np.float64) if resid_out is None else np.asarray(resid_out, dtype=np.float64)
    if resid.shape != y0.shape:
        raise ValueError("resid_out has the wrong shape")
    resid[:] = y0
    rank = 0
    compressed_rows = 0
    max_width = 0
    for block in design.blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        kb = int(len(block.columns))
        max_width = max(max_width, kb)
        if kb == 0:
            continue
        R = _compress_weighted_block(block.values, y0[rows], w0[rows], chunk_rows=chunk_rows)
        compressed_rows += int(R.shape[0])
        if R.size == 0:
            continue
        b, _, rb, _ = np.linalg.lstsq(R[:, :kb], R[:, kb], rcond=None)
        beta[block.columns] = np.asarray(b, dtype=np.float64)
        rank += int(rb)
        resid[rows] -= block.values @ beta[block.columns]
    info = PartitionedLeastSquaresInfo(
        block_count=len(design.blocks), compressed_rows=int(compressed_rows),
        original_rows=int(design.nobs), ncols=int(design.ncols), rank=int(rank),
        max_local_width=int(max_width), chunk_rows=max(1, int(chunk_rows)),
        method="disjoint_local_qr",
    )
    return beta, resid, info


def compress_partitioned_wls(
    design: BlockDesign,
    y,
    weights,
    *,
    chunk_rows: int = 250_000,
):
    """Build the exact compact least-squares problem for a row partition.

    For block ``b`` with global coefficient selector ``C_b``, let
    ``A_b = [sqrt(W_b) X_b, sqrt(W_b) y_b] = Q_b R_b``.  Orthogonality gives

    ``||sqrt(W_b) (y_b - X_b beta[C_b])||_2``
    ``= ||R_b[:, -1] - R_b[:, :-1] beta[C_b]||_2``.

    The final compact matrix is preallocated from the topology plan.  This
    avoids the previous list-plus-vstack peak, which could transiently hold two
    copies of an already wide fallback matrix.
    """
    if not isinstance(design, BlockDesign):
        raise TypeError("design must be a BlockDesign")
    design._ensure_layout()
    y0, w0 = _validate_wls_inputs(design, y, weights)
    solve_plan = plan_partitioned_wls(design)
    Xc = np.zeros((solve_plan.compact_fallback_rows, design.ncols), dtype=np.float64)
    yc = np.empty(solve_plan.compact_fallback_rows, dtype=np.float64)
    cursor = 0
    max_width = 0

    for block in design.blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        k_local = int(len(block.columns))
        max_width = max(max_width, k_local)
        if k_local == 0:
            continue
        R = _compress_weighted_block(
            block.values, y0[rows], w0[rows], chunk_rows=chunk_rows
        )
        if R.size == 0:
            continue
        nr = int(R.shape[0])
        hi = cursor + nr
        if hi > Xc.shape[0]:
            raise RuntimeError("partitioned WLS compact-row estimate was violated")
        Xc[cursor:hi, block.columns] = R[:, :k_local]
        yc[cursor:hi] = R[:, k_local]
        cursor = hi

    if cursor != Xc.shape[0]:
        # The topology estimate is exact for ordinary QR shapes, but retaining a
        # defensive trim makes the helper robust to future compression backends.
        Xc = Xc[:cursor]
        yc = yc[:cursor]
    return Xc, yc, max_width



def _block_angular_weighted_lstsq(
    design: BlockDesign,
    y0: np.ndarray,
    w0: np.ndarray,
    *,
    resid_out,
    chunk_rows: int,
    rank_tol: float | None = None,
    collect_border: bool = True,
):
    """Structure-exploiting orthogonal solve for bordered/block-angular WLS.

    Columns used in only one row component are local variables.  They are
    eliminated by each component's QR factorization.  Only columns appearing
    in multiple components enter the second-stage border problem.  This is the
    standard block-angular least-squares reduction, applied to ``BlockDesign``
    metadata so the global N x K matrix is never formed.

    Returns ``None`` when a local exclusive-column block is numerically rank
    deficient; callers then use the more general compact-QR fallback.
    """
    if rank_tol is None:
        # Local elimination is more sensitive than a one-shot global QR: an
        # ill-conditioned exclusive block can amplify rounding error before the
        # shared-border solve. sqrt(eps) is a conservative stability gate;
        # ambiguous cases retain correctness via the compact global-QR fallback.
        rank_tol = float(np.sqrt(np.finfo(np.float64).eps))
    counts = design.column_block_counts
    shared_global = np.flatnonzero(counts > 1).astype(np.int64, copy=False)
    if shared_global.size == 0:
        return None
    shared_map = np.full(design.ncols, -1, dtype=np.int64)
    shared_map[shared_global] = np.arange(len(shared_global), dtype=np.int64)

    border_R = None
    border_chunks = [] if collect_border else None
    border_input_rows = 0
    local_factors = []
    max_width = 0
    seen = np.zeros(design.nobs, dtype=bool)

    for block in design.blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        if np.any(seen[rows]):
            raise ValueError("BlockDesign row components must be disjoint")
        seen[rows] = True
        cols = np.asarray(block.columns, dtype=np.int64)
        max_width = max(max_width, int(len(cols)))
        if cols.size == 0:
            continue
        is_shared = counts[cols] > 1
        pos_local = np.flatnonzero(~is_shared)
        pos_shared = np.flatnonzero(is_shared)
        ordered_pos = np.r_[pos_local, pos_shared]
        local_cols = cols[pos_local]
        shared_cols = cols[pos_shared]
        values = block.values[:, ordered_pos]
        R = _compress_weighted_block(
            values, y0[rows], w0[rows], chunk_rows=chunk_rows
        )
        l = int(len(local_cols)); sb = int(len(shared_cols))
        if R.shape[0] < l:
            return None
        if l:
            Rll = np.asarray(R[:l, :l], dtype=np.float64)
            singular = np.linalg.svd(Rll, compute_uv=False)
            scale = max(float(singular[0]) if singular.size else 0.0, 1.0)
            if singular.size != l or float(singular[-1]) <= float(rank_tol) * scale:
                return None
            Rls = np.asarray(R[:l, l:l+sb], dtype=np.float64)
            rly = np.asarray(R[:l, l+sb], dtype=np.float64)
            local_factors.append((local_cols, shared_cols, Rll, Rls, rly))

        # After local columns are eliminated, only the trailing orthogonal
        # equations constrain the shared border coefficients.  Pure residual
        # rows with no shared columns are beta-independent constants and can be
        # omitted from the second-stage solve.
        trailing = R[l:, l:l+sb]
        rhs = R[l:, l+sb]
        if sb and trailing.shape[0]:
            informative = np.any(trailing != 0.0, axis=1)
            if np.any(informative):
                tr = np.asarray(trailing[informative], dtype=np.float64)
                rr = np.asarray(rhs[informative], dtype=np.float64)
                aug = np.zeros((tr.shape[0], len(shared_global) + 1), dtype=np.float64)
                aug[:, shared_map[shared_cols]] = tr
                aug[:, -1] = rr
                if collect_border:
                    border_chunks.append(aug)
                else:
                    border_R = _merge_augmented_qr(border_R, aug)
                border_input_rows += int(tr.shape[0])

    if design.nobs and not np.all(seen):
        raise ValueError("BlockDesign must cover every observation")

    if collect_border and border_chunks:
        border_matrix = np.vstack(border_chunks)
        Xs = np.asarray(border_matrix[:, :-1], dtype=np.float64)
        ys = np.asarray(border_matrix[:, -1], dtype=np.float64)
    elif border_R is not None and border_R.shape[0]:
        Xs = np.asarray(border_R[:, :-1], dtype=np.float64)
        ys = np.asarray(border_R[:, -1], dtype=np.float64)
    else:
        Xs = np.empty((0, len(shared_global)), dtype=np.float64)
        ys = np.empty(0, dtype=np.float64)

    if Xs.shape[0]:
        beta_s, _, rank_s, singular_s = np.linalg.lstsq(Xs, ys, rcond=None)
        beta_s = np.asarray(beta_s, dtype=np.float64)
        # Rank-deficient shared borders require a joint global minimum-norm
        # solution; local back-substitution alone does not preserve that norm.
        # Extremely ill-conditioned full-rank borders are also sent to the
        # global compact-QR safety path to avoid error amplification.
        if int(rank_s) < len(shared_global):
            return None
        singular_s = np.asarray(singular_s, dtype=np.float64)
        if singular_s.size and float(singular_s[-1]) <= float(rank_tol) * max(float(singular_s[0]), 1.0):
            return None
    else:
        beta_s = np.zeros(len(shared_global), dtype=np.float64)
        rank_s = 0

    beta = np.zeros(design.ncols, dtype=np.float64)
    beta[shared_global] = beta_s
    for local_cols, shared_cols, Rll, Rls, rly in local_factors:
        bs = beta[shared_cols]
        beta[local_cols] = np.linalg.solve(Rll, rly - Rls @ bs)

    if resid_out is None:
        resid = np.empty(design.nobs, dtype=np.float64)
    else:
        resid = np.asarray(resid_out, dtype=np.float64)
        if resid.shape != y0.shape:
            raise ValueError("resid_out has the wrong shape")
    resid[:] = y0
    for block in design.blocks:
        if block.values.shape[1]:
            resid[block.rows] -= block.values @ beta[block.columns]

    rank = int(rank_s + np.sum(counts == 1))
    info = PartitionedLeastSquaresInfo(
        block_count=len(design.blocks),
        compressed_rows=int(border_input_rows),
        original_rows=int(design.nobs),
        ncols=int(design.ncols),
        rank=rank,
        max_local_width=int(max_width),
        chunk_rows=max(1, int(chunk_rows)),
        method="block_angular_qr",
    )
    return beta, resid, info

def partitioned_weighted_lstsq(
    design: BlockDesign,
    y,
    weights,
    *,
    resid_out=None,
    chunk_rows: int = 250_000,
    fallback_memory_budget_mb: float | None = None,
    border_collect_budget_mb: float | None = 64.0,
):
    """Exact high-accuracy WLS for row-partitioned designs with shared columns.

    The preferred path is a classic block-angular QR reduction: component-local
    coefficients are eliminated locally and only genuinely shared coefficients
    enter the second-stage least-squares problem.  A generic two-stage compact
    QR fallback handles numerically ambiguous local blocks.  Neither path
    materializes the global ``N x K`` design or forms normal equations.
    """
    if not isinstance(design, BlockDesign):
        raise TypeError("design must be a BlockDesign")
    design._ensure_layout()
    y0, w0 = _validate_wls_inputs(design, y, weights)
    solve_plan = plan_partitioned_wls(design)
    if design.ncols == 0:
        if resid_out is None:
            resid = y0.copy()
        else:
            resid = np.asarray(resid_out, dtype=np.float64)
            if resid.shape != y0.shape:
                raise ValueError("resid_out has the wrong shape")
            resid[:] = y0
        info = PartitionedLeastSquaresInfo(
            len(design.blocks), 0, design.nobs, 0, 0, 0,
            max(1, int(chunk_rows)), "empty_design",
        )
        return np.empty(0, dtype=np.float64), resid, info

    if solve_plan.shared_width == 0:
        return _disjoint_weighted_lstsq(
            design, y0, w0, resid_out=resid_out, chunk_rows=chunk_rows
        )

    if border_collect_budget_mb is None:
        border_budget = None
    else:
        border_budget = plan_memory(
            requested_budget_bytes=int(float(border_collect_budget_mb) * 1024**2)
        ).effective_budget_bytes
    collect_border = border_budget is None or solve_plan.border_collect_bytes <= border_budget
    angular = _block_angular_weighted_lstsq(
        design, y0, w0, resid_out=resid_out, chunk_rows=chunk_rows,
        collect_border=collect_border,
    )
    if angular is not None:
        return angular

    if fallback_memory_budget_mb is not None:
        budget = plan_memory(
            requested_budget_bytes=int(float(fallback_memory_budget_mb) * 1024**2)
        ).effective_budget_bytes
        if solve_plan.compact_fallback_bytes > budget:
            raise MemoryError(
                "partitioned WLS local-rank fallback exceeds the configured memory budget; "
                f"estimated compact workspace={solve_plan.compact_fallback_bytes / 1024**2:.1f} MiB, "
                f"budget={budget / 1024**2:.1f} MiB"
            )
    Xc, yc, max_width = compress_partitioned_wls(
        design, y0, w0, chunk_rows=chunk_rows
    )
    if Xc.shape[0] == 0:
        beta = np.zeros(design.ncols, dtype=np.float64)
        rank = 0
    else:
        beta, _, rank, singular = np.linalg.lstsq(Xc, yc, rcond=None)
        beta = np.asarray(beta, dtype=np.float64)
        singular = np.asarray(singular, dtype=np.float64)
        stability_tol = float(np.sqrt(np.finfo(np.float64).eps))
        if int(rank) == design.ncols and singular.size:
            if float(singular[-1]) <= stability_tol * max(float(singular[0]), 1.0):
                raise np.linalg.LinAlgError(
                    "partitioned WLS is numerically ill-conditioned after compact QR; "
                    "use the pooled high-accuracy path or resolve near-collinearity before structured execution"
                )

    if resid_out is None:
        resid = np.empty(design.nobs, dtype=np.float64)
    else:
        resid = np.asarray(resid_out, dtype=np.float64)
        if resid.shape != y0.shape:
            raise ValueError("resid_out has the wrong shape")
    resid[:] = y0
    for block in design.blocks:
        if block.values.shape[1]:
            resid[block.rows] -= block.values @ beta[block.columns]

    info = PartitionedLeastSquaresInfo(
        block_count=len(design.blocks),
        compressed_rows=int(Xc.shape[0]),
        original_rows=int(design.nobs),
        ncols=int(design.ncols),
        rank=int(rank),
        max_local_width=int(max_width),
        chunk_rows=max(1, int(chunk_rows)),
        method="compact_qr_fallback",
    )
    return beta, resid, info

