from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .design_plan import StructuralDesignPlan


@dataclass(frozen=True, slots=True)
class ColumnMoments:
    """Sufficient statistics for columns of a numerical design.

    ``sum`` and ``sum_squares`` deliberately retain the full-sample semantics:
    structural zeros outside an active block contribute zero to both moments,
    while ``nobs`` remains the total number of observations.  This makes the
    object suitable for exact sample-variance/standardization calculations
    without materializing structural-zero blocks.
    """

    nobs: int
    sum: np.ndarray
    sum_squares: np.ndarray

    def sample_scale(self, *, floor: float = 1e-3) -> np.ndarray:
        if self.nobs <= 1:
            return np.full_like(self.sum, float(floor), dtype=np.float64)
        var = (self.sum_squares - self.sum * self.sum / float(self.nobs)) / float(self.nobs - 1)
        var = np.maximum(var, 0.0)
        scale = np.sqrt(var)
        scale = np.where(np.isfinite(scale), scale, 1.0)
        return np.maximum(scale, float(floor))


def _validate_structure(A: np.ndarray, structure: StructuralDesignPlan | None) -> StructuralDesignPlan | None:
    if structure is None:
        return None
    if int(structure.nobs) != int(A.shape[0]) or int(structure.ncols) != int(A.shape[1]):
        raise ValueError("structural design plan does not match X shape")
    return structure


def column_moments(X, *, structure: StructuralDesignPlan | None = None, frequency_weights=None) -> ColumnMoments:
    """Return exact first and second column moments.

    When an exact disconnected-component certificate is already available,
    only each block's active row/column rectangle is scanned.  No structural
    zeros are multiplied or copied.  Without such a certificate this reduces
    to the dense BLAS/NumPy path. Integer ``frequency_weights`` give the
    moments of the expanded rows (``nobs`` is their sum).
    """
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError("X must be a 2D numerical design")
    p = _validate_structure(A, structure)
    n, k = A.shape
    sums = np.zeros(k, dtype=np.float64)
    ss = np.zeros(k, dtype=np.float64)
    if k == 0:
        return ColumnMoments(int(n), sums, ss)

    if frequency_weights is not None:
        w = np.asarray(frequency_weights, dtype=np.float64)
        sums[:] = w @ A
        ss[:] = np.einsum("i,ij,ij->j", w, A, A, optimize=True)
        return ColumnMoments(int(round(float(np.sum(w)))), sums, ss)

    if p is not None and p.certified_block_separable:
        rb = np.asarray(p.row_blocks)
        for block in p.blocks:
            cols = np.fromiter(block.columns, dtype=np.int64, count=len(block.columns))
            rows = np.flatnonzero(rb == int(block.index))
            # Advanced indexing one column at a time avoids constructing the
            # full block rectangle merely to compute two K-space statistics.
            for j in cols:
                v = A[rows, int(j)]
                sums[int(j)] += float(np.sum(v, dtype=np.float64))
                ss[int(j)] += float(np.dot(v, v))
        return ColumnMoments(int(n), sums, ss)

    sums[:] = np.sum(A, axis=0, dtype=np.float64)
    ss[:] = np.einsum("ij,ij->j", A, A, optimize=True)
    return ColumnMoments(int(n), sums, ss)


def gram_matrix(X, *, weights=None, structure: StructuralDesignPlan | None = None) -> np.ndarray:
    """Return ``X' W X`` using an exact block certificate when available.

    The returned object is always a conventional dense K x K matrix so callers
    do not need special sparse semantics.  Structural off-block entries are
    filled with exact zeros rather than computed from a large dense design.
    """
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError("X must be a 2D numerical design")
    p = _validate_structure(A, structure)
    n, k = A.shape
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w is not None and (w.ndim != 1 or len(w) != n):
        raise ValueError("weights must have one value per observation")
    G = np.zeros((k, k), dtype=np.float64)
    if k == 0:
        return G

    if p is not None and p.certified_block_separable:
        rb = np.asarray(p.row_blocks)
        for block in p.blocks:
            cols = np.fromiter(block.columns, dtype=np.int64, count=len(block.columns))
            rows = np.flatnonzero(rb == int(block.index))
            B = A[np.ix_(rows, cols)]
            if w is None:
                Gb = B.T @ B
            else:
                Gb = B.T @ (B * w[rows, None])
            G[np.ix_(cols, cols)] += Gb
        return G

    if w is None:
        return A.T @ A
    return A.T @ (A * w[:, None])
