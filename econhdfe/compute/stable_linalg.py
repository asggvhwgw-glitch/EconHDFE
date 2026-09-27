"""Scale-aware small-design linear algebra.

No unscaled normal-equation pseudo inverse is used to decide the numerical
rank. QR compresses observation rows; one SVD of the compressed, equilibrated
design supplies the estimate, covariance bread and rank together. These are
established orthogonal-reduction methods, not new mathematical claims.
"""
from __future__ import annotations
import numpy as np
import scipy.linalg as la


def equilibrated_lstsq(X, y, *, chunk_rows=32768):
    """Return (beta, bread, rank) in original coefficient units.

    ``X`` and ``y`` must already include square-root observation weights.
    Chunked QR of [X / scale, y] avoids a second full observation-sized design.
    For rank-deficient inputs the solution is minimum-norm in scaled units;
    callers needing individually identified coefficients must resolve columns.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y):
        raise ValueError("X must be a matrix and y a matching vector")
    n, k = X.shape
    if k == 0:
        if not np.all(np.isfinite(y)):
            raise ValueError("least-squares inputs must be finite")
        return np.empty(0), np.empty((0, 0)), 0
    if n == 0:
        return np.zeros(k), np.zeros((k, k)), 0
    # Bound validation/scaling scratch as well as the QR row block. Reducing
    # absolute maxima also detects NaN/Inf, without a full n-by-k boolean mask.
    chunk = max(1, min(int(chunk_rows), 2_000_000 // (k + 1)))
    scale = np.zeros(k)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        maxima = np.max(np.abs(X[lo:hi]), axis=0)
        if not np.all(np.isfinite(maxima)) or not np.all(np.isfinite(y[lo:hi])):
            raise ValueError("least-squares inputs must be finite")
        np.maximum(scale, maxima, out=scale)
    scale[scale == 0] = 1.0
    R = np.empty((0, k + 1))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        # One Fortran-ordered buffer, not block + vstack + full-height R.
        # LAPACK's raw form returns compressed R without materializing Q.
        previous = len(R)
        block = np.empty((previous + hi - lo, k + 1), order="F")
        block[:previous] = R
        np.divide(X[lo:hi], scale, out=block[previous:, :k])
        block[previous:, k] = y[lo:hi]
        R = la.qr(block, mode="raw", overwrite_a=True, check_finite=False)[1]
    U, s, Vh = la.svd(R[:, :k], full_matrices=False, check_finite=False)
    cutoff = np.finfo(float).eps * max(n, k) * (s[0] if s.size else 0.0)
    keep = s > cutoff
    rank = int(np.count_nonzero(keep))
    inverse = (Vh[keep].T / s[keep]) / scale[:, None]
    beta = inverse @ (U[:, keep].T @ R[:, k])
    bread = inverse @ inverse.T
    return beta, (bread + bread.T) / 2.0, rank


def equilibrated_gram_inverse(G, *, return_rank=False):
    """Invert a Gram matrix in dimensionless coordinates before unscaling.

    This bounded-memory helper is for full-rank streamed/block covariance
    paths. It cannot recover precision already lost by forming a Gram matrix;
    dense estimation uses ``equilibrated_lstsq`` instead. ``return_rank`` uses
    the SAME dimensionless eigen-directions and 1e-15 relative cutoff as the
    inverse; it is a floating-point numerical rank, not exact categorical DoF.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.size == 0:
        return (G.copy(), 0) if return_rank else G.copy()
    scale = np.sqrt(np.maximum(np.diag(G), 0.0))
    scale[scale == 0] = 1.0
    C = (G / scale[:, None]) / scale[None, :]
    C = (C + C.T) / 2.0
    values, vectors = np.linalg.eigh(C)
    keep = np.abs(values) > 1e-15 * np.max(np.abs(values))
    inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    inverse = (inverse / scale[:, None]) / scale[None, :]
    inverse = (inverse + inverse.T) / 2.0
    return (inverse, int(np.count_nonzero(keep))) if return_rank else inverse
