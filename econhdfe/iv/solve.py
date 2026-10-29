from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la

from ..compute.wls import as_2d, weighted_arrays
from ..compute.stable_linalg import equilibrated_lstsq
from ..compute.partitioned_lstsq import _merge_augmented_qr, _augmented_qr_workspace
from ..errors import UnderidentifiedError, NumericalError


def _check_projected_rank(moment, nobs, n_parameters):
    """Certify relevance in original regressor units before fitting columns.

    ``moment`` is the projected design divided by ORIGINAL weighted regressor
    norms, in observation space or an orthogonally compressed row space.
    In particular a roundoff-sized fitted regressor must not be rescaled to
    unit norm and reported as an identified direction. This is a numerical
    rank check, not a weak-instrument test.
    """
    singular = la.svdvals(moment, check_finite=False)
    largest = float(singular[0]) if singular.size else 0.0
    cutoff = np.finfo(float).eps * max(nobs, *moment.shape) * max(1.0, largest)
    rank = int(np.count_nonzero(singular > cutoff))
    if rank < n_parameters:
        raise UnderidentifiedError(
            "Instrument moments do not separately identify all coefficients",
            details={"rank": rank, "n_parameters": int(n_parameters)},
            suggestion="Revise the endogenous roles or instrument specification.",
        )


def _check_solved_rank(rank, n_parameters):
    if rank < n_parameters:
        raise NumericalError(
            "Weighted IV solve lost an identifying direction after projection",
            details={"rank": int(rank), "n_parameters": int(n_parameters)},
            suggestion="Inspect nearly dependent instrument/regressor directions and their scaling.",
        )


@dataclass(frozen=True)
class Weighted2SLSResult:
    beta: np.ndarray
    residuals: np.ndarray
    bread: np.ndarray
    rank: int
    fitted: np.ndarray
    projected_X: np.ndarray
    first_stage: np.ndarray


def weighted_2sls(y, X, Z, *, weights=None) -> Weighted2SLSResult:
    """Solve a weighted 2SLS problem with already prepared X and Z.

    This is intentionally independent of the outcome model and HDFE layer.
    Linear IV uses it once; IV-PPML can reuse it at every IRLS iteration with
    changing weights after the working response and FE residualization have
    been updated.
    """
    y = np.asarray(y, dtype=np.float64)
    X = as_2d(X, len(y))
    Z = as_2d(Z, len(y))
    yw, Xw, Zw, _ = weighted_arrays(y, X, Z, weights)

    zscale = np.max(np.abs(Zw), axis=0)
    xscale = np.max(np.abs(Xw), axis=0)
    zscale[zscale == 0] = 1.0
    xscale[xscale == 0] = 1.0
    Xstd, Zstd = Xw / xscale, Zw / zscale
    first_std, *_ = np.linalg.lstsq(Zstd, Xstd, rcond=None)
    xhat_std = Zstd @ first_std
    norm_x = la.norm(Xstd, axis=0)
    norm_x[norm_x == 0] = 1.0
    # Reuse the first-stage projection. Checking H / ||X|| is equivalent to
    # Qz' X / ||X|| and needs no extra observation-sized instrument QR.
    _check_projected_rank(xhat_std / norm_x, len(y), X.shape[1])
    first_stage = first_std * xscale[None, :] / zscale[:, None]
    xhat_w = xhat_std * xscale
    beta, bread, rank = equilibrated_lstsq(xhat_w, yw)
    _check_solved_rank(rank, X.shape[1])
    fitted = X @ beta
    projected_X = xhat_w if weights is None else Z @ first_stage
    return Weighted2SLSResult(
        beta, y - fitted, bread, rank, fitted, projected_X, first_stage
    )


@dataclass(frozen=True)
class BlockWeighted2SLSResult:
    """Orthogonally compressed 2SLS for row-compatible BlockDesigns."""
    beta: np.ndarray
    residuals: np.ndarray
    bread: np.ndarray
    rank: int
    fitted: np.ndarray
    first_stage: np.ndarray


def _block_weighted_scale(design, sqrt_w, chunk_rows):
    """Column maxima without materializing the global weighted design."""
    scale = np.zeros(design.ncols)
    for block in design.blocks:
        if not len(block.columns):
            continue
        for lo in range(0, len(block.rows), chunk_rows):
            hi = min(len(block.rows), lo + chunk_rows)
            values = block.values[lo:hi] * sqrt_w[block.rows[lo:hi], None]
            np.maximum.at(scale, block.columns, np.max(np.abs(values), axis=0))
    scale[scale == 0] = 1.0
    return scale


def _compress_block_iv(y, X, Z, w, chunk):
    """TSQR of weighted [Z, X, y] with bounded row scratch.

    Orthogonal compression preserves first-stage and second-stage least
    squares without forming Z'Z or X'PzX. The small R carries all roles in
    the same row coordinates; structural zero columns remain exact zeros.
    """
    from ..compute.block_design import same_row_partition
    if not same_row_partition(X, Z):
        raise ValueError("block designs must have the same row partition")
    k, l = X.ncols, Z.ncols
    width = l + k + 1
    sqrt_w = np.sqrt(w)
    xscale = _block_weighted_scale(X, sqrt_w, chunk)
    zscale = _block_weighted_scale(Z, sqrt_w, chunk)
    R = None
    for xb, zb in zip(X.blocks, Z.blocks, strict=True):
        for lo in range(0, len(xb.rows), chunk):
            hi = min(len(xb.rows), lo + chunk)
            rows = xb.rows[lo:hi]
            sw = sqrt_w[rows, None]
            block = np.zeros((hi - lo, width), dtype=np.float64)
            block[:, zb.columns] = (zb.values[lo:hi] * sw) / zscale[zb.columns]
            block[:, l + xb.columns] = (xb.values[lo:hi] * sw) / xscale[xb.columns]
            block[:, -1] = y[rows] * sw[:, 0]
            R = _merge_augmented_qr(R, block)
    if R is None:
        R = np.empty((0, width))
    return R[:, -1], R[:, l:l + k], R[:, :l], xscale, zscale


def weighted_2sls_block(y, X, Z, *, weights=None) -> BlockWeighted2SLSResult:
    """Solve weighted 2SLS without materializing global N x K X or Z."""
    from ..compute.block_design import BlockDesign
    if not isinstance(X, BlockDesign) or not isinstance(Z, BlockDesign):
        raise TypeError("weighted_2sls_block requires BlockDesign X and Z")
    if X.nobs != Z.nobs:
        raise ValueError("X and Z must have the same observation count")
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (X.nobs,):
        raise ValueError("y must have one value per observation")
    w = np.ones(X.nobs, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (X.nobs,):
        raise ValueError("weights must have one value per observation")
    chunk, workspace = _augmented_qr_workspace(
        X.nobs, X.ncols + Z.ncols + 1,
        max((len(b.rows) for b in X.blocks), default=1),
        label="block_iv_orthogonal_solve",
    )
    with workspace:
        yw, Xw, Zw, xscale, zscale = _compress_block_iv(y, X, Z, w, chunk)
        # R has fewer rows than the original data. Keep the original observation
        # count in first-stage and identification tolerances, not R's row count.
        first_std = np.linalg.lstsq(
            Zw, Xw, rcond=np.finfo(float).eps * max(X.nobs, Z.ncols),
        )[0]
        projected = Zw @ first_std
        norm_x = la.norm(Xw, axis=0)
        norm_x[norm_x == 0] = 1.0
        _check_projected_rank(projected / norm_x, X.nobs, X.ncols)
        beta, bread, rank = equilibrated_lstsq(projected, yw)
        _check_solved_rank(rank, X.ncols)
        beta = beta / xscale
        bread = (bread / xscale[:, None]) / xscale[None, :]
        first_stage = first_std * xscale[None, :] / zscale[:, None]
        fitted = X.matvec(beta)
        return BlockWeighted2SLSResult(beta, y - fitted, bread, rank, fitted, first_stage)
