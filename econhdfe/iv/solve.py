from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..compute.wls import as_2d, weighted_arrays
from ..compute.stable_linalg import equilibrated_lstsq, equilibrated_gram_inverse


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
    first_std, *_ = np.linalg.lstsq(Zw / zscale, Xw / xscale, rcond=None)
    first_stage = first_std * xscale[None, :] / zscale[:, None]
    xhat_w = Zw @ first_stage
    beta, bread, rank = equilibrated_lstsq(xhat_w, yw)
    fitted = X @ beta
    projected_X = xhat_w if weights is None else Z @ first_stage
    return Weighted2SLSResult(
        beta, y - fitted, bread, rank, fitted, projected_X, first_stage
    )


@dataclass(frozen=True)
class BlockWeighted2SLSResult:
    """Weighted 2SLS sufficient-statistic solve for row-compatible BlockDesigns."""
    beta: np.ndarray
    residuals: np.ndarray
    bread: np.ndarray
    rank: int
    fitted: np.ndarray
    first_stage: np.ndarray


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
    zz = Z.gram(weights=w)
    ztx = Z.cross_gram(X, weights=w)
    zty = Z.t_matvec(w * y)
    zz_inv = equilibrated_gram_inverse(zz)
    first_stage = zz_inv @ ztx
    xpzx = ztx.T @ zz_inv @ ztx
    xpzy = ztx.T @ zz_inv @ zty
    bread, rank = equilibrated_gram_inverse(xpzx, return_rank=True)
    beta = bread @ xpzy
    fitted = X.matvec(beta)
    return BlockWeighted2SLSResult(beta, y - fitted, bread, rank, fitted, first_stage)
