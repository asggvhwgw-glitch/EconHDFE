"""Role-aware linear-IV solve; shared IV-PPML primitives stay unchanged."""
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from ...compute.stable_linalg import equilibrated_lstsq
from ...iv.solve import Weighted2SLSResult


@dataclass(frozen=True)
class _IVQR:
    """QR of [Z/zscale, E/escale, y] in the same weighted observation space."""
    R: np.ndarray
    zscale: np.ndarray
    escale: np.ndarray


def _role_qr(Zw, Ew, zs, es, *, chunk_rows, yw=None):
    """Bounded QR, with the original [Z,E] layout on the sensitive path."""
    n, l, q = len(Zw), Zw.shape[1], Ew.shape[1]
    width = l + q + int(yw is not None)
    chunk = max(1, min(chunk_rows, 1_000_000 // width))
    R = np.empty((0, width))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        previous = len(R)
        block = np.empty((previous + hi - lo, width), order="F")
        block[:previous] = R
        np.divide(Zw[lo:hi], zs, out=block[previous:, :l])
        np.divide(Ew[lo:hi], es, out=block[previous:, l:l+q])
        if yw is not None:
            block[previous:, -1] = yw[lo:hi]
        R = la.qr(block, mode="raw", overwrite_a=True, check_finite=False)[1]
    return R


def _solve_2sls_roles(y, X, Z, *, weights, n_exog, weighted, chunk_rows=16384):
    """Return the 2SLS result and a small QR factor for this fit only.

    X=[C,E] and Z=[C,I] are already equilibrated and identified by the
    linear-IV caller. `weighted` contains those exact arrays after W**.5;
    no sample, weight or scale may change between preparation and use.
    """
    y = np.asarray(y, dtype=float)
    yw, Xw, Zw, _ = weighted
    Ew = Xw[:, n_exog:]
    n, l, q = len(y), Z.shape[1], Ew.shape[1]
    chunk = max(1, min(chunk_rows, 1_000_000 // (l + q + 1)))
    zs, es = np.zeros(l), np.zeros(q)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        np.maximum(zs, np.max(np.abs(Zw[lo:hi]), axis=0), out=zs)
        np.maximum(es, np.max(np.abs(Ew[lo:hi]), axis=0), out=es)
    zs[zs == 0], es[es == 0] = 1.0, 1.0
    R = _role_qr(Zw, Ew, zs, es, chunk_rows=chunk_rows, yw=yw)
    pi, _, instrument_rank, instrument_singular = np.linalg.lstsq(
        R[:, :l], R[:, l:l+q], rcond=np.finfo(float).eps * max(n, l))
    # 压缩坐标中重建投影会在近共线工具下放大舍入误差。
    # 只为执行分支设保守门槛，不改变模型列、报告秩或诊断。
    compressed_safe = instrument_rank == l and instrument_singular[-1] > (
        np.sqrt(np.finfo(float).eps) * instrument_singular[0]
    )
    if not compressed_safe:
        # Appending y changes QR rounding. Near singular instruments amplify
        # that difference, so preserve the established [Z,E] factorization.
        R = _role_qr(Zw, Ew, zs, es, chunk_rows=chunk_rows)
        pi = np.linalg.lstsq(R[:, :l], R[:, l:],
                            rcond=np.finfo(float).eps * max(n, l))[0]
    pi = pi * es[None, :] / zs[:, None]
    first = np.zeros((l, X.shape[1]))
    first[:n_exog, :n_exog] = np.eye(n_exog)
    first[:, n_exog:] = pi
    xhat = Xw.copy()
    xhat[:, n_exog:] = Zw @ pi
    if not compressed_safe:
        # Reuse prepared arrays and scales; solve in observation coordinates.
        beta, bread, rank = equilibrated_lstsq(xhat, yw)
    else:
        # 沿用观测空间尺度、有限值检查和原始 N 的秩阈值。
        # 不能直接调用小矩阵 equilibrated_lstsq：它会重新选尺度与样本数。
        k = X.shape[1]
        scale = np.zeros(k)
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            maxima = np.max(np.abs(xhat[lo:hi]), axis=0)
            if not np.all(np.isfinite(maxima)) or not np.all(np.isfinite(yw[lo:hi])):
                raise ValueError("least-squares inputs must be finite")
            np.maximum(scale, maxima, out=scale)
        scale[scale == 0] = 1.0
        # Zw = Q Rz diag(zs)，所以 Q' Xhat = Rz diag(zs) first。
        compressed = (R[:, :l] @ (zs[:, None] * first)) / scale
        U, singular, Vh = la.svd(compressed, full_matrices=False, check_finite=False)
        cutoff = np.finfo(float).eps * max(n, k) * (singular[0] if singular.size else 0.)
        keep = singular > cutoff
        inverse = (Vh[keep].T / singular[keep]) / scale[:, None]
        beta = inverse @ (U[:, keep].T @ R[:, -1])
        bread = inverse @ inverse.T
        bread = (bread + bread.T) / 2.
        rank = int(np.count_nonzero(keep))
    fitted = X @ beta
    if weights is None:
        projected = xhat
    else:
        projected = X.copy()
        projected[:, n_exog:] = Z @ pi
    return Weighted2SLSResult(
        beta, y - fitted, bread, rank,
        fitted, projected, first,
    ), _IVQR(R, zs, es) if compressed_safe else None
