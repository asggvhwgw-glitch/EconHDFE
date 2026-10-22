"""Role-aware linear-IV solve; shared IV-PPML primitives stay unchanged."""
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from ...compute.stable_linalg import equilibrated_lstsq
from ...iv.solve import Weighted2SLSResult
from ...errors import NumericalError, UnderidentifiedError


@dataclass(frozen=True)
class _IVQR:
    """QR of [Z/zscale, E/escale, y] in the same weighted observation space."""
    R: np.ndarray
    zscale: np.ndarray
    escale: np.ndarray


@dataclass(frozen=True)
class _StableIVQR:
    """Small TSQR factors; observation-space coordinates are replayed in blocks.

    The referenced weighted arrays must remain unchanged until inference ends.
    No instrument inverse is used to reconstruct an orthogonal basis.
    """
    R: np.ndarray
    zscale: np.ndarray
    escale: np.ndarray
    instrument_coordinates: np.ndarray
    projected_coordinates: np.ndarray
    influence_coordinates: np.ndarray
    projected_left: np.ndarray
    residual_coordinates: np.ndarray
    diagnostic_coordinates: np.ndarray
    top_q: np.ndarray
    block_sizes: tuple
    Zw: np.ndarray
    Ew: np.ndarray
    yw: np.ndarray
    n_exog: int
    chunk_rows: int

    @property
    def nobs(self):
        return len(self.yw)

    @property
    def n_structural(self):
        return self.projected_coordinates.shape[1]

    def iter_transformed(self, coordinates):
        """Yield (lo, hi, Q @ coordinates) without storing the tall Q.

        Combine all inference coordinates into one call so this second QR pass
        supplies projected regressors, influence functions and moment scores.
        """
        offset = 0
        for block_index, lo in enumerate(range(0, self.nobs, self.chunk_rows)):
            hi = min(self.nobs, lo + self.chunk_rows)
            block = _stable_role_block(
                self.Zw, self.Ew, self.yw, self.zscale, self.escale, lo, hi,
            )
            local_q, _ = la.qr(block, mode="economic", overwrite_a=True,
                               check_finite=False)
            rows = self.block_sizes[block_index]
            small = self.top_q[offset:offset + rows] @ coordinates
            yield lo, hi, local_q @ small
            offset += rows


def _stable_role_block(Zw, Ew, yw, zs, es, lo, hi):
    l, q = Zw.shape[1], Ew.shape[1]
    block = np.empty((hi - lo, l + q + 1), order="F")
    np.divide(Zw[lo:hi], zs, out=block[:, :l])
    np.divide(Ew[lo:hi], es, out=block[:, l:l + q])
    block[:, -1] = yw[lo:hi]
    return block


def _check_partial_identification(Rz, Re, n_exog, nobs):
    """Check numerical identification after removing the control space.

    Canonical correlations separate near-dependent controls from a genuinely
    unresolved endogenous/instrument direction. This is not a weak-IV test.
    """
    if not Re.shape[1]:
        return
    if n_exog:
        control_q, _ = la.qr(Rz[:, :n_exog], mode="economic", check_finite=False)
        ep = Re - control_q @ (control_q.T @ Re)
        ip = Rz[:, n_exog:] - control_q @ (control_q.T @ Rz[:, n_exog:])
    else:
        ep, ip = Re, Rz
    bases = []
    for matrix in (ep, ip):
        scale = la.norm(matrix, axis=0)
        scale[scale == 0] = 1.
        left, singular, _ = la.svd(matrix / scale, full_matrices=False,
                                  check_finite=False)
        floor = np.finfo(float).eps * max(nobs, *matrix.shape)
        keep = singular > floor * (singular[0] if singular.size else 0.)
        bases.append(left[:, keep])
    eq, iq = bases
    if eq.shape[1] < Re.shape[1] or iq.shape[1] < Re.shape[1]:
        raise UnderidentifiedError("Partial IV roles lack an identifying direction")
    singular = la.svdvals(iq.T @ eq, check_finite=False)
    floor = np.finfo(float).eps * max(nobs, *Rz.shape)
    if singular[-1] <= floor:
        raise UnderidentifiedError("Excluded instruments do not identify all endogenous directions")
    if singular[-1] <= np.sqrt(np.finfo(float).eps):
        raise NumericalError(
            "Partial IV directions are too close to numerical non-identification",
            details={"smallest_partial_correlation": float(singular[-1])},
            suggestion="Inspect endogenous and excluded-instrument identification directions.",
        )


def _solve_2sls_stable_roles(y, X, Z, *, weights, n_exog, weighted,
                            chunk_rows=16384):
    """Solve difficult 2SLS designs through a two-level, bounded TSQR.

    The returned projected_X is None: inference replays the orthogonal factors
    once and reconstructs only the fitted endogenous columns it needs to keep.
    First-stage coefficients are reporting quantities, never projection inputs.
    """
    y = np.asarray(y, dtype=float)
    yw, Xw, Zw, _ = weighted
    Ew = Xw[:, n_exog:]
    n, l, q = len(y), Zw.shape[1], Ew.shape[1]
    width = l + q + 1
    chunk = max(width, min(chunk_rows, max(width, 1_000_000 // width)))
    zs, es = np.zeros(l), np.zeros(q)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        np.maximum(zs, np.max(np.abs(Zw[lo:hi]), axis=0), out=zs)
        np.maximum(es, np.max(np.abs(Ew[lo:hi]), axis=0), out=es)
    zs[zs == 0], es[es == 0] = 1., 1.
    local_factors = []
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        block = _stable_role_block(Zw, Ew, yw, zs, es, lo, hi)
        local_factors.append(la.qr(block, mode="raw", overwrite_a=True,
                                   check_finite=False)[1])
    block_sizes = tuple(len(factor) for factor in local_factors)
    top_q, R = la.qr(np.vstack(local_factors), mode="economic",
                     overwrite_a=True, check_finite=False)
    del local_factors
    rz = R[:, :l]
    uz, sz, vzh = la.svd(rz, full_matrices=False, check_finite=False)
    cutoff = np.finfo(float).eps * max(n, l) * sz[0]
    instrument_rank = int(np.count_nonzero(sz > cutoff))
    if instrument_rank != l:
        raise NumericalError(
            "Instrument design is numerically rank deficient in stable QR coordinates",
            details={"rank": instrument_rank, "n_instruments": l},
            suggestion="Inspect redundant or numerically inseparable instruments and controls.",
        )
    re = R[:, l:l + q] * es
    _check_partial_identification(rz, re, n_exog, n)
    rx = np.column_stack([rz[:, :n_exog] * zs[:n_exog], re])
    h = uz.T @ rx
    xscale = la.norm(h, axis=0)
    xscale[xscale == 0] = 1.
    ux, sx, vxh = la.svd(h / xscale, full_matrices=False, check_finite=False)
    cutoff = np.finfo(float).eps * max(n, X.shape[1]) * sx[0]
    rank = int(np.count_nonzero(sx > cutoff))
    if rank != X.shape[1]:
        raise UnderidentifiedError(
            "Projected design lacks an identifying coefficient direction",
            details={"rank": rank, "n_parameters": X.shape[1]},
        )
    inverse = (vxh.T / sx) / xscale[:, None]
    coefficient_map = inverse @ ux.T
    beta = coefficient_map @ (uz.T @ R[:, -1])
    bread = inverse @ inverse.T
    bread = (bread + bread.T) / 2.
    first = np.zeros((l, X.shape[1]))
    first[:n_exog, :n_exog] = np.eye(n_exog)
    first[:, n_exog:] = ((vzh.T / sz) @ (uz.T @ re)) / zs[:, None]
    fitted = X @ beta
    diagnostic = np.column_stack([re, rz[:, n_exog:] * zs[n_exog:]])
    if n_exog:
        control_q, _ = la.qr(rz[:, :n_exog], mode="economic", check_finite=False)
        diagnostic -= control_q @ (control_q.T @ diagnostic)
    factor = _StableIVQR(
        R, zs, es, uz, uz @ h, uz @ coefficient_map.T, ux,
        R[:, -1] - rx @ beta, diagnostic, top_q, block_sizes,
        Zw, Ew, yw, n_exog, chunk,
    )
    return Weighted2SLSResult(beta, y - fitted, bread, rank, fitted, None, first), factor


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
