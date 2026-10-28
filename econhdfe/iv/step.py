"""Bounded-workspace weighted IV steps for nonlinear outer estimators.

This is not a replacement for the public linear-IV estimator.  The fast route
uses a conditioned instrument Gram, a Cholesky whitening and a SMALL SVD; it
never forms a full weighted design or first-stage fitted design.  Uncertain
Gram rank/conditioning falls back to the established tall-design solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular
from .solve import weighted_2sls
from ..errors import UnderidentifiedError


@dataclass(slots=True)
class IVStep:
    beta: np.ndarray
    residuals: np.ndarray
    bread: np.ndarray
    first_stage: np.ndarray
    rank: int
    backend: str
    instrument_rcond: float
    slope_rcond: float


def _column_scale(A, chunk_rows):
    scale = np.zeros(A.shape[1], dtype=np.float64)
    for lo in range(0, len(A), chunk_rows):
        part = A[lo:lo + chunk_rows]
        scale = np.maximum(scale, np.maximum(np.abs(part.min(axis=0)),
                                             np.abs(part.max(axis=0))))
    return np.where(scale > 0.0, scale, 1.0)


def solve_iv_step(y, X, Z, weights, *, mode="auto", chunk_rows=65536,
                  gram_rcond=1e-8, resid_out=None) -> IVStep:
    """Solve one weighted IV system, with explicit conditioning safeguards.

    Additional fast-path workspace is O(chunk_rows*(K+L) + L**2 + L*K).
    Inputs and the returned N-vector residual are not included in this bound.
    ``stable`` deliberately calls the pre-existing scaled tall-design solver.
    """
    y, X, Z, w = (np.asarray(v, dtype=np.float64) for v in (y, X, Z, weights))
    if X.ndim != 2 or Z.ndim != 2 or y.ndim != 1 or w.shape != y.shape:
        raise ValueError("IV step expects vector y/weights and matrix X/Z")
    n, k = X.shape
    if n == 0 or Z.shape[0] != n or len(y) != n or Z.shape[1] < k or k == 0:
        raise ValueError("incompatible or underidentified IV step dimensions")
    if mode not in {"auto", "stable"}:
        raise ValueError("IV step mode must be auto/stable")
    if chunk_rows < 1 or not 0.0 < gram_rcond < 1.0:
        raise ValueError("invalid IV step chunk/rcond control")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("IV step requires finite positive weights")
    if resid_out is None:
        resid_out = np.empty(n, dtype=np.float64)

    ircond = srcond = float("nan")
    if mode == "auto":
        sx = _column_scale(X, chunk_rows)
        sz = _column_scale(Z, chunk_rows)
        wm = float(np.max(w))
        l = Z.shape[1]
        G = np.zeros((l, l)); C = np.zeros((l, k)); h = np.zeros(l)
        for lo in range(0, n, chunk_rows):
            hi = min(n, lo + chunk_rows)
            zb, xb = Z[lo:hi] / sz, X[lo:hi] / sx
            wb = w[lo:hi] / wm
            wz = zb * wb[:, None]
            G += zb.T @ wz
            C += wz.T @ xb
            h += wz.T @ y[lo:hi]
        try:
            if not (np.all(np.isfinite(G)) and np.all(np.isfinite(C))
                    and np.all(np.isfinite(h))):
                raise np.linalg.LinAlgError("non-finite sufficient statistics")
            sd = np.sqrt(np.diag(G))
            if np.any(sd <= 0):
                raise np.linalg.LinAlgError("zero weighted instrument")
            corr = G / sd[:, None] / sd[None, :]
            corr = (corr + corr.T) * .5
            eig = np.linalg.eigvalsh(corr)
            ircond = float(eig[0] / eig[-1])
            if ircond < gram_rcond:
                raise np.linalg.LinAlgError("ambiguous instrument Gram")
            L = np.linalg.cholesky(corr)
            cross = C / sd[:, None]
            K = solve_triangular(L, cross, lower=True, check_finite=False)
            rhs = solve_triangular(L, h / sd, lower=True, check_finite=False)
            U, s, Vt = np.linalg.svd(K, full_matrices=False)
            srcond = float(s[-1] / s[0]) if s[0] else 0.0
            if srcond < 1e-10:
                raise np.linalg.LinAlgError("ambiguous IV slope rank")
            gamma = Vt.T @ ((U.T @ rhs) / s)
            beta = gamma / sx
            stage = solve_triangular(L.T, K, lower=False, check_finite=False)
            first_stage = stage / sd[:, None] * sx[None, :] / sz[:, None]
            Vscaled = Vt.T / sx[:, None]
            bread = (Vscaled / s) @ (Vscaled / s).T / wm
            np.dot(X, beta, out=resid_out)
            resid_out *= -1.0; resid_out += y
            if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(resid_out))):
                raise np.linalg.LinAlgError("non-finite IV solution")
            return IVStep(beta, resid_out, bread, first_stage, k,
                          "sufficient_statistics", ircond, srcond)
        except np.linalg.LinAlgError:
            pass

    solved = weighted_2sls(y, X, Z, weights=w)
    if solved.rank < k:
        raise UnderidentifiedError("weighted IV slope system is numerically rank deficient")
    resid_out[:] = solved.residuals
    return IVStep(solved.beta, resid_out, solved.bread, solved.first_stage,
                  solved.rank, "stable_fallback" if mode == "auto" else "stable",
                  ircond, srcond)
