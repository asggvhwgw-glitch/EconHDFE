"""One original-target audit per accepted column; no weakened tolerances.

Pure categorical coefficients only. Normal equations and explicit native fit
reconstruction are computed in the same compiled pass. Group masses are reused;
no claim is made that this is a general forward-error certificate. No fastmath.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _audit(codes, w, y, r, scales, offsets, masses, coef, check_fit):
    n, k = y.shape
    q = codes.shape[1]
    normal = np.zeros(k)
    feasible = np.zeros(k)
    # O(sum K_q), not O(N*k), scratch. Every row contributes to every FE.
    sums = np.empty(len(masses))
    for j in range(k):
        sums[:] = 0.
        den = 0.
        ferr = 0.
        for i in range(n):
            yy = y[i, j] / scales[j]
            rr = r[i, j] / scales[j]
            if not np.isfinite(yy) or not np.isfinite(rr):
                normal[j] = np.inf
                feasible[j] = np.inf
                break
            wr = w[i] * rr
            fit = 0.
            for a in range(q):
                idx = offsets[a] + codes[i, a]
                sums[idx] += wr
                if check_fit:
                    fit += coef[idx, j]
            den += w[i] * yy * yy
            if check_fit:
                diff = yy - rr - fit
                ferr += w[i] * diff * diff
        if not np.isfinite(normal[j]):
            continue
        ne = 0.
        for a in range(len(masses)):
            ne += sums[a] * sums[a] / masses[a]
        # Scaled y has a unit entry unless zero. This tiny denominator only
        # handles zero RHS; nonzero residual on a zero RHS fails the audit.
        den = max(den, np.finfo(np.float64).tiny)
        normal[j] = np.sqrt(ne / den)
        feasible[j] = np.sqrt(ferr / den) if check_fit else 0.
    return normal, feasible


def column_scales(y):
    scales = np.max(np.abs(y), axis=0)
    scales[scales == 0.] = 1.
    return scales


def audit_columns(codes, w, y, r, offsets, masses, *, coef=None, scales=None):
    """Return normalized normal residual and native reconstruction error."""
    if scales is None:
        scales = column_scales(y)
    check = coef is not None
    if coef is None:
        coef = np.empty((0, 0), dtype=float)
    normal, feasible = _audit(codes, w, y, r, scales, offsets, masses, coef, check)
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(feasible)):
        raise FloatingPointError('nonfinite original-target audit')
    return normal, feasible
