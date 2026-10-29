from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from ..compute.vcov import _cluster_meat_xe, is_iid_vce, score_covariance


def additive_score_matrix(residual, instruments, *, weights=None) -> np.ndarray:
    """Observation-level additive IV scores ``q_i * residual_i``.

    For linear IV ``residual = y - X beta``.  For IV-PPML the same primitive
    applies with ``residual = y - mu`` and ``q`` equal to the full instrument
    matrix (included exogenous regressors plus excluded instruments).
    """
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    q = np.asarray(instruments, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    if q.shape[0] != len(r):
        raise ValueError("residual and instrument matrices must have the same number of rows")
    scores = q * r[:, None]
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(r):
            raise ValueError("weights must have the same number of rows")
        scores = scores * w[:, None]
    return scores


def additive_moments(residual, instruments, *, weights=None, normalize=True) -> np.ndarray:
    scores = additive_score_matrix(residual, instruments, weights=weights)
    total = scores.sum(axis=0)
    return total / len(scores) if normalize and len(scores) else total


def overid_test(resid, Z, *, X, kind="robust", clusters=None, center=False,
                time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None, score_scale=None,
                _xz=None, _meat=None):
    """Sargan (IID) or Hansen J statistic for the additive moments ``Z' resid``.

    ``X`` enters only through ``D = Z' X``. For a non-IID ``kind`` the moments
    first take one efficient GMM step under the fixed score covariance, so the
    statistic is chi2(L-K) at any consistent estimate, not only at efficient
    GMM. IV-PPML passes W-residualized instruments, ``w0 (y - mu)`` and ``W X``.
    """
    resid = np.asarray(resid, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    df = int(Z.shape[1] - X.shape[1])
    n_eff = float(len(resid) if effective_n is None else effective_n)
    if df <= 0:
        return {"stat": np.nan, "df": 0, "pvalue": np.nan}
    g = Z.T @ resid
    if is_iid_vce(kind):
        # Sargan uses the uncorrected residual variance, not HC1/fit DoF.
        sigma2 = float(resid @ resid) / n_eff
        S = sigma2 * (Z.T @ Z)
    elif kind == 'cluster' and not center:
        S = (_cluster_meat_xe(Z, resid, clusters, score_scale)[0]
             if _meat is None else _meat)
    else:
        S = score_covariance(
            Z * resid[:, None], kind=kind, clusters=clusters,
            k_total=0, center=center, small_sample=False,
            time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )
    W = np.linalg.pinv(S, hermitian=True)
    if not is_iid_vce(kind):
        # The reported coefficient remains 2SLS/k-class. Hansen J, however,
        # uses the efficient second-step moments under this fixed S (ivreg2).
        D = _xz.T if kind == 'cluster' and _xz is not None else Z.T @ X
        A = D.T @ W @ D
        delta = np.linalg.pinv((A + A.T) / 2, hermitian=True) @ (D.T @ W @ g)
        g = g - D @ delta
    stat = float(g.T @ W @ g)
    return {"stat": stat, "df": df, "pvalue": float(chi2.sf(stat, df))}
