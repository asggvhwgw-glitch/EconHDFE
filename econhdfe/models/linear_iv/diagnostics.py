from __future__ import annotations
from itertools import combinations
import numpy as np
from ...compute.wls import as_2d
import scipy.linalg as la
from scipy.stats import chi2, f
from ...compute.encoding import factorize_interaction
from ...compute.vcov import ols_vcov, score_covariance


# Only the joint first-stage/AP call owns this cache; never persist it in a
# workspace or result. Wide systems simply recompute instead of growing it.
_FIRST_STAGE_CACHE_BYTES = 4 * 1024**2



def _residualize_small(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.shape[1] == 0 or (A.ndim == 2 and A.shape[1] == 0):
        return A.copy()
    coef, *_ = la.lstsq(B, A, cond=None, lapack_driver="gelsy")
    return A - B @ coef


def _weighted_triplet(endog, exog, instruments, weights=None):
    E = np.asarray(endog, dtype=np.float64)
    if E.ndim == 1:
        E = E[:, None]
    n = len(E)
    C = as_2d(exog, n)
    Z = as_2d(instruments, n)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
        E, C, Z = E * sw, C * sw, Z * sw
    return E, C, Z


class _IVDiagnosticWorkspace:
    """Per-fit weighted roles and shared control projection; never session state.

    Coefficients still use GELSY with the established rank policy, not normal
    equations. Only observation-space projections and the diagnostic bread are
    shared. Callers must construct a new workspace after sample/weight changes.
    """

    def __init__(self, endog, exog, instruments, weights=None):
        self.E, self.C, self.Z = _weighted_triplet(endog, exog, instruments, weights)
        # Preserve the canonical/CD projection layout and rank-boundary behavior.
        self.Ep = _residualize_small(self.E, self.C)
        self.Zp = _residualize_small(self.Z, self.C)
        self.zz = self.Zp.T @ self.Zp
        # Conservative execution guard only: this Gram rank does NOT select
        # model columns, change GELSY's cutoff, or supply a reported rank.
        self._batch_safe = (
            (self.C.shape[1] == 0 or np.linalg.matrix_rank(self.C.T @ self.C) == self.C.shape[1])
            and (self.Zp.shape[1] == 0 or np.linalg.matrix_rank(self.zz) == self.Zp.shape[1])
        )
        self._bread = None
        self._shea = None
        self._reduced_form = None

    @property
    def bread(self):
        if self._bread is None:
            self._bread = np.linalg.pinv(self.zz, hermitian=True)
        return self._bread

    @property
    def shea(self):
        if self._shea is None:
            self._shea = _shea_partial_r2(self.E, self.C, self.Z)
        return self._shea

    @property
    def reduced_form(self):
        """Common coefficients, fitted values and residuals after removing C."""
        if self._reduced_form is None:
            coef, *_ = la.lstsq(self.Zp, self.Ep, cond=None, lapack_driver="gelsy")
            fitted = self.Zp @ coef
            self._reduced_form = coef, fitted, self.Ep - fitted
        return self._reduced_form

    def tests(self, dependent=None, *, _inverse_cache=None, **inference):
        """Test several RHS with one control projection and one GELSY solve."""
        target = self.E if dependent is None else dependent
        if not self._batch_safe:
            return [_first_stage_test(target[:, j], self.C, self.Z, **inference)
                    for j in range(target.shape[1])]
        ep = self.Ep if dependent is None else _residualize_small(dependent, self.C)
        if self.Zp.shape[1]:
            if dependent is None:
                coef, _, resid = self.reduced_form
            else:
                coef, *_ = la.lstsq(self.Zp, ep, cond=None, lapack_driver="gelsy")
                resid = ep - self.Zp @ coef
        else:
            coef = np.empty((0, ep.shape[1]))
            resid = ep
        out = []
        for j in range(ep.shape[1]):
            e, r = ep[:, j], resid[:, j]
            # Near-perfect fits amplify GEMM/GEMV rounding into very different
            # F statistics. Retain the original scalar evaluation there too.
            if float(r @ r) <= np.finfo(float).eps * float(e @ e):
                out.append(_first_stage_test(target[:, j], self.C, self.Z, **inference))
            else:
                inverse = None
                if _inverse_cache is not None and dependent is None and self.Zp.shape[1]:
                    if _inverse_cache[j] is None:
                        _inverse_cache[j] = _first_stage_inverse(
                            self.Zp, r, self.bread, self.C.shape[1], **inference,
                        )
                    inverse = _inverse_cache[j]
                out.append(_first_stage_statistics(
                    e, self.Zp, coef[:, j], r, self.bread, self.C.shape[1],
                    _inverse=inverse, **inference,
                ))
        return out


def _first_stage_test(dep, C, Z, *, vce, clusters=None, df_absorbed=0, nested_adj=0,
                      time=None, panel=None, bandwidth=None, kernel="bartlett", df1_override=None,
                      effective_n=None, score_scale=None):
    """Original scalar evaluation for rank/cancellation-sensitive designs."""
    zp = _residualize_small(Z, C)
    ep = _residualize_small(np.asarray(dep)[:, None], C).ravel()
    if zp.shape[1]:
        b, *_ = la.lstsq(zp, ep, cond=None, lapack_driver="gelsy")
        resid = ep - zp @ b
        bread = np.linalg.pinv(zp.T @ zp, hermitian=True)
    else:
        b = resid = bread = None
    return _first_stage_statistics(
        ep, zp, b, resid, bread, C.shape[1], vce=vce, clusters=clusters,
        df_absorbed=df_absorbed, nested_adj=nested_adj, time=time, panel=panel,
        bandwidth=bandwidth, kernel=kernel, df1_override=df1_override,
        effective_n=effective_n, score_scale=score_scale,
    )


def _first_stage_inverse(zp, resid, bread, ccols, *, vce, clusters,
                         df_absorbed, nested_adj, time=None, panel=None,
                         bandwidth=None, kernel="bartlett", df1_override=None,
                         effective_n=None, score_scale=None):
    # df1_override affects the test, NOT its covariance. Reuse is valid only
    # for the same residual and every other inference option held fixed.
    V = ols_vcov(
        zp, resid, bread, kind=vce, clusters=clusters,
        k_total=zp.shape[1] + ccols + int(df_absorbed), nested_adj=nested_adj,
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=effective_n, score_scale=score_scale,
    )
    return np.linalg.pinv(V, hermitian=True)


def _first_stage_statistics(ep, zp, b, resid, bread, ccols, *, vce, clusters,
                            df_absorbed, nested_adj, time=None, panel=None,
                            bandwidth=None, kernel="bartlett", df1_override=None,
                            effective_n=None, score_scale=None, _inverse=None):
    n = len(ep)
    n_eff = float(n if effective_n is None else effective_n)
    q = zp.shape[1]
    if q == 0:
        return {"partial_r2": np.nan, "f_classic": np.nan, "f_classic_pvalue": np.nan,
                "f_robust": np.nan, "f_robust_chi2_pvalue": np.nan, "excluded_instruments": 0}
    tss = float(ep @ ep)
    rss = float(resid @ resid)
    r2p = 0.0 if tss <= 0 else max(0.0, min(1.0, 1.0 - rss / tss))
    df1 = int(df1_override or q)
    df2 = max(n_eff - ccols - q - int(df_absorbed), 1)
    f_classic = ((tss-rss) / max(df1, 1)) / (rss / df2) if rss > 0 else np.inf
    p_classic = float(f.sf(f_classic, df1, df2)) if np.isfinite(f_classic) else 0.0
    Vinv = _inverse if _inverse is not None else _first_stage_inverse(
        zp, resid, bread, ccols, vce=vce, clusters=clusters,
        df_absorbed=df_absorbed, nested_adj=nested_adj,
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=n_eff, score_scale=score_scale,
    )
    wald = float(b.T @ Vinv @ b)
    return {
        "partial_r2": float(r2p),
        "f_classic": float(f_classic),
        "f_classic_pvalue": float(p_classic),
        "f_robust": float(wald / max(df1, 1)),
        "f_robust_chi2_pvalue": float(chi2.sf(wald, df1)),
        "excluded_instruments": int(q),
    }


def _shea_partial_r2(E, C, Z):
    X = np.column_stack([C, E])
    Q = np.column_stack([C, Z])
    xx_inv = np.linalg.pinv(X.T @ X, hermitian=True)
    q_inv = np.linalg.pinv(Q.T @ Q, hermitian=True)
    xpzx = X.T @ Q @ q_inv @ Q.T @ X
    iv_bread = np.linalg.pinv(xpzx, hermitian=True)
    start = C.shape[1]
    num = np.diag(xx_inv)[start:]
    den = np.diag(iv_bread)[start:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.clip(out, 0.0, 1.0)


def first_stage_diagnostics(endog, exog, instruments, *, weights=None, weight_info=None, vce="robust",
                            clusters=None, df_absorbed=0, nested_adj=0, time=None, panel=None,
                            bandwidth=None, kernel="bartlett", _workspace=None, _inverse_cache=None):
    work = _workspace or _IVDiagnosticWorkspace(endog, exog, instruments, weights)
    E, C, Z = work.E, work.C, work.Z
    n_eff = float(len(E) if weight_info is None else weight_info.effective_n)
    score_scale = None
    if weight_info is not None and weight_info.kind == "fweight" and str(vce).lower().replace("-", "_") in {"robust", "hc1", "heteroskedastic"}:
        score_scale = 1.0 / np.sqrt(np.asarray(weight_info.estimation, dtype=np.float64))
    out = work.tests(
        _inverse_cache=_inverse_cache, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
        nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=n_eff, score_scale=score_scale,
    )
    for j, d in enumerate(out):
        d["shea_partial_r2"] = float(work.shea[j])
    return out


def _first_stage_suite(endog, exog, instruments, *, _workspace=None, **inference):
    """Evaluate ordinary and conditional tests with one call-local VCE cache.

    The two consumers receive the exact same inference options. No array-id
    keys, global state, or cache surviving a changed sample/weight/VCE exists.
    Sensitive scalar fallbacks never write reusable inverses into the cache.
    """
    work = _workspace or _IVDiagnosticWorkspace(
        endog, exog, instruments, inference.get("weights"),
    )
    k, q = work.E.shape[1], work.Z.shape[1]
    cache = [None] * k if k * q * q * 8 <= _FIRST_STAGE_CACHE_BYTES else None
    first = first_stage_diagnostics(
        endog, exog, instruments, _workspace=work, _inverse_cache=cache, **inference,
    )
    conditional = sanderson_windmeijer_diagnostics(
        endog, exog, instruments, _workspace=work, _inverse_cache=cache, **inference,
    )
    return first, conditional


def _conditional_from_reduced_form(work, *, _inverse_cache=None, **inference):
    """AP/SW are contrasts of the SAME reduced form, not new tall regressions.

    Return None on rank/cancellation-sensitive systems so the original scalar
    definition remains the numerical fallback. The QR is orthogonal row
    compression, not a replacement of GELSY with normal equations.
    """
    E, Z = work.Ep, work.Zp
    n, k = E.shape
    if not work._batch_safe or n <= work.C.shape[1] + Z.shape[1]:
        return None
    coef, fitted, residual = work.reduced_form
    eps = np.finfo(float).eps
    if np.any(np.sum(residual ** 2, axis=0) <= eps * np.sum(E ** 2, axis=0)):
        return None
    # Guard the original [C,Z] scale as well as the conditional fitted span.
    # This 1e3 condition cap only selects execution; it is NOT a rank cutoff.
    R = la.qr(np.column_stack([work.C, work.Z]), mode="raw")[1]
    singular = la.svdvals(R)
    if singular[-1] <= 1e-3 * singular[0]:
        return None
    fitted_small = R[work.C.shape[1]:, work.C.shape[1]:] @ coef
    singular = la.svdvals(fitted_small)
    if len(singular) < k or singular[-1] <= 1e-3 * singular[0]:
        return None
    out = []
    for j in range(k):
        keep = np.arange(k) != j
        b, *_ = la.lstsq(fitted_small[:, keep], fitted_small[:, j],
                         cond=None, lapack_driver="gelsy")
        contrast = np.empty(k)
        contrast[j], contrast[keep] = 1.0, -b
        coefficient = coef @ contrast
        ap_residual = residual[:, j]
        ap = ap_residual + fitted @ contrast
        sw, sw_residual = E @ contrast, residual @ contrast
        if float(sw_residual @ sw_residual) <= eps * float(sw @ sw):
            return None
        inverse = None if _inverse_cache is None else _inverse_cache[j]
        ap_test = _first_stage_statistics(ap, Z, coefficient, ap_residual,
                                          work.bread, work.C.shape[1],
                                          _inverse=inverse, **inference)
        sw_test = _first_stage_statistics(sw, Z, coefficient, sw_residual,
                                          work.bread, work.C.shape[1], **inference)
        out.append({"ap": ap_test, "sw": sw_test, "df1": Z.shape[1] - k + 1})
    return out


def sanderson_windmeijer_diagnostics(endog, exog, instruments, *, weights=None, weight_info=None,
                                     vce="robust", clusters=None, df_absorbed=0,
                                     nested_adj=0, time=None, panel=None, bandwidth=None,
                                     kernel="bartlett", _workspace=None, _inverse_cache=None):
    """Angrist-Pischke and Sanderson-Windmeijer conditional first-stage tests.

    For one endogenous regressor these reduce to the ordinary excluded-IV test.
    For multiple endogenous regressors the numerator df is L1-K1+1.
    """
    work = _workspace or _IVDiagnosticWorkspace(endog, exog, instruments, weights)
    E, C, Z = work.E, work.C, work.Z
    n, k1 = E.shape
    n_eff = float(n if weight_info is None else weight_info.effective_n)
    score_scale = None
    if weight_info is not None and weight_info.kind == "fweight" and str(vce).lower().replace("-", "_") in {"robust", "hc1", "heteroskedastic"}:
        score_scale = 1.0 / np.sqrt(np.asarray(weight_info.estimation, dtype=np.float64))
    l1 = Z.shape[1]
    df1 = l1 - k1 + 1
    if df1 <= 0:
        return []
    if k1 == 1:
        base = work.tests(
            _inverse_cache=_inverse_cache, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
            nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )[0]
        base["shea_partial_r2"] = float(work.shea[0])
        return [{"ap": dict(base), "sw": dict(base), "df1": df1}]

    fast = _conditional_from_reduced_form(
        work, _inverse_cache=_inverse_cache, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
        nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        df1_override=df1, effective_n=n_eff, score_scale=score_scale,
    )
    if fast is not None:
        return fast

    Q = np.column_stack([C, Z])
    coef, *_ = la.lstsq(Q, E, cond=None, lapack_driver="gelsy")
    Ehat = Q @ coef
    X = np.column_stack([C, E])
    Xhat = np.column_stack([C, Ehat])
    out = []
    # At most eight dependent columns per batch: do not build an N x (2*K1)
    # AP/SW temporary for wide models. Each batch shares C and Zp factorizations.
    for lo in range(0, k1, 4):
        hi = min(k1, lo + 4)
        dependent = np.empty((n, 2 * (hi - lo)))
        for j in range(lo, hi):
            target_col = C.shape[1] + j
            keep = np.ones(X.shape[1], dtype=bool)
            keep[target_col] = False
            Xm = X[:, keep]
            Xhm = Xhat[:, keep]
            b1, *_ = la.lstsq(Xhm, E[:, j], cond=None, lapack_driver="gelsy")
            dependent[:, 2 * (j - lo)] = E[:, j] - Xhm @ b1
            dependent[:, 2 * (j - lo) + 1] = E[:, j] - Xm @ b1
        tests = work.tests(
            dependent, vce=vce, clusters=clusters, df_absorbed=df_absorbed,
            nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            df1_override=df1, effective_n=n_eff, score_scale=score_scale,
        )
        out.extend(
            {"ap": tests[2 * j], "sw": tests[2 * j + 1], "df1": int(df1)}
            for j in range(hi - lo)
        )
    return out


def _sym_sqrt(a):
    vals, vecs = np.linalg.eigh((a + a.T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _canonical_components(endog, exog, instruments, weights=None, effective_n=None, _workspace=None):
    work = _workspace or _IVDiagnosticWorkspace(endog, exog, instruments, weights)
    E, C, Z = work.E, work.C, work.Z
    n = len(E)
    n_eff = float(n if effective_n is None else effective_n)
    Ep, Zp = work.Ep, work.Zp
    qzz = work.zz / n_eff
    qzy = (Zp.T @ Ep) / n_eff
    qyy = (Ep.T @ Ep) / n_eff
    try:
        rzz = la.cholesky(qzz, lower=False, check_finite=False)
        ryy = la.cholesky(qyy, lower=False, check_finite=False)
        irzz = la.solve_triangular(rzz, np.eye(rzz.shape[0]), lower=False, check_finite=False)
        iryy = la.solve_triangular(ryy, np.eye(ryy.shape[0]), lower=False, check_finite=False)
    except la.LinAlgError:
        def root_and_invroot(a):
            vals, vecs = np.linalg.eigh((a+a.T)/2)
            tol = np.finfo(float).eps * max(a.shape) * max(float(np.max(np.abs(vals))), 1.0)
            if np.sum(vals > tol) < a.shape[0]:
                raise np.linalg.LinAlgError("rank-deficient canonical-correlation matrix")
            root = (vecs * np.sqrt(vals)) @ vecs.T
            invroot = (vecs / np.sqrt(vals)) @ vecs.T
            return root, invroot
        rzz, irzz = root_and_invroot(qzz)
        ryy, iryy = root_and_invroot(qyy)
    pihat = np.linalg.pinv(qzz, hermitian=True) @ qzy
    theta = rzz @ pihat @ iryy
    return Ep, Zp, pihat, theta, irzz, iryy, C.shape[1], n_eff


def _kron_scores(V, Z, block_rows=100_000):
    n, k1 = V.shape
    l1 = Z.shape[1]
    p = k1 * l1
    out = np.empty((n, p), dtype=np.float64)
    for lo in range(0, n, block_rows):
        hi = min(n, lo + block_rows)
        # Endogenous-major order equals column-major vec(Z'V).
        out[lo:hi] = (V[lo:hi, :, None] * Z[lo:hi, None, :]).reshape(hi-lo, p)
    return out


def _kp_factors(theta):
    l1, k1 = theta.shape
    U, _, Vh = np.linalg.svd(theta, full_matrices=True)
    V = Vh.T
    kk = k1 - 1
    if kk == 0:
        u12 = np.empty((l1, 0)); u22 = U
        v12 = np.empty((k1, 0)); v22 = V
    else:
        u12 = U[:kk, kk:]
        u22 = U[kk:, kk:]
        v12 = V[:kk, kk:]
        v22 = V[kk:, kk:]
    # Stack the source partitions to reconstruct full row dimension.
    ufull = np.vstack([u12, u22]) if kk else u22
    vfull = np.vstack([v12, v22]) if kk else v22
    us = _sym_sqrt(u22 @ u22.T)
    vs = _sym_sqrt(v22 @ v22.T)
    aq = ufull @ np.linalg.pinv(u22) @ us
    bq = vs @ np.linalg.pinv(v22.T) @ vfull.T
    return aq, bq


def _kp_stat(theta, kpvar, n):
    aq, bq = _kp_factors(theta)
    K = np.kron(bq, aq.T)
    vect = theta.reshape(-1, order="F")
    lam = K @ vect
    vlam = K @ kpvar @ K.T
    rank = int(np.linalg.matrix_rank(vlam))
    stat = float(n * lam.T @ np.linalg.pinv(vlam, hermitian=True) @ lam)
    return stat, rank


def kleibergen_paap_stats(endog, exog, instruments, *, weights=None, weight_info=None, vce="robust",
                           clusters=None, df_absorbed=0, nested_adj=0, time=None, panel=None,
                           bandwidth=None, kernel="bartlett", _workspace=None):
    """Kleibergen-Paap rk LM and rk Wald statistics.

    The covariance of the reduced-form Kronecker scores uses the same
    robust/cluster/HAC/Driscoll-Kraay engine as coefficient inference.
    """
    effective_n = None if weight_info is None else weight_info.effective_n
    Ep, Zp, pihat, theta, irzz, iryy, ccols, n_eff = _canonical_components(
        endog, exog, instruments, weights, effective_n=effective_n, _workspace=_workspace
    )
    n, k1 = Ep.shape
    score_scale = None
    if weight_info is not None and weight_info.kind == "fweight" and str(vce).lower().replace("-", "_") in {"robust", "hc1", "heteroskedastic"}:
        score_scale = 1.0 / np.sqrt(np.asarray(weight_info.estimation, dtype=np.float64))
    l1 = Zp.shape[1]
    if l1 < k1:
        return {"rk_lm": np.nan, "rk_lm_df": 0, "rk_lm_pvalue": np.nan,
                "rk_wald_chi2": np.nan, "rk_wald_df": 0, "rk_wald_f": np.nan}
    kind = (vce or "robust").lower().replace("-", "_")
    supported = {
        "robust", "hc1", "heteroskedastic", "cluster",
        "iid", "unadjusted", "homoskedastic",
        "hac", "newey_west", "neweywest",
        "dkraay", "driscoll_kraay", "dk",
    }
    if kind not in supported:
        raise ValueError(f"unsupported KP covariance kind: {vce}")
    if kind in {"dkraay", "driscoll_kraay", "dk"} and time is None:
        raise ValueError("Driscoll-Kraay Kleibergen-Paap statistic requires time=")
    aq, bq = _kp_factors(theta)
    K = np.kron(bq, aq.T)
    lam = K @ theta.reshape(-1, order="F")
    # A single endogenous variable has no score-dimension reduction. Sensitive
    # whitening keeps the established full-score multiplication order as well.
    compressed = k1 > 1 and np.linalg.cond(irzz) * np.linalg.cond(iryy) < 1e3
    projected_z = Zp @ (irzz @ aq) if compressed else None
    v_loading = iryy @ bq.T if compressed else None

    def stat_for(Vhat, *, full=False):
        reduced = compressed and not full
        scores = _kron_scores(Vhat @ v_loading, projected_z) if reduced else _kron_scores(Vhat, Zp)
        if kind in {"iid", "unadjusted", "homoskedastic"}:
            shat = (scores.T @ scores) / n_eff
        else:
            # Keep ranktest's Bartlett convention and the existing raw-score
            # engine, including multiway inclusion-exclusion BEFORE any repair.
            kp_kernel = "bartlett" if kind in {
                "hac", "newey_west", "neweywest",
                "dkraay", "driscoll_kraay", "dk"
            } else kernel
            shat = score_covariance(
                scores, kind=kind, clusters=clusters, k_total=0,
                small_sample=False, time=time, panel=panel, bandwidth=bandwidth, kernel=kp_kernel,
                effective_n=n_eff, score_scale=score_scale,
            ) / n_eff
        if not reduced:
            transform = np.kron(iryy.T, irzz.T)
            return _kp_stat(theta, transform @ shat @ transform.T, n_eff)
        singular = la.svdvals(shat)
        if singular[-1] <= 1e-10 * singular[0]:
            # Rank-boundary rounding is not an optimization opportunity.
            return stat_for(Vhat, full=True)
        rank = int(np.linalg.matrix_rank(shat))
        stat = float(n_eff * lam.T @ np.linalg.pinv(shat, hermitian=True) @ lam)
        return stat, rank

    lm, lm_rank = stat_for(Ep)
    Vrf = Ep - Zp @ pihat
    near_perfect = np.any(np.sum(Vrf ** 2, axis=0) <= np.finfo(float).eps * np.sum(Ep ** 2, axis=0))
    wald, wald_rank = stat_for(Vrf, full=near_perfect)
    df = max(l1 - k1 + 1, 1)
    if kind == "cluster" and clusters:
        g = min(len(np.unique(c)) for c in clusters)
        fscale = ((n_eff - (ccols + l1) - int(df_absorbed)) / max(n_eff - 1, 1)) * (max(g - 1, 1) / g)
        wald_f = (wald / max(l1, 1)) * fscale
    else:
        fscale = (n_eff - (ccols + l1) - int(df_absorbed)) / max(n_eff, 1)
        wald_f = (wald / max(l1, 1)) * fscale
    return {
        "rk_lm": float(lm), "rk_lm_df": int(df), "rk_lm_pvalue": float(chi2.sf(lm, df)),
        "rk_wald_chi2": float(wald), "rk_wald_df": int(df), "rk_wald_f": float(wald_f),
        "rank_lm_cov": int(lm_rank), "rank_wald_cov": int(wald_rank),
    }


def cragg_donald_stat(endog, exog, instruments, *, weights=None, df_absorbed=0, effective_n=None, _workspace=None):
    work = _workspace or _IVDiagnosticWorkspace(endog, exog, instruments, weights)
    E, C, Z = work.E, work.C, work.Z
    n = E.shape[0]
    n_eff = float(n if effective_n is None else effective_n)
    Ep, Zp = work.Ep, work.Zp
    if Zp.shape[1] == 0:
        return np.nan
    zz_inv = work.bread
    epz = Ep.T @ Zp
    explained = epz @ zz_inv @ epz.T
    residual = Ep.T @ Ep - explained
    try:
        eig = la.eigvalsh(explained, residual, check_finite=False)
        cdev = float(np.min(eig))
    except la.LinAlgError:
        cdev = float(np.min(np.linalg.eigvals(np.linalg.pinv(residual) @ explained).real))
    L = C.shape[1] + Zp.shape[1]
    L1 = Zp.shape[1]
    return float(max(n_eff - L - int(df_absorbed), 0) / max(L1, 1) * cdev)


def overid_test(resid, Z, *, X_cols: int, kind="robust", clusters=None, center=False,
                time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None, score_scale=None):
    resid = np.asarray(resid, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    df = int(Z.shape[1] - X_cols)
    n_eff = float(len(resid) if effective_n is None else effective_n)
    if df <= 0:
        return {"stat": np.nan, "df": 0, "pvalue": np.nan}
    g = Z.T @ resid
    if kind in (None, "iid", "unadjusted", "homoskedastic"):
        sigma2 = float(resid @ resid) / max(n_eff - X_cols, 1)
        S = sigma2 * (Z.T @ Z)
    else:
        S = score_covariance(
            Z * resid[:, None], kind=kind, clusters=clusters,
            k_total=0, center=center, small_sample=False,
            time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )
    stat = float(g.T @ np.linalg.pinv(S, hermitian=True) @ g)
    return {"stat": stat, "df": df, "pvalue": float(chi2.sf(stat, df))}
