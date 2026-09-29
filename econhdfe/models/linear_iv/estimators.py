from __future__ import annotations
import numpy as np
import scipy.linalg as la
from scipy.stats import chi2
from ...compute.vcov import kclass_vcov, gmm_vcov, score_covariance
from ...compute.vcov import block_cluster_meat_xe, fix_psd
from ...compute.weights import robust_score_scale
from ...compute.wls import weighted_arrays
from ...compute.block_design import BlockDesign, hstack_block_designs
from ...iv.design import IVDesign
from ...iv.solve import weighted_2sls
from .stock_yogo import stock_yogo_critical_values
from .diagnostics import (
    _IVDiagnosticWorkspace, _first_stage_suite, cragg_donald_stat, overid_test,
    kleibergen_paap_stats,
)

def _equilibrate_iv_roles(C, E, I):
    """Change units, not the IV estimator or the instrument column space."""
    arrays, scales = [], []
    for A in (C, E, I):
        if isinstance(A, BlockDesign):
            A._ensure_layout()
            scale = np.zeros(A.ncols)
            for block in A.blocks:
                if len(block.rows) and len(block.columns):
                    np.maximum.at(scale, block.columns, np.max(np.abs(block.values), axis=0))
            scale[scale == 0] = 1.0
            arrays.append(A.scale_columns(scale))
        else:
            scale = np.max(np.abs(A), axis=0) if len(A) else np.ones(A.shape[1])
            scale[scale == 0] = 1.0
            arrays.append(A / scale)
        scales.append(scale)
    cs, es, ins = scales
    return (*arrays, np.r_[cs, es], np.r_[cs, ins], es)


def _restore_iv_units(beta, V, first, meta, xscale, zscale, escale):
    first["coefficients"] = first["coefficients"] * xscale[None, :] / zscale[:, None]
    first["fitted_endog"] = first["fitted_endog"] * escale[None, :]
    if "first_step_params" in meta:
        meta["first_step_params"] = meta["first_step_params"] / xscale
    if "weight_matrix" in meta:
        meta["weight_matrix"] = meta["weight_matrix"] / zscale[:, None] / zscale[None, :]
    return beta / xscale, V / xscale[:, None] / xscale[None, :]


def _liml_kappa(yw: np.ndarray, Cw: np.ndarray, Ew: np.ndarray, Zw: np.ndarray) -> float:
    """Compute LIML kappa using only small cross-product matrices."""
    E = np.column_stack([yw, Ew])
    # The generalized eigenvalue is invariant to nonsingular congruence.
    # Balance outcome as well as endogenous columns before whitening; otherwise
    # changing y units can silently discard an eigen-direction and select k=0.
    scales = np.max(np.abs(E), axis=0)
    scales[scales == 0] = 1.0
    E = E / scales
    ete = E.T @ E
    zz = Zw.T @ Zw
    zze = Zw.T @ E
    mz = ete - zze.T @ np.linalg.pinv(zz, hermitian=True) @ zze
    if Cw.shape[1]:
        cc = Cw.T @ Cw
        cce = Cw.T @ E
        mx1 = ete - cce.T @ np.linalg.pinv(cc, hermitian=True) @ cce
    else:
        mx1 = ete
    # Equivalent to eigvalsh(inv_sqrth(mz) @ mx1 @ inv_sqrth(mz)), but
    # robust to semidefinite mz via a pseudo inverse square root.
    vals, vecs = np.linalg.eigh((mz + mz.T) / 2)
    tol = np.finfo(float).eps * max(mz.shape) * max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
    keep = vals > tol
    if not np.any(keep):
        raise np.linalg.LinAlgError("LIML residual cross-product matrix is singular")
    invroot = (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T
    q = invroot @ ((mx1 + mx1.T) / 2) @ invroot
    return float(np.min(np.linalg.eigvalsh((q + q.T) / 2)))


def _kclass_core(yw, Xw, Zw, kappa: float):
    zz_inv = np.linalg.pinv(Zw.T @ Zw, hermitian=True)
    xz = Xw.T @ Zw
    zty = Zw.T @ yw
    xpzx = xz @ zz_inv @ xz.T
    xpzy = xz @ zz_inv @ zty
    xx = Xw.T @ Xw
    xy = Xw.T @ yw
    A = (1.0 - kappa) * xx + kappa * xpzx
    rhs = (1.0 - kappa) * xy + kappa * xpzy
    bread = np.linalg.pinv((A + A.T) / 2, hermitian=True)
    beta = bread @ rhs
    return beta, bread, int(np.linalg.matrix_rank(A)), zz_inv


def _first_stage_projection(Cw, Ew, Iw):
    Zw = np.column_stack([Cw, Iw])
    Xw = np.column_stack([Cw, Ew])
    coef, *_ = la.lstsq(Zw, Xw, cond=None, lapack_driver="gelsy")
    xhat = Zw @ coef
    return coef, xhat


def _block_liml_kappa(y, C: BlockDesign, E: BlockDesign, Z: BlockDesign, w) -> float:
    y_scale = float(np.max(np.abs(y))) if len(y) else 1.0
    y = y / (y_scale if y_scale else 1.0)
    wy = w * y
    ete = np.empty((1 + E.ncols, 1 + E.ncols), dtype=np.float64)
    ete[0, 0] = float(y @ wy)
    yE = E.t_matvec(wy)
    ete[0, 1:] = yE; ete[1:, 0] = yE
    ete[1:, 1:] = E.gram(weights=w)
    zz = Z.gram(weights=w)
    zze = np.column_stack([Z.t_matvec(wy), Z.cross_gram(E, weights=w)])
    mz = ete - zze.T @ np.linalg.pinv(zz, hermitian=True) @ zze
    if C.ncols:
        cc = C.gram(weights=w)
        cce = np.column_stack([C.t_matvec(wy), C.cross_gram(E, weights=w)])
        mx1 = ete - cce.T @ np.linalg.pinv(cc, hermitian=True) @ cce
    else:
        mx1 = ete
    vals, vecs = np.linalg.eigh((mz + mz.T) / 2)
    tol = np.finfo(float).eps * max(mz.shape) * max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
    keep = vals > tol
    if not np.any(keep):
        raise np.linalg.LinAlgError("LIML residual cross-product matrix is singular")
    invroot = (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T
    q = invroot @ ((mx1 + mx1.T) / 2) @ invroot
    return float(np.min(np.linalg.eigvalsh((q + q.T) / 2)))


def _block_kclass_vcov(X: BlockDesign, Z: BlockDesign, resid, w, bread, *, kind,
                       clusters, k_total, nested_adj, effective_n, score_scale):
    kind0 = "iid" if kind is None else str(kind).lower().replace("-", "_")
    resid = np.asarray(resid, dtype=np.float64)
    gamma = np.linalg.pinv(Z.gram(weights=w), hermitian=True) @ Z.cross_gram(X, weights=w)
    if kind0 in {"iid", "unadjusted", "homoskedastic"}:
        sigma2 = float(np.dot(w * resid, resid)) / max(float(effective_n) - int(k_total), 1)
        return bread * sigma2
    score = w * resid
    if kind0 in {"robust", "hc1", "heteroskedastic"}:
        base = score if score_scale is None else score * np.asarray(score_scale, dtype=np.float64)
        Sz = Z.gram(weights=base * base)
        Sz *= float(effective_n) / max(float(effective_n) - int(k_total), 1)
    elif kind0 == "cluster":
        Sz, g = block_cluster_meat_xe(Z, score, clusters, score_scale=score_scale)
        from ...compute.vcov import _cluster_scale
        Sz *= _cluster_scale(float(effective_n), int(k_total), g, nested_adj=int(bool(nested_adj)))
    else:
        raise ValueError(f"partitioned IV covariance does not support kind: {kind0}")
    meat = gamma.T @ Sz @ gamma
    V = bread @ meat @ bread
    V = (V + V.T) / 2
    return fix_psd(V) if kind0 == "cluster" and clusters and len(clusters) > 1 else V


def fit_iv_kclass_block(
    y, exog: BlockDesign, endog: BlockDesign, instruments: BlockDesign, *,
    weights=None, weight_info=None, vce="iid", clusters=None, df_absorbed=0,
    nested_adj=0, estimator="2sls", kappa=None, fuller=0.0,
):
    """Block-native linear IV core for common robust/cluster specifications.

    Heavy weak-IV diagnostics still reuse the established dense implementation
    after the coefficient/inference solve; this keeps statistical compatibility
    while moving design construction, FE projection, cross-products and the
    main estimator off the global N x K representation.
    """
    y = np.asarray(y, dtype=np.float64)
    C, E, I, xscale, zscale, escale = _equilibrate_iv_roles(exog, endog, instruments)
    X = hstack_block_designs(C, E)
    Z = hstack_block_designs(C, I)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    XX = X.gram(weights=w); ZZ = Z.gram(weights=w); XZ = X.cross_gram(Z, weights=w)
    Xy = X.t_matvec(w * y); Zy = Z.t_matvec(w * y)
    zz_inv = np.linalg.pinv(ZZ, hermitian=True)
    xpzx = XZ @ zz_inv @ XZ.T
    xpzy = XZ @ zz_inv @ Zy

    est = str(estimator).lower()
    liml_kappa = None
    if est in {"2sls", "iv"}:
        est_kappa = 1.0
    elif est == "liml":
        liml_kappa = _block_liml_kappa(y, C, E, Z, w)
        est_kappa = liml_kappa if kappa is None else float(kappa)
    elif est == "kclass":
        if kappa is None:
            raise ValueError("estimator='kclass' requires kappa")
        est_kappa = float(kappa)
    else:
        raise ValueError("block k-class estimator must be '2sls', 'liml', or 'kclass'")
    if fuller:
        est_kappa -= float(fuller) / max(len(y) - Z.ncols, 1)
    A = (1.0 - est_kappa) * XX + est_kappa * xpzx
    rhs = (1.0 - est_kappa) * Xy + est_kappa * xpzy
    bread = np.linalg.pinv((A + A.T) / 2, hermitian=True)
    beta = bread @ rhs
    rank = int(np.linalg.matrix_rank(A))
    resid = y - X.matvec(beta)
    n_eff = float(len(y) if weight_info is None else weight_info.effective_n)
    score_scale = None if weight_info is None else robust_score_scale(weight_info, vce, len(y))
    k_total = rank + int(df_absorbed)
    V = _block_kclass_vcov(
        X, Z, resid, w, bread, kind=vce, clusters=clusters, k_total=k_total,
        nested_adj=nested_adj, effective_n=n_eff, score_scale=score_scale,
    )

    first_stage = zz_inv @ Z.cross_gram(X, weights=w)
    fitted_endog = Z.matmat(first_stage[:, C.ncols:]) if E.ncols else np.empty((len(y), 0))

    # Compatibility diagnostics use the existing, extensively validated code.
    Cd, Ed, Id = C.materialize(), E.materialize(), I.materialize()
    workspace = _IVDiagnosticWorkspace(Ed, Cd, Id, weights)
    first_diagnostics, conditional_diagnostics = _first_stage_suite(
        Ed, Cd, Id, weights=weights, weight_info=weight_info, vce=vce, clusters=clusters,
        _workspace=workspace,
        df_absorbed=df_absorbed, nested_adj=int(bool(nested_adj)),
    )
    first = {
        "coefficients": first_stage,
        "fitted_endog": fitted_endog,
        "diagnostics": first_diagnostics,
    }
    diagnostics = {
        "cragg_donald_f": cragg_donald_stat(
            Ed, Cd, Id, weights=weights, df_absorbed=df_absorbed, effective_n=n_eff,
            _workspace=workspace,
        ),
        "stock_yogo": stock_yogo_critical_values(I.ncols, E.ncols, estimator=est),
        "kleibergen_paap": kleibergen_paap_stats(
            Ed, Cd, Id, weights=weights, weight_info=weight_info, vce=vce, clusters=clusters,
            _workspace=workspace,
            df_absorbed=df_absorbed, nested_adj=int(bool(nested_adj)),
        ),
        "sanderson_windmeijer": conditional_diagnostics,
        "overidentification": overid_test(
            resid * np.sqrt(w), Z.materialize() * np.sqrt(w)[:, None], X_cols=X.ncols,
            kind=vce, clusters=clusters, effective_n=n_eff, score_scale=score_scale,
        ),
        "heterogeneous_spec_path": True,
        "heterogeneous_spec_diagnostics_dense_materialization": True,
    }
    meta = {"estimator": est, "kappa": float(est_kappa), "liml_kappa": liml_kappa, "fuller": float(fuller)}
    df_resid = max(n_eff - k_total, 0.0)
    if str(vce).lower().replace("-", "_") == "cluster" and clusters:
        df_resid = min(df_resid, float(min(len(np.unique(c)) for c in clusters) - 1))
    beta, V = _restore_iv_units(beta, V, first, meta, xscale, zscale, escale)
    return beta, V, resid, rank, df_resid, first, diagnostics, meta


def fit_iv_kclass(
    y,
    exog,
    endog,
    instruments,
    *,
    weights=None,
    weight_info=None,
    vce="iid",
    clusters=None,
    df_absorbed=0,
    nested_adj=0,
    estimator="2sls",
    kappa=None,
    fuller=0.0,
    time=None,
    panel=None,
    bandwidth=None,
    kernel="bartlett",
):
    y = np.asarray(y, dtype=np.float64)
    design = IVDesign.from_arrays(exog, endog, instruments, len(y))
    C, E, I, xscale, zscale, escale = _equilibrate_iv_roles(design.exog, design.endog, design.excluded)
    X, Z = np.column_stack([C, E]), np.column_stack([C, I])
    yw, Xw, Zw, sw = weighted_arrays(y, X, Z, weights)
    Cw = Xw[:, :C.shape[1]]; Ew = Xw[:, C.shape[1]:]; Iw = Zw[:, C.shape[1]:]

    estimator = estimator.lower()
    liml_kappa = None
    if estimator in {"2sls", "iv"}:
        est_kappa = 1.0
    elif estimator == "liml":
        liml_kappa = _liml_kappa(yw, Cw, Ew, Zw)
        est_kappa = liml_kappa if kappa is None else float(kappa)
    elif estimator == "kclass":
        if kappa is None:
            raise ValueError("estimator='kclass' requires kappa")
        est_kappa = float(kappa)
    else:
        raise ValueError("k-class estimator must be '2sls', 'liml', or 'kclass'")
    if fuller:
        est_kappa -= float(fuller) / max(len(y) - Zw.shape[1], 1)

    if estimator in {"2sls", "iv"} and not fuller and kappa is None:
        solved = weighted_2sls(y, X, Z, weights=weights)
        beta, bread, rank = solved.beta, solved.bread, solved.rank
    else:
        beta, bread, rank, _ = _kclass_core(yw, Xw, Zw, est_kappa)
    resid = y - X @ beta
    ew = resid if sw is None else resid * sw
    k_total = rank + int(df_absorbed)
    n_eff = float(len(y) if weight_info is None else weight_info.effective_n)
    score_scale = None if weight_info is None else robust_score_scale(weight_info, vce, len(y))
    V = kclass_vcov(
        Xw, Zw, ew, bread, kind=vce, clusters=clusters,
        k_total=k_total, nested_adj=int(bool(nested_adj)),
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=n_eff, score_scale=score_scale,
    )
    pi_x, xhat = _first_stage_projection(Cw, Ew, Iw)
    workspace = _IVDiagnosticWorkspace(E, C, I, weights)
    first_diagnostics, conditional_diagnostics = _first_stage_suite(
        E, C, I, weights=weights, weight_info=weight_info, vce=vce, clusters=clusters,
        _workspace=workspace,
        df_absorbed=df_absorbed, nested_adj=int(bool(nested_adj)),
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
    )
    first = {
        "coefficients": pi_x,
        "fitted_endog": xhat[:, C.shape[1]:],
        "diagnostics": first_diagnostics,
    }
    diagnostics = {
        "cragg_donald_f": cragg_donald_stat(
            E, C, I, weights=weights, df_absorbed=df_absorbed, effective_n=n_eff,
            _workspace=workspace,
        ),
        "stock_yogo": stock_yogo_critical_values(I.shape[1], E.shape[1], estimator=estimator),
        "kleibergen_paap": kleibergen_paap_stats(
            E, C, I, weights=weights, weight_info=weight_info, vce=vce, clusters=clusters,
            _workspace=workspace,
            df_absorbed=df_absorbed, nested_adj=int(bool(nested_adj)),
            time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        ),
        "sanderson_windmeijer": conditional_diagnostics,
        "overidentification": overid_test(
            ew, Zw, X_cols=Xw.shape[1], kind=vce, clusters=clusters,
            time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        ),
    }
    meta = {"estimator": estimator, "kappa": float(est_kappa), "liml_kappa": liml_kappa, "fuller": float(fuller)}
    df_resid = max(n_eff-k_total, 0.0)
    if str(vce).lower().replace("-", "_") == "cluster" and clusters:
        df_resid = min(df_resid, float(min(len(np.unique(c)) for c in clusters) - 1))
    beta, V = _restore_iv_units(beta, V, first, meta, xscale, zscale, escale)
    return beta, V, resid, rank, df_resid, first, diagnostics, meta


def _gmm_weight_matrix(Zw, ew, *, kind, clusters=None, center=False, time=None, panel=None,
                       bandwidth=None, kernel="bartlett", effective_n=None, score_scale=None):
    n = len(ew)
    if kind in (None, "iid", "unadjusted", "homoskedastic"):
        e0 = ew - ew.mean()
        sigma2 = float(e0 @ e0) / float(n if effective_n is None else effective_n)
        return sigma2 * (Zw.T @ Zw)
    return score_covariance(
        Zw * ew[:, None], kind=kind, clusters=clusters,
        k_total=0, center=center, small_sample=False,
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=effective_n, score_scale=score_scale,
    )


def fit_iv_gmm2s(
    y,
    exog,
    endog,
    instruments,
    *,
    weights=None,
    weight_info=None,
    vce="robust",
    clusters=None,
    df_absorbed=0,
    nested_adj=0,
    center=False,
    time=None,
    panel=None,
    bandwidth=None,
    kernel="bartlett",
):
    y = np.asarray(y, dtype=np.float64)
    design = IVDesign.from_arrays(exog, endog, instruments, len(y))
    C, E, I, xscale, zscale, escale = _equilibrate_iv_roles(design.exog, design.endog, design.excluded)
    X, Z = np.column_stack([C, E]), np.column_stack([C, I])
    yw, Xw, Zw, sw = weighted_arrays(y, X, Z, weights)
    n_eff = float(len(y) if weight_info is None else weight_info.effective_n)
    score_scale = None if weight_info is None else robust_score_scale(weight_info, vce, len(y))

    # First step is standard IV/2SLS through the shared outcome-agnostic core.
    first_step = weighted_2sls(y, X, Z, weights=weights)
    b1 = first_step.beta
    e1 = yw - Xw @ b1
    S1 = _gmm_weight_matrix(
        Zw, e1, kind=vce, clusters=clusters, center=center, time=time, panel=panel,
        bandwidth=bandwidth, kernel=kernel, effective_n=n_eff, score_scale=score_scale,
    )
    W = np.linalg.pinv((S1 + S1.T) / 2, hermitian=True)

    XZ = Xw.T @ Zw
    A = XZ @ W @ XZ.T
    rhs = XZ @ W @ (Zw.T @ yw)
    bread = np.linalg.pinv((A + A.T) / 2, hermitian=True)
    beta = bread @ rhs
    rank = int(np.linalg.matrix_rank(A))
    resid = y - X @ beta
    ew = resid if sw is None else resid * sw
    k_total = rank + int(df_absorbed)
    V = gmm_vcov(
        Xw, Zw, ew, W, kind=vce, clusters=clusters,
        k_total=k_total, nested_adj=int(bool(nested_adj)), center=center,
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        effective_n=n_eff, score_scale=score_scale,
    )

    pi_x, xhat = _first_stage_projection(Xw[:, :C.shape[1]], Xw[:, C.shape[1]:], Zw[:, C.shape[1]:])
    workspace = _IVDiagnosticWorkspace(E, C, I, weights)
    first_diagnostics, conditional_diagnostics = _first_stage_suite(
        E, C, I, weights=weights, weight_info=weight_info, vce=vce, clusters=clusters,
        _workspace=workspace,
        df_absorbed=df_absorbed, nested_adj=int(bool(nested_adj)),
        time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
    )
    first = {
        "coefficients": pi_x,
        "fitted_endog": xhat[:, C.shape[1]:],
        "diagnostics": first_diagnostics,
    }
    over_df = int(Zw.shape[1] - Xw.shape[1])
    if over_df > 0:
        g = Zw.T @ ew
        j = float(g.T @ W @ g)
        over = {"stat": j, "df": over_df, "pvalue": float(chi2.sf(j, over_df))}
    else:
        over = {"stat": np.nan, "df": 0, "pvalue": np.nan}
    diagnostics = {
        "cragg_donald_f": cragg_donald_stat(
            E, C, I, weights=weights, df_absorbed=df_absorbed, effective_n=n_eff,
            _workspace=workspace,
        ),
        "stock_yogo": stock_yogo_critical_values(I.shape[1], E.shape[1], estimator="gmm2s"),
        "kleibergen_paap": kleibergen_paap_stats(
            E, C, I, weights=weights, weight_info=weight_info, vce=vce, clusters=clusters,
            _workspace=workspace,
            df_absorbed=df_absorbed, nested_adj=int(bool(nested_adj)),
            time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
        ),
        "sanderson_windmeijer": conditional_diagnostics,
        "overidentification": over,
    }
    meta = {"estimator": "gmm2s", "center": bool(center), "first_step_params": b1, "weight_matrix": W}
    df_resid = max(n_eff-k_total, 0.0)
    if str(vce).lower().replace("-", "_") == "cluster" and clusters:
        df_resid = min(df_resid, float(min(len(np.unique(c)) for c in clusters) - 1))
    beta, V = _restore_iv_units(beta, V, first, meta, xscale, zscale, escale)
    return beta, V, resid, rank, df_resid, first, diagnostics, meta


def fit_iv_2sls(y, exog, endog, instruments, **kwargs):
    """Backward-compatible 2SLS entry point."""
    beta, V, resid, rank, df, first, _, _ = fit_iv_kclass(
        y, exog, endog, instruments, estimator="2sls", **kwargs
    )
    return beta, V, resid, rank, df, first
