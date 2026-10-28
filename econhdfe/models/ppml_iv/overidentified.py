"""Profile geometry and inference for iterated projected-IV Poisson equations.

No objective is re-labelled as efficient GMM.  The optional estimating-equation
sandwich differentiates BOTH the fitted instrument map and its empirical moments.
Its statistical scope is smooth, strongly identified fixed-dimensional M/Z
estimation; arbitrary growing-FE or weak-identification validity is not claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2
from ...errors import InferenceError, NumericalError, UnderidentifiedError
from ...compute.clusters import normalize_vce_clusters
from ...compute.vcov import cluster_subset_meats_xe


@dataclass(slots=True)
class ProfileGeometry:
    score: np.ndarray
    sensitivity: np.ndarray  # minus the exact profiled derivative, not symmetrized
    frozen_sensitivity: np.ndarray
    correction: np.ndarray
    tilt: np.ndarray         # Z_t (Z_t' W Z_t)^-1 Z_t' w0 (y-mu)
    instrument_scale: np.ndarray
    cross: np.ndarray        # column-scaled Z_t' W X_t
    correction_relative_norm: float


def profile_geometry(X, Z, fitted, weights, weighted_residual, *, chunk_rows=65536):
    """Exact profiled derivative at D' w0 (y-mu)=0, with W=w0*mu.

    X and Z must be W-residualized against the SAME FE space.  `fitted` is the
    weighted first-stage fitted X.  Additional tall workspace is one N-vector.
    No FE dummy matrix and no numerical parameter differences are constructed.
    """
    X, Z, H, w, r = (np.asarray(a, dtype=float) for a in
                     (X, Z, fitted, weights, weighted_residual))
    n, k = X.shape; l = Z.shape[1]
    if H.shape != X.shape or len(Z) != n or w.shape != (n,) or r.shape != (n,):
        raise ValueError("incompatible profile geometry inputs")
    if chunk_rows < 1 or l < k:
        raise ValueError("invalid profile geometry dimensions/chunk")
    scale = np.zeros(l)
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo+chunk_rows)
        scale += np.einsum('ij,i,ij->j', Z[lo:hi], w[lo:hi], Z[lo:hi])
    scale = np.sqrt(scale)
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise UnderidentifiedError("zero/nonfinite weighted instrument norm")
    C = np.zeros((l,l)); B = np.zeros((l,k)); m = np.zeros(l)
    frozen = np.zeros((k,k)); score = np.zeros(k)
    for lo in range(0,n,chunk_rows):
        hi=min(n,lo+chunk_rows); z=Z[lo:hi]/scale; wb=w[lo:hi]
        C += z.T @ (wb[:,None]*z)
        B += z.T @ (wb[:,None]*X[lo:hi])
        m += z.T @ r[lo:hi]
        frozen += H[lo:hi].T @ (wb[:,None]*X[lo:hi])
        score += H[lo:hi].T @ r[lo:hi]
    C=(C+C.T)*.5
    eig=np.linalg.eigvalsh(C)
    if eig[0] <= max(64.*np.finfo(float).eps*l,1e-12)*eig[-1]:
        raise UnderidentifiedError("profile instrument Gram is numerically rank deficient")
    a=cho_solve(cho_factor(C,lower=True,check_finite=False),m,check_finite=False)
    tilt=np.empty(n); correction=np.zeros((k,k))
    for lo in range(0,n,chunk_rows):
        hi=min(n,lo+chunk_rows)
        tilt[lo:hi]=(Z[lo:hi]/scale)@a
        R=X[lo:hi]-H[lo:hi]
        correction += R.T @ ((w[lo:hi]*tilt[lo:hi])[:,None]*X[lo:hi])
    sensitivity=frozen-correction
    # Report in equilibrated coefficient coordinates, not arbitrary raw units.
    sx=np.sqrt(np.maximum(np.abs(np.diag(frozen)),np.finfo(float).tiny))
    relative=float(np.linalg.norm(correction/sx[:,None]/sx[None,:]) /
                   max(np.linalg.norm(frozen/sx[:,None]/sx[None,:]),np.finfo(float).tiny))
    return ProfileGeometry(score,sensitivity,frozen,correction,tilt,scale,B,relative)


def inverse_sensitivity(A, *, rcond=1e-12):
    """General nonsymmetric inverse with column/row equilibration and rank guard."""
    A=np.asarray(A,dtype=float)
    col=np.maximum(np.max(np.abs(A),axis=0),np.finfo(float).tiny)
    row=np.maximum(np.max(np.abs(A),axis=1),np.finfo(float).tiny)
    scale=np.sqrt(np.sqrt(col)*np.sqrt(row))
    small=A/scale[:,None]/scale[None,:]
    s=np.linalg.svd(small,compute_uv=False)
    if not np.all(np.isfinite(s)) or s[0]<=0 or s[-1]<=rcond*s[0]:
        raise UnderidentifiedError("profile estimating-equation Jacobian is numerically singular")
    inv=np.linalg.solve(small,np.eye(len(A)))/scale[:,None]/scale[None,:]
    return inv,float(s[-1]/s[0])


def score_meat(scores, *, kind, clusters=None, base_weights=None,
               weight_kind="none", effective_n=None, chunk_rows=65536):
    """HC1 or CGM meat; frequency replication is not squared in the IID meat."""
    scores=np.asarray(scores,dtype=float); n,k=scores.shape
    if kind=='robust':
        meat=np.zeros((k,k))
        for lo in range(0,n,chunk_rows):
            hi=min(n,lo+chunk_rows); s=scores[lo:hi]
            if weight_kind=='fweight':
                if base_weights is None:
                    raise InferenceError("frequency meat requires base_weights")
                s=s/np.sqrt(np.asarray(base_weights)[lo:hi,None])
            meat+=s.T@s
        ne=float(n if effective_n is None else effective_n)
        if ne>1: meat*=ne/(ne-1.)
        return (meat+meat.T)*.5
    if kind!='cluster':
        raise InferenceError("profile inference supports robust/cluster only")
    clusters=normalize_vce_clusters(clusters,nobs=n)
    meat=np.zeros((k,k))
    for sign,S,g,_ in cluster_subset_meats_xe(scores,np.ones(n),clusters):
        if g<=1: raise InferenceError("at least two clusters are required")
        meat += sign*(g/(g-1.))*S
    return (meat+meat.T)*.5


def residualize_scaled(projector, data, weights, *, tol, copy=False):
    """Equilibrate RHS before projection; never accept nonconverged FE output.

    Tiny included-exogenous residual columns can otherwise trigger an absolute
    breakdown threshold in a fused multi-RHS backend and leave other columns
    unprojected. This adapter changes neither FE space nor projection target.
    """
    data=np.array(data,dtype=float,copy=True) if copy else np.asarray(data,dtype=float)
    scales=np.max(np.abs(data),axis=0)
    scales=np.where(scales>0,scales,1.)
    data/=scales
    try:
        out,info=projector.residualize(data,weights,tol=tol,return_info=True,copy=False)
    except Exception:
        # A rejected trial must not leave the persistent X/Z workspace in
        # equilibrated units before the next trial reuses it.
        data*=scales
        raise
    out*=scales
    if info is not None and not info.converged:
        raise NumericalError("FE projection failed convergence certification", code="numerical.iv_profile_projection")
    return out,info


def estimating_equation_scores(X, fitted, geometry, weights, weighted_residual,
                               *, projector=None, tol=1e-10, chunk_rows=65536):
    """Empirical influence scores, INCLUDING fitted-instrument estimation.

    psi_i = [H_i + P_D^W(h R)_i] w0_i u_i + w0_i mu_i h_i R_i.
    One extra K-RHS FE projection, only at final inference.  Merely replacing
    the old bread with the exact derivative and retaining H_i*w0_i*u_i would
    miss both generated-instrument and profiled-FE contributions.
    """
    X,H,w,r=(np.asarray(a,dtype=float) for a in (X,fitted,weights,weighted_residual))
    n,k=X.shape; scores=np.empty_like(X)
    for lo in range(0,n,chunk_rows):
        hi=min(n,lo+chunk_rows)
        scores[lo:hi]=geometry.tilt[lo:hi,None]*(X[lo:hi]-H[lo:hi])
    if projector is not None:
        scores,_=residualize_scaled(projector,scores,w,tol=tol,copy=False)
    # scores now contains M_D^W(hR); no dummy coefficients are needed.
    for lo in range(0,n,chunk_rows):
        hi=min(n,lo+chunk_rows)
        hR=geometry.tilt[lo:hi,None]*(X[lo:hi]-H[lo:hi])
        scores[lo:hi]=(H[lo:hi]+hR-scores[lo:hi])*r[lo:hi,None]+hR*w[lo:hi,None]
    return scores


def estimating_equation_vcov(X, fitted, geometry, weights, weighted_residual,
                             *, projector=None, tol=1e-10, kind='robust',
                             clusters=None, base_weights=None, weight_kind='none',
                             effective_n=None, chunk_rows=65536):
    scores=estimating_equation_scores(X,fitted,geometry,weights,weighted_residual,
                                     projector=projector,tol=tol,chunk_rows=chunk_rows)
    meat=score_meat(scores,kind=kind,clusters=clusters,base_weights=base_weights,
                    weight_kind=weight_kind,effective_n=effective_n,chunk_rows=chunk_rows)
    inverse,rcond=inverse_sensitivity(geometry.sensitivity)
    V=inverse@meat@inverse.T
    diagnostics=dict(mode='estimating_equation',bread='exact_profile_jacobian',
        generated_instrument_scores=True,profiled_fe_scores=True,
        sensitivity_rcond=rcond,correction_relative_norm=geometry.correction_relative_norm,
        score_sum_max=float(np.max(np.abs(scores.sum(axis=0)),initial=0.)),
        scope='regular_strong_identification_fixed_dimensional_random_design',
        weak_iv_robust=False,growing_fe_certified=False)
    return (V+V.T)*.5,diagnostics


def overidentification_test(Z, geometry, weighted_residual, *, kind='robust',
                            clusters=None,base_weights=None,weight_kind='none',
                            effective_n=None,chunk_rows=65536):
    """Orthogonalized moment Wald test for a consistent, not necessarily efficient, fit.

    N spans null(B'), B=Z_t' W X_t (in scaled instrument coordinates).
    T = (N'm)' [N' S N]^-1 (N'm) -> chi2(L-K) under the null and regularity.
    This is NOT the raw nonoptimal objective reported as a Hansen J statistic.
    Singular/indefinite restriction covariance suppresses p-values; no PSD
    repair or pseudoinverse silently changes the nominal degrees of freedom.
    """
    n,l=Z.shape; k=geometry.cross.shape[1]; df=l-k
    result=dict(name='orthogonalized_moment_wald',df=df,statistic=None,pvalue=None,
                status='exactly_identified' if df==0 else 'pending',
                weak_iv_robust=False,growing_fe_certified=False,
                null='all_additive_instrument_moments_correct')
    if df<=0: return result
    B=geometry.cross
    col=np.linalg.norm(B,axis=0)
    if np.any(col<=0):
        result['status']='rank_deficient_moment_jacobian'; return result
    U,s,_=np.linalg.svd(B/col,full_matrices=True)
    if s[-1]<=max(l,k)*np.finfo(float).eps*64*s[0]:
        result['status']='rank_deficient_moment_jacobian'; return result
    N=U[:,k:]
    # Restriction scores require only N*(L-K), not N*L, storage.
    scores=np.empty((n,df))
    for lo in range(0,n,chunk_rows):
        hi=min(n,lo+chunk_rows)
        scores[lo:hi]=((Z[lo:hi]/geometry.instrument_scale)@N)*weighted_residual[lo:hi,None]
    moment=scores.sum(axis=0)
    meat=score_meat(scores,kind=kind,clusters=clusters,base_weights=base_weights,
                    weight_kind=weight_kind,effective_n=effective_n,chunk_rows=chunk_rows)
    eig=np.linalg.eigvalsh(meat)
    if not np.all(np.isfinite(eig)) or eig[-1]<=0 or eig[0]<=1e-12*eig[-1]:
        result['status']='singular_or_indefinite_restriction_covariance'; return result
    stat=float(moment@np.linalg.solve(meat,moment))
    result.update(status='ok',statistic=stat,pvalue=float(chi2.sf(stat,df)),
                  tangent_orthogonality=float(np.linalg.norm(N.T@(B/col))),
                  covariance_rcond=float(eig[0]/eig[-1]))
    return result
