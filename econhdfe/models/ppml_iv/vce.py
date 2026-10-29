from __future__ import annotations
import numpy as np
from ...errors import InferenceError
from ...compute.clusters import normalize_vce_clusters
from ...compute.stable_linalg import equilibrated_gram_inverse
from ...compute.vcov import cluster_subset_meats_xe, sandwich_vcov_xe, normalize_vce, IID_VCE_KINDS
from ...compute.weights import frequency_score_scale


# IV-PPML names implemented by the robust sandwich.
_ROBUST_VCE_KINDS = ("robust", "model", *IID_VCE_KINDS)


def _bread(Xhat, X, w):
    A = np.asarray(Xhat, dtype=np.float64).T @ (np.asarray(w)[:, None] * np.asarray(X, dtype=np.float64))
    return equilibrated_gram_inverse((A + A.T) / 2.0)


def normalize_ivppml_vce(kind):
    """Return the implemented variance convention, not a legacy alias."""
    key = normalize_vce(kind, "robust")
    if key in _ROBUST_VCE_KINDS:
        return "robust"
    if key == "cluster":
        return key
    raise InferenceError("IV-PPML supports robust or cluster VCE", code="inference.unsupported_vce")


def ivppml_vcov(
    Xhat, X_dm, residual, irls_w, *, kind="robust", clusters=None,
    effective_n=None, base_weights=None, weight_kind="none", bread=None,
):
    """IV-PPML sandwich covariance matching the upstream score convention.

    Scores use first-stage fitted regressors ``Xhat`` and the IRLS residual
    multiplier ``irls_w * residual``.  Robust uses N/(N-1); cluster paths use
    G/(G-1) for every CGM subset, as in ``ivppmlhdfe``. Multiway-cluster
    results are unrepaired; see ``finalize_cluster_vcov``.
    """
    Xhat = np.asarray(Xhat, dtype=np.float64)
    X_dm = np.asarray(X_dm, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    irls_w = np.asarray(irls_w, dtype=np.float64)
    n = len(residual)
    n_eff = float(n if effective_n is None else effective_n)
    bread = _bread(Xhat, X_dm, irls_w) if bread is None else np.asarray(bread, dtype=np.float64)
    kind0 = normalize_ivppml_vce(kind)
    score_scalar = irls_w * residual

    if kind0 in _ROBUST_VCE_KINDS:
        # Frequency weights represent physical replication: the robust meat
        # contains one power of the user weight (the shared frequency score
        # scale), whereas probability weights keep the squared-score convention.
        weight_kind = str(weight_kind).lower()
        if weight_kind == "fweight" and base_weights is None:
            raise InferenceError("fweight robust VCE requires base_weights", code="inference.missing_base_weights")
        # k_total=1 gives ivppmlhdfe's N/(N-1) robust correction.
        V = sandwich_vcov_xe(
            Xhat, score_scalar, bread, kind="robust", k_total=1, effective_n=n_eff,
            score_scale=frequency_score_scale(weight_kind, base_weights, "robust"),
        )
        return (V + V.T) / 2.0

    if kind0 != "cluster":
        raise InferenceError("IV-PPML currently supports robust or cluster VCE", code="inference.unsupported_vce")
    clusters = normalize_vce_clusters(clusters, nobs=n)
    V = np.zeros_like(bread)
    for sign, meat, g, _ in cluster_subset_meats_xe(Xhat, score_scalar, clusters):
        V += sign * (g / (g - 1.0)) * (bread @ meat @ bread)
    return (V + V.T) / 2.0


def ivppml_ee_vcov(Xhat, X_dm, Z_dm, irls_w, weighted_residual, projector, *, tol, **vce_kw):
    """Estimating-equation sandwich for overidentified IV-PPML.

    The estimator solves ``Pi(b)' Z_dm' w0 (y - mu) = 0`` with first-stage
    coefficients ``Pi = (Z'WZ)^-1 Z'WX`` re-estimated at each mean
    (``W = w0 mu``, FE profiled out). When the sample overidentifying moments
    ``m = Z_dm' w0 (y - mu)`` are not zero, the standard sandwich omits two
    terms: the derivative of ``Pi`` in the Jacobian and the first-stage
    estimation error in the scores. Both vanish under correctly specified
    moments, where this equals ``ivppml_vcov`` asymptotically; under
    misspecification it is the sandwich of the stacked M-estimator.
    ``weighted_residual`` is ``w0 (y - mu)``; ``vce_kw`` follows ``ivppml_vcov``.
    """
    H, X, Z = (np.asarray(a, dtype=np.float64) for a in (Xhat, X_dm, Z_dm))
    W = np.asarray(irls_w, dtype=np.float64)
    u = np.asarray(weighted_residual, dtype=np.float64)
    n, k = X.shape
    G = Z.T @ (W[:, None] * Z)
    h = Z @ (equilibrated_gram_inverse((G + G.T) / 2.0) @ (Z.T @ u))
    R = X - H
    WX = W[:, None] * X
    sensitivity = H.T @ WX - R.T @ (h[:, None] * WX)
    Q = h[:, None] * R
    # Profiling the FE adds their W-projection of Q to the scores.
    PQ = Q - projector.residualize(Q.copy(), W, tol=tol, return_info=True, copy=False)[0]
    scores = (H + PQ) * u[:, None] + Q * W[:, None]
    meat = ivppml_vcov(scores, X, np.ones(n), np.ones(n), bread=np.eye(k), **vce_kw)
    try:
        inv = np.linalg.solve(sensitivity, np.eye(k))
    except np.linalg.LinAlgError as exc:
        raise InferenceError(
            "IV-PPML estimating-equation Jacobian is singular", code="inference.singular_jacobian",
        ) from exc
    V = inv @ meat @ inv.T
    return (V + V.T) / 2.0

