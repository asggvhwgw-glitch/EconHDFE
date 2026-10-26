from __future__ import annotations
import numpy as np
from ...errors import InferenceError
from ...compute.clusters import normalize_vce_clusters
from ...compute.stable_linalg import equilibrated_gram_inverse
from ...compute.vcov import cluster_subset_meats_xe, fix_psd, normalize_vce, IID_VCE_KINDS


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
    G/(G-1) for every CGM subset, as in ``ivppmlhdfe``.
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
        # Frequency weights represent physical replication.  The robust meat
        # therefore contains one power of the user weight, whereas probability
        # weights retain the usual squared-score convention.
        if str(weight_kind).lower() == "fweight":
            if base_weights is None:
                raise InferenceError("fweight robust VCE requires base_weights", code="inference.missing_base_weights")
            bw = np.asarray(base_weights, dtype=np.float64)
            score_scalar = (irls_w / np.sqrt(bw)) * residual
        meat = np.zeros((Xhat.shape[1], Xhat.shape[1]), dtype=np.float64)
        chunk = 250_000
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            S = Xhat[lo:hi] * score_scalar[lo:hi, None]
            meat += S.T @ S
        if n_eff > 1:
            meat *= n_eff / (n_eff - 1.0)
        V = bread @ meat @ bread
        return (V + V.T) / 2.0

    if kind0 != "cluster":
        raise InferenceError("IV-PPML currently supports robust or cluster VCE", code="inference.unsupported_vce")
    clusters = normalize_vce_clusters(clusters, nobs=n)
    V = np.zeros_like(bread)
    for sign, meat, g, _ in cluster_subset_meats_xe(Xhat, score_scalar, clusters):
        V += sign * (g / (g - 1.0)) * (bread @ meat @ bread)
    V = (V + V.T) / 2.0
    return fix_psd(V) if len(clusters) > 1 else V
