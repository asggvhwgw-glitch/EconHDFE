from __future__ import annotations
import numpy as np
from ...compute.vcov import sandwich_vcov_xe, normalize_vce, IID_VCE_KINDS, ROBUST_VCE_KINDS, _cluster_scale
from ...compute.clusters import normalize_vce_clusters
from ...compute.weights import frequency_score_scale
from ...compute.stable_linalg import equilibrated_gram_inverse
from ...errors import SpecificationError
from ...config import PPML_SMALL_SAMPLE_DEFAULT


# PPML VCE names served by the model-based inverse Hessian.
MODEL_VCE_KINDS = ("model", "asymptotic", *IID_VCE_KINDS)


def ppml_covariance_dof(small_sample, k_total, nested_adj):
    """Map PPML's explicit convention to the shared sandwich DoF arguments.

    ppmlhdfe requests reghdfe's ``vce_asymptotic`` convention. Setting K=1
    makes its robust factor N/(N-1) and its cluster factor G/(G-1), without
    charging the absorbed effects or nested-constant adjustment a second time.
    """
    if small_sample == "asymptotic":
        return 1, 0
    if small_sample == "residual":
        return int(k_total), int(bool(nested_adj))
    raise SpecificationError("ppml_small_sample must be asymptotic/residual")


def ppml_covariance_diagnostics(*, kind, small_sample, effective_n, k_total, nested_adj, clusters):
    """Report the finite-sample choice and multiplier actually used."""
    k, nested = ppml_covariance_dof(small_sample, k_total, nested_adj)
    kind0 = normalize_vce(kind, "model")
    n = float(effective_n)
    factor = 1.0
    if kind0 in (*ROBUST_VCE_KINDS, "hac", "newey_west", "neweywest"):
        factor = n / max(n - k, 1)
    elif kind0 == "cluster":
        # Estimator callers pass the already normalized contiguous codes.
        g = min(int(np.max(c)) + 1 for c in clusters)
        factor = _cluster_scale(n, k, g, nested_adj=nested)
    return {"convention": small_sample, "factor": float(factor),
            "effective_n": n, "k_total": k, "nested_adjustment": nested}


def _bread_streamed(X, w, *, chunk_rows=250_000):
    """Compute (X' W X)^-1 with bounded N x K temporaries."""
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    k = X.shape[1]
    H = np.zeros((k, k), dtype=np.float64)
    chunk = max(1, int(chunk_rows))
    for lo in range(0, len(w), chunk):
        hi = min(len(w), lo + chunk)
        xb = X[lo:hi]
        H += xb.T @ (xb * w[lo:hi, None])
    return equilibrated_gram_inverse(H)


def ppml_vcov(
    X_tilde, y, mu, true_w, *, kind="robust", clusters=None,
    df_absorbed=0, nested_adj=0, weight_kind="none", small_sample=PPML_SMALL_SAMPLE_DEFAULT,
):
    """PPML sandwich covariance without materializing a full score matrix."""
    X = np.asarray(X_tilde, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    tw = np.asarray(true_w, dtype=np.float64)
    bread = _bread_streamed(X, tw * mu)
    kind0 = normalize_vce(kind, "model")
    if kind0 in MODEL_VCE_KINDS:
        return bread

    if kind0 == "cluster":
        clusters = normalize_vce_clusters(clusters, nobs=len(y))

    # PPML score_i = X_tilde_i * true_w_i * (y_i - mu_i).
    score_scalar = tw * (y - mu)
    score_scale = frequency_score_scale(weight_kind, tw, kind0)
    effective_n = float(np.sum(tw)) if weight_kind == "fweight" else float(len(y))
    k_total, nested = ppml_covariance_dof(small_sample, X.shape[1] + int(df_absorbed), nested_adj)
    return sandwich_vcov_xe(
        X, score_scalar, bread,
        kind=kind0, clusters=clusters,
        k_total=k_total,
        nested_adj=nested, effective_n=effective_n, score_scale=score_scale,
    )
