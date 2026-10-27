from __future__ import annotations
import numpy as np
from ...compute.vcov import sandwich_vcov_xe, normalize_vce, IID_VCE_KINDS
from ...compute.clusters import normalize_vce_clusters
from ...compute.stable_linalg import equilibrated_gram_inverse


# PPML VCE names served by the model-based inverse Hessian.
MODEL_VCE_KINDS = ("model", "asymptotic", *IID_VCE_KINDS)


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
    df_absorbed=0, nested_adj=0, weight_kind="none", repair_psd=True,
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
    from ...compute.vcov import ROBUST_VCE_KINDS
    score_scale = 1.0 / np.sqrt(tw) if weight_kind == "fweight" and kind0 in ROBUST_VCE_KINDS else None
    effective_n = float(np.sum(tw)) if weight_kind == "fweight" else float(len(y))
    return sandwich_vcov_xe(
        X, score_scalar, bread,
        kind=kind0, clusters=clusters,
        k_total=X.shape[1] + int(df_absorbed),
        nested_adj=int(bool(nested_adj)), effective_n=effective_n, score_scale=score_scale,
        repair_psd=repair_psd,
    )
