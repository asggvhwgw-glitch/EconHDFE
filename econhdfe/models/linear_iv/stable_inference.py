"""Inference in orthogonal coordinates for the difficult dense 2SLS path."""
from __future__ import annotations

from itertools import combinations
import numpy as np
import scipy.linalg as la
from scipy.stats import chi2

from ...compute.encoding import factorize_interaction
from ...compute.vcov import score_covariance, _cluster_scale, fix_psd, normalize_vce, is_iid_vce, ROBUST_VCE_KINDS
from ...errors import InferenceError


_SCORE_BYTES = 256 * 1024**2


def _cluster_accumulators(clusters, p):
    """Bound retained codes, lookups, non-singleton sums and singleton HC0.

    The budget covers all arrays retained by the returned accumulators, not
    temporary factorization/counting arrays or the fit's total resident memory.
    """
    accumulators = []
    used = 0
    index_bytes = np.dtype(np.int64).itemsize
    score_bytes = np.dtype(np.float64).itemsize
    for size in range(1, len(clusters) + 1):
        for subset in combinations(clusters, size):
            codes = np.unique(subset[0], return_inverse=True)[1] if size == 1 else factorize_interaction(subset)[0]
            counts = np.bincount(codes)
            repeated = np.flatnonzero(counts > 1)
            required = (used + codes.nbytes + len(counts) * index_bytes
                        + (len(repeated) * p + p * p) * score_bytes)
            if required > _SCORE_BYTES:
                raise InferenceError(
                    "Stable IV cluster-score aggregation exceeds its memory budget",
                    details={"required_bytes": required, "budget_bytes": _SCORE_BYTES},
                    suggestion="This stable path needs a larger cluster-score memory budget or a streaming aggregation backend.",
                )
            used = required
            lookup = np.full(len(counts), -1, dtype=np.int64)
            lookup[repeated] = np.arange(len(repeated))
            accumulators.append((1 if size % 2 else -1, codes, lookup,
                                 np.zeros((len(repeated), p)), np.zeros((p, p))))
    return accumulators


def stable_2sls_inference(factor, *, n_exog, kind="iid", clusters=None,
                          k_total, nested_adj=0, effective_n=None, score_scale=None,
                          time=None, panel=None, bandwidth=None, kernel="bartlett"):
    """Replay the QR once for coefficient VCE, over-ID, and diagnostic roles.

    Scores are transformed before their outer products. This avoids cancellation
    in a sandwich containing the squared condition number of the design.
    Multiway inclusion-exclusion is applied before the existing final VCE repair.
    """
    n = factor.nobs
    l = factor.instrument_coordinates.shape[1]
    k = factor.influence_coordinates.shape[1]
    ke = k - n_exog
    n_eff = float(n if effective_n is None else effective_n)
    kind0 = normalize_vce(kind, "iid")
    iid = is_iid_vce(kind0)
    robust = kind0 in ROBUST_VCE_KINDS
    p = l + k
    partial_coordinates = factor.diagnostic_coordinates
    coordinates = np.column_stack([
        factor.instrument_coordinates, factor.influence_coordinates,
        factor.projected_coordinates[:, n_exog:],
        np.asarray(factor.residual_coordinates).reshape(-1, 1), partial_coordinates,
    ])
    ew, fitted = np.empty(n), np.empty((n, ke))
    partial = np.empty((n, partial_coordinates.shape[1]))
    g, hc0 = np.zeros(l), np.zeros((p, p))
    accumulators = []
    full_scores = None
    if kind0 == "cluster":
        if not clusters:
            raise ValueError("cluster covariance requires cluster arrays")
        groups = [len(np.unique(c)) for c in clusters]
        correction = _cluster_scale(n_eff, k_total, min(groups), nested_adj=nested_adj)
        accumulators = _cluster_accumulators(clusters, p)
    elif not iid and not robust:
        required = n * p * 8
        if required > _SCORE_BYTES:
            raise InferenceError(
                "Stable IV ordered-score covariance exceeds its memory budget",
                details={"required_bytes": required, "budget_bytes": _SCORE_BYTES},
                suggestion="This stable path needs a larger ordered-score memory budget or a streaming ordered-score backend.",
            )
        full_scores = np.empty((n, p))
    rss = 0.0
    for lo, hi, values in factor.iter_transformed(coordinates):
        q, influence = values[:, :l], values[:, l:p]
        e = values[:, p + ke]
        ew[lo:hi], fitted[lo:hi] = e, values[:, p:p+ke]
        partial[lo:hi] = values[:, p+ke+1:]
        rss += float(e @ e)
        g += q.T @ e
        if iid:
            continue
        scores = values[:, :p] * e[:, None]
        if score_scale is not None:
            scores *= np.asarray(score_scale)[lo:hi, None]
        if full_scores is not None:
            full_scores[lo:hi] = scores
            continue
        if robust:
            hc0 += scores.T @ scores
        for _, codes, lookup, sums, singleton_hc0 in accumulators:
            mapped = lookup[codes[lo:hi]]
            selected = mapped >= 0
            if np.any(selected):
                selected_scores = scores[selected]
                np.add.at(sums, mapped[selected], selected_scores)
            if np.any(~selected):
                singleton_scores = scores[~selected]
                singleton_hc0 += singleton_scores.T @ singleton_scores
    if iid:
        loading = factor.influence_coordinates
        V = (loading.T @ loading) * (rss / max(n_eff-k_total, 1))
        S = np.eye(l) * (rss / n_eff)
    else:
        if kind0 == "cluster":
            raw = np.zeros_like(hc0)
            for sign, _, _, sums, singleton_hc0 in accumulators:
                raw += sign * (sums.T @ sums + singleton_hc0)
            V = raw[l:, l:] * correction
        elif robust:
            raw = hc0
            V = raw[l:, l:] * (n_eff / max(n_eff-k_total, 1))
        else:
            raw = score_covariance(
                full_scores, kind=kind0, clusters=clusters, k_total=0,
                small_sample=False, time=time, panel=panel,
                bandwidth=bandwidth, kernel=kernel, effective_n=n_eff,
            )
            # Keep the generic covariance engine's HAC/DK finite-sample rules.
            V = score_covariance(
                full_scores[:, l:], kind=kind0, clusters=clusters, k_total=k_total,
                nested_adj=nested_adj, time=time, panel=panel,
                bandwidth=bandwidth, kernel=kernel, effective_n=n_eff,
            )
        S = raw[:l, :l]
        if kind0 == "cluster" and len(clusters) > 1:
            V = fix_psd(V)
    V = (V + V.T) / 2
    df = l - k
    over = {"stat": np.nan, "df": 0, "pvalue": np.nan}
    if df > 0:
        S = (S + S.T) / 2
        W = np.linalg.pinv(S, hermitian=True)
        if not iid:
            # An orthonormal basis spans the same derivative space as Qz'X.
            # It removes near-collinear controls before the efficient GMM step.
            D = factor.projected_left
            eigenvalues, eigenvectors = la.eigh(S, check_finite=False)
            cutoff = 1e-15 * np.max(np.abs(eigenvalues))
            if eigenvalues[0] >= -cutoff:
                keep = eigenvalues > cutoff
                root = (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])).T
                target, design = root @ g, root @ D
                delta, *_ = la.lstsq(design, target, cond=None, lapack_driver="gelsd")
                remainder = target - design @ delta
                stat = float(remainder @ remainder)
            else:
                # Preserve signed multiway inclusion-exclusion semantics.
                A = D.T @ W @ D
                delta = np.linalg.pinv((A+A.T)/2, hermitian=True) @ (D.T @ W @ g)
                g = g - D @ delta
                stat = float(g @ W @ g)
        else:
            stat = float(g @ W @ g)
        over = {"stat": stat, "df": df, "pvalue": float(chi2.sf(stat, df))}
    return {"V": V, "residuals_weighted": ew, "fitted_endog_weighted": fitted,
            "partial_endog": partial[:, :ke], "partial_instruments": partial[:, ke:],
            "overidentification": over}
