from __future__ import annotations
from itertools import combinations
import numpy as np
from ..errors import InferenceError
from .stable_linalg import equilibrated_gram_inverse
from .encoding import factorize_1d, factorize_interaction
from .kernels import hac_meat, driscoll_kraay_meat
from .block_design import BlockDesign


def _sym_inv(a: np.ndarray) -> np.ndarray:
    return equilibrated_gram_inverse(a)


def _cluster_meat(scores: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, int]:
    codes = np.asarray(codes, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    g = int(codes.max()) + 1 if len(codes) else 0
    p = scores.shape[1]
    # Near-observation-level cluster intersections can make G x K larger than
    # the score matrix itself. Use the exact identity
    #   cluster meat = HC0 + corrections for cells with n_g > 1
    # when that dense group-score matrix would exceed 256 MiB and repeated
    # observations are a minority.
    if g * max(p, 1) * 8 > 256 * 1024**2 and len(codes):
        counts = np.bincount(codes, minlength=g)
        repeated_groups = counts > 1
        repeated_obs = repeated_groups[codes]
        if np.count_nonzero(repeated_obs) <= len(codes) // 2:
            meat = scores.T @ scores
            if np.any(repeated_obs):
                rc = codes[repeated_obs]
                rs = scores[repeated_obs]
                _, inv = np.unique(rc, return_inverse=True)
                gr = int(inv.max()) + 1 if len(inv) else 0
                sums = np.empty((gr, p), dtype=np.float64)
                for j in range(p):
                    sums[:, j] = np.bincount(inv, weights=rs[:, j], minlength=gr)
                meat += sums.T @ sums - rs.T @ rs
            return meat, g
    sums = np.empty((g, p), dtype=np.float64)
    for j in range(p):
        sums[:, j] = np.bincount(codes, weights=scores[:, j], minlength=g)
    return sums.T @ sums, g


def _intersection_codes_many(arrays) -> np.ndarray:
    """Dense codes for a cluster intersection, using int64 mixed radix when safe."""
    arrays = [np.asarray(a, dtype=np.int64) for a in arrays]
    if len(arrays) == 1:
        return arrays[0].astype(np.int32, copy=False)
    raw = arrays[0].copy()
    limit = np.iinfo(np.int64).max
    for a in arrays[1:]:
        base = int(a.max()) + 1 if len(a) else 1
        maxraw = int(raw.max()) if len(raw) else 0
        if base > 0 and maxraw <= (limit - max(base - 1, 0)) // base:
            raw = raw * base + a
        else:
            codes, _ = factorize_interaction(arrays)
            return codes
    codes, _ = factorize_1d(raw)
    return codes


def _cluster_combined_meat(scores: np.ndarray, clusters) -> tuple[np.ndarray, int]:
    """Cameron-Gelbach-Miller inclusion-exclusion for arbitrary cluster ways."""
    clusters = tuple(clusters)
    if not clusters:
        raise ValueError("at least one cluster dimension is required")
    p = scores.shape[1]
    total = np.zeros((p, p), dtype=np.float64)
    g_base = [int(np.max(c)) + 1 if len(c) else 0 for c in clusters]
    if not g_base or min(g_base) < 2:
        raise InferenceError(
            "cluster inference requires at least two clusters in every dimension",
            code="inference.insufficient_clusters", details={"cluster_counts": g_base},
        )
    for r in range(1, len(clusters) + 1):
        sign = 1.0 if r % 2 else -1.0
        for idx in combinations(range(len(clusters)), r):
            if r == 1:
                codes = clusters[idx[0]]
            else:
                codes = _intersection_codes_many([clusters[j] for j in idx])
            meat, _ = _cluster_meat(scores, codes)
            total += sign * meat
    return (total + total.T) / 2.0, min(g_base)



def cluster_subset_meats_xe(X: np.ndarray, e: np.ndarray, clusters, score_scale=None):
    """Yield CGM cluster-subset meats for scores ``X_i e_i``.

    Each yielded tuple is ``(sign, meat, group_count, subset_size)``.  Keeping
    the group count per subset lets model-specific VCEs apply their own finite-
    sample convention without duplicating cluster aggregation logic.
    """
    X = np.asarray(X, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    base = e if score_scale is None else e * np.asarray(score_scale, dtype=np.float64)
    clusters = tuple(clusters)
    p = X.shape[1]
    for r in range(1, len(clusters) + 1):
        sign = 1.0 if r % 2 else -1.0
        for idx in combinations(range(len(clusters)), r):
            codes = clusters[idx[0]] if r == 1 else _intersection_codes_many([clusters[j] for j in idx])
            codes = np.asarray(codes, dtype=np.int64)
            g = int(codes.max()) + 1 if len(codes) else 0
            if g * max(p, 1) * 8 > 256 * 1024**2 and len(codes):
                counts = np.bincount(codes, minlength=g)
                repeated_obs = (counts > 1)[codes]
            else:
                repeated_obs = None
            if repeated_obs is not None and np.count_nonzero(repeated_obs) <= len(codes) // 2:
                meat = np.zeros((p, p), dtype=np.float64)
                chunk = 250_000
                for lo in range(0, len(base), chunk):
                    hi = min(len(base), lo + chunk)
                    xb = X[lo:hi] * base[lo:hi, None]
                    meat += xb.T @ xb
                if np.any(repeated_obs):
                    rc = codes[repeated_obs]
                    rs = X[repeated_obs] * base[repeated_obs, None]
                    _, inv = np.unique(rc, return_inverse=True)
                    gr = int(inv.max()) + 1 if len(inv) else 0
                    sums = np.empty((gr, p), dtype=np.float64)
                    for j in range(p):
                        sums[:, j] = np.bincount(inv, weights=rs[:, j], minlength=gr)
                    meat += sums.T @ sums - rs.T @ rs
            else:
                sums = np.empty((g, p), dtype=np.float64)
                for j in range(p):
                    sums[:, j] = np.bincount(codes, weights=X[:, j] * base, minlength=g)
                meat = sums.T @ sums
            yield sign, meat, g, r


def _cluster_meat_xe(X: np.ndarray, e: np.ndarray, clusters, score_scale=None) -> tuple[np.ndarray, int]:
    """Cluster meat for X_i e_i without materializing the N x K score matrix."""
    X = np.asarray(X, dtype=np.float64)
    clusters = tuple(clusters)
    p = X.shape[1]
    total = np.zeros((p, p), dtype=np.float64)
    g_base = [int(np.max(c)) + 1 if len(c) else 0 for c in clusters]
    if not g_base or min(g_base) < 2:
        raise InferenceError(
            "cluster inference requires at least two clusters in every dimension",
            code="inference.insufficient_clusters", details={"cluster_counts": g_base},
        )
    for sign, meat, _, _ in cluster_subset_meats_xe(X, e, clusters, score_scale=score_scale):
        total += sign * meat
    return (total + total.T) / 2.0, min(g_base)


def _cluster_scale(n: int, k_total: int, g: int, nested_adj: int = 0) -> float:
    if g <= 1:
        raise InferenceError(
            "cluster inference requires at least two clusters in every dimension",
            code="inference.insufficient_clusters", details={"n_clusters": int(g)},
        )
    if n <= k_total + nested_adj:
        raise InferenceError(
            "cluster finite-sample correction requires positive residual degrees of freedom",
            code="inference.invalid_degrees_of_freedom",
            details={"effective_n": float(n), "k_total": int(k_total), "nested_adj": int(nested_adj)},
        )
    return ((n - 1.0) / (n - k_total - nested_adj)) * (g / (g - 1.0))


def _fix_psd(v: np.ndarray, *, tol: float = 0.0) -> np.ndarray:
    """Clip negative eigenvalues caused by multiway inclusion-exclusion.

    reghdfe applies this repair to the final clustered covariance matrix for
    multiway clustering.  It is intentionally not applied to the raw meat so
    that sandwich transformations remain algebraically correct.
    """
    v = (np.asarray(v, dtype=np.float64) + np.asarray(v, dtype=np.float64).T) / 2.0
    evals, evecs = np.linalg.eigh(v)
    if np.min(evals) >= -tol:
        return v
    evals = np.clip(evals, 0.0, None)
    out = (evecs * evals) @ evecs.T
    return (out + out.T) / 2.0


def _is_multiway_cluster(kind, clusters) -> bool:
    return kind is not None and str(kind).lower().replace("-", "_") == "cluster" and clusters is not None and len(clusters) > 1


def score_covariance(
    scores: np.ndarray,
    *,
    kind: str,
    clusters=None,
    k_total: int = 0,
    nested_adj: int = 0,
    center: bool = False,
    small_sample: bool = True,
    time=None,
    panel=None,
    bandwidth=None,
    kernel="bartlett",
    effective_n=None,
    score_scale=None,
) -> np.ndarray:
    """Raw (sum-scaled) covariance of observation-level moment scores.

    ``effective_n`` controls finite-sample factors (not the number of stored
    rows), which is required for Stata frequency weights. ``score_scale`` can
    alter observation-level score multiplicity without changing the bread.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    n_eff = float(n if effective_n is None else effective_n)
    if score_scale is not None:
        scale = np.asarray(score_scale, dtype=np.float64)
        if scale.ndim != 1 or len(scale) != n:
            raise ValueError("score_scale must have one entry per observation")
        scores = scores * scale[:, None]
    if center:
        scores = scores - scores.mean(axis=0, keepdims=True)
    kind = "iid" if kind is None else kind.lower().replace("-", "_")
    if kind in ("robust", "hc1", "heteroskedastic"):
        out = scores.T @ scores
        if small_sample:
            out *= n_eff / max(n_eff - k_total, 1)
        return out
    if kind == "cluster":
        if not clusters:
            raise ValueError("cluster covariance requires cluster arrays")
        meat, g = _cluster_combined_meat(scores, clusters)
        if small_sample:
            meat *= _cluster_scale(n_eff, k_total, g, nested_adj=nested_adj)
        return meat
    if kind in {"hac", "newey_west", "neweywest"}:
        meat, _ = hac_meat(scores, bandwidth=bandwidth, kernel=kernel, time=time, panel=panel)
        if small_sample:
            meat *= n_eff / max(n_eff - k_total, 1)
        return meat
    if kind in {"dkraay", "driscoll_kraay", "dk"}:
        if time is None:
            raise ValueError("Driscoll-Kraay covariance requires time=")
        meat, _, n_periods = driscoll_kraay_meat(
            scores, time, bandwidth=bandwidth, kernel=kernel
        )
        # reghdfe documents vce(dkraay 1) (zero serial lags) as exactly
        # equivalent to clustering on the time dimension.  Applying the
        # cluster finite-sample factor with G=T preserves that invariant.
        if small_sample:
            meat *= _cluster_scale(
                n_eff, k_total, n_periods, nested_adj=nested_adj
            )
        return meat
    if kind in ("iid", "unadjusted", "homoskedastic"):
        return scores.T @ scores
    raise ValueError(f"unknown covariance kind: {kind}")


def sandwich_vcov_xe(
    X, e, bread, *, kind="robust", clusters=None, k_total=None, nested_adj=0,
    time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None,
    score_scale=None, chunk_rows=250_000,
):
    """Sandwich covariance for scores ``X_i * e_i`` without an N x K score copy.

    Robust and clustered paths stream or aggregate directly from ``X`` and the
    scalar score multiplier.  HAC/DK paths retain the generic score-matrix
    implementation because they need observation ordering across all columns.
    """
    X = np.asarray(X, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    n, k = X.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    kind0 = "iid" if kind is None else str(kind).lower().replace("-", "_")
    base = e if score_scale is None else e * np.asarray(score_scale, dtype=np.float64)
    if kind0 in {"robust", "hc1", "heteroskedastic"}:
        meat = np.zeros((k, k), dtype=np.float64)
        chunk = max(1, int(chunk_rows))
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            xb = X[lo:hi] * base[lo:hi, None]
            meat += xb.T @ xb
        meat *= n_eff / max(n_eff - k_total, 1)
    elif kind0 == "cluster":
        if not clusters:
            raise ValueError("cluster covariance requires cluster arrays")
        meat, g = _cluster_meat_xe(X, e, clusters, score_scale=score_scale)
        meat *= _cluster_scale(n_eff, k_total, g, nested_adj=nested_adj)
    else:
        scores = X * base[:, None]
        meat = score_covariance(
            scores, kind=kind0, clusters=clusters, k_total=k_total,
            nested_adj=nested_adj, small_sample=True, time=time, panel=panel,
            bandwidth=bandwidth, kernel=kernel, effective_n=n_eff,
        )
    V = bread @ meat @ bread
    return _fix_psd(V) if _is_multiway_cluster(kind0, clusters) else V



def block_cluster_subset_meats_xe(X: BlockDesign, e, clusters, *, score_scale=None):
    """Yield exact CGM subset meats for block-local scalar-multiplier scores.

    This is the BlockDesign counterpart of ``cluster_subset_meats_xe``.  It
    retains cluster ids that span row components by joining local group-score
    tables before forming cross-component products.
    """
    if not isinstance(X, BlockDesign):
        raise TypeError("X must be a BlockDesign")
    X._ensure_layout()
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 1 or len(e) != X.nobs:
        raise ValueError("e must have one value per BlockDesign observation")
    base = e if score_scale is None else e * np.asarray(score_scale, dtype=np.float64)
    clusters = tuple(np.asarray(c, dtype=np.int64) for c in clusters)
    if not clusters:
        raise ValueError("at least one cluster dimension is required")
    for c in clusters:
        if c.ndim != 1 or len(c) != X.nobs:
            raise ValueError("cluster arrays must have one value per observation")

    for r in range(1, len(clusters) + 1):
        sign = 1.0 if r % 2 else -1.0
        for idx in combinations(range(len(clusters)), r):
            codes = clusters[idx[0]] if r == 1 else _intersection_codes_many([clusters[j] for j in idx])
            codes = np.asarray(codes, dtype=np.int64)
            g = int(codes.max()) + 1 if len(codes) else 0
            local_tables = []
            for block in X.blocks:
                rows = block.rows
                local_codes = codes[rows]
                unique, inv = np.unique(local_codes, return_inverse=True)
                vals = block.values
                p = vals.shape[1]
                sums = np.empty((len(unique), p), dtype=np.float64)
                scalar = base[rows]
                for j in range(p):
                    sums[:, j] = np.bincount(inv, weights=vals[:, j] * scalar, minlength=len(unique))
                local_tables.append((unique, sums, block.columns))

            meat = np.zeros((X.ncols, X.ncols), dtype=np.float64)
            for a, (ua, sa, ca) in enumerate(local_tables):
                meat[np.ix_(ca, ca)] += sa.T @ sa
                for ub, sb, cb in local_tables[a + 1:]:
                    common, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
                    if common.size == 0:
                        continue
                    cross = sa[ia].T @ sb[ib]
                    meat[np.ix_(ca, cb)] += cross
                    meat[np.ix_(cb, ca)] += cross.T
            yield sign, meat, g, r


def _block_cluster_meat_xe(X: BlockDesign, e: np.ndarray, clusters, score_scale=None) -> tuple[np.ndarray, int]:
    """Exact cluster meat for block-local columns without an N x K score matrix."""
    clusters = tuple(np.asarray(c, dtype=np.int64) for c in clusters)
    if not clusters:
        raise ValueError("at least one cluster dimension is required")
    total = np.zeros((X.ncols, X.ncols), dtype=np.float64)
    g_base = [int(np.max(c)) + 1 if len(c) else 0 for c in clusters]
    if not g_base or min(g_base) < 2:
        raise InferenceError(
            "cluster inference requires at least two clusters in every dimension",
            code="inference.insufficient_clusters", details={"cluster_counts": g_base},
        )
    for sign, meat, _, _ in block_cluster_subset_meats_xe(
        X, e, clusters, score_scale=score_scale
    ):
        total += sign * meat
    return (total + total.T) / 2.0, min(g_base)


def sandwich_vcov_block_xe(
    X: BlockDesign, e, bread, *, kind="robust", clusters=None, k_total=None,
    nested_adj=0, effective_n=None, score_scale=None,
):
    """Sandwich covariance for block-local scores ``X_i * e_i``.

    The robust path uses block-local weighted Gram matrices.  Clustered paths
    preserve clusters that span multiple structural components by explicitly
    joining component group-score tables before constructing the meat.
    """
    if not isinstance(X, BlockDesign):
        raise TypeError("X must be a BlockDesign")
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 1 or len(e) != X.nobs:
        raise ValueError("e must have one value per observation")
    n_eff = float(X.nobs if effective_n is None else effective_n)
    k_total = X.ncols if k_total is None else int(k_total)
    kind0 = "iid" if kind is None else str(kind).lower().replace("-", "_")
    base = e if score_scale is None else e * np.asarray(score_scale, dtype=np.float64)

    if kind0 in {"robust", "hc1", "heteroskedastic"}:
        meat = X.gram(weights=base * base)
        meat *= n_eff / max(n_eff - k_total, 1)
    elif kind0 == "cluster":
        if not clusters:
            raise ValueError("cluster covariance requires cluster arrays")
        meat, g = _block_cluster_meat_xe(X, e, clusters, score_scale=score_scale)
        meat *= _cluster_scale(n_eff, k_total, g, nested_adj=nested_adj)
    else:
        # Ordered-score covariance estimators are not yet block-native because
        # they depend on the global observation sequence.  Refuse silently
        # materializing here; callers can use the established dense fallback.
        raise ValueError(f"block score covariance does not support kind: {kind0}")
    V = np.asarray(bread, dtype=np.float64) @ meat @ np.asarray(bread, dtype=np.float64)
    return _fix_psd(V) if _is_multiway_cluster(kind0, clusters) else V


def block_cluster_meat_xe(X: BlockDesign, e, clusters, *, score_scale=None):
    """Public exact cluster meat for a BlockDesign and scalar score multiplier."""
    return _block_cluster_meat_xe(X, e, clusters, score_scale=score_scale)


def cluster_meat_xe(X, e, clusters, *, score_scale=None):
    """Public streamed cluster meat for scalar-multiplier scores ``X_i e_i``."""
    return _cluster_meat_xe(X, e, clusters, score_scale=score_scale)


def fix_psd(V, *, tol=0.0):
    """Public PSD repair used by multi-way sandwich estimators."""
    return _fix_psd(V, tol=tol)


def ols_vcov(Xw, ew, bread, *, kind="iid", clusters=None, k_total=None, nested_adj=0,
             time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None,
             score_scale=None):
    n, k = Xw.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    if kind in (None, "iid", "unadjusted", "homoskedastic"):
        df = max(n_eff - k_total, 1)
        return bread * (float(ew @ ew) / df)
    return sandwich_vcov_xe(
        Xw, ew, bread, kind=kind, clusters=clusters, k_total=k_total,
        nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth,
        kernel=kernel, effective_n=n_eff, score_scale=score_scale,
    )


def kclass_vcov(
    Xw, Zw, ew, bread, *, kind="iid", clusters=None, k_total=None,
    nested_adj=0, time=None, panel=None, bandwidth=None, kernel="bartlett",
    effective_n=None, score_scale=None, kappa=1.0,
):
    """Sandwich using H=(1-kappa)X+kappa Pz X for fixed k-class.

    The default retains the conventional IV/LIML covariance path. Explicit
    fixed k-class callers pass their kappa; kappa=0 then reduces to OLS.
    """
    n, k = Xw.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    if kind in (None, "iid", "unadjusted", "homoskedastic") and kappa in (0.0, 1.0):
        df = max(n_eff - k_total, 1)
        return bread * (float(ew @ ew) / df)
    if kappa == 0.0:
        xhat = Xw
    else:
        gamma = _sym_inv(Zw.T @ Zw) @ (Zw.T @ Xw)
        xhat = Zw @ gamma
        if kappa != 1.0:
            xhat = (1.0 - kappa) * Xw + kappa * xhat
    if kind in (None, "iid", "unadjusted", "homoskedastic"):
        sigma2 = float(ew @ ew) / max(n_eff - k_total, 1)
        return sigma2 * (bread @ (xhat.T @ xhat) @ bread.T)
    scores = xhat * ew[:, None]
    meat = score_covariance(
        scores, kind=kind, clusters=clusters, k_total=k_total,
        nested_adj=nested_adj, small_sample=True, time=time, panel=panel,
        bandwidth=bandwidth, kernel=kernel, effective_n=n_eff, score_scale=score_scale,
    )
    V = bread @ meat @ bread
    return _fix_psd(V) if _is_multiway_cluster(kind, clusters) else V


def iv_vcov(Xw, Zw, ew, A_inv, **kwargs):
    return kclass_vcov(Xw, Zw, ew, A_inv, **kwargs)


def gmm_vcov(
    Xw, Zw, ew, W, *, kind="robust", clusters=None, k_total=None,
    nested_adj=0, center=False, time=None, panel=None, bandwidth=None, kernel="bartlett",
    effective_n=None, score_scale=None,
):
    n, k = Xw.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    XZ = Xw.T @ Zw
    D = XZ @ W
    bread = np.linalg.pinv(D @ XZ.T, hermitian=True)
    if kind in (None, "iid", "unadjusted", "homoskedastic"):
        sigma2 = float(ew @ ew) / max(n_eff - k_total, 1)
        S = sigma2 * (Zw.T @ Zw)
    else:
        S = score_covariance(
            Zw * ew[:, None], kind=kind, clusters=clusters,
            k_total=k_total, nested_adj=nested_adj, center=center,
            small_sample=True, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            effective_n=n_eff, score_scale=score_scale,
        )
    middle = D @ S @ D.T
    V = bread @ middle @ bread
    V = (V + V.T) / 2
    return _fix_psd(V) if _is_multiway_cluster(kind, clusters) else V
