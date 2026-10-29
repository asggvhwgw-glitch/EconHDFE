from __future__ import annotations
from itertools import combinations
import numpy as np
from ..errors import InferenceError
from .stable_linalg import equilibrated_gram_inverse
from .encoding import factorize_1d, factorize_interaction
from .kernels import hac_meat, driscoll_kraay_meat
from .block_design import BlockDesign
from .design_ops import column_moments


IID_VCE_KINDS = ("iid", "unadjusted", "homoskedastic")
ROBUST_VCE_KINDS = ("robust", "hc1", "heteroskedastic")
# VCE names served by the block-native linear paths; "" is an unset VCE.
BLOCK_VCE_KINDS = ("", *IID_VCE_KINDS, *ROBUST_VCE_KINDS, "cluster")


def normalize_vce(vce, default=None):
    """Lower-case VCE name with hyphens as underscores; ``None`` gives ``default``.

    Callers keep their own unset convention ("iid", "model", "robust", "").
    """
    if vce is None:
        return default
    return str(vce).lower().replace("-", "_")


def is_iid_vce(kind) -> bool:
    """Homoskedastic VCE aliases; an unset kind (``None``) is iid."""
    return kind is None or kind in IID_VCE_KINDS


def cluster_df_resid(df_resid, vce, clusters):
    """Cap residual DoF at ``min(G) - 1`` under clustered inference."""
    if normalize_vce(vce) == "cluster" and clusters:
        return min(df_resid, float(min(len(np.unique(c)) for c in clusters) - 1))
    return df_resid


def _sym_inv(a: np.ndarray) -> np.ndarray:
    return equilibrated_gram_inverse(a)


def _singleton_codes(codes):
    return bool(len(codes) and int(np.max(codes)) + 1 == len(codes)
                and np.all(np.bincount(codes, minlength=len(codes)) == 1))


def _independent_meat_xe(X, e):
    """Singleton intersections are HC0; bound the temporary score block."""
    p = X.shape[1]
    out = np.zeros((p, p))
    chunk = max(1, min(16384, 524288 // max(p, 1)))
    for lo in range(0, len(e), chunk):
        score = X[lo:lo+chunk] * e[lo:lo+chunk, None]
        out += score.T @ score
    return out


def _cluster_meat(scores: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, int]:
    codes = np.asarray(codes, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    g = int(codes.max()) + 1 if len(codes) else 0
    p = scores.shape[1]
    if _singleton_codes(codes):
        return scores.T @ scores, g
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
            if _singleton_codes(codes):
                yield sign, _independent_meat_xe(X, base), g, r
                continue
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


_KRON_GROUP_BYTES = 256 * 1024**2


def _cluster_kron_meat(V, Z, clusters):
    """Raw CGM covariance of Kronecker scores, aggregated without an N x K array."""
    clusters = tuple(clusters or ())
    if not clusters:
        raise ValueError("cluster covariance requires cluster arrays")
    groups = [int(np.max(c)) + 1 if len(c) else 0 for c in clusters]
    if min(groups) < 2:
        raise InferenceError(
            "cluster inference requires at least two clusters in every dimension",
            code="inference.insufficient_clusters", details={"cluster_counts": groups},
        )
    width = V.shape[1] * Z.shape[1]
    total = np.zeros((width, width))
    def scores(left, right):
        return (left[:, :, None] * right[:, None, :]).reshape(len(left), width)
    for size in range(1, len(clusters) + 1):
        for subset in combinations(range(len(clusters)), size):
            codes = clusters[subset[0]] if size == 1 else _intersection_codes_many([clusters[j] for j in subset])
            codes = np.asarray(codes, dtype=np.int64)
            g = int(codes.max()) + 1 if len(codes) else 0
            if _singleton_codes(codes):
                meat = np.zeros((width, width))
                chunk = max(1, min(16384, 524288 // max(width, 1)))
                for lo in range(0, len(V), chunk):
                    score = scores(V[lo:lo+chunk], Z[lo:lo+chunk])
                    meat += score.T @ score
            else:
                # Bound group-score storage too. Tile complete groups so the
                # cross-products sum exactly, without expanding N x width.
                group_rows = max(1, _KRON_GROUP_BYTES // (8 * max(width, 1)))
                order = np.argsort(codes, kind="stable") if g > group_rows else None
                sorted_codes = codes[order] if order is not None else None
                meat = np.zeros((width, width))
                for start in range(0, g, group_rows):
                    stop = min(start + group_rows, g)
                    if order is None:
                        rows, local_codes = slice(None), codes
                    else:
                        lo, hi = np.searchsorted(sorted_codes, [start, stop])
                        rows = order[lo:hi]
                        local_codes = sorted_codes[lo:hi] - start
                    sums = np.empty((stop - start, width))
                    for j in range(V.shape[1]):
                        for k in range(Z.shape[1]):
                            sums[:, j * Z.shape[1] + k] = np.bincount(
                                local_codes, weights=V[rows, j] * Z[rows, k],
                                minlength=stop - start)
                    meat += sums.T @ sums
            total += (1. if size % 2 else -1.) * meat
    return (total + total.T) / 2.


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


def _fix_psd(v: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues of a symmetric matrix (Cameron-Gelbach-Miller)."""
    v = (np.asarray(v, dtype=np.float64) + np.asarray(v, dtype=np.float64).T) / 2.0
    evals, evecs = np.linalg.eigh(v)
    if np.min(evals) >= 0.0:
        return v
    evals = np.clip(evals, 0.0, None)
    out = (evecs * evals) @ evecs.T
    return (out + out.T) / 2.0


def _is_multiway_cluster(kind, clusters) -> bool:
    return normalize_vce(kind) == "cluster" and clusters is not None and len(clusters) > 1


def regressor_scale(*designs, frequency_weights=None) -> np.ndarray:
    """reghdfe's standardization scale of the original regressor columns.

    Unweighted sample standard deviation (N-1 denominator) of each column on
    the estimation sample, before FE demeaning, floored at 1e-3
    (``reghdfe_standardize`` in reghdfe 6.12.5). Dense arrays and
    BlockDesigns are concatenated in argument order. With integer
    ``frequency_weights`` it is the stdev of the expanded rows (the unified
    policy; see ``SamplePolicy.psd_frequency_scale``).
    """
    parts = [
        (A.column_moments(frequency_weights=frequency_weights) if isinstance(A, BlockDesign)
         else column_moments(A, frequency_weights=frequency_weights)).sample_scale()
        for A in designs
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def finalize_cluster_vcov(V, kind, clusters, scale, constant_index=None):
    """Repair a multiway-cluster covariance in regressor-stdev coordinates.

    The signed CGM sandwich can be indefinite. Upstream clips its negative
    eigenvalues in the standardized coordinates of the original regressors.
    This separates the repair from solver scaling; regressor-unit invariance
    holds where the upstream stdev floor does not bind. ``scale()`` returns
    those column scales expressed in the
    coordinates of ``V`` (``regressor_scale`` for reported units); it is
    evaluated only when a repair can apply. With a reported constant at
    ``constant_index`` (scale 1), keep the independently clipped slope block
    so that reporting a constant cannot change slope inference. Merely
    inserting that block into the full spectral clip, as upstream reghdfe
    does, can leave the full covariance indefinite. Complete it instead:
    project the full clip's constant cross-covariances into the numerical
    range of the slope block and increase the constant variance, if needed,
    to satisfy its Schur-complement bound. This is a PSD completion policy,
    not a claim of a nearest-matrix solution or full-matrix Stata parity.
    Any other VCE is returned unchanged.
    """
    if not _is_multiway_cluster(kind, clusters):
        return V
    V = np.asarray(V, dtype=np.float64)
    V = (V + V.T) / 2.0
    s = np.array(scale(), dtype=np.float64)
    if constant_index is not None:
        s[constant_index] = 1.0
    outer = np.outer(s, s)
    W = V * outer
    evals, evecs = np.linalg.eigh(W)
    if np.min(evals) >= 0.0:
        return V
    fixed = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
    fixed = (fixed + fixed.T) / 2.0
    if constant_index is not None:
        indices = np.delete(np.arange(len(s)), constant_index)
        slopes = np.ix_(indices, indices)
        if len(indices):
            A = W[slopes]
            values, vectors = np.linalg.eigh(A)
            if values[0] < 0.0:
                A = (vectors * np.maximum(values, 0.0)) @ vectors.T
                A = (A + A.T) / 2.0
            # Keep the slope block itself unchanged. Numerically null slope
            # directions must have zero covariance with the constant; using
            # their noisy inverse would instead inflate its variance without
            # bound. The cutoff is relative to A, not to an absolute unit.
            cutoff = np.finfo(float).eps * len(indices) * max(values[-1], 0.0)
            active = values > cutoff
            coordinates = vectors[:, active].T @ fixed[indices, constant_index]
            cross = vectors[:, active] @ coordinates
            root = coordinates / np.sqrt(values[active])
            with np.errstate(over="ignore", invalid="ignore"):
                bound = float(root @ root)
                # An upward rounding guard prevents a binding Schur
                # complement from becoming negative through dot-product
                # roundoff. Null slope directions still have zero variance.
                bound *= 1.0 + 8.0 * np.finfo(float).eps * len(indices)
            if not np.isfinite(bound):
                raise InferenceError(
                    "constant covariance PSD completion exceeds numerical range",
                    details={"n_slopes": int(len(indices))},
                )
            fixed[slopes] = A
            fixed[indices, constant_index] = cross
            fixed[constant_index, indices] = cross
            fixed[constant_index, constant_index] = max(
                fixed[constant_index, constant_index], bound,
            )
    return fixed / outer


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
    if kind in ROBUST_VCE_KINDS:
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
    if is_iid_vce(kind):
        return scores.T @ scores
    raise ValueError(f"unknown covariance kind: {kind}")


def sandwich_vcov_xe(
    X, e, bread, *, kind="robust", clusters=None, k_total=None, nested_adj=0,
    time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None,
    score_scale=None, chunk_rows=250_000,
):
    """Sandwich covariance for scores ``X_i * e_i`` without an N x K score copy.

    Multiway-cluster results are unrepaired; see ``finalize_cluster_vcov``.

    Robust and clustered paths stream or aggregate directly from ``X`` and the
    scalar score multiplier.  HAC/DK paths retain the generic score-matrix
    implementation because they need observation ordering across all columns.
    """
    X = np.asarray(X, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    n, k = X.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    kind0 = normalize_vce(kind, "iid")
    base = e if score_scale is None else e * np.asarray(score_scale, dtype=np.float64)
    if kind0 in ROBUST_VCE_KINDS:
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
    return bread @ meat @ bread



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

    Multiway-cluster results are unrepaired; see ``finalize_cluster_vcov``.

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
    kind0 = normalize_vce(kind, "iid")
    base = e if score_scale is None else e * np.asarray(score_scale, dtype=np.float64)

    if kind0 in ROBUST_VCE_KINDS:
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
    return np.asarray(bread, dtype=np.float64) @ meat @ np.asarray(bread, dtype=np.float64)


def block_cluster_meat_xe(X: BlockDesign, e, clusters, *, score_scale=None):
    """Public exact cluster meat for a BlockDesign and scalar score multiplier."""
    return _block_cluster_meat_xe(X, e, clusters, score_scale=score_scale)


def ols_vcov(Xw, ew, bread, *, kind="iid", clusters=None, k_total=None, nested_adj=0,
             time=None, panel=None, bandwidth=None, kernel="bartlett", effective_n=None,
             score_scale=None):
    n, k = Xw.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    if is_iid_vce(kind):
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
    effective_n=None, score_scale=None, kappa=1.0, projected_X=None,
):
    """Sandwich using H=(1-kappa)X+kappa Pz X for fixed k-class.

    The default retains the conventional IV/LIML covariance path. Explicit
    fixed k-class callers pass their kappa; kappa=0 then reduces to OLS.
    """
    n, k = Xw.shape
    n_eff = float(n if effective_n is None else effective_n)
    k_total = k if k_total is None else int(k_total)
    if is_iid_vce(kind) and kappa in (0.0, 1.0):
        df = max(n_eff - k_total, 1)
        return bread * (float(ew @ ew) / df)
    if kappa == 0.0:
        xhat = Xw
    else:
        if projected_X is None:
            gamma = _sym_inv(Zw.T @ Zw) @ (Zw.T @ Xw)
            xhat = Zw @ gamma
        else:
            xhat = projected_X
        if kappa != 1.0:
            xhat = (1.0 - kappa) * Xw + kappa * xhat
    if is_iid_vce(kind):
        sigma2 = float(ew @ ew) / max(n_eff - k_total, 1)
        return sigma2 * (bread @ (xhat.T @ xhat) @ bread.T)
    return sandwich_vcov_xe(
        xhat, ew, bread, kind=kind, clusters=clusters, k_total=k_total,
        nested_adj=nested_adj, time=time, panel=panel, bandwidth=bandwidth,
        kernel=kernel, effective_n=n_eff, score_scale=score_scale,
    )


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
    if is_iid_vce(kind):
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
    return (V + V.T) / 2
