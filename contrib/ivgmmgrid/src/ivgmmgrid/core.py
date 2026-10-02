"""Single-FE, single-cluster two-step efficient GMM.

The second-step covariance uses the first-step cluster moment covariance.
No Stata process, foreign-language extension, or EconHDFE internals are used.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class IdentificationError(ValueError):
    """The retained regressors or cluster moments cannot identify the model."""


@dataclass(frozen=True)
class ModelResult:
    coefficients: np.ndarray  # endogenous candidate first, followed by exog
    covariance: np.ndarray
    sample_indices: np.ndarray
    omitted: np.ndarray
    nobs: int
    nclusters: int
    df_resid: int  # cluster inference: G - 1
    residual_df: int  # RMSE divisor: N - retained k - absorbed adjustment
    absorbed_df: int
    rss: float
    rmse: float
    hansen_j: float
    hansen_df: int
    condition_omega: float  # after instrument scaling


@dataclass(frozen=True)
class GridResult:
    models: tuple
    sample_groups: int
    engine: str

    @property
    def coefficients(self):
        return np.stack([m.coefficients for m in self.models])

    @property
    def covariance(self):
        return np.stack([m.covariance for m in self.models])

    @property
    def rmse(self):
        return np.array([m.rmse for m in self.models])


def _matrix(x, n, name):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows")
    return a


def _labels(a, n, name):
    a = np.asarray(a)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{name} must be a vector of length {n}")
    # NumPy numeric/string labels; None and NaN labels are missing.
    valid = np.array([v is not None and bool(v == v) for v in a])
    if a.dtype.kind in "fc":
        valid &= np.isfinite(a)
    return a, valid


def _within(a, groups):
    counts = np.bincount(groups)
    sums = np.zeros((len(counts), a.shape[1]))
    np.add.at(sums, groups, a)
    return a - (sums / counts[:, None])[groups]


def _columns(a, tol):
    """Retain independent columns in input order, with unit-invariant tests."""
    norms = np.linalg.norm(a, axis=0)
    retained, basis = [], []
    for j, scale in enumerate(norms):
        if scale == 0:
            continue
        v = a[:, j] / scale
        # Reorthogonalization avoids cancellation from nearly dependent columns.
        for _ in range(2):
            for u in basis:
                v = v - u * (u @ v)
        size = np.linalg.norm(v)
        if size > tol:
            retained.append(j)
            basis.append(v / size)
    return np.asarray(retained, dtype=int), norms


def _solve(a, b, context):
    a = (a + a.T) * 0.5
    try:
        factor = cho_factor(a, lower=True, check_finite=False)
        out = cho_solve(factor, b, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise IdentificationError(f"{context} is not positive definite") from exc
    if not np.all(np.isfinite(out)):
        raise IdentificationError(f"{context} solve produced non-finite values")
    return out


def _estimate(y, x, z, cluster, sample, absorbed_df, rank_tol, moments=None):
    n, p = x.shape
    keep, scales = _columns(x, rank_tol)
    if not len(keep) or 0 not in keep:
        raise IdentificationError("endogenous candidate is absorbed or unidentified")
    xs = x[:, keep] / scales[keep]
    k = len(keep)
    q = z.shape[1]
    g = int(cluster.max()) + 1
    if q < k:
        raise IdentificationError("fewer independent instruments than regressors")
    if g <= q:
        raise IdentificationError("cluster moment covariance needs more clusters than instruments")
    residual_df = n - k - absorbed_df
    if residual_df <= 0:
        raise IdentificationError("non-positive residual degrees of freedom")
    h = z.T @ z / n
    zy = z.T @ y / n
    zx = z.T @ xs / n
    if np.linalg.matrix_rank(zx, tol=rank_tol * np.linalg.norm(zx, 2)) < k:
        raise IdentificationError("instruments do not identify the retained regressors")
    b1 = _solve(zx.T @ _solve(h, zx, "instrument cross-product"),
                zx.T @ _solve(h, zy, "instrument cross-product"), "first-step normal matrix")
    if moments is None:
        # Observation-level reference route: rebuild each score for each fit.
        score = z * (y - xs @ b1)[:, None]
        cluster_score = np.zeros((g, q))
        np.add.at(cluster_score, cluster, score)
    else:
        cy, cx = moments
        cluster_score = cy - np.einsum("gqk,k->gq", cx[:, :, keep] / scales[keep], b1)
    if np.linalg.matrix_rank(cluster_score, tol=rank_tol * np.linalg.norm(cluster_score, 2)) < q:
        raise IdentificationError("cluster moment covariance is rank deficient")
    omega = cluster_score.T @ cluster_score / n
    ow = _solve(omega, np.column_stack([zx, zy]), "cluster moment covariance")
    a = zx.T @ ow[:, :k]
    b2 = _solve(a, zx.T @ ow[:, k], "second-step normal matrix")
    qsmall = (n - 1) / residual_df * g / (g - 1)
    v = _solve(a, np.eye(k), "second-step normal matrix") / n * qsmall
    residual = y - xs @ b2
    rss = float(residual @ residual)
    mean_score = zy - zx @ b2
    jstat = float(n * (mean_score @ _solve(omega, mean_score, "cluster moment covariance")))
    beta = np.zeros(p)
    beta[keep] = b2 / scales[keep]
    cov = np.zeros((p, p))
    cov[np.ix_(keep, keep)] = v / np.outer(scales[keep], scales[keep])
    omitted = np.ones(p, dtype=bool)
    omitted[keep] = False
    return ModelResult(beta, cov, sample.copy(), omitted, n, g, g - 1,
                       residual_df, absorbed_df, rss, float(np.sqrt(rss / residual_df)),
                       max(0.0, jstat), q - k, float(np.linalg.cond(omega)))


def fit_grid(y, exog, candidates, instruments, absorb, cluster, *, mask=None,
             engine="cached", verify=False, rank_tol=1e-10):
    """Fit one endogenous candidate per column, absorbing one fixed effect.

    Missing observations and singleton FE groups are removed independently for
    each candidate. Include exogenous controls in ``exog``; the function adds
    them to the instrument matrix. No intercept is needed after absorption.
    Collinear regressors are omitted in [candidate, exog] order. Rank failures
    raise IdentificationError; they are never concealed by a pseudoinverse.

    ``engine='reference'`` rebuilds observation-level cluster scores separately.
    ``verify=True`` compares the cached results with that Python reference.
    This check is not an independent external statistical implementation.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = len(y)
    e = _matrix(exog, n, "exog")
    c = _matrix(candidates, n, "candidates")
    z = _matrix(instruments, n, "instruments")
    if c.shape[1] == 0 or z.shape[1] == 0:
        raise ValueError("at least one candidate and excluded instrument are required")
    fe, fe_valid = _labels(absorb, n, "absorb")
    cl, cl_valid = _labels(cluster, n, "cluster")
    if engine not in ("cached", "reference"):
        raise ValueError("engine must be cached or reference")
    if not np.isfinite(rank_tol) or not 0 < rank_tol < 1:
        raise ValueError("rank_tol must be between zero and one")
    common = np.isfinite(y) & np.isfinite(e).all(1) & np.isfinite(z).all(1) & fe_valid & cl_valid
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (n,):
            raise ValueError("mask must be a boolean vector matching y")
        common &= mask
    partitions = {}
    for j in range(c.shape[1]):
        valid = common & np.isfinite(c[:, j])
        ids = np.flatnonzero(valid)
        if len(ids):
            _, codes = np.unique(fe[ids], return_inverse=True)
            ids = ids[np.bincount(codes)[codes] > 1]
        if not len(ids):
            raise ValueError(f"candidate {j} has no observations after sample filtering")
        key = ids.tobytes() if engine == "cached" else (j, ids.tobytes())
        if key not in partitions:
            partitions[key] = (ids, [])
        partitions[key][1].append(j)
    results = [None] * c.shape[1]
    for ids, js in partitions.values():
        _, f = np.unique(fe[ids], return_inverse=True)
        _, g = np.unique(cl[ids], return_inverse=True)
        # One FE: all levels nested in clusters => ivreghdfe small-sample +1.
        nested = all(len(np.unique(g[f == v])) == 1 for v in range(f.max() + 1))
        absorbed_df = 1 if nested else int(f.max()) + 1
        b = _within(np.column_stack([y[ids], e[ids], c[np.ix_(ids, js)]]), f)
        instruments_w = _within(np.column_stack([z[ids], e[ids]]), f)
        zk, zs = _columns(instruments_w, rank_tol)
        if not len(zk):
            raise IdentificationError("all instruments are absorbed")
        zw = instruments_w[:, zk] / zs[zk]
        cache = None
        if engine == "cached":
            cache = np.zeros((g.max() + 1, zw.shape[1], b.shape[1]))
            # Avoid an N x q x candidate tensor: memory scales with clusters.
            for col in range(b.shape[1]):
                np.add.at(cache[:, :, col], g, zw * b[:, col, None])
        for local, j in enumerate(js):
            cols = [1 + e.shape[1] + local, *range(1, 1 + e.shape[1])]
            moments = None if cache is None else (cache[:, :, 0], cache[:, :, cols])
            results[j] = _estimate(b[:, 0], b[:, cols], zw, g, ids, absorbed_df, rank_tol, moments)
    result = GridResult(tuple(results), len(partitions), engine)
    if verify and engine == "cached":
        ref = fit_grid(y, e, c, z, fe, cl, mask=mask, engine="reference", rank_tol=rank_tol)
        for actual, expected in zip(result.models, ref.models):
            for field in ("coefficients", "covariance", "rss", "rmse", "hansen_j"):
                np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=1e-7, atol=1e-10,
                                           err_msg=f"cached/reference mismatch: {field}")
    return result
