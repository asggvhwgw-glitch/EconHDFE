from __future__ import annotations

import math
import numpy as np

from ...compute.clusters import normalize_cluster_arrays
from ...errors import BootstrapError, InferenceError, InputError, error_boundary
from ...hdfe.protocol import capability
from ...resampling.sampling import wild_weights
from ...resampling.engine import run_replicates, wild_memory_estimate, check_worker_memory
from ...planner.scheduler import current_allocation, managed_execution
from .results import WildClusterTestResult


def _state_arrays(result):
    state = getattr(result, "state", None)
    if state is None:
        raise InputError(
            "refit with keep_state=True for wild cluster inference",
            code="resampling.missing_state", stage="resampling",
            suggestion="Call olshdfe(..., keep_state=True) before wild_cluster_test_ols().",
        )
    if getattr(result, "estimator", None) != "ols":
        raise InferenceError(
            "wild_cluster_test_ols supports OLS-HDFE results only",
            code="resampling.estimator", stage="resampling",
        )
    if not capability(state.absorber, "supports_wild_cluster_test"):
        raise InferenceError(
            f"wild cluster inference is not yet certified for absorber type {type(state.absorber).__name__}",
            code="resampling.absorber", stage="resampling",
        )
    clusters = normalize_cluster_arrays(state.clusters, nobs=len(state.y_within), required=True)
    if len(clusters) != 1:
        raise InferenceError(
            "wild_cluster_test_ols currently requires one-way clustering",
            code="resampling.multiway_cluster", stage="resampling",
            suggestion="Use CRV1 multi-way clustering for the main fit; WCR11/WCU11 are currently certified for one-way clustering.",
        )
    y = np.asarray(state.y_within, dtype=np.float64)
    X = np.asarray(state.X_within, dtype=np.float64)
    w = None if state.weights is None else np.asarray(state.weights, dtype=np.float64)
    return state, y, X, w, clusters[0]


def _restriction(result, *, param=None, R=None, r=0.0):
    p = len(np.asarray(result.params))
    if R is not None and param is not None:
        raise InputError("specify either param or R, not both", code="resampling.restriction", stage="resampling")
    if R is None:
        if param is None:
            if p != 1:
                raise InputError("param or R is required when the model has multiple coefficients", code="resampling.restriction", stage="resampling")
            idx = 0
        elif isinstance(param, str):
            try:
                idx = tuple(result.names).index(param)
            except ValueError as exc:
                raise InputError(
                    f"unknown coefficient name {param!r}", code="resampling.restriction", stage="resampling",
                    details={"names": tuple(result.names)},
                ) from exc
        else:
            idx = int(param)
            if idx < 0:
                idx += p
            if not 0 <= idx < p:
                raise InputError("param index is out of range", code="resampling.restriction", stage="resampling")
        Rm = np.zeros((1, p), dtype=np.float64)
        Rm[0, idx] = 1.0
    else:
        Rm = np.asarray(R, dtype=np.float64)
        if Rm.ndim == 1:
            Rm = Rm[None, :]
        if Rm.ndim != 2 or Rm.shape[1] != p or Rm.shape[0] < 1:
            raise InputError(
                "R must be q x p with p equal to the number of active coefficients",
                code="resampling.restriction", stage="resampling",
                details={"expected_columns": p, "shape": Rm.shape},
            )
    rv = np.asarray(r, dtype=np.float64)
    rv = np.full(Rm.shape[0], float(rv), dtype=np.float64) if rv.ndim == 0 else rv.reshape(-1)
    if len(rv) != Rm.shape[0]:
        raise InputError("r must have one value per row of R", code="resampling.restriction", stage="resampling")
    if np.linalg.matrix_rank(Rm) < Rm.shape[0]:
        raise InputError("rows of R must be linearly independent", code="resampling.restriction_rank", stage="resampling")
    return Rm, rv


def _cluster_scale(n_eff: float, k_total: int, G: int, nested_adj: int) -> float:
    if G <= 1 or n_eff <= k_total + nested_adj:
        return 1.0
    return ((n_eff - 1.0) / (n_eff - k_total - nested_adj)) * (G / (G - 1.0))


def _restriction_cov_from_residuals(residuals, influence_design, cluster, *, scale: float):
    q = influence_design.shape[1]
    G = int(cluster.max()) + 1
    sums = np.empty((G, q), dtype=np.float64)
    for j in range(q):
        sums[:, j] = np.bincount(cluster, weights=influence_design[:, j] * residuals, minlength=G)
    out = scale * (sums.T @ sums)
    return (out + out.T) / 2.0


def _test_stat(diff, covariance) -> float:
    q = len(diff)
    if q == 1:
        var = float(covariance[0, 0])
        if not np.isfinite(var) or var <= np.finfo(np.float64).tiny:
            raise BootstrapError("restriction variance is zero or non-finite", code="resampling.zero_variance", stage="resampling")
        return float(diff[0] / math.sqrt(var))
    stat = float(diff @ np.linalg.pinv(covariance, hermitian=True) @ diff)
    if not np.isfinite(stat):
        raise BootstrapError("non-finite bootstrap Wald statistic", code="resampling.nonfinite_statistic")
    return stat


def _alternative_pvalue(observed, draws, *, alternative: str, joint: bool, exact: bool = False) -> float:
    B = len(draws)
    if B == 0:
        raise BootstrapError("no finite bootstrap statistics were produced", code="resampling.no_valid_replicates")
    alt = str(alternative).lower().replace("-", "_")
    tol = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(float(observed)))
    if joint:
        if alt not in {"two_sided", "two_tailed", "greater"}:
            raise InputError("joint Wald bootstrap supports alternative='two_sided' only", code="resampling.alternative")
        exceed = int(np.sum(draws >= observed - tol))
    elif alt in {"two_sided", "two_tailed"}:
        exceed = int(np.sum(np.abs(draws) >= abs(observed) - tol))
    elif alt in {"greater", "right"}:
        exceed = int(np.sum(draws >= observed - tol))
    elif alt in {"less", "left"}:
        exceed = int(np.sum(draws <= observed + tol))
    elif alt in {"equal_tailed", "equal_tail"}:
        if exact:
            lo = float(np.mean(draws <= observed + tol)); hi = float(np.mean(draws >= observed - tol))
        else:
            lo = (1 + int(np.sum(draws <= observed + tol))) / (B + 1.0)
            hi = (1 + int(np.sum(draws >= observed - tol))) / (B + 1.0)
        return float(min(1.0, 2.0 * min(lo, hi)))
    else:
        raise InputError("alternative must be two_sided/equal_tailed/greater/less", code="resampling.alternative", stage="resampling")
    return float(exceed / B) if exact else float((1.0 + exceed) / (B + 1.0))


@error_boundary("ols_bootstrap")
@managed_execution
def wild_cluster_test_ols(
    result, *, param=None, R=None, r=0.0, reps=9_999,
    impose_null: bool = True, weight_distribution: str = "rademacher",
    alternative: str = "two_sided", n_jobs: int = 1, batch_size: int = 32,
    seed: int = 0, execution_config=None,
) -> WildClusterTestResult:
    """Studentized one-way WCR11/WCU11 test for OLS-HDFE.

    ``impose_null=True`` implements WCR11; ``False`` implements WCU11.
    Scalar hypotheses use a signed t statistic and joint hypotheses a Wald
    statistic. For Rademacher weights and sufficiently few clusters, all 2^G
    sign vectors are enumerated exactly when ``reps`` requests at least 2^G.
    """
    if int(reps) < 99:
        raise InputError("reps must be at least 99 for bootstrap inference", code="resampling.reps", stage="resampling")
    if int(batch_size) < 1:
        raise InputError("batch_size must be positive", code="resampling.batch_size", stage="resampling")
    state, y, X, w, cluster = _state_arrays(result)
    Rm, rv = _restriction(result, param=param, R=R, r=r)
    beta_hat = np.asarray(result.params, dtype=np.float64)
    n = len(y); G = int(cluster.max()) + 1
    requested_B = int(reps)
    dist0 = str(weight_distribution).lower().replace("-", "_")
    full_enumeration = bool(dist0 in {"rademacher", "radem", "rad"} and G <= 16 and (1 << G) <= requested_B)
    B = (1 << G) if full_enumeration else requested_B
    setup_bytes, worker_bytes = wild_memory_estimate(
        state.absorber.workspace_cost(), nobs=n, ncols=X.shape[1],
        groups=G, batch_size=batch_size,
        reps=B, restrictions=Rm.shape[0], enumeration=full_enumeration,
    )
    check_worker_memory(worker_bytes, setup_bytes=setup_bytes)
    sw = np.ones(n, dtype=np.float64) if w is None else np.sqrt(w)
    Xw = X * sw[:, None]
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)
    k_total = int(result.rank) + int(result.df_absorbed)
    n_eff = float(np.sum(w)) if getattr(state, "weight_type", "none") == "fweight" and w is not None else float(n)
    nested_adj = int(bool(getattr(getattr(result, "dof_info", None), "nested", 0)))
    scale = _cluster_scale(n_eff, k_total, G, nested_adj)

    diff_obs = Rm @ beta_hat - rv
    influence_design = Xw @ (bread @ Rm.T)
    ew_hat = np.asarray(result.residuals, dtype=np.float64) * sw
    observed = _test_stat(diff_obs, _restriction_cov_from_residuals(ew_hat, influence_design, cluster, scale=scale))

    if impose_null:
        middle = np.linalg.pinv(Rm @ bread @ Rm.T, hermitian=True)
        beta_r = beta_hat - bread @ Rm.T @ (middle @ diff_obs)
        base_resid = y - X @ beta_r
        boot_type = "WCR11"
    else:
        base_resid = np.asarray(result.residuals, dtype=np.float64)
        boot_type = "WCU11"
    fitted_unrestricted = X @ beta_hat

    if full_enumeration:
        B = 1 << G
        ids = np.arange(B, dtype=np.uint64)
        vg_full = np.empty((G, B), dtype=np.float64)
        for g in range(G):
            vg_full[g] = (((ids >> np.uint64(g)) & np.uint64(1)).astype(np.float64) * 2.0 - 1.0)
        starts = list(range(0, B, int(batch_size)))
        sizes = [min(int(batch_size), B - i) for i in starts]
        seeds = [None] * len(sizes)
    else:
        B = requested_B; vg_full = None
        starts = list(range(0, B, int(batch_size)))
        sizes = [min(int(batch_size), B - i) for i in starts]
        seeds = np.random.SeedSequence(seed).spawn(len(sizes))

    def one_batch(item):
        ss, b, start = item
        fe_threads = min(int(getattr(state.absorber, "absorb_threads", 1)), current_allocation().cpu_threads)
        vg = wild_weights(np.random.default_rng(ss), (G, b), weight_distribution) if vg_full is None else vg_full[:, start:start+b]
        ystar = fitted_unrestricted[:, None] + base_resid[:, None] * vg[cluster]
        ystar = state.absorber.residualize(ystar, absorb_threads=fe_threads)
        ystar_w = ystar * sw[:, None]
        betas = bread @ (Xw.T @ ystar_w)
        out = np.empty(b, dtype=np.float64)
        for j in range(b):
            resid = (ystar[:, j] - X @ betas[:, j]) * sw
            cov = _restriction_cov_from_residuals(resid, influence_design, cluster, scale=scale)
            try:
                out[j] = _test_stat(Rm @ (betas[:, j] - beta_hat), cov)
            except BootstrapError:
                out[j] = np.nan
        return out

    batch = run_replicates(
        one_batch, list(zip(seeds, sizes, starts, strict=True)), n_jobs=n_jobs, skip_exceptions=(),
        estimated_worker_bytes=worker_bytes,
    )
    draws = np.concatenate(batch.values)[:B]
    finite = draws[np.isfinite(draws)]
    if len(finite) < max(99, int(0.95 * B)):
        raise BootstrapError(
            "too many bootstrap replications produced invalid studentized statistics",
            code="resampling.too_many_failures", stage="resampling",
            details={"requested": B, "completed": int(len(finite))},
        )
    pvalue = _alternative_pvalue(observed, finite, alternative=alternative, joint=Rm.shape[0] > 1, exact=full_enumeration and len(finite) == B)
    return WildClusterTestResult(
        statistic=observed, pvalue=pvalue, bootstrap_statistics=finite,
        reps=B, completed_reps=len(finite), bootstrap_type=boot_type,
        weight_distribution=str(weight_distribution).lower(), alternative=str(alternative).lower(),
        cluster_count=G, restriction_matrix=Rm, restriction_value=rv,
        observed_difference=diff_obs, full_enumeration=full_enumeration,
        requested_reps=requested_B,
    )
