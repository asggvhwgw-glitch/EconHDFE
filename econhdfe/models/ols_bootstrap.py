from __future__ import annotations
import math
from contextlib import nullcontext
import numpy as np

from ..compute.clusters import normalize_cluster_arrays
from ..errors import InputError, error_boundary
from ..hdfe.absorber import HDFEAbsorber, HDFEConvergenceError
from ..hdfe.within_backend import WithinFEAbsorber
from ..hdfe.workspace import _absorption_workspace_cost
from ..resampling.sampling import wild_weights
from ..resampling.engine import (
    run_replicates, _wild_memory_estimate, _wild_score_memory_estimate,
    _check_worker_memory,
)
from ..planner.scheduler import current_allocation, managed_execution, reserve_memory


def _same_projection_metric(absorber, weights):
    """Only known CPU weighted-orthogonal projectors admit the adjoint shortcut.

    The low-level API historically permits a different WLS weight vector or a
    custom residualizer; those calls retain their original per-batch algorithm.
    Exact equality is conservative: proportional weights need not be detected.
    """
    # TwoWayFEAbsorber can report false nonconvergence when re-projecting an
    # already-within design (its PCG denominator is a near-zero Schur RHS).
    # Keep its legacy route until that solver has a reliable final certificate.
    if type(absorber) not in (HDFEAbsorber, WithinFEAbsorber):
        return False
    if getattr(absorber, "backend", "numpy") != "numpy":
        return False
    if np.dtype(getattr(absorber, "dtype", np.float64)) != np.dtype(np.float64):
        return False
    aw = absorber.weights
    if weights is not None and (not np.isfinite(weights).all() or np.any(weights <= 0)):
        return False
    if weights is None:
        return aw is None or bool(np.all(aw == 1.0))
    if aw is None:
        return bool(np.all(weights == 1.0))
    return bool(np.array_equal(weights, aw))


@error_boundary("ols_bootstrap")
@managed_execution
def wild_cluster_bootstrap_ols(*, absorber: HDFEAbsorber, y_within, X_within,
                               beta, residuals, clusters=None, weights=None,
                               reps=999, n_jobs=-1, batch_size=8, seed=0,
                               weight_distribution="rademacher", execution_config=None):
    """Coefficient-draw wild bootstrap with fixed-design score aggregation.

    Weighted fits use the same WLS metric as the fitted model. For hypothesis
    testing prefer ``wild_cluster_test_ols`` (WCR11/WCU11), which studentizes
    each bootstrap replication and can impose the null. Known same-metric CPU
    projectors absorb the design once, then reuse cluster scores for all draws.
    The general pinv center, multiplier batches and low-level fallback are kept.
    """
    y = np.asarray(y_within, dtype=np.float64)
    X = np.asarray(X_within, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    u = np.asarray(residuals, dtype=np.float64)
    n = len(y)
    if int(reps) < 1 or int(batch_size) < 1:
        raise InputError("reps and batch_size must be positive", code="resampling.batch_size")
    if clusters is None:
        dense = np.arange(n, dtype=np.int32); G = n
    else:
        dense = normalize_cluster_arrays(clusters, nobs=n, required=True)[0]
        G = int(dense.max()) + 1
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    score_route = 0 < X.shape[1] <= int(reps) and _same_projection_metric(absorber, w)
    memory_estimate = _wild_score_memory_estimate if score_route else _wild_memory_estimate
    setup_bytes, worker_bytes = memory_estimate(
        _absorption_workspace_cost(absorber), nobs=n, ncols=X.shape[1],
        groups=G, batch_size=batch_size, reps=reps,
    )
    if score_route and setup_bytes + worker_bytes > current_allocation().remaining_memory_bytes():
        # A wide one-time projection must not reject a formerly feasible call
        # whose narrow legacy RHS batches fit the same resource envelope.
        score_route = False
        setup_bytes, worker_bytes = _wild_memory_estimate(
            _absorption_workspace_cost(absorber), nobs=n, ncols=X.shape[1],
            groups=G, batch_size=batch_size, reps=reps,
        )
    _check_worker_memory(worker_bytes, setup_bytes=setup_bytes)
    scores = None
    preparation = (reserve_memory(setup_bytes, label="wild bootstrap score preparation")
                   if score_route else nullcontext())
    with preparation:
        sw = np.ones(n, dtype=np.float64) if w is None else np.sqrt(w)
        Xw = X * sw[:, None]
        gram = Xw.T @ Xw
        bread = np.linalg.pinv(gram, hermitian=True)
        if score_route:
            # Reordered products can amplify rounding in a weak retained
            # design. Preserve the legacy arithmetic without changing pinv's
            # rank/cutoff policy; exact redundant columns need not fall back.
            retained_condition = np.linalg.norm(gram, ord=np.inf) * np.linalg.norm(bread, ord=np.inf)
            if (not np.isfinite(retained_condition)
                    or retained_condition > 1 / np.sqrt(np.finfo(np.float64).eps)):
                score_route = False
                setup_bytes, worker_bytes = _wild_memory_estimate(
                    _absorption_workspace_cost(absorber), nobs=n, ncols=X.shape[1],
                    groups=G, batch_size=batch_size, reps=reps,
                )
                _check_worker_memory(worker_bytes, setup_bytes=setup_bytes)
        del gram
        fitted = X @ beta
        if score_route:
            fe_threads = min(int(getattr(absorber, "absorb_threads", 1)),
                             current_allocation().cpu_threads)
            projected, info = absorber.residualize(
                X, copy=True, return_info=True, absorb_threads=fe_threads,
            )
            if not info.converged:
                raise HDFEConvergenceError(info)
            # M'W=WM gives X'WM=(MX)'W, including raw/imperfectly within X.
            # Keep B=(X'WX)^+ and its general center; B Q beta need not be beta.
            center = bread @ (projected.T @ (fitted if w is None else w*fitted))
            weighted_residual = u if w is None else w*u
            scores = np.empty((G, X.shape[1]), dtype=np.float64)
            for j in range(X.shape[1]):
                scores[:, j] = np.bincount(
                    dense, weights=projected[:, j]*weighted_residual, minlength=G,
                )
            del projected, weighted_residual, fitted, Xw, sw
    seeds = np.random.SeedSequence(seed).spawn(math.ceil(reps / batch_size))
    sizes = [min(batch_size, reps-i) for i in range(0, reps, batch_size)]
    def one_batch(item):
        ss, b = item
        rng = np.random.default_rng(ss)
        mult_g = wild_weights(rng, (G, b), weight_distribution)
        if scores is not None:
            return (center[:, None] + bread @ (scores.T @ mult_g)).T
        fe_threads = min(int(getattr(absorber, "absorb_threads", 1)), current_allocation().cpu_threads)
        ystar = fitted[:, None] + u[:, None] * mult_g[dense]
        ystar = absorber.residualize(ystar, absorb_threads=fe_threads)
        coef = bread @ (Xw.T @ (ystar * sw[:, None]))
        return coef.T

    batches = run_replicates(
        one_batch, list(zip(seeds, sizes, strict=True)), n_jobs=n_jobs, skip_exceptions=(),
        estimated_worker_bytes=worker_bytes,
    )
    return np.vstack(batches.values)[:reps]


@error_boundary("ols_bootstrap")
@managed_execution
def wild_bootstrap(result, *, reps=999, n_jobs=-1, batch_size=8, seed=0, cluster_index=0,
                   weight_distribution="rademacher", execution_config=None):
    state = getattr(result, "state", None)
    if state is None:
        raise InputError(
            "refit with keep_state=True to reuse the FE absorber for bootstrap",
            code="resampling.missing_state", stage="resampling",
            suggestion="Call olshdfe(..., keep_state=True) before wild_bootstrap().",
        )
    cluster = None
    if state.clusters:
        idx = int(cluster_index)
        if not -len(state.clusters) <= idx < len(state.clusters):
            raise InputError("cluster_index is out of range", code="resampling.cluster_index", stage="resampling")
        cluster = state.clusters[idx]
    return wild_cluster_bootstrap_ols(
        absorber=state.absorber, y_within=state.y_within, X_within=state.X_within,
        beta=result.params, residuals=result.residuals, clusters=cluster, weights=state.weights,
        reps=reps, n_jobs=n_jobs, batch_size=batch_size, seed=seed,
        weight_distribution=weight_distribution,
    )
