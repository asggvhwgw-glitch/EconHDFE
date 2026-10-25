from __future__ import annotations
import math
import numpy as np

from ..compute.clusters import normalize_cluster_arrays
from ..errors import InputError, error_boundary
from ..hdfe.absorber import HDFEAbsorber
from ..hdfe.workspace import _absorption_workspace_cost
from ..resampling.sampling import wild_weights
from ..resampling.engine import run_replicates, _wild_memory_estimate, _check_worker_memory
from ..planner.scheduler import current_allocation, managed_execution


@error_boundary("ols_bootstrap")
@managed_execution
def wild_cluster_bootstrap_ols(*, absorber: HDFEAbsorber, y_within, X_within,
                               beta, residuals, clusters=None, weights=None,
                               reps=999, n_jobs=-1, batch_size=8, seed=0,
                               weight_distribution="rademacher", execution_config=None):
    """Backward-compatible coefficient-draw wild bootstrap for HDFE OLS.

    Weighted fits use the same WLS metric as the fitted model. For hypothesis
    testing prefer ``wild_cluster_test_ols`` (WCR11/WCU11), which studentizes
    each bootstrap replication and can impose the null.
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
    setup_bytes, worker_bytes = _wild_memory_estimate(
        _absorption_workspace_cost(absorber), nobs=n, ncols=X.shape[1],
        groups=G, batch_size=batch_size, reps=reps,
    )
    _check_worker_memory(worker_bytes, setup_bytes=setup_bytes)
    sw = np.ones(n, dtype=np.float64) if w is None else np.sqrt(w)
    Xw = X * sw[:, None]
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)
    fitted = X @ beta
    seeds = np.random.SeedSequence(seed).spawn(math.ceil(reps / batch_size))
    sizes = [min(batch_size, reps-i) for i in range(0, reps, batch_size)]
    def one_batch(item):
        ss, b = item
        fe_threads = min(int(getattr(absorber, "absorb_threads", 1)), current_allocation().cpu_threads)
        rng = np.random.default_rng(ss)
        mult_g = wild_weights(rng, (G, b), weight_distribution)
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
