from __future__ import annotations
import math
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from threadpoolctl import threadpool_limits

from ..compute.clusters import normalize_cluster_arrays
from ..errors import InputError, error_boundary
from ..hdfe.absorber import HDFEAbsorber
from ..resampling.sampling import wild_weights


@error_boundary("ols_bootstrap")
def wild_cluster_bootstrap_ols(*, absorber: HDFEAbsorber, y_within, X_within,
                               beta, residuals, clusters=None, weights=None,
                               reps=999, n_jobs=-1, batch_size=8, seed=0,
                               weight_distribution="rademacher"):
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
    if clusters is None:
        dense = np.arange(n, dtype=np.int32); G = n
    else:
        dense = normalize_cluster_arrays(clusters, nobs=n, required=True)[0]
        G = int(dense.max()) + 1
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.ones(n, dtype=np.float64) if w is None else np.sqrt(w)
    Xw = X * sw[:, None]
    bread = np.linalg.pinv(Xw.T @ Xw, hermitian=True)
    fitted = X @ beta
    seeds = np.random.SeedSequence(seed).spawn(math.ceil(reps / batch_size))
    sizes = [min(batch_size, reps-i) for i in range(0, reps, batch_size)]
    workers = max(1, effective_n_jobs(n_jobs))
    fe_threads = max(1, int(getattr(absorber, "absorb_threads", 1)) // workers)

    def one_batch(ss, b):
        rng = np.random.default_rng(ss)
        mult_g = wild_weights(rng, (G, b), weight_distribution)
        ystar = fitted[:, None] + u[:, None] * mult_g[dense]
        ystar = absorber.residualize(ystar, absorb_threads=fe_threads)
        coef = bread @ (Xw.T @ (ystar * sw[:, None]))
        return coef.T

    # BLAS limits are process-wide: enter/restore once, not in racing workers.
    with threadpool_limits(limits=1):
        batches = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(one_batch)(ss, b) for ss, b in zip(seeds, sizes, strict=False)
        )
    return np.vstack(batches)[:reps]


@error_boundary("ols_bootstrap")
def wild_bootstrap(result, *, reps=999, n_jobs=-1, batch_size=8, seed=0, cluster_index=0,
                   weight_distribution="rademacher"):
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
