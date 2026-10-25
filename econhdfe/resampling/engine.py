from __future__ import annotations
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from threadpoolctl import threadpool_limits

from ..errors import EconHDFEError, SpecificationError, error_boundary
from ..planner.scheduler import current_allocation, managed_execution, reserve_memory, worker_scope
from .results import ReplicateBatch, ReplicateFailure


_IN_REPLICATES = ContextVar("econhdfe_in_replicates", default=False)


def _wild_memory_estimate(workspace_cost, *, nobs, ncols, groups, batch_size, reps,
                          restrictions=0, enumeration=False):
    """Conservative CPU wild-bootstrap setup and per-worker peak bytes.

    Includes absorber scratch/lazy state, RHS copies, multiplier generation,
    matrix products, studentization and retained output/concatenation. Existing
    absorber resident state is conservatively included by the shared inventory;
    inputs supplied by the caller and Python/JIT/BLAS runtime overhead are not
    modeled. Full output storage is charged per worker to bound accumulation
    while earlier batches finish. This is an estimate, not an RSS guarantee.
    """
    fixed, per_column = workspace_cost
    n, k, g = int(nobs), int(ncols), int(groups)
    b, total, q = min(int(batch_size), int(reps)), int(reps), int(restrictions)
    setup = 8 * (n*k + 8*k*k + 6*n + n*q + k*q + 8*q*q)
    output_width = 1 if q else k
    output = 8 * (2*total*output_width + (g*total + 3*total if enumeration else 0))
    worker = fixed + per_column*b + output + 8 * (
        4*n*b + 3*g*b + 2*k*b + 3*n + g*q + 2*q*q + b
    )
    return int(setup), int(worker)


def _check_worker_memory(estimated_worker_bytes, *, setup_bytes=0):
    required = int(estimated_worker_bytes) + int(setup_bytes)
    available = current_allocation().remaining_memory_bytes()
    if required > available:
        raise SpecificationError(
            "One resampling batch does not fit the remaining memory budget.",
            code="resource.memory", stage="resampling",
            details={"estimated_worker_bytes": int(estimated_worker_bytes),
                     "setup_bytes": int(setup_bytes), "available_bytes": int(available)},
            suggestion="Reduce batch_size or explicitly increase the memory budget; batch size is not changed automatically.",
        )


def _failure(exc: BaseException) -> ReplicateFailure:
    if isinstance(exc, EconHDFEError):
        code = exc.code
    elif isinstance(exc, np.linalg.LinAlgError):
        code = "numerical.linalg"
    elif isinstance(exc, FloatingPointError):
        code = "numerical.floating_point"
    elif isinstance(exc, OverflowError):
        code = "numerical.overflow"
    else:
        code = f"resampling.{type(exc).__name__.lower()}"
    return ReplicateFailure(code=code, type=type(exc).__name__, message=str(exc))


@managed_execution
def run_replicates(
    worker: Callable[[np.random.SeedSequence], object],
    seeds: Sequence[np.random.SeedSequence], *, n_jobs: int = 1,
    skip_exceptions=(EconHDFEError, np.linalg.LinAlgError, FloatingPointError, OverflowError),
    execution_config=None,
    estimated_worker_bytes: int | None = None,
) -> ReplicateBatch:
    """Run independent replicates without hiding programming errors.

    Only explicitly declared statistical/numerical failures are skippable.
    AttributeError, NameError, AssertionError, etc. propagate immediately.
    An optional worker estimate bounds concurrency and reserves that workspace.
    Without it, arbitrary estimator memory use is unknown; the 32 MiB minimum
    is only an admission heuristic, not an estimate of the worker's peak.
    """
    allocation = current_allocation()
    if estimated_worker_bytes is not None:
        if not isinstance(estimated_worker_bytes, (int, np.integer)) or estimated_worker_bytes < 1:
            raise ValueError("estimated_worker_bytes must be a positive integer or None")
        _check_worker_memory(estimated_worker_bytes)
    worker_floor = max(32 * 1024**2, estimated_worker_bytes or 0)
    memory_workers = max(1, allocation.remaining_memory_bytes() // worker_floor)
    workers = max(1, min(effective_n_jobs(n_jobs), allocation.cpu_threads,
                         memory_workers, max(1, len(seeds))))
    worker_allocation = allocation.child(workers)

    def one(ss):
        # ContextVars do not automatically propagate into joblib threads.
        # Each replicate receives a share of the same parent envelope.
        with worker_scope(worker_allocation):
            token = _IN_REPLICATES.set(True)
            try:
                reservation = (nullcontext() if estimated_worker_bytes is None else
                               reserve_memory(estimated_worker_bytes, label="resampling worker"))
                with reservation:
                    return True, worker(ss)
            except skip_exceptions as exc:
                if isinstance(exc, EconHDFEError) and str(exc.code).startswith("resource."):
                    raise
                return False, _failure(exc)
            finally:
                _IN_REPLICATES.reset(token)

    # BLAS settings are process-wide. Only the outermost caller changes them;
    # nested resampling in workers must not race while restoring global state.
    limit = nullcontext() if _IN_REPLICATES.get() else threadpool_limits(limits=1)
    with limit:
        if workers == 1:
            raw = [one(s) for s in seeds]
        else:
            raw = Parallel(n_jobs=workers, prefer="threads", require="sharedmem")(
                delayed(one)(s) for s in seeds
            )
    batch = ReplicateBatch()
    for ok, value in raw:
        if ok:
            batch.values.append(value)
        else:
            batch.failures.append(value)
    return batch


@error_boundary("resampling")
@managed_execution
def parallel_pairs_bootstrap(estimator, nobs: int, *, reps=999, n_jobs=-1, seed=0,
                             execution_config=None):
    seeds = np.random.SeedSequence(seed).spawn(reps)
    def one(ss):
        rng = np.random.default_rng(ss)
        idx = rng.integers(0, nobs, size=nobs)
        return np.asarray(estimator(idx), dtype=np.float64)
    batch = run_replicates(one, seeds, n_jobs=n_jobs, skip_exceptions=())
    return np.vstack(batch.values)
