from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import RLock
import numpy as np
from ..planner import effective_cpu_count
from ..planner.calibration import auto_thread_count
from numba import config, get_num_threads, get_thread_id, njit, prange, set_num_threads, threading_layer


# workqueue aborts on concurrent entry, even with one masked Numba thread.
# All package parallel kernels use this module's reentrant execution guard.
_WORKQUEUE_LOCK = RLock()


@dataclass(slots=True)
class GroupIndex:
    """Compact CSR-like observation index for one dense FE code vector."""

    order: np.ndarray
    indptr: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.order.nbytes + self.indptr.nbytes)


@njit(cache=True, nogil=True)
def _build_group_index_int32(codes: np.ndarray, levels: int):
    """O(N + G) counting sort without an intermediate int64 argsort."""
    n = codes.size
    counts = np.zeros(levels, dtype=np.int64)
    for i in range(n):
        counts[codes[i]] += 1

    indptr = np.empty(levels + 1, dtype=np.int64)
    indptr[0] = 0
    for g in range(levels):
        indptr[g + 1] = indptr[g] + counts[g]

    cursor = indptr[:-1].copy()
    order = np.empty(n, dtype=np.int32)
    for i in range(n):
        g = codes[i]
        p = cursor[g]
        order[p] = i
        cursor[g] = p + 1
    return order, indptr


@njit(cache=True, nogil=True)
def _build_group_index_int64(codes: np.ndarray, levels: int):
    n = codes.size
    counts = np.zeros(levels, dtype=np.int64)
    for i in range(n):
        counts[codes[i]] += 1

    indptr = np.empty(levels + 1, dtype=np.int64)
    indptr[0] = 0
    for g in range(levels):
        indptr[g + 1] = indptr[g] + counts[g]

    cursor = indptr[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        g = codes[i]
        p = cursor[g]
        order[p] = i
        cursor[g] = p + 1
    return order, indptr


def build_group_index(codes: np.ndarray, levels: int) -> GroupIndex:
    """Build a compact observation index grouped by dense FE code."""
    codes = np.asarray(codes, dtype=np.int32)
    if codes.size <= np.iinfo(np.int32).max:
        order, indptr = _build_group_index_int32(codes, int(levels))
    else:
        order, indptr = _build_group_index_int64(codes, int(levels))
    return GroupIndex(order=order, indptr=indptr)


@njit(cache=True, nogil=True, parallel=True)
def _project_indexed_unweighted(a, order, indptr):
    """Project an N x K block on one intercept FE, parallel over groups."""
    levels = indptr.size - 1
    k = a.shape[1]
    for g in prange(levels):
        lo = indptr[g]
        hi = indptr[g + 1]
        if hi <= lo:
            continue
        inv_n = 1.0 / (hi - lo)
        for j in range(k):
            total = 0.0
            for p in range(lo, hi):
                total += a[order[p], j]
            mean = total * inv_n
            for p in range(lo, hi):
                a[order[p], j] -= mean


@njit(cache=True, nogil=True, parallel=True)
def _project_indexed_weighted(a, order, indptr, weights, denom):
    """Weighted intercept-FE projection, parallel over non-overlapping groups."""
    levels = indptr.size - 1
    k = a.shape[1]
    for g in prange(levels):
        lo = indptr[g]
        hi = indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv_d = 1.0 / d
        for j in range(k):
            total = 0.0
            for p in range(lo, hi):
                i = order[p]
                total += weights[i] * a[i, j]
            mean = total * inv_d
            for p in range(lo, hi):
                a[order[p], j] -= mean


@njit(cache=True, nogil=True, parallel=True)
def _group_sums_indexed_weighted_into(a, order, indptr, weights, out):
    """Weighted group sums for an N x K block, parallel over groups."""
    levels = indptr.size - 1
    k = a.shape[1]
    for g in prange(levels):
        lo = indptr[g]
        hi = indptr[g + 1]
        for j in range(k):
            total = 0.0
            for p in range(lo, hi):
                i = order[p]
                total += weights[i] * a[i, j]
            out[g, j] = total


def group_sums_indexed_weighted(
    a: np.ndarray,
    index: GroupIndex,
    weights: np.ndarray,
    *,
    threads: int | str | None = "auto",
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Return weighted group-by-column sums using the shared indexed topology.

    This is the reduction companion to ``project_indexed_inplace``.  Keeping
    it in the generic HDFE projection layer lets iterative two-way, multi-way,
    and future grouped operators share the same thread-control contract.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    w = np.asarray(weights, dtype=np.float64)
    if a.shape[0] != len(w):
        raise ValueError("weights must have one value per observation")
    shape = (index.indptr.size - 1, a.shape[1])
    if out is None:
        out = np.empty(shape, dtype=np.float64)
    else:
        out = np.asarray(out, dtype=np.float64)
        if out.shape != shape:
            raise ValueError("out has the wrong group-sum shape")
    nthreads = resolve_threads(threads, nobs=a.shape[0])
    with numba_thread_limit(nthreads):
        _group_sums_indexed_weighted_into(a, index.order, index.indptr, w, out)
    return out


def estimate_index_bytes(nobs: int, levels: list[int]) -> int:
    """Conservative resident bytes for cached group indexes."""
    order_bytes = 4 if nobs <= np.iinfo(np.int32).max else 8
    return int(sum(nobs * order_bytes + (int(g) + 1) * 8 for g in levels))


def resolve_threads(value, *, nobs: int | None = None, calibration_min_nobs: int = 100_000) -> int:
    """Resolve absorption threads; keep planner work out of hot kernels.

    ``auto`` is a planning decision and uses the shared planner once when an
    absorber/projector is compiled. Integer values are already-resolved policy
    in hot projection calls, so they take a constant-time cap-only path.
    """
    if value not in (None, "auto"):
        n = int(value)
        if n < 1:
            raise ValueError("absorb_threads must be >= 1 or 'auto'")
        return max(1, min(n, int(config.NUMBA_NUM_THREADS)))
    cap = min(int(get_num_threads()), int(effective_cpu_count()), int(config.NUMBA_NUM_THREADS))
    if nobs is not None and int(nobs) < max(1, int(calibration_min_nobs)):
        return 1
    selected, _calibration = auto_thread_count(max_threads=max(1, cap))
    return max(1, min(int(selected), int(config.NUMBA_NUM_THREADS)))


@contextmanager
def numba_thread_limit(nthreads: int):
    """Restore the caller's thread mask; serialize non-threadsafe workqueue.

    The lock covers package-managed parallel regions, not unrelated external
    Numba code. Reentrancy permits an absorber to call a guarded projection.
    OpenMP/TBB retain concurrent execution; no platform-specific defaults change.
    """
    old = int(get_num_threads())  # Also initializes the selected threading layer.
    n = max(1, min(int(nthreads), int(config.NUMBA_NUM_THREADS)))
    guard = _WORKQUEUE_LOCK if threading_layer() == "workqueue" else nullcontext()
    with guard:
        if n != old:
            set_num_threads(n)
        try:
            yield
        finally:
            if n != old:
                set_num_threads(old)


def project_indexed_inplace(
    a: np.ndarray,
    index: GroupIndex,
    *,
    weights: np.ndarray | None = None,
    denom: np.ndarray | None = None,
    threads: int | str | None = "auto",
) -> None:
    """Run the indexed CPU projection with bounded, temporary thread control."""
    nthreads = resolve_threads(threads, nobs=a.shape[0])
    with numba_thread_limit(nthreads):
        if weights is None:
            _project_indexed_unweighted(a, index.order, index.indptr)
        else:
            if denom is None:
                raise ValueError("weighted indexed projection requires denom")
            _project_indexed_weighted(a, index.order, index.indptr, weights, denom)

@njit(cache=False, nogil=True, parallel=True)
def _max_relative_update_parallel(current, previous, eps):
    """Maximum elementwise relative update without N x K temporaries."""
    n, k = current.shape
    thread_max = np.zeros(get_num_threads(), dtype=np.float64)
    for i in prange(n):
        tid = get_thread_id()
        local = thread_max[tid]
        for j in range(k):
            prev = previous[i, j]
            scale = abs(prev)
            if scale < 1.0:
                scale = 1.0
            value = abs(current[i, j] - prev) / (scale + eps)
            if value > local:
                local = value
        thread_max[tid] = local
    out = 0.0
    for t in range(thread_max.size):
        if thread_max[t] > out:
            out = thread_max[t]
    return out


def max_relative_update(
    current: np.ndarray,
    previous: np.ndarray,
    eps: float,
    *,
    threads: int | str | None = "auto",
) -> float:
    """Low-memory parallel convergence metric for NumPy MAP absorption."""
    nthreads = resolve_threads(threads, nobs=current.shape[0])
    with numba_thread_limit(nthreads):
        return float(_max_relative_update_parallel(current, previous, float(eps)))
