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
    contiguous: bool = False

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


@njit(cache=True, nogil=True)
def _groups_are_contiguous(order, indptr):
    # Counting sort preserves ascending row positions within each group.
    for g in range(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi > lo and order[hi - 1] - order[lo] != hi - lo - 1:
            return False
    return True


def build_group_index(codes: np.ndarray, levels: int) -> GroupIndex:
    """Build a compact observation index grouped by dense FE code."""
    codes = np.asarray(codes, dtype=np.int32)
    if codes.size <= np.iinfo(np.int32).max:
        order, indptr = _build_group_index_int32(codes, int(levels))
    else:
        order, indptr = _build_group_index_int64(codes, int(levels))
    return GroupIndex(order=order, indptr=indptr,
                      contiguous=bool(_groups_are_contiguous(order, indptr)))


@njit(cache=True, nogil=True, parallel=True)
def _project_indexed_unweighted(a, order, indptr):
    """Project columns with scalar accumulators, parallel over groups."""
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
    """Weighted scalar accumulation with the same within-group row order."""
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
def _project_contiguous_unweighted(a, order, indptr):
    """Project certified contiguous groups without per-row index reads."""
    levels = indptr.size - 1
    k = a.shape[1]
    for g in prange(levels):
        lo = indptr[g]
        hi = indptr[g + 1]
        if hi <= lo:
            continue
        start = order[lo]
        end = start + hi - lo
        inv_n = 1.0 / (hi - lo)
        for j in range(k):
            total = 0.0
            for p in range(start, end):
                total += a[p, j]
            mean = total * inv_n
            for p in range(start, end):
                a[p, j] -= mean


@njit(cache=True, nogil=True, parallel=True)
def _project_contiguous_weighted(a, order, indptr, weights, denom):
    """Weighted contiguous projection preserving within-group arithmetic."""
    levels = indptr.size - 1
    k = a.shape[1]
    for g in prange(levels):
        lo = indptr[g]
        hi = indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        start = order[lo]
        end = start + hi - lo
        inv_d = 1.0 / d
        for j in range(k):
            total = 0.0
            for p in range(start, end):
                i = p
                total += weights[i] * a[i, j]
            mean = total * inv_d
            for p in range(start, end):
                a[p, j] -= mean


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
        _project_indexed_inplace_current(a, index, weights, denom)


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar2_unweighted(a, order, indptr, contiguous):
    """Share row access across 2 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
        m0 = s0 * inv
        m1 = s1 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar2_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access across 2 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
        m0 = s0 * inv
        m1 = s1 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar3_unweighted(a, order, indptr, contiguous):
    """Share row access across 3 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar3_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access across 3 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar4_unweighted(a, order, indptr, contiguous):
    """Share row access across 4 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar4_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access across 4 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar5_unweighted(a, order, indptr, contiguous):
    """Share row access across 5 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
            s4 += a[i, 4]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar5_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access across 5 scalar accumulators; preserve column sums."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
            s4 += w * a[i, 4]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4

@njit(cache=True, nogil=True, parallel=True)
def _project_blocked5_unweighted(a, order, indptr, contiguous):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    k = a.shape[1]
    stop = k - k % 5
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        for offset in range(0, stop, 5):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                s0 += a[i, offset + 0]
                s1 += a[i, offset + 1]
                s2 += a[i, offset + 2]
                s3 += a[i, offset + 3]
                s4 += a[i, offset + 4]
            m0 = s0 * inv
            m1 = s1 * inv
            m2 = s2 * inv
            m3 = s3 * inv
            m4 = s4 * inv
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                a[i, offset + 0] -= m0
                a[i, offset + 1] -= m1
                a[i, offset + 2] -= m2
                a[i, offset + 3] -= m3
                a[i, offset + 4] -= m4
        # At most four trailing columns; keep the same per-column arithmetic.
        for j in range(stop, k):
            total = 0.0
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                total += a[i, j]
            mean = total * inv
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                a[i, j] -= mean

@njit(cache=True, nogil=True, parallel=True)
def _project_blocked5_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    k = a.shape[1]
    stop = k - k % 5
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        for offset in range(0, stop, 5):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                w = weights[i]
                s0 += w * a[i, offset + 0]
                s1 += w * a[i, offset + 1]
                s2 += w * a[i, offset + 2]
                s3 += w * a[i, offset + 3]
                s4 += w * a[i, offset + 4]
            m0 = s0 * inv
            m1 = s1 * inv
            m2 = s2 * inv
            m3 = s3 * inv
            m4 = s4 * inv
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                a[i, offset + 0] -= m0
                a[i, offset + 1] -= m1
                a[i, offset + 2] -= m2
                a[i, offset + 3] -= m3
                a[i, offset + 4] -= m4
        # At most four trailing columns; keep the same per-column arithmetic.
        for j in range(stop, k):
            total = 0.0
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                total += weights[i] * a[i, j]
            mean = total * inv
            for p in range(lo, hi):
                i = start + p - lo if contiguous else order[p]
                a[i, j] -= mean

@njit(cache=True, nogil=True, parallel=True)
def _project_scalar6_unweighted(a, order, indptr, contiguous):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
            s4 += a[i, 4]
            s5 += a[i, 5]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar6_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
            s4 += w * a[i, 4]
            s5 += w * a[i, 5]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar7_unweighted(a, order, indptr, contiguous):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
            s4 += a[i, 4]
            s5 += a[i, 5]
            s6 += a[i, 6]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar7_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
            s4 += w * a[i, 4]
            s5 += w * a[i, 5]
            s6 += w * a[i, 6]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar8_unweighted(a, order, indptr, contiguous):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
            s4 += a[i, 4]
            s5 += a[i, 5]
            s6 += a[i, 6]
            s7 += a[i, 7]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        m7 = s7 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6
            a[i, 7] -= m7


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar8_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
            s4 += w * a[i, 4]
            s5 += w * a[i, 5]
            s6 += w * a[i, 6]
            s7 += w * a[i, 7]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        m7 = s7 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6
            a[i, 7] -= m7


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar9_unweighted(a, order, indptr, contiguous):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        s8 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
            s4 += a[i, 4]
            s5 += a[i, 5]
            s6 += a[i, 6]
            s7 += a[i, 7]
            s8 += a[i, 8]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        m7 = s7 * inv
        m8 = s8 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6
            a[i, 7] -= m7
            a[i, 8] -= m8


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar9_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        s8 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
            s4 += w * a[i, 4]
            s5 += w * a[i, 5]
            s6 += w * a[i, 6]
            s7 += w * a[i, 7]
            s8 += w * a[i, 8]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        m7 = s7 * inv
        m8 = s8 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6
            a[i, 7] -= m7
            a[i, 8] -= m8


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar10_unweighted(a, order, indptr, contiguous):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        if hi <= lo:
            continue
        inv = 1.0 / (hi - lo)
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        s8 = 0.0
        s9 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            s0 += a[i, 0]
            s1 += a[i, 1]
            s2 += a[i, 2]
            s3 += a[i, 3]
            s4 += a[i, 4]
            s5 += a[i, 5]
            s6 += a[i, 6]
            s7 += a[i, 7]
            s8 += a[i, 8]
            s9 += a[i, 9]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        m7 = s7 * inv
        m8 = s8 * inv
        m9 = s9 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6
            a[i, 7] -= m7
            a[i, 8] -= m8
            a[i, 9] -= m9


@njit(cache=True, nogil=True, parallel=True)
def _project_scalar10_weighted(a, order, indptr, contiguous, weights, denom):
    """Share row access using bounded scalar accumulators, without scratch arrays."""
    for g in prange(indptr.size - 1):
        lo, hi = indptr[g], indptr[g + 1]
        d = denom[g]
        if hi <= lo or d == 0.0:
            continue
        inv = 1.0 / d
        start = order[lo] if contiguous else 0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        s6 = 0.0
        s7 = 0.0
        s8 = 0.0
        s9 = 0.0
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            w = weights[i]
            s0 += w * a[i, 0]
            s1 += w * a[i, 1]
            s2 += w * a[i, 2]
            s3 += w * a[i, 3]
            s4 += w * a[i, 4]
            s5 += w * a[i, 5]
            s6 += w * a[i, 6]
            s7 += w * a[i, 7]
            s8 += w * a[i, 8]
            s9 += w * a[i, 9]
        m0 = s0 * inv
        m1 = s1 * inv
        m2 = s2 * inv
        m3 = s3 * inv
        m4 = s4 * inv
        m5 = s5 * inv
        m6 = s6 * inv
        m7 = s7 * inv
        m8 = s8 * inv
        m9 = s9 * inv
        for p in range(lo, hi):
            i = start + p - lo if contiguous else order[p]
            a[i, 0] -= m0
            a[i, 1] -= m1
            a[i, 2] -= m2
            a[i, 3] -= m3
            a[i, 4] -= m4
            a[i, 5] -= m5
            a[i, 6] -= m6
            a[i, 7] -= m7
            a[i, 8] -= m8
            a[i, 9] -= m9


def _project_indexed_inplace_current(a, index, weights=None, denom=None):
    """Dispatch scalar and bounded-width kernels without changing thread policy."""
    # Float64 uses scalar kernels without an upper RHS-width cutoff.
    width = a.shape[1] if a.dtype == np.float64 else 0
    if weights is not None and denom is None:
        raise ValueError("weighted indexed projection requires denom")
    if width == 2:
        if weights is None:
            _project_scalar2_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar2_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 3:
        if weights is None:
            _project_scalar3_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar3_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 4:
        if weights is None:
            _project_scalar4_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar4_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 5:
        if weights is None:
            _project_scalar5_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar5_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 6:
        if weights is None:
            _project_scalar6_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar6_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 7:
        if weights is None:
            _project_scalar7_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar7_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 8:
        if weights is None:
            _project_scalar8_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar8_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 9:
        if weights is None:
            _project_scalar9_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar9_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width == 10:
        if weights is None:
            _project_scalar10_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_scalar10_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if width > 10:
        if weights is None:
            _project_blocked5_unweighted(a, index.order, index.indptr, index.contiguous)
        else:
            _project_blocked5_weighted(a, index.order, index.indptr, index.contiguous, weights, denom)
        return
    if weights is None:
        kernel = _project_contiguous_unweighted if index.contiguous else _project_indexed_unweighted
        kernel(a, index.order, index.indptr)
    else:
        kernel = _project_contiguous_weighted if index.contiguous else _project_indexed_weighted
        kernel(a, index.order, index.indptr, weights, denom)


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
