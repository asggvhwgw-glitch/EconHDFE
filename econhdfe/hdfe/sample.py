"""Exact categorical sample peeling, independent of estimators and IRLS weights.

Frequency-singleton tests use physical degree one AND unit multiplicity.
For positive integer frequencies this is exactly sum(f_i)==1, without summing
large floats or expanding records. A queue visits each removed incidence once.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(slots=True)
class FESampleSelection:
    keep: np.ndarray
    singletons: np.ndarray
    separated: np.ndarray


@njit(cache=True, nogil=True)
def _peel(codes, offsets, unit, positive, check_zero, eligible, drop_singletons):
    q, n = codes.shape
    nodes = offsets[-1]
    degree = np.zeros(nodes, np.int64)
    npositive = np.zeros(nodes, np.int64)
    for j in range(q):
        for i in range(n):
            v = offsets[j] + codes[j, i]
            degree[v] += 1
            npositive[v] += positive[i]
    alive = np.ones(n, np.bool_)
    queued = np.zeros(n, np.bool_)
    reason = np.zeros(n, np.uint8)
    queue = np.empty(n, np.int64)
    tail = 0
    zero_queued = npositive == 0
    for i in range(n):
        take = False
        for j in range(q):
            v = offsets[j] + codes[j, i]
            if check_zero and npositive[v] == 0:
                take = True
            if drop_singletons and unit[i] and degree[v] == 1 and eligible[j, i]:
                take = True
        if take:
            queue[tail] = i
            tail += 1
            queued[i] = True
    if tail == 0:
        return alive, reason
    ptr = np.empty(nodes + 1, np.int64)
    ptr[0] = 0
    for v in range(nodes):
        ptr[v + 1] = ptr[v] + degree[v]
    cursor = ptr[:-1].copy()
    incident = np.empty(n * q, np.int64)
    for j in range(q):
        for i in range(n):
            v = offsets[j] + codes[j, i]
            incident[cursor[v]] = i
            cursor[v] += 1
    head = 0
    while head < tail:
        i = queue[head]
        head += 1
        # Both criteria are monotone. Attribute a simultaneous trigger to
        # the all-zero level; this never labels a positive outcome separated.
        why = 1
        if check_zero:
            for j in range(q):
                if npositive[offsets[j] + codes[j, i]] == 0:
                    why = 2
                    break
        reason[i] = why
        alive[i] = False
        for j in range(q):
            v = offsets[j] + codes[j, i]
            degree[v] -= 1
            npositive[v] -= positive[i]
            all_zero = check_zero and npositive[v] == 0 and not zero_queued[v]
            one = drop_singletons and degree[v] == 1
            if all_zero:
                zero_queued[v] = True
            if all_zero or one:
                for pos in range(ptr[v], ptr[v + 1]):
                    other = incident[pos]
                    if alive[other] and not queued[other]:
                        if all_zero or (unit[other] and eligible[j, other]):
                            queue[tail] = other
                            tail += 1
                            queued[other] = True
    return alive, reason


def peel_sample(groups, *, nobs=None, frequency_weights=None, positive=None,
                singleton_eligible=None, drop_singletons=True):
    """Return disjoint removal reasons on the supplied positive-weight support.

    ``groups`` are dense nonnegative integer category codes. ``positive`` optionally
    enables zero-level peeling, while ``singleton_eligible`` can exclude a
    zero-valued slope-only row from the singleton criterion. Reasons follow
    deterministic first-trigger queue order; the final keep mask is the unique
    monotone fixed point and is independent of row/FE order. No leverage or
    numerical-weight threshold is used. Zero user weights must be handled by
    the caller's sample-validation policy, not passed as multiplicities.
    """
    arrays = tuple(np.asarray(g) for g in groups)
    n = (len(arrays[0]) if arrays else 0) if nobs is None else int(nobs)
    if n < 0:
        raise ValueError("nobs must be nonnegative")
    for g in arrays:
        if g.ndim != 1 or len(g) != n or not np.issubdtype(g.dtype, np.integer) or np.any(g < 0):
            raise ValueError("FE codes must be aligned one-dimensional nonnegative integers")
        if n and int(g.max()) >= n:
            raise ValueError("FE codes must be dense observed category codes below nobs")
    unit = np.ones(n, dtype=bool)
    if frequency_weights is not None:
        fw = np.asarray(frequency_weights, dtype=np.float64)
        if (fw.shape != (n,) or np.any(~np.isfinite(fw)) or np.any(fw <= 0)
                or not np.allclose(fw, np.rint(fw), rtol=0, atol=1e-12)):
            raise ValueError("frequency weights must be aligned positive integers")
        unit = np.rint(fw) == 1
    pos = np.zeros(n, dtype=bool) if positive is None else np.asarray(positive, dtype=bool)
    if pos.shape != (n,):
        raise ValueError("positive outcome mask length mismatch")
    q = len(arrays)
    eligible = np.ones((q, n), dtype=bool) if singleton_eligible is None else np.asarray(singleton_eligible, dtype=bool)
    if q and eligible.shape != (q, n):
        raise ValueError("singleton eligibility must have shape (number of FE terms, nobs)")
    if not arrays or n == 0 or (not drop_singletons and positive is None):
        return FESampleSelection(np.ones(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
    codes = np.ascontiguousarray(np.stack(arrays), dtype=np.int64)
    levels = np.asarray([int(g.max()) + 1 for g in arrays], dtype=np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(levels)]
    keep, reason = _peel(codes, offsets, unit, pos, positive is not None, eligible, bool(drop_singletons))
    return FESampleSelection(keep, reason == 1, reason == 2)
