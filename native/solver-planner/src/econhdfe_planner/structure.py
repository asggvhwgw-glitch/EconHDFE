"""Full-stream tuple sketch: ordering invariant; never used for correctness.

Unlike a small row sample, every row contributes. Hash collisions can only
change the performance admission policy, never the FE projection. The error
band is an approximate HLL sampling band, not a deterministic guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


def compact_codes(raw):
    a = np.asarray(raw)
    if a.ndim != 2 or a.dtype.kind not in "iu" or not 1 <= a.shape[1] <= 32:
        raise ValueError("codes must be a nonempty 2D integer array with 1..32 factors")
    n = len(a)
    if n == 0 or n > np.iinfo(np.uint32).max:
        raise ValueError("require 1..uint32_max rows")
    if a.min() < 0 or a.max() > np.iinfo(np.uint32).max:
        raise ValueError("encode missing/negative/oversized FE labels first")
    out = np.empty(a.shape, dtype=np.uint32, order="F")
    levels = []
    for j in range(a.shape[1]):
        c = a[:, j].astype(np.uint32, copy=False)
        if int(c.max()) < 2 * n:
            present = np.bincount(c) > 0
            mapping = np.cumsum(present, dtype=np.uint32) - np.uint32(1)
            out[:, j] = mapping[c]
            levels.append(int(present.sum()))
        else:
            labels, out[:, j] = np.unique(c, return_inverse=True)
            levels.append(len(labels))
    out.flags.writeable = False
    return out, tuple(levels)


def positive_weights(raw, n):
    w = np.ones(n) if raw is None else np.array(raw, dtype=np.float64, copy=True)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("weights must be finite, strictly positive and have shape (n,)")
    w /= w.max()
    if np.any(w == 0):
        raise ValueError("weight normalization underflows; sample is not modified")
    return w


def _mix(x):
    x = np.array(x, dtype=np.uint64, copy=True)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xbf58476d1ce4e5b9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return x


@dataclass(frozen=True)
class TupleSketch:
    rows: int
    distinct_estimate: float
    distinct_ratio: float
    ratio_upper: float
    relative_standard_error: float
    registers: int
    scanned_rows: int
    method: str = "full_stream_hll_64bit"

    def to_dict(self):
        return asdict(self)


def tuple_sketch(codes: np.ndarray, *, precision: int = 14,
                 chunk_rows: int = 131072) -> TupleSketch:
    if not 8 <= precision <= 18 or chunk_rows < 1:
        raise ValueError("invalid sketch precision/chunk_rows")
    n, k = codes.shape
    if n == 0 or k == 0:
        raise ValueError("empty geometry")
    m = 1 << precision
    registers = np.zeros(m, dtype=np.uint8)
    for start in range(0, n, chunk_rows):
        part = codes[start:start + chunk_rows]
        h = np.full(len(part), np.uint64(0x243f6a8885a308d3), dtype=np.uint64)
        for j in range(k):
            salt = np.uint64((0x9e3779b97f4a7c15 * (j + 1)) & ((1 << 64) - 1))
            h = _mix(h ^ _mix(part[:, j].astype(np.uint64) + salt))
        index = (h & np.uint64(m - 1)).astype(np.intp)
        tail = h >> np.uint64(precision)
        # frexp exponent is the integer bit length (up to insignificant hash
        # boundary rounding at powers of two); no log(0) or overflow.
        exponent = np.frexp(tail.astype(np.float64))[1]
        rank = (65 - precision - exponent).astype(np.uint8)
        np.maximum.at(registers, index, rank)
    alpha = 0.7213 / (1.0 + 1.079 / m)
    estimate = alpha * m * m / np.exp2(-registers.astype(np.float64)).sum()
    zeros = int((registers == 0).sum())
    if estimate <= 2.5 * m and zeros:
        estimate = -m * np.log(zeros / m)
    estimate = float(min(n, max(1, estimate)))
    rse = 1.04 / np.sqrt(m)
    return TupleSketch(n, estimate, estimate / n,
                       min(1.0, estimate / n * (1.0 + 3.0 * rse)),
                       float(rse), m, n)
