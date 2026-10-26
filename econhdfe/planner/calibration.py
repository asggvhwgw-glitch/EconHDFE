from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from threading import RLock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import config as numba_config
from numba import get_num_threads, njit, prange, set_num_threads

from .resources import effective_cpu_count, runtime_resources

_SCHEMA_VERSION = 2
_DEFAULT_TARGET_BYTES = 96 * 1024 * 1024
_DEFAULT_PASSES = 4
_DEFAULT_REPEATS = 3
_DEFAULT_NEAR_BEST = 1.05


@njit(cache=True, nogil=True, parallel=True)
def _memory_bandwidth_probe(x: np.ndarray, out: np.ndarray, passes: int) -> None:
    """Synthetic memory-bound kernel used only for runtime thread calibration.

    The probe intentionally contains no user data and no econometric logic.  It
    approximates the read/reduction/write traffic pattern that dominates large
    HDFE projection kernels, so the planner can identify the local saturation
    region without importing estimator or HDFE modules.
    """
    n, k = x.shape
    for p in range(passes):
        bias = p * 1e-12
        for i in prange(n):
            acc = 0.0
            for j in range(k):
                acc += x[i, j] * (j + 1.0)
            out[i] = acc + bias


@dataclass(frozen=True, slots=True)
class ThreadCalibration:
    schema_version: int
    calibration_id: str
    runtime_fingerprint: str
    max_threads: int
    candidates: tuple[int, ...]
    median_seconds: tuple[float, ...]
    selected_threads: int
    near_best_fraction: float
    target_bytes: int
    passes: int
    repeats: int
    source: str = "measured"  # measured | cache | fallback | single_thread

    def as_dict(self) -> dict:
        return {
            "schema_version": int(self.schema_version),
            "calibration_id": self.calibration_id,
            "runtime_fingerprint": self.runtime_fingerprint,
            "max_threads": int(self.max_threads),
            "candidates": [int(x) for x in self.candidates],
            "median_seconds": [float(x) for x in self.median_seconds],
            "selected_threads": int(self.selected_threads),
            "near_best_fraction": float(self.near_best_fraction),
            "target_bytes": int(self.target_bytes),
            "passes": int(self.passes),
            "repeats": int(self.repeats),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict, *, source: str = "cache") -> "ThreadCalibration":
        return cls(
            schema_version=int(payload["schema_version"]),
            calibration_id=str(payload["calibration_id"]),
            runtime_fingerprint=str(payload["runtime_fingerprint"]),
            max_threads=int(payload["max_threads"]),
            candidates=tuple(int(x) for x in payload["candidates"]),
            median_seconds=tuple(float(x) for x in payload["median_seconds"]),
            selected_threads=int(payload["selected_threads"]),
            near_best_fraction=float(payload["near_best_fraction"]),
            target_bytes=int(payload["target_bytes"]),
            passes=int(payload["passes"]),
            repeats=int(payload["repeats"]),
            source=source,
        )


def _cpu_model_digest() -> str | None:
    """Return only a non-reversible digest of the local CPU model string."""
    model = None
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if not model:
        model = platform.processor().strip() or None
    if not model:
        return None
    return hashlib.sha256(model.encode("utf-8", errors="ignore")).hexdigest()[:16]


def runtime_thread_fingerprint(*, max_threads: int | None = None) -> str:
    """Fingerprint only execution-relevant runtime properties, never host IDs."""
    try:
        import numba
        numba_version = numba.__version__
    except Exception:
        numba_version = "unknown"
    cap = _effective_numba_cap(max_threads)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    payload = {
        "schema": _SCHEMA_VERSION,
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "numba": numba_version,
        "cpu_model_digest": _cpu_model_digest(),
        "effective_cpu_count": int(effective_cpu_count()),
        "numba_cap": int(numba_config.NUMBA_NUM_THREADS),
        "calibration_cap": int(cap),
        "affinity": affinity,
        "threading_layer": str(numba_config.THREADING_LAYER),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _effective_numba_cap(max_threads: int | None = None) -> int:
    from .scheduler import current_allocation
    cap = min(int(effective_cpu_count()), int(numba_config.NUMBA_NUM_THREADS))
    active = current_allocation()
    if active is not None:
        cap = min(cap, active.cpu_threads)
    if max_threads is not None:
        cap = min(cap, max(1, int(max_threads)))
    return max(1, cap)


def calibration_candidates(max_threads: int) -> tuple[int, ...]:
    cap = max(1, int(max_threads))
    if cap <= 8:
        # Small machines are cheap enough to measure exactly; this also catches
        # 3/6-core quota configurations that powers-of-two-only probes miss.
        return tuple(range(1, cap + 1))
    vals = {1, 2, 4, 8, cap}
    x = 16
    while x < cap:
        vals.add(x)
        x *= 2
    return tuple(sorted(v for v in vals if v <= cap))


def _default_cache_dir() -> Path:
    explicit = os.environ.get("ECONHDFE_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser() / "planner"
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "econhdfe" / "planner"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "econhdfe" / "planner"
    return Path.home() / ".cache" / "econhdfe" / "planner"


def default_calibration_cache_path() -> Path:
    return _default_cache_dir() / f"thread-calibration-v{_SCHEMA_VERSION}.json"


def _load_cache(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return {"schema_version": _SCHEMA_VERSION, "entries": {}}
    if data.get("schema_version") != _SCHEMA_VERSION or not isinstance(data.get("entries"), dict):
        return {"schema_version": _SCHEMA_VERSION, "entries": {}}
    return data


def _write_cache(path: Path, data: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_raw = tempfile.mkstemp(prefix=".thread-calibration-", suffix=".json", dir=path.parent)
        tmp = Path(tmp_raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, sort_keys=True, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return True
    except OSError:
        return False


def _select_saturation_thread(candidates: tuple[int, ...], medians: tuple[float, ...], near_best: float) -> int:
    if not candidates:
        return 1
    best = min(medians)
    threshold = best * max(1.0, float(near_best))
    for threads, seconds in zip(candidates, medians):
        if seconds <= threshold:
            return int(threads)
    return int(candidates[int(np.argmin(np.asarray(medians)))])


def _allowed_probe_target(target_bytes=_DEFAULT_TARGET_BYTES):
    """Leave temporary/JIT headroom while choosing the probe's main array."""
    from .scheduler import current_allocation
    target = max(8 * 1024 * 1024, int(target_bytes))
    active = current_allocation()
    if active is not None:
        target = min(target, active.remaining_memory_bytes() // 2)
    resources = runtime_resources()
    if resources.memory_limit_bytes is not None and resources.memory_current_bytes is not None:
        headroom = max(0, resources.memory_limit_bytes - resources.memory_current_bytes
                       - int(0.12 * resources.memory_limit_bytes))
        target = min(target, headroom // 2)
    return target


def calibrate_threads(
    *,
    max_threads: int | None = None,
    target_bytes: int = _DEFAULT_TARGET_BYTES,
    passes: int = _DEFAULT_PASSES,
    repeats: int = _DEFAULT_REPEATS,
    near_best_fraction: float = _DEFAULT_NEAR_BEST,
) -> ThreadCalibration:
    """Measure the local memory-bound thread saturation point using synthetic data."""
    cap = _effective_numba_cap(max_threads)
    fingerprint = runtime_thread_fingerprint(max_threads=cap)
    if cap <= 1:
        cid = hashlib.sha256(f"{fingerprint}:1".encode()).hexdigest()[:16]
        return ThreadCalibration(
            _SCHEMA_VERSION, cid, fingerprint, 1, (1,), (0.0,), 1,
            float(near_best_fraction), 0, 0, 0, "single_thread",
        )

    from .scheduler import current_allocation, has_memory_reservations, reserve_memory
    active = current_allocation()
    if has_memory_reservations() or _parallel_worker(active):
        raise MemoryError("thread calibration is deferred during reserved or parallel work")
    target = _allowed_probe_target(target_bytes)
    if target < 8 * 1024 * 1024:
        raise MemoryError("insufficient budget/headroom for thread calibration")
    cols = 8
    rows = max(16_384, target // (cols * 8))
    # Deterministic, non-random, non-user data.  The small variation prevents
    # the compiler from reducing the workload to a trivial constant expression.
    # Reserve all known arrays atomically; the ledger lock is released before
    # initialization/JIT/timing. Compiler and runtime overhead is not modeled.
    with reserve_memory(rows * (cols + 3) * 8 + cols * 8, label="thread calibration probe"):
        base = np.arange(cols, dtype=np.float64)[None, :] + 1.0
        x = np.empty((rows, cols), dtype=np.float64)
        x[:] = base
        x[:, 0] += (np.arange(rows, dtype=np.float64) % 17.0) * 1e-9
        out = np.empty(rows, dtype=np.float64)

        old = int(get_num_threads())
        try:
            set_num_threads(1)
            _memory_bandwidth_probe(x[: min(rows, 32_768)], out[: min(rows, 32_768)], 1)
            candidates = calibration_candidates(cap)
            medians: list[float] = []
            reps = max(1, int(repeats))
            p = max(1, int(passes))
            for nthreads in candidates:
                set_num_threads(int(nthreads))
                timings = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    _memory_bandwidth_probe(x, out, p)
                    timings.append(time.perf_counter() - t0)
                medians.append(float(np.median(np.asarray(timings, dtype=np.float64))))
        finally:
            set_num_threads(old)
        measured_bytes = int(x.nbytes)
        del x, out, base

    medians_t = tuple(medians)
    selected = _select_saturation_thread(candidates, medians_t, near_best_fraction)
    identity_payload = json.dumps(
        {"fp": fingerprint, "candidates": candidates, "medians": [round(x, 9) for x in medians_t],
         "selected": selected, "near_best": near_best_fraction, "target": measured_bytes, "passes": p},
        sort_keys=True, separators=(",", ":"),
    )
    cid = hashlib.sha256(identity_payload.encode()).hexdigest()[:16]
    return ThreadCalibration(
        _SCHEMA_VERSION, cid, fingerprint, cap, candidates, medians_t, selected,
        float(near_best_fraction), measured_bytes, p, reps, "measured",
    )


_MEMORY_CACHE: dict[tuple[str, int], ThreadCalibration] = {}
_CALIBRATION_LOCK = RLock()


def _parallel_worker(allocation):
    while allocation is not None:
        if allocation.worker_divisor > 1:
            return True
        allocation = allocation.parent
    return False


def _valid_calibration(value, *, fingerprint, cap):
    from .scheduler import current_allocation, has_memory_reservations
    # A small probe can live in CPU cache. Reuse it under the same constrained
    # budget, but revisit it when at least twice as much probe space is safe.
    sufficient_probe = value.max_threads == 1 or value.target_bytes >= _DEFAULT_TARGET_BYTES
    if not sufficient_probe:
        deferred = has_memory_reservations() or _parallel_worker(current_allocation())
        allowed = 0 if deferred else _allowed_probe_target()
        sufficient_probe = deferred or allowed < min(_DEFAULT_TARGET_BYTES, 2 * value.target_bytes)
    return (
        value.schema_version == _SCHEMA_VERSION
        and value.runtime_fingerprint == fingerprint and value.max_threads == cap
        and value.source in {"measured", "cache", "single_thread"}
        and value.selected_threads in value.candidates
        and all(1 <= t <= cap for t in value.candidates)
        and len(value.candidates) == len(value.median_seconds)
        and all(np.isfinite(t) and t >= 0 for t in value.median_seconds)
        and (cap == 1 or value.target_bytes >= 8 * 1024 * 1024)
        and sufficient_probe
    )


def get_thread_calibration(
    *,
    max_threads: int | None = None,
    force: bool = False,
    cache_path: str | os.PathLike[str] | None = None,
) -> ThreadCalibration:
    """Load or measure the real-runtime thread calibration.

    Cache corruption or unwritable user cache directories are non-fatal: the
    planner falls back to an in-memory measurement and never changes estimator
    semantics because calibration controls only the thread count.
    """
    with _CALIBRATION_LOCK:
        return _get_thread_calibration(max_threads=max_threads, force=force, cache_path=cache_path)


def _get_thread_calibration(*, max_threads=None, force=False, cache_path=None):
    cap = _effective_numba_cap(max_threads)
    fp = runtime_thread_fingerprint(max_threads=cap)
    key = (fp, cap)
    if not force and key in _MEMORY_CACHE:
        cached = _MEMORY_CACHE[key]
        if _valid_calibration(cached, fingerprint=fp, cap=cap):
            return cached

    path = default_calibration_cache_path() if cache_path is None else Path(cache_path)
    data = _load_cache(path)
    if not force:
        raw = data.get("entries", {}).get(fp)
        if isinstance(raw, dict):
            try:
                cached = ThreadCalibration.from_dict(raw, source="cache")
                if raw.get("source") != "fallback" and _valid_calibration(cached, fingerprint=fp, cap=cap):
                    _MEMORY_CACHE[key] = cached
                    return cached
            except (KeyError, TypeError, ValueError):
                pass

    try:
        from .scheduler import current_allocation, has_memory_reservations
        if cap > 1 and (has_memory_reservations() or _parallel_worker(current_allocation())):
            raise MemoryError("thread calibration is deferred during reserved or parallel work")
        measured = calibrate_threads(max_threads=cap)
    except Exception:
        # Temporary pressure/deferred probes are not saturation measurements.
        # Stay serial now and retry at a later unreserved planning boundary.
        cid = hashlib.sha256(f"{fp}:fallback:{cap}".encode()).hexdigest()[:16]
        measured = ThreadCalibration(
            _SCHEMA_VERSION, cid, fp, cap, (1,), (0.0,), 1,
            _DEFAULT_NEAR_BEST, 0, 0, 0, "fallback",
        )
        return measured

    if measured.max_threads == 1 or measured.target_bytes >= _DEFAULT_TARGET_BYTES:
        data.setdefault("entries", {})[fp] = measured.as_dict()
        _write_cache(path, data)
    _MEMORY_CACHE[key] = measured
    return measured


def auto_thread_count(*, max_threads: int | None = None) -> tuple[int, ThreadCalibration]:
    calibration = get_thread_calibration(max_threads=max_threads)
    return int(calibration.selected_threads), calibration


def clear_in_memory_calibration_cache() -> None:
    _MEMORY_CACHE.clear()
