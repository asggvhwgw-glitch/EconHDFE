from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
import psutil


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_quota() -> int | None:
    raw = _read_text('/sys/fs/cgroup/cpu.max')
    if raw:
        parts = raw.split()
        if len(parts) >= 2 and parts[0] != 'max':
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, quota // period)
            except ValueError:
                pass
    return None


def affinity_cpus() -> int | None:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return None


def effective_cpu_count() -> int:
    vals = [v for v in (cpu_quota(), affinity_cpus(), os.cpu_count()) if v]
    return max(1, min(vals)) if vals else 1


def configure_numba_runtime() -> int:
    if 'numba' not in sys.modules and 'NUMBA_NUM_THREADS' not in os.environ:
        os.environ['NUMBA_NUM_THREADS'] = str(effective_cpu_count())
    raw = os.environ.get('NUMBA_NUM_THREADS')
    if raw is None:
        return effective_cpu_count()
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return effective_cpu_count()


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    cpu_threads: int
    memory_limit_bytes: int | None
    memory_current_bytes: int | None


def _memory_value(path: str) -> int | None:
    raw = _read_text(path)
    if not raw or raw == 'max':
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def runtime_resources() -> RuntimeResources:
    limit = _memory_value('/sys/fs/cgroup/memory.max')
    current = _memory_value('/sys/fs/cgroup/memory.current')
    try:
        host = psutil.virtual_memory()
    except (OSError, NotImplementedError):
        # Some sandboxes deny host counters; retain cgroup data when present.
        host = None
    if host is not None:
        host_limit = int(host.total)
        host_current = max(0, host_limit - int(host.available))
        if limit is None or current is None:
            limit, current = host_limit, host_current
        else:
            # Keep the tighter available headroom without weakening a
            # container limit when the host is also under memory pressure.
            headroom = min(max(0, limit - current), int(host.available))
            limit = min(limit, host_limit)
            current = max(0, limit - headroom)
    return RuntimeResources(
        cpu_threads=effective_cpu_count(),
        memory_limit_bytes=limit,
        memory_current_bytes=current,
    )
