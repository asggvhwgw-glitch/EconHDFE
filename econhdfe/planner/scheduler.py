"""One process-local resource envelope for an entire estimation call.

This is cooperative admission/planning, not an OS memory limiter. Independent
root calls are serialized because BLAS limits are process-global. Nested work
inherits the envelope; parallel workers receive explicit slices of it.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import wraps
from inspect import signature
from threading import RLock
from operator import index

import psutil
import sys
from threadpoolctl import ThreadpoolController, threadpool_limits

from ..config import ExecutionConfig
from ..errors import SpecificationError
from .resources import runtime_resources

_MIB = 1024**2
_RESERVE_FRACTION = 0.12
_ROOT_LOCK = RLock()
_CURRENT = ContextVar("econhdfe_resource_allocation", default=None)
_RESERVATIONS = ContextVar("econhdfe_memory_reservations", default=())


class _MemoryLedger:
    def __init__(self):
        self.lock = RLock()
        self.claims = {}



def _rss():
    return int(psutil.Process().memory_info().rss)


def _headroom(resources):
    if resources.memory_limit_bytes is None or resources.memory_current_bytes is None:
        return None
    reserve = int(resources.memory_limit_bytes * _RESERVE_FRACTION)
    return max(0, resources.memory_limit_bytes - resources.memory_current_bytes - reserve)


@dataclass(frozen=True)
class ResourceAllocation:
    config: ExecutionConfig
    cpu_threads: int
    memory_budget_bytes: int
    initial_rss_bytes: int
    reserve_bytes: int
    parent: ResourceAllocation | None = None
    worker_divisor: int = 1
    _ledger: _MemoryLedger = field(default_factory=_MemoryLedger, repr=False, compare=False)
    auto_threads: bool = False

    def __post_init__(self):
        if self.parent is not None:
            object.__setattr__(self, "_ledger", self.parent._ledger)

    def _root_and_cap(self):
        root, cap = self, self.memory_budget_bytes
        while root.parent is not None:
            root = root.parent
            cap = min(cap, root.memory_budget_bytes)
        return root, cap

    def _available_locked(self, *, credit_own):
        root, cap = self._root_and_cap()
        used = max(0, _rss() - root.initial_rss_bytes)
        total = sum(self._ledger.claims.values())
        own = sum(self._ledger.claims.get(key, 0) for key in _RESERVATIONS.get())
        charged = total - own if credit_own else total
        available = max(0, root.memory_budget_bytes - used - charged)
        live = _headroom(runtime_resources())
        if live is not None:
            available = min(available, max(0, live - charged))
        # Siblings share the root remainder, but retain their static slices.
        # Dividing the shared remainder again would charge siblings twice.
        local = cap if credit_own else max(0, cap - own)
        return min(local, available)

    def remaining_memory_bytes(self):
        with self._ledger.lock:
            return self._available_locked(credit_own=True)

    def unreserved_memory_bytes(self):
        """Capacity for new work, including this call's outstanding claims."""
        with self._ledger.lock:
            return self._available_locked(credit_own=False)

    def child(self, workers):
        workers = max(1, int(workers))
        threads = max(1, self.cpu_threads // workers)
        # Slice the admitted envelope, not its transient remainder. Runtime
        # consumption is still bounded dynamically by remaining_memory_bytes.
        budget = self.memory_budget_bytes // workers
        if budget < 32 * _MIB or self.remaining_memory_bytes() <= 0:
            raise SpecificationError("Not enough shared memory for the requested parallel workers.",
                                     code="resource.memory", stage="execution")
        return ResourceAllocation(
            replace(self.config, threads=threads, memory_budget_mb=budget / _MIB),
            threads, budget, self.initial_rss_bytes, self.reserve_bytes, self, workers,
            auto_threads=self.auto_threads,
        )

    def as_dict(self):
        return {"scope": "whole_call_incremental_workspace", "cpu_threads": self.cpu_threads,
                "thread_policy": "auto" if self.auto_threads else "explicit",
                "memory_budget_bytes": self.memory_budget_bytes, "reserve_bytes": self.reserve_bytes,
                "remaining_memory_bytes": self.remaining_memory_bytes(),
                "worker_divisor": self.worker_divisor,
                "memory_enforcement": "cooperative_planning_not_rss_limit"}


def current_allocation():
    return _CURRENT.get()


def inner_thread_request(requested):
    """Recover auto policy after public entry resolves the overall CPU cap."""
    active = current_allocation()
    if active is not None and active.auto_threads and requested == active.cpu_threads:
        return "auto"
    return requested


def has_memory_reservations():
    """Whether a live workspace makes a cold calibration probe unsuitable."""
    active = current_allocation()
    if active is None:
        return False
    with active._ledger.lock:
        return bool(active._ledger.claims)


def resolve_allocation(config=None):
    config = ExecutionConfig() if config is None else config
    config.validate()
    resources = runtime_resources()
    parent = current_allocation()
    auto_threads = config.threads == "auto" or (
        parent is not None and config is parent.config and parent.auto_threads
    )
    threads = resources.cpu_threads if config.threads == "auto" else min(config.threads, resources.cpu_threads)
    if parent is not None:
        threads = min(threads, parent.cpu_threads)
    available = _headroom(resources)
    if parent is not None:
        # Nested calls share an admitted envelope; do not reapply the public
        # 32 MiB admission floor to the already partly consumed remainder.
        available = parent.memory_budget_bytes
    if config.memory_budget_mb == "auto":
        if available is None:
            raise SpecificationError("Automatic memory scheduling needs available-memory counters; supply an explicit budget.",
                                     code="resource.unavailable", stage="execution")
        budget = available
    else:
        budget = int(config.memory_budget_mb * _MIB)
        if available is not None:
            budget = min(budget, available)
    if budget < 32 * _MIB:
        raise SpecificationError("Insufficient safe memory headroom for estimation.",
                                 code="resource.memory", stage="execution",
                                 details={"available_bytes": budget},
                                 suggestion="Free memory or reduce concurrent work before retrying.")
    resolved = replace(config, threads=max(1, threads), memory_budget_mb=budget / _MIB)
    return ResourceAllocation(resolved, max(1, threads), budget, _rss(),
                              int((resources.memory_limit_bytes or 0) * _RESERVE_FRACTION), parent,
                              auto_threads=auto_threads)


@contextmanager
def execution_scope(config=None):
    if current_allocation() is not None:
        allocation = resolve_allocation(config)
        token = _CURRENT.set(allocation)
        try:
            yield allocation
        finally:
            _CURRENT.reset(token)
    else:
        # Resolve after admission, so a queued call cannot use stale headroom.
        with _ROOT_LOCK:
            allocation = resolve_allocation(config)
            token = _CURRENT.set(allocation)
            try:
                with threadpool_limits(limits=allocation.cpu_threads):
                    # Linux OpenBLAS 0.3.28 can segfault in TRTRS and QR after
                    # raising its initial thread count. Restrict only that
                    # backend, once at root admission, and restore on exit.
                    # Workers inherit this process-wide setting unchanged.
                    compatibility = (ThreadpoolController().select(internal_api="openblas")
                                     .select(version="0.3.28").limit(limits=1)
                                     if sys.platform == "linux" else nullcontext())
                    with compatibility:
                        yield allocation
            finally:
                _CURRENT.reset(token)


@contextmanager
def worker_scope(allocation):
    token = _CURRENT.set(allocation)
    reservations = _RESERVATIONS.set(())
    try:
        yield allocation
    finally:
        _RESERVATIONS.reset(reservations)
        _CURRENT.reset(token)


@contextmanager
def reserve_memory(nbytes, label="workspace", *, reuse_parent=False):
    """Atomically reserve additional workspace until context exit.

    Nested requests are additive by default. Set ``reuse_parent=True`` only
    when the requested peak is already included in the nearest parent's
    envelope; a larger request then extends that envelope temporarily.
    Worker scopes never borrow their dispatcher's reservations.

    RSS cannot be attributed reliably to individual concurrent reservations.
    Admission therefore charges both observed RSS growth and outstanding
    claims conservatively. Planners inside a claim may use their own reserved
    envelope, but still see RSS consumption and other workers' claims.
    This is cooperative admission, not an operating-system memory guarantee.
    """
    try:
        amount = index(nbytes)
    except TypeError as exc:
        raise SpecificationError("Memory reservation must be an integer byte count.",
                                 code="resource.memory", stage="execution") from exc
    if amount < 0:
        raise SpecificationError("Memory reservation must be nonnegative.",
                                 code="resource.memory", stage="execution")
    allocation = current_allocation()
    if allocation is None:
        available = _headroom(runtime_resources())
        if available is None and amount:
            raise SpecificationError("Memory reservation needs available-memory counters outside execution_scope.",
                                     code="resource.unavailable", stage="execution")
        if available is not None and amount > available:
            raise SpecificationError("Requested workspace exceeds available memory headroom.",
                                     code="resource.memory", stage="execution",
                                     details={"label": str(label), "requested_bytes": amount,
                                              "available_bytes": available})
        yield
        return
    ledger = allocation._ledger
    with ledger.lock:
        chain = tuple(key for key in _RESERVATIONS.get() if key in ledger.claims)
        reuse = bool(reuse_parent and chain)
        key = chain[-1] if reuse else object()
        previous = ledger.claims[key] if reuse else 0
        claimed = max(previous, amount) if reuse else amount
        additional = claimed - previous
        available = allocation._available_locked(credit_own=False)
        _, cap = allocation._root_and_cap()
        if additional > available or amount > cap:
            raise SpecificationError("Requested workspace exceeds the shared memory allocation.",
                                     code="resource.memory", stage="execution",
                                     details={"label": str(label), "requested_bytes": amount,
                                              "additional_bytes": additional,
                                              "available_bytes": available})
        ledger.claims[key] = claimed
    token = _RESERVATIONS.set(chain if reuse else (*chain, key))
    try:
        yield
    finally:
        _RESERVATIONS.reset(token)
        with ledger.lock:
            if reuse:
                ledger.claims[key] = previous
            else:
                del ledger.claims[key]


def _record(result, allocation):
    if isinstance(result, (list, tuple)):
        for item in result:
            _record(item, allocation)
    elif hasattr(result, "absorb_info"):
        if result.absorb_info is None:
            result.absorb_info = {}
        result.absorb_info["resource_schedule"] = allocation.as_dict()
    elif isinstance(getattr(result, "diagnostics", None), dict):
        result.diagnostics["resource_schedule"] = allocation.as_dict()


def managed_execution(function):
    """Resolve policy once at public entry and restore reusable-model settings."""
    sig = signature(function)
    def invoke(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        owner = bound.arguments.get("self")
        attr = next((name for name in ("execution", "execution_config")
                     if owner is not None and hasattr(owner, name)), None)
        config = bound.arguments.get("execution_config")
        if config is None and bound.arguments.get("execution_context") is not None:
            config = bound.arguments["execution_context"].config
        if config is None and attr:
            config = getattr(owner, attr)
        if config is None:
            config = ExecutionConfig(
                threads=bound.arguments.get("absorb_threads", "auto"),
                memory_budget_mb=bound.arguments.get("memory_budget_mb", "auto"),
            )
        with execution_scope(config) as allocation:
            saved = getattr(owner, attr) if attr else None
            context = bound.arguments.get("execution_context")
            if context is None and owner is not None:
                context = getattr(owner, "context", None)
            context_config = getattr(context, "config", None)
            if attr:
                setattr(owner, attr, allocation.config)
            if context_config is not None:
                context.config = allocation.config
            if "execution_config" in sig.parameters:
                bound.arguments["execution_config"] = allocation.config
            if "memory_budget_mb" in sig.parameters:
                bound.arguments["memory_budget_mb"] = allocation.config.memory_budget_mb
            if "absorb_threads" in sig.parameters:
                bound.arguments["absorb_threads"] = allocation.cpu_threads
            try:
                result = function(*bound.args, **bound.kwargs)
                _record(result, allocation)
                return result
            finally:
                if attr:
                    setattr(owner, attr, saved)
                if context_config is not None:
                    context.config = context_config
    @wraps(function)
    def wrapped(*args, **kwargs):
        if current_allocation() is not None:
            return invoke(*args, **kwargs)
        # Read reusable-model policy only after admission: another call may
        # currently hold a temporary resolved configuration on that object.
        with _ROOT_LOCK:
            owner = sig.bind(*args, **kwargs).arguments.get("self")
            prepare = getattr(owner, "_prepare_resource_admission", None)
            if prepare is not None:
                prepare()
            return invoke(*args, **kwargs)
    return wrapped
