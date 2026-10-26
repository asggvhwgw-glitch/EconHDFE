from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from ..config import ExecutionConfig
from ..planner.resources import runtime_resources
from ..planner.scheduler import current_allocation
from .cache_accounting import retained_array_bytes


@dataclass(slots=True)
class ProfileRecorder:
    mode: str = "off"
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (perf_counter() - t0)

    def as_dict(self) -> dict[str, float]:
        return dict(self.timings)


@dataclass(slots=True)
class ExecutionContext:
    """Internal lifecycle container for caches, resource policy and profiling."""
    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    cache: dict[str, Any] = field(default_factory=dict)
    profiler: ProfileRecorder = field(init=False)
    _protected: tuple = field(default=(), init=False, repr=False)
    _policy_signature: tuple | None = field(default=None, init=False, repr=False)
    _resource_signature: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.config.validate()
        self.profiler = ProfileRecorder(self.config.profile)

    @property
    def retained_bytes(self) -> int:
        """Backing arrays held by cache, excluding caller-owned input storage.

        Dropping these references does not guarantee an RSS decrease: active
        callers or the allocator may retain the same storage.
        """
        return retained_array_bytes(self.cache, self._protected)

    def _capacity_bytes(self) -> int:
        resources = runtime_resources()
        headroom = None
        if resources.memory_limit_bytes is not None and resources.memory_current_bytes is not None:
            headroom = max(0, resources.memory_limit_bytes - resources.memory_current_bytes
                           - int(.12 * resources.memory_limit_bytes))
        requested = self.config.memory_budget_mb
        budget = headroom if requested == "auto" else int(requested * 1024**2)
        if budget is None:
            return 0
        # Release disposable references before the root's 32 MiB admission
        # floor is breached, even when their current size fits host headroom.
        if headroom is not None and headroom < 32 * 1024**2:
            return 0
        allocation = current_allocation()
        if allocation is not None:
            budget = min(budget, allocation.unreserved_memory_bytes())
        # Disposable caches must leave the majority of live capacity for work.
        capacity = budget // 4
        return capacity if headroom is None else min(capacity, headroom)

    def trim(self) -> None:
        if self.config.cache == "off":
            self.clear()
            return
        capacity = self._capacity_bytes()
        while self.cache and self.retained_bytes > capacity:
            del self.cache[next(iter(self.cache))]

    def prepare_admission(self, config, *, protected=()) -> None:
        """Release obsolete/cache-off entries before root resource admission."""
        self.config = config
        self._protected = tuple(protected)
        signature = (config.threads, config.memory_budget_mb)
        if self._policy_signature is not None and signature != self._policy_signature:
            self.clear()
        self._policy_signature = signature
        self.trim()

    @contextmanager
    def maintain(self, config, *, protected=()):
        """Bound reusable caches before and after lazily compiled objects grow."""
        self.config = config
        self._protected = tuple(protected)
        signature = (config.threads, config.memory_budget_mb)
        if self._resource_signature is not None and signature != self._resource_signature:
            self.clear()
        self._resource_signature = signature
        self.trim()
        try:
            yield
        finally:
            self.trim()

    def get(self, signature: str):
        self.trim()
        if signature not in self.cache:
            return None
        # Dict insertion order provides the LRU queue without separate state.
        value = self.cache.pop(signature)
        self.cache[signature] = value
        return value

    def put(self, signature: str, value: Any) -> None:
        if self.config.cache == "off":
            self.clear()
            return
        self.cache.pop(signature, None)
        self.cache[signature] = value
        self.trim()

    def clear(self) -> None:
        self.cache.clear()
