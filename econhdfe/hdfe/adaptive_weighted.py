"""Conservative, opt-in nonlinear FE routing; never changes the estimand.

This is a structural eligibility policy, not an optimal timing oracle. Only
three-way pure-intercept designs with strong repetition or a small Schur block
are admitted. Native trials are transactional and have a bounded iteration
allowance. No two expensive candidates are constructed for a timing race.
"""
from __future__ import annotations

import importlib
import math
from time import perf_counter

import numpy as np

from .weighted_projection import WeightedFEProjector
from .workspace import _absorption_build_cost
from ..errors import ConvergenceError, SpecificationError
from ..planner import scheduler

_MIN_ROWS = 100_000
_MAX_SCHUR_WIDTH = 64
_MAX_CELL_FRACTION = 0.25
_NATIVE_ITERATION_CAP = 256


def _native_available():
    try:
        native = importlib.import_module("_econhdfe_within")
    except (ImportError, OSError):
        return False
    return getattr(native, "ABI_VERSION", None) == 1 and hasattr(native, "NativeSolver")


class AdaptiveWeightedFEProjector(WeightedFEProjector):
    """Explicit adaptive engine with strict certificates and sticky fallback.

    The fallback has the optimized engine's numerical choices, including its
    two-way solver. Zero weights use LSMR. The ordinary native/legacy classes
    are unchanged. A declined/failed native candidate is not retried within
    this projector; a changed estimation sample gets a new projector.
    """

    def __init__(self, groups, *, engine="adaptive", method="map", **kwargs):
        if method != "map":
            raise SpecificationError("adaptive routing expects method='map'")
        kwargs["require_convergence"] = True
        super().__init__(groups, engine="optimized", method="map", **kwargs)
        self._total_max_iter = self.max_iter
        self._trial_cap = min(_NATIVE_ITERATION_CAP, self.max_iter // 4)
        self._selection = None
        self._strategy = None
        self._disabled = False
        self._native_required = 0
        self._events = []
        self._probe_seconds = 0.0

    def _capacity(self):
        budget = int(self.projection_memory_budget_mb * 1024**2)
        allocation = scheduler.current_allocation()
        return budget if allocation is None else min(budget, allocation.unreserved_memory_bytes())

    def _select(self):
        start = perf_counter()
        n = len(self.groups[0]) if self.groups else 0
        levels = [int(g.max()) + 1 if len(g) else 0 for g in self.groups]
        s = self._selection = dict(
            policy="nonlinear-structural-v1", nobs=n, levels=levels,
            candidate=None, reason="outside_validated_size_or_dimension",
            exact_cells=None, probe_bytes=0,
        )
        try:
            if n < _MIN_ROWS or len(levels) != 3 or self._trial_cap < 1:
                return
            if not _native_available():
                s["reason"] = "native_unavailable_or_incompatible"
                return
            big = max(levels)
            q = sum(levels) - big - len(levels) + 1
            s["schur_width"] = q
            extra = 0
            if 0 < q <= _MAX_SCHUR_WIDTH:
                extra = 8*(big*q + 6*q*q + 4*big + 8*q) + 16*len(levels) + 256
                candidate = "schur"
            else:
                # Exact integer encoding only. No wrapping/hashing collisions.
                if math.prod(levels) > np.iinfo(np.int64).max:
                    s["reason"] = "tuple_key_overflow"
                    return
                probe_bytes = 40*n  # keys, sorted copy, mask and unique output
                s["probe_bytes"] = probe_bytes
                if probe_bytes > self._capacity() // 8:
                    s["reason"] = "tuple_probe_budget"
                    return
                with scheduler.reserve_memory(probe_bytes, label="adaptive_fe_tuple_probe"):
                    keys = np.zeros(n, dtype=np.int64)
                    for g, L in zip(self.groups, levels):
                        keys *= L
                        keys += g
                    limit = int(n * _MAX_CELL_FRACTION)
                    # A prefix gives a rigorous LOWER bound on total cells.
                    # This rejects broad crossings early without extrapolation.
                    lower = len(np.unique(keys[:min(n, 2*limit+1)]))
                    s["cell_lower_bound"] = lower
                    if lower > limit:
                        s["reason"] = "insufficient_repetition"
                        return
                    cells = len(np.unique(keys))
                    s["exact_cells"] = cells
                    if cells > limit:
                        s["reason"] = "insufficient_repetition"
                        return
                candidate = "compressed"
            # Include the guarded native solver/fallback inventories, one RHS,
            # and Schur retained state. Actual input width is added at call time.
            base = 8*n*len(levels) + 24*n + 32*sum(levels)
            column = 64*n + 96*sum(levels)
            self._native_required = max(
                _absorption_build_cost(self.groups, "within") + extra,
                2*(base + column) + extra,
            )
            s.update(candidate=candidate, reason="structural_candidate",
                     estimated_native_bytes=self._native_required)
            self._strategy = candidate
        except SpecificationError as exc:
            if exc.code != "resource.memory":
                raise
            s["reason"] = "tuple_probe_admission"
        finally:
            self._probe_seconds += perf_counter() - start

    def _demote(self, reason):
        self._events.append(reason)
        self._disabled = True
        # Release native references before a different numerical engine is built.
        self._absorbers.clear()
        self._absorber = None
        self._active_kind = None
        self.method = "map"

    def _validate_weights(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        n = len(self.groups[0]) if self.groups else 0
        if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
            raise SpecificationError("adaptive weights must be finite, nonnegative and have positive support")
        return w

    def prepare(self, weights, *, tol):
        w = self._validate_weights(weights)
        if self._selection is None:
            self._select()
        if self._strategy is not None and not self._disabled:
            if np.any(w == 0):
                self._demote("zero_weight_support")
            else:
                self.method = "within"
        if self.method != "within":
            self.method = "lsmr" if np.any(w == 0) else "map"
            # Switching MAP/LSMR must not reuse the other's compiled object.
            old = self._absorbers.get("generic")
            if old is not None and old.method != self.method:
                self._absorbers.pop("generic")
                self._absorber = None
        absorber = super().prepare(w, tol=tol)
        absorber.max_iter = self._trial_cap if self.method == "within" else self.max_iter
        return absorber

    def _build(self, weights, tol, kind):
        if kind == "within":
            from .within_backend import WithinFEAbsorber
            return WithinFEAbsorber(
                self.groups, weights=weights, tol=tol, max_iter=self._trial_cap,
                absorb_threads=self.absorb_threads_requested,
                projection_memory_budget_mb=self.projection_memory_budget_mb,
                strategy=self._strategy,
            )
        return super()._build(weights, tol, kind)

    def residualize(self, data, weights, *, tol, return_info=False, copy=True):
        source = np.asarray(data, dtype=np.float64)
        n = len(self.groups[0]) if self.groups else 0
        if source.ndim not in (1, 2) or len(source) != n or not np.isfinite(source).all():
            raise SpecificationError("adaptive RHS must be finite with one row per observation")
        if not copy and not source.flags.writeable:
            raise SpecificationError("copy=False requires a writable RHS")
        if not np.isfinite(tol) or not 0 < tol <= 1:
            raise SpecificationError("adaptive tolerance must lie in (0, 1]")
        w = self._validate_weights(weights)
        if self._selection is None:
            self._select()
        self.max_iter = self._total_max_iter
        if self._strategy is not None and not self._disabled:
            if np.any(w == 0):
                self._demote("zero_weight_support")
            elif self._native_required + source.nbytes > self._capacity():
                self._demote("native_workspace_admission")
        attempt_native = self._strategy is not None and not self._disabled
        if not attempt_native:
            return super().residualize(source, w, tol=tol, return_info=return_info, copy=copy)
        failure = None
        try:
            # Always isolate a native trial, including copy=False callers.
            # This output reservation is part of the existing scheduler ledger.
            out, info = super().residualize(source, w, tol=tol, return_info=True, copy=True)
            if not np.isfinite(out).all():
                raise ConvergenceError("Native result is nonfinite", code="convergence.weighted_fe")
        except ConvergenceError as exc:
            if exc.code != "convergence.weighted_fe":
                raise
            failure = "native_certificate_failed"
        except SpecificationError as exc:
            if exc.code != "resource.memory" and "within budget cannot fit" not in str(exc):
                raise
            failure = "native_workspace_admission"
        except ValueError as exc:
            if not str(exc).startswith("Schur rejected:"):
                raise
            failure = "schur_numeric_gate"
        except MemoryError:
            failure = "native_allocation_failed"
        if failure is not None:
            # Conservatively charge the FULL trial cap per RHS, even on an
            # early build failure. No trial + fallback can exceed max_iter.
            self._demote(failure)
            self.max_iter = self._total_max_iter - self._trial_cap
            try:
                out, info = super().residualize(source, w, tol=tol, return_info=True, copy=copy)
                info.iterations += self._trial_cap
                info.criterion = "adaptive_budgeted_fallback_" + info.criterion
            finally:
                self.max_iter = self._total_max_iter
        elif not copy:
            source[...] = out
            out = source
        return (out, info) if return_info else out

    @property
    def resource_info(self):
        info = super().resource_info
        info["engine"] = "adaptive"
        info["selection"] = dict(
            **(self._selection or {"reason": "not_prepared"}),
            probe_seconds=self._probe_seconds, fallback_events=tuple(self._events),
            native_disabled=self._disabled, native_iteration_cap=self._trial_cap,
            max_iter_per_rhs=self._total_max_iter,
            budget_scope="known_workspace_estimates_not_total_native_rss",
        )
        if self._active_kind == "within":
            info["native"] = self._absorber.resource_info
        return info
