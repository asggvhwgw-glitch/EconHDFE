"""Optional structured native FE absorption; inference stays on original rows.

The Rust extension depends on a pinned within revision. This module never selects
it implicitly: method='within' is opt-in, and the base installation stays pure
Python. Numerical weights are not frequency counts or inference multiplicities.
"""
from __future__ import annotations
from threading import RLock
import numpy as np
from .absorber import AbsorbInfo, HDFEAbsorber
from .projection import resolve_threads
from ..errors import SpecificationError


class WithinFEAbsorber:
    """Pure-intercept adapter preserving the standard residualizer contract."""
    def __init__(self, groups, *, weights=None, tol=1e-8, max_iter=16_000,
                 absorb_threads="auto", projection_memory_budget_mb=512,
                 strategy="auto"):
        try:
            import _econhdfe_within as native
        except ImportError as exc:
            raise SpecificationError(
                "method='within' requires the optional native extension; "
                "install ./native/within-accel from the source checkout."
            ) from exc
        if getattr(native, "ABI_VERSION", None) != 1:
            raise SpecificationError("incompatible econhdfe-within extension ABI")
        arrays = [np.asarray(g) for g in groups]
        if not arrays or any(g.ndim != 1 for g in arrays):
            raise SpecificationError("within requires nonempty one-dimensional FE groups")
        n = len(arrays[0])
        if not n or any(len(g) != n or g.dtype.kind not in "iu" or
                        np.any(g < 0) or np.any(g > np.iinfo(np.int32).max)
                        for g in arrays):
            raise SpecificationError("within requires equal-length dense nonnegative integer FE codes")
        self.groups = [np.asarray(g, dtype=np.int32) for g in arrays]
        self.levels = [int(g.max()) + 1 for g in self.groups]
        self.nobs = n
        self.weights = self._validated_weights(weights)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.absorb_threads = resolve_threads(absorb_threads, nobs=n)
        self.projection_memory_budget_mb = float(projection_memory_budget_mb)
        if not np.isfinite(self.projection_memory_budget_mb) or self.projection_memory_budget_mb <= 0:
            raise SpecificationError("within memory budget must be finite and positive")
        self.method = "within"
        self.transform = "structured_native"
        self.acceleration = "none"
        self.preconditioner = "within_adaptive"
        self.projection_backend = "within_structured"
        self.slopes = [None] * len(arrays)
        self.intercepts = [True] * len(arrays)
        self._lock = RLock()
        self._profile_enabled = False
        self._iteration_profile = {}
        self._native = native.NativeSolver(
            np.asfortranarray(np.column_stack(arrays), dtype=np.uint32),
            weights=self.weights, strategy=strategy, threads=self.absorb_threads,
            workspace_bytes=int(self.projection_memory_budget_mb * 1024**2),
        )
        self.index_bytes = 0  # Native allocations are not ndarray-owned bytes.

    def _validated_weights(self, weights):
        if weights is None:
            return None
        w = np.array(weights, dtype=np.float64, copy=True)
        if w.shape != (self.nobs,) or not np.isfinite(w).all() or np.any(w <= 0):
            raise SpecificationError("native within requires strictly positive finite weights")
        return w

    def update_weights(self, weights, *, tol=None):
        w = self._validated_weights(weights)
        if w is None:
            w = np.ones(self.nobs)
        with self._lock:
            self._native.update_weights(w)
            self.weights = w
            if tol is not None:
                self.tol = float(tol)
        return self

    @property
    def supports_fe_orthogonality_certificate(self):
        return False

    @property
    def supports_projection_shortcut(self):
        return type(self) is WithinFEAbsorber

    supports_wild_cluster_test = False

    def pending_cost(self):
        return 0

    def absorb_info_extras(self):
        return {"within": self.resource_info}

    def workspace_cost(self):
        # Own arrays + conservative observation/coefficient work estimate.
        # Upstream Schwarz factorization has no hard memory-budget API. This is
        # explicitly not a bound on native peak RSS or upstream factor fill-in.
        levels = sum(self.levels)
        base_resident = 8*self.nobs*len(self.groups) + 24*self.nobs + 32*levels
        primary_column = 64*self.nobs + 96*levels
        extra = getattr(self._native, "extra_resident_bytes", None)
        guarded = extra is not None
        legacy_schur = not guarded and self._native.route == "schur"
        if extra is None:
            extra = 0
            if legacy_schur:
                big = max(self.levels)
                q = levels - big - len(self.levels) + 1
                # Older ABI-1 wheels lack an exact retained-state getter; their
                # Schur build estimate conservatively covers its live storage.
                extra = 8*(big*q + 6*q*q + 4*big + 8*q) + 16*len(self.levels) + 256
        resident = base_resident + int(extra)
        # Certification can keep the primary engine/result alive while a fresh
        # original-row diagonal solver runs. Reserve its state before pooling
        # when the budget can admit it; otherwise retain primary-only operation.
        # Native admission independently declines fallback below this total.
        fallback_fits = 2*(base_resident + primary_column) + int(extra) <= int(
            self.projection_memory_budget_mb * 1024**2
        )
        # A legacy Schur wheel cannot disable its fallback, so reserve it even
        # if that means rejecting an insufficient budget before native entry.
        reserve_fallback = (guarded and fallback_fits) or legacy_schur
        per_column = base_resident + 2*primary_column if reserve_fallback else primary_column
        return int(resident), max(1, int(per_column))

    @property
    def resource_info(self):
        return {
            "backend": "within", "route": self._native.route,
            "nobs": self.nobs, "n_cells": self._native.n_cells,
            "threads": self.absorb_threads,
            "weight_generation": self._native.weight_generation,
            "schur_fallbacks": getattr(
                self._native, "schur_fallback_count", self._native.fallback_count,
            ),
            "certification_fallbacks": self._native.fallback_count,
            "schur_condition": self._native.schur_condition,
            "budget_scope": "reduction_workspace_and_rhs_pool_not_total_native_rss",
        }

    def residualize(self, data, *, copy=True, return_info=False, absorb_threads=None):
        a = np.asarray(data, dtype=np.float64)
        one_dim = a.ndim == 1
        if one_dim:
            a = a[:, None]
        if a.ndim != 2 or a.shape[0] != self.nobs or not np.isfinite(a).all():
            raise SpecificationError("within RHS must be finite with one row per observation")
        if not copy and not a.flags.writeable:
            raise SpecificationError("copy=False requires a writable RHS")
        out = a.copy() if copy else a
        nthreads = self.absorb_threads if absorb_threads is None else resolve_threads(absorb_threads)
        # Only one Krylov solve is live; pool width also bounds the bridge buffer.
        resident, per = self.workspace_cost()
        cap = (int(self.projection_memory_budget_mb*1024**2) - resident)//per
        if cap < 1 and a.shape[1]:
            raise SpecificationError("within budget cannot fit one RHS; increase memory_budget_mb")
        width = max(1, min(a.shape[1] or 1, cap))
        worst = AbsorbInfo(True, 0, 0.0, "within_empty")
        with self._lock:
            self._native.set_threads(nthreads)
            try:
                for lo in range(0, a.shape[1], width):
                    block, converged, iterations, errors = self._native.solve(
                        a[:, lo:lo+width], tol=self.tol, max_iter=self.max_iter,
                    )
                    out[:, lo:lo+width] = block
                    worst.converged = worst.converged and all(converged)
                    worst.iterations = max(worst.iterations, max(iterations, default=0))
                    worst.max_update = max(worst.max_update, max(errors, default=0.0))
                worst.criterion = "within_" + self._native.route + "_stationarity"
                self._iteration_profile = self.resource_info
            finally:
                self._native.set_threads(self.absorb_threads)
        result = out[:, 0] if one_dim else out
        return (result, worst) if return_info else result

    def solve_effects(self, target):
        # Keep reporting/normalization on the established econhdfe recovery path.
        recovery = HDFEAbsorber(
            self.groups, weights=self.weights, method="lsmr", acceleration="none",
            tol=self.tol, max_iter=self.max_iter,
        )
        return recovery.solve_effects(target)
