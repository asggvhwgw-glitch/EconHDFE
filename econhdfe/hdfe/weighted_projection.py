from __future__ import annotations

from dataclasses import replace
import numpy as np

from .factory import build_absorber, projection_kind
from .projection import resolve_threads
from .protocol import ReweightableAbsorber
from .workspace import absorption_build_cost
from ..planner import scheduler
from ..compute.execution import _plan_workspace
from ..errors import SpecificationError, ConvergenceError


class WeightedFEProjector:
    """Reusable weighted FE projector for iterative WLS estimators.

    Categorical topology is compiled once. Subsequent calls update only the
    numerical weights through the public absorber ``update_weights`` contract.
    Resource controls are explicit so iterative estimators can honor the same
    ``ExecutionConfig`` thread and memory policy as the linear estimators.
    """

    def __init__(
        self,
        groups,
        *,
        engine: str,
        method: str = "map",
        max_iter: int = 16_000,
        absorb_threads="auto",
        projection_memory_budget_mb: float = 512,
    ):
        self.groups = tuple(np.asarray(g, dtype=np.int32) for g in groups)
        self.engine = str(engine)
        # PPML-family APIs share this entry, including block-local projectors.
        # Preserve an explicit LSMR request (e.g. a separation auxiliary solve).
        self.method = "within" if self.engine == "within" and method == "map" else str(method)
        self.max_iter = int(max_iter)
        self.absorb_threads_requested = absorb_threads
        self.projection_memory_budget_mb = float(projection_memory_budget_mb)
        self._absorber = None
        # Optimized two-way projection cannot be updated to a weight vector
        # that removes an entire FE level. Keep separate compiled variants so
        # zero-weight simplex projections never poison the positive-weight IRLS
        # fast path.
        self._absorbers: dict[str, ReweightableAbsorber] = {}
        self._active_kind: str | None = None

    def prepare(self, weights, *, tol):
        """Bind one numerical weight state and return the compiled absorber."""
        w = np.asarray(weights, dtype=np.float64)
        kind, solver = projection_kind(self.method, engine=self.engine, ngroups=len(self.groups), weights=w)
        absorber = self._absorbers.get(kind)
        if absorber is None:
            # Probe only at an unreserved planning boundary; construction below
            # may use the cached decision while holding its memory claim.
            resolve_threads(self.absorb_threads_requested, nobs=len(w))
            with scheduler.reserve_memory(
                absorption_build_cost(self.groups, kind), label="weighted_fe_build",
            ):
                absorber = build_absorber(
                    kind, self.groups, weights=w, tol=tol, max_iter=self.max_iter,
                    absorb_threads=self.absorb_threads_requested,
                    memory_budget_mb=self.projection_memory_budget_mb,
                    method=solver, transform="symmetric",
                    acceleration="cg" if self.method == "map" else "none",
                    projection_backend="auto",
                )
            self._absorbers[kind] = absorber
        else:
            absorber.update_weights(w, tol=tol)
        self._absorber = absorber
        self._active_kind = kind
        return absorber

    def residualize(self, data, weights, *, tol, return_info=False, copy=True):
        absorber = self.prepare(weights, tol=tol)
        source = np.asarray(data, dtype=np.float64)
        one_dim = source.ndim == 1
        matrix = source[:, None] if one_dim else source
        with scheduler.reserve_memory(matrix.nbytes if copy else 0, label="weighted_fe_output"):
            if copy:
                matrix = matrix.copy()
            resident, per_column = absorber.workspace_cost()
            plan = _plan_workspace(
                nobs=absorber.nobs, ncols=matrix.shape[1], pool_size="auto",
                memory_budget_mb=self.projection_memory_budget_mb,
                requested_threads=self.absorb_threads_requested,
                bytes_per_column=per_column, resident_solver_bytes=resident,
            )
            pending = absorber.pending_cost()
            allocation = scheduler.current_allocation()
            if allocation is not None:
                # Output reservations stay live through this loop. Planning
                # must leave them and lazy topology room before choosing RHS
                # width, rather than discovering the conflict at admission.
                available = max(0, allocation.unreserved_memory_bytes() - pending)
                cap = available // per_column
                if cap < 1:
                    raise SpecificationError(
                        "No memory remains for one weighted projection column.",
                        code="resource.memory", stage="execution",
                        details={"available_bytes": available, "bytes_per_column": per_column},
                    )
                if cap < plan.pool_size:
                    pool = 1 << (int(cap).bit_length() - 1)
                    plan = replace(plan, pool_size=pool, mode="pooled_memory",
                                   reason="bounded_by_shared_reservations")
            with scheduler.reserve_memory(
                plan.pool_size * per_column + pending, label="weighted_fe_scratch",
            ):
                infos = []
                for lo in range(0, matrix.shape[1], plan.pool_size):
                    hi = min(matrix.shape[1], lo + plan.pool_size)
                    projected, info = absorber.residualize(
                        matrix[:, lo:hi], copy=False, return_info=True,
                        absorb_threads=plan.threads,
                    )
                    # IRLS, rank filters and final VCE callers may discard info.
                    # An opt-in native solve (or its zero-support LSMR route)
                    # must never turn an uncertified projection into an estimate.
                    if self.method == "within" and (
                        not info.converged or not np.isfinite(info.max_update)
                    ):
                        raise ConvergenceError(
                            "within weighted FE projection failed to converge",
                            code="convergence.hdfe", stage="absorption",
                            details={"backend": self._active_kind,
                                     "iterations": int(info.iterations),
                                     "max_update": float(info.max_update),
                                     "criterion": str(info.criterion),
                                     "tolerance": float(tol),
                                     "max_iter": self.max_iter,
                                     "rhs_start": lo, "rhs_stop": hi},
                        )
                    matrix[:, lo:hi] = projected
                    infos.append(info)
            output = matrix[:, 0] if one_dim else matrix
            if return_info:
                info = max(infos, key=lambda item: (not item.converged, item.iterations, item.max_update)) if infos else None
                return output, info
            return output

    @property
    def compiled(self) -> bool:
        return bool(self._absorbers)

    @property
    def resource_info(self) -> dict:
        absorber = self._absorber
        return {
            "engine": self.engine,
            "method": self.method,
            "requested_threads": self.absorb_threads_requested,
            "actual_threads": None if absorber is None else int(getattr(absorber, "absorb_threads", 1)),
            "memory_budget_mb": self.projection_memory_budget_mb,
            "backend": self._active_kind,
            "projection_backend": None if absorber is None else str(getattr(absorber, "projection_backend", getattr(absorber, "method", "unknown"))),
            "index_bytes": None if absorber is None else int(getattr(absorber, "index_bytes", 0)),
            "native": getattr(absorber, "resource_info", None) if self._active_kind == "within" else None,
        }
