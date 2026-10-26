from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..errors import SpecificationError
from ..planner.scheduler import current_allocation, inner_thread_request

from ..planner import (
    PlanCertificate, RepresentationCandidate, choose_representation,
    plan_execution, runtime_resources,
)


@dataclass(frozen=True, slots=True)
class WorkspacePlan:
    mode: str
    pool_size: int
    threads: int
    memory_limit_bytes: int | None
    memory_current_bytes: int | None
    headroom_bytes: int | None
    reason: str

    def as_dict(self):
        return {
            'mode': self.mode,
            'pool_size': int(self.pool_size),
            'threads': int(self.threads),
            'memory_limit_bytes': self.memory_limit_bytes,
            'memory_current_bytes': self.memory_current_bytes,
            'headroom_bytes': self.headroom_bytes,
            'reason': self.reason,
        }


def _pow2_floor(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n).bit_length() - 1)


def _plan_workspace(
    *, nobs, ncols, pool_size, memory_budget_mb, requested_threads,
    bytes_per_column, resident_solver_bytes=0, reserve_fraction=0.12,
):
    """Internal byte-based plan; resident solver state shares the user budget."""
    nobs = int(nobs); ncols = max(1, int(ncols))
    requested_threads = inner_thread_request(requested_threads)
    budget = int(float(memory_budget_mb) * 1024**2)
    if budget <= 0:
        raise SpecificationError("memory_budget_mb must be positive")
    resources = runtime_resources()
    execution = plan_execution(
        requested_budget_bytes=budget,
        requested_threads=1 if requested_threads in (None, "auto") and nobs < 100_000 else requested_threads,
        resources=resources,
        reserve_fraction=reserve_fraction, calibrate_auto_threads=(requested_threads in (None, "auto") and nobs >= 100_000),
    )
    threads = execution.parallel.inner_threads
    memory = execution.memory
    # Resident state is already reflected in runtime current usage. Charge it
    # against the configured solver budget, not twice against host headroom.
    usable = budget - int(resident_solver_bytes)
    if memory.headroom_bytes is not None:
        usable = min(usable, memory.headroom_bytes)
    per_col = max(1, int(bytes_per_column))
    cap = max(0, usable // per_col)
    explicit = pool_size not in (None, 'auto')
    requested = max(1, min(int(pool_size), ncols)) if explicit else 1
    if requested > cap:
        raise SpecificationError(
            "Requested HDFE workspace does not fit the memory budget or available headroom.",
            code="resource.memory", stage="execution",
            details={"pool_size": requested, "bytes_per_column": per_col,
                     "resident_solver_bytes": int(resident_solver_bytes),
                     "requested_budget_bytes": budget, "available_workspace_bytes": max(0, usable)},
            suggestion="Reduce pool_size or use 'auto'; if one column cannot fit, increase the budget or free memory.",
        )
    if pool_size not in (None, 'auto'):
        p = requested
        return WorkspacePlan('full_width_memory' if p == ncols else 'pooled_memory', p, threads,
                             resources.memory_limit_bytes, resources.memory_current_bytes,
                             memory.headroom_bytes, 'explicit_pool_size')

    raw = max(1, min(ncols, cap))
    p = min(ncols, _pow2_floor(raw))
    mode = 'full_width_memory' if p >= ncols else 'pooled_memory'
    reason = 'fits_full_width' if mode == 'full_width_memory' else 'bounded_by_memory_headroom'
    return WorkspacePlan(
        mode, p, threads, resources.memory_limit_bytes, resources.memory_current_bytes,
        memory.headroom_bytes, reason,
    )


def _check_allocation_headroom(nbytes):
    """Check a known upcoming allocation; inputs are not a solver-workspace budget."""
    active = current_allocation()
    if active is not None and int(nbytes) > active.remaining_memory_bytes():
        raise SpecificationError("Allocation exceeds the remaining shared computation budget.",
                                 code="resource.memory", stage="execution",
                                 details={"allocation_bytes": int(nbytes),
                                          "headroom_bytes": active.remaining_memory_bytes()})
    resources = runtime_resources()
    if resources.memory_limit_bytes is None or resources.memory_current_bytes is None:
        return
    reserve = (active.reserve_bytes if active is not None else
               int(0.12 * resources.memory_limit_bytes))
    headroom = max(0, resources.memory_limit_bytes - resources.memory_current_bytes - reserve)
    if int(nbytes) > headroom:
        raise SpecificationError(
            "Insufficient available memory for the dense HDFE design block.",
            code="resource.memory", stage="execution",
            details={"allocation_bytes": int(nbytes), "headroom_bytes": headroom},
            suggestion="Free memory or reduce the design width before estimation.",
        )


def should_probe_plain_map(*, nobs: int, ncols: int, method: str, acceleration: str, plan: WorkspacePlan) -> bool:
    """Whether auto execution should try bounded plain MAP before CG."""
    return (
        method == 'map' and acceleration == 'cg' and nobs >= 1_000_000 and ncols >= 4
        and plan.pool_size < ncols
    )


class ReusableRHSWorkspace:
    """One contiguous workspace reused for all RHS pools."""
    def __init__(self, nobs: int, width: int, dtype=np.float64):
        self.array = np.empty((int(nobs), int(width)), dtype=dtype, order='C')

    def load(self, source: np.ndarray, lo: int, hi: int) -> np.ndarray:
        width = int(hi - lo)
        # A short final pool must not retain the full pool row stride.
        nobs = self.array.shape[0]
        view = self.array.reshape(-1)[:nobs * width].reshape(nobs, width)
        np.copyto(view, source[:, lo:hi])
        return view

    @staticmethod
    def write(target: np.ndarray, lo: int, hi: int, work: np.ndarray) -> None:
        target[:, lo:hi] = work[:, :hi-lo]


@dataclass(frozen=True, slots=True)
class DesignStoragePlan:
    """Estimator-agnostic representation decision for a compiled design."""

    representation: str
    expected_passes: int
    dense_bytes: int
    representation_bytes: int
    savings_fraction: float
    reason: str

    def as_dict(self):
        return {
            "representation": self.representation,
            "expected_passes": int(self.expected_passes),
            "dense_bytes": int(self.dense_bytes),
            "representation_bytes": int(self.representation_bytes),
            "savings_fraction": float(self.savings_fraction),
            "reason": self.reason,
        }


def plan_design_storage(
    structure,
    *,
    expected_passes: int = 1,
    memory_budget_mb: float | None = None,
    min_savings_fraction: float = 0.20,
) -> DesignStoragePlan:
    """Choose dense vs. block-dense storage through shared planner contracts.

    Exactness is certified before cost policy.  The phase-1 unified planner
    intentionally preserves the established one-shot/repeated-pass policy so
    architecture can change without silently changing model execution.
    """
    expected = max(1, int(expected_passes))
    dense = int(structure.dense_bytes)
    column_refs = sum(len(b.columns) for b in structure.blocks)
    block = int(
        structure.block_dense_bytes
        + 8 * structure.nobs
        + 8 * column_refs
        + 64 * len(structure.blocks)
    )
    savings = 0.0 if dense <= 0 else max(0.0, 1.0 - block / dense)
    budget = None if memory_budget_mb is None else int(float(memory_budget_mb) * 1024**2)

    block_exact = bool(structure.certified_block_separable)
    block_cert = PlanCertificate(
        name="block_dense", eligible=block_exact, exact=block_exact,
        reason="exact_block_structure" if block_exact else "not_certified_block_separable",
    )
    dense_cert = PlanCertificate.exact_yes("dense", "dense_representation_is_always_exact")
    candidates = (
        RepresentationCandidate("dense", dense_cert, dense, pass_bytes=dense),
        RepresentationCandidate("block_dense", block_cert, block, pass_bytes=block),
    )

    if not block_exact:
        required, reason = "dense", "not_certified_block_separable"
    elif block >= dense or savings < float(min_savings_fraction):
        required, reason = "dense", "insufficient_storage_savings"
    elif budget is not None and dense > budget and block <= budget:
        required, reason = "block_dense", "dense_exceeds_memory_budget"
    elif expected >= 2:
        required, reason = "block_dense", "repeated_operator_passes"
    else:
        required, reason = "dense", "one_shot_dense_avoids_compile_cost"

    # Use the common representation contract to enforce exact eligibility.
    choose_representation(candidates, expected_passes=expected, required_name=required)
    rep_bytes = block if required == "block_dense" else dense
    rep_savings = savings if required == "block_dense" else 0.0
    return DesignStoragePlan(required, expected, dense, rep_bytes, rep_savings, reason)
