from __future__ import annotations

from dataclasses import replace

from ...config import ExecutionConfig
from ...planner.scheduler import current_allocation, resolve_allocation
from ...compute.context import ExecutionContext
from ...hdfe.plan import FEPlan
from ...hdfe.weighted_projection import WeightedFEProjector


def resolve_execution(
    execution_config: ExecutionConfig | None = None,
    execution_context: ExecutionContext | None = None,
) -> tuple[ExecutionConfig, ExecutionContext]:
    """Return one validated PPML execution policy and lifecycle context."""
    if execution_config is None:
        execution_config = execution_context.config if execution_context is not None else ExecutionConfig()
    execution_config.validate()
    allocation = current_allocation() or resolve_allocation(execution_config)
    if execution_config is not allocation.config:
        execution_config = replace(
            execution_config, threads=allocation.cpu_threads,
            memory_budget_mb=allocation.memory_budget_bytes / (1024 ** 2),
        )
    if execution_context is None:
        execution_context = ExecutionContext(execution_config)
    else:
        execution_context.config = execution_config
    return execution_config, execution_context


def _projector_key(plan: FEPlan, *, engine: str, method: str, execution: ExecutionConfig) -> str:
    allocation = current_allocation()
    automatic = allocation.auto_threads if allocation is not None else execution.threads == "auto"
    return (
        f"ppml.projector:{plan.fingerprint}:{engine}:{method}:"
        f"threads={execution.threads}:auto={automatic}:memory={execution.memory_budget_mb}"
    )


def get_projector(
    plan: FEPlan,
    *,
    engine: str,
    method: str = "map",
    execution: ExecutionConfig,
    context: ExecutionContext | None = None,
    cache: bool = True,
) -> WeightedFEProjector | None:
    """Build or reuse a resource-aware weighted FE projector.

    Only estimator projectors should normally be placed in a reusable model's
    persistent ``ExecutionContext``.  Separation projectors are intentionally
    short-lived because simplex can introduce zero weights, which changes the
    numerical topology available to optimized absorbers.
    """
    if not plan.groups:
        return None
    key = _projector_key(plan, engine=engine, method=method, execution=execution)
    if cache and context is not None:
        hit = context.get(key)
        if isinstance(hit, WeightedFEProjector):
            return hit
    out = WeightedFEProjector(
        plan.groups,
        engine=engine,
        method=method,
        absorb_threads=execution.threads,
        projection_memory_budget_mb=execution.memory_budget_mb,
    )
    if cache and context is not None:
        context.put(key, out)
    return out
