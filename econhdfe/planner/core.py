from __future__ import annotations

from .contracts import (
    CandidateCost,
    ExecutionPlan,
    MemoryPlan,
    ParallelPlan,
    PlanCertificate,
    RepresentationCandidate,
    RepresentationPlan,
)
from .resources import RuntimeResources, runtime_resources
from .calibration import auto_thread_count
from .scheduler import current_allocation


def plan_memory(
    *,
    requested_budget_bytes: int,
    estimated_peak_bytes: int | None = None,
    resources: RuntimeResources | None = None,
    reserve_fraction: float = 0.12,
) -> MemoryPlan:
    """Build one memory envelope shared by ingestion/design/HDFE planning.

    ``requested_budget_bytes`` is the package workspace policy. Runtime cgroup
    headroom can tighten it, but never loosen it. This is resource planning only.
    """
    r = runtime_resources() if resources is None else resources
    requested = max(1, int(requested_budget_bytes))
    reserve = 0
    headroom = None
    effective = requested
    reason = "user_budget"
    if r.memory_limit_bytes is not None and r.memory_current_bytes is not None:
        reserve = max(0, int(r.memory_limit_bytes * max(0.0, float(reserve_fraction))))
        headroom = max(0, int(r.memory_limit_bytes - r.memory_current_bytes - reserve))
        effective = max(1, min(requested, headroom))
        if effective < requested:
            reason = "runtime_headroom_tighter_than_user_budget"

    active = current_allocation()
    if active is not None:
        remaining = active.remaining_memory_bytes()
        effective = min(effective, remaining)
        headroom = remaining if headroom is None else min(headroom, remaining)
        reason = "shared_call_budget"

    peak = None if estimated_peak_bytes is None else max(0, int(estimated_peak_bytes))
    feasible = peak is None or peak <= effective
    if peak is None:
        pressure = "unknown"
    else:
        ratio = peak / max(effective, 1)
        if ratio <= 0.30:
            pressure = "low"
        elif ratio <= 0.60:
            pressure = "normal"
        elif ratio <= 0.80:
            pressure = "high"
        else:
            pressure = "critical"
        if not feasible:
            reason = "estimated_peak_exceeds_effective_budget"
    return MemoryPlan(
        requested_budget_bytes=requested,
        effective_budget_bytes=effective,
        runtime_limit_bytes=r.memory_limit_bytes,
        runtime_current_bytes=r.memory_current_bytes,
        headroom_bytes=headroom,
        reserve_bytes=reserve,
        estimated_peak_bytes=peak,
        pressure=pressure,
        feasible=feasible,
        reason=reason,
    )


def plan_parallelism(
    *,
    requested_threads: int | str | None = "auto",
    resources: RuntimeResources | None = None,
    outer_tasks: int = 1,
    prefer_outer: bool = False,
    blas_threads: int = 1,
    calibrate_auto: bool | None = None,
) -> ParallelPlan:
    """Choose a non-oversubscribed parallel layout.

    ``auto`` can use one cached real-machine synthetic calibration to find the
    local memory-bandwidth saturation point.  Injected ``resources`` default to
    the deterministic historical policy unless ``calibrate_auto=True`` is
    requested explicitly; this keeps tests/simulations independent of the host.
    Calibration changes only execution resources, never the econometric model.
    """
    use_runtime = resources is None
    r = runtime_resources() if resources is None else resources
    active = current_allocation()
    cap = max(1, min(int(r.cpu_threads), active.cpu_threads)) if active else max(1, int(r.cpu_threads))
    auto_source = "explicit"
    calibration_id = None
    if requested_threads in (None, "auto"):
        if calibrate_auto is None:
            calibrate_auto = use_runtime
        if calibrate_auto and cap > 1 and not (prefer_outer and int(outer_tasks) > 1):
            requested, calibration = auto_thread_count(max_threads=cap)
            requested = max(1, min(int(requested), cap))
            auto_source = calibration.source
            calibration_id = calibration.calibration_id
        else:
            requested = cap
            auto_source = "runtime_cap"
        requested_label = "auto"
    else:
        requested = max(1, min(int(requested_threads), cap))
        requested_label = int(requested_threads)

    tasks = max(1, int(outer_tasks))
    if prefer_outer and tasks > 1 and requested > 1:
        outer = min(tasks, requested)
        inner = max(1, requested // outer)
        mode = "outer"
        reason = "independent_tasks_share_cpu_budget"
    else:
        outer = 1
        inner = requested
        mode = "inner"
        reason = "calibrated_single_inner_kernel" if requested_label == "auto" and calibrate_auto else (
            "single_inner_kernel" if requested_label == "auto" else "explicit_thread_budget"
        )
    return ParallelPlan(
        mode=mode,
        outer_workers=outer,
        inner_threads=inner,
        blas_threads=max(1, min(int(blas_threads), cap)),
        effective_cpu_threads=cap,
        requested_threads=requested_label,
        reason=reason,
        auto_source=auto_source,
        calibration_id=calibration_id,
    )


def choose_representation(
    candidates,
    *,
    expected_passes: int = 1,
    memory_budget_bytes: int | None = None,
    required_name: str | None = None,
) -> RepresentationPlan:
    """Choose among already-certified physical representations.

    Statistical eligibility is supplied by ``PlanCertificate``. The cost model
    uses byte traffic as a transparent machine-independent first approximation;
    it never turns an uncertified candidate into a valid one.
    """
    expected = max(1, int(expected_passes))
    budget = None if memory_budget_bytes is None else max(1, int(memory_budget_bytes))
    costs = []
    by_name = {}
    for cand in tuple(candidates):
        if not isinstance(cand, RepresentationCandidate):
            raise TypeError("candidates must be RepresentationCandidate objects")
        eligible = bool(cand.certificate.eligible and cand.certificate.exact)
        peak = int(max(0, cand.payload_bytes + cand.metadata_bytes + cand.setup_bytes))
        feasible = eligible and (budget is None or peak <= budget)
        traffic = int(max(0, cand.setup_bytes + cand.metadata_bytes + expected * cand.per_pass_bytes + cand.dispatch_penalty_bytes))
        if not eligible:
            score = float("inf")
            reason = cand.certificate.reason
        elif not feasible:
            score = float("inf")
            reason = "memory_budget_exceeded"
        else:
            score = float(traffic)
            reason = "eligible_and_feasible"
        cost = CandidateCost(cand.name, eligible, feasible, peak, traffic, score, reason)
        costs.append(cost); by_name[cand.name] = cost

    if not costs:
        raise ValueError("at least one representation candidate is required")
    if required_name is not None:
        chosen = by_name.get(required_name)
        if chosen is None:
            raise ValueError(f"required representation {required_name!r} was not supplied")
        if not chosen.feasible:
            raise ValueError(f"required representation {required_name!r} is not feasible")
        reason = "required_representation"
    else:
        feasible = [c for c in costs if c.feasible]
        if not feasible:
            raise MemoryError("no exact execution representation fits the memory budget")
        chosen = min(feasible, key=lambda c: (c.score, c.estimated_peak_bytes, c.name))
        reason = "minimum_estimated_byte_traffic"
    return RepresentationPlan(chosen.name, expected, chosen, tuple(costs), reason)


def exact_certificate(name: str, condition: bool, *, yes: str, no: str) -> PlanCertificate:
    return PlanCertificate.exact_yes(name, yes) if condition else PlanCertificate.no(name, no)


def plan_execution(
    *,
    requested_budget_bytes: int,
    requested_threads: int | str | None = "auto",
    resources: RuntimeResources | None = None,
    estimated_peak_bytes: int | None = None,
    representation_candidates=(),
    expected_passes: int = 1,
    reuse_count: int = 1,
    outer_tasks: int = 1,
    prefer_outer: bool = False,
    reserve_fraction: float = 0.12,
    notes=(),
    calibrate_auto_threads: bool = False,
) -> ExecutionPlan:
    """Assemble one explainable resource/representation/parallel plan.

    This is deliberately generic: callers must provide representation
    certificates rather than asking the planner to infer econometric validity.
    """
    r = runtime_resources() if resources is None else resources
    envelope = plan_memory(
        requested_budget_bytes=requested_budget_bytes, resources=r,
        reserve_fraction=reserve_fraction,
    )
    rep = None
    cands = tuple(representation_candidates)
    if cands:
        rep = choose_representation(
            cands, expected_passes=expected_passes,
            memory_budget_bytes=envelope.effective_budget_bytes,
        )
        if estimated_peak_bytes is None:
            estimated_peak_bytes = rep.selected_cost.estimated_peak_bytes
    memory = plan_memory(
        requested_budget_bytes=requested_budget_bytes,
        estimated_peak_bytes=estimated_peak_bytes, resources=r,
        reserve_fraction=reserve_fraction,
    )
    parallel = plan_parallelism(
        requested_threads=requested_threads, resources=r,
        outer_tasks=outer_tasks, prefer_outer=prefer_outer,
        calibrate_auto=calibrate_auto_threads,
    )
    return ExecutionPlan(
        memory=memory, parallel=parallel, representation=rep,
        reuse_count=max(1, int(reuse_count)), notes=tuple(map(str, notes)),
    )
