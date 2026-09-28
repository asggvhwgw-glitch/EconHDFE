from __future__ import annotations

import numpy as np

from econhdfe.planner import (
    PlanCertificate, RepresentationCandidate, RuntimeResources,
    choose_representation, plan_execution, plan_memory, plan_parallelism,
)


def test_certificate_blocks_uncertified_fast_path_even_when_cheaper():
    dense = RepresentationCandidate(
        "dense", PlanCertificate.exact_yes("dense"), payload_bytes=1_000,
    )
    unsafe = RepresentationCandidate(
        "unsafe", PlanCertificate.no("unsafe", "no_exact_certificate"), payload_bytes=10,
    )
    plan = choose_representation((dense, unsafe), expected_passes=20)
    assert plan.representation == "dense"
    by_name = {c.name: c for c in plan.candidates}
    assert not by_name["unsafe"].eligible
    assert np.isinf(by_name["unsafe"].score)


def test_representation_cost_model_prefers_lower_repeated_byte_traffic():
    exact = PlanCertificate.exact_yes("exact")
    dense = RepresentationCandidate("dense", exact, payload_bytes=1_000)
    block = RepresentationCandidate("block", exact, payload_bytes=300, setup_bytes=250)
    one = choose_representation((dense, block), expected_passes=1)
    repeated = choose_representation((dense, block), expected_passes=4)
    assert one.representation == "block"  # 550 < 1000 in generic cost model
    assert repeated.representation == "block"
    assert repeated.selected_cost.estimated_traffic_bytes < 4_000


def test_memory_plan_tightens_user_budget_to_runtime_headroom():
    resources = RuntimeResources(
        cpu_threads=8,
        memory_limit_bytes=1_000,
        memory_current_bytes=200,
    )
    plan = plan_memory(
        requested_budget_bytes=900,
        estimated_peak_bytes=650,
        resources=resources,
        reserve_fraction=0.10,
    )
    # Runtime headroom is 1000 - 200 - 100 = 700.
    assert plan.effective_budget_bytes == 700
    assert plan.feasible
    assert plan.pressure == "critical"
    assert plan.reason == "runtime_headroom_tighter_than_user_budget"


def test_memory_plan_marks_infeasible_without_changing_semantics():
    resources = RuntimeResources(4, None, None)
    plan = plan_memory(
        requested_budget_bytes=100,
        estimated_peak_bytes=101,
        resources=resources,
    )
    assert not plan.feasible
    assert plan.reason == "estimated_peak_exceeds_effective_budget"


def test_parallel_plan_never_oversubscribes_shared_cpu_budget():
    resources = RuntimeResources(12, None, None)
    p = plan_parallelism(
        requested_threads="auto", resources=resources,
        outer_tasks=8, prefer_outer=True,
    )
    assert p.outer_workers <= 8
    assert p.outer_workers * p.inner_threads <= 12
    assert p.mode == "outer"


def test_parallel_plan_preserves_existing_single_kernel_auto_semantics():
    resources = RuntimeResources(6, None, None)
    p = plan_parallelism(requested_threads="auto", resources=resources)
    assert p.outer_workers == 1
    assert p.inner_threads == 6
    assert p.mode == "inner"


def test_composite_execution_plan_is_explainable_and_resource_bounded():
    resources = RuntimeResources(8, 2_000, 500)
    exact = PlanCertificate.exact_yes("dense")
    plan = plan_execution(
        requested_budget_bytes=1_200, requested_threads=6, resources=resources,
        estimated_peak_bytes=700,
        representation_candidates=(RepresentationCandidate("dense", exact, 700),),
        expected_passes=3, reuse_count=3, notes=("test",),
    )
    assert plan.memory.effective_budget_bytes == 1_200
    assert plan.memory.feasible
    assert plan.parallel.inner_threads == 6
    assert plan.representation.representation == "dense"
    assert plan.reuse_count == 3
    assert plan.as_dict()["notes"] == ("test",)
