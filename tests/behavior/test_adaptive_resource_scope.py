"""Automatic inner policy survives numeric CPU admission and nested calls."""
import pytest
from econhdfe import ExecutionConfig
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources


@pytest.fixture
def host(monkeypatch):
    state = {"current": 128 * 1024**2}
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: RuntimeResources(
        4, 1024 * 1024**2, state["current"]))
    monkeypatch.setattr(scheduler, "_rss", lambda: 0)
    return state


def test_auto_origin_survives_nested_and_worker_scopes(host):
    with scheduler.execution_scope(ExecutionConfig(memory_budget_mb=128)) as root:
        assert root.cpu_threads == 4 and root.auto_threads
        assert scheduler.inner_thread_request(root.config.threads) == "auto"
        with scheduler.execution_scope(root.config) as nested:
            assert nested.auto_threads
            assert scheduler.inner_thread_request(nested.cpu_threads) == "auto"
        with scheduler.worker_scope(root.child(2)) as worker:
            assert worker.cpu_threads == 2 and worker.auto_threads
            assert scheduler.inner_thread_request(worker.cpu_threads) == "auto"
        with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)):
            assert scheduler.inner_thread_request(4) == 4
    assert scheduler.inner_thread_request(4) == 4


def test_explicit_threads_are_not_reinterpreted(host):
    with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)) as root:
        assert not root.auto_threads
        assert scheduler.inner_thread_request(4) == 4
        assert not root.child(2).auto_threads


def test_reservation_signal_restores_after_failure(host):
    with scheduler.execution_scope(ExecutionConfig(memory_budget_mb=128)):
        assert not scheduler.has_memory_reservations()
        with pytest.raises(RuntimeError):
            with scheduler.reserve_memory(1024):
                assert scheduler.has_memory_reservations()
                raise RuntimeError("abort")
        assert not scheduler.has_memory_reservations()


def test_cache_reclamation_precedes_resource_admission(host):
    host["current"] = 1000 * 1024**2
    events = []
    class Model:
        execution = ExecutionConfig()
        def _prepare_resource_admission(self):
            assert scheduler.current_allocation() is None
            events.append("reclaim")
            host["current"] = 128 * 1024**2
        @scheduler.managed_execution
        def fit(self):
            events.append("fit")
            assert scheduler.current_allocation().auto_threads
    Model().fit()
    assert events == ["reclaim", "fit"]


def test_nonlinear_context_keeps_auto_policy_between_calls(host):
    from econhdfe.compute.context import ExecutionContext
    from econhdfe.models.ppml.execution import resolve_execution
    policy = ExecutionConfig(memory_budget_mb=128)
    context = ExecutionContext(policy)
    seen = []
    @scheduler.managed_execution
    def inner(execution_config=None):
        seen.append(scheduler.current_allocation().auto_threads)
    @scheduler.managed_execution
    def outer(execution_config=None, execution_context=None, fail=False):
        resolved, _ = resolve_execution(execution_config, execution_context)
        assert resolved is scheduler.current_allocation().config
        inner(execution_config=resolved)
        if fail:
            raise RuntimeError("abort")
    outer(execution_context=context)
    assert context.config is policy
    with pytest.raises(RuntimeError):
        outer(execution_context=context, fail=True)
    assert context.config is policy
    assert seen == [True, True]


def test_projector_cache_distinguishes_auto_from_explicit_cpu_cap(host):
    import numpy as np
    from econhdfe.compute.context import ExecutionContext
    from econhdfe.hdfe.plan import FEPlan
    from econhdfe.models.ppml.execution import get_projector
    plan = FEPlan.from_arrays([np.arange(100) % 10])
    context = ExecutionContext()
    with scheduler.execution_scope(ExecutionConfig(memory_budget_mb=128)) as auto:
        first = get_projector(plan, engine="replica", execution=auto.config, context=context)
        assert get_projector(plan, engine="replica", execution=auto.config, context=context) is first
    with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)) as explicit:
        second = get_projector(plan, engine="replica", execution=explicit.config, context=context)
        assert second is not first
