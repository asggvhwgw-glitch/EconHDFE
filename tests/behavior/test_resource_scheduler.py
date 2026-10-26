"""Whole-call admission and shared-budget invariants, without large allocations."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
import pytest
from econhdfe import ExecutionConfig
from econhdfe.errors import SpecificationError
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources
from econhdfe.planner.core import plan_memory, plan_parallelism

MIB = 1024**2

@pytest.fixture
def host(monkeypatch):
    state = {"rss": 100*MIB, "resources": RuntimeResources(8, 8192*MIB, 2048*MIB)}
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: state["resources"])
    monkeypatch.setattr(scheduler, "_rss", lambda: state["rss"])
    return state


def test_auto_uses_safe_headroom_and_shared_remaining(host):
    with scheduler.execution_scope() as allocation:
        assert allocation.cpu_threads == 8
        assert allocation.memory_budget_bytes == 8192*MIB-2048*MIB-int(.12*8192*MIB)
        host["rss"] += 400*MIB
        expected = allocation.memory_budget_bytes - 400*MIB
        assert allocation.remaining_memory_bytes() == expected
        plan = plan_memory(requested_budget_bytes=20_000*MIB)
        assert plan.effective_budget_bytes <= expected
        with scheduler.execution_scope(ExecutionConfig(threads=16)) as nested:
            assert nested.cpu_threads == 8
            assert nested.remaining_memory_bytes() <= expected
    assert scheduler.current_allocation() is None


def test_budget_and_threads_are_divided_not_duplicated(host):
    with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)) as parent:
        child = parent.child(4)
        assert child.cpu_threads == 1 and child.memory_budget_bytes == 32*MIB
        with scheduler.worker_scope(child):
            plan = plan_parallelism(requested_threads=100, resources=host["resources"])
            assert plan.inner_threads == 1
            with scheduler.execution_scope() as nested:
                assert nested.memory_budget_bytes == 32*MIB
        assert scheduler.current_allocation() is parent


def test_pressure_rejects_before_body_and_restores_context(host):
    host["resources"] = RuntimeResources(8, 1024*MIB, 1000*MIB)
    @scheduler.managed_execution
    def forbidden():
        pytest.fail("entered estimator under exhausted headroom")
    with pytest.raises(SpecificationError, match="headroom"):
        forbidden()
    assert scheduler.current_allocation() is None


def test_unknown_memory_requires_explicit_budget(host):
    host["resources"] = RuntimeResources(2, None, None)
    with pytest.raises(SpecificationError, match="counters"):
        scheduler.resolve_allocation()
    assert scheduler.resolve_allocation(ExecutionConfig(memory_budget_mb=64)).memory_budget_bytes == 64*MIB


def test_exception_restores_process_limits_and_model_policy(host, monkeypatch):
    events = []
    class Limits:
        def __init__(self, **kwargs): events.append(kwargs["limits"])
        def __enter__(self): return self
        def __exit__(self, *args): events.append("restore")
    monkeypatch.setattr(scheduler, "threadpool_limits", Limits)
    class Model:
        execution = ExecutionConfig()
        @scheduler.managed_execution
        def fit(self):
            assert self.execution.threads == 8
            raise RuntimeError("failure")
    model=Model(); original=model.execution
    with pytest.raises(RuntimeError): model.fit()
    assert model.execution is original
    assert events == [8, "restore"] and scheduler.current_allocation() is None


def test_native_blas_compatibility_and_restore(host):
    import sys
    import scipy.linalg as la
    from threadpoolctl import threadpool_info
    before = threadpool_info()
    with pytest.raises(RuntimeError, match="restore probe"):
        with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)):
            for pool in threadpool_info():
                if pool['internal_api'] == 'openblas':
                    expected = 1 if sys.platform == 'linux' and pool['version'] == '0.3.28' else 4
                    assert pool['num_threads'] == expected
            for n in (1, 2, 20):
                np.testing.assert_allclose(la.solve_triangular(np.eye(n)*2, np.ones((n, 3))), .5)
            a = np.random.default_rng(927).normal(size=(100, 8))
            q, r = la.qr(a, mode='economic')
            np.testing.assert_allclose(q @ r, a, rtol=1e-12, atol=1e-12)
            raise RuntimeError("restore probe")
    assert threadpool_info() == before


def test_weighted_projection_honors_remaining_memory(host, monkeypatch):
    from econhdfe.hdfe.weighted_projection import WeightedFEProjector
    from econhdfe.compute import execution
    monkeypatch.setattr(execution, "runtime_resources", lambda: host["resources"])
    n=400
    projector=WeightedFEProjector([np.arange(n)%20],engine="replica",absorb_threads=1,
                                  projection_memory_budget_mb=32)
    rhs=np.random.default_rng(5).normal(size=(n, 4))
    with scheduler.execution_scope(ExecutionConfig(threads=1,memory_budget_mb=32)):
        projector.residualize(rhs, np.ones(n), tol=1e-8)
        host["rss"] += 32*MIB
        original = rhs.copy()
        with pytest.raises(SpecificationError) as rejected:
            projector.residualize(rhs, np.ones(n), tol=1e-8)
        assert rejected.value.code == "resource.memory"
        np.testing.assert_array_equal(rhs, original)
