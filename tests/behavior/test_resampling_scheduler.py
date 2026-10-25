"""Shared resampling budgets and process-wide BLAS lifecycle."""
from contextlib import contextmanager
from threading import get_ident

import numpy as np
import pytest
from joblib import parallel_config

from econhdfe import ExecutionConfig
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources
from econhdfe.resampling import engine


@pytest.fixture
def stable_runtime(monkeypatch):
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: RuntimeResources(
        8, 2048 * 1024**2, 0,
    ))
    monkeypatch.setattr(scheduler, "_rss", lambda: 0)
    # Resolve joblib's host-dependent negative worker counts deterministically.
    monkeypatch.setattr(engine, "effective_n_jobs", lambda n: 8 if n == -1 else n)


@pytest.mark.parametrize("n_jobs,cpus,memory,workers", [
    (-1, 2, 128, 2), (8, 8, 64, 2), (1, 4, 128, 1),
])
def test_workers_share_parent_and_force_shared_memory(stable_runtime, monkeypatch, n_jobs, cpus, memory, workers):
    observed = []
    original = engine.Parallel
    def parallel(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "Parallel", parallel)
    def worker(seed):
        allocation = scheduler.current_allocation()
        return allocation.cpu_threads, allocation.memory_budget_bytes
    with scheduler.execution_scope(ExecutionConfig(threads=cpus, memory_budget_mb=memory)) as parent:
        with parallel_config(backend="loky"):
            result = engine.run_replicates(worker, list(range(4)), n_jobs=n_jobs)
        assert scheduler.current_allocation() is parent
    assert result.values == [(cpus // workers, memory * 1024**2 // workers)] * 4
    if workers > 1:
        assert observed[0]["n_jobs"] == workers
        assert observed[0]["require"] == "sharedmem"
    else:
        assert not observed
    assert scheduler.current_allocation() is None


def test_nested_workers_change_blas_only_once(stable_runtime, monkeypatch):
    events = []
    caller = get_ident()
    @contextmanager
    def limits(*, limits):
        events.append(("enter", get_ident(), limits))
        try:
            yield
        finally:
            events.append(("exit", get_ident(), limits))
    monkeypatch.setattr(engine, "threadpool_limits", limits)
    def inner(seed):
        allocation = scheduler.current_allocation()
        return allocation.cpu_threads, allocation.memory_budget_bytes
    def outer(seed):
        return engine.run_replicates(inner, [0, 1], n_jobs=2).values
    with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)):
        result = engine.run_replicates(outer, [0, 1], n_jobs=2)
    assert result.values == [[(1, 32 * 1024**2)] * 2] * 2
    assert events == [("enter", caller, 1), ("exit", caller, 1)]
    assert scheduler.current_allocation() is None
    assert not engine._IN_REPLICATES.get()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_worker_error_restores_scope_and_blas(stable_runtime, monkeypatch, n_jobs):
    events = []
    @contextmanager
    def limits(*, limits):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")
    monkeypatch.setattr(engine, "threadpool_limits", limits)
    def fail(seed):
        raise RuntimeError("worker failed")
    with scheduler.execution_scope(ExecutionConfig(threads=2, memory_budget_mb=128)) as parent:
        with pytest.raises(RuntimeError, match="worker failed"):
            engine.run_replicates(fail, [0, 1], n_jobs=n_jobs, skip_exceptions=())
        assert scheduler.current_allocation() is parent
    assert events == ["enter", "exit"]
    assert scheduler.current_allocation() is None
    assert not engine._IN_REPLICATES.get()


def test_pairs_bootstrap_seed_order_unchanged(stable_runtime):
    def estimator(rows):
        allocation = scheduler.current_allocation()
        assert allocation is not None
        return np.array([rows.mean(), rows.std()])
    with scheduler.execution_scope(ExecutionConfig(threads=2, memory_budget_mb=128)):
        serial = engine.parallel_pairs_bootstrap(estimator, 40, reps=9, n_jobs=1, seed=91)
        parallel = engine.parallel_pairs_bootstrap(estimator, 40, reps=9, n_jobs=-1, seed=91)
    np.testing.assert_array_equal(serial, parallel)


def test_studentized_wild_bootstrap_parallel_matches_serial(stable_runtime):
    import pandas as pd
    from econhdfe import olshdfe, wild_cluster_test_ols
    rng = np.random.default_rng(71)
    n = 240
    x = rng.normal(size=n)
    group = np.arange(n) % 12
    data = pd.DataFrame(dict(x=x, y=.4*x + group*.1 + rng.normal(size=n), group=group))
    with scheduler.execution_scope(ExecutionConfig(threads=2, memory_budget_mb=128)):
        fit = olshdfe(data, y="y", x=["x"], absorb=["group"], cluster="group", keep_state=True)
        serial = wild_cluster_test_ols(fit, reps=99, batch_size=13, n_jobs=1, seed=14)
        parallel = wild_cluster_test_ols(fit, reps=99, batch_size=13, n_jobs=2, seed=14)
    np.testing.assert_allclose(serial.bootstrap_statistics, parallel.bootstrap_statistics, atol=1e-12)
    assert serial.pvalue == parallel.pvalue


def test_standalone_pairs_bootstrap_accepts_explicit_budget(stable_runtime, monkeypatch):
    seen = []
    def no_parallel(*args, **kwargs):
        pytest.fail("32 MiB admits only one worker")
    monkeypatch.setattr(engine, "Parallel", no_parallel)
    def estimator(rows):
        allocation = scheduler.current_allocation()
        seen.append((allocation.cpu_threads, allocation.memory_budget_bytes))
        return [rows.mean()]
    result = engine.parallel_pairs_bootstrap(
        estimator, 20, reps=3, n_jobs=-1,
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=32),
    )
    assert result.shape == (3, 1)
    assert seen == [(2, 32 * 1024**2)] * 3
    assert scheduler.current_allocation() is None


def test_worker_allocation_created_once(stable_runtime, monkeypatch):
    original = scheduler.ResourceAllocation.child
    calls = []
    def child(self, workers):
        calls.append(workers)
        return original(self, workers)
    monkeypatch.setattr(scheduler.ResourceAllocation, "child", child)
    engine.run_replicates(
        lambda seed: seed, list(range(5)), n_jobs=2,
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
    )
    assert calls == [2]


def test_admitted_minimum_budget_survives_small_rss_growth(stable_runtime, monkeypatch):
    rss = [0]
    monkeypatch.setattr(scheduler, "_rss", lambda: rss[0])
    def jobs(n):
        # Simulate setup allocation after root admission but before dispatch.
        rss[0] = 1024**2
        return 2
    monkeypatch.setattr(engine, "effective_n_jobs", jobs)
    def worker(seed):
        allocation = scheduler.current_allocation()
        return allocation.memory_budget_bytes, allocation.remaining_memory_bytes()
    batch = engine.run_replicates(
        worker, [0, 1], n_jobs=2,
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=32),
    )
    assert batch.values == [(32 * 1024**2, 31 * 1024**2)] * 2
