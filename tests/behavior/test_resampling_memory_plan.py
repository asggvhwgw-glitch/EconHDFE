"""Small-fixture checks for workload-aware bootstrap admission."""
import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, olshdfe, wild_bootstrap, wild_cluster_test_ols
from econhdfe.errors import SpecificationError
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources
from econhdfe.resampling import engine


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: RuntimeResources(
        8, 2048 * 1024**2, 0,
    ))
    monkeypatch.setattr(scheduler, "_rss", lambda: 0)
    monkeypatch.setattr(engine, "effective_n_jobs", lambda n: 8 if n == -1 else n)


@pytest.mark.parametrize("estimate,expected", [(50, 2), (80, 1)])
def test_estimated_peak_limits_parallel_workers(runtime, monkeypatch, estimate, expected):
    observed = []
    original = engine.Parallel
    def parallel(*args, **kwargs):
        observed.append(kwargs["n_jobs"])
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "Parallel", parallel)
    batch = engine.run_replicates(
        lambda seed: scheduler.current_allocation().memory_budget_bytes,
        list(range(4)), n_jobs=-1, estimated_worker_bytes=estimate * 1024**2,
        execution_config=ExecutionConfig(threads=8, memory_budget_mb=128),
    )
    assert batch.values == [128 * 1024**2 // expected] * 4
    assert observed == ([expected] if expected > 1 else [])


def test_unfit_worker_rejected_before_execution(runtime):
    def forbidden(seed):
        pytest.fail("worker executed before memory admission")
    with pytest.raises(SpecificationError) as raised:
        engine.run_replicates(
            forbidden, [0], estimated_worker_bytes=33 * 1024**2,
            execution_config=ExecutionConfig(threads=2, memory_budget_mb=32),
        )
    assert raised.value.code == "resource.memory"
    assert scheduler.current_allocation() is None


def test_resource_failure_is_never_a_skipped_replicate(runtime):
    def fail(seed):
        raise SpecificationError("budget exhausted", code="resource.memory")
    with pytest.raises(SpecificationError, match="budget exhausted"):
        engine.run_replicates(
            fail, [0, 1], n_jobs=2, estimated_worker_bytes=1024,
            execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
        )
    assert scheduler.current_allocation() is None
    # Reservation and worker scopes must be released after the exception.
    assert engine.run_replicates(lambda seed: seed, [1]).values == [1]


def test_real_workspace_memory_failure_is_not_skipped(runtime):
    from econhdfe.compute.execution import _plan_workspace
    def worker(seed):
        return _plan_workspace(
            nobs=100, ncols=1, pool_size="auto", memory_budget_mb=32,
            requested_threads=1, bytes_per_column=64 * 1024**2,
        )
    with pytest.raises(SpecificationError) as raised:
        engine.run_replicates(
            worker, [0], execution_config=ExecutionConfig(threads=1, memory_budget_mb=64),
        )
    assert raised.value.code == "resource.memory"
    assert scheduler.current_allocation() is None


def test_unknown_estimator_does_not_claim_peak_reservation(runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unknown arbitrary estimator was assigned a claimed peak")
    monkeypatch.setattr(engine, "reserve_memory", forbidden)
    assert engine.run_replicates(lambda seed: seed, [1]).values == [1]


@pytest.fixture
def fitted(runtime):
    rng = np.random.default_rng(174)
    x = rng.normal(size=120)
    df = pd.DataFrame(dict(x=x, y=.3*x + rng.normal(size=120), g=np.arange(120) % 12))
    return olshdfe(
        df, y="y", x=["x"], absorb=["g"], cluster="g", keep_state=True,
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
    )


@pytest.mark.parametrize("studentized", [False, True])
def test_wild_large_batch_rejected_before_matrix_setup(fitted, monkeypatch, studentized):
    def forbidden(*args, **kwargs):
        pytest.fail("matrix setup ran before oversized batch admission")
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    function = wild_cluster_test_ols if studentized else wild_bootstrap
    with pytest.raises(SpecificationError) as raised:
        function(
            fitted, reps=100_000, batch_size=100_000, n_jobs=8,
            weight_distribution="normal",
            execution_config=ExecutionConfig(threads=8, memory_budget_mb=32),
        )
    assert raised.value.code == "resource.memory"


def test_memory_limited_workers_preserve_draws(fitted):
    kwargs = dict(reps=23, batch_size=5, seed=77)
    serial = wild_bootstrap(fitted, n_jobs=1, **kwargs,
                            execution_config=ExecutionConfig(threads=2, memory_budget_mb=32))
    parallel = wild_bootstrap(fitted, n_jobs=8, **kwargs,
                              execution_config=ExecutionConfig(threads=8, memory_budget_mb=128))
    np.testing.assert_allclose(serial, parallel, atol=1e-12)


def test_coefficient_unfit_design_and_batches_fail_before_setup(runtime, monkeypatch):
    from econhdfe.hdfe.absorber import HDFEAbsorber
    from econhdfe.models.ols_bootstrap import wild_cluster_bootstrap_ols
    n, k = 150_000, 24
    absorber = HDFEAbsorber([np.arange(n, dtype=np.int32) % 200], absorb_threads=1)
    def forbidden(*args, **kwargs):
        pytest.fail("wide design matrix setup ran before memory admission")
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    with pytest.raises(SpecificationError) as raised:
        wild_cluster_bootstrap_ols(
            absorber=absorber, y_within=np.zeros(n), X_within=np.zeros((n, k)),
            beta=np.zeros(k), residuals=np.ones(n), clusters=np.arange(n) % 10,
            reps=32, batch_size=1, n_jobs=1,
            execution_config=ExecutionConfig(threads=1, memory_budget_mb=32),
        )
    assert raised.value.code == "resource.memory"


def test_coefficient_wide_design_uses_feasible_legacy_batches(runtime):
    from econhdfe.hdfe.absorber import HDFEAbsorber
    from econhdfe.models.ols_bootstrap import wild_cluster_bootstrap_ols
    rng = np.random.default_rng(219)
    n, k = 25_000, 24
    absorber = HDFEAbsorber([np.arange(n, dtype=np.int32) % 200], absorb_threads=1)
    kwargs = dict(absorber=absorber, y_within=np.zeros(n), X_within=rng.normal(size=(n, k)),
                  beta=rng.normal(size=k), residuals=rng.normal(size=n),
                  clusters=np.arange(n) % 17, reps=32, batch_size=1, n_jobs=1)
    low = wild_cluster_bootstrap_ols(
        **kwargs, execution_config=ExecutionConfig(threads=1, memory_budget_mb=32),
    )
    high = wild_cluster_bootstrap_ols(
        **kwargs, execution_config=ExecutionConfig(threads=1, memory_budget_mb=128),
    )
    np.testing.assert_allclose(low, high, rtol=1e-10, atol=1e-12)
