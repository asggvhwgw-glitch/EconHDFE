"""Automatic inner saturation must not reduce the outer CPU envelope."""
import numpy as np
import pytest

from econhdfe import ExecutionConfig
from econhdfe.compute import execution
from econhdfe.compute.execution import _plan_workspace
from econhdfe.hdfe import projection
from econhdfe.planner import calibration, core, scheduler
from econhdfe.planner.resources import RuntimeResources


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    resources = lambda: RuntimeResources(4, 1024**3, 0)
    monkeypatch.setattr(scheduler, "runtime_resources", resources)
    monkeypatch.setattr(calibration, "runtime_resources", resources)
    monkeypatch.setattr(execution, "runtime_resources", resources)
    monkeypatch.setattr(calibration, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(projection, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(scheduler, "_rss", lambda: 0)
    monkeypatch.setenv("ECONHDFE_CACHE_DIR", str(tmp_path))
    calibration.clear_in_memory_calibration_cache()
    yield
    calibration.clear_in_memory_calibration_cache()


def _measurement(cap, selected=2):
    return calibration.ThreadCalibration(
        calibration._SCHEMA_VERSION, "fixture", calibration.runtime_thread_fingerprint(max_threads=cap),
        cap, (1, selected), (.2, .1), selected, 1.05, calibration._DEFAULT_TARGET_BYTES, 1, 1,
    )


def test_managed_auto_recovers_inner_policy_but_explicit_integer_does_not(runtime, monkeypatch):
    measured = []
    def measure(*, max_threads):
        measured.append(max_threads)
        return _measurement(max_threads)
    monkeypatch.setattr(calibration, "calibrate_threads", measure)
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=128)) as allocation:
        assert allocation.cpu_threads == 4
        assert projection.resolve_threads(allocation.config.threads, nobs=100_000) == 2
        plan = _plan_workspace(nobs=100_000, ncols=2, pool_size="auto", memory_budget_mb=128,
                               requested_threads=allocation.cpu_threads, bytes_per_column=1024)
        assert plan.threads == 2
        assert projection.resolve_threads(allocation.cpu_threads, nobs=100) == 1
        assert allocation.cpu_threads == 4
    with scheduler.execution_scope(ExecutionConfig(threads=4, memory_budget_mb=128)) as allocation:
        assert projection.resolve_threads(allocation.cpu_threads, nobs=100_000) == 4
    assert measured == [4]


def test_outer_parallelism_uses_cpu_envelope_not_bandwidth_saturation(runtime, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("outer parallelism must not be capped by a single memory kernel")
    monkeypatch.setattr(core, "auto_thread_count", forbidden)
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=128)):
        plan = core.plan_parallelism(requested_threads="auto", resources=RuntimeResources(4, None, None),
                                    outer_tasks=8, prefer_outer=True, calibrate_auto=True)
    assert plan.outer_workers == 4 and plan.inner_threads == 1


def test_factory_calibrates_before_memory_reservation(runtime, monkeypatch):
    from econhdfe.pipeline import make_standard_absorber
    from econhdfe.hdfe.weighted_projection import WeightedFEProjector
    seen = []
    def measure(*, max_threads):
        assert not scheduler.has_memory_reservations()
        seen.append(max_threads)
        return _measurement(max_threads)
    monkeypatch.setattr(calibration, "calibrate_threads", measure)
    groups = [np.arange(100_000, dtype=np.int32) % 20]
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=128)) as allocation:
        absorber = make_standard_absorber(
            groups, [None], [True], weights=None, tol=1e-8, max_iter=100,
            backend="numpy", method="map", transform="symmetric", acceleration="cg",
            preconditioner="diagonal", projection_backend="auto", absorb_threads=allocation.cpu_threads,
            memory_budget_mb=128,
        )
        assert absorber.absorb_threads == 2
        weighted = WeightedFEProjector(groups, engine="replica", absorb_threads=allocation.cpu_threads)
        assert weighted.prepare(np.ones(100_000), tol=1e-8).absorb_threads == 2
    assert seen == [4]


def test_cold_reserved_or_parallel_work_defers_probe_without_poisoning_cache(runtime, monkeypatch):
    calls = []
    def measure(*, max_threads):
        calls.append(max_threads)
        return _measurement(max_threads)
    monkeypatch.setattr(calibration, "calibrate_threads", measure)
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=128)) as allocation:
        with scheduler.reserve_memory(1024, label="test"):
            assert calibration.get_thread_calibration(max_threads=4).source == "fallback"
        with scheduler.worker_scope(allocation.child(2)):
            assert calibration.get_thread_calibration(max_threads=2).selected_threads == 1
        assert not calls and not calibration._MEMORY_CACHE
        assert calibration.get_thread_calibration(max_threads=4).selected_threads == 2
        with scheduler.reserve_memory(1024, label="test"):
            assert calibration.get_thread_calibration(max_threads=4).selected_threads == 2
    assert calls == [4]


def test_failed_probe_is_retryable_and_not_persisted(runtime, monkeypatch):
    def failed(**kwargs):
        raise MemoryError("temporary pressure")
    monkeypatch.setattr(calibration, "calibrate_threads", failed)
    assert calibration.get_thread_calibration(max_threads=4).selected_threads == 1
    assert not calibration._MEMORY_CACHE
    assert not calibration.default_calibration_cache_path().exists()
    monkeypatch.setattr(calibration, "calibrate_threads", lambda **kwargs: _measurement(4))
    assert calibration.get_thread_calibration(max_threads=4).selected_threads == 2


def test_probe_arrays_scale_to_explicit_budget(runtime, monkeypatch):
    sizes = []
    def probe(x, out, passes):
        assert scheduler.has_memory_reservations()
        sizes.append(x.nbytes + out.nbytes)
    monkeypatch.setattr(calibration, "_memory_bandwidth_probe", probe)
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=32)):
        result = calibration.calibrate_threads(max_threads=2, repeats=1, passes=1)
    assert result.target_bytes <= 16 * 1024**2
    assert max(sizes) < 32 * 1024**2
    assert not scheduler.has_memory_reservations()


def test_runtime_fingerprint_invalidates_cached_thread_choice(runtime, monkeypatch):
    fp = ["cpu-set-a"]
    monkeypatch.setattr(calibration, "runtime_thread_fingerprint", lambda **kwargs: fp[0])
    calls = []
    def measure(*, max_threads):
        calls.append(fp[0])
        return _measurement(max_threads)
    monkeypatch.setattr(calibration, "calibrate_threads", measure)
    calibration.get_thread_calibration(max_threads=4)
    fp[0] = "cpu-set-b"
    calibration.get_thread_calibration(max_threads=4)
    assert calls == ["cpu-set-a", "cpu-set-b"]


def test_small_probe_reused_in_process_but_remeasured_with_more_headroom(runtime, monkeypatch):
    from dataclasses import replace
    sizes = []
    def measure(*, max_threads):
        target = calibration._allowed_probe_target()
        sizes.append(target)
        return replace(_measurement(max_threads), target_bytes=target)
    monkeypatch.setattr(calibration, "calibrate_threads", measure)
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=32)):
        calibration.get_thread_calibration(max_threads=4)
        calibration.get_thread_calibration(max_threads=4)
        assert sizes == [16 * 1024**2]
        assert not calibration.default_calibration_cache_path().exists()
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=256)):
        calibration.get_thread_calibration(max_threads=4)
        calibration.get_thread_calibration(max_threads=4)
    assert sizes == [16 * 1024**2, 96 * 1024**2]
    assert calibration.default_calibration_cache_path().exists()


def test_probe_reservation_rechecks_stale_headroom_before_allocation(runtime, monkeypatch):
    from econhdfe.errors import SpecificationError
    monkeypatch.setattr(calibration, "_allowed_probe_target", lambda value=None: 96 * 1024**2)
    def forbidden(*args, **kwargs):
        pytest.fail("probe arrays allocated before atomic admission")
    monkeypatch.setattr(calibration.np, "empty", forbidden)
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=32)):
        with pytest.raises(SpecificationError) as raised:
            calibration.calibrate_threads(max_threads=2)
        assert raised.value.code == "resource.memory"
        assert not scheduler.has_memory_reservations()


@pytest.mark.parametrize("response", [False, True])
def test_block_prewarm_precedes_output_claim_even_for_reused_absorber(runtime, monkeypatch, response):
    from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
    from econhdfe.hdfe.block_projection import BlockWeightedFEProjector
    from econhdfe.hdfe.plan import FEPlan
    calls = []
    def measure(*, max_threads):
        assert not scheduler.has_memory_reservations()
        calls.append(max_threads)
        return _measurement(max_threads)
    monkeypatch.setattr(calibration, "calibrate_threads", measure)
    n = 100_000
    rows = np.arange(n)
    design = BlockDesign(n, 1, (DenseDesignBlock(0, rows, np.array([0]), np.ones((n, 1))),))
    plan = FEPlan.from_arrays([rows % 20])
    with scheduler.execution_scope(ExecutionConfig(threads="auto", memory_budget_mb=128)) as allocation:
        projector = BlockWeightedFEProjector.from_design(
            design, plan, engine="replica", absorb_threads=allocation.cpu_threads,
        )
        for repeat in range(2):
            if repeat:
                calibration.clear_in_memory_calibration_cache()
                calibration.default_calibration_cache_path().unlink()
            if response:
                result = projector.residualize_response_design(np.ones(n), design, np.ones(n), tol=1e-8)
                np.testing.assert_allclose(result.response, 0, atol=1e-12)
            else:
                result = projector.residualize_design(design, np.ones(n), tol=1e-8)
                np.testing.assert_allclose(result.blocks[0].values, 0, atol=1e-12)
    assert calls == [4, 4]
