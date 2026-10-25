"""Atomic cooperative memory admission across nested and worker scopes."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import pytest

from econhdfe import ExecutionConfig
from econhdfe.errors import SpecificationError
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources

MIB = 1024**2


@pytest.fixture
def host(monkeypatch):
    state = {"rss": 100 * MIB, "resources": RuntimeResources(4, 2048 * MIB, 256 * MIB)}
    monkeypatch.setattr(scheduler, "_rss", lambda: state["rss"])
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: state["resources"])
    return state


def scope(mb=128):
    return scheduler.execution_scope(ExecutionConfig(threads=1, memory_budget_mb=mb))


def test_default_nested_requests_are_additive_and_failure_releases(host):
    with scope() as allocation:
        with scheduler.reserve_memory(80 * MIB, "output"):
            assert allocation.remaining_memory_bytes() == 128 * MIB
            assert allocation.unreserved_memory_bytes() == 48 * MIB
            with pytest.raises(SpecificationError) as error:
                with scheduler.reserve_memory(49 * MIB, "scratch"):
                    pytest.fail("oversized nested workspace was admitted")
            assert error.value.code == "resource.memory"
            assert error.value.details["label"] == "scratch"
            with scheduler.reserve_memory(48 * MIB):
                assert sum(allocation._ledger.claims.values()) == 128 * MIB
            assert sum(allocation._ledger.claims.values()) == 80 * MIB
        assert not allocation._ledger.claims
        with pytest.raises(RuntimeError, match="work failed"):
            with scheduler.reserve_memory(128 * MIB):
                raise RuntimeError("work failed")
        assert not allocation._ledger.claims
        assert allocation.remaining_memory_bytes() == 128 * MIB


def test_explicit_reuse_borrows_and_extends_parent_without_double_charging(host):
    with scope() as allocation:
        with scheduler.reserve_memory(80 * MIB, "peak envelope"):
            with scheduler.reserve_memory(60 * MIB, reuse_parent=True):
                assert sum(allocation._ledger.claims.values()) == 80 * MIB
            with scheduler.reserve_memory(100 * MIB, reuse_parent=True):
                assert sum(allocation._ledger.claims.values()) == 100 * MIB
                with scheduler.reserve_memory(28 * MIB):
                    assert sum(allocation._ledger.claims.values()) == 128 * MIB
            assert sum(allocation._ledger.claims.values()) == 80 * MIB
            with pytest.raises(SpecificationError):
                with scheduler.reserve_memory(129 * MIB, reuse_parent=True):
                    pass
        assert not allocation._ledger.claims


def test_two_worker_slices_can_reserve_concurrently_without_redividing(host):
    barrier = Barrier(2)
    with scope() as allocation:
        child = allocation.child(2)
        def work():
            with scheduler.worker_scope(child):
                with scheduler.reserve_memory(40 * MIB):
                    barrier.wait(timeout=5)
                    assert child.remaining_memory_bytes() == 64 * MIB
                    barrier.wait(timeout=5)
            return True
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(work) for _ in range(2)]
            assert all(job.result(timeout=10) for job in jobs)
        assert not allocation._ledger.claims


def test_workers_reject_shared_overcommit_atomically(host):
    held, release = Event(), Event()
    with scope() as allocation:
        def hold():
            with scheduler.worker_scope(allocation.child(1)):
                with scheduler.reserve_memory(80 * MIB):
                    held.set()
                    assert release.wait(5)
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(hold)
            try:
                assert held.wait(5)
                with scheduler.worker_scope(allocation.child(1)):
                    with pytest.raises(SpecificationError):
                        with scheduler.reserve_memory(49 * MIB):
                            pass
                    with scheduler.reserve_memory(48 * MIB):
                        assert sum(allocation._ledger.claims.values()) == 128 * MIB
            finally:
                release.set()
            job.result(timeout=10)
        assert not allocation._ledger.claims


def test_worker_static_slice_and_dispatcher_claim_are_both_enforced(host):
    with scope() as allocation:
        child = allocation.child(2)
        with scheduler.reserve_memory(80 * MIB):
            with scheduler.worker_scope(child):
                with pytest.raises(SpecificationError):
                    with scheduler.reserve_memory(49 * MIB, reuse_parent=True):
                        pass  # workers cannot borrow the dispatcher's envelope
                with scheduler.reserve_memory(40 * MIB):
                    with pytest.raises(SpecificationError):
                        with scheduler.reserve_memory(25 * MIB):
                            pass
        with scheduler.worker_scope(child):
            with pytest.raises(SpecificationError):
                with scheduler.reserve_memory(65 * MIB):
                    pass
        assert not allocation._ledger.claims


def test_rss_and_outstanding_claims_never_create_extra_capacity(host):
    with scope() as allocation:
        with scheduler.reserve_memory(64 * MIB):
            host["rss"] += 64 * MIB
            assert allocation.remaining_memory_bytes() == 64 * MIB
            with scheduler.worker_scope(allocation):
                assert allocation.remaining_memory_bytes() == 0
                with pytest.raises(SpecificationError):
                    with scheduler.reserve_memory(1):
                        pass
        # Releasing a claim does not release retained arrays reflected in RSS.
        assert allocation.remaining_memory_bytes() == 64 * MIB
        with scheduler.reserve_memory(64 * MIB):
            pass
        with pytest.raises(SpecificationError):
            with scheduler.reserve_memory(65 * MIB):
                pass


def test_nested_execution_uses_same_ledger_and_cannot_reuse_above_child_cap(host):
    with scope() as allocation:
        with scheduler.reserve_memory(80 * MIB):
            with scope(64) as nested:
                assert nested._ledger is allocation._ledger
                with pytest.raises(SpecificationError):
                    with scheduler.reserve_memory(65 * MIB, reuse_parent=True):
                        pass
                with scheduler.reserve_memory(64 * MIB, reuse_parent=True):
                    assert nested.remaining_memory_bytes() == 64 * MIB


def test_no_scope_checks_live_headroom_and_unknown_counters(host):
    host["resources"] = RuntimeResources(2, 100 * MIB, 50 * MIB)
    with scheduler.reserve_memory(38 * MIB):
        pass
    with pytest.raises(SpecificationError):
        with scheduler.reserve_memory(38 * MIB + 1):
            pass
    host["resources"] = RuntimeResources(2, None, None)
    with pytest.raises(SpecificationError) as error:
        with scheduler.reserve_memory(1):
            pass
    assert error.value.code == "resource.unavailable"
    with scheduler.reserve_memory(0):
        pass


@pytest.mark.parametrize("value", [-1, 1.5, "100"])
def test_invalid_byte_counts_fail_before_body(host, value):
    with pytest.raises(SpecificationError):
        with scheduler.reserve_memory(value):
            pytest.fail("invalid byte count accepted")
