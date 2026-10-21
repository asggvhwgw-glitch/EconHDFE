"""Resource-policy regressions; small numerical fixtures, no performance tests."""

import numpy as np
import pytest
from numba import get_num_threads
from econhdfe import ExecutionConfig, olshdfe
from econhdfe.errors import SpecificationError
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.two_way import TwoWayFEAbsorber
from econhdfe.planner import RuntimeResources


def groups_and_rhs(width=5):
    rng = np.random.default_rng(20261021)
    n = 480
    groups = [np.tile(np.arange(k), n // k).astype(np.int32) for k in (16, 24, 30)]
    rng.shuffle(groups[1])
    rng.shuffle(groups[2])
    return groups, rng.normal(size=(n, width))


def test_absorber_resolves_threads_once_and_restores_on_failure(monkeypatch):
    import econhdfe.hdfe.absorber as module
    import econhdfe.hdfe.projection as projection
    groups, rhs = groups_and_rhs()
    a = HDFEAbsorber(groups, projection_backend="indexed", absorb_threads=2)
    original = module._project_indexed_inplace_current
    seen = []

    def forbidden_auto(**kwargs):
        pytest.fail("hot projection must not recalibrate/reselect explicit threads")

    def observe(*args, **kwargs):
        seen.append(get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(projection, "auto_thread_count", forbidden_auto)
    monkeypatch.setattr(module, "_project_indexed_inplace_current", observe)
    outer = get_num_threads()
    a.residualize(rhs, absorb_threads=1)
    assert seen and set(seen) == {1}
    assert a.absorb_threads == 2 and get_num_threads() == outer
    seen.clear()
    a.residualize(rhs)
    assert set(seen) == {2}

    def fail(*args, **kwargs):
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(module, "_project_indexed_inplace_current", fail)
    with pytest.raises(RuntimeError, match="kernel failure"):
        a.residualize(rhs, absorb_threads=1)
    assert get_num_threads() == outer
    monkeypatch.setattr(module, "_project_indexed_inplace_current", observe)
    a.residualize(rhs)  # Workqueue lock was released as well.


def test_two_way_public_factory_and_local_thread_override(monkeypatch):
    import econhdfe.hdfe.two_way as module
    groups, rhs = groups_and_rhs(3)
    original = module._subtract_effects_inplace
    seen = []

    def observe(*args):
        seen.append(get_num_threads())
        return original(*args)

    monkeypatch.setattr(module, "_subtract_effects_inplace", observe)
    r = olshdfe(None, y=rhs[:, 0], x=rhs[:, 1:], absorb=groups[:2],
                execution_config=ExecutionConfig(threads=2))
    assert r.absorb_info["method"] == "twoway"
    assert r.absorb_info["absorb_threads"] == 2 and seen[-1] == 2
    a = TwoWayFEAbsorber(groups[:2])._configure_execution(absorb_threads=2)
    outer = get_num_threads()
    one = a.residualize(rhs, absorb_threads=1)
    assert seen[-1] == 1 and a.absorb_threads == 2 and get_num_threads() == outer
    two = a.residualize(rhs)
    assert seen[-1] == 2
    np.testing.assert_allclose(one, two, atol=1e-12, rtol=1e-12)


@pytest.fixture
def unknown_memory(monkeypatch):
    import econhdfe.compute.execution as execution
    monkeypatch.setattr(execution, "runtime_resources", lambda: RuntimeResources(4, None, None))


def test_auto_cg_memory_inventory_and_failure_preserve_input(unknown_memory):
    from econhdfe.pipeline import _absorption_workspace_cost, _residualize_blocks
    groups, rhs = groups_and_rhs()
    a = HDFEAbsorber(groups, acceleration="auto", projection_backend="indexed", absorb_threads=2)
    fixed, per_col = _absorption_workspace_cost(a)
    assert per_col >= 4 * rhs.shape[0] * 8 and fixed >= a.index_bytes
    original = rhs.copy()
    with pytest.raises(SpecificationError):
        _residualize_blocks(a, rhs, 5, memory_budget_mb=(fixed + per_col * 4) / 1024**2)
    np.testing.assert_array_equal(rhs, original)
    _, info = _residualize_blocks(a, rhs, "auto", memory_budget_mb=(fixed + per_col * 4) / 1024**2)
    assert info.converged and a.execution_plan.pool_size == 4
    assert a.execution_memory["estimated_solver_bytes"] <= a.execution_memory["budget_bytes"]


def test_reused_scratch_is_safe_for_same_absorber_calls():
    from concurrent.futures import ThreadPoolExecutor
    groups, rhs = groups_and_rhs()
    a = HDFEAbsorber(groups, projection_backend="fused", absorb_threads=2)
    inputs = [rhs, rhs * 2 + 1]
    references = [a.residualize(x) for x in inputs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(a.residualize, inputs))
    for got, ref in zip(outputs, references):
        np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)
