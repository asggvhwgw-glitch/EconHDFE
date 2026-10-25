"""Disposable nonlinear caches are bounded without invalidating active values."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, PPMLHDFE, PPMLConfig, IVPPMLHDFE, IVPPMLConfig
from econhdfe.compute.context import ExecutionContext
from econhdfe.planner.resources import RuntimeResources

MIB = 1024**2


@pytest.fixture
def capacity(monkeypatch):
    import econhdfe.compute.context as context_module
    machine = [RuntimeResources(4, 1024*MIB, 0)]
    monkeypatch.setattr(context_module, 'runtime_resources', lambda: machine[0])
    return machine


def test_owner_storage_deduplicated_and_inputs_excluded(capacity):
    ctx = ExecutionContext(ExecutionConfig(memory_budget_mb=32))
    source = np.zeros(100)
    shared = np.zeros(200)
    ctx.prepare_admission(ctx.config, protected=(source,))
    ctx.put('a', [source, shared[:100]])
    ctx.put('b', shared[100:])
    assert ctx.retained_bytes == shared.nbytes


def test_lru_capacity_evicts_oldest_but_active_local_survives(capacity):
    ctx = ExecutionContext(ExecutionConfig(memory_budget_mb=32))
    a, b, c = [np.zeros(3*MIB//8) for _ in range(3)]
    ctx.put('a', a)
    ctx.put('b', b)
    active = ctx.get('a')
    ctx.put('c', c)
    assert list(ctx.cache) == ['a', 'c']
    assert ctx.retained_bytes == 6*MIB
    assert active is a


def test_lazy_cache_growth_is_trimmed_at_fit_exit(capacity):
    ctx = ExecutionContext(ExecutionConfig(memory_budget_mb=32))
    active = {}
    with ctx.maintain(ctx.config):
        ctx.put('lazy', active)
        active['compiled'] = np.zeros(9*MIB//8)
    assert not ctx.cache
    assert active['compiled'].nbytes == 9*MIB


def test_off_and_tight_headroom_release_cache_references(capacity):
    ctx = ExecutionContext(ExecutionConfig(memory_budget_mb=32))
    active = np.zeros(100)
    ctx.put('a', active)
    capacity[0] = RuntimeResources(4, 1024*MIB, 1000*MIB)
    ctx.prepare_admission(ctx.config)
    assert not ctx.cache
    capacity[0] = RuntimeResources(4, 1024*MIB, 0)
    ctx.put('a', active)
    ctx.config = replace(ctx.config, cache='off')
    assert ctx.get('a') is None and not ctx.cache
    assert active.size == 100


@pytest.mark.parametrize('kind', ['ppml', 'ivppml'])
def test_model_off_clears_before_root_admission_and_policy_changes_drop_old_keys(kind, capacity, monkeypatch):
    from econhdfe.planner import scheduler
    rng = np.random.default_rng(343)
    n = 360
    z = rng.normal(size=n)
    x = z + rng.normal(scale=.2, size=n)
    data = pd.DataFrame(dict(y=rng.poisson(np.exp(.3+.2*x)), x=x, z=z, g=np.arange(n)%12))
    policy = ExecutionConfig(threads=1, memory_budget_mb=32)
    if kind == 'ppml':
        model = PPMLHDFE(data, absorb=('g',), config=PPMLConfig(separation=()), execution_config=policy)
        fit = lambda: model.fit('y', ('x',))
    else:
        model = IVPPMLHDFE(data, absorb=('g',), config=IVPPMLConfig(separation=()), execution_config=policy)
        fit = lambda: model.fit('y', endog=('x',), instruments=('z',))
    first = fit()
    old = [value for key, value in model.context.cache.items() if key.startswith('ppml.projector:')]
    assert old
    model.execution_config = replace(policy, memory_budget_mb=64)
    second = fit()
    assert all(all(value is not prior for prior in old) for value in model.context.cache.values())
    np.testing.assert_allclose(second.coef, first.coef, atol=1e-9)
    model.execution_config = replace(policy, cache='off')
    original = scheduler.resolve_allocation
    def check(config=None):
        assert not model.context.cache
        return original(config)
    monkeypatch.setattr(scheduler, 'resolve_allocation', check)
    third = fit()
    assert not model.context.cache
    np.testing.assert_allclose(third.coef, first.coef, atol=1e-9)


def test_capacity_shrinks_with_active_unreserved_memory(capacity, monkeypatch):
    from types import SimpleNamespace
    import econhdfe.compute.context as context_module
    ctx = ExecutionContext(ExecutionConfig(memory_budget_mb=32))
    ctx.put('older', np.zeros(3*MIB//8))
    ctx.put('newer', np.zeros(3*MIB//8))
    monkeypatch.setattr(context_module, 'current_allocation', lambda: SimpleNamespace(
        unreserved_memory_bytes=lambda: 16*MIB,
    ))
    ctx.trim()
    assert list(ctx.cache) == ['newer']
    assert ctx.retained_bytes == 3*MIB
    assert ctx._capacity_bytes() == 4*MIB


def test_sub_admission_headroom_clears_even_small_cache(capacity):
    ctx = ExecutionContext(ExecutionConfig(memory_budget_mb=128))
    active = np.zeros(MIB//8)
    ctx.put('small', active)
    limit = 1024*MIB
    capacity[0] = RuntimeResources(4, limit, limit-int(.12*limit)-20*MIB)
    ctx.prepare_admission(ctx.config)
    assert ctx._capacity_bytes() == 0
    assert not ctx.cache
    assert active.nbytes == MIB
