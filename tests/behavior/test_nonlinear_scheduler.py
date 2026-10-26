"""Nonlinear estimators share one live resource scope across preprocessing/IRLS."""
import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, IVPPMLConfig, IVPPMLHDFE, PPMLConfig, PPMLHDFE, ivppmlhdfe, ppmlhdfe
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.models.ppml.execution import resolve_execution
from econhdfe.planner.scheduler import current_allocation


def _data():
    rng = np.random.default_rng(752)
    n = 600
    group = np.arange(n) % 12
    z = rng.normal(size=n)
    x = z + rng.normal(scale=.2, size=n)
    y = rng.poisson(np.exp(.4 + .15*x))
    y[group == 0] = 0
    return pd.DataFrame(dict(y=y, x=x, z=z, group=group))


@pytest.fixture
def phases(monkeypatch):
    import econhdfe.models.ppml.estimator as ppml
    import econhdfe.models.ppml_iv.estimator as ivppml
    seen = []

    def capture(module, name):
        original = getattr(module, name)
        def wrapped(*args, **kwargs):
            allocation = current_allocation()
            assert allocation is not None, name
            seen.append((name, allocation.cpu_threads, allocation.memory_budget_bytes))
            return original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)

    for module, irls in ((ppml, 'fit_irls'), (ivppml, 'fit_iv_irls')):
        capture(module, 'detect_separation')
        capture(module, irls)
    original = HDFEAbsorber.__init__
    def absorber(self, *args, **kwargs):
        allocation = current_allocation()
        assert allocation is not None
        assert kwargs['absorb_threads'] == allocation.cpu_threads
        assert kwargs['projection_memory_budget_mb'] * 1024**2 == allocation.memory_budget_bytes
        seen.append(('absorber', allocation.cpu_threads, allocation.memory_budget_bytes))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(HDFEAbsorber, '__init__', absorber)
    return seen


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
@pytest.mark.parametrize('automatic', [False, True])
def test_public_nonlinear_phases_share_scope(model, automatic, phases):
    data = _data()
    execution = ExecutionConfig() if automatic else ExecutionConfig(threads=1, memory_budget_mb=32)
    kwargs = dict(absorb=[data.group.to_numpy()], execution_config=execution)
    if model == 'ppml':
        result = ppmlhdfe(y=data.y, x=data.x, config=PPMLConfig(separation=('fe',)), **kwargs)
    else:
        result = ivppmlhdfe(y=data.y, endog=data.x, instruments=data.z,
                           config=IVPPMLConfig(separation=('fe',)), **kwargs)
    assert result.converged
    assert {'absorber', 'detect_separation'} <= {stage for stage, _, _ in phases}
    assert len({(threads, memory) for _, threads, memory in phases}) == 1
    assert all(isinstance(threads, int) and threads >= 1 and memory > 0 for _, threads, memory in phases)
    if not automatic:
        assert {(threads, memory) for _, threads, memory in phases} == {(1, 32*1024**2)}
    assert current_allocation() is None


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
def test_reusable_model_refreshes_resource_policy(model, phases):
    data = _data()
    if model == 'ppml':
        instance = PPMLHDFE(data, absorb=('group',), config=PPMLConfig(separation=('fe',)),
                            execution_config=ExecutionConfig(threads=1, memory_budget_mb=32))
        fit = lambda: instance.fit('y', ('x',))
    else:
        instance = IVPPMLHDFE(data, absorb=('group',), config=IVPPMLConfig(separation=('fe',)),
                              execution_config=ExecutionConfig(threads=1, memory_budget_mb=32))
        fit = lambda: instance.fit('y', endog=('x',), instruments=('z',))
    fit()
    assert {(threads, memory) for _, threads, memory in phases} == {(1, 32*1024**2)}
    phases.clear()
    changed = ExecutionConfig(threads=2, memory_budget_mb=64, cache='off')
    instance.execution_config = changed
    fit()
    assert {(threads, memory) for _, threads, memory in phases} == {(2, 64*1024**2)}
    assert instance.execution_config is changed
    assert current_allocation() is None


def test_direct_execution_resolution_turns_auto_into_numeric_policy():
    config, context = resolve_execution(ExecutionConfig())
    assert isinstance(config.threads, int)
    assert isinstance(config.memory_budget_mb, (int, float))
    assert config.memory_budget_mb > 0
    assert context.config == config
    assert current_allocation() is None


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
def test_reusable_auto_model_rechecks_available_resources(model, phases, monkeypatch):
    from types import SimpleNamespace
    import econhdfe.planner.scheduler as scheduler
    machine = SimpleNamespace(cpu_threads=4, memory_limit_bytes=1024*1024**2,
                              memory_current_bytes=128*1024**2)
    monkeypatch.setattr(scheduler, 'runtime_resources', lambda: machine)
    monkeypatch.setattr(scheduler, '_rss', lambda: 0)
    data = _data()
    policy = ExecutionConfig()
    if model == 'ppml':
        instance = PPMLHDFE(data, absorb=('group',), config=PPMLConfig(separation=('fe',)),
                            execution_config=policy)
        fit = lambda: instance.fit('y', ('x',))
    else:
        instance = IVPPMLHDFE(data, absorb=('group',), config=IVPPMLConfig(separation=('fe',)),
                              execution_config=policy)
        fit = lambda: instance.fit('y', endog=('x',), instruments=('z',))
    fit()
    first = {(threads, memory) for _, threads, memory in phases}
    phases.clear()
    machine.cpu_threads = 2
    machine.memory_current_bytes = 512*1024**2
    fit()
    second = {(threads, memory) for _, threads, memory in phases}
    assert len(first) == len(second) == 1
    old_threads, old_memory = first.pop()
    new_threads, new_memory = second.pop()
    assert (old_threads, new_threads) == (4, 2)
    assert new_memory < old_memory
    assert instance.execution_config is policy
    assert instance.context.config == policy


def test_direct_array_entry_preserves_context_policy(monkeypatch):
    from econhdfe.compute.context import ExecutionContext
    from econhdfe.hdfe.plan import FEPlan
    from econhdfe.models.ppml.estimator import fit_arrays
    from econhdfe.models.ppml import estimator
    seen = []
    original = estimator.detect_separation
    def capture(*args, **kwargs):
        allocation = current_allocation()
        seen.append(allocation.config)
        return original(*args, **kwargs)
    monkeypatch.setattr(estimator, 'detect_separation', capture)
    data = _data()
    policy = ExecutionConfig(threads=1, memory_budget_mb=64, cache='off', profile='full')
    context = ExecutionContext(policy)
    result = fit_arrays(
        data.y.to_numpy(), data[['x']].to_numpy(), FEPlan.from_arrays([data.group.to_numpy()]),
        offset=np.zeros(len(data)), true_w=np.ones(len(data)), vce='robust',
        clusters=None, names=('x',), config=PPMLConfig(separation=('fe',)),
        execution_context=context,
    )
    assert result.converged
    assert len(seen) == 1
    assert seen[0].threads == 1 and seen[0].memory_budget_mb == 64
    assert seen[0].cache == 'off' and seen[0].profile == 'full'
    assert not context.cache
    assert current_allocation() is None
