"""Resource contracts must hold during rank filtering as well as final IRLS."""
import numpy as np
import pytest
from econhdfe import ExecutionConfig, IVPPMLConfig, PPMLConfig, ivppmlhdfe, ppmlhdfe
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml_iv.heterogeneous import fit_block_arrays


def _data(separated):
    rng = np.random.default_rng(42)
    n = 600
    group = np.arange(n) % 12
    instrument = rng.normal(size=n)
    endog = instrument + rng.normal(scale=.3, size=n)
    y = rng.poisson(np.exp(.4 + .2 * endog))
    if separated:
        y[group == 0] = 0
    return y, endog, instrument, group


@pytest.fixture
def absorber_calls(monkeypatch):
    calls = []
    original = HDFEAbsorber.__init__
    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        calls.append((self.nobs, kwargs.get('absorb_threads'), self.projection_memory_budget_mb))
    monkeypatch.setattr(HDFEAbsorber, '__init__', capture)
    return calls


@pytest.mark.parametrize('model', ['ppml', 'ivppml'])
@pytest.mark.parametrize('engine', ['replica', 'optimized'])
def test_post_separation_rank_filter_preserves_resources(model, engine, absorber_calls):
    y, endog, instrument, group = _data(True)
    kwargs = dict(absorb=[group], execution_config=ExecutionConfig(threads=1, memory_budget_mb=32))
    if model == 'ppml':
        result = ppmlhdfe(y, endog, config=PPMLConfig(engine=engine, separation=('fe',)), **kwargs)
    else:
        result = ivppmlhdfe(y, endog=endog, instruments=instrument,
                           config=IVPPMLConfig(engine=engine, separation=('fe',)), **kwargs)
    assert result.converged
    assert np.sum(result.separation_mask) == 50
    # Confirm a new absorber handles the changed sample rather than a stale cache.
    assert {n for n, _, _ in absorber_calls} == {600, 550}
    assert all(threads == 1 and memory == 32 for _, threads, memory in absorber_calls)


def _as_block(values, group):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    blocks = []
    for index in (0, 1):
        rows = np.flatnonzero(group % 2 == index)
        blocks.append(DenseDesignBlock(index, rows, np.arange(values.shape[1]), values[rows]))
    return BlockDesign(len(group), values.shape[1], tuple(blocks))


@pytest.mark.parametrize('separated', [False, True])
@pytest.mark.parametrize('engine', ['replica', 'optimized'])
def test_structured_iv_rank_filters_preserve_resources(separated, engine, absorber_calls):
    y, endog, instrument, group = _data(separated)
    n = len(y)
    result = fit_block_arrays(
        y, _as_block(np.empty((n, 0)), group), _as_block(endog, group),
        _as_block(instrument, group), FEPlan.from_arrays([group]),
        offset=np.zeros(n), true_w=np.ones(n), vce='robust',
        endog_names=('e',), instrument_names=('z',),
        config=IVPPMLConfig(engine=engine, separation=('fe',)),
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=32),
    )
    assert result.converged
    assert np.sum(result.separation_mask) == (50 if separated else 0)
    assert {n for n, _, _ in absorber_calls} == ({250, 300} if separated else {300})
    assert all(threads == 1 and memory == 32 for _, threads, memory in absorber_calls)
