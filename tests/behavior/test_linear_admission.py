"""Linear FE constructors must be admitted before allocating indexes/operators."""
from types import SimpleNamespace
import numpy as np
import pytest
from econhdfe import ExecutionConfig
from econhdfe.errors import SpecificationError
from econhdfe import pipeline
from econhdfe.hdfe import factory
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources

MIB = 1024**2


@pytest.fixture
def host(monkeypatch):
    state = {'rss': 0}
    monkeypatch.setattr(scheduler, 'runtime_resources', lambda: RuntimeResources(4, 1024*MIB, 0))
    monkeypatch.setattr(scheduler, '_rss', lambda: state['rss'])
    return state


def _standard(method='map'):
    return pipeline.make_standard_absorber(
        [np.arange(100)%10, np.arange(100)%7], [None, None], [True, True],
        weights=None, tol=1e-10, max_iter=1000, backend='numpy', method=method,
        transform='symmetric', acceleration='cg', preconditioner='none',
        projection_backend='indexed', absorb_threads=1, memory_budget_mb=32,
    )


def _group(individual=True, incidence='auto'):
    prep = dict(groups=[np.arange(100)%10], slopes=[None], intercepts=[True], weights=None,
                individual_spec=SimpleNamespace(intercept=True) if individual else None,
                membership_group=np.repeat(np.arange(100), 2),
                individual_codes=np.arange(200)%13, individual_slopes=None)
    return pipeline.make_group_absorber(
        prep, method='lsmr', transform='symmetric', backend='numpy', acceleration='none',
        preconditioner='diagonal', tol=1e-10, max_iter=1000, aggregation='mean',
        incidence_backend=incidence, memory_budget_mb=32, absorb_threads=1,
    )


@pytest.mark.parametrize('kind', ['map', 'twoway', 'group', 'csr', 'matrix_free', 'auto'])
def test_rejects_before_fe_constructor(host, monkeypatch, kind):
    def forbidden(*a, **k):
        pytest.fail('constructor entered before memory admission')
    for module, name in ((factory, 'HDFEAbsorber'), (factory, 'TwoWayFEAbsorber'),
                         (pipeline, 'GroupIndividualAbsorber')):
        monkeypatch.setattr(module, name, forbidden)
    with scheduler.execution_scope(ExecutionConfig(threads=1, memory_budget_mb=32)):
        host['rss'] = 32*MIB-1
        with pytest.raises(SpecificationError) as error:
            if kind in ('map', 'twoway'):
                _standard(kind)
            else:
                _group(kind != 'group', kind if kind != 'group' else 'auto')
        assert error.value.code == 'resource.memory'


@pytest.mark.parametrize('method', ['map', 'twoway'])
def test_admitted_standard_projection_matches_dummy_reference(host, method):
    rng = np.random.default_rng(102)
    values = rng.normal(size=(100, 2))
    dummies = np.column_stack([np.eye(10)[np.arange(100)%10], np.eye(7)[np.arange(100)%7]])
    expected = values - dummies @ np.linalg.lstsq(dummies, values, rcond=None)[0]
    with scheduler.execution_scope(ExecutionConfig(threads=1, memory_budget_mb=32)):
        actual = _standard(method).residualize(values)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_auto_retains_matrix_free_under_tight_construction_room(host):
    with scheduler.execution_scope(ExecutionConfig(threads=1, memory_budget_mb=32)):
        # This shape fits matrix-free construction, but its CSR build exceeds 20 KB.
        host['rss'] = 32*MIB - 20_000
        result = _group(incidence='auto')
        assert result.incidence_backend == 'matrix_free'


def test_auto_respects_outer_claim_when_selecting_incidence(host):
    with scheduler.execution_scope(ExecutionConfig(threads=1, memory_budget_mb=32)):
        with scheduler.reserve_memory(32*MIB - 20_000, label="caller workspace"):
            result = _group(incidence='auto')
            assert result.incidence_backend == 'matrix_free'
