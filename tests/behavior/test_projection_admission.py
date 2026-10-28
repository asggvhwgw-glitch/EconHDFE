"""Projection reservations precede construction and retain output lifetimes."""
from contextlib import contextmanager

import numpy as np
import pytest

from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.errors import SpecificationError
from econhdfe.hdfe.block_projection import BlockWeightedFEProjector
from econhdfe.hdfe.plan import FEPlan
from econhdfe.hdfe import weighted_projection
from econhdfe.hdfe.weighted_projection import WeightedFEProjector
from econhdfe.planner import scheduler


@contextmanager
def deny(nbytes, label=""):
    raise SpecificationError("test budget exhausted", code="resource.memory", stage="execution")
    yield


@pytest.mark.parametrize("engine,groups", [
    ("optimized", [np.arange(12) % 3, np.arange(12) % 4]),
    ("reference", [np.arange(12) % 3]),
])
def test_build_admission_precedes_index_construction(monkeypatch, engine, groups):
    projector = WeightedFEProjector(groups, engine=engine)
    monkeypatch.setattr(scheduler, "reserve_memory", deny)
    monkeypatch.setattr(weighted_projection, "build_absorber",
                        lambda *a, **k: pytest.fail("build preceded admission"))
    with pytest.raises(SpecificationError) as err:
        projector.prepare(np.ones(12), tol=1e-8)
    assert err.value.code == "resource.memory"
    assert not projector.compiled


def layout():
    blocks = tuple(DenseDesignBlock(i, np.arange(i * 4, (i + 1) * 4),
                                   np.array([i]), np.arange(4.).reshape(4, 1))
                   for i in range(2))
    design = BlockDesign._from_trusted(8, 2, blocks)
    return design, BlockWeightedFEProjector.from_design(
        design, FEPlan.from_arrays([]), engine="reference",
    )


@pytest.mark.parametrize("with_response", [False, True])
def test_block_admission_precedes_output_allocation(monkeypatch, with_response):
    design, projector = layout()
    response, weights = np.arange(8.), np.ones(8)
    monkeypatch.setattr(scheduler, "reserve_memory", deny)
    monkeypatch.setattr(np, "empty", lambda *a, **k: pytest.fail("empty preceded admission"))
    monkeypatch.setattr(np, "array", lambda *a, **k: pytest.fail("array preceded admission"))
    with pytest.raises(SpecificationError) as err:
        if with_response:
            projector.residualize_response_design(response, design, weights, tol=1e-8)
        else:
            projector.residualize_design(design, weights, tol=1e-8)
    assert err.value.code == "resource.memory"


def test_block_reservation_covers_accumulated_output_until_return(monkeypatch):
    design, projector = layout()
    response, weights = np.arange(8.), np.ones(8)
    active, observed = [], []

    @contextmanager
    def track(nbytes, label=""):
        active.append((nbytes, label))
        try:
            yield
        finally:
            active.pop()

    original = np.empty
    def allocate(*args, **kwargs):
        assert active and active[-1][1] == "block_fe_outputs"
        observed.append(active[-1][0])
        return original(*args, **kwargs)

    monkeypatch.setattr(scheduler, "reserve_memory", track)
    monkeypatch.setattr(np, "empty", allocate)
    result = projector.residualize_response_design(response, design, weights, tol=1e-8)
    assert observed and min(observed) >= 8 * 8 + 8 * 2 * 8
    assert not active
    np.testing.assert_array_equal(result.response, response)
    for before, after in zip(design.blocks, result.design.blocks):
        np.testing.assert_array_equal(before.values, after.values)


@pytest.mark.parametrize("engine", ["reference", "optimized"])
def test_weighted_projection_numeric_result_with_real_reservations(engine):
    rng = np.random.default_rng(17)
    groups = [np.arange(60) % 5, np.arange(60) % 3]
    weights = rng.uniform(0.5, 2, 60)
    data = rng.normal(size=(60, 3))
    projector = WeightedFEProjector(groups, engine=engine, absorb_threads=1)
    got = projector.residualize(data, weights, tol=1e-10)
    dummy = np.column_stack([np.eye(int(g.max()) + 1)[g] for g in groups])
    coef = np.linalg.lstsq(dummy * np.sqrt(weights)[:, None],
                           data * np.sqrt(weights)[:, None], rcond=None)[0]
    np.testing.assert_allclose(got, data - dummy @ coef, atol=1e-8)


def test_weighted_pool_shrinks_to_unreserved_memory(monkeypatch):
    from types import SimpleNamespace

    groups = [np.arange(60) % 5]
    weights = np.ones(60)
    data = np.random.default_rng(5).normal(size=(60, 4))
    projector = WeightedFEProjector(groups, engine="reference", absorb_threads=1)
    absorber = projector.prepare(weights, tol=1e-10)
    _, per_column = absorber.workspace_cost()
    original = absorber.residualize
    widths = []

    def capture(matrix, **kwargs):
        widths.append(matrix.shape[1])
        return original(matrix, **kwargs)

    @contextmanager
    def reserve(nbytes, label=""):
        yield

    monkeypatch.setattr(scheduler, "reserve_memory", reserve)
    monkeypatch.setattr(scheduler, "current_allocation", lambda: SimpleNamespace(
        auto_threads=False,
        unreserved_memory_bytes=lambda: per_column + 1,
    ))
    monkeypatch.setattr(absorber, "residualize", capture)
    got = projector.residualize(data, weights, tol=1e-10)
    assert widths == [1, 1, 1, 1]
    expected = data.copy()
    for group in range(5):
        rows = groups[0] == group
        expected[rows] -= expected[rows].mean(axis=0)
    np.testing.assert_allclose(got, expected, atol=1e-9)
