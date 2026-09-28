import numpy as np
import pytest

from econhdfe.compute.block_design import BlockDesign
from econhdfe.compute.design_plan import analyze_design_structure
from econhdfe.hdfe.block_projection import BlockWeightedFEProjector
from econhdfe.hdfe.plan import FEPlan
from econhdfe.hdfe.weighted_projection import WeightedFEProjector


def _fixture(seed=1401, blocks=4, rows_per=240, width=3):
    rng = np.random.default_rng(seed)
    n = blocks * rows_per
    which = np.repeat(np.arange(blocks), rows_per)
    X = np.zeros((n, blocks * width))
    firm = np.empty(n, dtype=np.int32)
    cell = np.empty(n, dtype=np.int32)
    for b in range(blocks):
        rows = np.flatnonzero(which == b)
        X[rows, b * width:(b + 1) * width] = rng.normal(size=(len(rows), width))
        # Both FE dimensions are block-local but cross-cut each other within a block.
        firm[rows] = b * 60 + np.repeat(np.arange(60), 4)
        cell[rows] = b * 12 + np.tile(np.arange(12), 20)
    plan = FEPlan.from_arrays([firm, cell])
    structure = analyze_design_structure(X, groups=plan.groups)
    assert structure.certified_block_separable
    B = BlockDesign.from_dense(X, structure, verify=False)
    return rng, X, plan, structure, B


@pytest.mark.parametrize("engine", ["replica", "optimized"])
def test_block_weighted_fe_projection_matches_pooled_dense(engine):
    rng, X, plan, structure, B = _fixture()
    response = rng.normal(size=len(X))
    weights = np.exp(rng.normal(scale=0.25, size=len(X)))
    dense = np.column_stack([response, X])
    pooled = WeightedFEProjector(plan.groups, engine=engine, method="map")
    got_dense, _ = pooled.residualize(
        dense, weights, tol=1e-10, return_info=True, copy=True
    )

    blocked = BlockWeightedFEProjector(
        structure, plan, engine=engine, method="map", absorb_threads=1
    )
    got = blocked.residualize_response_design(response, B, weights, tol=1e-10)
    np.testing.assert_allclose(got.response, got_dense[:, 0], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(got.design.materialize(), got_dense[:, 1:], rtol=2e-8, atol=2e-9)
    assert got.iterations >= 0
    assert blocked.resource_info["block_count"] == structure.block_count
