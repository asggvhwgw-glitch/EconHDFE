"""Known allocations must be admitted before materializing their payloads."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from econhdfe.compute import execution
from econhdfe.data import DataFrameSource, materialize_required_data
from econhdfe.data.planner import plan_ingestion
from econhdfe.design import (
    _compile_structured_design, _materialize_compiled_dense,
    _materialize_compiled_block, factor,
)
from econhdfe.compute.design_plan import analyze_execution_structure
from econhdfe.errors import SpecificationError


def deny_allocation(monkeypatch):
    monkeypatch.setattr(execution, "current_allocation", lambda: SimpleNamespace(
        remaining_memory_bytes=lambda: 1,
    ))


def test_resident_source_is_checked_before_projection(monkeypatch):
    src = DataFrameSource(pd.DataFrame({"x": np.arange(20.)}))
    deny_allocation(monkeypatch)
    monkeypatch.setattr(src, "materialize", lambda *a, **k: pytest.fail("allocated before admission"))
    with pytest.raises(SpecificationError) as err:
        materialize_required_data(src, ["x"])
    assert err.value.code == "resource.memory"


@pytest.mark.parametrize("block", [False, True])
def test_design_rejects_before_materialization(monkeypatch, block):
    data = {"g": np.repeat(np.arange(3), 4)}
    compiled = _compile_structured_design(
        data, [factor("g", drop_base=False)], 12, prefix="x", row_mask=None,
        absorbed_groups=(), absorbed_names=(), structural=False,
        protected_terms=(), omit=None,
    )
    structure = analyze_execution_structure(compiled.terms, groups=())
    assert structure.certified_block_separable
    deny_allocation(monkeypatch)
    monkeypatch.setattr(np, "zeros", lambda *a, **k: pytest.fail("allocated before admission"))
    with pytest.raises(SpecificationError) as err:
        if block:
            _materialize_compiled_block(data, compiled, structure)
        else:
            _materialize_compiled_dense(data, compiled)
    assert err.value.code == "resource.memory"


def test_ingestion_minimum_cannot_exceed_byte_budget():
    src = DataFrameSource(pd.DataFrame({"x": np.arange(20.)}))
    plan = plan_ingestion(src, ["x"], memory_budget_mb=0.001, min_batch_rows=4096)
    assert plan.batch_rows < 4096
    assert plan.batch_rows * plan.estimated_bytes_per_row <= plan.target_batch_bytes


def test_ingestion_rejects_a_row_wider_than_budget():
    src = DataFrameSource(pd.DataFrame({"x": ["x" * 1000]}))
    with pytest.raises(SpecificationError) as err:
        plan_ingestion(src, ["x"], memory_budget_mb=0.0001)
    assert err.value.code == "resource.memory"


def test_ndarray_conversion_is_admitted_but_zero_copy_view_is_not_charged(monkeypatch):
    from econhdfe.design import build_design

    deny_allocation(monkeypatch)
    existing = np.arange(20.).reshape(10, 2)
    assert build_design(None, existing, 10).values is existing
    with pytest.raises(SpecificationError) as err:
        build_design(None, existing.astype(np.int32), 10)
    assert err.value.code == "resource.memory"
