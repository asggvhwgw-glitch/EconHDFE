import numpy as np
import pandas as pd

from econhdfe.design import build_design, factor, reg_interaction
from econhdfe.compute.design_plan import analyze_design_structure, analyze_structural_terms
from econhdfe.hdfe.plan import FEPlan


def _canonical_partition(labels):
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


def test_symbolic_plan_matches_dense_connectivity_for_factor_interactions():
    rng = np.random.default_rng(9021)
    years = 6
    per = 80
    n = years * per
    year = np.repeat(np.arange(years), per)
    # FE levels are year-local, so the specification has six exact components.
    firm = year * 1000 + np.tile(np.arange(per), years)
    x = rng.normal(size=n) + 0.25
    z = rng.normal(size=n) - 0.4
    df = pd.DataFrame({"year": year, "firm": firm, "x": x, "z": z})

    specs = [
        reg_interaction(factor("year", drop_base=False), "x"),
        reg_interaction(factor("year", drop_base=False), "z"),
    ]
    design = build_design(df, specs, n, structural=False)
    fe = FEPlan.from_arrays([firm, year])

    symbolic = analyze_structural_terms(design.structural_terms, groups=fe.groups)
    dense = analyze_design_structure(design.values, groups=fe.groups)

    assert symbolic.certified_block_separable
    assert dense.certified_block_separable
    assert symbolic.ncols == dense.ncols == 2 * years
    assert symbolic.block_count == dense.block_count == years
    np.testing.assert_array_equal(
        _canonical_partition(symbolic.row_blocks),
        _canonical_partition(dense.row_blocks),
    )
    np.testing.assert_array_equal(symbolic.column_nnz, dense.column_nnz)
    assert symbolic.block_dense_bytes == dense.block_dense_bytes


def test_symbolic_plan_refuses_false_split_when_fe_bridges_factor_blocks():
    rng = np.random.default_rng(9022)
    years = 4
    per = 50
    n = years * per
    year = np.repeat(np.arange(years), per)
    # Same firm ids repeat in every year, so firm FE bridges all year columns.
    firm = np.tile(np.arange(per), years)
    x = rng.normal(size=n) + 1.0
    df = pd.DataFrame({"year": year, "firm": firm, "x": x})
    design = build_design(
        df,
        [reg_interaction(factor("year", drop_base=False), "x")],
        n,
        structural=False,
    )
    fe = FEPlan.from_arrays([firm, year])

    symbolic = analyze_structural_terms(design.structural_terms, groups=fe.groups)
    dense = analyze_design_structure(design.values, groups=fe.groups)
    assert symbolic.block_count == dense.block_count == 1
    assert not symbolic.certified_block_separable


def test_symbolic_support_is_conservative_for_accidental_continuous_zeros():
    # x is exactly zero in one category. The numerical design can exploit that,
    # but symbolic support intentionally does not: it must never infer a split
    # from data-dependent numerical zeros before materialization.
    group = np.repeat(np.arange(3), 20)
    x = np.ones(len(group))
    x[group == 1] = 0.0
    df = pd.DataFrame({"g": group, "x": x})
    design = build_design(
        df,
        [reg_interaction(factor("g", drop_base=False), "x")],
        len(df), structural=False,
    )
    symbolic = analyze_structural_terms(design.structural_terms)
    dense = analyze_design_structure(design.values)

    # Symbolic nnz counts structural activation rather than actual floating
    # nonzeros, hence it is allowed to over-estimate support.
    assert np.all(symbolic.column_nnz >= dense.column_nnz)
    assert symbolic.structural_zero_fraction <= dense.structural_zero_fraction


def test_pre_materialization_block_design_matches_dense_values_and_metadata():
    from econhdfe.design import _build_heterogeneous_design, omit_column

    rng = np.random.default_rng(9023)
    years = 5
    per = 60
    n = years * per
    year = np.repeat(np.arange(years), per)
    firm = year * 10_000 + np.tile(np.arange(per), years)
    x = rng.normal(size=n) + 0.7
    z = rng.normal(size=n) - 0.2
    keep_rows = np.ones(n, dtype=bool)
    keep_rows[::17] = False
    df = pd.DataFrame({"year": year, "firm": firm, "x": x, "z": z})
    specs = [
        reg_interaction(factor("year", drop_base=False), "x"),
        reg_interaction(factor("year", drop_base=False), "z"),
    ]

    # Compile FE codes on exactly the same active sample as the design compiler.
    fe = FEPlan.from_arrays([firm[keep_rows], year[keep_rows]])
    dense = build_design(
        df, specs, n, row_mask=keep_rows, structural=False,
        omit=omit_column("year[4]#z"),
    )
    block = _build_heterogeneous_design(
        df, specs, n, groups=fe.groups, row_mask=keep_rows, structural=False,
        omit=omit_column("year[4]#z"),
    )

    assert block.names == dense.names
    assert block.origins == dense.origins
    assert block.requested_names == dense.requested_names
    assert block.user_omissions == dense.user_omissions
    np.testing.assert_allclose(block.values.materialize(), dense.values, rtol=0, atol=0)
    assert block.values.payload_bytes < dense.values.nbytes


def test_pre_materialization_block_compiler_does_not_call_dense_materializer(monkeypatch):
    import econhdfe.design as design_module

    rng = np.random.default_rng(9024)
    groups = 4
    per = 40
    g = np.repeat(np.arange(groups), per)
    local_fe = g * 1000 + np.tile(np.arange(per), groups)
    df = pd.DataFrame({"g": g, "x": rng.normal(size=len(g)) + 1.0})
    specs = [reg_interaction(factor("g", drop_base=False), "x")]
    fe = FEPlan.from_arrays([local_fe, g])

    def forbidden(*args, **kwargs):
        raise AssertionError("dense materializer must not be used")

    monkeypatch.setattr(design_module, "_materialize_compiled_dense", forbidden)
    out = design_module._build_heterogeneous_design(
        df, specs, len(df), groups=fe.groups, structural=False,
    )
    assert out.values.ncols == groups
    assert out.values.payload_bytes == len(df) * 8


def test_execution_design_planner_selects_block_before_materialization():
    from econhdfe.design import _compile_execution_design
    from econhdfe.compute.block_design import BlockDesign

    rng = np.random.default_rng(9025)
    G, per = 6, 70
    g = np.repeat(np.arange(G), per)
    local_fe = g * 1000 + np.tile(np.arange(per), G)
    df = pd.DataFrame({"g": g, "x": rng.normal(size=len(g)) + 0.2})
    specs = [reg_interaction(factor("g", drop_base=False), "x")]
    fe = FEPlan.from_arrays([local_fe, g])

    out = _compile_execution_design(
        df, specs, len(df), groups=fe.groups, expected_passes=4,
        memory_budget_mb=64, structural=False,
    )
    assert out.storage_plan.representation == "block_dense"
    assert isinstance(out.values, BlockDesign)
    assert out.execution_structure.certified_block_separable


def test_execution_design_planner_keeps_dense_for_single_connected_component():
    from econhdfe.design import _compile_execution_design

    rng = np.random.default_rng(9026)
    G, per = 4, 50
    g = np.repeat(np.arange(G), per)
    # Repeated FE levels bridge all factor-specific columns.
    repeated = np.tile(np.arange(per), G)
    df = pd.DataFrame({"g": g, "x": rng.normal(size=len(g)) + 0.2})
    specs = [reg_interaction(factor("g", drop_base=False), "x")]
    fe = FEPlan.from_arrays([repeated, g])

    out = _compile_execution_design(
        df, specs, len(df), groups=fe.groups, expected_passes=20,
        structural=False,
    )
    assert out.storage_plan.representation == "dense"
    assert isinstance(out.values, np.ndarray)
    assert not out.execution_structure.certified_block_separable


def test_partitioned_plan_preserves_fe_components_with_shared_global_columns():
    from econhdfe.design import _compile_structured_design, _materialize_compiled_block
    from econhdfe.compute.design_plan import analyze_structural_partition

    rng = np.random.default_rng(9027)
    G, per = 6, 55
    g = np.repeat(np.arange(G), per)
    within = np.tile(np.arange(per), G)
    fe1 = g * 1000 + within // 5
    fe2 = g * 1000 + within % 5
    df = pd.DataFrame({
        "g": g,
        "x": rng.normal(size=len(g)) + .4,
        "z": rng.normal(size=len(g)) + .7,
    })
    specs = [
        reg_interaction(factor("g", drop_base=False), "x"),
        "z",  # one globally shared coefficient
    ]
    fe = FEPlan.from_arrays([fe1, fe2])
    dense = build_design(df, specs, len(df), structural=False)
    compiled = _compile_structured_design(
        df, specs, len(df), prefix="x", row_mask=None,
        absorbed_groups=(), absorbed_names=(), structural=False,
        protected_terms=(), omit=None,
    )

    # Coefficient-connectivity analysis collapses because z bridges every FE
    # component, while row-partition analysis keeps the six exact FE pieces.
    exact = analyze_structural_terms(compiled.terms, groups=fe.groups)
    part = analyze_structural_partition(compiled.terms, groups=fe.groups)
    assert exact.block_count == 1
    assert part.block_count == G
    assert not part.columns_disjoint
    assert part.shared_columns == (G,)

    block = _materialize_compiled_block(df, compiled, part)
    assert not block.columns_disjoint
    np.testing.assert_allclose(block.materialize(), dense.values, rtol=0, atol=0)
    assert block.payload_bytes < dense.values.nbytes

    w = np.exp(rng.normal(scale=.1, size=len(g)))
    beta = rng.normal(size=dense.values.shape[1])
    v = rng.normal(size=len(g))
    np.testing.assert_allclose(block.column_moments().sum, dense.values.sum(axis=0))
    np.testing.assert_allclose(block.gram(weights=w), dense.values.T @ (dense.values * w[:, None]))
    np.testing.assert_allclose(block.matvec(beta), dense.values @ beta)
    np.testing.assert_allclose(block.t_matvec(v), dense.values.T @ v)


def test_execution_design_skips_topology_for_small_mixed_dense(monkeypatch):
    import econhdfe.compute.design_plan as design_plan_module
    from econhdfe.design import _compile_execution_design

    rng = np.random.default_rng(9061)
    G, per = 4, 90
    g = np.repeat(np.arange(G), per)
    within = np.tile(np.arange(per), G)
    fe1 = g * 10_000 + within // 9
    fe2 = g * 10_000 + within % 9
    df = pd.DataFrame({
        "g": g,
        "x": rng.normal(size=len(g)) + .2,
        "z1": rng.normal(size=len(g)),
        "z2": rng.normal(size=len(g)),
    })
    specs = [
        "z1", "z2",
        reg_interaction(factor("g", drop_base=False), "x"),
    ]
    fe = FEPlan.from_arrays([fe1, fe2])

    def forbidden(*args, **kwargs):
        raise AssertionError("small mixed dense design must skip topology analysis")

    monkeypatch.setattr(design_plan_module, "analyze_execution_structure", forbidden)
    out = _compile_execution_design(
        df, specs, len(df), groups=fe.groups, expected_passes=2,
        memory_budget_mb=64, structural=False,
    )
    assert out.storage_plan.representation == "dense"
    assert out.storage_plan.reason == "small_mixed_design_avoids_topology_setup"
    assert out.execution_structure is None
    assert isinstance(out.values, np.ndarray)
