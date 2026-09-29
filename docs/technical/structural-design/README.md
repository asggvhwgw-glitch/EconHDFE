# Exact partition-refinement structural design reduction

**Status:** current EconHDFE project technical documentation  
**Scope:** theorem-backed structural column-space reduction with WLS/FWL and finite-PPML consequences  
**Formal verification:** see [machine-checked coverage](../../../formal/mathematical-coverage.md)

This directory contains the project technical document for exact HDFE partition canonicalization and pre-materialization structural design reduction.

## Artifacts

- `exact_partition_refinement_hdfe_design.tex`: canonical LaTeX source.
- `exact_partition_refinement_hdfe_design.pdf`: compiled project technical document.

## Implementation correspondence

```text
exact_partition_refinement_hdfe_design.{tex,pdf}
        -> econhdfe/hdfe/structure.py
        -> econhdfe/design_structure.py
        -> tests/numerics/test_fe_canonicalization.py
        -> tests/behavior/test_structural_omissions.py
        -> tests/behavior/test_dependency_omissions.py
        -> benchmarks/bench_structural_collinearity.py
        -> benchmarks/bench_dependency_dag.py
```

## Claim boundary

Partition refinement, functional dependencies and ordinary dummy collinearity are established concepts. The project technical document covers their HDFE-specific exact composition, active/reference-coded reductions, common-multiplier cases, coefficient semantics and finite-PPML consequences under stated assumptions. Planner implementation correctness and historical originality are separate claims.
