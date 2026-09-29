# Exact structural-design reduction

This directory contains the EconHDFE project technical documentation for exact HDFE partition canonicalization and pre-materialization structural design reduction.

## Artifacts

- `exact_partition_refinement_hdfe_design.tex`: canonical LaTeX source.
- `exact_partition_refinement_hdfe_design.pdf`: compiled project technical document.
- `../../../formal/appendices/structural-design.tex`: machine-checked mathematical verification appendix integrated by the LaTeX source.

## Implementation and evidence

```text
exact_partition_refinement_hdfe_design.{tex,pdf}
        -> econhdfe/hdfe/structure.py
        -> econhdfe/design_structure.py
        -> tests/behavior/
        -> tests/numerics/
        -> benchmarks/bench_structural_collinearity.py
        -> benchmarks/bench_dependency_dag.py
```

The theorem-by-theorem Lean mapping is maintained in `formal/mathematical-coverage.md`.

## Scope

Partition refinement and functional dependencies are established concepts. The project technical contribution documented here is their exact HDFE-specific composition for FE canonicalization and pre-materialization design reduction. Machine-checked status applies to the mapped mathematics, not to planner control flow, floating-point execution, unrelated IV/GMM/covariance results, or historical originality.
