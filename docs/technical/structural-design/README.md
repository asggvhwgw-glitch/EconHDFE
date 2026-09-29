# Exact structural-design reduction

**Status:** EconHDFE Project Technical Documentation  
**Verification:** Core mathematical statements are machine-checked in Lean 4 under the assumptions mapped in `formal/mathematical-coverage.md`.  
**Claim boundary:** Mathematical correctness, implementation correctness, numerical behavior, and historical originality are separate claims.

This directory contains the project technical document for exact HDFE partition canonicalization and pre-materialization structural design reduction.

## Artifacts

- `exact_partition_refinement_hdfe_design.tex` — canonical LaTeX source.
- `exact_partition_refinement_hdfe_design.pdf` — compiled project technical document.
- `formal/appendices/structural-design.tex` — machine-checked mathematics appendix included by the source.

## Implementation correspondence

```text
exact_partition_refinement_hdfe_design.{tex,pdf}
        -> econhdfe/hdfe/structure.py
        -> econhdfe/design_structure.py
        -> structural canonicalization / omission tests
        -> benchmarks/bench_structural_collinearity.py
        -> benchmarks/bench_dependency_dag.py
```

Partition refinement and functional dependencies are established concepts. The registered contribution is the HDFE-specific exact compilation framework and its stated mathematical consequences. Planner control flow and Python implementation correctness are not formal-verification claims.
