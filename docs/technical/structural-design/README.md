# Exact structural-design reduction technical document

> **Document status:** Current EconHDFE project technical documentation.


This directory contains the project technical document for the package's exact HDFE partition canonicalization and pre-materialization structural collinearity planner.

- `exact_partition_refinement_hdfe_design.tex`: canonical LaTeX source.
- `exact_partition_refinement_hdfe_design.pdf`: compiled technical document.

Implementation correspondence:

```text
exact_partition_refinement_hdfe_design.{tex,pdf}
        -> econhdfe/hdfe/structure.py
        -> econhdfe/design_structure.py
        -> tests/test_fe_canonicalization.py
        -> tests/test_v073_structural_collinearity.py
        -> tests/test_v074_dependency_omission.py
        -> benchmarks/bench_structural_collinearity.py
        -> benchmarks/bench_dependency_dag.py
```

Claim boundary: partition refinement and functional dependencies are established concepts. The technical contribution documented here is the exact HDFE-specific canonicalization and design-compilation framework, including certified interaction reductions, component dependency closure, singleton-sample recertification and proof-trace separation between requested inference topology and numerical execution topology.
