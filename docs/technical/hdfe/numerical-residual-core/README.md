# Exact multiway HDFE residual-core technical document

> **Document status:** Current EconHDFE project technical documentation.


This directory contains the project technical document for the arbitrary-multiway numerical residual-core reduction used by the HDFE solver.

- `exact_multiway_hdfe_residual_core.tex`: canonical LaTeX source.
- `exact_multiway_hdfe_residual_core.pdf`: compiled technical document.

Implementation correspondence:

```text
exact_multiway_hdfe_residual_core.{tex,pdf}
        -> econhdfe/hdfe/numerical_core.py
        -> econhdfe/hdfe/absorber.py
        -> tests/test_hdfe_multiway_solver_opt.py
        -> tests/test_hdfe_complex_fe_corpus.py
        -> benchmarks/hdfe/solver_v044_integration.json
```

Claim boundary: two-way graph leaf pruning is established in the HDFE literature. The technical contribution documented here is the exact arbitrary-G hypergraph projection reduction, its positive-weight theorem, and the residual-core reconstruction rule. Adaptive solver routing, fused kernels, threading and buffer reuse are engineering optimizations rather than separate technical innovations.
