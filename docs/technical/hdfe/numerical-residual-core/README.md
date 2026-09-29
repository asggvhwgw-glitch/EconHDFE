# Exact multiway HDFE residual-core reduction

**Status:** current EconHDFE project technical documentation  
**Scope:** theorem-backed exact residual-core projection mathematics and implementation correspondence  
**Formal verification:** see [machine-checked coverage](../../../../formal/mathematical-coverage.md)

This directory contains the project technical document for the arbitrary-multiway numerical residual-core reduction used by the HDFE solver.

## Artifacts

- `exact_multiway_hdfe_residual_core.tex`: canonical LaTeX source.
- `exact_multiway_hdfe_residual_core.pdf`: compiled project technical document.

## Implementation correspondence

```text
exact_multiway_hdfe_residual_core.{tex,pdf}
        -> econhdfe/hdfe/numerical_core.py
        -> econhdfe/hdfe/absorber.py
        -> tests/numerics/test_residual_core.py
        -> tests/numerics/test_complex_fe_corpus.py
        -> benchmarks/hdfe/solver_v044_integration.json
```

## Claim boundary

Two-way graph leaf pruning is established in the HDFE literature. The mapped mathematical contribution is the exact arbitrary-G categorical-hypergraph projection reduction, its positive-weight theorem, residual reconstruction, and stated corollaries. Queue/encoder implementation, floating-point convergence, adaptive routing, threading, fused kernels and buffer reuse remain separate engineering or validation claims.
