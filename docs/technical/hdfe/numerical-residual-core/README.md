# Exact multiway HDFE residual-core reduction

This directory contains the EconHDFE project technical documentation for the arbitrary-multiway numerical residual-core reduction used by the HDFE solver.

## Artifacts

- `exact_multiway_hdfe_residual_core.tex`: canonical LaTeX source.
- `exact_multiway_hdfe_residual_core.pdf`: compiled project technical document.
- `../../../../formal/appendices/residual-core.tex`: machine-checked mathematical verification appendix integrated by the LaTeX source.

## Implementation and evidence

```text
exact_multiway_hdfe_residual_core.{tex,pdf}
        -> econhdfe/hdfe/numerical_core.py
        -> econhdfe/hdfe/absorber.py
        -> tests/numerics/
        -> benchmarks/hdfe/solver_v044_integration.json
```

The theorem-by-theorem Lean mapping is maintained in `formal/mathematical-coverage.md`.

## Scope

Two-way graph leaf pruning is established in the HDFE literature. The project technical contribution documented here is the exact arbitrary-G categorical-hypergraph projection reduction, its positive-weight conditions, and exact residual reconstruction. Machine-checked status applies to the mapped mathematics, not to queue/encoding code, floating-point convergence, solver routing, threading, buffer reuse, or historical originality.
