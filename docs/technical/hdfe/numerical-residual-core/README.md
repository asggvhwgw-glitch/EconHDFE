# Exact multiway HDFE residual-core reduction

**Status:** EconHDFE Project Technical Documentation  
**Verification:** Core mathematical statements are machine-checked in Lean 4 under the assumptions mapped in `formal/mathematical-coverage.md`.  
**Claim boundary:** Mathematical correctness, implementation correctness, numerical behavior, and historical originality are separate claims.

This directory contains the project technical document for the arbitrary-multiway numerical residual-core reduction used by the HDFE solver.

## Artifacts

- `exact_multiway_hdfe_residual_core.tex` — canonical LaTeX source.
- `exact_multiway_hdfe_residual_core.pdf` — compiled project technical document.
- `formal/appendices/residual-core.tex` — machine-checked mathematics appendix included by the source.

## Implementation correspondence

```text
exact_multiway_hdfe_residual_core.{tex,pdf}
        -> econhdfe/hdfe/numerical_core.py
        -> econhdfe/hdfe/absorber.py
        -> residual-core numerical tests
        -> benchmarks/hdfe/solver_v044_integration.json
```

Two-way graph leaf pruning is established in the HDFE literature. The registered contribution is the arbitrary-G categorical-hypergraph projection theorem, exact recursive reconstruction, and its positive-weight validity conditions. Queue/encoding code, floating-point solver behavior, adaptive routing, threading, and buffer reuse are separate software or engineering claims.
