# Exact multiway HDFE structural DoF

This directory contains the EconHDFE project technical documentation for **Exact Structural Degrees of Freedom for Multiway High-Dimensional Categorical Fixed Effects**.

## Artifacts

- `exact_multiway_hdfe_dof.tex`: canonical LaTeX source.
- `exact_multiway_hdfe_dof.pdf`: compiled project technical document.
- `../../../../formal/appendices/exact-rank.tex`: machine-checked mathematical verification appendix integrated by the LaTeX source.

## Implementation and evidence

- structural rank engine: `econhdfe/hdfe/rank.py`;
- absorbed-DoF integration: `econhdfe/hdfe/dof.py`;
- exact-rank benchmark evidence: `benchmarks/hdfe/exact_rank.json`;
- theorem-by-theorem Lean coverage: `formal/mathematical-coverage.md`.

## Scope

The document records the mathematical framework, implementation correspondence, validation boundary, and prior-art boundary. Machine-checked status applies only to the explicitly mapped mathematical statements and assumptions; it does not certify the production rank backend or establish historical originality.

The TeX/PDF pair is included in source and release documentation artifacts but intentionally excluded from the runtime wheel.
