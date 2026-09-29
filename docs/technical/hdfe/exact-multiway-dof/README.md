# Exact multiway HDFE structural DoF

**Status:** EconHDFE Project Technical Documentation  
**Verification:** Core mathematical statements are machine-checked in Lean 4 under the assumptions mapped in `formal/mathematical-coverage.md`.  
**Claim boundary:** Mathematical correctness, implementation correctness, numerical behavior, and historical originality are separate claims.

This directory contains the project technical document **Exact Structural Degrees of Freedom for Multiway High-Dimensional Categorical Fixed Effects**.

## Artifacts

- `exact_multiway_hdfe_dof.tex` — canonical LaTeX source.
- `exact_multiway_hdfe_dof.pdf` — compiled project technical document.
- `formal/appendices/exact-rank.tex` — machine-checked mathematics appendix included by the source.

## Implementation and evidence

- structural rank engine: `econhdfe/hdfe/rank.py`;
- absorbed-DoF integration: `econhdfe/hdfe/dof.py`;
- exact-rank benchmark evidence: `benchmarks/hdfe/exact_rank.json`.

The document concerns characteristic-zero rank of the requested categorical FE design. It does not certify numerical convergence, heterogeneous slopes, group-individual multi-membership, cluster finite-sample conventions, or the production exact-rank backend as a formally verified program.

## Distribution

The TeX/PDF technical document is included in source/release artifacts for auditability and intentionally excluded from the runtime wheel.
