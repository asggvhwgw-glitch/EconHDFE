# Exact multiway HDFE structural DoF

**Status:** current EconHDFE project technical documentation  
**Scope:** theorem-backed mathematical framework with implementation and validation correspondence  
**Formal verification:** see [machine-checked coverage](../../../../formal/mathematical-coverage.md)

This directory contains the project technical document **Exact Structural Degrees of Freedom for Multiway High-Dimensional Categorical Fixed Effects**.

## Artifacts

- `exact_multiway_hdfe_dof.tex`: canonical LaTeX source.
- `exact_multiway_hdfe_dof.pdf`: compiled project technical document.

## Implementation correspondence

- structural rank engine: `econhdfe/hdfe/rank.py`;
- absorbed-DoF integration: `econhdfe/hdfe/dof.py`;
- exact-rank benchmark evidence: `benchmarks/hdfe/exact_rank.json`.

## Claim boundary

Machine-checked status applies to the mapped mathematical statements and assumptions. It does not by itself certify Python backend correctness, floating-point execution, external-package parity, performance claims, or historical originality.

## Distribution

The technical document is included in source/release artifacts for auditability and intentionally excluded from the runtime wheel. `pip install econhdfe` remains a runtime-code installation rather than a documentation distribution.
