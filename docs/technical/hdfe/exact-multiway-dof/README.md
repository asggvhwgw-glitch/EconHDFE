# Exact multiway HDFE structural DoF technical document

> **Document status:** Current EconHDFE project technical documentation.


This directory is the repository copy of the project technical document **Exact Structural Degrees of Freedom for Multiway High-Dimensional Categorical Fixed Effects**.

Files:

- `exact_multiway_hdfe_dof.tex`: canonical LaTeX source.
- `exact_multiway_hdfe_dof.pdf`: compiled technical document distributed with source/release artifacts for auditability.

Implementation mapping:

- structural rank engine: `econhdfe/hdfe/rank.py`;
- absorbed-DoF integration: `econhdfe/hdfe/dof.py`;
- exact-rank benchmark evidence: `benchmarks/hdfe/exact_rank.json`.

Artifact policy: the technical document is included in the source archive and release bundle but intentionally excluded from the wheel. `pip install econhdfe` therefore remains a runtime-code installation rather than a documentation distribution.
