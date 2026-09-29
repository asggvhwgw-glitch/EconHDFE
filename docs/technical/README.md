# Technical documentation and innovation policy

This directory contains mathematical/algorithmic documentation, implementation notes and the package-wide technical-innovation audit.

## Technical overview documents

- [`overview.md`](overview.md) — end-to-end technical contract: estimator semantics, HDFE absorption, data layer, structural compilation, execution planning, repeated workflows, fixed-effect recovery and validation boundaries.
- [`performance-architecture.md`](performance-architecture.md) — where large HDFE workloads actually spend time, and how repeated data preparation, design construction and projection work is removed before any kernel is optimized.
- [`identified-fixed-effects.md`](identified-fixed-effects.md) — fixed-effect recovery: realized-sample identification, connected components, normalization, singleton/separation diagnostics and partial-block salvage.

## Innovation source of truth

- `innovation-audit.md` — complete package audit distinguishing genuine technical innovation from established methods and engineering.
- `innovation-registry.json` — machine-readable release contract. Every item classified as `technical_innovation` must have a formal `.tex` manuscript and compiled `.pdf`, plus implementation/test/evidence mappings.

The current registered innovations are:

1. exact arbitrary-G categorical HDFE structural rank / absorbed DoF;
2. exact arbitrary-G numerical residual-core reduction;
3. exact partition-refinement HDFE canonicalization and structural design reduction.

No other current feature should be described as an original econhdfe technical contribution without updating the audit, literature boundary, registry and project technical document in the same release.

## Machine-checked mathematical verification

The three registered theorem-backed contributions now have a separate Lean 4 proof library under [`formal/`](../../formal/README.md). The canonical coverage table is [`formal/mathematical-coverage.md`](../../formal/mathematical-coverage.md), with manuscript-ready appendix fragments under [`formal/appendices/`](../../formal/appendices/).

The verification scope is deliberately mathematical only: explicitly mapped theorem statements, assumptions, and corollaries are machine-checked. Python/Numba implementation correctness, floating-point convergence, benchmark performance, external Stata parity, and historical originality remain separate claims. See [formal verification status](formal-verification.md).

## Domain documentation

- `hdfe/` — exact rank/DoF mathematics, exact residual-core projection, solver implementation notes and exact-rank backends.
- `structural-design/` — exact partition-refinement / dependency-DAG design reduction manuscript.
- `cluster-inference.md` — established CRV/WCR cluster-inference behavior and support boundaries; this is technical documentation, not an originality claim.
- [`heterogeneous-specification-optimization.md`](heterogeneous-specification-optimization.md): internal acceleration for interaction-rich and heterogeneous-coefficient empirical specifications, including model integration and dense-fallback boundaries.


## Historical technical reviews

Version-bound mathematical and nonlinear/resource reviews are retained under [`history/`](history/) for provenance. They are not current sources of truth for package behavior or theorem coverage; use the active technical documents and [formal verification status](formal-verification.md).


## Documentation format

Current technical Markdown documents use a lightweight common header: **Status**, **Scope**, and **Boundary**. Theorem-backed LaTeX documents identify the author as **EconHDFE Project** and the document class as **Project Technical Documentation**. Version-bound reviews and obsolete checkpoints belong under a `history/` directory rather than beside current sources of truth.
