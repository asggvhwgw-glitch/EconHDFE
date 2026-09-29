# Technical documentation and innovation policy

> **Document status:** Current EconHDFE project technical documentation.


This directory contains mathematical/algorithmic documentation, implementation notes and the package-wide technical-innovation audit.

## Technical overview documents

- [`overview.md`](overview.md) — end-to-end technical contract: estimator semantics, HDFE absorption, data layer, structural compilation, execution planning, repeated workflows, fixed-effect recovery and validation boundaries.
- [`performance-architecture.md`](performance-architecture.md) — where large HDFE workloads actually spend time, and how repeated data preparation, design construction and projection work is removed before any kernel is optimized.
- [`identified-fixed-effects.md`](identified-fixed-effects.md) — fixed-effect recovery: realized-sample identification, connected components, normalization, singleton/separation diagnostics and partial-block salvage.

## Innovation source of truth

- `innovation-audit.md` — complete package audit distinguishing genuine technical innovation from established methods and engineering.
- `innovation-registry.json` — machine-readable release contract. Every item classified as `technical_innovation` must have a formal `.tex` technical document and compiled `.pdf`, plus implementation/test/evidence mappings.

The current registered innovations are:

1. exact arbitrary-G categorical HDFE structural rank / absorbed DoF;
2. exact arbitrary-G numerical residual-core reduction;
3. exact partition-refinement HDFE canonicalization and structural design reduction.

No other current feature should be described as an original econhdfe technical contribution without updating the audit, literature boundary, registry and formal technical document in the same release.

## Machine-checked mathematical verification

The three registered theorem-backed contributions now have a separate Lean 4 proof library under [`formal/`](../../formal/README.md). The canonical coverage table is [`formal/mathematical-coverage.md`](../../formal/mathematical-coverage.md), with appendix fragments for the three project technical documents under [`formal/appendices/`](../../formal/appendices/).

The verification scope is deliberately mathematical only: explicitly mapped theorem statements, assumptions, and corollaries are machine-checked. Python/Numba implementation correctness, floating-point convergence, benchmark performance, external Stata parity, and historical originality remain separate claims. See [formal verification status](formal-verification.md).

## Domain documentation

- `hdfe/` — exact rank/DoF mathematics, exact residual-core projection, solver implementation notes and exact-rank backends.
- `structural-design/` — exact partition-refinement / dependency-DAG design reduction technical document.
- `cluster-inference.md` — established CRV/WCR cluster-inference behavior and support boundaries; this is technical documentation, not an originality claim.
- [`heterogeneous-specification-optimization.md`](heterogeneous-specification-optimization.md): internal acceleration for interaction-rich and heterogeneous-coefficient empirical specifications, including model integration and dense-fallback boundaries.


## Format convention

Current project technical documents use one top-level title, an explicit document-status line, a clear scope or claim boundary, and links to validation or related material where relevant. Version-specific reviews, migration notes, checkpoints, and closeout records are historical records and live under dedicated `history/` directories rather than beside current source-of-truth documents.


## Historical technical reviews

Version-specific correctness/resource reviews are retained under [`history/`](history/) for provenance. They are not the current source of truth for package behavior. Current mathematical status is recorded in [formal verification](formal-verification.md), while historical originality remains separately qualified in the innovation registry.
