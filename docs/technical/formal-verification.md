# Machine-checked mathematical verification

> **Status:** Current project technical documentation.  
> **Scope:** Machine-checked Lean coverage for the three theorem-backed mathematical lines.  
> **Boundary:** Mathematical verification only; software implementation, floating-point behavior, performance, external parity, and historical originality are separate claims.


EconHDFE keeps mathematical verification separate from software verification.

The three registered theorem-backed contributions have Lean 4 proofs for their core mathematical statements under the assumptions recorded in `formal/mathematical-coverage.md`. The proof library is rooted at `formal/README.md`; project-technical-document appendix fragments are in `formal/appendices/`.

## Verified mathematical lines

- **Exact multiway categorical FE rank / absorbed DoF:** actual matrix rank, duplicate-row reduction, component additivity, peeling rank recursion, multipartite upper bound and column reduction, proper-connectivity formula, characteristic-zero equivalence, finite-field lower bounds, exact modular acceptance, prefix-rank identities and positive-weight rank invariance.
- **Residual-core reduction:** finite positive-weight WLS existence and uniqueness, legal categorical peeling, exact reconstruction, total residual-core identity, normal-equation/loss transport, terminal-core existence and order independence, recoding and multi-RHS consequences.
- **Partition-refinement structural design:** refinement and dependency closure, full and active/reference-coded basis reductions, common multipliers, finite composition, full/within WLS equivalence, identified or consistently selected coefficients, estimable functions and attained finite-PPML predictor invariance.

## What this does not verify

The Lean work does **not** certify Python/Numba control flow, sparse data structures, modular or rational backend implementations, floating-point convergence, memory/time budgets, benchmark performance, Stata parity, covariance formulas not explicitly mapped, or historical novelty.

A mathematical theorem can therefore be machine-checked while its production implementation remains covered by ordinary tests, numerical oracles and release validation. Likewise, Lean verification establishes correctness under stated assumptions; it does not establish that a result is historically novel.

## Audit and reproduction

The library pins Lean 4.19.0 and Mathlib commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`. The dedicated workflow builds every proof module and runs `#print axioms` for every named project theorem. Proof placeholders and project-defined unproved assertions are rejected. The accepted ambient axioms are `propext`, `Classical.choice`, and `Quot.sound`.

The canonical theorem-by-theorem mapping is `formal/mathematical-coverage.md`. More detailed exact-rank mapping is retained in `formal/rank-correspondence.md` and `formal/arithmetic-correspondence.md`.

## Technical-document integration

The Lean appendix fragments target the corrected 2026-09-28 project technical document sources and are now included directly by those corrected TeX sources on `dev/lean-foundations`. A dedicated technical-document workflow recompiles the three PDFs and checks references, citations, overfull boxes, and the presence of the Lean appendix text. The historical base PDFs remain provenance only; machine-checked status belongs to the mapped mathematical statements in the exact audited source commit.


## Compiled technical-document artifacts

The corrected sources and integrated Lean appendices are compiled by the dedicated
`Technical document PDFs` workflow. The final closeout build produced:

- exact multiway rank/DoF: 17 pages;
- residual-core reduction: 8 pages;
- structural-design reduction: 9 pages.

The build rejects undefined references/citations and overfull boxes, and confirms
that each PDF contains its Lean formalization appendix. The PDFs are committed at
their canonical project technical-document paths in `docs/technical/`; the workflow artifact is
an additional reproducibility bundle rather than the sole copy.
