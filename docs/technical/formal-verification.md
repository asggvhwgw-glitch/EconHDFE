# Machine-checked mathematical verification

EconHDFE keeps mathematical verification separate from software verification.

The three registered theorem-backed contributions have Lean 4 proofs for their core mathematical statements under the assumptions recorded in `formal/mathematical-coverage.md`. The proof library is rooted at `formal/README.md`; manuscript-ready appendix fragments are in `formal/appendices/`.

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

## Manuscript integration

The current Lean appendix fragments target the corrected 2026-09-28 manuscript texts. They should be integrated into those corrected sources and compiled together; the historical manuscript PDFs should not be relabeled as machine-checked merely because the independent Lean library exists.
