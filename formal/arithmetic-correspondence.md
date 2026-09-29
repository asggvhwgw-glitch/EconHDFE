# Exact-rank arithmetic: mathematical correspondence

This document concerns the corrected 2026-09-28 manuscript identified in
[README.md](README.md). Statements are kernel-checked only when the exact source
commit builds and passes its dependency audit. The agreed scope is mathematics;
verification of executable elimination, backends and resource managers is not
required to complete this mathematical workstream. See
[mathematical-coverage.md](mathematical-coverage.md) for the consolidated mapping.

## Cross-field rank and modular certificates

`RankMinors.lean` proves that every finite matrix over a field has a nonzero
minor whose order is its actual rank. It selects independent spanning columns
from the original matrix and then independent spanning rows. Both selections
are injective. The resulting square minor has full rank and nonzero determinant.
At rank zero the empty square minor has determinant one; no fictitious pivot is
created. This is mathematical existence, not an executable minor search.

A nonzero minor bounds rank from below. Since determinants commute with ring
homomorphisms, maximal minors prove rank invariance under field embeddings.
`CharZeroRank.lean` specializes this to rational/real scalar extension and to
the same integer matrix interpreted over Q or any characteristic-zero field.
The categorical specialization uses actual 0/1 entries, connecting exact
rational rank to the real FE fitted space.

`ModularRankCertificate.lean` does not posit a homomorphism from Q to a finite
field. It starts with integer entries. A nonzero modular determinant reflects
a nonzero integer determinant, which stays nonzero in characteristic zero.
Maximal-minor existence yields

```text
rank_Fp(A mod p) <= rank_Q(A) = rank_R(A).
```

Exact acceptance additionally requires reaching an independently proved upper
bound. The proof handles both a supplied minor reaching that bound and the
actual finite-field rank reaching it. The categorical corollary uses the proved
multipartite bound, not an assumed characteristic-zero rank answer.

| Mathematical result | Lean declaration | Scope |
|---|---|---|
| Nonzero minor lower bound | `rank_lower_of_nonzero_minor` | Finite rectangular matrices over a field |
| Maximal minor exists | `matrix_exists_minor_of_rank`, `matrix_exists_rank_minor` | Injective row/column selections, including order zero |
| Field-embedding invariance | `matrix_rank_map_field` | Field homomorphism |
| Rational/real and integer invariance | `rational_real_rank`, `integer_rank_charZero`, `integer_rational_real_rank` | Finite rational or integer matrices |
| Categorical specialization | `incidence_int_map`, `categorical_rational_real_rank` | Actual full indicators |
| Finite-field lower bound | `modular_rank_le_charZero` | Integer entries and prime modulus |
| Exact acceptance | `modular_minor_rank_exact`, `modular_rank_exact` | Lower bound reaches a valid upper bound |
| Categorical acceptance | `categorical_modular_rank_exact` | Complete blocks and the proved multipartite bound |

`ModularRankExamples.lean` proves that for every prime p, `diag(p,1)` has rational
rank two and modular rank one. A larger prime does not itself certify arbitrary
inputs. A minor modulo three certifies `diag(2,1)`. These are generic integer
witnesses, not claimed to be complete categorical FE designs.

## Exact row transformations as supporting mathematics

`ExactRowReduction.lean` proves preservation of row span under nonzero scaling
plus linear combinations of retained rows, and deletion of a zero row. Its
finite trace definition does not contain a rank identity or an assumed equality
of row spaces; that equality follows by induction on legitimate operations.

The cross-elimination lemmas distinguish pivot cancellation from preservation
of span. The latter requires nonzero pivot and divisor. At an echelon endpoint,
nonzero selected pivots and zeros at earlier selected pivots give a triangular
nonzero minor. The minor bound and row count imply that rank equals the number
of retained rows. `exact_trace_rank` composes this with the span invariant.

| Supporting result | Lean declaration | Conditions |
|---|---|---|
| Row replacement and scaling | `span_insert_row_replace`, `span_insert_row_scale` | Nonzero coefficient on replaced row; other row combination retained |
| Cross-multiplication | `cross_elimination_pivot_zero`, `cross_elimination_span` | Cancellation and invertibility conditions distinguished |
| Finite span invariant | `exact_row_step_span`, `exact_row_trace_span` | Legitimate mathematical transformations |
| Endpoint rank | `echelon_rows_rank`, `exact_trace_rank` | Actual nonzero echelon pivots and completed trace |
| Proof-side normal return | `exact_fallback_normal_return` | Completed certificate; not program refinement |

`ExactEchelonCertificate` packages actual rows, pivots, operations and terminal
conditions; it does not assume its own rank answer. The optional-return adapter
is retained as a small consequence of the certificate. It is not a model of the
Python budget manager. No additional code-level normalization or state-machine
proof is planned as a completion requirement under the mathematical-only scope.

## Composition with categorical structure

`IntegerRankEvidence` provides two mathematical evidence routes for an actual
integer block: a modular hit at a valid upper bound or a completed rational row
trace. Each route gives the rational and real rank. `categorical_certified_rank`
then composes tuple coverage, legal peeling, genuine separated blocks and their
evidence:

```text
rank_R(original FE matrix) = peeled observation count + sum(certified block ranks).
```

The blocks are actual surviving integer submatrices, not a disconnected list of
reported integers. This is a conditional mathematical correctness theorem under
explicit certificates. It does not assert their production by a concrete backend
for every input or within a budget. The proper-connectivity formula is separately
proved and can provide a block rank under its stated structural hypotheses.

## What is outside this workstream

Python/Numba dictionary mutation, modular kernels, integer gcd normalization
programs, FLINT/SymPy internals, execution termination, work/bit/storage budgets
and resource-exhaustion behavior are outside this mathematical formalization.
These are boundaries, not unresolved mathematical premises hidden inside the
proved declarations. Floating-point solvers, covariance conventions, external
package parity and historical originality likewise require separate evidence.

The toolchain remains Lean 4.19.0 and the pinned Mathlib commit. A finite local
elaboration budget in maximal-minor construction does not change the theorem's
hypotheses or bypass the kernel. The build and dependency audit still cover all
modules. The appendix's names are checked against declarations, while semantic
correspondence between prose and formal statements remains an explicit review
responsibility.
