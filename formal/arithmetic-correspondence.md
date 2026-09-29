# Exact-rank arithmetic: proof correspondence and boundaries

This increment targets the corrected 2026-09-28 manuscript identified in
[README.md](README.md). It does not replace the separately delivered manuscript
corrections. Statements in this document describe the same source tree; they
are kernel-checked only when that exact tree builds and passes the proof audit.
No claim about a Python execution or historical originality is inferred from
mathematical formalization.

## Cross-field rank and modular certificates

`RankMinors.lean` proves that every finite matrix over a field has a nonzero
minor whose order is its actual rank. The proof selects independent spanning
columns from the original matrix and then independent spanning rows. The row
and column selections are injective. The square minor has full rank, hence
nonzero determinant. Rank zero uses the conventional determinant one on an
empty square matrix; it does not create a fictitious pivot. The existence
proof is noncomputable mathematics, not an executable minor search.

A nonzero minor supplies a rank lower bound. Determinants commute with ring
homomorphisms, so maximal minors prove rank invariance under field embeddings.
`CharZeroRank.lean` specializes this to rational-to-real scalar extension and
to the same integer matrix interpreted over Q or any characteristic-zero
field. The categorical corollary uses actual 0/1 indicator entries, connecting
the rational exact-rank target to the real FE space used in projection.

`ModularRankCertificate.lean` does not posit a homomorphism from Q to a finite
field. It starts with an integer matrix. A modular nonzero minor reflects a
nonzero integer determinant, which remains nonzero in characteristic zero.
Maximal-minor existence then supplies the complete inequality

```text
rank_Fp(A mod p) <= rank_Q(A) = rank_R(A).
```

Acceptance additionally needs an independently valid upper bound. It is proved
both from a supplied nonzero minor reaching that bound and from the actual
finite-field rank reaching it. The categorical corollary composes the existing
multipartite upper-bound theorem; it does not introduce a rank-equality
hypothesis for the characteristic-zero matrix.

| Result | Declaration | Scope |
|---|---|---|
| Nonzero minor lower bound | `rank_lower_of_nonzero_minor` | Finite rectangular matrices over a field |
| Maximal nonzero minor exists | `matrix_exists_minor_of_rank`, `matrix_exists_rank_minor` | Actual injective row/column selections, including order zero |
| Rank preserved by scalar extension | `matrix_rank_map_field` | Field homomorphism; no finite-field cast of arbitrary rationals |
| Rational/real and integer invariance | `rational_real_rank`, `integer_rank_charZero`, `integer_rational_real_rank` | General finite rational or integer matrix |
| Categorical specialization | `incidence_int_map`, `categorical_rational_real_rank` | Actual full categorical indicators |
| Finite-field lower bound | `modular_rank_le_charZero` | Integer entries and prime modulus |
| Exact modular acceptance | `modular_minor_rank_exact`, `modular_rank_exact` | Modular evidence reaches a proved upper bound |
| Categorical acceptance | `categorical_modular_rank_exact` | Complete FE blocks and the proved multipartite bound |

`ModularRankExamples.lean` proves an adversarial family for **every prime**:
`diag(p, 1)` has rational rank two and rank one modulo p. Increasing the chosen
prime therefore does not itself prove correctness for arbitrary inputs. A
minor modulo three certifies rank two for `diag(2, 1)`. These are generic
integer-matrix witnesses, not claimed to be full categorical FE designs.

## Exact fallback: what is and is not proved

`ExactRowReduction.lean` gives trace-based partial correctness. A valid step
replaces a row by a nonzero multiple plus a linear combination of retained
rows, or removes a zero row. The trace definition contains neither a rank
identity nor a row-space equality. The latter is proved by induction on the
finite trace. This covers the algebra of exact cross-multiplication and
nonzero primitive-row scaling used in the native rational path.

The cross-elimination lemma explicitly separates cancellation from rank
preservation: cancellation alone even holds at a zero divisor parameter in
Lean's total division convention, but the **span-preservation theorem requires
both the pivot entry and the divisor to be nonzero**. The native integer gcd
normalization still needs its own representation and divisibility bridge.

At termination, the retained rows must have nonzero selected pivots and zero
entries at all earlier selected pivots. These properties prove a triangular
nonzero minor. Its determinant and the row-count upper bound establish that
the row rank equals the number of retained rows. Combining this with the
proved trace invariant gives `exact_trace_rank`.

`ExactEchelonCertificate` packages the actual rows, pivots, legal finite trace
and terminal conditions; it does not contain an assumed matrix-rank answer.
`exact_fallback_normal_return` proves the reported count correct when this
mathematical evidence is present. `exact_fallback_unresolved` says that absence
of completed evidence produces no answer in the proof-side optional-result
adapter. This small adapter is **not** the Python budget manager, a memory
bound, or a proof that an executable search terminates.

| Result | Declaration | Scope |
|---|---|---|
| Row replacement preserves span | `span_insert_row_replace` | Nonzero coefficient on the replaced row; pivot retained in the other span |
| Primitive scaling | `span_insert_row_scale` | Nonzero scalar |
| Cross-multiplication | `cross_elimination_pivot_zero`, `cross_elimination_span` | Cancellation and invertible replacement conditions kept separate |
| Finite reduction invariant | `exact_row_step_span`, `exact_row_trace_span` | Derived from allowed operations, not assumed span equality |
| Terminal pivot count is rank | `echelon_rows_rank`, `exact_trace_rank` | Actual nonzero echelon pivots and a completed legal trace |
| Normal-return correctness | `exact_fallback_normal_return` | Completed proof-side certificate; no code refinement claim |

No theorem here yet establishes that the production dictionary-based loop
emits these traces, that its gcd divisions realize the indicated rational
scalings, that it always terminates with enough resources, or that work/bit/
storage exhaustion is represented by this proof-side adapter. The FLINT and
SymPy implementations are not verified by this module either.

## Composition with the categorical structural stage

`CertifiedRankPipeline.lean` accepts two mathematical evidence routes for an
integer core block: a modular hit at a valid upper bound, or a completed exact
rational row trace. Both routes prove its rational and real ranks. The
`categorical_certified_rank` theorem combines these with the previously proved
tuple-coverage, legal-peeling and actual cross-block-zero conditions:

```text
rank_R(original FE matrix) = peeled observation count + sum(certified block ranks).
```

The blocks in this statement are the **actual integer submatrices** of the
surviving categorical design. No free-standing list of integers is accepted
as a list of block ranks. Every unresolved block needs its own evidence. This
is a conditional correctness theorem under explicit structural and arithmetic
certificates, not a proof that a particular backend generates them for every
input or within a specified resource budget.

## Remaining obligations

The next algorithmic step is a bridge from the actual sparse elimination state
to valid row traces, including pivot ordering, zero-entry removal, gcd scaling,
normal completion and resource-exhaustion semantics. The modular numerical
kernels likewise need verification of their returned rank or an emitted minor
certificate. Prefix-rank bookkeeping, mixed-slope designs, floating-point
solvers, covariance conventions and external-package parity remain separate.

The library remains pinned to Lean 4.19.0 and the same Mathlib commit. One
finite local elaboration budget is raised for the maximal-minor construction;
this does not add a mathematical hypothesis or bypass kernel checking. The
normal build and per-theorem dependency audit still apply to every module.
