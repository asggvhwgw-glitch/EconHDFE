# EconHDFE Lean formalization

This is a separate mathematical proof library, not a dependency of the Python
package. Its target is the **2026-09-28 corrected manuscript bundle**
`econhdfe-manuscripts-corrected-20260928.zip`, prepared against
`main@06cd0a5efe34de50c52d8b7c0db68c240b9d2673`. Those corrections were delivered
separately and are not silently substituted into this branch. The older
manuscripts on the base commit are not the authority for the corrected
coefficient-selection and residual-feasibility conditions.

The current increment connects actual categorical indicator columns and legal
observation-level peeling to the weighted residual-core reconstruction theorem.
It also covers the first increment's partition-refinement and WLS foundations.
It does **not** formalize all three manuscripts or verify the Python
implementation, floating-point accuracy, numerical convergence, statistical
assumptions, or historical originality. Only precise Lean statements in a
successfully compiled and audited commit are kernel-checked.

## Reproduce and audit

Install the official Lean toolchain manager `elan`, then run from this directory:

```bash
lake update
lake exe cache get
lake build
```

`lean-toolchain` pins Lean 4.19.0. `lakefile.lean` pins mathlib to commit
`c44e0c8ee63ca166450922a373c7409c5d26b00b`, the matching v4.19.0 library. This pair
is selected for reproducibility, not claimed to be the latest release.

The isolated `.github/workflows/lean.yml` workflow compiles the library and audits
every named project theorem with `#print axioms`. It rejects proof placeholders
and project-defined unproved assertions. The only permitted ambient axioms are
`propext`, `Classical.choice`, and `Quot.sound`. This is not a claim of axiom-free
foundational mathematics. The evidence artifact retains the complete generated
dependency manifest, compiler identity, source commit and hashes, theorem list,
build log, audit log, and proof sources. Python CI and release workflows are not
changed by the proof library.

## Categorical residual-core proof chain

`CategoricalDesign.lean` defines `feColumn` as an actual categorical indicator,
and `feSpace` as the span of those columns. Its coordinates are **observation
identities**: equal FE tuples remain separate rows. `PeelingTrace` records a
finite sequence of distinct removed observations and selected FE levels. Its
only incidence condition is that the selected level occurs on exactly the
selected row among observations not removed earlier. It does not assume a
matrix rank, invertible block, column-space equality, or residual conclusion.

`PeelingProjection.lean` constructs the actual pivot matrix
`T[i,j] = feColumn code (pivot j) (row i)`. Trace legality implies diagonal entries
one, zeros below the diagonal, and zeros on surviving rows. The determinant is
therefore one, including the empty-trace case, and its coefficient map is
surjective. A linear combination of original pivot columns can consequently
fit any vector supported on the removed observations.

Together with the separately proved restriction of the original indicator span,
this establishes the precise feasibility equivalence

```text
z belongs to the original FE column space
    if and only if
z restricted to surviving rows belongs to the restricted FE column space.
```

Core residuals are inserted into their original row coordinates, with zeros
elsewhere. Finite weighted inner-product transport proves the full normal
equations, while the feasibility equivalence proves that the reconstructed fit
is attainable. Combining these two facts with WLS optimality and uniqueness
closes the projection proof; orthogonality alone is never used as a substitute
for attainable fitted values.

| Mathematical result | Lean declaration | Exact scope |
|---|---|---|
| Restrict the actual full-dummy span | `feSpace_restrict` | Original level labels, arbitrary row map |
| Legal trace gives the triangular pivot matrix | `peeling_column_diagonal`, `peeling_column_later_zero`, `peeling_column_core_zero` | Derived from active singleton incidence |
| Pivot block is invertible | `peelMatrix_det_one`, `peelMatrix_surjective` | No invertibility hypothesis in the categorical theorem |
| Reconstruct all removed coordinates | `off_core_mem`, `peeling_feasible_iff` | Original indicator span and original row identities |
| Lift feasibility and normal equations | `categorical_residual_lifts` | Exact core residual; no numerical solver assumption |
| Exact recursive residual-core theorem (`thm:core`) | `categorical_residual_core` | Finite sample, positive diagonal weights, any legal trace, including partial traces |
| Core minimizer produces a full minimizer | `categorical_core_wls_fit` | Assumes an attained core optimum, not a full-sample solver |
| Multi-RHS and positive weight updates | `categorical_multi_rhs` | Same trace, each RHS may have different strictly positive weights |
| Terminal trace exists | `terminal_trace_exists`, `terminal_extension_exists`, `peeling_length_le` | A finite sample bounds the number of valid removals; no verification of the Python queue |
| Terminal core is order independent | `noSingletons_survive`, `terminal_core_unique` | Any two legal traces ending with no active singleton |
| Core redensification preserves the result | `feSpace_recode`, `categorical_reencoded_residual_core` | Recoding preserves equality of labels on realized core rows; unused labels need not be retained |

`PeelingTermination.lean` constructs a valid one-row extension whenever a
surviving singleton exists and proves terminal existence by decreasing remaining
sample size. Its extension theorem states existence of a terminal legal trace
whose survivor set is contained in the initial trace's survivor set; a separate
API for literal trace-prefix refinement is not asserted. A leafless subset
survives every legal trace. Two terminal traces must therefore have the same
surviving set.

`CategoricalRecode.lean` removes an otherwise hidden redensification assumption:
only equality relations between labels on actual rows must be preserved.
Unused levels can disappear and the label types may differ.

`PeelingExamples.lean` supplies small kernel-checked witnesses. A three-FE
cascade with tuples `000, 001, 011, 111, 111` removes the first three observations
and retains both equal terminal tuples. A separate duplicate-pair result proves
that no row can be legally removed from two identical tuples. These examples
check the definitions' intended interpretation; the general theorem does not
rely on finite enumeration.

The earlier `ResidualCore.lean` remains a reusable block-algebra module with an
explicit surjectivity assumption. The categorical chain now derives that
property from incidence instead of assuming it. The production queue, encoding
implementation and floating-point core solver remain separate obligations.

## Partition refinement and WLS foundations

`PartitionRefinement.lean` defines refinement by an actual level map and relates
the categorical coefficient-map range to the full indicator span. It proves
refinement transitivity, restriction under row deletion, column-space inclusion,
equivalent partitions, one-step FE canonicalization, a shared continuous
multiplier, component interaction certificates, full-dummy fine-column
elimination, and finite composition of certified space equalities.

The relevant declarations are `refines_trans`, `refines_restrict`,
`partitionSpace_eq_span`, `partitionSpace_le`, `equivalent_partition_spaces`,
`canonicalize_one`, `shared_multiplier_le`, `interaction_refines`,
`coarse_indicator_sum`, `drop_one_fine`, and
`finite_reduction_preserves_space`. Shared multipliers may be zero or negative,
but must be the same on both blocks. `drop_one_fine` requires a retained coarse
space and the complete fine block; arbitrary reference-coded partial blocks
are not covered by that statement.

`WeightedLeastSquares.lean` defines `IsWLSFit` as an attainable fitted vector
minimizing the explicit finite weighted sum of squared residuals. `IsResidual`
requires both fitted-value feasibility and orthogonality to the fitted space.
`wlsFit_iff_residual` proves their equivalence under strictly positive diagonal
weights; `residual_unique` proves uniqueness. `wls_fitted_invariance` gives the
same fitted vector for equal spaces on the same sample, outcome and weights.
It does not assert equality of arbitrary rank-deficient coefficient coordinates.

## Remaining boundaries

The categorical residual-core reconstruction theorem, mathematical terminal
core existence and order independence now have a connected proof chain. This
is narrower than full-manuscript certification. General WLS existence for every
input is not separately established here: the reconstruction result assumes an
exact core residual or attained core minimizer. Explicit zero-weight and
non-diagonal-GLS counterexamples, quantitative floating-point error bounds, and
correctness or complexity of the production queue and encoder remain outside
the current library. Positive weight reuse refers to topology, not reusing a
weighted solution after changing its weights.

The structural-design line still needs active/reference-coded block elimination,
its complete composition with common multipliers, explicit within-coefficient
selection, and the finite PPML corollary. No IV/GMM/covariance invariance or
estimator convergence follows merely from equality of fitted spaces.

The exact multiway rank line is not yet implemented here. Its remaining chain
includes characteristic-zero equivalence, duplicate rows, component direct
sums, rank-one leaf elimination, multipartite nullity bounds, certified column
removal, proper-connectivity rank, finite-field lower bounds, and composition
with a correct characteristic-zero fallback. A numerical oracle or a theorem
that assumes the desired rank is not a replacement for those proofs.

No innovation-registry status, homepage claim, software version, Python API or
release status is promoted by this directory. Mathematical theorem coverage,
implementation correctness and historical originality remain distinct claims.
