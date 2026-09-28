# EconHDFE Lean formalization — first increment

This is a separate mathematical proof library, not a dependency of the Python
package. Its target is the **2026-09-28 corrected manuscript bundle**
`econhdfe-manuscripts-corrected-20260928.zip`, prepared against
`main@06cd0a5efe34de50c52d8b7c0db68c240b9d2673`. Those manuscript corrections
were delivered separately and are not silently substituted into this branch.
The older manuscripts still present on the base commit are not the authority
for the corrected coefficient-selection and residual-feasibility conditions.

This increment does **not** formalize all three manuscripts. It does not verify
the Python implementation, finite-precision error bounds, numerical convergence,
statistical assumptions or historical originality. Only the precise Lean
statements in a successfully compiled and audited commit are kernel-checked.

## Reproduce

Install the official Lean toolchain manager `elan`, then run from this directory:

```bash
lake update
lake exe cache get
lake build
```

`lean-toolchain` pins Lean 4.19.0. `lakefile.lean` pins mathlib to commit
`c44e0c8ee63ca166450922a373c7409c5d26b00b` (the matching v4.19.0 library).
This compatible pair is selected for reproducibility, not claimed to be the
latest release. CI retains the generated complete dependency manifest, compiler
identity, source hashes, theorem list, build log and axiom audit.

The isolated workflow is `.github/workflows/lean.yml`; it does not modify the
Python test or release workflows. A successful workflow requires `lake build`
and an axiom audit of every named project theorem. The audit rejects proof
placeholders and project-defined unproved assertions. The only allowed ambient
axioms are the standard `propext`, `Classical.choice`, and `Quot.sound`.
This is not a claim of axiom-free foundational mathematics.

## What the current modules state

### PartitionRefinement.lean

`Refines Q P` is an actual level map satisfying `P i = f (Q i)` on every sample
row. `partitionSpace P` is the range of the categorical coefficient map; the
separate `partitionSpace_eq_span` theorem connects it to the span of the full
indicator columns. Thus a column-space claim is not hidden inside an unrelated
abstract definition.

| Corrected manuscript result | Lean declaration | Scope |
|---|---|---|
| Refinement transitivity | `refines_trans` | Exact composition of level maps |
| Preservation of dependencies under row restriction | `refines_restrict` | Arbitrary row-selection map |
| Dummy-space inclusion (`lem:dummy`) | `partitionSpace_le` | Entire categorical column space |
| Equivalent partitions | `equivalent_partition_spaces` | Both directions of refinement |
| FE canonicalization (`thm:canon`) | `canonicalize_one`, `finite_reduction_preserves_space` | Retained fine witness; finite composition of certified equalities |
| Common continuous multiplier (`lem:slope`) | `shared_multiplier_le` | Identical multiplier on both blocks; zero/negative values allowed |
| Component interaction certificate (`prop:interaction`) | `interaction_refines` | Each coarse component determined by a selected fine component |
| Fine-column elimination (`prop:basis`) | `coarse_indicator_sum`, `drop_one_fine` | Full-dummy case, one removed fine column and retained coarse space |

The active/reference-coded partial-block extension and the concrete Python
certification/closure algorithms remain outside this increment. Whole-model
Moore–Penrose coefficient invariance is neither asserted nor used. Equality of
spaces alone does not identify arbitrary coefficient coordinates.

### WeightedLeastSquares.lean

`IsWLSFit` means an attainable fitted vector minimizing the explicit finite sum
`sum_i w_i * residual_i^2`. `IsResidual` requires **both** attainable fitted
values and orthogonality to the fitted subspace.

`wlsFit_iff_residual` proves the equivalence under strictly positive diagonal
weights. `residual_unique` proves uniqueness. `wls_fitted_invariance` then
proves identical fitted vectors for equal spaces, on the same sample with the
same outcome and positive diagonal weights. `singleton_residual_zero` proves
that a genuine singleton indicator in the fitted space forces zero residual
at that observation.

These statements concern unpenalized real WLS. They do not assert PPML/IV/GMM
convergence, estimator-wide parity, or covariance invariance. Existence of a
minimizer for every input and the corrected PPML corollary are not separately
formalized in this increment.

### ResidualCore.lean

`blockMap T B D` is the actual block predictor `(a,b) -> (T a + B b, D b)`.
`block_range_eq_liftCore` constructively proves that, when `T` is surjective,
its range consists exactly of vectors with unrestricted peeled coordinates
and attainable core coordinates. This proves reconstruction feasibility.

`normal_transport` proves weighted orthogonality transport in original column
coordinates. `residual_lifts`, `block_residual_reconstruction`, and
`block_multi_rhs` combine feasibility, normal equations and uniqueness, and
show that full residuals are zero extensions of core residuals.

The block theorem explicitly assumes surjectivity of `T`. This is a genuine
algebraic hypothesis. It is **not** a proof that every legal hypergraph peeling
sequence produces a unit-triangular `T`, or that the production queue is correct.
Those connections, recursive graph invariants and peel-order independence must
still be formalized before claiming the full categorical residual-core theorem.

## Remaining work before full-manuscript certification

The structural-design line still needs active/reference-coded block elimination,
composition with common multipliers, explicit within-coefficient selection, and
the finite PPML corollary under its exact assumptions.

The residual-core line still needs a concrete categorical incidence model,
legal peeling trace, induced triangular block, termination, order independence,
row/column reindexing, and explicit zero-weight/non-diagonal counterexamples.

The exact multiway rank line is not yet implemented here. Its remaining chain
includes characteristic-zero equivalence, duplicate rows, component direct
sums, rank-one leaf elimination, multipartite nullity bounds, certified column
removal, proper-connectivity rank, finite-field lower bounds, and certified
composition with a correct characteristic-zero fallback. A numerical rank oracle
or a theorem that merely assumes the desired rank is not a substitute.

No theorem-backed registry status or README marketing claim is promoted by the
existence of this directory. Full-paper and Python-implementation certification
remain distinct from this first proof library.
