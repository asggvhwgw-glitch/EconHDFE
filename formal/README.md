# EconHDFE: Lean proofs of the mathematical theory

This is an independent mathematical library, not a dependency of the Python
package. Its scope is the mathematical results of the three corrected technical
manuscripts. Verification of Python/Numba programs, sparse dictionaries, modular
kernels, queue code, gcd normalization code, numerical convergence, resource
budgets and backend completion is **not a completion requirement** for this work.

The manuscript target remains the separately delivered 2026-09-28 corrected
bundle, `econhdfe-manuscripts-corrected-20260928.zip`, prepared against
`main@06cd0a5efe34de50c52d8b7c0db68c240b9d2673`. The corrected bundle has not been
silently substituted into this branch. The older manuscript sources/PDFs on the
base commit still contain superseded wording. The appendix fragments below are
provided separately for the corrected manuscripts; this Lean workflow does not
compile or replace manuscript PDFs.

## What is verified

Verification applies to the exact Lean declarations in a commit whose complete
`lake build` and per-theorem dependency audit both succeed. It is not a blanket
claim that every sentence, benchmark, illustrative example, algorithm description
or historical originality claim in a manuscript is machine-checked. Progress is
recorded by mathematical correspondence, not estimated percentages or counts of
helper lemmas.

| Manuscript | Mathematical proof chain | Appendix fragment |
|---|---|---|
| Residual-core reduction | WLS existence and uniqueness; actual categorical incidence; legal leaf removal; triangular reconstruction; exact loss/normal-equation transport; total residual-core identity; terminal existence and order independence; recoding and multiple RHS | [Residual-core appendix](appendices/residual-core.tex) |
| Exact multiway FE rank/DoF | Actual matrix rank; duplicate rows; component additivity; peeling rank; complete-block reduction; proper-connectivity formula; Q/R equivalence; modular lower bounds and exact certificates; conditional mathematical composition; prefix allocation and positive weights | [Exact-rank appendix](appendices/exact-rank.tex) |
| Structural design reduction | Refinement and dependency closure; full and active/reference-coded block identities; shared multipliers; finite algebraic composition; full/within WLS equivalence; identified or consistently selected coefficients; attained finite PPML predictors | [Structural-design appendix](appendices/structural-design.tex) |

[Mathematical coverage](mathematical-coverage.md) gives the manuscript statements,
Lean names and exact qualifications. More detailed descriptions of the existing
rank layers remain in [rank-correspondence.md](rank-correspondence.md) and
[arithmetic-correspondence.md](arithmetic-correspondence.md).

## Common WLS foundation

`WeightedLeastSquares.lean` defines a residual by both attainable fitted values
and weighted orthogonality. It proves equivalence with the explicit weighted
least-squares objective and uniqueness under strictly positive diagonal weights.
`LeastSquaresExistence.lean` now also proves existence for every finite input,
using the normal-equation map into the dual of the fitted space. It constructs
the unique mathematical residual and its linear within operator. No numerical
optimizer is assumed to exist or converge.

`ResidualCoreTotal.lean` uses this existence theorem to remove the former
assumption that an exact core residual or attained minimizer has already been
provided. The categorical proof starts from actual indicator columns and legal
singleton incidence; it does not assume an invertible block. Duplicate tuples
retain separate row identities. Positive weight changes may reuse topology,
not an old weighted solution. `core_inner_transport` gives the corrected normal
equations in original column coordinates, and `core_loss_transport` preserves
the objective on zero extension.

## Structural reductions and statistical consequences

`ActivePartitionRefinement.lean` permits partial/reference-coded blocks only
when every realized fine level needed inside the reduced coarse cell remains
represented. The retained coarse witness is explicit. Common multipliers act
on both the coarse witness and fine columns, and need not be nonzero or positive.

`StructuralConsequences.lean` describes finite sequences of these mathematical
reductions. A step carries actual refinement/coverage data, not a pre-assumed
space equality. The proof derives the equality and then its WLS and PPML
consequences. This mathematical trace is not a model of the planner's program.

`FrischWaughLovell.lean` connects full and within WLS optima in both directions
and proves identified non-FE coefficient invariance. `WithinInvariance.lean`
handles the same selector on the same transformed problem in rank-deficient
cases. It does not equate arbitrary full-model minimum-norm coordinates.
Estimable coefficient functionals must annihilate the design kernel.

`PPMLInvariance.lean` uses the actual weighted exponential objective and proves
strict midpoint convexity and uniqueness of an attained finite predictor.
Equal design spaces, unchanged response, sample, positive weights and offset
then give identical eta and mu. The theorem does not assert existence of finite
optima for separated samples. No IV/GMM or covariance invariance is inferred
from column-space equality alone.

## Exact rank and its mathematical boundary

The rank is that of actual 0/1 matrices, not a sparsity-pattern matching rank.
The library proves maximal nonzero minors, field-embedding invariance and
`rank_Fp(A mod p) <= rank_Q(A) = rank_R(A)` for integer A and prime p. Modular
acceptance needs a proved upper bound. A modular shortfall is not a deficiency
certificate, regardless of the size of the prime.

The mathematical composition theorem requires valid evidence for each actual
core block. Existing finite row-trace lemmas are retained as algebraic support;
producing such traces from production programs is outside the present scope.
`RankConsequences.lean` supplies prefix-rank redundancy identities and proves
that multiplication by a strictly positive diagonal square-root weight matrix
does not change structural rank. Cluster finite-sample conventions remain
separate from the rank target.

## Reproduce and audit

Install the official Lean toolchain manager, then run in this directory:

```bash
lake update
lake exe cache get
lake build
```

`lean-toolchain` pins Lean 4.19.0 and `lakefile.lean` pins Mathlib to
`c44e0c8ee63ca166450922a373c7409c5d26b00b`. This pair is chosen for reproducibility,
not as a latest-version claim. The independent Lean workflow compiles every
module and runs `#print axioms` for every named project theorem. It rejects
proof placeholders and unproved project assertions. Permitted foundational
axioms are `propext`, `Classical.choice` and `Quot.sound`; this is not axiom-free
mathematics. The evidence artifact preserves dependency versions, theorem names,
build/audit logs, the exact source commit, hashes and proof sources.

The library does not change Python APIs, test organization, release status or
innovation-registry claims. Mathematical verification, software correctness and
historical novelty remain separate.
