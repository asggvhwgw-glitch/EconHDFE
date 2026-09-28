# Exact multiway rank: formalization correspondence

This document maps the structural part of the corrected exact multiway FE
rank/DoF manuscript to the Lean library. It describes the declarations in the
same source tree; a declaration is verified only when that tree compiles and
passes the axiom audit. It is not a completion claim for the full manuscript,
the exact-arithmetic backend, or the Python implementation.

The manuscript target is the separately delivered 2026-09-28 corrected bundle
identified in [README.md](README.md). The manuscripts on the original base
commit still contain superseded wording and are not silently rewritten here.

## Mathematical object and domain

`incidenceMatrix F code` is the actual matrix whose `(i,v)` entry is one exactly
when observation `i` has the level named by `v`, and zero otherwise. Its rank is
Mathlib's dimension of the image of matrix multiplication. It is not the
maximum matching rank of a sparsity pattern. `incidence_rank_eq_finrank` connects
the real matrix rank to the indicator span used by the projection library.

Observation types retain row identities. Generic rank reductions allow repeated
rows because row multiplicity does not enlarge the row span. The projection
library continues to retain all observations and their separate weights and
outcomes. No deduplication theorem is applied to a weighted loss function.

Most new linear-algebra results are generic over a field. They can be
instantiated separately over the rationals, reals or a finite field. This does
not establish that an arbitrary integer matrix has equal ranks across those
fields. The peeling recursion and the composed categorical pipeline currently
use real vector spaces, reusing the previous incidence reconstruction proof.
General rational-to-real rank invariance remains a separate proof obligation.

## Correspondence

| Corrected manuscript result | Lean declarations | Conditions actually used |
|---|---|---|
| Actual categorical rank/DoF target | `incidenceMatrix`, `incidence_rank_eq_finrank` | Full categorical indicators; finite level blocks |
| Duplicate-edge reduction, `lem:dedup` | `matrix_rank_of_same_rows`, `matrix_rank_dedup`, `categorical_rank_dedup` | Every original tuple has a retained representative; arbitrary field |
| Component additivity, `lem:components` | `component_mulVec`, `componentRangeEquiv`, `matrix_rank_block_diagonal`, `matrix_rank_components` | Finite rectangular blocks, explicit row/column reindexings, actual off-block zeros |
| One-unit leaf elimination, `lem:peel` | `categorical_leaf_rank` | A valid additional singleton removal in a legal categorical trace; real rank |
| Recursive peeling rank | `peelingFitMap_bijective`, `categorical_peeling_rank`, `categorical_peeling_matrix_rank` | Any legal finite trace, including empty or partial traces; real rank |
| Complete-block column reduction, `lem:blockdrop` | `multipartite_column_span`, `multipartite_rank_drop` | Keep all columns of one reference partition; omit one selected column from every other complete block |
| Deterministic upper bound, `prop:upper` | `retained_levels_card`, `multipartite_rank_upper` | Finite sample and blocks; chosen reference partition and baseline levels; arbitrary field |
| Proper-connectivity formula, `prop:proper` | `one_coordinate_kernel`, `coordinate_path_kernel`, `proper_kernel_levels`, `proper_reduced_kernel`, `proper_connectivity_rank` | Nonempty sample, all declared levels observed, paths changing at most one coordinate; arbitrary field |
| Label renaming and removal of unused levels | `categorical_recode_rank` | Preserve equality of labels on all realized rows; real rank |
| Connected structural preprocessing identity | `categorical_structural_rank` | Tuple coverage, legal trace on representatives, separated actual residual blocks; real rank |

## How the rank recursion is obtained

The trace does not contain a rank assumption. The existing categorical
projection proof constructs actual pivot columns and proves their upper
triangular matrix has determinant one. This implies that any vector supported
on removed rows lies in the original FE span. Along with actual row restriction,
it yields an explicit linear bijection

```text
original FE fitted space
    <-> (arbitrary values on removed rows) x (core FE fitted space).
```

Taking finite dimensions gives

```text
rank(original categorical matrix) = number of removed rows + rank(core matrix).
```

A row incident to several singleton levels is removed once and contributes one
rank unit. Original level labels may be retained on the core: pivot levels then
have zero columns. Equality-preserving recoding supplies the separate bridge to
dropping unused level labels.

`peeling_residual_dimension` states that `N - rank(D)` is unchanged by a legal
projection trace. It is an FE-only residual-space dimension identity, not a
claim about residual DoF after adding regressors, or a cluster small-sample
correction. In particular, this sample-dimension statement must not be applied
after rank-only deduplication to infer the original regression's sample size.

## Components and exact column reduction

The component result constructs an explicit equivalence between the image of
the actual block diagonal matrix and the product of its component images.
Rank additivity then follows from finite-product dimension. Supplied row and
column equivalences are checked against actual zero cross-component entries;
a name such as `components` is not treated as evidence about a graph routine.
Blocks may have different row and column counts. Empty blocks are allowed.

The complete-block reduction reconstructs each removed column from the retained
reference block's all-ones sum minus the other columns of its own block. This
preserves the actual column span over every field. There are exactly
`V - (G - 1)` retained columns, so rank is at most the minimum of that count and
the row count. This upper bound alone does not certify exact rank.

## What proper connectivity means here

`OneCoordinateStep` permits a pair of tuples that differ in at most one FE
coordinate. Allowing equal tuples only adds redundant steps; on unique tuples,
nontrivial steps are precisely the manuscript's one-coordinate adjacency.
`ProperConnected` requires a finite chain of such steps between every pair of
observations. It is not ordinary incidence connectivity.

For any kernel coefficient vector, subtracting equations at adjacent rows
forces the affected level coefficients to agree. Path induction and observed
level surjectivity force all level coefficients to be constant within each
partition. Extending the reduced coefficients by zero at omitted baseline
levels then forces every non-reference partition's constant to zero. The root
row equation forces the remaining constant to zero. Thus the reduced matrix is
injective and has rank exactly `V - (G - 1)`.

The proof is generic over fields; there is no hidden division by the number of
partitions. Nonempty sample and observed-level surjectivity are explicit inputs,
not inferred from dense integer labels. Proper connectivity is sufficient only;
its failure does not imply a rank deficiency.

## Small semantic witnesses

`RankExamples.lean` proves that repeated all-equal three-FE tuples have rank one.
It also takes a representative of the duplicate pair and performs a legal
one-row rank peel. This contrasts with `duplicate_trace_empty`, which proves
that the original two-observation projection trace cannot remove either row.

For `000,001,011,111,111`, the three legal removals contribute rank three and the
two retained observations have a one-dimensional categorical fitted space.
The actual total rank is therefore four. These are exact kernel-checked
witnesses of the definitions and composition. The general results do not rely
on enumeration or a floating-point rank test.

## Remaining exact-arithmetic obligations

The general characteristic-zero equivalence between rational and real rank is
not yet proved in this library. The finite-field lower-bound theorem, extraction
of a nonzero minor or equivalent certificate, and the full modular acceptance
rule are also not yet formalized. No modular estimate is promoted to a final
answer merely because an upper bound was proved.

The rational/integer fallback still needs its elimination invariants, exact rank
return theorem, and explicit normal-return/resource-exhaustion distinction.
The current structural pipeline leaves an actual rank for each unresolved core
block; it does not solve all such blocks by proper connectivity. Prefix-rank
bookkeeping and the full resource-aware algorithm composition remain separate.

This library does not verify Python row encoding, deduplication, connectivity
search, peeling queues, modular kernels or fallback code, and it does not prove
floating-point accuracy, covariance conventions or historical originality.
