# Exact multiway rank: mathematical correspondence

This document maps the structural rank/DoF theory to Lean declarations in the
same source tree. A declaration is verified only when that exact tree compiles
and passes the dependency audit. The scope is mathematical theory, not program
verification. The arithmetic part is mapped in
[arithmetic-correspondence.md](arithmetic-correspondence.md), and the consolidated
manuscript coverage is in [mathematical-coverage.md](mathematical-coverage.md).

The target is the corrected 2026-09-28 exact-rank manuscript text identified in
[README.md](README.md). The corresponding source on this branch now includes
[appendices/exact-rank.tex](appendices/exact-rank.tex), and the dedicated
manuscript workflow recompiles its PDF. The proof library still verifies only the
mapped mathematics, not the Python backend or every implementation statement in
the manuscript.

## Mathematical object and domain

`incidenceMatrix F code` is the actual matrix whose entry `(i,v)` is one when
observation i has the level named by v, and zero otherwise. Rank is the dimension
of the image of matrix multiplication, not the maximum matching rank of a
sparsity pattern. `incidence_rank_eq_finrank` identifies its real rank with the
indicator span used by the projection theory.

Row multiplicities do not enlarge the rank span, so duplicate-row reduction is
legitimate for rank. Projection retains all observation identities and their
separate outcomes and weights. No deduplication theorem is applied to a weighted
loss function. In particular, rank-only deduplication does not change the sample
size used for an original regression's residual degrees of freedom.

Most structural lemmas are generic over fields. The peeling proof uses real
fitted spaces. The separate `CharZeroRank.lean` bridge supplies rational/real
rank invariance; finite-field rank equality is not assumed when a prime loses
rank.

## Correspondence

| Mathematical statement | Lean declarations | Conditions |
|---|---|---|
| Actual categorical rank target | `incidenceMatrix`, `incidence_rank_eq_finrank` | Full indicators, finite level blocks |
| Duplicate-edge reduction (`lem:dedup`) | `matrix_rank_of_same_rows`, `matrix_rank_dedup`, `categorical_rank_dedup` | Every original row value has a representative |
| Component additivity (`lem:components`) | `componentRangeEquiv`, `matrix_rank_block_diagonal`, `matrix_rank_components` | Actual finite rectangular blocks and off-block zeros |
| Leaf elimination (`lem:peel`) | `categorical_leaf_rank` | A legal additional singleton removal |
| Recursive peeling | `peelingFitMap_bijective`, `categorical_peeling_matrix_rank` | Any finite legal trace, including partial or empty traces |
| Complete-block reduction (`lem:blockdrop`) | `multipartite_column_span`, `multipartite_rank_drop` | Retain one complete reference block, omit one column from each other block |
| Upper bound (`prop:upper`) | `retained_levels_card`, `multipartite_rank_upper` | Reference partition and baseline levels supplied |
| Proper-connectivity formula (`prop:proper`) | `one_coordinate_kernel`, `coordinate_path_kernel`, `proper_reduced_kernel`, `proper_connectivity_rank` | Nonempty sample, every level observed, one-coordinate paths |
| Core recoding | `categorical_recode_rank` | Realized level equality preserved |
| Structural composition | `categorical_structural_rank` | Tuple coverage, legal peeling, actual separated core blocks |
| Characteristic-zero bridge | `categorical_rational_real_rank` | Actual integer indicator entries |
| Exact modular certificate | `categorical_modular_rank_exact` | Prime modulus; actual modular rank reaches the proved bound |
| Structural/arithmetic composition | `categorical_certified_rank` | Valid mathematical evidence for each actual core submatrix |
| Prefix dimension accounting | `prefix_increment_total`, `matrix_prefix_redundancy_total` | Actual matrix block spaces and width bounds |
| Reordering and positive weights | `rank_column_permutation`, `positive_weight_sqrt_rank` | Column permutation; strictly positive diagonal weights |

## Why peeling adds one rank unit

The legal trace does not contain a rank assumption. Incidence conditions imply
that its actual pivot matrix is upper triangular with determinant one. Together
with the actual restriction of the categorical span, this gives a linear
bijection

```text
original fitted space <-> removed-row coordinates x core fitted space.
```

Finite dimensions give `rank(D) = number of removed rows + rank(core)`.
One row incident to several singleton levels is removed once and contributes
one rank unit. Keeping original labels on the core merely leaves zero columns
for unused levels; equality-preserving recoding supplies the separate bridge
for removing those labels. `peeling_residual_dimension` concerns the FE-only
quantity N - rank(D), not additional regressors or cluster corrections.

## Components and block relations

Component additivity constructs an equivalence of the actual matrix image and
the product of component images. Row/column reindexings must satisfy genuine
cross-block zeros. The statement does not certify a connectivity routine.
Rectangular components of different sizes and empty blocks are allowed.

The complete-block reduction reconstructs each removed column as the sum of
reference-block columns minus the remaining columns of its own partition.
Exactly V - (G - 1) columns remain; the rank is bounded by their count and by
the number of rows. That upper bound alone does not certify equality.

## Proper connectivity is a sufficient condition

`OneCoordinateStep` allows two tuples to differ in at most one coordinate.
Equal tuples add only redundant steps; on unique tuples, nontrivial steps are
the manuscript's one-coordinate adjacency. `ProperConnected` requires paths
between every pair of observations. Ordinary incidence connectivity is weaker.

Subtracting adjacent row equations makes kernel coefficients agree in the
changed coordinate. Path induction and observed-level coverage make them
constant within each partition. Extending reduced coefficients by zero on the
omitted baseline columns forces all nonreference constants to zero; the root
row equation forces the remaining constant to zero. Thus the reduced matrix is
injective. No division by G is used. Failure of the sufficient connectivity
condition is not a deficiency certificate.

## Semantic examples and limits

`RankExamples.lean` proves the duplicate three-FE rank-one case and the rank-four
cascade `000,001,011,111,111`. The duplicate-pair projection example separately
shows why its original rows cannot be removed by a projection trace.
`ModularRankExamples.lean` gives a generic integer family `diag(p,1)` with rank
two in characteristic zero and one modulo p. These are exact witnesses, not an
exhaustive formalization of every illustrative matrix in the manuscript.

Mathematical evidence composition is established. Generation of evidence by a
Python program, dictionary/gcd representation, backend termination, budgets and
error-state refinement are outside the agreed scope, not pending completion
gates. The optional-result adapter is not a production budget manager. Covariance
conventions, floating-point accuracy and historical novelty are separate claims.
