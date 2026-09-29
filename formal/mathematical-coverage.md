# Mathematical coverage of the three manuscripts

The scope is mathematical theorem verification, not program verification.
Each row below names actual Lean declarations; its verified status requires a
successful complete build and dependency audit for the exact source commit.
These rows replace informal completion percentages. They are not a claim that
all prose, numerical experiments or illustrative counterexamples are formalized.

The target is the corrected 2026-09-28 manuscript text. The corrected TeX sources
on this branch now include the three fragments under `appendices/`, and the
dedicated manuscript workflow recompiles the corresponding PDFs from those
sources. Stable manuscript labels are given where available. Machine-checked
status still applies only to the explicitly mapped mathematical statements, not
to every sentence or implementation claim in the PDFs.

## Residual-core reduction

| Mathematical statement | Lean declarations | Qualification |
|---|---|---|
| Projection problem: attainable optimum and unique fitted vector/residual | `wlsFit_iff_residual`, `wls_residual_exists_unique`, `wls_fit_exists_unique` | Every finite sample, any fitted subspace, strictly positive diagonal weights |
| Unique-level zero residual (`lem:zero`) | `singleton_residual_zero` | The singleton indicator is in the fitted space |
| Leaf elimination and reconstruction (`lem:elim`) | `peelMatrix_det_one`, `off_core_mem`, `peeling_feasible_iff`, `core_loss_transport` | Actual categorical incidence and any legal finite trace; one step is a special case |
| Recursive residual-core theorem (`thm:core`) | `categorical_residual_core_total` | Core and full projections are proved to exist, not supplied as premises |
| Recoding on the core | `categorical_reencoded_core_total` | Realized label equality must be preserved |
| Corrected normal-equation corollary | `core_inner_transport`, `categorical_residual_lifts` | Original FE-column equations, plus separate fitted-value feasibility |
| Multiple RHS and positive weight updates | `categorical_multi_rhs_total` | Topology is reusable; weighted solutions are recomputed mathematically |
| Terminal existence and order independence | `terminal_trace_exists`, `terminal_core_unique`, `categorical_terminal_core_projection` | Finite observation identities; no queue implementation assumed |

These declarations include empty and partial traces and retain duplicate rows as
distinct observations. Positive finite weights and categorical intercept columns
are mathematical premises. The zero-weight and non-diagonal-GLS exclusions are
not software-verification tasks. General numerical accuracy, tolerance attainment
and complexity are outside this appendix.

## Exact multiway categorical rank / absorbed DoF

| Mathematical statement | Lean declarations | Qualification |
|---|---|---|
| Actual FE rank target | `incidenceMatrix`, `incidence_rank_eq_finrank` | Actual indicator matrix, not generic sparsity-pattern rank |
| Characteristic-zero equivalence (`lem:RQ`) | `matrix_exists_rank_minor`, `rational_real_rank`, `categorical_rational_real_rank` | Finite matrices, actual nonzero minors |
| Duplicate-edge reduction (`lem:dedup`) | `categorical_rank_dedup` | Every original tuple is represented |
| Component additivity (`lem:components`) | `matrix_rank_components` | Actual row/column reindexings and off-block zeros |
| Leaf rank recursion (`lem:peel`) | `categorical_leaf_rank`, `categorical_peeling_matrix_rank` | One rank unit per removed observation; real rank with the separate Q/R bridge |
| Multipartite upper bound (`prop:upper`) | `multipartite_rank_upper` | Complete partitions with a reference block and baseline levels |
| Redundant column removal (`lem:blockdrop`) | `multipartite_rank_drop` | Omitted columns explicitly reconstructed |
| Proper-connectivity formula (`prop:proper`) | `proper_connectivity_rank` | Nonempty sample, observed levels, one-coordinate paths; sufficient only |
| Finite-field lower bound (`lem:finitefield`) | `modular_rank_le_charZero` | Integer matrix and prime modulus |
| Exact modular certificate (`thm:certificate`) | `modular_rank_exact`, `categorical_modular_rank_exact` | Actual modular rank reaches a separately proved upper bound |
| Main mathematical composition | `categorical_certified_rank` | Structural hypotheses and valid mathematical evidence for actual core blocks |
| Algebraic rational row-trace support | `exact_row_trace_span`, `exact_trace_rank` | Finite allowed row operations and actual echelon terminal conditions |
| Prefix contributions and redundancies (`eq:Mj`) | `prefix_increment_total`, `prefix_redundancy_total`, `matrix_prefix_redundancy_total` | Nonnegative increments; actual matrix widths give the dimension bounds |
| Total rank under reordering and positive weights | `rank_column_permutation`, `positive_weight_sqrt_rank` | Complete column permutations and strictly positive diagonal weights |

The program-form version of Algorithm 1, generation of modular/echelon evidence,
integer dictionaries, normalization implementations, exact-backend completion
and resource exhaustion are excluded from the agreed scope. The mathematical
composition remains conditional on real structural and arithmetic certificates;
it does not assume the final characteristic-zero rank but also does not promise
a concrete program produces certificates for every input. Prefix sums concern
structural rank, not cluster-specific small-sample adjustments.

## Partition refinement and structural design

| Mathematical statement | Lean declarations | Qualification |
|---|---|---|
| Dummy-space inclusion (`lem:dummy`) | `partitionSpace_eq_span`, `partitionSpace_le` | Actual level map on the entire fixed sample |
| Equivalent partitions | `equivalent_partition_spaces` | Mutual refinement |
| FE canonicalization (`thm:canon`) | `canonicalize_one`, `structural_trace_space_eq`, `structural_trace_within` | Witness retained at each mathematical step |
| Absorbed factor block (`prop:absorb`) | `partitionSpace_le`, `absorbed_block_within_zero` | Refined block lies in the actual absorbed space |
| Fine basis elimination (`prop:basis`) | `coarse_active_sum`, `drop_active_fine`, `drop_one_fine` | Complete realized coverage in the reduced coarse cell, retained coarse witness |
| Shared multiplier (`lem:slope`) | `shared_multiplier_le`, `drop_active_fine_multiplier`, `finite_reduction_common_multiplier` | Same multiplier on both sides; zero/negative entries permitted |
| Dependency closure (`lem:trans`) | `refines_trans`, `refines_restrict` | Fixed sample or row restriction; no acyclicity assumption |
| Interaction certificate (`prop:interaction`) | `interaction_refines` | Component-level determining maps supplied |
| Main structural column-space theorem (`thm:main`) | `structural_step_space_eq`, `structural_trace_space_eq` | Equality derived from allowed algebraic constructors, not inserted as a trace premise |
| WLS fitted-value corollary | `wls_fit_exists_unique`, `structural_trace_wls_fit` | Same sample, response and positive diagonal weights |
| Full / within WLS connection | `full_wls_implies_within`, `within_wls_implies_full` | Same unpenalized model spaces |
| Identified coefficients | `full_wls_identified_fe_invariance`, `within_identified_coefficients` | Injective within design |
| Rank-deficient coefficient selection | `within_selection_invariance` | Same selector on the same transformed problem, not an arbitrary full-model minimum-norm rule |
| Estimable linear functions | `estimable_coefficient_fiber`, `estimable_fit_function_invariance` | Kernel-annihilating coefficient functionals or the same linear functional of fits |
| Finite PPML corollary | `ppml_midpoint_strict`, `finite_ppml_predictor_unique`, `structural_trace_ppml`, `finite_ppml_response_invariance` | Attained finite optima, positive weights, same response and offset |

The finite trace is a mathematical composition of the paper's transformations,
not the planner's program state. Missing fine reference cells inside a required
coarse cell invalidate the full-sum witness; the theorem does not claim such a
reduction is always available or maximal. PPML uniqueness is proved from the
actual exponential objective, but finite optimum existence is not claimed for
separated samples. IV/GMM/covariance conclusions require additional mathematics
and are not inferred here.

## Integration and claims

Appendix sources are `appendices/residual-core.tex`, `appendices/exact-rank.tex`
and `appendices/structural-design.tex`. They are now included by the corrected
manuscript sources and compiled into the branch PDFs. Their purpose is to identify
precisely which mathematical statements are machine-checked under which premises.

The mathematical library has no runtime dependency on an AI agent. The trusted
basis is the pinned Lean kernel plus the foundational axioms reported in the
audit. Historical novelty, Python correctness, floating-point behavior and
external Stata parity remain separate claims, not pending gates for this
mathematical formalization workstream.
