# Row-partitioned HDFE weighted least squares: implementation note

> **Status:** Current project technical documentation.  
> **Scope:** Row-partitioned weighted least-squares execution for structurally sparse HDFE designs.  
> **Boundary:** Established numerical method adapted to EconHDFE; not registered as a new theorem-backed contribution.


## Status and originality boundary

This note documents an **established numerical method adapted to econhdfe's structural execution layer**. It is not registered in `innovation-registry.json` as a new technical contribution.

The prior-art boundary is important. Least-squares problems with block-angular observation matrices and QR-based structure-exploiting solvers are classical; see M. G. Cox, *The least-squares solution of linear equations with block-angular observation matrix*, in *Reliable Numerical Computation* (Oxford University Press, 1990), pp. 227–239. Modern numerical-linear-algebra treatments also cover block-angular least squares. Recursive row-block QR/TSQR and communication-avoiding QR are established by Demmel, Grigori, Hoemmen, and Langou, *SIAM Journal on Scientific Computing* 34(1), 2012, A206–A239, DOI 10.1137/080731992.

The econhdfe-specific engineering contribution is to expose this class of numerical method automatically from **pre-materialization factor-variable support metadata and exact FE row topology**, so an estimator can avoid constructing a global dense `N x K` design.

## Problem

After weighted HDFE projection, suppose observations are partitioned into disjoint FE row components `b = 1,...,B`. The numerical design is stored as local dense blocks `X_b`, but coefficient columns use global indices and may be shared across row components. The WLS problem is

\[
\min_{\beta}\sum_{b=1}^B
\|W_b^{1/2}(y_b-X_b\beta_{C_b})\|_2^2,
\]

where `C_b` is the global coefficient set active in component `b`.

A coefficient used in one component is *local*. A coefficient used in multiple components is a *border/shared* coefficient.

## Orthogonal local compression

For each component, order columns as local then shared and factor the weighted augmented matrix

\[
A_b = [W_b^{1/2}L_b,\;W_b^{1/2}S_b,\;W_b^{1/2}y_b]
    = Q_bR_b.
\]

Because `Q_b` is orthogonal, replacing `A_b` by `R_b` preserves the least-squares norm. In implementation, the local factorization is itself performed as a bounded-workspace TSQR reduction: row chunks are QR-factorized, their small `R` factors are stacked, and that stack is QR-factorized again.

Partition the resulting factor as

\[
R_b =
\begin{bmatrix}
R_{LL,b} & R_{LS,b} & r_{Ly,b}\\
0 & R_{SS,b} & r_{Sy,b}
\end{bmatrix}.
\]

When the exclusive local columns have full local rank, for any fixed shared coefficient vector the first block of equations can be satisfied by

\[
\beta_{L,b}=R_{LL,b}^{-1}
(r_{Ly,b}-R_{LS,b}\beta_{S,b}).
\]

Therefore only the trailing equations constrain the shared coefficients. The implementation embeds `R_{SS,b}` into global shared-column coordinates, stacks these small systems across components, solves the shared least-squares problem, and finally back-substitutes each component's local coefficients.

If an exclusive local block is numerically ambiguous, econhdfe does not force this elimination. It falls back to the more general compact-QR construction that embeds each local augmented `R_b` into global coefficient coordinates and solves the resulting compressed problem directly.

## Why this matters for HDFE estimators

The method separates three objects that a dense estimator would otherwise conflate:

1. FE projection is local to exact row components;
2. physical design storage contains only columns that can be nonzero in each component;
3. the coefficient solve remains globally correct through the small shared border system.

Thus a global control does not force the HDFE projector or the repeated PPML IRLS workspace back to an `N x K` dense representation.

## Numerical policy

Intermediate PPML iterations may still use block-accumulated normal equations when the existing `fast_solver` policy requests that approximation. Accurate IRLS iterations and the final WLS use the QR-based partitioned solver. The heterogeneous-specification fast path is enabled only when sample selection, separation, final WLS, covariance, DoF and reporting semantics are certified; otherwise the estimator falls back to the pooled dense path.

## Implementation map

- `econhdfe/compute/design_plan.py`: exact FE row partition and symbolic column support.
- `econhdfe/compute/block_design.py`: row-partitioned local-dense physical representation.
- `econhdfe/compute/partitioned_lstsq.py`: TSQR compression and block-angular WLS solve.
- `econhdfe/hdfe/block_projection.py`: component-local weighted FE projection.
- `econhdfe/models/ppml/heterogeneous.py`: PPML consumer for certified heterogeneous-specification fast paths.

## Validation boundary

The implementation must be compared against the established pooled dense path on coefficients, residuals, convergence, final WLS, robust covariance and clustered covariance, including clusters that span row components. Performance claims must include structural-analysis and design-compilation cost, not only the hot solver kernel.

## Hardening rules added after the initial prototype

The compute-layer module is intentionally stricter than a performance-only prototype.

### Layout invariants

`BlockDesign` operators require a true row partition: block rows must be in range, unique within each block, disjoint across blocks, and cover every observation exactly once. Column indices must be in range and unique within a block. Externally constructed layouts are validated lazily on first use and their row/column metadata is detached into read-only internal copies, preventing a caller from mutating an already-certified layout. Internal compiler/projector paths use a private trusted constructor after an exact certificate has already been established.

### Numerical stability gate

Local block-angular elimination is used only when the exclusive-column triangular factor is numerically well conditioned. The implementation uses a conservative `sqrt(machine epsilon)` singular-value gate rather than treating every formally full-rank local factor as safe. If local elimination or the shared border is numerically ambiguous, the solver does not silently accept the fast path.

The generic compact-QR fallback preserves the global least-squares objective, but a full-rank compact problem that remains extremely ill conditioned is explicitly rejected with a numerical error. The intended estimator-level behavior is then to use the pooled high-accuracy path or resolve near-collinearity before structured execution. Exact rank-deficient compact problems may still be solved by the standard minimum-norm least-squares convention.

### Bounded-memory border solve

For a small shared border, collecting the reduced border equations and solving them once is fastest. For a large border, the second stage switches to streaming TSQR so memory is bounded by a small retained `R` factor plus one incoming component. `PartitionedSolvePlan` reports both the collect-mode and streaming-mode workspace estimates; selection is controlled by an explicit border-memory budget rather than by model names or data-specific special cases.

### Bounded-memory fallback

The compact fallback is preallocated once from the certified topology. Earlier prototypes accumulated one global-width compact matrix per component and then called `vstack`, transiently holding almost two copies of the fallback matrix. The hardened implementation writes each local `R` directly into one preallocated compact matrix. A separate fallback-memory budget can reject the fallback before allocation if the estimated compact workspace is too large.

## Planner diagnostics

`plan_partitioned_wls()` reports topology and memory quantities before a high-accuracy solve:

- shared and exclusive coefficient widths;
- maximum component-local width;
- estimated number of border equations;
- collect-mode and streaming-border workspace;
- retained local back-substitution factors;
- compact-fallback rows and bytes;
- dense-equivalent bytes;
- algebraic dense-QR and partitioned-QR work proxies;
- shared-column fraction.

The work proxies are deliberately **not** a wall-clock performance guarantee. Dense LAPACK kernels can outperform many small structured QR operations when the shared border becomes dominant, even if a simple flop proxy still favors the structured representation. They are diagnostics for a future execution planner, not a hard-coded dispatch threshold.

## Stress-test boundary

Randomized parity testing covers varying block counts, local/shared widths, zero-weight observations, TSQR chunk sizes, and deliberately ill-conditioned local columns. In well-conditioned full-rank cases the hardened partitioned WLS matches pooled dense WLS to near machine precision. Deliberately ill-conditioned cases are rejected rather than returning a lower-accuracy structured solution.

A dedicated border-width benchmark (`benchmarks/ppml/bench_partitioned_border_scaling.py`) records the expected crossover: narrow borders benefit substantially from the block-angular representation, while a border that dominates the total coefficient set can make dense QR faster. This is precisely why the module exposes planner diagnostics and does not equate “certified partitionable” with “always execute partitioned.”
