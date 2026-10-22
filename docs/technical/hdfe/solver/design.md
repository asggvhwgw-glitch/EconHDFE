# HDFE numerical solver design

The numerical solver is a distinct layer from structural DoF/rank. Its job is to compute the within transformation efficiently while preserving the column space requested by the estimator.

## Three representations

1. **Requested/inference topology**: the FE terms supplied by the user. This representation governs absorbed DoF, nesting, reporting, and compatibility semantics.
2. **Canonical numerical plan**: algebraically equivalent FE representation chosen for efficient execution.
3. **Transient numerical residual core**: optional row/level reduction used only during projection when degree-one structure proves some residual rows exactly zero.

Only layers 2 and 3 are solver optimizations. They must not rewrite layer 1.

## Main execution paths

- specialized two-way paths where their structural assumptions hold;
- generic MAP/CG absorption for arbitrary categorical FE systems;
- adaptive `acceleration="auto"` in optimized FE plans;
- weighted multi-RHS grouped projection for large pooled residualization;
- optional 3+ FE numerical core reduction for eligible pure-intercept NumPy systems.

## Numerical core reduction

For an intercept-only categorical FE level appearing in exactly one active observation, orthogonality to that dummy requires the residual at the corresponding row to be zero. Recursively peeling such rows can reduce the iterative system without changing the exact within result. Unlike structural-rank logic, this operation never deduplicates observations: each row carries its own RHS and weight.

The topology can be reused across strictly positive weight updates. Zero weights change the effective numerical graph and invalidate that reuse.

## Adaptive MAP/CG

`acceleration="auto"` performs real symmetric-MAP sweeps on the actual RHS block and estimates contraction from those iterates. Fast-contraction systems continue plain MAP; difficult systems switch to CG from the partially residualized state. Direct low-level `HDFEAbsorber` retains the conservative historical default `acceleration="cg"`; optimized `FEPlan` may select `auto` internally.

The large pooled public-estimator path uses the same two-sweep contraction
probe on its first RHS pool. It retains a maximum 64-sweep plain-MAP allowance
for fast-contracting problems, while difficult problems immediately continue
CG without reloading the original pool. Later pools use the first pool's
selected method. Probe, continued plain MAP and CG together respect the
configured iteration limit for that pool. This differs from the former
64-sweep trial followed by a restart from the original RHS.

The NumPy CG path reuses the symmetric-operator output buffer and fuses
solution/residual updates with Numba. CPU float64 blocks with at least 100,000
rows, at least two columns, and more than one active thread additionally use
parallel direction updates and fuse residual updates with their squared norm,
and the final projection difference with its inner product. Reductions use
16 fixed row chunks, independent of the thread mask, without fastmath.
The existing planner reserves 17 float64 values per column for these reductions.
Widths 6–10 and wider 11/16/22/32/64 blocks were checked against the original
serial path across group layouts and weights before removing the five-column
cutoff; this is empirical evidence, not a speed guarantee for every width.
Other blocks retain serial vector operations. Vector fusion changes reduction
order and can slightly change iteration counts; it does not add a preconditioner.

## Near-collinear dense OLS

Pure-intercept NumPy MAP OLS screens the scaled design for ill-conditioning.
A condition estimate above 10,000 triggers a bounded-row QR/SVD verification;
Gram eigenvalues alone never decide numerical rank. Full-rank difficult
designs use reversible symmetric scaling of the weak right-singular subspace during absorption, with an
effective absorption tolerance no larger than `1e-11`. Singular designs keep
the existing order-preserving omission policy. If near-collinearity appears
only after FE removal, the retained original columns are reprojected in the
within-informed basis. This fallback costs an additional absorption pass.

Well-conditioned directions are not whitened. OLS and its covariance are
computed in the conditioned basis; coefficients and the full covariance are
then mapped back together. The retained within columns are restored for
prediction and state consumers. No ridge penalty or
additional omission is introduced. `absorb_info["conditioning"]` records the
trigger stage, condition estimate and effective tolerance. This path currently
excludes block designs, two-way solvers, heterogeneous slopes, GPU and other
estimators; the CG safeguards below are shared by their applicable CG paths.

CG locks initially projection-orthogonal RHS columns at a small multiple of
machine roundoff. Nonpositive curvature or a step exceeding the remaining
energy budget triggers a fresh symmetric-operator residual and at most two
restarts, within the existing iteration budget. Failure after these checks
remains nonconvergence. A small recursive residual, FE orthogonality, or
Hestenes metric alone is not a universal coefficient-error bound: weak graph
connectivity and weak design directions can amplify projection error.

These are established orthogonal preconditioning and residual-replacement
techniques, not new mathematical or identification claims. Near-collinearity
still increases statistical uncertainty. Validation compares original-unit
coefficients and covariance with independent dense-FWL/analytic references;
performance comparisons must use the same numerical acceptance gates.

`OLSHDFESession` applies the same screening, original-column reprojection,
and stable-coordinate inference to each specification. Ordinary specifications
retain their batched, FE/sample-aware column cache. Conditioned projections
are specification-local: they do not replace shared within columns, and the
shared absorber's requested tolerance is restored even on failure. Repeated
conditioned fits may therefore reproject; completed persistent results can be
reused. Numerical/result cache ABIs invalidate results from before this repair.

Screening reuses the full-data raw and within Gram matrices required by the
omission checks. It retains column-max scaling and QR/SVD verification for
suspicious designs; it does not sample rows or relax the trigger. Each Gram
matrix is formed once per specification and reused only for that same sample,
weights and column order.

### Scale-aware stopping

The closeout numerical audit removed the absolute `recent < 1e-15` success
condition and absolute epsilon guards on CG inner products. Each RHS now uses
its initial weighted energy E: the remaining-potential denominator is bounded
below by eps * E, and Krylov residuals below eps² * E are treated as roundoff.
The latter is necessary to avoid amplifying projection noise after nearly exact
FE removal. Active directions require a positive finite curvature; nonfinite
norms or nonzero inputs whose initial norm underflows return nonconvergence
with `cg_breakdown`, rather than a zero-step success.

Auto probes and plain-MAP residualization work in per-column max-absolute
units and restore the original units in a `finally` block. This includes later
pools after the first pool chooses plain MAP. Two column reductions avoid an
N x RHS absolute-value temporary; normalization adds linear scans and O(RHS)
state, without another full matrix or an automatic retry. CG itself uses the
energy-relative test without normalizing its input.

`test_cg_scale_contract.py` retains small regressions for dense FWL, mixed RHS
units, auto/plain MAP and later pools, invalid initial energy and exception
restoration. Wide CG coverage reuses the existing dense/serial oracle with a
representative width above five. Exhaustive development matrices are kept
outside the source repository. These tests do not certify every floating-point
range or GPU execution. Explicit
low-level orthogonality-checkpoint helpers keep their documented mixed
absolute/relative error metric; they are not the default pooled execution path.

For standard OLS, `ExecutionConfig(profile="summary")` or `"full"` additionally
records `result.profile["stages"]`: data/FE preparation, design construction,
absorber setup, absorption, cluster encoding/collinearity, DoF, OLS/VCE, and
result construction. Standard pooled absorption adds
`result.profile["absorption"]["pools"]` with each pool's width, elapsed time,
iterations, stopping criterion and available solver-stage timings. CG timings
separate symmetric-operator applications from remaining algebra, allocation
and stopping checks. On the parallel fused path, the projection timing also
includes the fused inner product; the pool stages explicitly report
`cg_parallel_fused` and `cg_projection_includes_inner_product`. Compare total
CG/absorption time across these paths, not their projection subtotals as if
the boundaries were identical. First-call timings can include JIT compilation.

Reported estimator `iterations` remains the maximum final pool iteration
count, not the sum over all RHS pools. The first auto pool now includes its
probe in that count. The profile is optional and does not change sample,
inference topology or convergence tolerances. Specialized two-way and
heterogeneous-design solvers expose only the applicable timing detail;
fine-grained coverage of other estimators is not implied.

## Memory rule

Core reduction may allocate the reduced residual-core block, but `copy=False` must reuse the caller/workspace output buffer rather than materializing a second full `N x RHS` result. Memory planning is therefore based on the full workspace plus bounded reduced-core and grouped-projection scratch.

## Thread and projection execution

The absorber resolves its thread policy outside the iteration loop. Internal
indexed projections and MAP convergence scans inherit that thread context;
they do not reinterpret an explicit thread count as `auto`. Standalone
projection helpers still resolve their own policy. Workqueue guarding and
thread-mask restoration remain in place, including exception paths. Two-way
OLS factories configure the Schur absorber, and per-call overrides apply
without mutating its saved default.

Indexed projection uses scalar accumulators, parallel over groups. Float64
RHS widths 2 through 10 share each row/index/weight access across explicitly
unrolled scalar accumulators. Every larger width uses a bounded five-column
scalar block and a per-column tail of at most four columns. There is no upper
RHS-width cutoff for the float64 indexed projection acceleration. The existing
backend eligibility and workspace budget still apply.
Both paths preserve column-wise row order, reciprocal multiplication and
zero-denominator behavior without a temporary sum array. The same kernels
support certified contiguous groups and indirect indexes. Other dtypes and
single-column inputs retain the generic kernels. Dispatch uses only dtype,
RHS width and the existing topology certificate, with no dataset-specific rule.
CG vector fusion also has no upper column-width cutoff; its row-count, CPU
float64 and multi-thread eligibility remains in force. Neither acceleration
changes the automatic pool selection policy.
The experimental eight-column tile implementation was reverted after narrow
pool regressions. The same row order is retained for each projected column;
no fastmath or reduced precision is introduced.
A stable counting-sort index also certifies whether every group's row positions
form one contiguous interval. The certificate uses the first/last row and group
length, so arbitrary group-label order and empty levels are supported. Certified
indexes use direct row intervals with the same within-group arithmetic; other
indexes retain indirect row access. This is an O(G) initialization check with
no extra row/group arrays, not a sort or a dataset-specific execution rule.
Compiled indexed backends remain indexed for wide RHS blocks rather than
silently switching to a serial kernel at eight columns. Fused fallback
group-sum scratch is reused across sweeps and FE terms. A per-instance lock
protects reusable scratch during concurrent calls even on OpenMP/TBB.

Short final RHS pools reshape the used prefix of the reusable C-order buffer.
They remain contiguous without allocating another block or retaining the full
pool row stride. Previously returned views are valid only until the next load,
as before. The allocated capacity and memory planner charges do not change.

## Workspace-budget behavior

The pooled CPU plan charges resident FE solver state and estimated pool scratch
to `memory_budget_mb`. Both `cg` and `auto` reserve Krylov buffers. The full
input/design matrix is reported separately; this is not a total-process RSS
cap. Two-way coefficient buffers and lazy indexes, and numerical-core buffers,
receive conservative allowances. Multi-thread CG/auto plans conservatively
reserve another 17 float64 values per column for the parallel reduction's
16 partials and result, including when the selected block falls back to the
original implementation.

An explicit pool that does not fit raises `SpecificationError` before pool
allocation; it is not silently reduced. Automatic planning also fails if one
column cannot fit. The former implicit 64 MiB floor is removed. The result's
`absorb_info["execution_memory"]` distinguishes input bytes, resident solver
bytes, per-column workspace and estimated solver bytes.

Host available memory is read with psutil and combined with cgroup headroom
when present, with the existing 12% reserve. Standard dense OLS also checks
headroom before constructing the stacked response/design block. These are
point-in-time checks, not reservations: other processes, allocator/JIT costs,
input materialization and solver-specific transient storage can still affect
RSS. GPU device memory and LSMR/group-individual allocations are not certified
by this CPU categorical-CG inventory.
