# Performance architecture: removing repeated work before making kernels faster

> **Status:** Current project technical documentation.  
> **Scope:** Performance architecture, workload decomposition, reuse, and execution policy.  
> **Boundary:** Benchmark evidence is workload-specific and does not change econometric semantics.


`econhdfe` is designed for empirical workloads in which data movement, representation and repeated transformations can cost more than the final coefficient solve. The package's performance strategy is therefore broader than "use a faster HDFE algorithm."

This document describes where time and memory are spent, which layer owns each optimization, and what the benchmark evidence does—and does not—support.

## 1. The large-data cost decomposition

For a regression with high-dimensional FE, total runtime can be decomposed conceptually as

\[
T \approx T_{io} + T_{encode} + T_{design} + T_{FE} + T_{solve} + T_{infer} + T_{repeat}.
\]

A package that only optimizes `T_solve` can still perform poorly when:

- a wide CSV/Stata file is reread for every specification;
- string identifiers are repeatedly factorized;
- a sparse-by-structure interaction is expanded into a dense matrix;
- the same FE geometry is projected again for each table column;
- PPML rebuilds weighted projection state unnecessarily;
- parallel BLAS/Numba layers oversubscribe the machine;
- a nominally "optimized" path has more setup cost than the dense path.

`econhdfe` assigns these costs to separate modules so that each can be optimized without changing estimator semantics.

## 2. Data Layer: reduce bytes before estimation

The Data Layer first asks which raw columns the specification actually needs. For supported file-backed workflows it can project those columns rather than materializing the entire source table.

Identifier-only FE/cluster columns can then be stored as compact codes. Columns that are also used as explicit numeric/factor regressors retain their original value semantics; encoding is role-aware rather than a blanket conversion.

For repeated OLS/linear-IV workflows, an immutable encoded dataset can be reused by a session. This changes the cost model from "read and encode everything again" toward "materialize only newly needed state."

The optimization is intentionally conservative:

- missing-value/sample semantics remain estimator-controlled;
- cache state is keyed to source/specification/sample signatures;
- deleting a persistent cache removes reuse, not econometric information;
- fully out-of-core PPML/HDFE solving is not claimed.

## 3. Structural compilation: avoid creating work that is exactly redundant

Before numerical linear algebra, the package uses exact categorical structure to detect cases such as:

- one FE partition refining another;
- factor columns implied by existing FE/design structure;
- interaction dependencies propagated through a dependency DAG;
- supported event-study/group-specific-slope support that is local by row component.

This can reduce both explicit column count and physical payload before a dense matrix is created.

The important distinction is:

```text
mathematical reduction: preserve column space exactly
physical representation: choose how to store/operate on that space
```

The first requires a certificate. The second is a planner decision.

## 4. Structured heterogeneous designs

Interaction-rich specifications can be pathological for dense storage even when the economic model is simple. Suppose each of `G` groups has local coefficients that are zero outside its own observations. Dense expansion stores all of those structural zeros.

The row-partitioned path instead keeps:

- local coefficient blocks by certified row component;
- a small shared/global border when present;
- enough metadata to perform WLS/IV/IRLS and robust/cluster score aggregation without rebuilding the full dense design.

A development microbenchmark in `benchmarks/heterogeneous_spec_models_96k.json` uses 96,000 observations, 12 groups and 8 local slopes per group:

| Quantity | Dense | Structured |
|---|---:|---:|
| Design payload | 75.3 MB | 7.7 MB |
| OLS time | 1.145 s | 0.109 s |
| Linear-IV time | 5.009 s | 3.851 s |
| PPML time | 2.331 s | 0.363 s |
| IV-PPML time | 0.396 s | 0.191 s |

Coefficient differences relative to the package's dense path are at machine precision in that benchmark.

This evidence is intentionally narrow. It demonstrates the value of a certified structured representation on a workload with many structural zeros. It is not a general comparison to `reghdfe`, `ppmlhdfe`, `fixest`, Stata, R, or every dense regression.

## 5. HDFE projection: exact structural reduction plus established solvers

The HDFE runtime combines established iterative/direct numerical methods with exact package-specific structural reductions.

### Two-way FE

A specialized Schur/PCG path exploits the bipartite structure of two categorical FE partitions.

### Generic multiway FE

MAP/CG/LSMR-style paths operate on compact group codes and grouped projection kernels rather than explicit dummy matrices.

### Numerical residual-core reduction

For eligible 3+ pure-intercept systems, exact recursive degree-one elimination can shrink the observation set that must enter the numerical projection solve. Peeling is exact under its stated topology/weight conditions; the remaining core is solved with established numerical methods.

This is different from exact structural rank/DoF, which is a column-space/inference object.

## 6. Weighted iterative models: reuse topology, not invalid state

PPML repeatedly changes weights. The HDFE layer therefore separates topology that remains valid under strictly positive weight updates from numerical quantities that must be recomputed.

The optimized PPML path can reuse:

- FE encoding/topology;
- validated execution resources;
- work buffers and weighted projection infrastructure.

A zero weight can change the effective numerical graph, so optimizations that require positive-weight topology are disabled when their validity condition fails.

PPML warm starts are similarly conservative: a previous converged predictor can initialize a nearby specification, but the new fit still solves the new estimating equations to the requested tolerance.

## 7. Repeated specifications: cache exact transformations, not results by analogy

Empirical work commonly changes:

- the outcome;
- one or two controls;
- an FE set;
- a regression-table column while most inputs stay fixed.

Linear HDFE geometry makes some transformations exactly reusable. Sessions therefore cache by FE/sample geometry and can residualize unions of newly requested columns in one multi-RHS operation.

The cache boundary is the crucial part of the design:

- changing FE sets does not reuse transforms from another FE space;
- changing the effective sample invalidates sample-dependent structure;
- IV diagnostics are recomputed by specification;
- PPML/IV-PPML weighted states are not treated as fixed linear geometry.

Persistent caches are an execution optimization, not a second source of truth.

## 8. Execution Planner: avoid optimizing the wrong path

An exact fast path can still lose on setup overhead. The planner therefore treats candidate legality and candidate profitability separately.

A typical decision is:

```text
1. certify exact candidate representations/solvers
2. estimate payload, passes, memory and resource headroom
3. choose dense / block / threads / memory policy
4. expose the chosen plan for diagnostics
```

This design was strengthened after same-host validation found a complex event-study case where topology analysis itself cost more than the eventual dense solve. The fix was a deliberately narrow dense fast path for small surviving designs—not a blanket bypass of the structured planner.

## 9. Automatic threading

Parallelism is useful only above an overhead threshold. `threads="auto"` therefore keeps small HDFE jobs single-threaded and calibrates only sufficiently large workloads.

The calibration:

- uses synthetic memory-bound work, not user observations;
- is keyed to an anonymous execution fingerprint;
- selects the smallest thread count within a tolerance of the best measured median;
- can be bypassed with an explicit thread count;
- does not change model semantics.

Planner feedback reports are privacy-minimized and never uploaded automatically.

## 10. FE recovery cost

The v0.6 identified-FE recovery layer is post-estimation and level-sized. It does not retain an `N`-length contribution array per FE dimension.

Development smokes include:

- 200,000-row AKM-style linear design: 3,640 recursively pruned singleton observations; 50,457 recovered worker+firm level coefficients; median recovery about 0.110 s; max reconstruction error about `5.3e-11`;
- 40,000-row origin-destination PPML design: 1,500 recovered levels; median recovery about 0.015 s; reconstruction error about `2.3e-11`;
- 1,000 disconnected 3-way blocks with one unidentified block: component localization preserves the 999 identified blocks and marks only the bad block unavailable.

These are same-host development measurements.

## 11. How to read repository benchmarks

Every useful benchmark should answer four questions:

1. **What workload?** `N`, FE dimensions/cardinalities, explicit columns, interactions, weights, clustering, repeated passes.
2. **What baseline?** Prior econhdfe version, internal dense/reference path, or an external package/version.
3. **What parity check?** Coefficients, SE/VCOV, sample, DoF, iterations, separation/omission metadata as applicable.
4. **What claim is justified?** A local regression gate, a microbenchmark of one optimization, or a portable external comparison.

Repository-local timings are not automatically claims against Stata/R packages. The strongest external comparisons should use the external-validation harness and report package versions, hardware and model semantics explicitly.

## 12. What is not being claimed

The current project does **not** claim that:

- every regression is faster than every upstream implementation;
- all data larger than RAM can be estimated out of core;
- automatic threading always beats every explicit thread count;
- a structured-design microbenchmark generalizes to ordinary narrow regressions;
- local Python parity tests replace licensed Stata/upstream external certification;
- engineering optimizations such as caching, block QR, threading or planning are new econometric methods.

The performance objective is narrower and more defensible:

> eliminate provably unnecessary work, reuse transformations only when their econometric geometry is unchanged, and select execution policies that respect the actual data shape and runtime resources.
