# Technical overview

> **Status:** Current project technical documentation.  
> **Scope:** Package-wide econometric, mathematical, and execution contracts.  
> **Boundary:** Architecture and semantics; version-specific evidence lives in validation/release records.


This document explains the technical contract of `econhdfe` without conflating three different things:

1. the **econometric model** the researcher requests;
2. the **mathematical transformations** that are exactly equivalent to that model;
3. the **physical execution plan** used to evaluate those transformations on a particular dataset and machine.

That separation is the central architectural rule of the package.

## 1. Econometric model families

The current estimator families share additive high-dimensional fixed effects but retain separate estimating equations.

### OLS-HDFE

For explicit regressors `X` and fixed-effect design `D`,

\[
y = X\beta + D\gamma + \varepsilon.
\]

The structural coefficients can be estimated from the Frisch-Waugh-Lovell problem

\[
\hat\beta = (X'M_DX)^{-1}X'M_Dy,
\qquad
M_D = I - P_D,
\]

without materializing the dummy matrix `D`.

### Linear IV-HDFE

The HDFE transformation is shared with OLS, while the estimator-specific layer owns instrument roles, k-class/2SLS/LIML/GMM calculations and weak-identification diagnostics. Fixed effects are nuisance controls unless explicitly recovered after estimation.

### PPML-HDFE

The conditional mean is

\[
\mu_i = \exp(x_i'\beta + d_i'\gamma + o_i),
\]

with optional offset `o_i`. IRLS repeatedly changes the positive weights, so the numerical HDFE projector must support a dynamic weighted lifecycle. FE/simplex/ReLU separation remains part of the PPML identification/finite-estimate problem and is not an execution-planner decision.

### IV-PPML-HDFE

IV-PPML uses its own additive moment condition and an iteratively reweighted IV solve. It reuses the weighted HDFE projector and common IV primitives but remains a distinct estimator family.

## 2. Shared HDFE representation

Categorical fixed effects are represented by compact integer group codes rather than explicit dummy matrices. The shared HDFE layer owns:

- FE specification and interaction coding;
- exact canonicalization of certifiably redundant categorical partitions;
- numerical absorption/projection;
- exact or conservative absorbed DoF accounting;
- specialized two-way Schur/PCG and generic multiway iterative paths;
- topology reuse under valid positive-weight updates;
- singleton handling and requested-vs-effective FE metadata.

The numerical solver is not allowed to redefine the requested model. Solver canonicalization is therefore kept separate from inference topology and requested-design DoF.

## 3. Exact structural compilation before materialization

A large empirical specification can be expensive even before HDFE projection. Factor interactions, event-study expansions and group-specific slopes may mechanically generate a wide `N x K` dense matrix whose useful support is highly structured.

The design pipeline separates:

```text
requested specification
      -> symbolic terms
      -> exact structural certificates
      -> reduced / structured representation
      -> numerical estimator
```

Exact partition-refinement and functional-dependency certificates may remove columns or FE partitions only when the combined design column space is preserved. Certificates are recertified when the relevant sample changes.

For supported heterogeneous-coefficient patterns, row support can also justify a block/local physical representation. That is an execution representation, not a different regression.

## 4. Numerical HDFE projection

The package uses established numerical primitives—MAP, CG, LSMR/LSQR, Schur complements, grouped projection kernels—but adds package-specific exact structure where formally registered.

Two ideas should not be confused:

- `hdfe/rank.py` studies structural column rank/absorbed DoF;
- numerical residual-core reduction studies which observation rows must actually participate in a multiway projection solve.

The latter does **not** deduplicate observations: each observation has its own RHS and weight.

## 5. Data Layer

For large empirical datasets, the cost of getting clean arrays into the estimator can rival the final solve. The Data Layer therefore owns:

- required-column projection from supported sources;
- compact encoding of identifier-only FE/cluster columns;
- estimation-sample bookkeeping;
- immutable encoded datasets for repeated linear workflows;
- safe reuse of source columns and within-transformed columns;
- optional persistent cache validation.

The Data Layer is intentionally below estimator semantics. It does not decide which variables are endogenous, which FE should be absorbed, or which covariance estimator should be used.

Current boundary: file-backed OLS/linear-IV workflows can exploit source projection and persistent reuse. Fully out-of-core HDFE/PPML solving is not claimed.

## 6. Execution Planner

The planner uses a three-stage contract:

```text
certificate -> cost/resource model -> execution policy
```

A certificate answers whether a candidate execution path is exact for the requested specification. Cost/resource logic then considers memory, representation size, repeated passes, runtime headroom and thread behavior. Only then is an execution policy selected.

This matters because an exact optimization can still be slower on a small problem. Conversely, a compact representation can be essential on an interaction-rich large problem.

The planner cannot change:

- the estimation sample;
- regressors, FE, instruments or weights;
- separation semantics;
- inference choice;
- convergence mathematics.

`threads="auto"` follows the same rule. It may benchmark a small synthetic memory-bound calibration on the current runtime for sufficiently large HDFE jobs, but it never consumes user observations.

## 7. Repeated empirical workflows

Repeated specification work is treated as an execution lifecycle rather than a different estimator.

`OLSHDFESession` and `IVHDFESession` can reuse:

- encoded source columns;
- FE encodings/topology for the same requested FE combination;
- singleton samples;
- DoF metadata;
- within-transformed columns.

Cache keys include the relevant sample/FE/design geometry. Changing FE sets creates separate entries. Linear-IV diagnostics remain specification-specific.

PPML and IV-PPML are not given the same fixed-geometry cache semantics because their IRLS weights change. PPML warm starts reuse a starting predictor only; convergence is still to the current specification's equations and tolerances.

## 8. Identified categorical fixed-effect recovery

FE recovery is a post-estimation module because fixed effects are nuisance parameters in many models but primary research objects in others.

The recovery problem is

\[
g = D\gamma,
\]

where `g` is the estimated aggregate FE contribution supplied by an estimator adapter.

The module distinguishes:

- ordinary additive shift indeterminacy;
- disconnected components;
- extra data/specification rank deficiency;
- support changes created by singleton pruning or PPML separation;
- reporting normalization.

Normalization chooses a representative of an identified equivalence class; it does not create identifying variation. Fully identified independent components are recoverable even if another disconnected block is not. Unidentified block coefficients are reported as unavailable rather than filled by an arbitrary solver normalization.

See [`identified-fixed-effects.md`](identified-fixed-effects.md).

## 9. Inference ownership

Inference is deliberately separated from point-estimation numerics.

- standard sandwich covariance, including multiway CRV1, is shared compute infrastructure;
- cluster diagnostics and wild-cluster procedures live in inference/resampling layers;
- model-specific diagnostics remain with their model family;
- performance optimizations may aggregate sufficient statistics, but they may not silently replace an inference method with a different one.

## 10. Technical-innovation boundary

Implementation difficulty is not treated as evidence of originality. The package currently registers exactly three entries, classified as `theorem_backed_framework` / `theorem_backed_application`:

1. exact arbitrary-G categorical FE structural rank / absorbed DoF;
2. exact arbitrary-G residual-core reduction for categorical HDFE projection;
3. exact partition-refinement HDFE structural design reduction.

Each has a formal manuscript and explicit prior-art boundary, and the registry records mathematical support rather than proven novelty (`originality_status: not_independently_established`; see the 0.6.1 mathematical review). PPML-HDFE, IV-PPML, Schur/CG/LSMR, clustered covariance, wild bootstrap, TSQR/block-angular least squares, caching, threading, Data Layer and Execution Planner work are documented as established methods or engineering unless separately audited and registered.

See [`innovation-audit.md`](innovation-audit.md).

## 11. Correctness-first fallback policy

Fast paths are optional. The intended rule is:

> use exact structure when certified and worthwhile; otherwise use the established pooled/dense/reference implementation.

Representative fallback triggers include unsupported FE modes, incompatible structured role partitions, separation problems that globally couple shared coefficients, poor local conditioning, memory/resource policy, and cases where dense execution is already cheaper.

This policy is why benchmark results must be read by workload rather than as a package-wide constant speedup.

## 12. Validation contract

A release is expected to preserve several independent contracts:

- estimator/regression tests;
- statistical parity and result metadata;
- public API compatibility snapshots;
- architecture dependency/freshness checks;
- technical-innovation registry/manuscript presence;
- clean wheel installation and smoke tests;
- source-archive retest;
- benchmark evidence with explicit baselines;
- release-bundle verification and hashes.

External licensed/upstream certification is kept separate from repository-local validation so that local test success is not overstated as external parity.
