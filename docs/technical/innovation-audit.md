# 0.6.1 review status

> **Document status:** Current EconHDFE project technical documentation.


Current registry classes are `theorem_backed_framework` and `theorem_backed_application`. They describe mathematical support, **not confirmed novelty**. The historical claim discussion below is retained for traceability and is subordinate to this qualification. Full proof/implementation corrections and tests: [mathematical-review-0.6.1.md](mathematical-review-0.6.1.md).

# Package-wide technical-innovation audit

This document records the package mathematical contributions and their prior-art boundaries. **The 0.6.1 review does not certify historical originality or priority.** It deliberately separates mathematical or algorithmic contributions from high-quality engineering, upstream reproduction, compatibility work, and product/API design.

The audit is conservative. A feature is not called an innovation merely because it is difficult to implement, much faster than another package, or absent from one particular upstream implementation. To qualify as `technical_innovation`, the package must contain a nontrivial algorithmic, mathematical, inferential, or exact structural result whose claim can be stated and defended independently of Python engineering. Every such item must be registered in `innovation-registry.json` and must ship a formal `.tex` manuscript and compiled `.pdf`.

## Audited innovations

The following three contributions have theorem-backed artifacts. Their correctness and implementation have been reviewed under the assumptions stated in `mathematical-review-0.6.1.md`; none is advertised as independently established new mathematics.

| ID | Technical contribution | Formal manuscript | Claim boundary |
| --- | --- | --- | --- |
| `HDFE-EXACT-RANK-001` | Exact arbitrary-G categorical FE structural-rank / absorbed-DoF computational framework | `hdfe/exact-multiway-dof/` | The matrix-rank identity itself is established; novelty is limited to the exact general computational/certification framework. |
| `HDFE-NUMCORE-002` | Exact arbitrary-G residual-core reduction for multiway categorical HDFE projection | `hdfe/numerical-residual-core/` | Two-way degree-one pruning is prior art; the claim is the arbitrary-G hypergraph projection theorem and exact reconstruction/weight-validity conditions. |
| `HDFE-STRUCTURAL-DESIGN-003` | Exact partition-refinement / functional-dependency HDFE canonicalization and pre-materialization design reduction | `structural-design/` | Functional dependencies and dummy collinearity are prior art; the claim is the exact HDFE composition, dependency closure, recertification and joint-column-space reduction framework. |

The machine-readable registry is the source of truth for release validation:

```text
docs/technical/innovation-registry.json
```

## Full package classification

The following table records the major computational and inferential subsystems reviewed through 0.6.3. A classification of `established_method` or `engineering_optimization` is intentional and must not be rewritten as an originality claim without a new literature audit and formal manuscript.

| Subsystem / feature | Classification | Audit conclusion |
| --- | --- | --- |
| Exact multiway categorical FE structural rank / DoF | **theorem_backed_framework/application** | Registered as `HDFE-EXACT-RANK-001`; formal paper retained. |
| Arbitrary-G numerical residual-core peeling | **theorem_backed_framework/application** | Registered as `HDFE-NUMCORE-002`; formal paper retained. |
| Exact FE partition canonicalization + structural design reduction + dependency graph | **theorem_backed_framework/application** | Registered as `HDFE-STRUCTURAL-DESIGN-003`; formal paper retained. |
| Exact arithmetic backends (GF(2), SymPy, optional FLINT) | implementation of registered innovation | These are certification/backend choices inside exact-rank innovation #1, not separate technical claims. |
| Two-way HDFE specialized Schur/PCG path | established numerical adaptation | Schur complements, conjugate gradients and two-way FE reductions are established; package contribution is performance engineering. |
| Symmetric MAP, CG, LSMR/LSQR absorption | established_method | Standard iterative linear-algebra methods adapted to HDFE. |
| Adaptive MAP/CG route selection | engineering_optimization | Empirical planner/heuristic; no new convergence theorem is claimed. |
| Fused weighted multi-RHS projection | engineering_optimization | Kernel fusion and memory-bandwidth optimization, not a new estimator or theorem. |
| Pre-materialization row-partitioned design + block-angular/TSQR WLS | established numerical adaptation / engineering_optimization | Exact FE row-topology and symbolic support metadata are used to select a compact physical representation, while the shared-column solve uses established block-angular least-squares and TSQR/orthogonal-reduction ideas. This execution extension is not registered as a new numerical method. |
| Positive-weight topology reuse / buffer reuse / memory budgets | engineering_optimization | Execution-state and memory-management improvements. |
| Runtime thread planning, Numba/backend dispatch, GPU hooks | engineering_optimization | Hardware/runtime implementation. |
| Repeated-specification sessions and FE/design caches | engineering_optimization | Reuse of validated transformations; no new statistical object. |
| Data Layer projected I/O, compact encoding and persistent linear-workflow reuse | engineering_optimization | Reduces ingestion/representation/repeated-workflow cost while preserving estimator sample/specification semantics; not a new estimator or theorem. |
| Unified Execution Planner and automatic runtime thread calibration | engineering_optimization | Separates exact candidate legality from memory/representation/thread policy; no new econometric or convergence result is claimed. |
| Identified categorical FE recovery, normalization and component salvage | established_method / product integration | Recovers additive indicator FE from an estimated contribution with explicit rank/component/normalization semantics. The mathematical ingredients are established; the 0.6 integration is not registered as a new identification theorem. |
| Post-absorption Gram collinearity detection | established_method / engineering | Standard numerical-rank/Gram-matrix logic used with explicit package semantics. |
| Explicit omitted/reference-variable selection | product/API semantics | Important for event-study workflows but not a technical estimator contribution. |
| Group + individual FE support | compatibility_implementation | Reproduces established reghdfe-style multi-membership semantics; current DoF treatment is explicitly conservative. |
| OLS-HDFE | established_method | Standard least squares after FE projection. |
| Linear IV / 2SLS / LIML / k-class / GMM | established_method | Established econometric estimators and diagnostics. |
| Stock–Yogo / conventional linear weak-IV diagnostics | compatibility_implementation | Existing diagnostics; package provides integration, not a new statistic. |
| PPML-HDFE IRLS | established_method / compatibility_implementation | Established PPML-HDFE estimator family; package contribution is faithful, scalable implementation. |
| PPML FE/simplex/ReLU separation checks | established_method / compatibility_implementation | Separation detection follows the ppmlhdfe literature/implementation lineage. |
| PPML execution-resource propagation and reusable projectors | engineering_optimization | Correctness/engineering fixes to runtime policy; no new statistical method. |
| IV-PPML additive-moment estimator | established_method / compatibility_implementation | Implements an established IV-PPML construction; not claimed as invented by econhdfe. |
| IV-PPML SPJ bias correction | established_method / compatibility_implementation | Split-panel jackknife is established; package implementation is not a new bias-correction theory. |
| IV-PPML weak-identification/KP work carried in separate experimental modules | research/experimental, not core-package innovation | Any future novel nonlinear weak-IV statistic requires its own theory audit and manuscript before core registration. |
| CRV1 one-/multi-way clustered covariance | established_method | Standard cluster-robust covariance/inclusion-exclusion machinery. |
| Cluster diagnostics | established_method / product integration | Useful diagnostics assembled for applied work; no originality claim. |
| WCR11/WCU11 wild cluster bootstrap | established_method / compatibility_implementation | Established wild-cluster bootstrap inference; package adds tested implementation. |
| Wild bootstrap weight distributions / exact small-G enumeration | established_method | Standard bootstrap choices/enumeration, not new inference theory. |
| CRV3 prototype retained outside the public production path | established_method / experimental engineering | Cluster jackknife is established; current issue is scalable implementation, not novelty. |
| Generic resampling engine | engineering_optimization | Parallel/reproducible draw mechanics. |
| Structured error taxonomy and publication reporting | product engineering | Stability and reproducibility contracts, not econometric innovation. |
| Agent Skill, architecture visualization, release/compatibility gates | product/release engineering | Governance and tooling only. |

## Why several sophisticated components are not registered

### Solver sophistication is not automatically novelty

The package contains specialized HDFE solvers, adaptive routing, fused matrix projection and extensive memory planning. Most of this work is high-value engineering around established numerical primitives. The solver-side mathematical claim is the arbitrary-G **exact residual-core reduction theorem**, an elementary singleton-column elimination application; formalizing it does not by itself establish novelty. Its paper explicitly does not claim that degree-one pruning itself is new; graph-based HDFE work already discusses degree-one pruning in the two-way setting.

### Block-angular QR and TSQR are prior art

The row-partitioned execution path may look algorithmically distinctive because it combines HDFE row components, factor-variable support metadata, local dense storage and a small shared-coefficient solve. The numerical least-squares reduction itself is nevertheless established. Cox (1990) gives an orthogonal-transformation method for least squares with a block-angular observation matrix; standard numerical-linear-algebra references treat QR methods for block-angular least squares; and TSQR/communication-avoiding QR formalizes recursive row-block QR reduction for tall-skinny matrices. `econhdfe` therefore classifies local QR compression, border-variable elimination and the second-stage shared solve as established numerical adaptation. Their integration with the package's structural compiler is performance engineering unless a future, separately audited HDFE-specific theorem goes beyond those prior results.

### Functional dependency is prior art

Database and structure-aware learning literature has long used functional dependencies to simplify learning problems. Likewise, dummy-variable dependence under nested categories is elementary linear algebra. The registered structural-design claim is therefore narrow: an exact HDFE compilation framework that proves and composes partition-refinement certificates, recertifies after sample changes, propagates component dependencies through a directed graph (possibly cyclic before quotienting equivalent partitions), and reduces factor/factor-slope designs before materialization while preserving the combined column space.

### Econometric estimators retain upstream attribution

PPML-HDFE, IV-PPML, split-panel jackknife, conventional linear-IV diagnostics, clustered covariance and wild-cluster bootstrap are established methods. Reimplementing them in a faster shared HDFE runtime does not make the estimator itself new. Their technical documentation should focus on parity, numerical semantics, supported boundaries and validation rather than originality claims.

## Release policy

A new item may be added to `innovation-registry.json` only when all of the following exist in the same release:

1. a precise novelty statement and prior-art boundary;
2. a formal LaTeX manuscript with definitions, assumptions, propositions/theorems and proofs or a clearly delimited computational result;
3. the compiled PDF retained in the source and release bundle;
4. explicit implementation correspondence;
5. regression/correctness tests addressing the claimed result;
6. benchmark, validation or exact-oracle evidence appropriate to the claim;
7. a release-gate check proving that the paper artifacts are present and identical between source and bundle.

Technical manuscripts are **not** wheel payload. Runtime users should not pay installation-size cost for audit artifacts.

If a contribution is later shown to be established prior art, the registry must be corrected in a subsequent release. Historical manuscripts remain for provenance, but the current audit must no longer advertise novelty.
