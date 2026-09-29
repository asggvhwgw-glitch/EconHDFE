# econhdfe heterogeneous-specification development closeout

Base: **econhdfe v0.4.10.1**. This is an internal development closeout, not a release. The package version and public estimator signatures are unchanged.

## Scope

This closeout deliberately stops expanding the topology handled by the structural engine. Its role is now narrow and economics-facing: recognize common **explicit heterogeneous-coefficient / interaction-rich specifications**, avoid materializing or repeatedly processing proven structural zeros, and otherwise fall back to the established dense estimator path.

Typical intended cases are factor-by-continuous group-specific slopes, event-study/cohort interactions, and the same specifications with a relatively small set of globally shared controls. It is not a general sparse-matrix framework and it does not attempt to optimize arbitrary overlapping coefficient support.

## Model integration

### OLS

The DataFrame HDFE OLS path can consume a compiled heterogeneous design, project fixed effects by certified row component, solve the partitioned weighted least-squares problem, and compute robust/cluster covariance from block scores. Ordinary specifications remain on the original path.

### Linear IV

2SLS/LIML/k-class can use row-compatible heterogeneous designs for exogenous regressors, endogenous regressors and excluded instruments. FE projection and the main IV solve use block sufficient statistics. Robust/cluster covariance is block-native.

The established weak-IV/first-stage diagnostic implementation is intentionally retained and currently materializes the role designs once. This is a known stop line: the main estimator is connected, but linear-IV acceleration is materially smaller than the other model families and peak-memory elimination is incomplete. We do not duplicate or rewrite the weak-IV diagnostic stack merely to force a larger benchmark speedup.

GMM2S, HAC/Driscoll-Kraay and group+individual FE remain on the existing specialized/dense paths.

### PPML-HDFE

The former private block prototype has been consolidated into an internal heterogeneous-specification path. It preserves global `mu`, `eta`, deviance, step-halving and convergence semantics while using component-local HDFE projection and structured WLS. Final robust/cluster inference remains pooled in score space.

When a design has coefficients shared across row components and the requested separation method requires global simplex/ReLU logic, the optimizer explicitly falls back to the established dense PPML path. FE/mu-only separation remains compatible with the structured path.

### IV-PPML

Common heterogeneous exogenous/endogenous/instrument specifications can be compiled before materialization and used by structured IRLS/2SLS, standardization and robust/multi-way-cluster inference. Global convergence semantics are unchanged. Unsafe separation/topology cases fall back to dense execution.

## Shared compute-layer contract

The model integrations consume the same internal primitives rather than carrying estimator-specific block parsers:

- symbolic heterogeneous-specification compilation before dense expansion;
- certified FE row partition / coefficient support metadata;
- `BlockDesign` physical storage;
- component FE projection;
- block-aware Gram/cross-products and column selection;
- partitioned high-accuracy WLS;
- block weighted 2SLS;
- pooled robust/multi-way-cluster score aggregation;
- exact dense fallback when the certificate, cost decision or numerical-stability gate fails.

The intended architecture is therefore **specification optimization upstream of the estimator**, not separate heterogeneous versions of OLS/IV/PPML.

## Validation

Final local checks for this closeout:

- **419 tests passed**;
- compatibility gate: **0 public-contract changes**;
- technical-innovation validation: **PASS, 3 registered innovations (unchanged)**;
- architecture map regenerated and current.

The dedicated integration tests cover OLS, linear-IV 2SLS/LIML with robust and cluster VCE, PPML, and IV-PPML including standardization and cluster identifiers spanning structural components.

## Development benchmark

`benchmarks/heterogeneous_spec_models.py` measures full design-build + estimation time for the same interaction-rich specification under the dense reference path and the structured path. These are development microbenchmarks, not release performance claims.

### 24,000 observations; 6 groups; 6 local slopes/group; 38 total columns

- dense design payload: 7.30 MB;
- planned block payload: 1.54 MB;
- OLS: about **3.44x**;
- linear IV: about **1.11x**;
- PPML: about **2.26x**;
- IV-PPML: about **2.13x**;
- coefficient differences remain at machine precision.

### 96,000 observations; 12 groups; 8 local slopes/group; 98 total columns

- dense design payload: 75.26 MB;
- planned block payload: 7.68 MB;
- OLS: about **10.51x**;
- linear IV: about **1.30x**;
- PPML: about **6.42x**;
- IV-PPML: about **2.07x**;
- coefficient differences remain around `1e-15`.

Linear IV is deliberately not presented as a headline acceleration result because the mature weak-IV diagnostic layer currently dominates enough work to reduce the benefit of structured main estimation.

## Originality boundary

No new technical-innovation claim is made. Block-angular least squares, QR reduction and TSQR are established numerical methods. This closeout is an engineering integration of the existing structural-design machinery with multiple estimators. The package innovation registry remains unchanged at three registered items.

## Stop line

This branch is now considered feature-complete for the current heterogeneous-specification objective. Further work on this layer should be limited to bug fixes, real-hardware validation and planner calibration. It should not continue expanding to exotic coefficient-support topologies unless a concrete empirical workload demonstrates a material need.
