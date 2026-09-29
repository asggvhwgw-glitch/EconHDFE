# Heterogeneous-specification optimization

**Status:** Current internal engineering documentation  
**Scope:** Exact physical-representation optimization for interaction-rich and heterogeneous-coefficient specifications.  
**Claim boundary:** The layer preserves the requested economic model and does not define a new estimator or infer unrequested heterogeneity.
## Purpose

This is an internal performance layer for empirical specifications that request many explicit heterogeneous coefficients, especially factor interactions and group-specific slopes. Typical examples include event-study expansions, cohort-by-event-time terms, industry-specific slopes, region-specific policy effects, and related interaction-rich specifications.

The layer does **not** define a new estimator and does not infer economic heterogeneity that the user did not request. Its purpose is to avoid the performance cliff created when a compact economic specification is mechanically expanded into a very wide dense `N x K` matrix containing large regions of structural zeros.

The economic model is unchanged. The optimization asks only whether the requested coefficient support and FE topology admit a cheaper exact physical representation.

## Execution contract

The internal sequence is:

```text
factor / interaction specification
        -> symbolic support compiler
        -> FE row-topology certificate
        -> storage / execution planner
        -> dense or row-partitioned design
        -> established estimator
```

A certificate answers **whether** a structured representation is exact. The planner separately decides whether the representation is worth using. A certified partition is therefore never an execution mandate.

The common accelerated cases are intentionally narrow:

1. factor-by-continuous group-specific slopes;
2. event-study / cohort-by-event-time style mutually supported terms;
3. the same structures plus a small set of globally shared controls.

Arbitrary sparse matrices, highly overlapping coefficient-support graphs, or unusually wide shared borders are not targets of this layer.

## Model integration

The same compute-layer contract is consumed by the main estimator families:

- OLS-HDFE: block/local FE projection, partitioned high-accuracy WLS, robust/cluster score aggregation;
- linear IV-HDFE: block/local FE projection and k-class/2SLS/LIML sufficient-statistic solve; established weak-IV diagnostics are retained, currently with a one-time dense diagnostic materialization;
- PPML-HDFE: structured design, component FE projection, global IRLS convergence, partitioned final WLS, robust/cluster inference;
- IV-PPML-HDFE: structured role designs, component FE projection, global IV-IRLS state, block sufficient-statistic 2SLS, robust/cluster inference.

This integration deliberately does not replace the established dense paths.

## Conservative fallback boundaries

The structured path is refused or abandoned when its correctness or benefit is not certified. Important examples include:

- ordinary narrow continuous designs for which dense storage is already efficient;
- incompatible role partitions in IV / IV-PPML;
- group-plus-individual FE modes not yet covered by the row-partition certificate;
- absorbed heterogeneous FE slopes whose topology is not represented by the current categorical row planner;
- linear-IV GMM2S and ordered-score covariance estimators (HAC / Driscoll-Kraay) not yet block-native;
- shared coefficient columns when PPML/IV-PPML requests simplex or ReLU separation, because those separation problems globally couple the shared coefficients;
- numerical rank or local-elimination cases that fail the conservative conditioning gate;
- planner estimates indicating too little structural-zero saving or an excessively dominant shared border.

In all such cases the original estimator implementation remains the correctness reference.

## Separation boundary

FE-only separation decomposes over certified disconnected FE row components. `mu` separation remains part of the global IRLS state. By contrast, simplex and ReLU separation generally couple any coefficient shared across row components. The implementation therefore does not run component-local simplex/ReLU when shared coefficients are present; it falls back to the pooled path instead.

This restriction is intentional. It avoids turning a performance optimization into a change in the finite-estimate identification problem.

## Numerical methods and originality boundary

The shared-coefficient least-squares reduction uses established block-angular QR / TSQR ideas. It is classified as an established numerical adaptation / engineering optimization, not a new numerical estimator. The package innovation registry therefore remains unchanged.

The separately registered structural-design innovation has a narrower claim concerning exact HDFE structural compilation and pre-materialization reduction. This document does not expand that claim.

## Design principle

The intended rule is:

> Recognize common explicit coefficient heterogeneity, remove calculations that are provably structural zeros, and otherwise get out of the estimator's way.

The module is not intended to become a general-purpose sparse linear-algebra framework.
