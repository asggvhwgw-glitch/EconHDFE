# 0.7 Post-estimation design

This note records the implementation boundary for post-estimation work starting from
the 0.6.5 main branch.

## Design principles

Post-estimation must use the exact reported coefficient order, covariance matrix,
realized estimation sample, fixed-effect identification state, and normalization
chosen by the fitted model. It must not reconstruct these objects by guessing from
user input after estimation.

The result object should remain the public entry point. Estimator internals should
expose only immutable/read-only state needed for post-estimation.

## Phase 1: single-model coefficient inference

Implemented on this branch:

- one linear combination of reported coefficients;
- named coefficient weights as well as numeric vectors;
- nonzero null values;
- joint Wald tests with F or chi-square reference distributions;
- effective-rank handling for singular restriction covariance;
- explicit rejection of directions with no estimated sampling variance.

This layer depends only on params, vcov, names, confidence level, and residual DoF,
so it is shared safely by OLS and linear-IV RegressionResult objects without touching
the estimator or HDFE numerical path.

## Phase 2: prediction state (PRED-00)

Before out-of-sample predict(), add a frozen prediction-state contract containing:

1. active reported design-column order after user omission and collinearity handling;
2. enough factor-variable metadata to reproduce expansion and reference levels;
3. realized estimation-sample mapping;
4. categorical FE level maps and connected-component/identification metadata when
   FE contributions are requested;
5. normalization metadata for recovered FE coefficients;
6. model-specific link/response semantics.

Do not store the original DataFrame merely to make prediction convenient.

## Phase 3: prediction API (PRED-01)

Start with categorical fixed effects. The API should distinguish at least:

- linear index / link;
- response scale;
- beta-only prediction;
- prediction including identified FE contributions.

Unknown factor or FE levels must never be silently encoded as zero. Return NaN or
raise according to an explicit policy. Cross-component FE combinations that are not
identified must follow the same rule.

Varying-slope FE should be a separate extension because prediction requires both
level and slope-variable semantics.

## Phase 4: margins and nonlinear transformations (POST-01)

Margins/AME should be built on the prediction contract, not directly on raw params.
For models with absorbed FE, beta-only covariance is not a complete uncertainty
model for arbitrary level predictions. The API must therefore state whether
uncertainty is conditional on recovered FE, excludes FE uncertainty, or is not
available.

## Non-goals for the first 0.7 step

- no cross-model/suest-like tests without a joint covariance;
- no automatic independence assumption across separately fitted models;
- no prediction for unseen FE levels by setting their contribution to zero;
- no generic prediction standard errors that ignore absorbed-effect uncertainty;
- no mutable estimator objects exposed through RegressionResult.

## Validation

The first phase belongs in behavior tests because it is an inference semantic
contract. Prediction-state work should additionally have contract tests for column
ordering and failure semantics, plus numerical tests against explicit dummy-variable
fits for categorical FE.
