# Identified categorical fixed-effect recovery

**Status:** Current fixed-effect recovery documentation  
**Scope:** Realized-sample identification, connected components, normalization, diagnostics, and recovery semantics.  
**Claim boundary:** Recovery and normalization do not create identification that is absent from the realized design.
High-dimensional fixed-effect estimators usually absorb FE because the coefficients are nuisance parameters. In several important empirical designs, however, the FE coefficients themselves are economic objects—for example worker and firm effects in AKM-style models or origin/destination value components used in a later structural stage.

`econhdfe.effects` is therefore a small post-estimation layer. It is intentionally not another estimator family.

## 1. Recovery problem

After the structural coefficients have been estimated, the additive FE contribution can be written as

\[
g = D\gamma,
\]

where `D` is the categorical indicator design and `gamma` stacks the FE-level coefficients.

Examples:

- linear OLS/IV adapter: the fitted FE contribution after removing `X beta`;
- PPML/IV-PPML adapter: the converged linear predictor after removing `X beta`, offset and any reported normalized constant.

The recovery layer receives `g`, the categorical FE groups, the realized estimation sample and optional recovery weights. It does not rerun the structural estimator.

## 2. Identification is sample-dependent

A specification that is identifiable in principle can fail in the realized estimation sample. Identification therefore belongs to

\[
D(\text{final sample}),
\]

not to the FE names alone.

Important causes include:

- insufficient worker-firm or origin-destination mobility;
- disconnected support;
- nested/redundant categorical partitions;
- user/sample restrictions that remove identifying links;
- recursive singleton pruning;
- PPML separation or other finite-MLE support removal.

The module runs identification diagnostics on the realized sample supplied by the estimator adapter.

## 3. Rank, nullity and ordinary normalization

The recovered FE system is not unique whenever

\[
\mathcal N(D) = \{v : Dv=0\}
\]

is nontrivial.

For a connected two-way additive system such as worker+firm, the familiar indeterminacy is

\[
\alpha_i + c,\qquad \psi_j-c.
\]

That is an ordinary normalization direction. It can be fixed by a reference level or a mean-zero restriction without changing fitted values or identified within-component contrasts.

For `K` categorical FE dimensions, each independent connected incidence component has at least `K-1` ordinary additive shift directions. The package compares this ordinary shift nullity with exact realized-sample rank/nullity. Additional null directions are reported as `extra_nullity`.

## 4. Normalization does not create identification

Supported reporting conventions include:

- deterministic canonical normalization;
- reference-level normalization;
- unweighted mean zero;
- observation-mass-weighted mean zero.

A normalization transforms one valid representative into another while preserving

\[
D\gamma.
\]

It cannot make a previously unidentified contrast economically meaningful.

For example, if two worker-firm components are disconnected, setting a global firm mean to zero does not identify the level difference between those components.

## 5. Independent-component salvage

A practical data problem is that one independent FE block may be unidentified while another is perfectly usable.

The recovery layer therefore diagnoses rank at independent-component granularity:

```text
component A: extra_nullity = 0  -> recover
component B: extra_nullity > 0  -> unavailable
component C: extra_nullity = 0  -> recover
```

For unidentified blocks:

- the FE levels remain in the result metadata;
- their reported coefficients are `NaN`;
- `identified=False` is recorded at level size;
- the diagnostic records the component and its extra nullity;
- normalization skips the unidentified block.

If every independent component is unidentified, strict recovery raises `FixedEffectIdentificationError`.

The package deliberately does **not** search for a maximal identified subspace inside one connected-but-rank-deficient component. That problem can require model-specific structural restrictions and is outside the default empirical workflow.

## 6. Singleton pruning

Recursive singleton pruning removes observations/levels that provide no usable support under the estimator's FE geometry.

For linear adapters, full raw categorical groups can be supplied even when the fitted result was estimated after singleton removal. The adapter reconstructs the same recursive keep mask using the shared HDFE singleton algorithm and cross-checks both the dropped count and final sample size before recovery.

Removed levels are not assigned artificial FE coefficients. Surviving identified components remain recoverable.

## 7. PPML separation and finite estimates

PPML has a second support problem beyond linear rank: some observations/levels may not belong to a finite-MLE sample because of separation.

Recovery therefore occurs **after** PPML's singleton/separation decisions. The diagnostic distinguishes:

- support that was already rank-deficient in the raw specification;
- support removed by singleton pruning;
- additional rank loss that appears only after PPML separation;
- terminal cases where no finite estimable sample remains.

Changing FE normalization cannot repair separation or create a finite PPML estimate.

## 8. Solver independence

The numerical decomposition used internally need not define the reporting normalization.

For example:

- the specialized two-way path may use a component anchor;
- generic recovery may obtain a minimum-norm LSMR solution.

Both can represent the same aggregate contribution `D gamma`. The result is therefore passed through a separate normalization layer so reporting semantics are not silently solver-dependent.

## 9. Storage contract

Recovery results are level-sized:

```text
worker levels + firm levels + ...
```

rather than observation-sized per FE dimension.

The result stores levels, coefficients, component IDs, level mass, identification masks and diagnostics. An observation-length FE contribution is not retained by default. This is important for matched employer-employee and origin-destination data where `N` may greatly exceed the number of FE levels.

## 10. Scope

Recovery is supported for **categorical/indicator intercept FE**.

In scope:

- worker;
- firm;
- city;
- year;
- origin/destination;
- pure categorical interactions such as firm-by-year treated as categorical partitions.

Not in scope:

- varying-slope recovery such as firm-specific age slopes;
- FE standard errors;
- AKM/KSS leave-out bias corrections;
- automatic structural restrictions for connected-but-rank-deficient systems;
- cross-component comparisons not identified by the data/model.

Those can be added only when their identification and inference contracts are explicit.

## 11. Advanced structural use

`strict_identification=False` exists for advanced workflows that deliberately need to inspect an unresolved numerical decomposition. Such coefficients are not marked identified and should not be interpreted as if a numerical normalization supplied missing economic identification.

A future structural-constraint interface may allow model-supplied restrictions. The default package will not invent them automatically.
