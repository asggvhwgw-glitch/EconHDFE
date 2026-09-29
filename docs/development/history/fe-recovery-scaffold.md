# Identified Fixed-Effect Recovery scaffold

Status: integrated v0.6.0 recovery module; exposed as the separate `econhdfe.effects` post-estimation surface.

## Purpose

`econhdfe.effects` is a thin estimator-agnostic layer for recovering additive categorical fixed effects from an already identified contribution

\[
    g = D\gamma.
\]

The estimator remains responsible for producing `g`. The recovery layer is responsible for level mapping, identification metadata, decomposition, and reporting normalization.

Typical adapters are:

- linear OLS/IV: the absorbed contribution after estimating the structural coefficients;
- PPML/IV-PPML: `eta - X beta - offset` after convergence.

No estimator code is changed by this scaffold.

## Current scope

Implemented:

- categorical intercept fixed effects only;
- raw level -> dense code -> recovered coefficient mapping;
- exact categorical rank and nullity through the existing HDFE rank engine;
- incidence-connected components for arbitrary K-way categorical FE specifications;
- `structural_shift_nullity = components * (K - 1)`;
- `extra_nullity` diagnostics for dependencies not explained by ordinary component shifts;
- two-way Schur recovery using the existing two-way numerical core;
- generic K-way matrix-free LSMR recovery;
- level-sized result storage (no observation-sized FE contribution retained);
- canonical, reference, mean-zero, and observation-weighted mean-zero reporting normalizations;
- zero-cost post-estimation re-normalization along ordinary component-shift directions;
- solver-parity tests between Schur and LSMR after canonical normalization;
- thin post-estimation adapters for OLS/linear-IV and PPML/IV-PPML;
- PPML recovery using the final IRLS mass `w * mu`;
- application-level validation with AKM-style worker-firm OLS and Artuç–McLaren-style origin-destination PPML.

Deliberately not implemented yet:

- heterogeneous/varying-slope FE recovery;
- a general sparse basis for `null(D)` when `extra_nullity > 0`;
- FE standard errors or leave-out / AKM bias corrections;
- cross-component economic comparisons that are not identified by the model.

## Identification contract

For K FE dimensions, every incidence-connected component has at least `K-1` additive shift directions. The existing exact rank engine gives

\[
q = \dim N(D) = L - \operatorname{rank}(D).
\]

The scaffold reports

\[
q_{shift} = C(K-1),
\qquad
q_{extra} = q-q_{shift}.
\]

When `q_extra == 0`, component-wise reference or mean-zero restrictions fully normalize the ordinary additive indeterminacy. When `q_extra > 0`, recovery localizes the failure to independent incidence components. Components with no additional null direction are recovered and normalized normally; levels in components with `extra_nullity > 0` are retained but reported as unavailable (`NaN`). If every component is unidentified, strict recovery raises `FixedEffectIdentificationError`. Advanced structural workflows may set `strict_identification=False` to inspect an unresolved numerical decomposition, but those coefficients are explicitly not marked identified.

Normalization never creates identification.

## Scope boundary

The recovery contract is intentionally limited to **indicator/categorical intercept FE**. Pure categorical interactions such as `firm#year` are in scope because they are still indicator partitions. Continuous/varying-slope terms such as `firm#c.age` remain estimable by the HDFE engine but are a recovery non-goal unless a later literature-driven use case justifies the extra identification machinery.

## Public development API

```python
from econhdfe.effects import recover_fixed_effects, NormalizationSpec

res = recover_fixed_effects(
    target,
    [worker_id, firm_id],
    names=["worker", "firm"],
    normalization=NormalizationSpec("weighted_mean_zero", baseline="worker"),
)

firm = res.term("firm")
firm.levels
firm.coefficients
firm.component

res_ref = res.renormalize(
    NormalizationSpec("reference", baseline="worker", references={"firm": reference_firm})
)
```

`baseline` is the FE dimension that absorbs compensating shifts. With `baseline="worker"`, normalizing firm effects leaves fitted FE contributions unchanged by shifting worker effects in the opposite direction.

## Initial performance smoke

On the current validation host, a synthetic connected AKM design with 200,000 observations, 50,000 worker levels, and 5,000 firm levels recovered 55,000 level coefficients with the two-way Schur path in roughly 0.18 seconds. Reconstruction error was below `8e-11`. This is a development smoke result, not a cross-hardware performance claim.

## Adapter contract

The adapters remain post-estimation helpers rather than estimator branches:

- OLS/linear-IV: `fitted - X beta`;
- PPML: `eta - offset - X beta`;
- IV-PPML: the same eta contract after removing the reported normalized constant.

They do not rerun an estimator and do not add observation-sized FE storage. The next meaningful extension is only a compact representation of additional K-way null-space directions when a concrete application requires `extra_nullity > 0`; varying-slope recovery remains out of scope.


## Identification diagnostics

Recovery now defaults to strict identification at the independent-component level. A bad block no longer discards unrelated identified blocks: fully identified components are recovered, while levels in components with extra null-space directions are reported as unavailable and carry structured diagnostics. If all components are unidentified, `FixedEffectIdentificationError` is raised. The diagnostic layer distinguishes realized-data rank deficiency from support removed by PPML separation when stage masks are available, while recursive singleton pruning is reported explicitly without falsely claiming it caused rank loss. Advanced structural workflows may set `strict_identification=False` to inspect an unresolved decomposition; automatic structural restrictions are intentionally out of scope.
