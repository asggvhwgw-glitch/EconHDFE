# Empirical research workflows

## Contents

- Model selection: OLS, linear IV, PPML and IV-PPML
- Specification preflight and interaction fixed effects
- Event-study reference control
- Group + individual fixed effects
- Repeated specifications, saved FE and publication output
- Omitted-variable/structural-simplification reporting
- Cluster diagnostics and one-way wild-cluster inference


Use this guide for ordinary econometric use: choosing an estimator, specifying FE/IV/event-study models, repeated tables and publication output.

## Model selection

### OLS with HDFE

```python
from econhdfe import olshdfe

res = olshdfe(
    df,
    y="outcome",
    x=["x1", "x2"],
    absorb=["firm_id", "year"],
    cluster=["firm_id", "year"],
    vce="cluster",
)
```

### Linear IV / 2SLS

```python
from econhdfe import ivhdfe

res = ivhdfe(
    df,
    y="y",
    exog=["control1", "control2"],
    endog=["endogenous_x"],
    instruments=["z1", "z2"],
    absorb=["firm_id", "year"],
    cluster="firm_id",
    estimator="2sls",
)
```

Included exogenous controls are part of the instrument set by construction. Before interpreting first-stage/weak-ID statistics, verify that excluded instruments remain linearly independent after FE absorption and structural pruning.

### PPML-HDFE

```python
from econhdfe import PPMLConfig, ppmlhdfe

res = ppmlhdfe(
    y,
    X,
    absorb=[firm, year],
    clusters=[firm, year],
    vce="cluster",
    config=PPMLConfig(engine="optimized"),
)
```

Use PPML for the intended Poisson PML/multiplicative conditional-mean specification; do not choose it merely because the outcome has zeros.

### IV-PPML

```python
from econhdfe import IVPPMLConfig, ivppmlhdfe

res = ivppmlhdfe(
    y,
    exog=controls,
    endog=endog,
    instruments=z,
    absorb=[firm, year],
    vce="robust",
    config=IVPPMLConfig(engine="optimized"),
)
```

IV-PPML is not linear IV. Do not transfer linear KP/SW/Stock-Yogo critical values or diagnostics into this model unless a statistic is explicitly derived/implemented for IV-PPML.

## Semantic specification preflight

Before fitting an interaction-heavy model:

1. identify known categorical hierarchies and FE interactions;
2. treat user-supplied relations such as `city -> province` as candidate dependencies;
3. distinguish semantic suspicion from estimator-certified structural dependence;
4. inspect whether exogenous controls, endogenous regressors or excluded instruments may be absorbed/spanned;
5. preserve role priority in IV: exogenous controls should not be silently sacrificed to save an excluded instrument;
6. use `collinearity="raise"` once during strict specification validation when the result is consequential.

After estimation inspect the canonicalization/solver-selection information and omitted-variable metadata. Report material simplifications that affect interpretation.

## Interaction fixed effects and heterogeneous slopes

Prefer the public interaction representation rather than precomputed string cells when available:

```python
from econhdfe import interaction, FixedEffect

absorb = [interaction("city", "year"), "firm"]
slope_fe = FixedEffect("firm", slopes=("tenure",), intercept=True)
```

Do not remove the intercept portion of a slope-and-intercept FE merely because a related intercept FE looks nested; the heterogeneous-slope term changes the column space.

## Python-safe factor-variable expressions

For compact categorical/continuous interactions in OLS or linear IV DataFrame estimators, prefer `fv()` when it is clearer than manually nesting `factor()` and `reg_interaction()`:

```python
from econhdfe import fv, olshdfe

res = olshdfe(
    df,
    y="y",
    x=["control", fv("i(region)##i(year_group)##c(exposure)")],
    absorb=["firm", "year"],
)
```

The DSL is deliberately Python-safe: use `i(x)` and `c(x)`, not Stata-style `i.x` / `c.x`. `#` means interaction-only; `##` means full factorial; `+` joins terms inside groups. `i(a,b)##c(z)` is shorthand for `(i(a)+i(b))##c(z)`. Base controls are `i(x, base=<value|first|last|freq|none>)`. Quoted column names are accepted. Up to eight-way interactions are supported.

Treat `fv()` as a symbolic frontend, not as a dummy-matrix constructor. In `x=` / `exog=` / instrument roles it compiles into the existing `factor()` / `reg_interaction()` design representation. In `absorb=` it compiles into the canonical HDFE `FixedEffect` / categorical-interaction representation. Thus `fv("i(firm)#i(year)")` is a joint absorbed FE, `fv("i(firm)#c(trend)")` is slope-only, and `fv("i(firm)##c(trend)")` absorbs both the group intercept and group-specific trend. Full-factorial absorbed expressions are symbolically reduced to an equivalent minimal FE column space before encoding. Do not use dot-style Stata syntax in Python. Repeated-spec sessions accept categorical-only `fv()` absorb expressions; for heterogeneous-slope `fv()` absorption use direct `olshdfe()` / `ivhdfe()`.

## Explicit event-study reference control

Never let rank resolution accidentally choose the event-study reference category.

```python
from econhdfe import factor, reg_interaction, omit_level, olshdfe

event_study = reg_interaction(
    factor("event_time", drop_base=False, name="event_time"),
    "ever_treated",
    name="event_study",
)

res = olshdfe(
    df,
    y="y",
    x=["x1", "x2", event_study],
    absorb=["firm", "year"],
    omit=[omit_level("event_time", -1, term="event_study")],
)
```

Use `omit_column`, `omit_term`, or `omit_level` deliberately. After fitting, verify the intended reference appears as a user-directed omission and inspect automatic omissions separately. Never relabel an automatically omitted event-time coefficient as the chosen reference period.

For IV use role-specific `omit_exog`, `omit_endog`, and `omit_instruments`. Do not omit instruments to conceal underidentification.

## Group + individual fixed effects

For patent-inventor, paper-author or other long-form group/individual data:

```python
res = olshdfe(
    df,
    y="patent_value",
    x=["x1"],
    absorb=["year", "inventor_id"],
    group="patent_id",
    individual="inventor_id",
    aggregation="mean",
    incidence_backend="auto",
)
```

The group-level outcome/regressors must be constant within group. Do not treat repeated group outcomes as ordinary independent row-level observations.

## Repeated linear specification tables

When the DataFrame is fixed and the workflow repeatedly changes y, adds/removes numeric controls, or toggles among a small set of intercept-only FE combinations, prefer `OLSHDFESession` / `IVHDFESession` and `fit_many()` over independent calls.

Batch related specifications so the session can residualize the union of needed columns once per FE structure. Use ordinary estimator calls for nonlinear PPML/IV-PPML, heterogeneous-slope FEs, complex design objects not supported by sessions, or group+individual FE.

## Identified fixed-effect recovery

When FE level coefficients are themselves research objects, prefer the separate `econhdfe.effects` recovery layer rather than interpreting an arbitrary solver decomposition. The v0.6.0 recovery contract is for additive categorical/indicator intercept FE only. It reports realized-sample rank/nullity and components, supports reference/mean-zero/weighted-mean-zero reporting normalizations, and salvages fully identified independent components when another component has additional unidentified directions. Unidentified component levels are retained with `NaN` coefficients and explicit diagnostics; `FixedEffectIdentificationError` is raised when no component is recoverable.

Normalization selects a representative of an identified equivalence class; it cannot create identification. Compare effects across disconnected components only when an external structural restriction justifies doing so. For PPML/IV-PPML, recovery applies to the final finite-MLE sample after singleton/separation decisions. Pure categorical interactions such as `firm#year` are supported; varying-slope terms such as `firm#c.age` are not a recovery feature in this release.

Use `strict_identification=False` only for advanced structural workflows that explicitly intend to inspect an unresolved numerical decomposition; such coefficients are not marked identified.

For a fitted linear model, full raw indicator groups may be supplied even when the estimator recursively removed singletons; the adapter reconstructs and verifies the same singleton-pruned sample before recovery:

```python
from econhdfe.effects import recover_linear_result, NormalizationSpec

fx = recover_linear_result(
    fit,
    df[["x1", "x2"]].to_numpy(),
    [df["worker"].to_numpy(), df["firm"].to_numpy()],
    x_names=["x1", "x2"], fe_names=["worker", "firm"],
    normalization=NormalizationSpec("weighted_mean_zero", baseline="worker"),
)
```

## Saved fixed effects

```python
res = olshdfe(..., save_fe=True)
fe = res.fixed_effects
```

With multiple FE dimensions, individual level coefficients generally depend on normalization. Validate the summed FE contribution/fitted-value reconstruction rather than treating each saved level as uniquely identified without checking normalization.

## Cluster diagnostics and wild-cluster inference

The main estimators already support one-way and multi-way CRV1. Do not replace a substantively justified multi-way clustering design merely because advanced WCB is currently one-way.

For consequential one-way clustered OLS with few or strongly unbalanced clusters, retain state and inspect diagnostics:

```python
from econhdfe import cluster_diagnostics, wild_cluster_test_ols

fit = olshdfe(
    df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
    cluster="state", vce="cluster", keep_state=True,
)

diag = cluster_diagnostics(fit)
wcr = wild_cluster_test_ols(
    fit, param="x1", r=0.0, reps=9_999,
    impose_null=True, weight_distribution="rademacher",
)
```

Interpret diagnostic flags as prompts for review, not automatic specification rules. `few_clusters` currently means G<30; size/score concentration flags are heuristic. `impose_null=True` is WCR11 and is the normal restricted-bootstrap sensitivity test; `False` is WCU11. Rademacher, Mammen, Webb and normal wild weights are available. When Rademacher support is small enough and requested repetitions cover all `2**G` sign patterns, the implementation performs full enumeration.

Current boundary: WCR11/WCU11 is certified only for one-way OLS-HDFE. Multi-way CRV1 remains available from the estimator, but do not claim multi-way WCB, IV WCB, PPML score bootstrap or CRV3 from this release.

## Testing-release cross-check

econhdfe should currently be treated as testing/beta software for consequential empirical work. For each substantive regression run or table, randomly select at least one representative specification and rerun the same specification in a mature reference implementation when available. Compare N/sample decisions, coefficients, standard errors, inference and absorbed DoF, and the major fit statistics relevant to the estimator. Use `reghdfe` for OLS-HDFE, `ivreghdfe`/`ivreg2` for linear IV, and `ppmlhdfe` for PPML when those are the appropriate references.

Do not force apparent parity by changing the user's FE, clusters, weights, omitted category, separation policy or DoF method. If definitions differ (notably some IV R2/fit-statistic conventions), record the convention difference rather than treating it as a numerical failure. See [benchmarking.md](benchmarking.md) for the full parity/benchmark protocol.

## Publication output

Prefer the paper-facing reporting surface:

```python
pub = res.publication_output()
coef_table = pub["coefficients"]
model_stats = pub["model"]
```

A publication result should include the coefficient/inference quantities and model metadata required for the table. Full solver/canonicalization/profile diagnostics should remain off unless requested or needed to explain a problem.

For OLS and linear IV, distinguish fit DoF from inference DoF. `result.df_resid` is the inference/reference-distribution DoF and can be `G-1` under clustered VCE. The four reported fit statistics (`r2`, `r2_adjusted`, `r2_within`, `r2_adjusted_within`) use the HDFE fit denominator instead; do not recompute adjusted R-squared from `result.df_resid`. When reproducing `reghdfe`, use the same absorbed-DoF method. An explicit econhdfe `dof_method="exact"` in a 3+ FE design can intentionally change adjusted R-squared relative to `reghdfe`'s approximate multiway DoF accounting.

For linear IV, report the package's applicable first-stage and identification/overidentification diagnostics. For nonlinear IV-PPML, report only diagnostics whose interpretation is valid for that estimator.

## Reporting omitted variables and structural simplification

The normal default warns/reports omitted variables rather than silently discarding them. Distinguish:

- deliberate user reference/basis omissions;
- structural absorption or exact dependency certified before materialization;
- numeric linear combinations detected after absorption.

Unexpected automatic omissions can change the estimand or identified dynamic effects and must be surfaced before presenting a regression table.
## PPML performance diagnostics

When PPML runtime is unexpectedly high, first distinguish separation from IRLS. In v0.4.5+ inspect `result.diagnostics["separation_seconds"]` and `result.diagnostics["projection_resources"]`. Keep the full requested separation policy for formal estimates; treat FE-only separation as a labelled sensitivity/performance decomposition rather than silently changing the estimator workflow.


## 0.7 后估计与预测

先核实已安装版本：以下在 0.7.0 源码及相应版本制品中可用；PyPI 发布状态需单独核实。res.predict() 返回最终样本 fitted；
restore_sample=True 按物理行补 NaN。predict(newdata, kind="xb") 使用冻结保留系数设计；
stdp 仅为 beta 部分标准误，不是完整预测区间或 FE 不确定性。标准 OLS/IV 的新数据含 FE
预测需拟合时 save_fe=True；FE 必须为命名 categorical intercept。未知 level、未见
interaction cell、跨 component/extra-nullity 或破坏训练嵌套的新组合默认拒绝，
unknown="nan" 只将这些行标 NaN，不更改模型。自动遗漏回归项时新行 response 尚未认证，
不能把 xb 改称完整预测。IV 结构预测不需要排除工具变量。

linear_combination/wald_test 仅对一个结果的参数/V 作常规推断，不是弱 IV 稳健检验。
PPML/IV-PPML、varying-slope、group+individual 样本外预测、margins/AME 和跨模型联合推断
仍不支持。不得以新样本重拟合、未知类别填零、beta-only V 冒充含 FE 不确定性来补齐边界。
