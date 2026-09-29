# Migrating from 0.6.1 to 0.6.2

No top-level estimator signature, public dataclass field, default model, default
DoF method, or root/effects export is intentionally changed. Numerical and
failure/reporting semantics change where 0.6.1 was incorrect:

- Rerun affected nonlinear-IV results involving extreme units, nonzero offsets,
  or standardization-dependent first-stage interpretation. Block PPML with
  `standardize=False` and disparate units also needs recomputation.
- Cluster labels may be strings, shifted integers or fractional categories.
  Results no longer depend on the spelling/numbering of those categories.
  `vce="cluster"` requires at least two observed groups in EVERY dimension of
  the estimation sample; a single dimension cannot silently become robust VCE.
- IV-PPML legacy `None/model/iid/unadjusted/homoskedastic` requests still compute
  the historical robust covariance, but now report `vce="robust"` accurately.
  `diagnostics["vce_requested"]` retains the request and
  `diagnostics["first_stage_units"]` is `"original"`. These are added dictionary
  entries, not new dataclass fields. Fixed dictionary consumers should allow them.
- The normalized IV-PPML `_cons` retains stored `stderr=0`; its reported p-value
  and confidence interval are unavailable, not evidence of zero uncertainty.
- Iteration counts reject booleans, fractions and non-finite values. Keep them
  as positive integers; `max_step_halving` separately permits integer zero.
- Block numeric rank now refers to the same equilibrated directions as the
  covariance bread. It is NOT the exact categorical FE rank/DoF.

References, scope and local evidence:
`docs/technical/history/nonlinear-and-resource-review-0.6.2.md`,
`docs/development/test-status.md`, `docs/release/performance-0.6.2.md`.

---

# Migrating 0.6.0 to 0.6.1

Existing top-level calls remain valid. Recompute results produced with severe
column-unit differences, PPML standardize=True and model/IID VCE, one-cluster
inference, or recovery condition-limit stops. These are corrections, not a new
estimator or a change to the default DoF method.

A one-cluster covariance request now raises a structured InferenceError instead
of returning spuriously precise inference. Invalid t degrees of freedom produce
unavailable statistics/p-values/intervals without significance stars. Non-finite
recovery inputs and nonintegral/noncontiguous `assume_dense` codes now fail
validation. Use normal raw-label encoding rather than pretending gapped labels
are dense. LSMR/LSQR condition-limit stops report nonconvergence.

`RecoveryDiagnostics` adds four defaulted fields (`stop_code`, `stop_reason`,
`condition_estimate`, `normal_equation_residual_norm`). Named and prior positional
construction remains available. Consumers serializing an exact fixed schema
should tolerate these additive fields. The first explicit effects-submodule
snapshot is now included; no previous snapshot was overwritten.

The mathematical registry retains its IDs but replaces unsupported automatic
originality implications with theorem-backed framework/application labels and
an explicit unverified-priority status. A formal manuscript plus passing tests
is not a certificate of novelty. See the complete 0.6.1 mathematical review.

The new GitHub workflow validates artifacts only and does not publish releases.
Configure the real repository URLs, private vulnerability reporting, and release
permissions at the destination repository. The local deliverable cannot verify
those remote settings.

---

## Historical migration notes

# 0.5.0 -> 0.6.0

Version 0.6.0 is backward compatible for existing estimator calls. `olshdfe`, `ivhdfe`, `ppmlhdfe`, `ivppmlhdfe`, `reghdfe`, and `ivreghdfe` require no source changes. The established automated top-level public-contract diff against v0.5.0 is empty.

The new functionality is opt-in under `econhdfe.effects`. Use it when fixed-effect level coefficients themselves are research objects. Recovery is limited to categorical/indicator intercept FE. By default, extra realized-sample null-space directions raise `FixedEffectIdentificationError`; changing normalization does not resolve lack of identification. Existing `save_fe=True` behavior remains available and unchanged.

# 0.4.10.2 -> 0.5.0

Version 0.5.0 is a backward-compatible architecture/performance MINOR release. Existing calls to `olshdfe`, `ivhdfe`, `ppmlhdfe`, `ivppmlhdfe`, `reghdfe`, and `ivreghdfe` do not require source changes, and the automated public-contract diff against 0.4.10.2 is empty.

The main user-visible change is execution capability rather than estimator semantics. OLS/linear-IV can use the new Data Layer for projected file/DataSource inputs; repeated-session workflows can reuse immutable encoded datasets and may opt into disposable cross-process result/within-column resume; interaction-rich specifications may take a certified structured path instead of materializing a full dense design; and the unified planner coordinates memory, representation, and thread decisions. `threads="auto"` may perform a one-time cached package-generated calibration on sufficiently large HDFE jobs. Explicit thread counts continue to bypass that calibration.

No migration is required for existing in-memory DataFrame workflows. Users who need strictly fixed runtime-resource behavior should set an explicit positive integer thread count instead of `"auto"`. Planner developer reports are local, opt-in artifacts and do not upload data or telemetry automatically.

The 0.4.10.2 `econhdfe-report` CLI and `econhdfe.support_reports` API remain available unchanged. The wheel continues to ship the canonical privacy-safe error and benchmark templates; benchmark consent remains separate from template generation.

# 0.4.10.1 -> 0.4.10.2

No estimator/public-Python-API migration is required. Version 0.4.10.2 is a reporting-tooling REVISION. Installed builds add the `econhdfe-report` console helper for writing canonical error/benchmark Markdown templates and `econhdfe.support_reports` for strict allowlist-based in-process parameter-only error reports. Benchmarking still requires separate explicit user consent.

# 0.4.10 -> 0.4.10.1

No estimator/API/runtime migration is required. Version 0.4.10.1 is a documentation/Skill/release-governance REVISION.

The canonical error report is now privacy-minimized and parameter-only. Agents/developers should not ask users to reconstruct/synthesize data or return raw/sample observations, exact identifiers, paths, commands/scripts, raw logs, full tracebacks, or attachments. Use anonymous model-role aliases and return only configuration, counts, structured error fields, aggregate numerical diagnostics and parity-difference magnitudes. If more detail is required, request narrower metadata/counters first. Real-data benchmarking remains a separate explicit-consent workflow.

# 0.4.9 -> 0.4.10

Version 0.4.10 is a backward-compatible factor-variable/HDFE integration **PATCH**. The public `fv(expression)` API is unchanged, but `fv()` expressions can now be passed directly to `absorb=` in DataFrame linear OLS/IV estimators.

Absorb-context compilation is role-aware rather than explicit-dummy expansion. `fv("i(firm)#i(year)")` compiles to one joint categorical FE; `fv("i(firm)#c(trend)")` compiles to a slope-only heterogeneous FE; `fv("i(firm)##c(trend)")` compiles to an intercept-plus-slope FE. Full factorials are reduced symbolically to an equivalent minimal absorbed column space before FE encoding/DoF, so `fv("i(firm)##i(year)##c(trend)")` becomes the joint `firm#year` intercept-plus-trend-slope representation rather than a redundant list of lower-order slope FEs.

Base selectors are a parameterization concept for explicit factor regressors and are therefore rejected in absorb context (except the implicit/default first-base parser state). Pure continuous absorbed terms and products with more than one continuous atom remain unsupported by the current HDFE slope primitive and fail with structured `absorb.factorvars.*` errors. Repeated-spec sessions accept categorical-only `fv()` absorb expressions and reject heterogeneous-slope ones; direct `olshdfe()` / `ivhdfe()` remains the path for slope absorption. PPML/IV-PPML low-level FE plans are unchanged.

# 0.4.8.1 -> 0.4.9

Version 0.4.9 is a backward-compatible frontend/API **PATCH**. Existing `factor()` / `reg_interaction()` code remains valid. The release adds one public helper:

```python
from econhdfe import fv

fv("i(x)##i(y)##c(z)")
```

The DSL deliberately uses `i(x)` / `c(x)` rather than Stata-style `i.x` / `c.x`; dot notation is rejected with structured error code `design.factorvars.syntax`. `#` means interaction-only and `##` means full factorial. `+` joins grouped terms, `i(x,y)` distributes the factor prefix over multiple columns, continuous powers such as `c(x)#c(x)` are supported, and interaction order is capped at eight. Reference levels use `i(x, base=<value|first|last|freq|none>)`.

`fv()` compiles into the existing `Factor` / `RegressorInteraction` design engine before structural-collinearity and HDFE processing. It does not introduce a second numerical design path. The compact DSL is currently certified on the full DataFrame linear OLS/IV estimator path. Repeated-spec sessions remain optimized for ordinary numeric columns; low-level PPML/IV-PPML APIs continue to accept constructed numeric matrices.

# 0.4.8 -> 0.4.8.1

No estimator, runtime, public API/schema, result, configuration, error-code, or dependency migration is required. Version 0.4.8.1 is a documentation/Skill/release-governance **REVISION**.

The release adds `docs/development/ERROR_REPORT_TEMPLATE.md` and a byte-identical standalone-Skill copy. Agents should use it for crashes, material numerical/reference-package mismatches, convergence anomalies, regressions, and API/documentation defects. Raw-data inspection, replication-script access, and performance benchmarking remain separate explicit-consent choices; file attachment alone is not permission.

# 0.4.7 -> 0.4.8

Version 0.4.8 is a backward-compatible statistics/reporting **PATCH**. Estimator call signatures are unchanged; public result dataclasses gain optional ancillary fields.

For OLS, `RegressionResult` now exposes `rss`, `tss`, `tss_within`, `mss`, `rmse`, `loglike`, `loglike_null`, `f_statistic`, `f_pvalue`, `df_model`, `df_resid_fit`, and `vcov_rank`. These follow current `reghdfe` fit-statistic conventions. `df_resid` remains the inference/reference-distribution DoF; `df_resid_fit` is the separate fit-statistics denominator introduced in the 0.4.7 DoF repair. `rank` retains its historical econhdfe meaning (active design/model rank); use `vcov_rank` when comparing the covariance-rank stored result reported by mature reference packages.

For linear IV, the new ancillary fields are deliberately limited to quantities with a stable partialled-out/`ivreg2`-small interpretation: RSS, within TSS, RMSE, model F/p-value, model DoF, fit residual DoF, and covariance rank. `tss`, `mss`, and Gaussian likelihood fields remain unset for IV because `ivreghdfe`/`ivreg2` conventions do not supply an unambiguous reconstructed overall-fit analogue. Existing econhdfe overall IV R-squared fields keep their prior meaning and should not be advertised as strict `ivreghdfe` parity.

For PPML, a correctness bug is fixed: when internal outcome standardization is enabled, final `deviance` is now multiplied back to the original outcome scale. Previous releases could therefore report a deviance differing by the internal outcome scale factor even when coefficients, SEs and log likelihood agreed. `PPMLResult` also adds optional `loglike_null`, `pseudo_r2`, `chi2`, `chi2_pvalue`, `df_model`, `nobs_full`, and `vcov_rank`.

The Agent Skill now treats econhdfe explicitly as testing/beta software for consequential empirical work. For every substantive run/table, randomly cross-check at least one representative specification against a mature reference implementation when available. Real-user-data benchmarking requires explicit user agreement; source data/scripts remain read-only by default and the result should use the canonical benchmark Markdown template.

# 0.4.6.1 -> 0.4.7

Version 0.4.7 fixes OLS/linear-IV fit-statistic degrees-of-freedom semantics to match `reghdfe`. Existing estimator call signatures are unchanged.

`RegressionResult` adds the optional field `r2_adjusted_within`. OLS and linear-IV results now report the four standard HDFE fit measures: `r2`, `r2_adjusted`, `r2_within`, and `r2_adjusted_within`.

The public `df_resid` field remains the **inference/reference-distribution DoF**. Under clustered VCE this may be capped at `G-1`, exactly as before. It is no longer reused for adjusted R-squared. The adjusted fit measures instead use a separate internal fit residual DoF that charges the absorbed FE rank (including FE dimensions nested in clusters) and the active regressor rank. Consequently, clustered and non-clustered fits of the same sample/design now have identical R-squared statistics even when their inference `df_resid` differs.

For strict numerical comparison with `reghdfe`, use the same absorbed-DoF method. In 3+ FE designs, econhdfe's explicit `dof_method="exact"` can legitimately produce a different adjusted R-squared denominator than `reghdfe`'s approximate/default multiway DoF accounting; the unadjusted overall/within R-squared measures are unaffected by that distinction.

# 0.4.6 -> 0.4.6.1

No estimator/API/runtime migration is required. Version 0.4.6.1 is a documentation/governance REVISION.

The release adds the package-wide technical-innovation audit and machine-readable registry, plus formal manuscripts for the two previously undocumented registered HDFE innovations. Runtime modules, estimator equations, numerical defaults, public signatures/result schemas/configs/error codes and dependencies are unchanged.

Third-party developers must now pass `python scripts/technical_innovation.py`. Any future contribution described as `technical_innovation` must be registered with a formal `.tex + .pdf`, a prior-art boundary, implementation correspondence, tests and evidence in the same release.

# 0.4.5 -> 0.4.6

Existing estimator calls remain source-compatible. Version 0.4.6 adds optional advanced one-way OLS cluster-inference APIs and fixes weighted execution of the legacy coefficient-draw wild bootstrap.

New public surfaces:

```python
from econhdfe import cluster_diagnostics, wild_cluster_test_ols
```

Both require an OLS result fitted with `keep_state=True`; `wild_cluster_test_ols` additionally requires exactly one clustering dimension. Multi-way CRV1 remains supported by the estimator, but multi-way WCB is intentionally rejected rather than silently approximated.

`wild_bootstrap()` remains backward compatible. Weighted calls now refit each pseudo-outcome in the fitted WLS metric; users who previously relied on the erroneous unweighted coefficient draws should expect corrected weighted bootstrap draws. The function also accepts an additive `weight_distribution=` option.

CRV3/cluster-jackknife is not part of the 0.4.6 public API.

# 0.4.4.4 -> 0.4.5

No estimator call-signature migration is required. Existing PPML/IV-PPML code remains source-compatible.

Behavioral correction: when `ExecutionConfig(threads=..., memory_budget_mb=...)` is supplied to `ppmlhdfe`, `ivppmlhdfe`, `PPMLHDFE`, or `IVPPMLHDFE`, those values now reach the internal weighted-HDFE projector. Code that previously assumed the settings were advisory may observe different thread usage, memory planning, and runtime.

Optimized PPML separation no longer forces simplex through the replica engine or ReLU through a fixed LSMR absorber. The default separation method set is unchanged, and the estimator continues to run the requested methods unless an existing logically safe skip applies.

For performance/reproducibility audits, inspect:

```python
result.diagnostics["execution"]
result.diagnostics["projection_resources"]
result.diagnostics["separation_seconds"]
result.diagnostics["separation_iterations"]
result.diagnostics["separation_solvers"]
```

`memory_budget_mb` is a projector/workspace policy, not a hard cap on total process RSS.

# Migrating from pyreghdfe to econhdfe

New code should use:

```python
from econhdfe import olshdfe, ppmlhdfe, ivhdfe
```

The former estimator names remain aliases:

```python
from econhdfe import reghdfe, ivreghdfe
```

Existing code that imports `pyreghdfe` continues to work in this alpha through a compatibility shim. No numerical implementation remains in that namespace.

Internal imports should not target the compatibility namespace. Extension code should import from the domain layer it needs, for example `econhdfe.hdfe`, `econhdfe.iv`, or `econhdfe.compute`.


## 0.4.4 to 0.4.4.1

The fourth component is a maintenance REVISION, not a new estimator/runtime patch line. The release adds no public-contract change. Future maintenance revisions on this base continue as `0.4.4.2`, `0.4.4.3`, and so on; material runtime changes advance to `0.4.5`.

No estimator-call, runtime, or public-API migration is required. Version 0.4.4.1 reorganizes repository/release documentation only.

For developers and release tooling, former root documents now live under `docs/development/` or `docs/release/`; the machine-readable maintenance manifest is `docs/release/maintenance.json`. HDFE exact-rank and solver benchmark evidence is under `benchmarks/hdfe/`. The exact 3+ FE structural-DoF manuscript is now a source/release artifact under `docs/technical/hdfe/exact-multiway-dof/` and is intentionally not installed in the wheel.


## 0.4.3 to 0.4.4

No estimator-call migration is required. Existing `olshdfe`, `ivhdfe`, `ppmlhdfe`, `ivppmlhdfe`, session, result, configuration, and structured-error contracts remain valid.

Low-level users constructing `HDFEAbsorber` directly may optionally use the new additive controls `core_reduction`, `core_min_peel_fraction`, `auto_plain_limit`, `auto_polish_limit`, and `fused_rhs_memory_mb`, or set `acceleration="auto"`. The direct absorber default remains `acceleration="cg"`; existing calls therefore keep their previous acceleration choice. Optimized `FEPlan` uses the adaptive planner internally.

Numerical core reduction applies only to eligible NumPy 3+ pure-intercept categorical FE systems with strictly positive weights. A later zero-weight update disables that structural fast path because the effective topology changes. These controls affect execution only; requested FE topology used for inference/DoF is not rewritten by the numerical core.

## 0.4.2 to 0.4.3

No estimator-call migration is required. Public estimator signatures, result fields, configuration defaults and structured-error codes are unchanged.

Agent-skill packaging changed: the canonical skill is now `skills/econhdfe/`, and releases ship `econhdfe-skill-v0.4.3.zip`. The former duplicated repository-root `SKILL.md` and `skill/SKILL.md` entrypoints were removed. Users who installed one of those files manually should replace it with the complete `econhdfe/` skill directory so its `references/`, `scripts/`, and `agents/` resources remain available.

Third-party developers preparing a release must now also pass `scripts/compatibility.py check`; any intentional public-contract difference from the previous release requires a change-id approval with a written reason.

## 0.4.1 to 0.4.2

No estimator-call migration is required. Existing OLS/IV/PPML/IV-PPML and repeated-spec session calls remain valid. Code that intentionally depended on raw `KeyError`, `TypeError`, or `ValueError` from malformed repeated-session specifications or invalid preflight levels should instead catch `EconHDFEError` / `SpecificationError` and inspect the stable error code.

## 0.6.2 to 0.6.3

No exported signatures, dataclass fields, defaults or structured-error schemas
change. The NumPy `_weighted_colsum` uses fused contraction; floating summation
order can differ across layouts/platforms. This does not relax convergence or
accuracy contracts and does not enable `fastmath` or lower precision.

Developer workflow: use `python scripts/run_tests.py -- -q` for the full suite;
a deliberate `NUMBA_NUM_THREADS=1` cap fails preflight rather than skipping the
multithread tests. Performance probes use their own single-thread subprocesses.

`version.py gate` remains a maintenance-review/compatibility gate, not evidence
of formal release readiness. The additional `release_acceptance.py` checks
actual execution records. `build_release.sh` builds candidates by default;
strict isolated construction remains the default unless the explicit offline
candidate flag is set. Formal authorization uses already-frozen artifacts and
a detached execution record; see acceptance.md. Unexecuted checks remain blocked.
