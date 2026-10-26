# Statistical-result parity audit

This note records the package-level audit of reported model statistics against the current `reghdfe`, `ivreghdfe`/`ivreg2`, and `ppmlhdfe` conventions. It distinguishes (i) incorrect calculations, (ii) missing reportable scalars, and (iii) intentionally different conventions.

## OLS-HDFE

Current `reghdfe` stores, in addition to coefficients and covariance, N/sample metadata, RSS, total and within TSS, MSS, the four R-squared measures, absorbed/model/residual DoF, RMSE, Gaussian log likelihood, fixed-effect-only log likelihood, model F and covariance rank.

`econhdfe` now computes these from one shared reporting path for direct OLS and repeated OLS sessions:

- `rss`: weighted residual sum of squares;
- `tss`: centered total sum of squares when an intercept is in the absorbed design, otherwise uncentered;
- `tss_within`: sum of squares after FE concentration;
- `mss = tss - rss`;
- `rmse = sqrt(rss / df_resid_fit)`;
- `loglike = -N/2 * [1 + log(2*pi) + log(rss/N)]`;
- `loglike_null`: same Gaussian expression using within TSS, matching the FE-only reference model;
- `f_statistic`: Wald test of all active explicit regressors divided by `df_model`;
- `f_pvalue`: F-reference p-value using inference `df_resid`;
- `df_model`: active explicit-regressor rank;
- `df_resid_fit`: fit-statistics DoF, distinct from cluster inference DoF;
- `vcov_rank`: rank of the reported covariance matrix, corresponding to upstream `e(rank)`.

The covariance is never pseudo-inverted merely to manufacture a finite model F. If the tested covariance block is rank deficient, model F/p-value are missing (`NaN`).

## Linear IV-HDFE

`ivreghdfe` delegates much of its fit-statistic behavior to `ivreg2`, and with absorbed FE the historical output is based on partialled-out data. In particular, its R-squared convention is not a drop-in replacement for the older overall `reghdfe` IV output.

To avoid relabeling different conventions as parity, `econhdfe` keeps its existing extended overall/within R-squared fields and adds only the unambiguous `ivreg2`-small/HDFE quantities:

- RSS;
- within TSS;
- RMSE using fit residual DoF;
- model Wald/F statistic and p-value;
- model DoF and fit residual DoF;
- covariance rank.

The 2026-10-20 external audit corrected three linear-IV quantities:

- Cluster CRV1 uses the ivreg2 denominator `N-K`, where K includes explicit
  rank and non-nested absorbed DoF. It does not apply OLS/reghdfe's additional
  nested-FE subtraction. If every absorbed effect is nested and `df_a=0`,
  ivreghdfe retains one absorbed intercept; the IV path does the same.
- IV RMSE uses `N-rank-max(df_a, nested_present)` as its fit denominator.
  It no longer adds all nested FE levels to the reporting penalty. OLS's
  fit-statistics policy is unchanged.
- Robust/cluster/HAC overidentification uses efficient second-step moments
  under the covariance estimated from the reported k-class residuals. The
  k-class coefficient vector remains the reported estimate. A raw quadratic
  form in inefficient 2SLS moments is not reported as Hansen J. The iid
  Sargan calculation uses the uncorrected residual variance. GMM2S retains
  its existing first-step weight matrix for its J statistic.

The full-result persistent ABI is now `linear-session-result-4`, preventing
restoration of pre-correction statistics. Within-transformed column caches
retain their numerical ABI. See [the implementation and validation note](iv-parity-correction-20261020.md).

`tss`, `mss`, and Gaussian log-likelihood fields remain unset for linear IV rather than presenting an overall quantity as if it were the partialled-out `ivreghdfe` scalar. Benchmark reports must mark these convention-sensitive quantities accordingly.

## PPML-HDFE

Current `ppmlhdfe` reports N/full N, separated/singleton counts, model/absorbed/residual DoF, covariance rank, deviance, log pseudolikelihood, null log pseudolikelihood, pseudo-R2 and model Wald chi-square.

Sample definition follows `ppmlhdfe` 2.3.0: `fixed_effects(absorb, touse, "iweight", depvar)` prunes singletons (row counts) and zero-outcome FE levels jointly, all counted in `e(num_singletons)`; simplex/ReLU/mu trims then reload the HDFE and prune new singletons into `e(num_singletons)`, while `e(num_separated)` counts only rows the separation methods flag. IV-PPML follows `ivppmlhdfe` 0.9.4, which counts post-trim singletons in `e(num_separated)`. Small Stata 18 reference cases are pinned in `tests/behavior/test_singleton_upstream.py`.

The audit found one correctness bug: when `PPMLConfig.standardize=True`, `econhdfe` unscaled coefficients, covariance, `mu`, `eta`, and log likelihood but returned deviance in standardized-outcome units. Poisson deviance scales linearly when both `y` and `mu` are scaled, and current `ppmlhdfe` explicitly multiplies converged deviance by the outcome scale before posting it. `econhdfe` now performs the same unscaling.

The result also now reports:

- `loglike_null` from the constant-only weighted mean outcome;
- `pseudo_r2 = 1 - loglike/loglike_null`;
- `chi2` and `chi2_pvalue` for the non-constant reported slopes;
- `df_model`;
- `nobs_full`;
- `vcov_rank`.

The PPML Wald statistic is omitted if its covariance block is rank deficient rather than using a generalized inverse silently.

## Deliberate non-parity / boundaries

- 3+ FE adjusted R-squared can intentionally differ from `reghdfe` when `econhdfe` is explicitly run with exact absorbed DoF while the reference uses its approximate DoF method. Compare like with like.
- Linear-IV overall R-squared remains an econhdfe extension; do not label it `ivreghdfe`-parity when the reference reports only the partialled-out/within convention.
- Iteration counts are algorithm diagnostics, not required to match when two solvers reach the same estimator under the requested tolerance.
- IV-PPML has no exact upstream `ivreghdfe`/`ppmlhdfe` analogue; compare shared subproblems and model-specific validated diagnostics rather than importing linear-IV fit statistics.

## Validation policy

Because econhdfe remains a testing/beta package, consequential empirical workflows should randomly select at least one representative specification per run/table and compare it with the mature reference implementation when one exists. The canonical user-return format is `benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md` and the Agent Skill includes the same template for standalone use.
