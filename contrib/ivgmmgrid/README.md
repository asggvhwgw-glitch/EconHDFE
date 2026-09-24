# ivgmmgrid

A Stata helper for repeated `ivreghdfe ..., gmm2s` models whose outcome, included controls, instruments, fixed effect and cluster variable are shared. Supply the actual stored candidate regressor columns; each column becomes the single endogenous regressor in one model.

This is a computational implementation of the existing estimator. It does not introduce a new estimator, choose a grid or change bootstrap inference. No performance guarantee or priority claim is made here.

## 与 EconHDFE 的关系

本目录补充 Stata 用户的重复 IV-GMM 工作流，独立于 EconHDFE 的 Python 主包。它重用一组模型的固定效应变换和矩统计量，并以原生 `ivreghdfe` 校准和必要时回退。合并后会提供可调用的 Stata 命令、帮助页、合成数据示例和契约检查；不会改变 Python 主包的估计器或默认结果。

Original project source is **GPL-3.0-only**, Copyright (c) 2026 ivgmmgrid contributors; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Stata and native dependencies are installed separately. This is an optional source companion, not a component of the BSD Python wheel.

## Setup

Use a fresh Stata session after updating this package or loading an experimental
prototype. The current Mata loader checks function presence rather than an
in-memory version handshake; a previously loaded prototype can otherwise remain
active. The study's batch runners always start fresh sessions.

Requires Stata 18 or later and a working native `ivreghdfe` installation, including its `ivreg2`, `reghdfe`, `ftools` and associated dependencies. The helper does not install dependencies or access the network.

Keep these files together:

- `stata/ivgmmgrid.ado`
- `stata/ivgmmgrid.sthlp`
- `stata/water_gmm_moments.mata`

Add that directory to the Stata ado-path:

```stata
adopath ++ "/absolute/path/to/EconHDFE/contrib/ivgmmgrid/stata"
which ivreghdfe
which ivgmmgrid
help ivgmmgrid
```

The initial native reference is ivreghdfe 1.1.4, commit [bfb5577a6dbdfb029ab4ab6a7e93f7257a827b42](https://github.com/sergiocorreia/ivreghdfe/tree/bfb5577a6dbdfb029ab4ab6a7e93f7257a827b42). Pin dependency versions for a replication and repeat parity checks after changing them. The bundled Mata file is the validated main kernel, not the separate experimental micro-optimization variants.

## API

```stata
ivgmmgrid y [if] [in], exog(w1 w2) candidates(c1 c2 c3) ///
    instruments(z1 z2) absorb(time_id) cluster(cluster_id) [verifyall]
```

All options shown before `verifyall` are required. Each candidate represents:

```stata
ivreghdfe y w1 w2 (candidate = z1 z2) [if] [in], ///
    absorb(time_id) cluster(cluster_id) gmm2s
```

Use physical numeric variables. Expand time-series or factor expressions beforehand with the correct panel declaration and storage choices. Candidate columns and included controls must not overlap; this ambiguity is rejected with error 198. Keep parameter values in a separate vector if the columns represent a gamma or other parameter grid.

The prepared engine supports:

- One endogenous regressor per model, with one or more fixed included exogenous controls.
- One numeric intercept fixed effect and one numeric cluster variable.
- Unweighted, uncentered, full-rank two-step GMM.
- Native efficient-GMM covariance from first-step residuals and native small-sample corrections.

Weights, multiple simultaneous endogenous regressors, multiple FEs/clusters, heterogeneous slopes, HAC, centered moments, LIML and CUE are outside the API. Unsupported syntax is rejected. Rank/omission cases that native `ivreghdfe` can handle are evaluated by the native fallback.

## Preparation, checks and fallback

The first candidate is always estimated with native `ivreghdfe`. Its actual sample, degrees of freedom, ranks, coefficients, covariance and RMSE anchor the batch. The prepared engine reuses the within-FE transformation and global/cluster cross-products. It still computes a different initial 2SLS coefficient and clustered GMM weighting matrix for every candidate.

Candidates with different missing-value patterns are sent to separate native regressions. The command **does not intersect candidate samples**. Kernel unavailability, incompatible ranks or omissions, kernel failures and failed checks select native regression for the whole batch. Native failures propagate their original return code; failed models are not dropped, regularized, replaced with 2SLS or silently retried on another sample.

By default, only the first point is compared against native numerical output. The check covers coefficients, every covariance element, RSS, RMSE, Hansen J and exact sample/count/df metadata. Numeric tolerance is `1e-10 + 1e-8*abs(native)`; compared RMSE values must also match after Stata float rounding.

**A passing first-point check is not full-grid or near-tie certification.** With `verifyall`, every point is additionally estimated natively. Any discrepancy returns the complete native grid. This validation work is additional computation and is excluded from default performance claims.

The input dataset, variable types and observation order are restored. The helper clears internal `e()` on successful completion and on execution errors. This prevents accidentally treating the first anchor as the selected regression; store or refit earlier estimates if needed.

## Results

The helper is r-class:

| Result | Meaning |
|---|---|
| `r(grid)` | One row per candidate in supplied order |
| `r(engine)` | `moments` or `stock` |
| `r(fallback)` | 0 for moments; 1 for a wholly native grid |
| `r(fallback_reason)` | Diagnostic reason, empty for moments |
| `r(sample_common)` | Whether actual sample masks are shared |
| `r(sample_n)` | Common sample N; missing if masks differ, even when N counts coincide |
| `r(sample_counts)` | Each candidate's observation count |
| `r(candidates)` | Number of evaluated candidate columns |
| `r(verifyall)` | Whether validation was requested; not an independent certification flag |

For k coefficients (one candidate plus included controls), `r(grid)` contains 1+k+k²+8 columns:

1. `candidate`: candidate index.
2. `b_candidate`, followed by `b_<exog>`.
3. Every covariance element in column-major order: `V_1_1, V_2_1, ..., V_k_1, V_1_2, ..., V_k_k`.
4. `rss rmse N N_clust df_r hansen_j cond_omega rss_direct`.

The condition number concerns the first-step moment covariance; it may be missing on the native path if that matrix cannot be read. Direct RSS is marked by `rss_direct=1`. Long exogenous variable names can require truncated output labels on fallback; keep the original exogenous varlist as the mapping.

With two controls, the matrix has 21 columns: coefficients 2–4, covariance 5–13, RSS 14 and RMSE 15. Prefer `colnumb(G,"rmse")` to fixed indices. Copy `r(grid)` immediately if subsequent commands can overwrite r-class results.

## Selection and final estimation

Returned values are double precision. Preserve the application's original candidate construction, parameter sequence, objective storage type, exclusions and tie-breaking rule. For example, sequential float replacements need not equal one double-precision polynomial calculation, and choosing a minimum from double RMSE need not equal the original float-RMSE decision.

The helper does not provide a complete estimation result or postestimation interface. After selecting a candidate under the application's own rules, refit it explicitly:

```stata
ivreghdfe y w1 w2 (chosen_candidate = z1 z2), ///
    absorb(time_id) cluster(cluster_id) gmm2s
```

Apply any final-model sample restriction used by the original analysis, which can differ from the grid sample. Use this native refit for coefficient tables, inference and postestimation.

## Example and validation boundary

[examples/ivgmmgrid_example.do](examples/ivgmmgrid_example.do) generates its own small clustered dataset and candidate columns, calls `verifyall`, applies a clearly stated demonstration selection rule, and refits the winner. It assumes the package and native dependencies are already on the ado-path. It does not download data or install software.

Run `do examples/ivgmmgrid_example.do` and `do checks/ivgmmgrid_contract.do` from this directory after adding `stata` to adopath. See [VALIDATION.md](VALIDATION.md) for this contribution's fresh synthetic checks. No original real-data speed claims are re-certified here. Default anchor checking, full-grid validation and final native refitting are distinct safeguards and should not be described interchangeably.
