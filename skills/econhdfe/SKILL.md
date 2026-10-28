---
name: econhdfe
description: Use econhdfe for high-dimensional fixed-effect OLS, linear IV, PPML, IV-PPML, repeated specifications, publication-ready inference, cluster diagnostics/wild-cluster testing, performance tuning, validation, and package extension. Use when an empirical economics workflow would otherwise rely on reghdfe/ivreghdfe/ppmlhdfe-style estimation, when diagnosing or benchmarking econhdfe, or when developing against its public API.
---

# econhdfe

Treat this file as the router for the skill. Load only the references needed for the current task; do not preload every guide.

## Route the task

- For installing the Python package, installing this skill, optional dependencies, or checking the runtime, read [references/installation.md](references/installation.md).
- For solver, inference, execution, weights, clustering, GPU, memory, caching, automatic thread calibration, and advanced parameter choices, read [references/configuration.md](references/configuration.md).
- For empirical research workflows, model selection, event studies, repeated specifications, result interpretation, and publication output, read [references/empirical-research.md](references/empirical-research.md).
- For consequential replication, parity testing, performance benchmarking, numerical diagnostics, and release-grade validation, read [references/advanced-validation.md](references/advanced-validation.md).
- For the testing-release parity policy or user-authorized real-data benchmarks and the standard benchmark return format, read [references/benchmarking.md](references/benchmarking.md).
- For the fill-in Markdown returned to developers after an authorized real-machine benchmark, use [references/benchmark-report-template.md](references/benchmark-report-template.md).
- For crashes, numerical anomalies, parity mismatches, suspected bugs, or API/documentation failures, use [references/error-report-template.md](references/error-report-template.md) to return a structured developer report.
- For suspicious `threads="auto"` choices, planner regressions, or machine-specific performance anomalies, read [references/planner-feedback.md](references/planner-feedback.md) and use [references/planner-report-template.md](references/planner-report-template.md).
- For extending estimators/backends, changing public APIs, adding errors/results/config fields, contributing tests, or preparing a release, read [references/developer-guide.md](references/developer-guide.md).
- For codebase maps, dependency diagrams, contributor onboarding visuals, or checking whether architecture documentation is stale, read [references/architecture-visualization.md](references/architecture-visualization.md).

Read more than one reference only when the task genuinely spans those roles. Keep setup questions out of the empirical guide and development/release mechanics out of ordinary empirical workflows.

## Core estimator routing

Choose the estimator by the economic model, not by which backend is fastest:

- `olshdfe`: linear OLS with high-dimensional fixed effects.
- `ivhdfe`: linear IV/HDFE; supports 2SLS and the package's linear-IV alternatives/diagnostics.
- `ppmlhdfe`: Poisson PML with high-dimensional fixed effects and nonnegative outcomes.
- `ivppmlhdfe`: nonlinear additive-moment IV-PPML. Do not report linear-IV KP/SW/Stock-Yogo statistics as IV-PPML diagnostics.
- `OLSHDFESession` / `IVHDFESession`: repeated linear specifications on the same DataFrame when y, controls, or a small set of FE combinations change repeatedly.

Never create huge dummy matrices manually when the HDFE interface can absorb the effects directly.
For compact linear factor-variable specifications, use `fv("i(group)##c(age)")`; the Python-safe DSL uses `i(x)` / `c(x)`, not Stata-style dot prefixes. The expression is symbolic: regressor/instrument roles compile into the canonical design engine, while `absorb=` compiles into canonical HDFE FE/slope specifications.

## Global empirical safeguards

1. Preserve the user's economic specification when tuning execution. Thread count, memory budget, pool size and solver backend are execution choices; controls, fixed effects, instruments, clustering and event-study reference periods are econometric choices.
2. Use explicit omission selectors for event-study/reference-category control. Do not rely on incidental column order to choose the omitted category.
3. Treat semantic nesting/collinearity reasoning as a hypothesis. Let econhdfe certify structural dependencies or detect numeric rank deficiency after absorption.
4. Inspect omitted variables, FE canonicalization and solver-selection metadata before interpreting coefficients in interaction-heavy designs.
5. Catch `EconHDFEError` and use its structured `code`, `stage`, `details` and `suggestion`; do not parse exception text.
6. Do not claim Stata or upstream package parity unless the relevant external validation actually ran.
7. Treat econhdfe as testing/beta software for consequential empirical work. For every substantive run/table, randomly select at least one representative specification and compare it with `reghdfe`, `ivreghdfe`/`ivreg2`, `ppmlhdfe`, or another mature reference when available. Surface any mismatch in sample, coefficients, SEs, DoF or major fit statistics.
8. If the user explicitly agrees, a real-data benchmark may be run on the user's data. Keep source data/scripts read-only, preserve the econometric specification, and return the benchmark in `references/benchmark-report-template.md` format. Never benchmark user data without explicit agreement.
9. When a material error or parity mismatch is found, organize the report with `references/error-report-template.md` using **parameter-only, privacy-minimized metadata**. Do not request or return raw/sampled/synthetic observations, exact variable identifiers, file paths, exact commands/scripts, raw logs, full tracebacks, or attachments. Alias model roles (`y`, `x1`, `fe1`, `cluster1`) and return only configuration, counts, structured error fields, aggregate diagnostics, and parity-difference magnitudes. If more information is needed, ask for narrower metadata/counters first. Benchmarking remains a separate explicit-consent workflow.

## Bundled deterministic helpers

Run these from the skill directory when useful:

```bash
python scripts/check_environment.py
python scripts/smoke_test.py
python scripts/architecture_map.py --root /path/to/econhdfe-source
econhdfe-report error --output econhdfe-error-report.md
econhdfe-report benchmark --output econhdfe-benchmark-report.md
```

`econhdfe-report error` writes the canonical blank privacy-minimized report template; `econhdfe-report benchmark` writes the canonical benchmark template and does **not** grant permission to benchmark user data. For an in-process failure, prefer `econhdfe.support_reports.write_error_report(...)`: it reads only an explicit allowlist of scalar/counter fields from result, error, and configuration objects; it never serializes the result object, DataFrame, coefficient/residual arrays, variable names, paths, error messages, raw logs, or full traceback text.

`check_environment.py` reports the installed econhdfe/Python/dependency/runtime capabilities as JSON. `smoke_test.py` runs small OLS, IV and PPML fits against the installed package and exits nonzero on failure. `planner_report.py` emits a privacy-minimized real-runtime thread-calibration/planner report without reading model data. `architecture_map.py` recovers internal Python imports with AST parsing and emits deterministic JSON/Markdown/HTML architecture maps for a source tree; pass `--check` to detect stale committed maps.

## Output discipline

For normal empirical work, return the requested estimates and publication-relevant diagnostics, not internal solver telemetry. Surface additional numerical/structural diagnostics when they materially affect identification, interpretation, convergence, reproducibility, or when the user asks for them.


### IV-Poisson outer solver candidate

For explicit nonlinear outer-solver development or testing, use the opt-in
`IVPPMLConfig(outer_solver="moment")` documented in `references/configuration.md`
and `docs/technical/ivppml/outer-moment-estimator.md`. Preserve the legacy default,
separate FE/backend choice from outer-solver choice, and use
`validation/iv_outer/README.md` for the executed scope and reproduction. Never
report Poisson-deviance stability as an IV moment convergence certificate.
