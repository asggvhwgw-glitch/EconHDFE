# econhdfe

[English](README.md) | [简体中文](README.zh-CN.md)

**High-performance high-dimensional fixed-effect econometrics for modern empirical research.**

`econhdfe` is a Python package for OLS-HDFE, linear IV-HDFE, PPML-HDFE, and IV-PPML-HDFE. It is designed for empirical work with large datasets, high-cardinality fixed effects, clustered inference, rich interactions, and many closely related specifications.

The project is currently **alpha research software**. This source tree is version 0.7.0, including Post/Prediction and IV correctness fixes. Repository-local validation is extensive, but licensed-Stata and complete upstream external certification remain separate validation boundaries.

## Performance at a glance

The main goal of `econhdfe` is to avoid computational work that the requested econometric model never required in the first place.

In one recorded interaction-rich benchmark with 96,000 observations, a certified structured representation reduced the physical design from approximately **75.3 MB to 7.7 MB**. Against the package's own dense execution path on the same specification, measured runtime improved by **10.51× for OLS-HDFE**, **6.42× for PPML-HDFE**, **2.07× for IV-PPML-HDFE**, and **1.30× for linear IV-HDFE**, while maximum coefficient differences remained between roughly `1e-16` and `1e-15`.

| Estimator | Structured vs. dense | Max coefficient difference |
|---|---:|---:|
| OLS-HDFE | **10.51×** | `3.1e-16` |
| Linear IV-HDFE | **1.30×** | `3.5e-15` |
| PPML-HDFE | **6.42×** | `5.8e-16` |
| IV-PPML-HDFE | **2.07×** | `1.7e-15` |

This is deliberately a workload-specific benchmark, not a claim that every regression becomes ten times faster. Other workloads benefit through different mechanisms, including projected data access, compact identifier encoding, exact structural reduction, memory-aware execution, and reuse across nearby specifications.

Detailed benchmark inputs, baselines, timing rules, and parity checks are kept in [the benchmark documentation](docs/development/benchmarks.md).

## Why econhdfe?

Consider a researcher working with 50 million rows of product-level trade data. A baseline specification might absorb exporter-year, importer-year, and product fixed effects, include several controls, and cluster inference by trade pair.

After the first regression, the actual empirical workflow rarely stops. The researcher changes the outcome, adds or removes controls, changes one fixed effect, tries another clustering rule, estimates an event-study specification, and eventually produces a large robustness table.

Econometrically, many of these specifications are close to one another. Computationally, however, a one-regression-at-a-time workflow can repeatedly pay for reading the same columns, encoding millions of categorical identifiers, expanding the same interactions, rediscovering fixed-effect structure, projecting similar variables through the same HDFE geometry, allocating large temporary arrays, and rebuilding inference objects.

Once the dataset becomes large enough, the expensive part of the workflow is often no longer the final small coefficient solve. It is data movement, representation, fixed-effect projection, memory allocation, and repeated work across specifications.

`econhdfe` is built around that observation.

```text
Traditional workflow

data
  ↓
regression
  ↓
discard intermediate work
  ↓
next regression


econhdfe

data source
  ↓
validated encoded state
  ↓
symbolic econometric design
  ↓
exact structural analysis
  ↓
shared HDFE geometry
  ↓
execution planner
  ↓
OLS / IV / PPML / IV-PPML
  ↓
reuse what remains valid
```

This design has become especially practical because the modern Python ecosystem is very different from the Python of early scientific computing. The rise of machine learning, large-scale data systems, and AI has concentrated enormous engineering investment around Python-facing high-performance infrastructure. Heavy computation does not need to run in interpreted Python loops: NumPy, SciPy, Numba/JIT, optimized BLAS/LAPACK libraries, Arrow-style columnar data, modern dataframe engines, parallel runtimes, memory-mapped storage, and optional GPU backends can all sit behind a Python interface.

`econhdfe` uses Python as the coordination layer for those numerical and data systems. The package does not try to make an interpreted loop compete with Mata or compiled code. It tries to choose a better representation, avoid unnecessary work, and send each part of the problem to the appropriate numerical backend.

The econometric model remains the constraint. Execution choices may change how the computation is performed; they must not silently change the regressors, instruments, sample, weights, fixed effects, or inference requested by the researcher.

## Quick start

Python 3.10–3.13 is supported.

```bash
pip install econhdfe
```

A standard OLS-HDFE specification can be written directly against a pandas DataFrame:

```python
from econhdfe import olshdfe

res = olshdfe(
    df,
    y="log_wage",
    x=["experience", "experience2"],
    absorb=["worker", "firm"],
    cluster=["firm"],
    vce="cluster",
)

print(res.params)
print(res.stderr)
```

The main estimator entry points are deliberately small:

| Model | Entry point |
|---|---|
| OLS-HDFE | `olshdfe(...)` |
| Linear IV-HDFE | `ivhdfe(...)` |
| PPML-HDFE | `ppmlhdfe(...)` |
| IV-PPML-HDFE | `ivppmlhdfe(...)` |

A linear IV specification, for example, looks like:

```python
from econhdfe import ivhdfe

res = ivhdfe(
    df,
    y="outcome",
    exog=["control"],
    endog=["price"],
    instruments=["cost_shifter"],
    absorb=["firm", "year"],
    estimator="2sls",
    cluster=["firm"],
    vce="cluster",
)
```

For repeated regression tables, `econhdfe` can preserve transformations that remain valid across nearby specifications:

```python
from econhdfe import OLSHDFESession

session = OLSHDFESession(
    df,
    cluster=["firm"],
    vce="cluster",
)

results = session.fit_many_y(
    y=["outcome_1", "outcome_2", "outcome_3"],
    x=["treatment", "control"],
    absorb=["firm", "year"],
)
```

The reuse rules are deliberately conservative. A different sample, relevant data state, weights, or FE geometry must not silently read stale numerical state.

### Or give it to an AI agent

`econhdfe` is designed to be **agent-native** as well as directly callable from Python.

The repository ships a package-specific Agent Skill under `skills/econhdfe/`. The README is meant to help a researcher understand the project; the Skill contains the more detailed operational knowledge needed for installation, model specification, diagnostics, advanced execution settings, benchmarking, validation, privacy-safe support, and third-party development.

A researcher using a coding agent can start with the following prompt:

> Clone or access the EconHDFE repository at `https://github.com/asggvhwgw-glitch/EconHDFE`.
>
> Locate `skills/econhdfe/` and read `skills/econhdfe/SKILL.md` before using the package. Treat that Skill and its linked references as the authoritative operational guide.
>
> Inspect my Python environment and available compute resources, install or configure the appropriate EconHDFE version and optional dependencies, and verify that the package and core estimator interfaces work. Use the Skill when choosing advanced execution settings instead of guessing parameters from general Python knowledge.
>
> Do not modify my data or econometric specification merely to make the computation faster. Performance choices must preserve the requested model. If something fails, use EconHDFE's structured diagnostics and privacy-safe support workflow.
>
> Once setup is complete, ask me for the empirical specification or research task I want to run.

The agent is an interface to the software, not part of the estimator. Samples, coefficients, fixed effects, convergence rules, inference, and numerical tolerances remain controlled by the same tested public API used by human callers.

## What econhdfe does differently

The package is designed around the full empirical workflow rather than one isolated solver. File-backed data sources can expose only the columns required by a specification; categorical identifiers can be encoded once and reused where valid; symbolic design information can be simplified before unnecessary dense matrices are created; and the execution planner can choose among certified representations according to the actual workload and resource budget.

OLS, linear IV, PPML, and IV-PPML share the same underlying HDFE infrastructure for encoding, topology, projection, degrees of freedom, weighted projection, and low-level numerical work. This avoids maintaining four unrelated fixed-effect implementations and means improvements to the common computational core can benefit multiple estimator families.

Repeated specifications are treated as a first-class use case. For many empirical projects, avoiding a second expensive fixed-effect transformation can matter more than making one small dense matrix multiplication marginally faster. `econhdfe` therefore distinguishes the econometric specification from reusable computational state and explicitly invalidates reuse when the relevant sample or FE geometry changes.

The package also separates execution policy from econometric semantics. Memory budgets, thread counts, structured representations, and cache strategies are computational decisions. They are not allowed to redefine the model.

## Beyond engineering

`econhdfe` is not only an engineering project. Building a general multiway HDFE system exposes structural problems that cannot be solved by caching or parallelism alone.

One example is the absorbed degrees of freedom associated with three or more categorical fixed-effect dimensions. If \(D_{FE}\) denotes the combined fixed-effect design matrix, the relevant quantity is

\[
\mathrm{DoF}_{FE} = \mathrm{rank}(D_{FE}).
\]

For one or two FE dimensions, redundancy has familiar graph-based structure. For general multiway fixed effects, exact structural rank is substantially more difficult. `econhdfe` contains a theorem-backed framework for computing the exact structural rank and absorbed DoF of arbitrary-`G` categorical FE designs within its documented assumptions and resource bounds.

That distinction matters because numerical simplification and econometric rank are not the same object. The implementation keeps the requested FE topology, structural rank/DoF reasoning, numerical representation, and projection solver conceptually separate so that an optimization cannot silently redefine the design being estimated.

The project currently tracks two other theorem-backed pieces of work. An exact residual-core reduction can eliminate eligible parts of the multiway FE incidence structure before the expensive numerical solve and reconstruct them afterwards while preserving the target projection. An exact partition-refinement reduction can detect nested and redundant categorical structure before full materialization, allowing the same requested column space to be represented by a smaller exact basis.

The core mathematical statements of these three theorem-backed lines are now machine-checked in Lean 4 under their documented assumptions. Formal statements, proof coverage, implementation mappings, tests, and prior-art boundaries are maintained separately in the [technical documentation](docs/technical/README.md) and [formal verification status](docs/technical/formal-verification.md). This does not amount to formal verification of the Python implementation, floating-point execution, benchmark claims, or historical originality: mathematical correctness, implementation correctness, numerical validation, and independent priority are treated as different claims.

## Main capabilities

| Area | Current support |
|---|---|
| OLS-HDFE | General multiway FE absorption, IID/robust covariance, multiway clustering, HAC and Driscoll-Kraay paths |
| Linear IV-HDFE | 2SLS, LIML, k-class, two-step GMM and conventional weak-IV diagnostics |
| PPML-HDFE | IRLS, high-dimensional FE projection, separation handling, robust and clustered inference |
| IV-PPML-HDFE | Additive-moment IV-PPML on the shared weighted HDFE infrastructure |
| Fixed effects | Multiway categorical FE, interactions, heterogeneous slopes, optional exact structural rank/DoF |
| Repeated specifications | Reusable OLS and linear-IV sessions with FE/sample-aware invalidation |
| Post-estimation | Publication results, linear contrasts/Wald, chunked OLS/IV predictions and explicitly saved categorical FE (0.7.0) |
| Execution | Projected data access, memory budgets, structured representations, bounded parallelism and automatic HDFE thread selection |
| Compatibility | `reghdfe`, `ivreghdfe`, and legacy `pyreghdfe` entry points |
| Agent interface | Native `econhdfe` Skill for agent-driven empirical and development workflows |

## Validation

Performance is not useful if an optimization silently changes the model. `econhdfe` therefore separates public-interface testing, statistical behavior, independent numerical checks, and release engineering.

The current test suite is organized around **contracts, behavior, numerics, and tooling** rather than historical version numbers. Statistical tests cover realized samples, weights, singleton handling, FE semantics, omitted variables, clustering, DoF, PPML separation, result semantics, and cache invalidation. Critical numerical paths are additionally checked against independent constructions such as dense dummy matrices, SVD-based references, exact rational arithmetic, deliberately difficult rank cases, extreme scales, rank deficiency, and resource-boundary failures.

CI covers Linux, Windows, and macOS across supported Python versions, together with feasible minimum dependency environments. Source archives and installed wheels are also tested outside the ordinary development import path.

The claim boundary remains explicit. Extensive repository-local testing is not equivalent to complete licensed-Stata or upstream external-corpus certification, and those external checks are not claimed merely because the internal suite is large.

See [Testing](docs/development/testing.md) and the [validation status](docs/development/test-status.md) for details.

## Post-estimation and prediction

Version 0.7.0 provides `linear_combination()` and `wald_test()`,
plus chunked predictions for standard OLS and linear IV. `result.predict()`
returns fitted values for the estimation sample; `restore_sample=True` restores
original row positions. New-data `kind="xb"` uses the frozen design, while
`kind="stdp"` includes beta covariance only, not fixed-effect uncertainty.
New-data `response` and `fe` require supported categorical fixed effects saved
explicitly with `save_fe=True`; unknown levels and unidentified combinations
are rejected. PPML/IV-PPML prediction, varying-slope or group+individual FE
prediction, margins and AME are not supported in this version.
See the [post-estimation contract](docs/development/postestimation-0.7.md).
These features require 0.7.0. Check `python -c "import econhdfe; print(econhdfe.__version__)"` after installation. Use a published 0.7.0 wheel when available, or run `python -m pip install .` from this source checkout; the package index may still serve an earlier release. See [release status and validation](docs/development/test-status.md).

## Fixed-effect recovery

Some applications need fixed effects as economic objects rather than nuisance parameters. Worker-firm models, mobility applications, origin-destination models, and structural gravity are common examples.

`econhdfe.effects` provides component-aware recovery of identified categorical or indicator fixed effects with explicit normalization. Identification and normalization are treated separately: changing a normalization can change reported levels, but it cannot create identification where none exists.

See [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md).

## Optional Python IV-GMM grid companion

[ivgmmgrid](contrib/ivgmmgrid/README.md) evaluates repeated two-step clustered IV-GMM specifications with one absorbed fixed effect and one endogenous candidate per fit. Its computation and public-data replay run entirely in Python: `python -m pip install ./contrib/ivgmmgrid`. It is a focused grid-search companion, with a distinct covariance convention, rather than a replacement for the main IV API. GPL-3.0-only; Stata scripts are optional validation aids only. See [real-data parity and scope](contrib/ivgmmgrid/VALIDATION.md).

## Compatibility

New Python code should normally import directly from `econhdfe`:

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

Compatibility entry points are also available:

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

These aliases are intended to ease migration and interoperability. They do not imply that every Stata syntax construct or every upstream edge case has completed external certification.

## Documentation

The main documentation index is [docs/README.md](docs/README.md). The most useful technical entry points are the [technical overview](docs/technical/overview.md), [performance architecture](docs/technical/performance-architecture.md), [economics-first architecture](docs/development/architecture.md), [execution planner](docs/development/execution-planner.md), [benchmark documentation](docs/development/benchmarks.md), and [technical innovation documentation](docs/technical/README.md).

Current development priorities are tracked in [TODO.md](TODO.md). Migration information between public releases is kept under [docs/release/](docs/release/).

## Contributing

Contributions should preserve the separation between econometric semantics, HDFE mathematics, estimator-specific equations, inference, data preparation, numerical kernels, and execution policy. Performance changes should carry numerical parity checks as well as timing evidence, and new technical claims should include their mathematical statement, implementation correspondence, tests, and prior-art boundary.

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer guide](skills/econhdfe/references/developer-guide.md).

## License

The EconHDFE core is BSD licensed; see [LICENSE](LICENSE). Optional companion packages and data fixtures carry the licenses declared in their own directories.
