# econhdfe

**High-performance high-dimensional fixed-effect econometrics for empirical research in Python.**

`econhdfe` provides OLS-HDFE, linear IV-HDFE, PPML-HDFE, and IV-PPML-HDFE on top of a shared fixed-effect, inference, and execution core. It is built for the way empirical researchers actually work: large datasets, several high-dimensional fixed effects, clustered inference, and many nearby specifications in the same regression table.

> **Project status:** active **alpha** research software. The repository may be ahead of the latest published release. Published versions are available from PyPI and GitHub Releases. Repository-local validation is extensive, but licensed-Stata / upstream external certification remains a separate boundary.

## Why econhdfe

On a large empirical dataset, the final coefficient solve is often not the expensive part. Time and memory are spent repeatedly:

- reading the same columns;
- encoding high-cardinality firm, worker, product, city, or year identifiers;
- expanding interactions;
- absorbing the same fixed-effect geometry;
- rebuilding weighted projectors;
- and rerunning nearly identical robustness specifications.

`econhdfe` treats those operations as shared computational infrastructure rather than rebuilding them estimator by estimator.

The central design rule is simple:

> **The econometric specification determines the model. The execution layer only decides how to compute that same model efficiently.**

That separation makes it possible to optimize data access, fixed-effect projection, repeated specifications, memory use, and parallel execution without silently changing regressors, instruments, samples, weights, or inference.

## Installation

Python **3.10–3.13** is supported.

```bash
pip install econhdfe
```

Optional I/O support for file-backed workflows:

```bash
pip install "econhdfe[io]"
```

Optional CUDA dependencies:

```bash
pip install "econhdfe[gpu]"
```

From a checked-out source tree:

```bash
pip install .
```

## Quick start

### OLS with high-dimensional fixed effects

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

The same public API accepts ordinary categorical fixed effects, interaction-rich designs, weights, multiway clustering, and explicit inference/execution configuration.

### Linear IV-HDFE

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

`estimator=` also supports LIML, k-class, and two-step GMM paths.

### PPML-HDFE

```python
from econhdfe import ppmlhdfe

res = ppmlhdfe(
    y,
    X,
    absorb=[exporter_year, importer_year, pair],
    clusters=[pair],
    vce="cluster",
)
```

PPML includes high-dimensional FE projection, separation handling, robust inference, and multiway clustering.

### IV-PPML-HDFE

```python
from econhdfe import ivppmlhdfe

res = ivppmlhdfe(
    y,
    exog=controls,
    endog=endogenous,
    instruments=instruments,
    absorb=[firm, year],
    clusters=[firm],
    vce="cluster",
)
```

IV-PPML uses an additive-moment formulation with the same shared weighted HDFE infrastructure.

## What is supported

| Area | Current support |
|---|---|
| OLS-HDFE | Multiway categorical FE absorption, robust/IID covariance, multiway clustering, HAC and Driscoll-Kraay paths |
| Linear IV-HDFE | 2SLS, LIML, k-class, two-step GMM, conventional weak-IV diagnostics |
| PPML-HDFE | IRLS, high-dimensional FE projection, separation checks, robust and clustered inference |
| IV-PPML-HDFE | Additive-moment IV-PPML with shared weighted HDFE machinery |
| Fixed effects | General multiway categorical FE, interactions, heterogeneous slopes, optional exact structural rank/DoF machinery |
| Repeated specifications | Reusable OLS and linear-IV sessions for changing outcomes, controls, and FE sets |
| Inference | IID, robust, multiway-cluster covariance, cluster diagnostics, selected wild-cluster procedures |
| Post-estimation | Publication-oriented result objects and identified recovery of categorical/indicator fixed effects |
| Execution | Memory budgets, bounded parallelism, automatic HDFE thread selection, dense/structured execution planning |
| Compatibility | `reghdfe`, `ivreghdfe`, and legacy `pyreghdfe` compatibility entry points |

## Repeated regression tables

A common empirical workflow holds most of the specification fixed while changing outcomes, controls, or fixed effects. Recomputing every transformation from scratch wastes work.

`OLSHDFESession` and `IVHDFESession` reuse only state that is valid for the same realized sample and FE geometry.

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

Changing the FE structure or sample creates a different cache identity; stale transformed state is not silently reused. Persistent file-backed reuse is available for exploratory workflows, but it is an execution optimization rather than part of estimator semantics.

## Where the performance comes from

There is no single "fast solver" behind the package. Performance comes from keeping several layers separate.

### 1. Read and encode only what the model needs

The data layer can project required columns from supported file-backed sources and compactly encode identifier-heavy FE/cluster columns. Repeated linear workflows can reuse validated encoded and within-transformed state.

### 2. Remove exact structure before numerical work

The symbolic design layer detects exact nesting, partition refinement, redundant FE structure, and supported interaction structure before unnecessary dense materialization. Exact reductions preserve the requested column space.

### 3. Share one HDFE core across estimators

OLS, IV, PPML, and IV-PPML reuse common FE encoding, topology, projection, DoF, and low-level numerical infrastructure. Improvements therefore do not need to be reimplemented independently for every estimator.

### 4. Separate execution policy from econometrics

The execution planner decides between valid physical representations, memory budgets, and thread counts only after the statistical structure is fixed.

```text
specification
    ↓
exact structural information
    ↓
cost / resource estimate
    ↓
execution plan
    ↓
estimator
```

### 5. Reuse work across nearby specifications

For regression tables and robustness exercises, safe reuse can matter more than optimizing one isolated matrix operation.

For implementation details and benchmark evidence, see:

- [Performance architecture](docs/technical/performance-architecture.md)
- [Benchmark index and interpretation rules](docs/development/benchmarks.md)
- [Execution planner](docs/development/execution-planner.md)

Repository benchmarks are workload-specific development evidence, not universal speed claims.

## Fixed-effect recovery

Most regressions treat fixed effects as nuisance parameters, but some applications need the FE coefficients themselves—for example worker/firm models, mobility models, or structural gravity.

`econhdfe.effects` provides component-aware recovery of **identified categorical/indicator fixed effects** with explicit normalization. Identification and normalization are kept separate, and unidentified components are not converted into arbitrary coefficient levels.

See [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md).

## Validation and claim boundaries

The repository maintains separate checks for:

- public API contracts;
- statistical behavior and failure semantics;
- independent numerical oracles and resource-boundary cases;
- Linux, Windows, and macOS across supported Python versions;
- feasible minimum dependency versions;
- source and installed-wheel behavior;
- release artifacts and reproducibility metadata.

The test suite is organized around **contracts, behavior, numerics, and tooling**, not historical version numbers. See [Testing](docs/development/testing.md).

Two boundaries are intentionally explicit:

1. **Package-local validation is not the same as licensed-Stata / upstream external certification.**
2. **A formal mathematical result is not automatically a proven novelty claim.**

Technical manuscripts, implementation mappings, prior-art boundaries, and the originality registry are kept under [technical documentation](docs/technical/README.md).

## Documentation

Start with the repository [documentation index](docs/README.md).

Useful entry points:

- [Technical overview](docs/technical/overview.md) — estimator and computational contracts
- [Performance architecture](docs/technical/performance-architecture.md) — large-data design and reuse strategy
- [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md) — FE recovery and normalization
- [Economics-first architecture](docs/development/architecture.md) — module ownership and package structure
- [Testing](docs/development/testing.md) — contract / behavior / numerics / tooling test model
- [Migration notes](docs/release/migration.md) — public release migration
- [Roadmap](TODO.md) — current development priorities

The repository also ships a portable Agent Skill under `skills/econhdfe/` for installation, empirical workflows, validation, support reports, and third-party development.

## Compatibility

New code should import directly from `econhdfe`:

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

Compatibility aliases are also available:

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

Compatibility does **not** mean that every Stata syntax construct or every upstream edge case has completed external certification.

## Contributing

Contributions should preserve the separation between econometric semantics, HDFE numerics, inference, data preparation, and execution policy.

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [developer guide](skills/econhdfe/references/developer-guide.md).

## License

BSD licensed. See [LICENSE](LICENSE).
