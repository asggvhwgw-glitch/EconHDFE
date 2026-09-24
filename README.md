# econhdfe

**High-performance high-dimensional fixed-effect econometrics for empirical research.**

`econhdfe` provides OLS-HDFE, linear IV-HDFE, PPML-HDFE, IV-PPML-HDFE, modern inference utilities, repeated-specification workflows, and identified recovery of categorical fixed effects on top of one shared HDFE and execution core.

The package is designed around a practical empirical problem: once a dataset becomes large, the expensive part of a regression table is often no longer the final small linear solve. Time and memory are spent repeatedly reading columns, encoding high-cardinality identifiers, expanding interactions, projecting out fixed effects, rebuilding weighted projectors, and rerunning nearly identical specifications. `econhdfe` treats those operations as shared computational infrastructure rather than reimplementing them estimator by estimator.

> **Project status:** v0.6.3 is an alpha research release. The local release suite and package-level parity gates are extensive, but licensed-Stata / upstream external certification remains a separate release boundary. See [Validation and claims](#validation-and-claims).

## Why econhdfe exists

A typical empirical project may combine millions of observations, several high-dimensional fixed effects, clustered inference, factor-variable expansions, event-study terms, and dozens of nearby robustness specifications. A conventional workflow can repeatedly pay for the same work:

```text
raw data
  -> read many columns
  -> encode firm / worker / city / year IDs
  -> expand interactions
  -> build a wide design matrix
  -> absorb the same FE geometry
  -> solve one specification
  -> throw most intermediate work away
  -> repeat
```

`econhdfe` reorganizes that workflow:

```text
                    exact specification / structure certificates
                                   │
raw data -> Data Layer -> symbolic design -> HDFE core -> estimator
                │              │          │          │
                │              │          │          └─ OLS / IV / PPML / IV-PPML
                │              │          └─ shared projection / DoF / topology
                │              └─ dense or structured physical representation
                └─ projected reads / encoded IDs / repeated-workflow reuse
                                   │
                              Execution Planner
                          memory / threads / representation
```

The econometric specification stays fixed. The package changes **how the same requested model is executed**, and falls back to established dense/reference paths whenever a faster representation is not certified or not worthwhile.

## Main capabilities

| Area | Current support |
|---|---|
| Linear HDFE | OLS, robust/IID covariance, multiway clustering, HAC and Driscoll-Kraay paths |
| Linear IV-HDFE | 2SLS, LIML, k-class, GMM paths and conventional weak-IV diagnostics |
| PPML-HDFE | IRLS, HDFE projection, FE/simplex/ReLU separation, robust and multiway-cluster inference |
| IV-PPML-HDFE | Additive-moment IV-PPML with shared weighted HDFE infrastructure |
| Fixed effects | 2-way and general multiway categorical FE absorption; exact optional 3+ FE structural rank/DoF |
| Heterogeneous specifications | Factor interactions, event-study-style terms and group-specific slopes with conservative structured execution when certified |
| Repeated regressions | Reusable OLS/linear-IV sessions for changing outcomes, controls and FE sets; optional file-backed persistent reuse |
| Identified FE recovery | Post-estimation categorical/indicator FE recovery with component-aware identification and user-selectable normalization |
| Inference | IID/robust/multiway-cluster covariance, cluster diagnostics, one-way OLS WCR11/WCU11 wild-cluster tests |
| Execution | Memory budgets, bounded parallelism, automatic HDFE thread calibration, dense/block representation planning |
| Compatibility | `reghdfe`, `ivreghdfe`, and legacy `pyreghdfe` compatibility entry points |

## Installation

Python 3.10-3.13 is supported.

From a checked-out source tree:

```bash
pip install .
```

Optional I/O backends:

```bash
pip install ".[io]"
```

Development/test dependencies:

```bash
pip install ".[test]"
```

Optional CUDA support is packaged separately:

```bash
pip install ".[gpu]"
```

The core runtime depends on NumPy, SciPy, pandas, Numba, joblib, and threadpoolctl. The wheel does not include the formal technical manuscripts; those remain in the source/release bundle for auditability.

## Quick start

### OLS-HDFE

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
    estimator="liml",
    vce="robust",
)
```

### PPML-HDFE

```python
from econhdfe import ppmlhdfe, PPMLConfig

res = ppmlhdfe(
    trade,
    X,
    absorb=[exporter_year, importer_year, pair],
    clusters=[pair],
    vce="cluster",
    config=PPMLConfig(engine="optimized"),
)
```

### IV-PPML-HDFE

```python
from econhdfe import ivppmlhdfe

res = ivppmlhdfe(
    y,
    exog=controls,
    endog=endogenous,
    instruments=instruments,
    absorb=[firm, year],
    vce="robust",
)
```

## Optional quantile-regression companion: BoundedQR

[BoundedQR](contrib/boundedqr/README.md) adds separately installed clustered wild-gradient quantile regression, a CPU Python API and an R bridge (GPL-3.0-or-later). It complements the mean/IV/PPML estimators with a different estimand. It requires an explicit full-rank design and does not absorb HDFE or justify demeaning quantile regressions. See its guide for runnable examples and numerical limits.

## Recovering fixed effects as economic objects

Most HDFE regressions treat fixed effects as nuisance parameters. Some research designs do not: AKM-style worker/firm models, origin-destination mobility models, structural gravity, and two-stage models may need the FE coefficients themselves.

Fixed-effect recovery is therefore kept outside the estimator core:

```python
from econhdfe.effects import recover_linear_result, NormalizationSpec

fx = recover_linear_result(
    res,
    df[["experience", "experience2"]].to_numpy(),
    [df["worker"].to_numpy(), df["firm"].to_numpy()],
    x_names=["experience", "experience2"],
    fe_names=["worker", "firm"],
    normalization=NormalizationSpec(
        "weighted_mean_zero",
        baseline="worker",
    ),
)

firm_fx = fx.term("firm")
```

Recovery is deliberately limited to **categorical/indicator intercept FE**. Pure categorical interactions such as `firm#year` are in scope; varying-slope recovery such as `firm#c.age` is not.

Identification and normalization are separate contracts:

- recursive singletons and PPML separation are diagnosed on the realized estimation sample;
- disconnected but internally identified components are recovered separately;
- if one independent block is unidentified, identified blocks are retained while the bad block's FE coefficients are returned as `NaN` with diagnostics;
- if no independent block is recoverable, strict recovery raises `FixedEffectIdentificationError`;
- reference, mean-zero, and weighted-mean-zero normalizations change reported levels but cannot create identification.

See [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md).

## Where the performance comes from

The performance work is intentionally modular. No single solver explains the package.

### 1. Read and encode only what the specification needs

The Data Layer projects required columns from supported file-backed sources, encodes identifier-only FE/cluster columns compactly, and exposes reusable estimation-ready state. OLS/linear-IV research sessions can reuse validated encoded columns and within-transformed columns across nearby specifications.

This matters for the common workflow "change `y`, add/drop a control, change an FE set". It avoids turning every regression-table column into a fresh data-ingestion and FE-encoding job.

### 2. Remove exact structure before numerical work

The symbolic design compiler recognizes exact partition refinement, nested categorical structure, redundant FE partitions and supported interaction structure before dense materialization. Exact structural reductions preserve the requested combined column space; they do not change the model.

For interaction-rich designs, the package can keep local coefficients in a row-partitioned representation instead of constructing a wide dense matrix full of structural zeros.

### 3. Share one HDFE core across estimators

OLS, IV, PPML and IV-PPML reuse the same FE encoding, canonicalization, projection, DoF and topology machinery. Improvements to multiway FE projection, weighted projector reuse or memory behavior therefore apply across estimator families.

### 4. Plan execution separately from econometrics

The Execution Planner separates:

```text
exactness certificate -> cost/resource estimate -> execution policy
```

A representation is used only when it is both valid and expected to help. Memory budgets, row/block representation and thread counts are execution choices; they do not alter regressors, FE, instruments, samples, weights or inference.

`threads="auto"` calibrates only on sufficiently large HDFE workloads. Small jobs remain single-threaded so calibration and parallel overhead cannot dominate the regression.

### 5. Reuse work across empirical specifications

`OLSHDFESession` and `IVHDFESession` cache only transformations that are valid for the same FE/sample geometry. A different FE combination gets a different cache entry. PPML/IV-PPML iteration states are not treated as interchangeable with linear HDFE state; PPML warm starts change only the starting predictor, not the converged estimating equations.

A deeper explanation is in [Performance architecture](docs/technical/performance-architecture.md).

## Performance evidence

Performance numbers in this repository are **same-host development evidence**, not universal hardware claims and not substitutes for external Stata/R package benchmarking.

A representative structured-design microbenchmark (recorded in [`benchmarks/heterogeneous_spec_models_96k.json`](benchmarks/heterogeneous_spec_models_96k.json)) uses 96,000 observations, 12 groups and 8 local slopes per group. The physical dense design is about 75.3 MB; the planned block representation is about 7.7 MB. Against the package's own dense path, the measured speedups were:

| Estimator | Structured / dense speedup | Max coefficient difference |
|---|---:|---:|
| OLS | 10.51x | 3.1e-16 |
| Linear IV | 1.30x | 3.5e-15 |
| PPML | 6.42x | 5.8e-16 |
| IV-PPML | 2.07x | 1.7e-15 |

That benchmark is deliberately narrow: it measures an interaction-rich design for which a structured representation is certified. It should **not** be read as "econhdfe is 10x faster on every regression."

The recorded v0.5.0 -> v0.6.0 release regression matrix ([`benchmarks/effects/v060_pre_release/summary.json`](benchmarks/effects/v060_pre_release/summary.json); 120,000 observations, three alternating-process rounds, seven OLS/IV/PPML/IV-PPML cases) reports exact coefficient/SE/VCOV and applicable result-metadata parity across versions, with median timing ratios between **0.775x and 1.026x** and no systematic estimator regression. The ongoing calibration guard is [`benchmarks/planner/final_calibration/summary.json`](benchmarks/planner/final_calibration/summary.json).

All benchmark interpretation rules and machine-readable evidence are indexed in [`docs/development/benchmarks.md`](docs/development/benchmarks.md) and `benchmarks/`.

## Architecture in one sentence

`econhdfe` separates **econometric semantics** from **data preparation**, **exact structural simplification**, **numerical HDFE projection**, and **runtime execution policy**.

The main ownership boundaries are:

- `models/` — estimator equations and estimator-specific diagnostics;
- `hdfe/` — FE representation, topology, projection, exact rank/DoF and specialized solvers;
- `design.py` / `design_structure.py` — symbolic regressors and exact pre-materialization structural reduction;
- `data/` — projected/encoded data and safe repeated-workflow reuse;
- `planner/` — estimator-agnostic memory/thread/representation policy;
- `compute/` — low-level kernels, linear algebra and covariance primitives;
- `iv/` — shared instrument-role and weighted-IV primitives;
- `inference/` / `resampling/` — post-estimation inference and repeated-draw mechanics;
- `effects/` — post-estimation identified categorical FE recovery;
- `frontend/` — role-aware validation and preflight reporting.

See [Technical overview](docs/technical/overview.md) and the generated [architecture map](docs/development/architecture-map/architecture.md).

## Technical contributions and prior-art boundary

`econhdfe` deliberately does **not** label every difficult implementation as a research contribution. The current registry contains exactly three entries, classified as `theorem_backed_framework` / `theorem_backed_application`:

1. **Exact arbitrary-G categorical FE structural rank / absorbed DoF** — a certified exact computational framework for general multiway categorical FE rank.
2. **Exact arbitrary-G residual-core reduction for HDFE projection** — an exact multiway hypergraph leaf-elimination/reconstruction result for eligible weighted projection problems.
3. **Exact partition-refinement HDFE structural design reduction** — exact FE/design canonicalization and dependency-aware pre-materialization basis reduction.

Each registered entry ships with a formal LaTeX manuscript, compiled PDF, implementation map, tests and benchmark/exact-oracle evidence. The registry records mathematical support, **not proven novelty**: every entry carries `originality_status: not_independently_established`, and proof/implementation review under stated assumptions is documented in the [0.6.1 mathematical review](docs/technical/mathematical-review-0.6.1.md). Established methods—OLS, 2SLS/LIML/GMM, PPML-HDFE, IV-PPML, Schur/CG/LSMR, clustered covariance, wild-cluster bootstrap, TSQR/block-angular least squares, caching, threading and execution planning—retain their upstream attribution and are **not** claimed as original econhdfe methods.

See [`docs/technical/innovation-audit.md`](docs/technical/innovation-audit.md) and [`docs/technical/innovation-registry.json`](docs/technical/innovation-registry.json).

## Validation and claims

The source tree contains regression tests, release gates, public-contract snapshots, architecture freshness checks, benchmark evidence and installation smokes. The 0.6.x line also tests FE-recovery identification, normalization invariance, singleton handling and PPML finite-MLE/separation support.

Important claim boundary:

- repository-local and source-archive validation is extensive;
- same-host performance evidence is retained with exact benchmark inputs and parity checks;
- existing top-level estimator APIs are guarded by compatibility snapshots;
- **licensed Stata golden parity and upstream external-corpus certification are separate external gates and are not claimed as newly completed for 0.6.3.**

See [`docs/development/test-status.md`](docs/development/test-status.md), [`docs/development/external-validation.md`](docs/development/external-validation.md), and [`docs/development/statistical-parity.md`](docs/development/statistical-parity.md).

## Documentation

Start at [`docs/README.md`](docs/README.md).

For most contributors and advanced users:

- [Technical overview](docs/technical/overview.md)
- [Performance architecture](docs/technical/performance-architecture.md)
- [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md)
- [Economics-first architecture](docs/development/architecture.md)
- [Execution Planner](docs/development/execution-planner.md)
- [Data Layer](docs/development/data-layer.md)
- [Economics-facing module map](docs/development/economic-module-map.md)
- [Technical-innovation audit](docs/technical/innovation-audit.md)
- [Release validation status](docs/development/test-status.md)

The repository also ships a portable Agent Skill under `skills/econhdfe/` for installation/configuration guidance, empirical workflows, validation, benchmarking, support reports and third-party development.

## Compatibility

New code should import from `econhdfe`.

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

Compatibility aliases remain available:

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

Compatibility does not imply that every Stata syntax construct or every upstream edge case has already completed external certification. Supported behavior is documented and release-tested in the Python package.

## Contributing

Changes should preserve the package boundaries above. In particular:

- estimator-specific logic belongs in `models/`, not in `hdfe` or `planner`;
- `planner/` must not decide econometric specifications;
- data/cache reuse must be keyed to the exact sample/FE/design geometry it represents;
- a new technical-innovation claim requires a prior-art audit, formal manuscript, implementation/test/evidence mapping, and innovation-registry entry in the same release;
- benchmarks must report their workload and baseline explicitly rather than turning one microbenchmark into a general speed claim.

Developer workflow and release gates are documented under `docs/development/`, `docs/release/`, and `skills/econhdfe/references/developer-guide.md`.

## License

The EconHDFE core is BSD-licensed; see [`LICENSE`](LICENSE). Optional companion projects under `contrib/` retain their own directory-level licenses and are installed separately. They are not included in the core Python wheel.
