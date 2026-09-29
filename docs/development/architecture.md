# econhdfe architecture — 0.6.0

`econhdfe` is organized around **econometric responsibilities**, not around numerical algorithms or Stata command names. The data layer first ensures that only variables required by the empirical specification enter the runtime; HDFE is the shared solution to nuisance high-dimensional heterogeneity; IV is the shared solution to exclusion-restriction/endogeneity mechanics; outcome-model implementations live under `models`; the compute layer exists only to execute these econometric objects efficiently.

A detailed module-by-module economics-facing map is maintained in [`economic-module-map.md`](economic-module-map.md). It states, for every runtime Python module, the empirical/econometric problem served by the module and only then its computational responsibility.

## Economics-first architecture

| Empirical problem | Package responsibility | Main modules |
|---|---|---|
| Raw wide microdata and common estimation samples | Econometric data access projects only requested variables, plans bounded-memory batches, keeps categorical identities stable across chunks, and records sample exclusions | `data/`, `frontend/columns.py` |
| Linear partial effects with many nuisance controls | OLS-HDFE partials out firm/time/geography/industry effects and estimates the remaining slopes | `models/ols.py`, `hdfe/` |
| Endogenous regressors | Linear IV supplies 2SLS/LIML/k-class/GMM and weak-identification diagnostics; generic IV primitives represent exclusion restrictions | `models/linear_iv/`, `iv/` |
| Nonnegative multiplicative outcomes, especially trade flows | PPML-HDFE estimates multiplicative conditional means while retaining zeros and absorbing HDFE | `models/ppml/`, `models/poisson.py`, `hdfe/` |
| Endogeneity in multiplicative/PPML models | IV-PPML combines excluded instruments with the PPML mean and HDFE; SPJ/bootstrap are model-specific | `models/ppml_iv/`, `iv/`, `hdfe/` |
| Firm/time/industry/etc. nuisance heterogeneity | HDFE removes nuisance FE without explicit dummy expansion and accounts for absorbed DoF | `hdfe/absorber.py`, `rank.py`, `dof.py` |
| Recovered FE as economic objects | A separate effects layer recovers additive categorical FE, audits realized-sample identification, and applies reporting normalization without changing the underlying estimator | `effects/` |
| Event studies and group-specific slopes with many explicit heterogeneous coefficients | Heterogeneous-specification optimization preserves the requested coefficients but avoids materializing structural zeros when exact and worthwhile | `design.py`, `compute/design_plan.py`, `compute/block_design.py`, `compute/partitioned_lstsq.py`, model `heterogeneous.py` consumers |
| Correlated shocks and few/unbalanced clusters | Shared covariance plus cluster diagnostics/wild-cluster procedures provide inference at the economic dependence level | `compute/vcov.py`, `inference/cluster/` |
| Regression-table robustness exercises | Session/cache infrastructure reuses FE/sample work across specifications without reusing invalid specification-specific diagnostics | `sessions.py`, `compute/context.py` |
| Post-estimation of fitted linear models | Coefficient restrictions and future predictions reuse the exact reported parameter order, realized sample and frozen design/FE semantics rather than reconstructing them from mutable estimator internals | `postestimation.py`, `prediction.py`, `results.py` |
| Large-sample feasibility | A shared execution planner separates exactness certificates from memory/representation/thread policy, then compute/HDFE kernels execute the chosen plan without changing the estimand | `planner/`, `compute/`, `hdfe/projection.py` |

The architecture therefore follows the sequence **economic specification → data requirements/sample state → canonical design and nuisance structure → estimator → inference**. A separate execution-planner layer acts across data, HDFE and compute stages: certificates answer whether an optimization is exact, while cost/resource policy answers whether an exact optimization is worthwhile. Numerical objects such as QR, MAP, PCG, TSQR or sparse/block storage are implementation mechanisms, not top-level package concepts.

## Visual architecture map

The generated contributor-facing map lives under `docs/development/architecture-map/`:

- `architecture.md`: GitHub-readable Mermaid/code map;
- `architecture.html`: self-contained interactive dependency/flow view;
- `architecture.json`: machine-readable AST/import evidence.

Regenerate after structural changes with `python scripts/generate_architecture_map.py`. CI/release checks may run the same command with `--check`; do not hand-edit the generated dependency topology.

For an economics-facing explanation of every runtime module, use `docs/development/economic-module-map.md`; the generated map remains the source of truth for code dependencies.

## Dependency rule

```text
frontend ─────► data
   │             ▲
   └──── specification/column requirements

models ───────────────► hdfe ───────────────► compute
  │                      │                       │
  ├──────────────► iv ───┘                       ▼
  ├──────────────► data ───────────────────► planner
  └──────────────► resampling                 ▲
                         hdfe ─────────────────┘

`planner` is estimator-agnostic infrastructure: data, HDFE and compute may report resource/candidate metadata to it, while the planner must not import estimator/data/HDFE implementations.
```

The direction is strict:

- `data` owns source scanning, batching, stable encoding and sample-state infrastructure; it never imports estimator models. A thin specification bridge may consume frontend column requirements.
- `planner` owns execution certificates/resource policy only. It imports no data, HDFE, IV or model implementation and cannot define statistical eligibility by inspecting estimator internals. Its calibration sublayer may benchmark only package-generated synthetic memory traffic, and its developer-report sublayer may expose only aggregate runtime/planner metadata.
- `compute` depends on no estimator, HDFE, IV, or data-source layer.
- `hdfe` may depend on `compute`, but not on `iv` or `models`.
- `iv` is outcome-model agnostic and may depend only on `compute`; it must not import HDFE or any model implementation.
- `models` owns estimating equations and may compose `hdfe`, `iv`, and `compute`.
- `resampling` is outcome-model agnostic and must not import `models`, `hdfe`, or `iv`; model-specific bootstrap definitions call into it.
- `frontend` provides cheap role-aware input checks and structured reports; `errors` provides the public failure taxonomy.
- root-level `pipeline`, `design`, `results`, `postestimation`, `prediction`, and `collinearity` are thin orchestration/shared-interface modules; root `bootstrap.py` is compatibility-only.

The stronger IV rule is deliberate: endogeneity is not a synonym for linear 2SLS. Linear IV and future IV-PPML should consume the same instrument/moment/weighted-2SLS primitives without depending on one another.

## Package layout

```text
econhdfe/
  api.py
  pipeline.py
  results.py
  postestimation.py / prediction.py
  design.py / design_structure.py
  collinearity.py
  errors.py
  config.py                 # HDFE / inference / execution strategy configs
  reporting.py              # publication result protocol/table helpers
  sessions.py               # repeated linear OLS/IV specification lifecycle
  frontend/
    roles.py / validate.py / report.py / columns.py
  data/
    source.py              # projected/batched DataFrame, CSV, Stata, Parquet sources
    planner.py             # ingestion batch and memory planning
    dataset.py             # materialize only required econometric columns
    encoding.py            # stable categorical codes across chunks
    sample.py              # monotone estimation-sample state
  resampling/
    engine.py / sampling.py / results.py
  inference/
    cluster/
      diagnostics.py         # cluster balance/score concentration diagnostics
      wild.py                # one-way OLS WCR11/WCU11
      results.py
  bootstrap.py               # compatibility re-exports

  models/
    ols.py
    linear_iv/
      api.py                 # public linear IV-HDFE orchestration
      estimators.py          # 2SLS / LIML / k-class / GMM2S
      diagnostics.py         # first stage, CD, KP, SW, over-ID
      stock_yogo.py
    ppml/
      estimator.py / irls.py
      separation*.py
      standardize.py
      config.py / results.py / vce.py
    ppml_iv/
      api.py / estimator.py / irls.py / vce.py
      standardize.py / config.py / results.py
      bias.py                  # Class A/B/C SPJ bias correction
      bootstrap.py             # model-specific cluster bootstrap

  iv/
    design.py                # IVDesign: exog / endog / excluded roles
    solve.py                 # outcome-agnostic weighted 2SLS inner solve
    moments.py               # additive IV score/moment primitives

  effects/                 # post-estimation categorical FE recovery; no estimator ownership
    topology.py             # connected components + exact rank/nullity metadata
    diagnostics.py          # singleton/separation/data-induced identification explanations
    recover.py              # Schur/LSMR decomposition + normalization
    adapters.py             # OLS/IV and PPML/IV-PPML target adapters
    results.py              # level-sized effect/result contracts

  hdfe/
    specs.py / encoding.py
    structure.py / plan.py
    absorber.py / numerical_core.py / projection.py
    two_way.py / weighted_projection.py
    group_individual.py / dof.py

  planner/
    contracts.py             # exactness, memory, representation, parallel contracts
    resources.py             # shared CPU/memory resource discovery
    core.py                  # memory/representation/thread policy

  compute/
    backend.py / encoding.py / clusters.py / kernels.py
    linalg.py / wls.py / vcov.py / weights.py
    runtime.py / execution.py / context.py
```

## Econometric data-layer boundary

The data layer answers a narrower question than the estimator: **which raw observations and columns need to reach the canonical design at all?** It does not decide coefficients, instruments, FE identification, separation, or covariance formulas. The frontend compiles a specification into raw-column requirements; `data.source` pushes that projection into CSV/Stata/Parquet readers where possible; `data.planner` chooses bounded batches; categorical encoding and sample-state tracking preserve group identity and observation provenance. The resulting projected frame then enters the existing design/HDFE/model pipeline.

The integrated path is intentionally conservative: ordinary in-memory DataFrames retain the established behavior, while OLS-HDFE and linear-IV estimators and repeated-spec Sessions may additionally consume file/DataSource inputs. For a repeated regression table, Sessions first compile the union of raw columns and economic roles, build one immutable `EncodedEconometricDataset`, and then reuse the existing exact FE/sample/within-column caches. Identifier-only FE/cluster columns can be encoded once; a column that appears as an explicit factor/regressor in any specification keeps its original value semantics. Missing-value semantics remain unchanged; `EstimationSampleState` is an audit/infrastructure object, not an implicit drop-missing policy. Fully out-of-core HDFE/PPML remains outside the current checkpoint.

## Why linear-IV code moved out of `iv/`

The old `iv/estimators.py` mixed two different concepts: generic instrument mechanics and the linear-IV outcome model. That would create the wrong dependency once IV-PPML is added. In 0.1.0a2:

- generic `iv` knows how columns are assigned instrument roles, how additive IV moments are formed, and how to solve a weighted 2SLS inner problem;
- `models.linear_iv` owns LIML, Fuller/k-class, two-step linear GMM and linear weak-identification diagnostics;
- `models.ppml` remains pure PPML;
- `models.ppml_iv` composes PPML state/link primitives, generic `iv`, weighted HDFE and compute kernels without importing `models.linear_iv`.

The implemented path is:

```text
models/ppml_iv
      │
      ├──► models/ppml     # shared Poisson numerical primitives
      ├──► iv              # additive moments and weighted 2SLS
      ├──► hdfe            # weighted FE concentration
      └──► compute         # VCE/runtime/kernels
```

SPJ bias correction and bootstrap remain model-specific in `models.ppml_iv.bias` and `models.ppml_iv.bootstrap`; generic `iv` therefore stays usable by other nonlinear outcome models without inheriting IV-PPML incidental-parameter assumptions.

## Public estimator names

```python
from econhdfe import olshdfe, ppmlhdfe, ivhdfe, ivppmlhdfe
```

`ivhdfe` remains the public linear-IV estimator for compatibility. The nonlinear IV-PPML estimator is exposed explicitly as `ivppmlhdfe`; `ivhdfe` remains the linear-IV estimator. Legacy `reghdfe`, `ivreghdfe`, and `pyreghdfe` remain compatibility shims.

## Error boundary rule

Numerical kernels may keep concise internal exceptions. Public estimator/resampling boundaries expose `EconHDFEError` subclasses with machine-readable reason codes. The boundary deliberately does not catch arbitrary `Exception`: programmer failures such as `AttributeError`, `NameError`, and assertions remain visible instead of being mislabeled as user/data failures.

## Resampling and inference rule

The top-level `resampling` layer owns only repeated-draw mechanics (seeds, cluster-row groups, wild-weight generation, parallel execution, failure aggregation). Standard sandwich covariance—including multi-way CRV1—remains in `compute/vcov.py`. Advanced post-estimation cluster inference lives under `inference/cluster/`: diagnostics and one-way OLS WCR11/WCU11 are inference procedures rather than estimator definitions. Model-specific bootstrap statistics such as IV-PPML SPJ remain with their model.

CRV3/cluster jackknife is intentionally not public in 0.4.6. The current prototype requires delete-cluster HDFE refits and extra raw estimation-state arrays; it should enter the inference layer only after a large-data execution/memory design is validated.

## Result/reporting boundary

Estimator results expose a common `PublicationResult` protocol: `coef_table()`, `model_stats()`, and `publication_output()`. The default publication output contains econometric quantities suitable for regression tables and appendices, while internal solver diagnostics/profile traces remain opt-in. Linear-IV publication output retains reportable first-stage and identification/over-identification tests without returning large observation-level fitted first-stage arrays.

## Configuration and execution lifecycle

Stable public controls are grouped into `HDFEConfig`, `InferenceConfig`, and `ExecutionConfig`. Only strategy-level choices are public; implementation constants remain private. `ExecutionContext` owns reusable caches and optional profiling state for reusable estimators. `FEPlan.fingerprint` is content based, and DataFrame PPML/IV-PPML models can validate source FE signatures before reusing compiled topology.


## Exact multiway HDFE rank (0.4.0)

`hdfe/rank.py` owns certified characteristic-zero rank for intercept-only categorical FE designs. It is estimator-independent and is consumed by `hdfe/dof.py`. Numerical canonicalization is solver-only; requested/inference topology is retained separately for DoF and cluster nesting. Optional SymPy/FLINT backends accelerate hard residual cores but are not mandatory dependencies.

## Multiway numerical solver core (0.4.4)

`hdfe/numerical_core.py` is a numerical-execution reduction, not an inference-rank engine. For eligible NumPy pure-intercept systems with three or more FE partitions, it recursively peels degree-one FE levels: orthogonality to a singleton dummy forces that row residual to zero, so only the surviving row core must be solved. This is distinct from `hdfe/rank.py`, which computes structural absorbed DoF and may deduplicate FE tuples; numerical core reduction never deduplicates observations because every row carries its own RHS and weight.

`HDFEAbsorber(acceleration="auto")` performs two real symmetric-MAP sweeps on the actual RHS block. Fast contraction continues plain MAP; hard systems continue CG from the partially residualized state. Direct `HDFEAbsorber` keeps `acceleration="cg"` as its public default. Optimized `FEPlan` selects `auto` for generic MAP systems. The weighted many-RHS projection kernel can aggregate a bounded column block in one grouped pass.

The core topology is reusable across strictly positive weight updates. A zero weight changes the effective numerical graph and disables core reduction. The integrated path reuses the caller/workspace output buffer after materializing only the reduced core input, avoiding an additional full `N x RHS` allocation. Solver canonicalization/core reduction never changes the requested FE representation retained for DoF, nesting, or reporting.

## Repeated-spec session boundary (0.4.1)

`OLSHDFESession` and `IVHDFESession` are execution/lifecycle objects, not new estimators. They cache DataFrame numeric columns, FE component encodings, FE-combination-specific absorbers, singleton samples, DoF metadata, and within-transformed columns. Each distinct requested FE combination owns a distinct cache entry; transformed data are never reused across different FE/sample spaces.

`fit_many()` groups specifications by FE combination and residualizes the union of newly requested columns in one multi-RHS pass. This is exact for fixed linear HDFE geometry. Linear-IV diagnostics remain specification-specific and are recomputed. PPML/IV-PPML are intentionally excluded because their IRLS weights change with the outcome/specification, so fixed linear within/cross-product caching is not generally valid.


## Release-governance and documentation boundary (0.4.2–0.4.4.1)

Release engineering is treated as a separate contract from estimator implementation. `scripts/version.py` owns version consistency and bump discipline; `scripts/release_maintenance.py` owns the auditable cross-cutting impact review. A bump resets all review areas, while the release builder calls the strict gate before building artifacts. This prevents a new model/session/backend change from being released without an explicit review of frontend exports, structured errors, agent skill, linked modules, result/config/cache contracts, tests, performance and packaging.

0.4.3 adds a second, automated guard: `scripts/compatibility.py` compares the live public contract against the immutable previous-release snapshot. Exports, callable signatures, result/config dataclass schemas, exported dataclasses and structured-error codes/stages cannot drift silently. Intentional differences require an exact change-id approval and rationale; stale approvals also fail the gate.

The Agent Skill is a release-side consumer interface rather than estimator code. `skills/econhdfe/` is the single canonical skill root; its thin `SKILL.md` routes to one-level `references/`, deterministic `scripts/`, and `agents/openai.yaml`. Release assembly produces a standalone skill zip and verifies it is identical to this canonical tree.

The maintenance manifest contains review evidence only; it must not encode estimator logic or become a second configuration system.

0.4.4.1 makes documentation layout part of the release contract. Runtime code remains under `econhdfe/`; current technical/developer documentation lives under `docs/`; machine-readable benchmark evidence lives under `benchmarks/`. The exact multiway-DoF manuscript is shipped in source/release artifacts for auditability but is explicitly excluded from wheels.


## Automatic thread calibration and feedback

`threads="auto"` is a planner decision, not an estimator decision. For sufficiently large HDFE workloads, the planner can run a short synthetic memory-bound Numba calibration on the actual runtime, select the smallest thread count within 5% of the best measured median time, and persist that anonymous calibration by execution fingerprint. Small HDFE jobs stay single-threaded so calibration/parallel overhead cannot dominate the fit. Explicit thread counts bypass calibration. Calibration never consumes user observations and cannot change samples, regressors, FE, instruments, weights, inference, or solver mathematics.

A separate planner developer report returns only aggregate hardware capabilities, calibration timings, selected plan metadata, anonymous dimensional counts and optional aggregate fit timing. It never sends data automatically and intentionally excludes raw/synthetic observations, variable names, file paths, commands, hostnames, logs and tracebacks.
