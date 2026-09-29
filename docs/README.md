# EconHDFE documentation

This index separates current user/developer documentation from historical reviews, checkpoints, and release evidence. Runtime code, mathematical verification, validation evidence, and release governance are maintained as distinct layers.

## Start here

- [Project README](../README.md) / [中文 README](../README.zh-CN.md)
- [Technical overview](technical/overview.md) — estimator semantics, HDFE mathematics, data flow and execution boundaries.
- [Technical documentation index](technical/README.md) — current mathematical, numerical and architecture documents.
- [Machine-checked mathematical verification](technical/formal-verification.md) — Lean scope, theorem mapping, audit boundary and evidence.
- [Testing and validation](development/testing.md) — test organization and validation philosophy.
- [Current validation status](development/test-status.md) — repository-local versus external validation boundaries.
- [Unified roadmap](../TODO.md)

## Empirical users

- [Identified categorical fixed effects](technical/identified-fixed-effects.md) — FE recovery, identification and normalization.
- [Cluster inference](technical/cluster-inference.md) — cluster-robust and advanced cluster-inference boundaries.
- [Statistical parity](development/statistical-parity.md) — reported statistics and reference-package conventions.
- [Migration guide](release/migration.md) — public-release migration notes.
- `skills/econhdfe/` — installation, empirical, validation and developer guidance for agent-driven workflows.

## Technical architecture

- [Technical overview](technical/overview.md)
- [Performance architecture](technical/performance-architecture.md)
- [Economics-first package architecture](development/architecture.md)
- [Economic problem map](development/economic-module-map.md)
- [Generated architecture map](development/architecture-map/architecture.md)
- [Econometric data layer](development/data-layer.md)
- [Execution planner](development/execution-planner.md)
- [Partitioned WLS](technical/partitioned-wls.md)
- [Heterogeneous-specification optimization](technical/heterogeneous-specification-optimization.md)

## Mathematical theory

The three registered theorem-backed technical documents are:

- [Exact multiway categorical FE rank / absorbed DoF](technical/hdfe/exact-multiway-dof/README.md)
- [Exact multiway residual-core reduction](technical/hdfe/numerical-residual-core/README.md)
- [Exact partition-refinement structural design reduction](technical/structural-design/README.md)

Their mathematical theorem coverage is machine-checked separately under `formal/`; see the [formal verification status](technical/formal-verification.md). Mathematical correctness, production implementation correctness, and historical originality remain separate claims.

## Validation and benchmarks

- [Benchmark index and interpretation rules](development/benchmarks.md)
- [Test environment](development/test-environment.md)
- [External validation](development/external-validation.md)
- [Upstream references](development/upstream-references.md)
- [Release acceptance](release/acceptance.md)
- [0.6.3 measured performance](release/performance-0.6.3.md)

Repository-local timing evidence is not a general performance claim unless the workload, package versions, hardware and parity checks are stated.

## Developer documentation

- [Testing](development/testing.md)
- [Architecture](development/architecture.md)
- [Execution planner](development/execution-planner.md)
- [Error report template](development/ERROR_REPORT_TEMPLATE.md)
- [Planner report template](development/PLANNER_REPORT_TEMPLATE.md)
- [Developer guide](../skills/econhdfe/references/developer-guide.md)
- [Contributing](../CONTRIBUTING.md)

## Release engineering

- [Versioning policy](release/versioning.md)
- [Release checklist](release/checklist.md)
- [Publishing workflow](release/publishing.md)
- [Release manifest](release/manifest.md)
- [Maintenance contract](release/maintenance.json)
- [Release evidence](release/evidence/)

The latest published release and the current unreleased development line are distinct. A passing development CI run is not by itself a detached release.

## Claim policy

The repository distinguishes:

1. **econometric behavior** — estimator/specification/inference semantics;
2. **engineering behavior** — data, design, projection, caching and execution choices;
3. **mathematical correctness** — explicitly stated theorem results and assumptions;
4. **technical originality** — independent prior-art status recorded separately from correctness.

The source of truth for registered theorem-backed contributions is [technical/innovation-registry.json](technical/innovation-registry.json).

## Historical material

Historical documents are retained for provenance but are not the current source of truth.

- `development/history/` — development checkpoints, scaffolds and migration records.
- `release/history/` — release closeouts and forward-port audits.
- `technical/history/` — version-specific mathematical/resource reviews.
- `legacy/` — older superseded architecture and package documentation.

Notable historical reviews:

- [0.6.1 mathematical correctness review](technical/history/mathematical-review-0.6.1.md)
- [0.6.2 nonlinear/resource review](technical/history/nonlinear-and-resource-review-0.6.2.md)
