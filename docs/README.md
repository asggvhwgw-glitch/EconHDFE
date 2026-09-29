# 0.6.1 correctness review

- [Mathematical proofs, applications, counterexamples and scope](technical/mathematical-review-0.6.1.md)
- [Local validation and outstanding external checks](development/test-status.md)
- [Migration from 0.6.0](release/migration.md)

# econhdfe documentation

The repository keeps runtime code, technical evidence, and release governance separate.

## GitHub / technical entry points

- [Technical overview](technical/overview.md) — end-to-end econometric and computational contract.
- [Performance architecture](technical/performance-architecture.md) — large-data bottlenecks, modular performance strategy and benchmark interpretation.
- [Identified categorical fixed effects](technical/identified-fixed-effects.md) — FE recovery, identification, normalization, singleton/separation diagnostics and block salvage.
- [Technical documentation index](technical/README.md) — formal manuscripts, numerical notes and innovation policy.
- [Machine-checked mathematical verification](technical/formal-verification.md) — Lean scope, manuscript mapping, audit boundary and evidence.
- [Economics-first architecture](development/architecture.md) — module ownership and dependency boundaries.
- [Economic problem map](development/economic-module-map.md) — every runtime module mapped to the empirical/econometric problem it serves.
- [Generated architecture map](development/architecture-map/architecture.md) — AST-backed code dependency view.
- [Release validation status](development/test-status.md) — current regression, parity and release-gate evidence.

## Documentation policy

The repository distinguishes three kinds of claims:

1. **econometric behavior** — what estimator/specification/inference is implemented;
2. **engineering behavior** — how data, designs, FE projection, caching and execution are accelerated without changing the requested model;
3. **technical originality** — only items registered in `technical/innovation-registry.json`, each backed by a formal manuscript and prior-art boundary.

Repository-local benchmark timings are development evidence unless an external comparison explicitly states its package versions, hardware, workload and parity checks.

## User-facing entry points

- `README.md`: package overview and public examples.
- `skills/econhdfe/`: portable agent skill with installation, configuration, empirical, validation, and developer references.
- `docs/release/migration.md`: migration notes between public releases.

## Technical documentation

- `docs/technical/README.md`: package-wide technical-innovation policy and source-of-truth registry.
- `docs/technical/innovation-audit.md`: complete novelty audit distinguishing technical innovation from established methods and engineering.
- `docs/technical/overview.md`: end-to-end technical contract from estimator semantics through execution planning to validation boundaries.
- `docs/technical/performance-architecture.md`: where HDFE runtime is spent and why removing repeated work precedes kernel optimization.
- `docs/technical/identified-fixed-effects.md`: fixed-effect recovery, realized-sample identification and normalization.
- `docs/technical/hdfe/`: HDFE mathematics, formal innovation manuscripts, numerical solver design, implementation correspondence, and performance evidence.
- `docs/technical/structural-design/`: formal manuscript for exact partition-refinement HDFE structural design reduction.
- `docs/development/architecture.md`: economics-first package architecture, dependency boundaries and module ownership.
- `docs/development/execution-planner.md`: unified certificate → cost/resource → execution policy and its current calibration boundary.
- `docs/development/economic-module-map.md`: every runtime module mapped to the economic/econometric problem it serves and its computational responsibility.
- `docs/development/architecture-map/`: generated AST-backed Markdown/JSON plus an interactive HTML architecture map.
- `docs/development/benchmarks.md`: benchmark index and interpretation rules.
- `docs/development/test-status.md`: local versus external validation status.
- `docs/development/ERROR_REPORT_TEMPLATE.md`: canonical privacy-minimized, parameter-only crash/mismatch/bug report returned by users or agents to developers; it intentionally excludes data reconstruction and raw artifacts.
- `docs/development/external-validation.md`: tests that require licensed/reference hardware or software.
- `docs/development/upstream-references.md`: upstream implementations and literature used for compatibility work.

## Release engineering

- `docs/release/versioning.md`: version policy and mandatory closeout workflow.
- `docs/release/checklist.md`: human-readable release checklist.
- `docs/release/maintenance.json`: machine-readable release-impact review consumed by the release gate.
- `docs/release/manifest.md`: current release manifest.
- `docs/release/performance-release.md`: performance-release notes.

## Historical material

`docs/legacy/` is retained for provenance only. It is not the canonical source for current behavior.

- `technical/cluster-inference.md` — CRV1/resampling/advanced-cluster-inference boundary and v0.4.6 scope.

- [`development/statistical-parity.md`](development/statistical-parity.md): parity audit for secondary/main reported model statistics and reference-package conventions.

- [Planner/performance developer report template](development/PLANNER_REPORT_TEMPLATE.md) — privacy-minimized feedback for automatic threading and execution-planner anomalies.

- `econhdfe.support_reports` / `econhdfe-report`: installed privacy-safe support-report helpers; error extraction is scalar/counter allowlist only, while real-data benchmark authorization remains separate.

## Current maintenance entry points

- [Unified TODO / roadmap](../TODO.md)
- [Execution acceptance, distinct from review](release/acceptance.md)
- [Functional versus benchmark test environment](development/test-environment.md)
- [0.6.3 measured performance](release/performance-0.6.3.md)

Release publication mechanics: [release/publishing.md](release/publishing.md).
