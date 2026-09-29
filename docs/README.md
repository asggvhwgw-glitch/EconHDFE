# EconHDFE Documentation

The documentation is organized by **current behavior**, **mathematical/technical foundations**, **development/validation**, and **release history**. Historical checkpoints are retained for provenance but are not the source of truth for current behavior.

## Start here

- [Project README](../README.md) / [中文首页](../README.zh-CN.md)
- [Technical overview](technical/overview.md) — end-to-end econometric and computational contract.
- [Testing and validation status](development/test-status.md) — current local and CI validation boundaries.
- [Unified roadmap](../TODO.md) — current priorities and open work.

## Empirical users

- [Identified categorical fixed effects](technical/identified-fixed-effects.md) — FE recovery, identification and normalization.
- [Testing guide](development/testing.md) — how behavior and numerical parity are checked.
- [Migration notes](release/migration.md) — changes between public releases.
- `skills/econhdfe/` — installation, empirical use, diagnostics, advanced settings and developer guidance for agents.

## Mathematical theory and formal verification

- [Technical documentation index](technical/README.md) — theorem-backed project technical documents and technical-policy boundaries.
- [Machine-checked mathematical verification](technical/formal-verification.md) — Lean scope, proof assumptions and audit boundary.
- [Lean proof library](../formal/README.md) — reproducible formal proof environment.
- [Mathematical coverage table](../formal/mathematical-coverage.md) — theorem-by-theorem mapping.

## Architecture and performance

- [Economics-first architecture](development/architecture.md)
- [Economic problem map](development/economic-module-map.md)
- [Generated architecture map](development/architecture-map/architecture.md)
- [Execution planner](development/execution-planner.md)
- [Data layer](development/data-layer.md)
- [Performance architecture](technical/performance-architecture.md)
- [Benchmark documentation](development/benchmarks.md)

## Inference and specialized technical notes

- [Cluster inference](technical/cluster-inference.md)
- [Heterogeneous specification optimization](technical/heterogeneous-specification-optimization.md)
- [Partitioned WLS](technical/partitioned-wls.md)
- [HDFE technical documents](technical/hdfe/)

## Validation and external references

- [Test environment](development/test-environment.md)
- [External validation](development/external-validation.md)
- [Statistical parity](development/statistical-parity.md)
- [Upstream references](development/upstream-references.md)
- [Validation harness](../validation/README.md)

Repository-local tests and benchmarks are not a substitute for licensed-Stata or upstream external-corpus certification.

## Release engineering

- [Versioning](release/versioning.md)
- [Release checklist](release/checklist.md)
- [Execution acceptance](release/acceptance.md)
- [Release manifest](release/manifest.md)
- [Publishing](release/publishing.md)
- [Performance release notes](release/performance-release.md)

## Historical material

Historical development checkpoints are under [development/history/](development/history/), historical technical reviews under [technical/history/](technical/history/), and release closeout records under [release/history/](release/history/). Older pre-current architecture material remains under [legacy/](legacy/).

Historical files are kept for provenance. They do not override current API, architecture, mathematical coverage, or release documentation.
