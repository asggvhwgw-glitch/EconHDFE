# EconHDFE documentation

The repository keeps user guidance, current technical specifications, development records, validation evidence, and historical material separate.

## Start here

- [Project README](../README.md) / [中文 README](../README.zh-CN.md)
- [Technical overview](technical/overview.md) — end-to-end econometric and computational contract.
- [Performance architecture](technical/performance-architecture.md) — large-data bottlenecks, representation strategy, and benchmark interpretation.
- [Testing and validation status](development/test-status.md) — current repository-local execution evidence and external-validation boundary.
- [Agent Skill](../skills/econhdfe/SKILL.md) — operational guide for agent-driven installation, empirical work, validation, and development.

## Empirical and technical users

- [Identified categorical fixed effects](technical/identified-fixed-effects.md)
- [Cluster inference](technical/cluster-inference.md)
- [HDFE technical documentation](technical/hdfe/)
- [Structural design reduction](technical/structural-design/)
- [Partitioned WLS](technical/partitioned-wls.md)

## Mathematical theory and formal verification

- [Technical documentation index](technical/README.md)
- [Machine-checked mathematical verification](technical/formal-verification.md)
- [Lean proof library](../formal/README.md)
- [Theorem-by-theorem mathematical coverage](../formal/mathematical-coverage.md)

The Lean work verifies explicitly mapped mathematical statements under their stated assumptions. It is not a blanket formal verification of Python/Numba code, floating-point convergence, benchmark claims, external Stata parity, or historical originality.

## Architecture and development

Current design documents:

- [Economics-first architecture](development/architecture.md)
- [Economic module map](development/economic-module-map.md)
- [Generated architecture map](development/architecture-map/architecture.md)
- [Data layer](development/data-layer.md)
- [Execution planner](development/execution-planner.md)
- [Testing](development/testing.md)
- [Test environment](development/test-environment.md)
- [Benchmarks](development/benchmarks.md)
- [External validation](development/external-validation.md)
- [Statistical parity](development/statistical-parity.md)
- [Upstream references](development/upstream-references.md)

Historical development checkpoints are archived under [development/history/](development/history/). They document how the current architecture evolved and are not current specifications.

## Release engineering

- [Versioning](release/versioning.md)
- [Release checklist](release/checklist.md)
- [Publishing](release/publishing.md)
- [Release manifest](release/manifest.md)
- [Migration](release/migration.md)
- [Execution acceptance](release/acceptance.md)
- [Release evidence](release/evidence/)

Version-specific closeouts and maintenance notes are archived under [release/history/](release/history/). Archived records are provenance, not current release status.

## Technical-claim governance

The repository distinguishes:

1. **econometric behavior** — the estimator/specification/inference implemented;
2. **engineering behavior** — how data, design, FE projection, caching, and execution are accelerated without changing the requested model;
3. **machine-checked mathematics** — explicitly mapped theorems and assumptions in the Lean library;
4. **historical originality** — a separate prior-art question that is not established by formal verification.

The machine-readable technical-claim source of truth is [technical/innovation-registry.json](technical/innovation-registry.json). Historical technical reviews are archived under [technical/history/](technical/history/).

## Support and reporting

- [Privacy-minimized error report template](development/ERROR_REPORT_TEMPLATE.md)
- [Planner/performance developer report template](development/PLANNER_REPORT_TEMPLATE.md)
- `econhdfe.support_reports` / `econhdfe-report` — installed privacy-safe reporting helpers.

## Legacy material

[docs/legacy/](legacy/) is retained for provenance only. It is not the canonical source for current behavior.

Current development priorities are tracked in [TODO.md](../TODO.md).
