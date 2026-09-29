# EconHDFE Documentation

The repository keeps runtime code, current technical documentation, historical development records, validation evidence, and release governance separate.

## Start here

- [Project README](../README.md) / [中文 README](../README.zh-CN.md)
- [Technical overview](technical/overview.md) - end-to-end econometric and computational contract.
- [Performance architecture](technical/performance-architecture.md) - large-data bottlenecks, reuse, and benchmark interpretation.
- [Economics-first architecture](development/architecture.md) - module ownership and dependency boundaries.
- [Generated architecture map](development/architecture-map/architecture.md) - AST-backed code dependency view.
- [Current roadmap](../TODO.md)

## Empirical users

- [Testing and validation status](development/test-status.md)
- [Identified categorical fixed effects](technical/identified-fixed-effects.md)
- [Statistical parity notes](development/statistical-parity.md)
- [Migration guidance](release/migration.md)
- [Agent Skill](../skills/econhdfe/SKILL.md)

## Technical architecture

- [Technical documentation index](technical/README.md)
- [Execution planner](development/execution-planner.md)
- [Data layer](development/data-layer.md)
- [Economic module map](development/economic-module-map.md)
- [Benchmark documentation](development/benchmarks.md)
- [External validation](development/external-validation.md)
- [Upstream references](development/upstream-references.md)

## Mathematical theory and formal verification

- [Machine-checked mathematical verification](technical/formal-verification.md)
- [Lean proof library](../formal/README.md)
- [Mathematical coverage table](../formal/mathematical-coverage.md)
- [Exact multiway FE rank / DoF](technical/hdfe/exact-multiway-dof/)
- [Exact residual-core reduction](technical/hdfe/numerical-residual-core/)
- [Exact structural-design reduction](technical/structural-design/)

## Validation, benchmarks, and release engineering

- [Testing](development/testing.md)
- [Test environments](development/test-environment.md)
- [Release acceptance](release/acceptance.md)
- [Release checklist](release/checklist.md)
- [Release manifest](release/manifest.md)
- [Publishing process](release/publishing.md)
- [Versioning](release/versioning.md)

Repository-local benchmark timings are development evidence unless an external comparison explicitly states package versions, hardware, workload, and parity checks.

## Documentation policy

The repository distinguishes:

1. **econometric behavior** - what estimator, specification, and inference are implemented;
2. **engineering behavior** - how data, designs, FE projection, caching, and execution are accelerated without changing the requested model;
3. **mathematical verification** - machine-checked statements under explicit assumptions;
4. **historical originality** - a separate prior-art question that is not established by mathematical correctness alone.

## Developer and support entry points

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [Error report template](development/ERROR_REPORT_TEMPLATE.md)
- [Planner report template](development/PLANNER_REPORT_TEMPLATE.md)
- [Developer guide](../skills/econhdfe/references/developer-guide.md)

## Historical reviews and archived material

Historical records are preserved for provenance but are not current sources of truth:

- [Historical technical reviews](technical/history/)
- [Development history](development/history/)
- [Release history](release/history/)
- [Legacy documentation](legacy/)

Current behavior should be read from the active documentation above rather than inferred from an old version-bound checkpoint.
