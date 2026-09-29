# Benchmark layout

**Status:** current benchmark index  
**Policy:** benchmark evidence is workload-specific development evidence unless an external comparison records package versions, hardware, workload, parity checks and timing scope.

## Current domain directories

- `hdfe/` — HDFE numerical-solver and exact-rank evidence.
- `ppml/` — PPML benchmark programs and results.
- `repeated/` — repeated-specification/session benchmarks.
- `planner/` — execution-planner and calibration evidence.
- `effects/` — fixed-effect recovery benchmarks.
- `real_world/` — externally executed, user-authorized real-data benchmark records; preserve original runs immutably and append follow-ups.
- `release/` — benchmark material tied to release validation.

## Root-level historical scripts

Several inherited linear/HDFE benchmark scripts remain directly under `benchmarks/`. They are retained at their existing paths to avoid breaking provenance links in historical reports and release evidence.

**Do not add new benchmark families at the root.** New work should use the closest domain directory. A separate provenance migration should precede any future relocation of inherited root-level benchmark files.

## Interpretation rules

A benchmark must separate:

1. statistical/numerical parity;
2. timing scope (end-to-end, setup, projection, solve, inference, etc.);
3. cold versus warm execution;
4. environment and thread/resource settings;
5. workload-specific speed or memory results.

A single benchmark is not a package-wide performance guarantee.

## Real-machine return format

Use `real_world/BENCHMARK_REPORT_TEMPLATE.md` for user-authorized real-data benchmarks. Preserve prior dated records; never overwrite historical evidence. Reports should avoid raw/private data and keep parity evidence separate from speed claims.
