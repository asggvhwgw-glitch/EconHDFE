# Benchmark layout

Benchmark evidence is organized by stable domain where possible. New benchmark artifacts should go into a domain subdirectory rather than adding more files at the root.

## Current domain directories

- `hdfe/`: HDFE numerical-solver and exact-rank evidence.
- `effects/`: fixed-effect recovery and related post-estimation benchmarks.
- `planner/`: execution-planner and calibration evidence.
- `ppml/`: PPML benchmark programs and results.
- `repeated/`: repeated-specification/session benchmarks.
- `real_world/`: externally executed, user-authorized real-data benchmark records; preserve original runs immutably and append follow-ups.
- `release/`: release-facing benchmark summaries or reproducibility artifacts.
- `data/`: benchmark fixtures or generated inputs used by benchmark programs.

## Root-level historical scripts

Some inherited linear/HDFE benchmark scripts remain directly under `benchmarks/` to avoid breaking historical provenance and documentation links. They are retained intentionally; **new work should not add more root-level benchmark files unless a release or compatibility workflow requires it**.

This directory is evidence, not a claim that every stored timing is representative of all machines or workloads. Current performance claims must identify the exact benchmark input, package version, hardware/runtime, warm/cold scope, and numerical parity checks.

## Real-machine benchmark return format

Use `real_world/BENCHMARK_REPORT_TEMPLATE.md` for new user-authorized real-data benchmarks. Preserve prior dated records; never overwrite historical benchmark evidence. A benchmark report must separate parity from speed, state timing scope/cold-vs-warm behavior, record machine/package versions and resource settings, and avoid embedding raw/private data.
