# Planner final calibration checkpoint

This checkpoint closes the architecture-level planner work after the post-architecture validation. It adds only two bounded engineering capabilities: real-runtime automatic HDFE thread calibration and a privacy-minimized developer feedback/report mechanism.

## Automatic threads

`threads="auto"` no longer means “use every effective logical CPU” for sufficiently large HDFE work. For HDFE workloads with at least 100k observations, econhdfe runs a deterministic package-generated memory-bound Numba probe the first time a new anonymous runtime fingerprint is seen. It measures candidate thread counts, selects the smallest candidate within 5% of the best median calibration time, and stores only the anonymous fingerprint, candidate counts and aggregate timing curve in the local planner cache.

Small HDFE jobs stay single-threaded to avoid paying the one-time calibration and parallel-dispatch cost. Explicit positive integer thread settings bypass calibration completely.

Calibration affects execution resources only. It cannot change the sample, design, FE definition, instruments, weights, covariance estimator, solver mathematics or convergence criteria.

## Current real-runtime evidence

On the current 4-thread container:

- cold calibration: about 1.02 s once;
- cache reload in a new process/in-memory state: about 0.45 ms;
- selected automatic HDFE threads: 4;
- alternating five-round 300k x 8, four-FE validation:
  - auto median: about 0.2846 s;
  - best explicit median: about 0.2863 s;
  - auto / best explicit: about 0.994x;
  - engineering near-best-10% check: PASS.

The calibration is intentionally a near-saturation policy, not a claim that one microbenchmark can identify the exact optimum for every FE topology.

## Developer feedback

`econhdfe.planner.write_planner_developer_report(...)` produces Markdown or JSON with aggregate runtime resources, the anonymous calibration curve/ID, optional execution-plan information, and optional anonymous result counts/timings. `skills/econhdfe/scripts/planner_report.py` can generate an environment/calibration-only report without loading model data.

Nothing is sent automatically. The report excludes raw/sampled/synthetic observations, variable identifiers, data paths, exact commands/scripts, hostnames, raw logs and full tracebacks. Real-data benchmarking remains a separate explicit-consent workflow.

Canonical template: `docs/development/PLANNER_REPORT_TEMPLATE.md`.

## Validation

- tests: 466 / 466 PASS;
- public contract: 0 changes;
- technical innovation registry: 3 unchanged;
- architecture map: current; economics-facing module map covers all 108 runtime Python modules;
- seven-model OLS/IV/PPML/IV-PPML statistical parity: exact for coefficients, SE, N and absorbed DoF in the validation matrix;
- package version remains 0.4.10.1 because this is still a development checkpoint rather than a release.

No further planner architecture expansion is recommended at this stage. Future work should be restricted to real-machine calibration evidence, bug fixes, and narrowly justified heuristic corrections.
