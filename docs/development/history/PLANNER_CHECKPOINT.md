# Planner checkpoint 1

This development checkpoint introduces a shared execution-planner layer without changing econometric semantics or public estimator signatures.

## What changed

- one canonical runtime CPU/memory resource contract;
- exactness/eligibility certificates separated from performance cost;
- one memory envelope used by ingestion, HDFE workspaces and partitioned-WLS memory guards;
- one parallel-budget contract used by HDFE automatic thread selection;
- dense/block representation candidates validated through the same exactness contract;
- composite `ExecutionPlan` for explainable memory + representation + parallel policy.

## What deliberately did not change

The production dense/block heuristic still uses the established conservative policy: block execution requires an exact certificate, meaningful storage savings, and either repeated operator passes or a dense representation that exceeds its configured memory budget. The new generic byte-traffic score is not yet allowed to override that rule.

The reason is empirical. Dense BLAS throughput depends strongly on operator mix and thread count, so physical bytes alone overpredict block speedups on some one-shot operations. Hardware calibration is therefore a later planner phase, not folded into this architectural checkpoint.

## Validation

- 461 / 461 tests passed;
- public contract: 0 changes;
- technical innovation registry: 3, unchanged;
- architecture map: current;
- planner overhead: ~12 microseconds per composite plan in the development microbenchmark;
- structured-model coefficients remain at machine-precision parity with pooled dense paths in the planner checkpoint benchmark.

## Post-architecture validation

A dedicated post-architecture validation pass now compares this checkpoint to the immediately preceding Data Layer checkpoint. Seven representative OLS/IV/PPML/IV-PPML specifications are numerically identical including full covariance matrices and applicable metadata. Generic four-FE and complex event-study stress tests likewise retain exact result parity, and repeated timing runs show no stable performance regression. See `POST_ARCHITECTURE_VALIDATION.md` and `benchmarks/planner/post_architecture/validation_summary.json`.

## Next planner phase

The next useful work is not more planner abstraction. It is calibration of two decisions with real machine evidence:

1. automatic HDFE thread count under memory-bandwidth saturation;
2. dense vs block representation under actual operator mixes and expected reuse.

Only after those calibrations are stable should the planner alter current production defaults.


## Final calibration addendum

The post-architecture planner now includes real-runtime automatic HDFE thread calibration for sufficiently large workloads and a privacy-minimized developer feedback/report mechanism. The calibration is synthetic, cached by anonymous runtime fingerprint, and bounded to execution policy only. The canonical multi-round 300k x 8 / 4-FE validation is stored in `benchmarks/planner/thread_calibration.json`; the auto choice must remain within the declared near-best engineering tolerance rather than being hard-coded from one benchmark.
