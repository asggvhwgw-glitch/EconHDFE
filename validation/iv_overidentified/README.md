# Overidentified IV-PPML development validation — 2026-10-28

Local follow-on to PR #36 at `76523437e7e3f2ab5911f17d802c74a9b7041420`.
The code is NOT integrated with main `85e5ff90` / PR #35. `0.10.1.dev1`
is a branch-local development checkpoint; it is not a new published release line.
The default remains legacy/standard inference/no overidentification test.

## Executed checks on the final runtime

- `python -m pytest tests -q`: **589 passed, 1 skipped**; optional native within absent.
- Parent + follow-on numerical suites: **64 passed**, of which 28 are new.
- Installed wheel, outside the source tree, host dependencies: **116 passed**,
  including the parent and new numerical tests. This is not a clean dependency
  installation or isolated build.
- Offline non-isolated wheel build passed; source/runtime byte parity is checked
  in `artifact-runtime-parity.json` in the final bundle.
- Version/contract/maintenance gate and generated architecture check passed.
- Candidate execution gate valid; release readiness is deliberately false.

Reproduce with `NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
The early two-thread host-capability setting was incompatible with unrelated
existing tests which deliberately request four threads; final runs use four.

## Final benchmark protocol

`formal/ledger.json` records 14 serial fresh-process benchmark runs, with no
other test or simulation workload running concurrently. DGP generation and JIT
warmup are excluded; estimator, projection, outer iterations and inference are
included. Three repeats per 1M configuration, one repeat per 5M configuration.
Four threads; same three categorical FE specifications; K=2 slopes and L=4
instruments including the exogenous regressor. Requested workspace budget is
2048 MiB: that is not a total-process RSS cap. The EE configurations also run
the optional overidentification test; standard configurations do not.

| N | Outer | Inference | Runs | Median seconds | Median peak RSS MiB | Outer iterations |
|---:|---|---|---:|---:|---:|---:|
| 1,000,000 | moment | standard | 3 | 5.125 | 749.97 | 7 |
| 1,000,000 | moment | estimating equation + overid | 3 | 6.034 | 757.70 | 7 |
| 1,000,000 | profile | standard | 3 | 6.738 | 687.77 | 3 |
| 1,000,000 | profile | estimating equation + overid | 3 | 7.236 | 687.72 | 3 |
| 5,000,000 | moment | standard | 1 | 49.168 | 2540.88 | 7 |
| 5,000,000 | profile | estimating equation + overid | 1 | 48.983 | 2385.07 | 3 |

Fewer outer iterations are not automatically lower total cost. The 5M timings
are single exploratory observations, not a statistically established speedup.
Coefficient differences between modern modes are below 6.74e-10 in these
benchmarks. At 1M the EE covariance difference across solvers is below 2.0e-15.
The 5M covariance parity is not tested across equal inference configurations.

## Fixed-dimensional Monte Carlo

`simulation.json` and `simulation_12k.json` were rerun on final source: 300
correctly specified and 300 misspecified replications at each of N=3000 and
N=12000; all 1200 completed. Four fixed FE groups; true endogeneity, not simply
an exogenous regressor called endogenous. The independent Gaussian omitted
instrument term has a known projected-IV pseudo-true slope, so the misspecified
coverage target is not an unknown structural value guessed from the DGP.

For the endogenous slope, at N=3000 the misspecified standard/full-EE 95%
coverage is 91.7%/94.0%; at N=12000 it is 91.0%/93.0%. Correctly specified
coverage at N=12000 is 94.7% for both and the overid rejection rate is 4.7% at
nominal 5%. Not every parameter/specification attains nominal coverage. Three
hundred replications also imply Monte Carlo error; improvements are diagnostic,
not finite-sample coverage or arbitrary-HDFE certification. The overid test
rejects this deliberately strong misspecification in every replication; this
is not a general power guarantee.

## Retained failures and development evidence

- `full-tests.log`: early 577 passes / 8 failures, due to two-thread capability
  and stale generated contract/architecture/release metadata.
- `full-tests-before-map-regeneration.log`: 588 passes / 1 failure from stale
  architecture after the final code changes. Regenerated, then full rerun passes.
- `before_memory_fix/`: the first 5M profile fit was rejected by resource
  admission at the original 2048 MiB workspace budget. Removed duplicate raw
  stacked designs and released stale/rejected trial state before reallocation;
  did not raise the budget. Final 5M run passed.
- `bench_profile_1m_r1.*` is a preliminary concurrent run; not used in formal
  timing statistics. `pre_memory_fix_simulation*` are predecessor simulation
  logs; final summaries use the reruns without that prefix.

## Theory and implementation limitations

The precise derivative and full empirical-distribution influence formula are in
`docs/technical/ivppml/overidentified-estimation-inference.md`. Independent full-
dummy parameter and observation-weight finite differences, rather than a
second copy of the symbolic formula, are the main oracles.

Strong identification and regular fixed-dimensional asymptotics remain
necessary. No arbitrary growing-FE limit, short-panel bias removal, weak-IV-
robust confidence set, few-cluster guarantee, licensed Stata parity, or new
Windows/macOS/dependency-floor certification is claimed. Scalar overid p-values
are suppressed for singular/indefinite restriction covariance, not rescued by
PSD clipping with unchanged nominal degrees of freedom. The posted constant
remains a normalization, not a zero-uncertainty estimated intercept.
