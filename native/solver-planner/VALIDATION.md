# Three-route planner v0.2: validation and performance boundaries

Date: 2026-10-29. Status: **validated experimental planning adapter, not a production-default replacement**.

## Implemented scope

`econhdfe_planner.PlannedAbsorber` admits three execution routes: batched `within`, existing exact-tuple `compressed_within`, and persistent `fereg`. The new code is a Python planning layer over pinned Rust adapters, not a new implementation of either numerical core. The existing compressed-within bridge still loops over RHS; this release does not claim a newly fused compressed block Krylov implementation.

The full-stream tuple sketch visits every observation, instead of estimating duplicates from a tiny row sample. A conservative approximate distinct-ratio band gates whether to measure compression. Sketch hashes never define FE equality: actual compression remains exact. Size counts order trials but no longer veto fereg on three comparable large factors.

Auto measures admitted candidates on the full graph, actual RHS block, current weights and common requested tolerance. It includes a first projection and a warm projection, and selects by measured future-call cost. With workload='reweight', the warm score also includes an identical-weight refresh, which is a proxy for future changing-weight costs, not a guarantee about an actual IRLS path. The already computed winning projection is returned; only its solver is retained afterwards.

All calibration time, losing candidates, copies, numerical validation and structure scanning are included in the first-call timing. A low warm cost is NOT a first-call speedup. The route stays stable until an explicit retune or certification failover. Weight updates validate before public state mutation, and invalidate the numerical cache on internal refresh failure.

No existing EconHDFE estimator API, singleton/separation policy, rank/DoF, covariance estimator or default backend has been changed. Main has not been merged into or updated by this work. Categorical intercept FE, strictly positive finite weights and fixed samples only.

## Source and verification

- New branch: `feat/tri-route-planner-v02-20261029`.
- Runtime implementation commit: `815af529466774ea1a39256658492c6604c6c184`.
- Independently rebuilt and tested GitHub CI commit: `4ed1e0d5dc936c922d83caad49016ff9e1fe788c`.
- Successful CI run: [40819593931](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40819593931).
- CI artifact: `tri-planner-validation-40819593931`, ID `12874022917`, SHA-256 `6606d958cb236f04bdede3dc20db496f35e75ae3adffe90313f212e4dc9cd67b`.
- within pinned at `fc44282cea6124f5f18bb142908605a656d35a7c`.
- fereg-core pinned at `0fa59e3b539addbddaee8bd71acd031c497166a0`.
- Local performance uses the previously validated v0.1 native wheels. Two local-source comments differ from the connector-uploaded source; comment-stripped Git blob hashes and Python AST equivalence were verified. Runtime statements are identical.

The CI independently compiles both native dependencies, passes all 41 tests and runs two 100k-row smoke cases. The million/five-million-row performance numbers below are from the local test corpus, **not CI smoke timing**.

## Numerical validation

- Local unit tests: **41/41 passed**. Independently rebuilt GitHub tests: **41/41 passed**.
- **8 synthetic structures, 15 size/RHS/seed configurations, 38 method records and 114 returned projections**, all valid.
- Maximum relative weighted L2 difference from an independent analytic oracle or SciPy reference: **5.910528484088863e-11**, below the common `2e-8` benchmark threshold.
- **37 measured candidate trials all accepted**; each includes cold and warm measurements. No failed candidate is ranked as fast. Performance tests disable backend failover; failover is covered separately by fault injection.
- Requested original-row certificate tolerance remains `1e-10`. No acceptance gate was relaxed to improve speed.

Tests cover independent dense SVD least squares, weighted 1/2/8 RHS, updates, invalid-update rollback, internal update failure and rebuild, sparse labels, duplicate/nested FE, disconnected components, retained singletons, zero/constant/empty RHS, candidate disagreement, soft-budget accounting and stable routes. A common **test-only PPML outer loop on 600 rows** checks coefficients, fitted means and raw scores. This is not production PPML/IV-PPML integration or covariance parity.

## Independent oracles and synthetic scope

The primary unique-tuple fixtures superimpose random four-edge cycles `(a0,b0),(a0,b1),(a1,b0),(a1,b1)`. Assign weighted flow `W r=(+e,-e,-e,+e)` and hold a third factor constant on the cycle. Then every factor's weighted residual sum is zero. Constructing `y=D alpha+r` gives a known residual without repeating complete tuples. Controlled weight changes multiply all edges in a cycle by the same factor; this preserves the oracle but is not a real PPML weight path.

Paired and 16-fold repeated-tuple fixtures are retained. A denser repeated-cell holdout is included rather than silently discarding sparse compressed graphs.

The additional `iid_three` case uses independent randomly permuted balanced FE factors and unit weights, not cycles. Its reference is computed separately with SciPy LSMR (`atol=btol=1e-13`, maxiter 3000), with termination and normal-equation residual checks. Its reported difference is a **reference-solve difference**, not an analytic-error theorem. All data are synthetic; they do not establish representative administrative-data performance.

## Same-machine measurements

Seconds. Cold = constructor plus first call, including full calibration. Reuse = median of the next two calls, including numerical weight refresh when applicable. Paid-3x gain is existing-auto total time / new-planner total time over three actually measured calls: only values above one are a measured cumulative improvement. Estimated break-even assumes the two observed reuse costs persist; it is not a measured long-horizon guarantee.

Baseline: unchanged `NativeSolver(strategy='auto')`, not bare within or a full regression estimator. Every compared method/configuration runs in a separate process, with two solver threads and one BLAS thread. Source generation and independent oracle checks are excluded, but all planner/solver work is included.

| Structure | Rows / RHS | Existing cold | v0.2 cold | Existing reuse | v0.2 reuse | Reuse gain | Paid-3x gain | Estimated calls to break even |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unique one-big, seed 8411 | 1M / 8 | 0.965 | 5.953 | 0.714 | 0.452 | 1.58x | 0.35x | 21 |
| Unique two-big, changed weights | 1M / 8 | 5.151 | 18.855 | 3.394 | 1.962 | 1.73x | 0.52x | 11 |
| Unique three-big, seed 8411 | 1M / 8 | 20.390 | 43.310 | 20.321 | 3.155 | 6.44x | 1.23x | 3 |
| Paired three-big, seed 8411 | 1M / 8 | 23.145 | 124.222 | 30.637 | 1.962 | 15.62x | 0.66x | 5 |
| Sparse repeated cells | 1M / 8 | 2.257 | 23.629 | 1.888 | 0.485 | 3.89x | 0.25x | 17 |
| Weakly connected cycles | 1M / 8 | 5.399 | 12.914 | 4.901 | 0.542 | 9.05x | 1.09x | 3 |
| Unique two-big, changed weights | 1M / 2 | 0.839 | 3.513 | 0.742 | 0.427 | 1.74x | 0.53x | 10 |
| Paired three-big | 1M / 2 | 8.094 | 31.080 | 5.412 | 0.830 | 6.52x | 0.58x | 7 |
| Sparse repeated cells | 1M / 2 | 0.574 | 4.367 | 0.417 | 0.122 | 3.41x | 0.31x | 14 |
| Weakly connected cycles | 5M / 2 | 34.627 | 48.334 | 11.687 | 2.989 | 3.91x | 1.07x | 3 |
| Sparse repeated cells | 5M / 2 | 6.636 | 68.053 | 5.831 | 0.828 | 7.04x | 0.26x | 14 |
| Unique three-big, seed 17323 | 250k / 8 | 1.614 | 3.165 | 1.198 | 0.454 | 2.64x | 0.98x | 4 |
| Paired three-big, seed 17323 | 250k / 8 | 3.248 | 12.223 | 2.769 | 0.309 | 8.97x | 0.68x | 5 |
| IID mixed three-big, seed 71031 | 1M / 2 | 0.854 | 3.113 | 0.654 | 0.455 | 1.44x | 0.54x | 13 |
| Dense repeated cells, seed 71031 | 1M / 8 | 1.025 | 11.981 | 0.930 | 0.460 | 2.02x | 0.22x | 25 |

**Only 3 of 15 configurations improve actual cumulative time over the first three calls.** Others are still paying back calibration. Estimated total-call break-even ranges from 3 to 25. Timing confidence intervals are not claimed.

The v0.1 misrouting can be directly observed: in the 1M/eight-RHS paired three-big case, old v0.1 auto chooses within and its reuse median is 21.433 seconds. v0.2 measures the candidates and selects fereg, whose reuse is 1.962 seconds, but first-call calibration costs 124.222 seconds. **Correct selection and cheap selection are distinct engineering problems.**

All 15 configurations selected fereg on this machine under this adapter contract. This does not prove universal dominance, does not justify deleting other routes and does not reproduce the earlier bare-core benchmark scope.

## Environment and limits

Local performance: Linux x86-64, Intel Xeon Platinum 8573C, Python 3.13.5, 4 GiB container memory limit. Maximum measured isolated-process peak RSS, including inputs and reference residuals, was **2922.5 MiB**. Calibration can retain an incumbent and a challenger simultaneously; `scratch_bytes` is not a hard process-memory or fill bound. A soft probe budget cannot interrupt an in-progress Rust call.

Method order was fixed and only two reuse calls were timed, so host scheduling and cache noise remain. Sampled seeds do not characterize every topology. Large statistical-estimator/inference parity, real datasets, heterogeneous slopes, zero-weight handling and Windows/macOS binaries remain outside this release.

## Integration decision

Keep this experimental and opt-in. Explicitly choose a previously benchmarked route for one-shot fits; paid calibration is only a candidate when enough future projections can amortize it. The next engineering goal is **low-cost selection conditioned on expected reuse**, with capped probes, reliable cost accounting and additional estimator-level validation. Do not replace the production auto backend with this full-calibration prototype.
