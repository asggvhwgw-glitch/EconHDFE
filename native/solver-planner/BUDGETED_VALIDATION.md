# Budgeted planner v0.3: validation and paid-total performance

Date: 2026-10-29. Status: **numerically validated, opt-in experimental adapter; not a production-default improvement**.

## Result in one paragraph

The finite-horizon/cancellation mechanism works. Short workloads no longer pay v0.2's full calibration. However, the conservative default trial budget usually does not admit and promote the faster alternative. Only **2 of 12** primary configurations have lower measured cumulative time than existing auto; only **1 of 12** improves by more than 5%, while the other is a 1.2% difference without a confidence interval. v0.3 beats full-calibration v0.2 in **7 of 12** configurations. This release must not replace production auto. Negative and timed-out optional trials remain in the evidence.

## Implementation and numerical contract

`BudgetedAbsorber` supplements, rather than replaces, v0.2 `PlannedAbsorber`. It first obtains a certified result using the unchanged production `NativeSolver(strategy="auto")`, retaining its exact compression and SmallSchur behavior. By default optional exploration is deferred until the second real call. With default controls, total expected horizons of one through four calls do not start trials. No speculative incumbent warm-up solve is purchased.

Candidate kernels run in fresh private Python processes. They use all observations and a small actual RHS block, not a sampled graph. A 32-iteration stage can escalate to 128 on numerical convergence/certificate failure only, sharing one overall wall deadline. This reuses structural/preconditioner state, not an unaccepted residual or a continued Krylov basis. Unconverged or timed-out trials are never ranked as fast.

For observed current incumbent-call cost c and expected remaining calls R, default optional child-work budget is `min(3 seconds, 0.5*c, 0.1*R*c)`. A small-RHS pilot only screens candidates. Promotion requires certified output for the actual full RHS block, cross-route agreement and forecast payback: `R * max(0,c-candidate_full_call_cost) >= 1.25 * paid_optional_cost`. A promoted worker retains state and is idle between synchronous calls. `close()` releases it.

This is a forecast, not a worst-case performance theorem. Workload='reweight' uses an identical-weight refresh in the promotion measurement; it cannot predict an arbitrary future IRLS trajectory. Real changed-weight tests and arbitrary-weight unit oracles are separate checks.

The optional deadline can terminate native child work, but cannot preempt parent array copying, file I/O, process startup or OS cleanup. It is **not a hard total-call timeout**; necessary incumbent solves and subsequent selected-worker calls are outside this optional budget. Overrun is explicitly measured. Native iteration caps apply per inner projection attempt, not to total matvecs, symbolic setup or factorization work. `scratch_bytes` is not an aggregate RSS/fill cap.

No existing EconHDFE estimator default, sample policy, singleton/separation behavior, rank/DoF or covariance calculation was changed. Categorical intercept FE, strictly positive finite weights and fixed samples only. Linux x86-64 is the only certified native binary platform here.

## Source, builds and tests

- Branch: `feat/budgeted-planner-v03-20261029`.
- Runtime/tests/CI source commit: `23a0c12f04983514a64e7cc9b10145d1e34c584e`.
- Successful independent GitHub CI: [run 40828420235](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40828420235).
- CI artifact ID: `12876864543`, name `bounded-planner-validation-40828420235`.
- CI artifact SHA-256: `5e1177bf350255bc868c8017191a09ffa55f42ee6efa0403f77f892bf4426105`.
- within revision: `fc44282cea6124f5f18bb142908605a656d35a7c`.
- fereg revision: `0fa59e3b539addbddaee8bd71acd031c497166a0`.
- Local source tests: **86 passed**. GitHub's independently rebuilt native/Python packages: **86 passed**. Installed CI-built Python wheel with the measured native wheels: **86 passed**.
- CI additionally runs two 100k-row smoke cases. The million/five-million-row times below are **local performance data, not CI smoke timings**.

Tests include dense least-squares/SVD oracles, actual termination of an unresponsive child process, cleanup without a retained reader-thread reference cycle, invalid-weight rollback, internal refresh failure, arbitrary changing weights/RHS, sparse labels, zero/constant/empty RHS, full-block promotion, disagreement rejection, horizon accounting, numerical-cap escalation under one deadline and retained incumbent results after trial failure. A common test-only PPML outer loop on 600 rows checks coefficients, fitted means and scores. It is not production PPML/IV-PPML integration or covariance parity.

The new runtime modules match the CI wheel byte-for-byte. The inherited local v0.2 planner has one comment-only difference from the uploaded source; AST equivalence of all runtime modules was verified. Performance uses the previously validated v0.1 native wheels. The independent CI also rebuilt both native adapters; those binaries are a separate build-validation exercise.

## Benchmark method and accuracy

The primary grid has **12 configurations x 3 methods = 36 records and 231 returned projections**, all valid. Four additional v0.3 budget/pilot-width sensitivity records contribute 32 returned projections; combined total **40 records, 263 projections**, all valid. The maximum relative weighted L2 difference from an independent analytic residual or reference solve is **6.205895088857551e-11**, below the common `2e-8` benchmark threshold. Original-row requested certification tolerance remains `1e-10`.

Most unique-tuple fixtures construct weighted zero-sum residual flows around four-edge cycles; paired and repeated-cell fixtures are retained. The IID-three fixture instead uses independent balanced random factors with a separate SciPy LSMR reference. Its difference is reference-solve disagreement, not a proved forward-error bound. Changed weights in large analytic fixtures preserve the known cycle residual and are not a real PPML path. RHS columns are rescaled between calls; answers are not simply cached. All data are synthetic.

Local machine: AMD EPYC 9V74, Linux x86-64, Python 3.13.5, 4 GiB container memory limit; 2 native threads and 1 BLAS thread. Each method/configuration executes in a separate process; primary method order alternates across configurations. This is one workload trajectory per method/configuration, not repeated timing trials or a confidence interval. Do not compare absolute local times to older measurements on a different CPU.

Reported time = constructor + every actually executed update and residualization call in the stated horizon. It includes optional trial startup, copies, IPC, certification and rejected-trial cleanup. Data generation, independent oracle checking and final end-of-object close/teardown are excluded. The final close is executed, but was not included in the accumulated solver timing.

## Primary paid cumulative timings

Seconds; all calls in H were actually executed. Baseline is existing `NativeSolver(strategy="auto")`, not bare within. v0.2 is full calibration; v0.3 uses default budget and one-column pilot, with expected_calls set to H and scratch_bytes=1 GiB for all compared adapters. Main seed 31733, except IID/dense-repeat seed 71031 and the second unique-three seed 89173.

| Structure | Rows / RHS | H | Existing auto | v0.2 | v0.3 | v0.3 selected |
|---|---:|---:|---:|---:|---:|---|
| Unique one-big | 1M / 8 | 1 | 0.554 | 2.977 | 0.547 | existing auto |
| Unique two-big, changed weights | 1M / 2 | 3 | 1.704 | 3.231 | 1.917 | existing auto |
| Weak cycles | 1M / 8 | 8 | 24.075 | 9.633 | 25.774 | existing auto |
| Paired three-big | 250k / 8 | 8 | 22.913 | 14.153 | 9.214 | fereg |
| Unique three-big | 1M / 2 | 8 | 27.860 | 16.051 | 30.447 | existing auto |
| Unique two-big, changed weights | 1M / 8 | 8 | 19.183 | 14.904 | 20.992 | existing auto |
| Sparse repeated cells | 1M / 8 | 8 | 12.481 | 14.527 | 13.822 | existing auto |
| Weak cycles | 5M / 2 | 8 | 54.992 | 48.837 | 59.435 | existing auto |
| Sparse repeated cells | 5M / 2 | 1 | 4.573 | 31.091 | 4.839 | existing auto |
| IID three-big | 1M / 2 | 8 | 4.426 | 6.513 | 5.244 | existing auto |
| Dense repeated cells | 1M / 8 | 8 | 4.577 | 9.770 | 4.970 | existing auto |
| Unique three-big, second seed | 250k / 8 | 8 | 16.260 | 5.137 | 16.703 | existing auto |

The paired-three case demonstrates successful paid promotion: 0.971 seconds of optional work led to 9.214 seconds total versus 22.913 existing-auto and 14.153 full-calibration. This is approximately 2.49x faster than existing auto for that workload. It is the only >5% gain in this primary grid.

The large one-shot repeated-cell case avoids v0.2's 31.091-second calibration-heavy result, but v0.3's 4.839 seconds is still slower than existing auto's 4.573 seconds. Skipping calibration is not equivalent to having zero adapter overhead. Likewise, constrained trials miss clear opportunities in the weak-cycle and unique-three cases where v0.2 is faster.

Primary optional elapsed time is at most **2.931 seconds**; maximum measured overrun above that invocation's effective budget is **0.066 seconds**. This is an empirical cancellation result, not a real-time guarantee. Separate parent and child peak RSS measurements reach 3021.13 MiB and 2488.88 MiB respectively; these are not a simultaneously measured aggregate peak and should not be added or described as a hard bound.

## Budget/pilot sensitivity — negative results retained

Both sensitivity workloads use 1M rows, 8 RHS and 8 calls. They raise max_overhead_fraction to 1 and exploration_fraction to 0.25, keeping max_probe_seconds=3 and the 32/128 stage policy. Results are subsequent same-machine runs, not simultaneous controls or confidence intervals.

| Case | Default v0.3 | Wider budget, 1-column pilot | Wider budget, 8-column pilot |
|---|---:|---:|---:|
| Weak cycles | 25.774 | 27.763 | 28.178 |
| Two-big, changed weights | 20.992 | 21.928 | 22.097 |

All four sensitivity records are numerically valid and retain existing auto. The one-column screening estimate can falsely reject a fast fused multi-RHS solver because it extrapolates a cold scalar solve linearly. Using the full block avoids that particular extrapolation, but then consumes most of the budget before the required promotion/warm measurement. The full-block weak case pays 3.017 seconds of optional work against a 3-second child-work budget; parent cleanup accounts for the recorded overrun.

## Decision

Do not merge this as a replacement production auto policy. It provides a working experiment for finite-horizon costs, safe candidate cancellation and retaining a valid incumbent, but its conservative default performance is insufficient. Increasing constants alone did not fix the tested failures.

The next justified engineering change is to reduce the cost of obtaining useful candidate information: native cooperative cancellation during setup and iteration, reusable RHS buffers/geometry instead of child .npy transfers, and separating first-use factorization from steady-state fused-RHS cost. Keep explicit backend selection and existing auto available while testing that change. No universal speedup, bounded-regret theorem, real-data representativeness, production PPML/IV-PPML or covariance parity is claimed.

Reproduction: `run_budgeted_suite.py` and `benchmark_budgeted.py`; use `--pilot-rhs 8 --overhead 1 --explore .25` for full-block sensitivity. The validated bundle retains complete per-call JSON, logs, unit/CI evidence, source provenance and SHA-256 manifests. Development-only cap-32 experiments are archived separately and excluded from the final counts and table.
