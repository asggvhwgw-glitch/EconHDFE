# Post-architecture consistency and performance validation

This checkpoint validates the package after the Data Layer and unified Execution Planner architecture work. It is a development validation checkpoint, not a versioned release; the package version remains `0.4.10.1`.

## Scope and acceptance rule

The planner refactor is required to preserve econometric semantics exactly. Performance is evaluated with repeated medians on the same runtime rather than single wall-time observations. On the shared development container, changes within roughly ±10% are treated as ordinary timing noise unless they reproduce consistently across alternating runs or can be traced to a changed execution plan.

The test matrix covers:

- OLS-HDFE and linear IV-HDFE;
- PPML-HDFE and IV-PPML-HDFE;
- two- and three-way FE;
- robust and two-way clustered covariance;
- generic four-way HDFE projection;
- complex event-study/factor specifications with structural omissions and FE canonicalization;
- heterogeneous/block designs;
- automatic thread selection;
- planner overhead and dense/block operator calibration;
- retained legacy Python expectation fixtures where applicable.

## 1. Full regression and governance gates

- Regression suite: **461 / 461 passed**.
- Public-contract gate: **0 changes**.
- Technical-innovation registry: **PASS, 3 registered innovations**.
- Architecture map: **PASS/current**.

The monolithic pytest invocation exceeded the tool execution window after ~78% without failures, so the suite was completed in two deterministic shards: **265 + 196 = 461**.

## 2. Cross-check against the pre-planner checkpoint

A deterministic seven-model DGP was run against the immediately preceding Data Layer checkpoint and the unified-planner checkpoint.

| Model | Coef max diff | SE max diff | VCV max diff | Metadata |
|---|---:|---:|---:|---|
| OLS, 2 FE, robust | 0 | 0 | 0 | exact |
| OLS, 3 FE, 2-way cluster | 0 | 0 | 0 | exact |
| IV, 2 FE, robust | 0 | 0 | 0 | exact |
| IV, 3 FE, 2-way cluster | 0 | 0 | 0 | exact |
| PPML, 2 FE, robust | 0 | 0 | 0 | exact |
| PPML, 3 FE, 2-way cluster | 0 | 0 | 0 | exact |
| IV-PPML, 2 FE, robust | 0 | 0 | 0 | exact |

“Metadata” includes the applicable N, rank, absorbed DoF, inference DoF, convergence/iterations, separation counts, cluster counts and solver selection.

This is the strongest planner consistency result: the architecture refactor produced **bit-identical reported numerical results on this matrix**.

## 3. Generic 4-FE stress test

A 200k-observation, eight-regressor, four-FE design was run with explicit 1/2/4 HDFE threads. Coefficients and standard errors are exactly identical between the pre-planner and planner checkpoints at every thread count; solver and iteration counts also match.

| Threads | Pre-planner median | Planner median | Current / baseline |
|---:|---:|---:|---:|
| 1 | 0.6400 s | 0.6369 s | 0.995 |
| 2 | 0.7233 s | 0.6268 s | 0.867 |
| 4 | 0.5995 s | 0.6333 s | 1.056 |

The first batch run showed an apparent large regression during a period of shared-container contention; alternating reruns eliminated it. This is why release performance gates should never be based on one wall-time observation.

With `absorb_threads="auto"`, the planner resolved to four threads and produced a 0.6026 s median in a separate run. That is within the best explicit-thread performance region on this machine. No thread-policy change is justified by this evidence.

## 4. Complex empirical-style event study

A 300k-observation event-study specification with factor expansions, explicit omissions, clustered inference and a deliberately redundant FE hierarchy was tested in three alternating pre-planner/planner runs.

The specification canonicalizes to `firm + city#year`; 1,904 structural/automatic omitted columns are handled identically. N, rank, DoF, solver, iteration count, effective FE set, omission metadata and all coefficient errors are exactly equal across checkpoints.

- Pre-planner median: **1.0048 s**.
- Planner median: **0.9480 s**.
- Current / baseline: **0.943**.
- Peak RSS: effectively unchanged (~323 MiB).

The historical benchmark harness was also repaired to pass a DataFrame rather than the obsolete raw-dict data container expected before the Data Layer contract was introduced. This is a benchmark-maintenance fix, not an estimator change.

## 5. Heterogeneous/block execution

At 72k observations with six coefficient blocks and six local slopes per block, three independent runs were compared using the existing four-model heterogeneous-specification benchmark.

| Model | Pre-planner median | Planner median | Current / baseline |
|---|---:|---:|---:|
| OLS | 0.0840 s | 0.0723 s | 0.861 |
| Linear IV | 0.5590 s | 0.5849 s | 1.046 |
| PPML | 0.1417 s | 0.1405 s | 0.992 |
| IV-PPML | 0.1267 s | 0.1328 s | 1.048 |

All structured-vs-dense coefficient differences remain at machine precision. The two modest slowdowns are below 5% and do not reproduce as a broad planner regression; they are kept as benchmark evidence rather than tuned away.

## 6. Planner overhead and representation calibration

A composite planner call (memory + representation + parallel + `ExecutionPlan`) costs approximately:

- **12.4 microseconds per plan** over 100,000 calls.

Planner work is confined to preparation/compilation boundaries; integer thread decisions are consumed directly inside hot projection kernels.

The dense/block operator calibration reinforces the decision not to let byte traffic alone determine production representation. With 72–89% storage savings, block execution is consistently strong for transpose-matvec, but at four BLAS threads the least sparse case is only approximately tied on ordinary matvec. Representative current measurements:

- 72% storage savings, 4 BLAS threads: dense/block matvec ratio ~**1.02x**, transpose-matvec ~**2.43x**.
- 81% savings: matvec ~**1.31x**, transpose-matvec ~**4.72x**.
- 89% savings: matvec ~**2.79x**, transpose-matvec ~**4.22x**.

Therefore the existing conservative production rule remains appropriate: exact certificate first, meaningful storage savings, then repeated-pass or memory-pressure justification. Hardware/operator calibration may refine this later, but no default is changed in this checkpoint.

## 7. Legacy expectation fixtures

The IV-PPML stored Python expectation fixture remains essentially exact:

- coefficient max difference: `2.78e-16`;
- SE max difference: `9.37e-17`;
- VCV max difference: `8.58e-17`;
- N, absorbed DoF, separation and mu-separation counts all match.

The older PPML Python fixture has a known absorbed-DoF mismatch for two-/three-FE cases (for example 20 vs current exact 19). This mismatch already exists in the pre-planner checkpoint and reflects the later exact FE-rank convention, not the planner refactor. Coefficients remain within `7.11e-15`; the largest SE difference is `3.82e-06`.

## Decision

The `econhdfe/` runtime source tree is byte-for-byte unchanged relative to Planner checkpoint 1; this validation checkpoint adds benchmark/validation evidence, documentation, generated architecture outputs, and one stale benchmark-harness repair only.

The architecture work passes the current consistency/performance gate.

1. No econometric result change was detected from the planner refactor.
2. No stable end-to-end performance regression was detected.
3. Automatic HDFE threading is in the local performance sweet spot; no new hardware-specific default is justified yet.
4. The production dense/block heuristic should remain conservative.
5. Further planner abstraction has low expected marginal value. Future planner work should be limited to real-machine calibration and bug fixes unless new evidence shows a systematic miss.

Raw benchmark evidence and the aggregate machine-readable summary are under `benchmarks/planner/post_architecture/`.
