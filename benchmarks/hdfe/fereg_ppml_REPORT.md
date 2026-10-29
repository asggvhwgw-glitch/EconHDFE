# End-to-end PPML: fereg 0.5.0 versus EconHDFE (within backend)

**Date:** 2026-10-29. **Source snapshot:** `b899ce84155c3a7712b48e9e73fd439b72d902dd`.
[GitHub Actions full results](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40809995049).

## Scope

- Two complete Python estimators on the **identical synthetic dataset in the same process**: `fereg.fepois` (native core exact global FE elimination + PCG) versus `econhdfe.ppmlhdfe(PPMLConfig(engine="within"))`.
- 5 FE connection topologies (one_big, two_big, three_big, weak_links, repeated_cells), 120k and 1M observations, 2 worker threads. Each case is run on one GitHub-hosted Ubuntu 24.04 runner; runners may have different CPUs across cases.
- Generated log conditional mean with one observed regressor and FE; outcome is **positive continuous**, so zero-group separation does not affect parity. No missing rows or intentional singleton filtering. Underlying true slope 0.18.
- fereg source pinned at `0fa59e3b539addbddaee8bd71acd031c497166a0`; EconHDFE source pinned by workflow. Estimator variance calculation in both is iid/model-based, not necessarily identical DOF convention.
- fereg uses `iwls_tol=1e-9, fixef_tol=1e-10`; EconHDFE uses `tolerance=1e-9, target_inner_tol=1e-10`. Both are full-function estimation times including setup, FE absorption and inference, **not inner-kernel timing**.
- Each case size is fitted twice with alternate tool order. The **second repetition** is reported below because the first 120k run includes substantial one-time JIT/prewarm overhead inside EconHDFE. Both repetitions and their sequence are retained in JSON.
- **All 10 experiments passed**; max absolute cross-estimator coefficient difference **3.0531e-15**, observational sample sizes match.

## 1,000,000 observations; second estimator call (warmed)

| FE topology | fereg fepois (s) | EconHDFE PPML-within (s) | fereg gain | slope absolute gap |
|---|---:|---:|---:|---:|
| one_big | 0.464 | 4.752 | 10.23x | 1.36e-15 |
| two_big | 0.932 | 2.317 | 2.49x | 1.44e-15 |
| three_big | 1.285 | 3.489 | 2.71x | 3.05e-15 |
| weak_links | 4.002 | 10.191 | 2.55x | 2.22e-16 |
| repeated_cells | 0.232 | 0.827 | 3.57x | 1.69e-15 |

## 120,000 observations; second estimator call (warmed)

| FE topology | fereg fepois (s) | EconHDFE PPML-within (s) | fereg gain | slope absolute gap |
|---|---:|---:|---:|---:|
| one_big | 0.068 | 0.418 | 6.20x | 1.39e-16 |
| two_big | 0.115 | 0.312 | 2.72x | 7.77e-16 |
| three_big | 0.132 | 0.464 | 3.52x | 1.05e-15 |
| weak_links | 0.112 | 0.601 | 5.39x | 2.78e-17 |
| repeated_cells | 0.035 | 0.118 | 3.35x | 6.11e-16 |

## Interpretation

1. All five 1M topologies show a clear **end-to-end fereg advantage** versus EconHDFE's currently integrated PPML-within route (approximately 2.5x to 10x in warmed calls). This is true even for the 3-big class where **bare within Rust wins** the direct two-RHS projection; estimator architecture and IRLS work dominate beyond the core alone.
2. Fereg returns practically identical regression slopes across these DGPs: gap ~1e-15, consistent with full estimator parity at this synthetic setting. The PPML benchmark did not independently verify all final FE raw scores or sandwich covariance equality; it establishes only successful convergence, equal nobs and beta parity.
3. No direct *fereg-compatible IV-PPML* API was found in the public fereg Python estimator interface; only EconHDFE internally implements IV-PPML. Do not claim end-to-end IV-PPML parity.
4. Differences in complete estimators may also include sample handling, Python formula parsing, dense numerical linear algebra, inference, IRLS speed and FE kernel overhead. Optimizing only the FE kernel will not reproduce the entire gain without profiling.
5. Cold-start numbers at 120k especially cannot be compared naively due to JIT compilation in EconHDFE. Both first/second calls are in the JSON.
6. For unweighted PPML, IRLS still changes the Hessian weight vector each outer iteration; fereg stateful `set_weights` from the **direct Rust test** is a plausible mechanism for improved end-to-end performance. Instrument per-iteration weighted solver cost before choosing production integration.

[Raw results and errors](fereg_ppml_20261029.json).
