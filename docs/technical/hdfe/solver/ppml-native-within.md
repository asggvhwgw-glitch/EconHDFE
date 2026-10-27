# Native within for weighted Poisson-family estimators

PPML and IV-PPML support `PPMLConfig(engine="within")` and
`IVPPMLConfig(engine="within")`. The default `engine="replica"`
and existing `engine="optimized"` remain unchanged. This adds an
engine choice without changing published public dataclass signatures
or the release-0.9.1 contract snapshot.

For the within engine, dense and block-separable Poisson estimators
use the existing `WeightedFEProjector(method="within")`.
The selection applies to post-FE rank screening, IRLS weighted solves,
separation-induced sample rebuilding, and the final high-precision
FWL/inference projection. Separation-specific simplex and ReLU
operators remain unchanged.

Native within requires strictly positive finite numerical weights.
The FE topology is reusable, but numerical weights are updated on each
IRLS iteration. For zero-weight states the existing generic LSMR
backend is selected, returning to native when weights are positive.
Absent the optional `_econhdfe_within` extension, an explicitly
requested within solve raises a configuration error.

Numerical tolerances, rank policy, moment conditions, separation,
sample semantics and VCE remain unchanged. Inspect
`result.diagnostics["projection_resources"]` to see the method
and effective backend. Speedup requires end-to-end validation.

## Million-row end-to-end profile (2026-10-27)

Single-thread Linux x86_64, 1,000,000 observations, three repetitions with alternating engine order. Timings include complete fits, not just residualization; generation/JIT and post-fit audits are excluded. `separation=("fe",)` is used and both engines receive the same tolerances.

| Structure | Estimator | optimized (s) | within (s) | Speed ratio | Update share of native fit |
| --- | --- | ---: | ---: | ---: | ---: |
| repeated | ppml | 4.317 | 1.882 | 2.29x | 6.8% |
| repeated | ivppml | 7.751 | 4.299 | 1.80x | 2.7% |
| crossed | ppml | 3.688 | 4.426 | 0.83x | 7.0% |
| crossed | ivppml | 9.563 | 12.022 | 0.80x | 2.8% |
| weak | ppml | 10.192 | 6.742 | 1.51x | 5.0% |
| weak | ivppml | 22.420 | 17.841 | 1.26x | 1.5% |
| schur | ppml | 3.424 | 2.165 | 1.58x | 22.5% |
| schur | ivppml | 6.149 | 3.780 | 1.63x | 9.1% |

See `benchmarks/hdfe/ppml_native_profile_1m_20261027.json` for measured-source identity, native-library and raw-record hashes, exact medians, and limitations. Reproduce with `benchmarks/hdfe/ppml_end_to_end_profile.py`.

Native is faster on repeated cells and small-Schur designs, but slower on the general crossing in this corpus. Do not change the default engine on this evidence. Eliminating numerical weight updates altogether would save only the measured update share, not a factor of ten. The weak-community design also compresses repeated cells; no pure preconditioner attribution is made.

## Certification propagation

The opt-in native nonlinear projector now requires a successful finite inner certificate at rank filtering, IRLS and final inference. A failure raises `ConvergenceError(code="convergence.weighted_fe")`; it is not hidden behind outer deviance convergence. The raw projector and linear paths retain their existing status-returning contract, and legacy replica/optimized behavior is unchanged. Tests include an independent full-dummy nonlinear moment root, six failure-injection locations, an actually exhausted native iteration budget, and zero-support/positive-weight transitions.

The original dynamic-weight microbenchmark generated non-dense labels and was rejected before timing native at 100k rows. It now uses `FEPlan.from_arrays`, like formal estimator inputs. Failed runs are not reported as speedups.
