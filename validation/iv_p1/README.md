# P1 inference checkpoint — 2026-10-28

This directory records the weak-identification / growing-FE follow-on to the **local overidentified dev1 artifact**. It is not a source push, main integration, release, or new cross-platform/Stata certification. Existing estimator defaults are unchanged.

## Implemented interfaces

```python
from econhdfe.inference.iv_poisson import (
    RestrictedPoissonS, PreparedPanelS, panel_s_test,
    circuit_s_test, invert_scalar_grid,
)

# Every endogenous slope must be fixed in beta0. C is regular nuisance.
null_model = RestrictedPoissonS(
    y, E, Z_excluded, exog=C, absorb=[firm, year], clusters=firm,
)
null_result = null_model.test(beta0)
# Validity requires regular null nuisance and an independent-cluster CLT.
# Multiple FE arrays are computationally supported, not automatically
# statistically certified in any growing-dimensional asymptotic regime.

# Scalar inversion only: use when E has exactly one target coefficient.
# result is a discrete accepted set, never a certified continuum interval.
# grid_result = invert_scalar_grid(lambda b: null_model.test([b]), grid)

# One-way short panel: all slopes in X are fixed, other effects are known offsets.
# This is a different moment family under EXTRA cross-time assumptions.
panel = PreparedPanelS(
    y, X, Z_excluded, firm, method="cross_product", offset=known_offset,
    assume_cross_time_independence=True,
)
panel_result = panel.test(beta_all)

# Long-panel bias correction: predetermined equal halves within every unit.
long_panel = PreparedPanelS(
    y, X, Z_excluded, firm, method="spj", offset=known_offset,
    split=predetermined_half, assume_long_panel=True,
)
```

Read `docs/technical/ivppml/p1-weak-identification-growing-fe.md` before selecting a method. In particular `assume_*` is an assertion by the user, not a condition verified by software. No new method is a misspecification-robust confidence set for the parent estimator's pseudo-true value. Weak-IV robustness does not repair invalid instruments, fixed-short-panel nuisance bias, few clusters, or arbitrary serial/dyadic dependence.

## Reproduce tests

From source root, with the parent package dependencies installed:

```sh
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/p1 -q
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/ -q
```

Numerical test cases compare against independent full-dummy null roots/projections, exact finite-state expectations, frequency replication, explicit pair/product sums, and independently computed cluster covariance. Grid unknown points are not treated as rejected. `full_tests_initial.log` deliberately preserves the first run's stale-map and layer-ownership failures; these are corrected without changing statistical calculations. `tests_first.log` and `tests_second.log` retain discovery of the parent projection range problem. Final logs are separate.

## Reproduce simulation and timings

```sh
export NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python benchmarks/p1/run_inference_mc.py --suite fixed --reps 1000 --output validation/iv_p1/mc_fixed.json
python benchmarks/p1/run_inference_mc.py --suite panel --reps 1000 --output validation/iv_p1/mc_panel.json
python benchmarks/p1/run_inference_mc.py --suite circuits --reps 1000 --output validation/iv_p1/mc_circuits.json
python benchmarks/p1/run_inference_mc.py --suite inversion --output validation/iv_p1/inversion_examples.json
python benchmarks/p1/bench_inference.py --help
python benchmarks/p1/reproduce_parent_projection.py
```

There are 14,000 formal MC datasets: three fixed-nuisance cells, eight one-way panel cells and three circuit cells, 1000 replications each. Point-null rejection rates are **not** confidence-grid coverage results. JSON retains p-values, failures, Wilson MC intervals, deliberate invalid-assumption counterexamples and low-power outcomes. The fixed-T profile/SPJ stress tests intentionally violate their stated long-panel rate conditions. Serial dependence intentionally violates the cross-product route's off-diagonal assumptions while maintaining contemporaneous additive-IV validity.

Timing tables concern **one null test**, not complete IV point estimation or full confidence inversion. Preparation is separate; 3-repeat medians use same-host serial execution. Process RSS is a high-water mark. Large-sample timing does not certify inference under the benchmark's FE dimensions.

## Integration caveats

- Runtime label remains the parent's local `0.10.1.dev1`; use the artifact manifest and hashes to identify this P1 checkpoint. Main's 0.9.2 version policy is not changed.
- The new lower-layer `hdfe/scaled_projection.py` duplicates the numerically identical short dev1 equilibration adapter to avoid an inference-to-model dependency; parent behavior is not modified.
- No source was pushed to PR36. Main and post-#35 compatibility must be tested separately.
- Circuits require caller-supplied **observation-disjoint, FE-balanced** index sets and stronger conditional moment assumptions. The code does not claim a solver for arbitrary sparse/unbalanced FE networks.
