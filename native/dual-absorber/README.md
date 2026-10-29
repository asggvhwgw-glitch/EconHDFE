# Dual absorber prototype v0.1

An independently installable, explicitly opted-in categorical fixed-effect residualizer.
It adds **no production engine**, changes **no existing estimator/default**, and lives alongside
`native/within-accel`. The package is `econhdfe-dual-prototype`, import `econhdfe_dual`.

## What is implemented

- Direct pinned Rust calls to `within::Solver::solve_batch` and `fereg_core::FeSolver`.
- Original-order, multi-RHS weighted residualization through one Python/Rust call.
- Persistent topology: cached within `Design`, or the fereg solver with `set_weights`.
- A conservative structural heuristic, explicit `within`/`fereg` overrides, and an independently certified fallback.
- Input column scaling, positive-weight validation and normalization, label compaction, preservation of original observation order.
- A common original-row **maximum weighted group-mean** acceptance test for every RHS.
- Diagnostic records for route/reason, setup, all numerical attempts, corrections, fallback and weight generations.
- Dense least-squares oracles, a common safeguarded test-only PPML IRLS loop, and large known-residual tests.

This is an adapter around the upstream fereg solver, **not a new independent implementation of its algorithm**.

## Build and use

A Rust toolchain, C compiler and Python >=3.10 are needed to build locally.

```sh
python -m pip install ./native/dual-absorber
python -m pip install pytest
python -m pytest native/dual-absorber/tests -q
```

```python
import numpy as np
from econhdfe_dual import DualAbsorber

rng = np.random.default_rng(29)
n = 100_000
codes = np.column_stack([
    rng.integers(10_000, size=n),
    rng.integers(50, size=n),
    rng.integers(10, size=n),
]).astype(np.uint32)
Y = rng.normal(size=(n, 8))
w = np.exp(.3 * rng.normal(size=n))

absorber = DualAbsorber(codes, weights=w, backend="auto", threads=2)
R, diagnostics = absorber.residualize(Y, tol=1e-10, return_info=True)
assert diagnostics["accepted"]

# IRLS-style weights: same observations and same FE topology.
absorber.update_weights(w * np.exp(.05 * rng.normal(size=n)))
R2 = absorber.residualize(Y)
print(absorber.backend, absorber.weight_generation)
```

Set `backend="within"` or `backend="fereg"` and `allow_fallback=False` for an isolated backend test.
The return shape follows the input vector/matrix. Inputs are not changed. All returned rows remain in original order.

## Auto policy v0.1

The route is selected on the **first RHS block**, then held for state reuse unless numerical failover occurs.

1. One FE, or total other-factor cardinality <=4096: fereg.
2. At least three comparable large factors (third >=4096, first <=4x third): within.
3. Evidence of duplicate tuples in a small deterministic sample: fereg candidate.
4. Two-way FE, strong factor-size asymmetry, or >=4 RHS outside the balanced-three-way guard: fereg candidate.
5. Otherwise: within.

The thresholds are **heuristics**, not a trained classifier, certified runtime guarantee or a measurement of graph spectral gaps.
The duplicate sample fraction is **not U/N**; it is only a hint and can miss compressible structures or overstate their prevalence.
There is no full tuple-index construction solely for route selection. Fereg carries out its own exact cell preprocessing.
The existing EconHDFE `compressed_within` route is retained unchanged and benchmarked separately.

## Numerical contract

For strictly positive weights W and the categorical indicator matrix D, return R approximately equal to
`Y - D (D' W D)^+ D' W Y`. The mean test for column j is

```
max_over_factors_and_levels |sum_i_in_level w_i R_ij / sum_i_in_level w_i|
--------------------------------------------------------------------
                       max_i |Y_ij|
```

Zero columns are handled without division by zero. The gate is `max(tol, 128*eps)`.
The source is scaled before solving and restored afterwards. Each underlying algorithm constructs the
subtracted fit in the FE span; a mean test alone would not establish that property for an arbitrary candidate residual.

**This is a backward/stationarity check, not a condition-number-independent forward-error certificate.**
Small tests compare full residuals with an independent dense SVD/lstsq oracle. Large benchmarks use an analytically
known residual orthogonal to all FE in the current weights, which supplies an additional forward-error check.

A kernel's own convergence flag cannot bypass the independent gate. One residual-replacement correction is allowed.
Failure can try the other backend once for that block; failure there raises rather than returning unverified results.
`max_iter` applies **per kernel projection attempt**, not to total work across correction/fallback. The attempts array
records accepted and failed work; final `iterations` reports the accepted backend path, not omitted failed work.
No FE coefficient warm starts are used in v0.1.

Invalid weights leave the old state intact. If a native numerical update fails after validation, cached numerical state
is invalidated while the old public weights are retained; a subsequent solve rebuilds for those old weights.

## Scope and exclusions

- Categorical intercept FE only; labels must be nonnegative uint32-compatible integers. Arbitrary label gaps are compacted.
- Strictly positive finite weights only. Zero-weight sample trimming belongs to the estimator and is **not** implemented here.
- No singleton deletion, statistical sample rules, rank/DoF policy, separation handling, covariance estimator or IV-PPML outer loop.
- Test-only common PPML IRLS is not a replacement for the production PPML estimator; no production inference claim follows from it.
- `scratch_bytes` controls an estimated multi-RHS block width; it is **not** a hard process-memory cap and does not bound upstream sparse-factor fill.
- Fereg uses its public scoped pool interface. A pool is opened per native fereg call; the numerical state still persists.
- Current validation workflow targets Linux x86-64; other platforms are not certified by this prototype workflow.

## Reproducibility

Both dependencies are pinned to exactly the revisions used in the preceding core comparison:

- within: `fc44282cea6124f5f18bb142908605a656d35a7c`.
- fereg: `0fa59e3b539addbddaee8bd71acd031c497166a0`.

CI archives the actual `Cargo.lock`, source SHA, compiler build logs, Python environment, numerical test report,
benchmark results, and wheels. The default library code remains on its existing backend path.

The large benchmark deliberately constructs paired identical FE tuples to obtain an exact residual oracle.
Consequently it covers the target size/degree patterns, but **not fully unique-row topology, real administrative data,
or a comprehensive worst-case graph family**. It must not be advertised as universal speed dominance.

## Attribution

The adapter is BSD-3-Clause, like EconHDFE. The pinned `within` and `fereg` upstream projects are MIT licensed.
Their algorithms and source remain upstream dependencies; preserve upstream notices when redistributing their artifacts.
