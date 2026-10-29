# Three-route FE planner — v0.2.0a1

An **opt-in experimental planning layer** over the already pinned Rust cores.
It does not change any EconHDFE estimator, singleton/separation rule, rank/DoF
calculation, covariance estimator, or default backend. It is a separately
installed Python package, not a rewrite of either Rust solver.

## Execution routes

* `within`: the v0.1 adapter's actual Rust `solve_batch` route, preserving Design
  across numerical weight updates.
* `compressed_within`: the existing native `NativeSolver(strategy="compressed")`.
  Exact full-tuple compression and original-row certification are retained. This
  legacy compressed adapter still loops over RHS; v0.2 does not falsely claim a
  newly fused compressed Rust implementation.
* `fereg`: the v0.1 adapter over pinned `FeSolver`, fused residualization and
  stateful `set_weights`.

The numerical kernels are unchanged: within `fc44282cea6124f5f18bb142908605a656d35a7c`,
fereg `0fa59e3b539addbddaee8bd71acd031c497166a0`. Both dependency notices remain in
`native/dual-absorber/THIRD_PARTY_NOTICES.md`.

## What changed from the first prototype

A **full-stream 64-bit tuple sketch** replaces the misleading small-row duplicate
sample. Every row contributes, with bounded-size registers and chunk buffers.
An approximate error allowance admits the compression candidate only when the
upper estimated distinct-tuple ratio is at most 0.7. This sketch is for cost
selection only: it never merges rows, defines FE equivalence, or changes a
numerical projection. Hash collisions can affect admission, not correctness.

There is no rule saying "three large factors implies within". With
`backend="auto"`, admitted candidates are measured on the **complete graph,
actual RHS block and current weights**, using the same requested tolerance and
independent numerical gates. A first solve and one or more warm solves are timed.
For `workload="reweight"`, warm timings also include an identical-weight refresh.
Candidates that fail their certificates are not ranked. Certified projections
are cross-checked; material disagreement fails closed.

The objective is the measured cost of a **future call**, not speed on the first
call. After calibration, only the winner is retained and its already-computed
projection is returned. Calibration cost, copies, validation, losing candidates
and geometry scanning are NOT subtracted from first-call timing. A measured
planner can be a poor choice for one-shot estimation.

The route stays stable after the first call. Changed RHS width is flagged in the
diagnostics, not silently retuned. `retune()` explicitly requests calibration on
the next nonempty block. Weight changes preserve the chosen core's supported
state; invalid inputs leave public weights unchanged, and an internal refresh
error invalidates numerical state so the next solve rebuilds using old weights.

## Usage

```python
from econhdfe_planner import PlannedAbsorber

p = PlannedAbsorber(codes, weights=w, backend="auto", threads=2,
                   workload="reweight", probe_repeats=1)
residual, info = p.residualize(Y, tol=1e-10, return_info=True)
print(info["plan"]["selected"], info["plan"]["calibration_seconds"])
p.update_weights(w_next)
residual_next = p.residualize(Y_next, tol=1e-10)

# Avoid calibration cost for a known workload:
q = PlannedAbsorber(codes, weights=w, backend="compressed_within", threads=2)
```

`probe_budget_seconds` is checked **between** native calls. It cannot interrupt a
native call and is not a hard timeout. An untested candidate is labelled untested,
never labelled slower. `scratch_bytes` controls existing adapter workspace
admission; it does not bound total process RSS or third-party factorization fill.
During calibration at most an incumbent and a challenger are retained, and even
that may be too large for a constrained machine.

## Installation

From this repository (Rust and a C toolchain required for native dependencies):

```sh
python -m pip install ./native/within-accel ./native/dual-absorber
python -m pip install ./native/solver-planner
python -m pytest native/solver-planner/tests -q
```

The downloadable experimental bundle additionally contains the validated Linux
x86-64 native wheels and the pure-Python planner wheel. Those native binaries are
NOT macOS or Windows builds. With compatible Python and NumPy, install the three
wheels from the bundle; otherwise build the native dependencies from source.

## Validation boundaries

Tests use an independent dense SVD/least-squares oracle, state rollback,
row-order/layout changes, sparse labels, deliberately failed candidates, solver
failover, and a common **test-only** PPML outer loop. Production PPML/IV-PPML and
covariance parity are not introduced in this version.

Large fixtures now include both repeated tuples and **mostly unique four-edge
cycles**. For a cycle with edges (a0,b0),(a0,b1),(a1,b0),(a1,b1), set
`W r = (+e,-e,-e,+e)` and hold any third FE constant on that cycle. Each factor's
weighted sum is zero. Adding an explicit `D alpha` gives an independent known
residual without duplicating full tuples. These are still designed synthetic
fixtures, not representative administrative data or a universal complexity test.

All compared paths use positive weights and the same original-row accuracy
requirements. Benchmark time excludes synthetic-data generation and independent
oracle checks, but includes every actual planner/solver call. The large suite
runs each method/configuration in a separate process. First-call and repeated
costs must be reported separately; a projected break-even is not an observed
long-horizon speedup.
