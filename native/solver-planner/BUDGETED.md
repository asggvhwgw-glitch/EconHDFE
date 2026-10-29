# BudgetedAbsorber v0.3 — bounded optional route exploration

This is an opt-in planning experiment. It does not replace an EconHDFE estimator or alter its default backend, sample, singleton/separation, rank/DoF or inference policy. `PlannedAbsorber` retains the v0.2 full-calibration interface for controlled comparison.

## Policy

1. Obtain a certified result from the unchanged existing `NativeSolver(strategy="auto")`. Its exact-tuple compression and SmallSchur routes remain available.
2. Do not speculate on the first call by default. Observe a second **real** call before forecasting repeated-call cost; no extra incumbent warm-up solve is purchased.
3. With the defaults, horizons of one through four total calls never start optional trials. `expected_calls` is a workload estimate supplied by the caller, not a claim that any future work is scheduled.
4. Trial candidates use fresh Python processes and pinned Rust adapters. They receive all observations and a small actual RHS block; no row-subsampled graph is used. A deadline can terminate a nonresponsive native trial without corrupting the incumbent's solver state.
5. Start with an iteration cap of 32. On a numerical convergence/certificate failure only, try at most 128 using the same candidate's structural state and the same overall deadline. Unaccepted residuals are never returned or ranked. This is not Krylov-vector continuation.
6. A scalar-pilot cost extrapolation screens promotion. It is heuristic: fused multi-RHS scaling and first-use factorization can make it pessimistic. This remains a known source of missed opportunities, not a proven lower bound.
7. A candidate is retained only after completing the actual full RHS block at the requested accuracy, agreeing with the incumbent and passing an estimated finite-horizon payback test. One accepted challenger ends calibration. The selected worker stays idle between synchronous calls and retains its numerical state.

Let c be the observed current incumbent-call cost (including a pending weight refresh), R the expected remaining calls, and B the optional budget. Defaults use:

    B = min(3 seconds, 0.5*c, 0.1*R*c)

A promotion requires:

    R * max(0, c - candidate_full_call_cost) >= 1.25 * paid_optional_cost

This is a forecast. A changed RHS, changed weights, host scheduling or factorization policy can invalidate it. Actual paid cumulative performance must still be measured; no universal speedup or regret theorem is asserted.

`workload="reweight"` includes an identical-weight refresh in the promotion measurement. It is not a simulation of an arbitrary future IRLS trajectory.

## Deadline and resource contract

`max_probe_seconds` is **not a total-call deadline**. It applies to optional child work after an incumbent answer exists. Parent array copies, .npy I/O, process creation, response parsing and OS cleanup cannot be preempted by this mechanism; actual overruns are reported as `budget_overrun_seconds`. The necessary incumbent solve and later retained-worker projections are outside the optional deadline.

An iteration cap applies per native adapter projection attempt, not to symbolic setup, factorization work or a global count of all matrix-vector products. Native certification may do a corrective attempt. A deadline remains necessary even with the iteration cap.

`scratch_bytes` is the existing adapter's working-buffer admission hint, not a hard aggregate RSS/factor-fill cap. During a trial, the incumbent and challenger can coexist. All IPC and data movement are included in external wall timings. Use a single caller per absorber; this protocol is not a concurrent shared-object API.

## Usage

```python
from econhdfe_planner import BudgetedAbsorber

with BudgetedAbsorber(
    codes,
    weights=w,
    expected_calls=8,
    workload="reweight",
    threads=2,
    max_probe_seconds=3.0,
) as absorber:
    residual, info = absorber.residualize(Y, tol=1e-10, return_info=True)
    absorber.update_weights(w_next)
    residual_next, info = absorber.residualize(Y_next, tol=1e-10, return_info=True)
    print(info["backend"], info["plan"])
```

`expected_calls=1` is the default and performs no optional calibration. Explicit backends are `within`, `compressed_within`, `fereg` and `legacy_auto`. `retune(expected_calls=...)` retains the incumbent and requests a new bounded exploration on a later synchronous nonempty call. The revised expected count includes calls already made.

Use the context manager or `close()` to release a retained child promptly. The top-level `backend` field reports the active route; a stored plan records its calibration decision, while `fallback_errors` separately records certification failover.

## Validation scope

Dense SVD/least-squares oracles, arbitrary changed weights, invalid-update rollback, same-outer-loop PPML checks, real process cancellation, timeout fallback, numeric-cap retry, full-block promotion, cross-route disagreement and process cleanup are tested. Million/five-million-row measurements use analytic cycle/paired residuals and an independent SciPy reference for IID FE data. All are synthetic.

Strictly positive finite weights and categorical intercept FE only. No production PPML/IV-PPML integration or covariance parity is claimed. Zero-weight handling, heterogeneous slopes, real administrative datasets and non-Linux binary certification remain out of scope. The Python worker avoids forking initialized native thread pools, but this release's binaries and process behavior were validated on Linux x86-64 only.
