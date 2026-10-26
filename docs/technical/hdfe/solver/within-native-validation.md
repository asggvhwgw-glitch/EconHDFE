# Native within validation — 2026-10-26

## What actually ran

Runtime commit: `266bee7717857773e48b67e0e93ab6b7dca4bd7b`.
Pinned upstream: `py-econometrics/within@fc44282cea6124f5f18bb142908605a656d35a7c`.
The final CI job compiled Rust in release mode, built and installed the Linux
x86_64 abi3 wheel, and tested the exact committed tree with no uncommitted diff.

- Rust unit tests: **5 passed**.
- Native / public integration tests: **39 passed**, no failures or skips.
- Existing full econhdfe suite: **396 passed**, four existing omission warnings.
- Installed CI wheel independently retested on local Python 3.13.5: **39 passed**.

The evidence is from Linux, not Windows/macOS certification or a package release.
Initial trial runs exposed a stopping-metric mismatch and missing shell pipefail;
those runs are not counted as accepted results. The final run uses explicit bash
pipefail, stricter inner stopping and bounded original-row residual correction.
No independent test tolerance was relaxed.

## Performance

One million synthetic rows, four RHS, two Rust threads, single-thread BLAS,
three repetitions, median wall time. First includes native setup and solve;
reuse changes the RHS. Every column passes the independent known-residual gate
`||r-r_true||_W / ||r_true||_W <= 1e-8`.

| Design | Original within first | Optimized first | First speedup | Original reuse | Optimized reuse |
|---|---:|---:|---:|---:|---:|
| Large firm + small region/year | 0.859 s | 0.198 s | 4.33x | 0.663 s | 0.125 s |
| Repeated four-way FE tuples | 0.864 s | 0.179 s | 4.82x | 0.829 s | 0.113 s |
| Repeated tuples, unequal weights | 0.920 s | 0.185 s | 4.96x | 0.889 s | 0.117 s |

The baseline is unmodified pinned within using the same wrapper, sequential RHS
policy and achieved-accuracy checks. These are NOT whole-estimator speedups or a
comparison against upstream solve_batch's different parallel scheduling policy.
This small controlled corpus is not a universal no-regression certificate.

A separate single-run local extension completed 10,000,000 rows / 4 RHS in
4.024 s first / 1.643 s changed-RHS reuse, compressing to 624,996 rows, with
8.61e-14 relative residual error. Process peak RSS was 2774 MiB, including input,
DGP and reference arrays. The 10M baseline attempt exhausted its 90-second entire-
script budget; it supplies no completed solve timing and no reported speedup.

Machine-readable summaries and provenance are in
`benchmarks/hdfe/within_native_20261026.json`. Full per-run JSON, build/test logs,
executed source archive and wheel are in GitHub Actions run **40272121466**, artifact
**12629006778**. The run trigger SHA precedes a controlled feature-branch generated-
file commit; `native-results/source-sha.txt` records the actual tested SHA above.

## Opt-in usage from this PR checkout

```bash
python -m pip install .
python -m pip install ./native/within-accel
```

The second command requires a Rust toolchain and the locked upstream dependencies.
There is no automatic PyPI installation, default-backend switch or release claim.

```python
from econhdfe import olshdfe, ivhdfe, HDFEConfig, ExecutionConfig

result = olshdfe(
    df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
    cluster=["firm"],
    hdfe_config=HDFEConfig(solver="within", tolerance=1e-10),
    execution_config=ExecutionConfig(threads=2, memory_budget_mb=1024),
)
print(result.absorb_info["within"])
```

`ivhdfe(..., method="within")` uses the same adapter. Remove the explicit selection
to return to established defaults. Python retains original-row sample, weights,
DoF, normalization and covariance semantics. Weak-design conditioning is preserved
for native OLS and OLS sessions. WeightedFEProjector supports the optional native
primitive and retains its established zero-weight fallback; high-level PPML and
IV-PPML routing are not changed in this PR.

Known reduction scratch and RHS concurrency are controlled; upstream factor fill
and whole-process RSS are not hard-bounded. See `within-native.md` for numerical
scope, rejection gates and remaining limitations. The PR is reviewable, not an
instruction to merge or publish automatically.
