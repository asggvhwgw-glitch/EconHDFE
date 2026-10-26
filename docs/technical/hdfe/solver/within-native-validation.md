# Native within validation — 2026-10-26

## Integration on 0.9.0 and local macOS validation

The backend was integrated on main `45ab1bc9dc69ef6fb4b85d263cc65d24b35d7692`
(PR #26). The Session conflict preserves both the new `entry.fe.groups` layout
and weak-design conditioning for `method="within"`; generated architecture maps
were rebuilt. Base/default estimator routing is unchanged.

On macOS arm64 with research Python 3.12, a locally compiled abi3 wheel passed
5 Rust tests, **43 native/integration tests**, and **464 base tests**. The four
new native Session cases exercise weak and hidden weak directions, weighted and
unweighted fits, prewarm, reordered columns, data mutation and persistent cache
reuse against the established standalone backend. Native CI now includes Linux
and macOS; these local results are not Windows native certification.

Whole-estimator comparisons use the existing project samples, four threads and
a 512 MiB workspace budget. Timings exclude data loading/imports. Each row
passed coefficient, SE/covariance, residual, sample and rank checks; the complex
IV case additionally uses its existing independent observation-space reference.

| Existing sample | Default median | Native median | Interpretation |
|---|---:|---:|---|
| 100k composite IV, 21 parameters, three FEs | 11.523 s | 4.070 s | 2.83x faster |
| 500k two-FE OLS, four controls | 0.152 s | 0.157 s | about 3% slower |
| 500k two-FE IV | 0.284 s | 0.289 s | about 2% slower |
| 500k Session, construction + four specifications | 0.319 s | 0.392 s | about 23% slower |

These are three-run medians (alternating backend order for the 500k rows).
The 100k comparison uses tol=1e-12, retains 97,056 observations after 2,944
singleton deletions, and keeps rank 21 and absorbed DoF 26,890. All independent
reference gates pass; numerical tolerances were not relaxed. The 500k runs use
tol=1e-10 and preserve the refactor benchmark's seed and DGP.

The original 5M composite IV attempt was rejected during design materialization
by available-memory admission, at both 512 and 2048 MiB requested budgets.
It supplies no completed model timing or speedup claim. Native peak RSS and
Schwarz fill-in remain outside a strict workspace-memory guarantee.

Machine-readable results, runtime source hash and same-wrapper 1M microbenchmarks
are in `benchmarks/hdfe/within_native_macos_090_20261026.json`. The mixed results
support keeping this backend explicitly opt-in rather than changing defaults.
High-level PPML/IV-PPML, slopes, GPU and group/individual routes remain outside
this native adapter's scope. No package publication was performed.

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

## Additional 5M full-IV retry with relaxed memory admission

A subsequent local run using `memory_policy="relaxed"`, 8192 MiB requested
workspace, four threads and the unchanged `tol=1e-12` cleared the earlier
memory-admission failure. It did **not** complete estimation: the native
original-row gate reported `convergence.hdfe` after 202 iterations, with a
metric of `1.0456878418718674e-12`. Fit time before failure was 243.153 seconds;
process peak RSS was 3.439 GiB. The configured iteration cap was 16000, so this
was not exhaustion of that cap. The native wrapper permits only two residual
replacement passes before final certification. No tolerance was loosened.

This result does not establish a numerical regression, but it prevents a claim
that the 5M full-IV specification is validated. There is no successful 5M
within timing or speedup. Smaller validated models and projection-only
microbenchmarks do not override this failure.

The default backend subsequently completed the same specification in 970.075
seconds (single cold-process fit, excluding loading), with N=4,850,080,
rank=21, 149,920 singletons dropped and absorbed DoF=1,094,160. Sample/rank,
coefficients, residuals, RSS and covariance conditional on the reported DoF
passed the existing independent reference. Maximum scaled coefficient error
was 1.32e-8 and relative residual L2 error was 1.68e-10. Peak RSS was 4.088 GiB.
Both runs used the same source and numerical settings; this confirms that
relaxed admission removes the resource blocker, not that native convergence
is validated. The local candidate was then relabelled 0.9.1 to preserve the
immutable 0.9.0 public-contract snapshot; no solver code changed after timing.
Machine-readable results and resource counters are in
`benchmarks/hdfe/within_native_macos_091_5m_20261026.json`.

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
