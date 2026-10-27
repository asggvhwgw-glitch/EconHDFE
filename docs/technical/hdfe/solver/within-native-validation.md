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

## Budgeted certification regressions and a new large-sample corpus

### Initial candidate record (`999900d`)

The initial follow-up keeps `tol=1e-12`, the original-row certificate and the shared
`max_iter=16000` limit. It replaces the fixed two-replacement stop with measured
continuation, then permits an admitted original-row diagonal-preconditioned
fallback. The pinned upstream revision and estimator defaults are unchanged.
The detailed rules and known-buffer accounting are in `within-native.md`.

Two reproducible 4,000-row adversarial cases exercise distinct recovery paths.
Both use strongly unequal positive weights; they are accuracy regressions,
not representative workload timings.

| Case | Baseline | Updated native | Independent reference |
|---|---|---|---|
| Progressing residual replacement | Rejected at metric 3.16e-12 after 236 iterations | Accepted at 3.77e-13 after 240 iterations; no fallback | Reference-coded dense weighted least squares retains all 583 columns; maximum residual difference 1.19e-9 |
| Stalled adaptive preconditioner | Rejected at metric 2.40e-11 after 85 iterations | Accepted at 3.82e-13 after 302 total iterations; one diagonal fallback | Exactly representable paired residuals; maximum forward error 5.06e-11 |

The baseline wheel is built from unchanged native code at main `15eda97`.
The independent original-row mean recomputations also pass for the updated
outputs. Every reported norm is retained in the evidence: passing the required
maximum-group-mean gate does not imply that every other error norm improves.
Regression tests also cover RHS/weight units, row/FE/column permutations,
iteration exhaustion, declined fallback under tight workspace, successful
primary-only operation, other buffered RHS columns, and retained Schur storage.

The new large corpus has one million and five million rows, six RHS columns,
three categorical terms with weakly linked communities, unequal dyadic weights,
nearly collinear controls and a nearly absorbed excluded instrument. Equal-weight
observation pairs give an exact projection of the stored float inputs. The full-IV
reference independently uses QR/SVD on the known within data, cluster-score
aggregation and an active-graph DoF calculation. Sample/rank, coefficients,
covariance, standard errors, residuals, RSS and original-row stationarity pass
their fixed gates for both baseline and updated native code.

| Updated native sample | Maximum projection relative error | Scaled coefficient error | Scaled covariance error | Relative residual error |
|---|---:|---:|---:|---:|
| 1,000,000 rows | 5.74e-11 | 5.81e-11 | 1.08e-12 | 1.30e-11 |
| 5,000,000 rows | 3.63e-11 | 1.02e-11 | 1.55e-13 | 6.65e-12 |

Maximum independently recomputed projection stationarity is 2.62e-13 and
1.30e-13 respectively. None of the large runs needs a certification fallback.
Single-run full-IV fit times are 2.705 / 3.780 seconds for baseline / updated
native at 1M, and 15.597 / 16.275 seconds at 5M. They are validation timings,
with separate checkout/JIT caches, not a performance or timing-regression
certificate. Updated whole-process peak RSS is 994 MiB and 3549 MiB, including
inputs, generation, projection, independent references and full IV; requested
workspace is 2048 MiB with relaxed memory policy and is not an RSS limit.

**This is not a replay of the historical 5M composite IV failure above.** Its
original generator/input was unavailable; the historical 21-parameter,
singleton-pruned case remains unvalidated by this follow-up. The new corpus
provides large-scale independent accuracy coverage without replacing that
historical evidence or establishing a native 5M speedup for that specification.

Reproduce the new corpus with `native/within-accel/validate_native_dyadic_iv.py`
using `--rows 1000000` or `--rows 5000000`, `--mode both --threads 2`, initial
single-thread BLAS and a matching process/proc namespace. Its exact executed
source, runtime hashes, environment, acceptance limits and measurements are
recorded in `benchmarks/hdfe/within_native_certification_20261027.json`.
Local Linux x86_64 / Python 3.12 validation passed **6 Rust tests, 51 native and
public-integration tests, and all 468 tests in the full base suite**. The
version/compatibility gate remains 0.9.1 and the regenerated architecture map
passes its check. These are local checks, not a package publication or a claim
of validation on other operating systems.

### CPU-rounding follow-up

The initial candidate's Linux CI run passed 50 native tests but failed the
progressing-continuation regression. A controlled local reproduction with both
the exact CI wheel and the local wheel isolated a one-ULP input sensitivity:
disabling NumPy's AVX512 `exp` dispatch changed 188 of 4,000 positive weights,
with maximum relative difference 2.19e-16. Codes and outcomes were unchanged.
Changing NumPy 2.3.5 to 2.5.4 alone on the same CPU did not change those inputs
or the solver outcome. Actual CI input hashes were not logged; this establishes
a reproducible dispatch mechanism, not the byte identity of the CI inputs.

The final runtime allows one compensated original-row weighted cyclic sweep
after a successful upstream solve stalls above the certificate, provided an
iteration remains. The sweep counts against the same total budget and uses the
existing primary scratch reservation. Acceptance still requires the original
`max(tol, 64*eps)` gate; the best residual is retained if the sweep regresses.
The diagonal fallback remains available only after checked workspace admission.

| Input / case | Initial candidate | Final candidate |
|---|---|---|
| NumPy AVX512 exponential weights | Pass, 240 iterations, metric 3.77e-13 | Pass, 240 iterations, metric 3.77e-13 |
| Scalar / AVX512-disabled exponential weights | Fail, 16,000 iterations, metric 1.72e-12, one fallback | Pass, 245 iterations, metric 1.09e-15, no fallback |
| Exact dyadic weak direction, one RHS | Pass, 302 iterations, metric 3.82e-13, one fallback | Pass, 90 iterations, metric 5.93e-15, no fallback |

Permanent regression coverage includes both NumPy `exp` and scalar `math.exp`
on the same seed, geometry and outcome. Both retain all 583 reference-coded
dummy columns and the existing dense-reference `atol=2e-9`, `rtol=2e-8` limits.
An extended-precision original-row group-mean check independently passes the
same 1e-12 gate. No reference direction, numerical threshold or original-main
assertion was removed.

The 600,000-byte primary-only, 100-iteration and two-RHS 980,000-byte dyadic
cases now certify through the sweep. Their newly added route/expected-failure
assertions were updated to check this successful result at the same limits,
including the independent projection oracle. `max_iter=1` still fails honestly
without extra work. Two Rust tests guard the actual fallback constructor:
they check the exact byte boundary, retained Schur/other-RHS charges, overflow,
exhausted iterations, and delivery of only the remaining iteration allowance.

The final local runtime passes **8 Rust tests, 53 native/integration tests and
468 full-suite tests**. The unchanged 8,192-, 1M- and 5M-row corpus also passes
all projection and full-IV gates with the final binary. The maximum projection
relative errors at 1M and 5M are 5.74e-11 and 3.63e-11; neither run needs a
fallback. Frozen pre-fix diagnosis, final source/binary hashes,
both CPU-dispatch reproductions, boundary measurements and final large-corpus
reruns are recorded separately in
`benchmarks/hdfe/within_native_certification_cpu_rounding_20261027.json`.
The initial record above is preserved unchanged. Neither record is a replay
of the unavailable historical 5M input or a native performance claim.

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
