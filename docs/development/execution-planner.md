# Unified execution planner

The execution planner is an estimator-agnostic layer between **econometric correctness** and **physical execution**. It does not infer coefficients, samples, fixed effects, instruments, separation, or covariance formulas.

Its invariant is:

```text
exactness / eligibility certificate
              ↓
resource and cost comparison
              ↓
real-runtime calibration where appropriate
              ↓
physical execution choice
```

A candidate that lacks an exact certificate is never allowed to win because it is faster or smaller.

## Current scope

The planner now unifies four execution responsibilities that were previously distributed across modules:

1. **Memory envelope.** Data ingestion and HDFE workspaces use the same user-budget + runtime-headroom policy.
2. **Representation.** Dense and block-dense design paths express exact eligibility through the shared certificate contract. The established conservative one-shot/repeated-pass production policy remains in place because storage-byte savings alone do not predict wall time reliably.
3. **Parallel budget.** Outer/inner workers share one non-oversubscribed CPU budget.
4. **Real-runtime automatic thread calibration.** On the first sufficiently large HDFE workload (currently 100k+ observations), `threads="auto"` runs a short deterministic synthetic memory-bound Numba probe at candidate thread counts, selects the smallest candidate within 5% of the measured best median time, and caches that decision by an anonymous runtime fingerprint. Smaller HDFE workloads stay single-threaded so calibration and parallel overhead cannot dominate the fit.

Explicit integer thread counts always override calibration. Calibration controls only execution resources; it cannot change the model, sample, FE, instruments, weights, clustering, covariance formula, or convergence criterion.

HDFE hot kernels now inherit the resolved outer thread context; they do not
perform another calibration lookup per FE sweep. Two-way standard factories
also propagate the requested thread policy.

Runtime headroom combines psutil host availability (including macOS) with
Linux cgroup limits when readable. An unavailable host counter leaves cgroup
information intact. Memory values describe the effective planning envelope,
not process RSS. Pooled HDFE charges known resident solver state plus
conservative per-column scratch to the user budget, while already-resident
state is not charged twice against live headroom. Explicit pools are validated;
automatic planning fails if even one column cannot fit. See
[solver execution and budget details](../technical/hdfe/solver/design.md).

## Why calibration is deliberately bounded

Thread scaling for HDFE is usually memory-bandwidth limited. The maximum logical CPU count is therefore not a reliable default. At the same time, microbenchmarks are noisy and cannot guarantee the exact optimum for every topology/operator mix. The auto policy is consequently designed to find a **stable near-saturation region**, not to promise the globally fastest thread count for every regression.

The canonical validation is an alternating multi-round comparison of the calibrated `auto` choice against explicit candidate thread counts on an unchanged synthetic HDFE workload. A machine-specific anomaly is reportable through the planner developer-report mechanism rather than hidden by a benchmark-specific hard-coded threshold.

Calibration state is disposable. Removing the local planner cache can only cause a future recalibration; it cannot change econometric semantics.

## Planner developer feedback

`econhdfe.planner.write_planner_developer_report(...)` and `skills/econhdfe/scripts/planner_report.py` generate a privacy-minimized Markdown/JSON report containing only:

- aggregate runtime CPU/memory capability;
- anonymous calibration ID and candidate timing curve;
- selected thread budget and optional execution-plan metadata;
- anonymous workload counts and aggregate timing/iteration counters when a result is supplied.

The report contains no raw/sampled/synthetic observations, variable identifiers, confidential paths, exact commands/scripts, hostnames, raw logs, full tracebacks, or source data. Nothing is transmitted automatically. Real-data performance benchmarking remains a separate explicit-consent workflow.

The canonical fill-in template is `docs/development/PLANNER_REPORT_TEMPLATE.md` and is mirrored in the standalone skill.

## Why the representation byte-cost model remains advisory

A compact representation reduces memory traffic, but dense BLAS may still win on a one-shot contiguous operation because of vectorization and dispatch overhead. Development microbenchmarks show exactly this at moderate structural sparsity. Therefore the generic byte-traffic model remains an explainable comparison primitive; it does **not** replace the conservative production representation policy solely from structural-zero ratios.

## Dependency boundary

```text
data ───────┐
compute ────┼──► planner
hdfe ───────┘

planner ─X─► data / compute / hdfe / iv / models
```

`planner.calibration` benchmarks only package-generated synthetic memory traffic and imports no HDFE/model implementation. HDFE consumes the selected thread budget. `planner.report` consumes only contracts/aggregate result metadata and likewise does not define estimation logic.

`planner.resources` remains the canonical runtime-resource contract. `compute.runtime` is a compatibility surface that re-exports it.

## Non-goals

- no ML planner;
- no remote telemetry or automatic report upload;
- no persistent database of user workload performance;
- no estimator-specific planner logic;
- no dynamic changes to the econometric specification;
- no default change to dense/block representation thresholds based only on byte savings.
