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
4. **Whole-call scheduling.** `ExecutionConfig(threads="auto", memory_budget_mb="auto")` admits the call using the current effective CPU capacity and available host/container memory less a 12% total-memory reserve. Its resolved limits govern BLAS, Numba, ingestion, design allocation, FE workspace and resampling. The root CPU ceiling remains available to independent tasks; automatic inner HDFE kernels select a measured saturation point within that ceiling.

Explicit integer thread counts always override calibration. Calibration controls only execution resources; it cannot change the model, sample, FE, instruments, weights, clustering, covariance formula, or convergence criterion.

For a monitored large run, opt in with
`ExecutionConfig(memory_policy="relaxed", memory_budget_mb=8192)`.
The default `"conservative"` policy remains unchanged. Relaxed mode reserves
the smaller of 3% of effective total memory and 512 MiB, instead of 12%.
An explicit budget remains the requested whole-call incremental workspace
ceiling rather than being permanently clipped to initial host headroom.
Every allocation/reservation still obeys live host/container headroom, RSS
growth, outstanding reservations, and worker slices. Thus newly reclaimed host
memory can become usable during a call, but an explicit large budget cannot
override live memory pressure. With `memory_budget_mb="auto"`, the initial
safe headroom still determines the budget. Nested calls and workers inherit
the root policy; it is restored at scope exit.

This option does not limit total process RSS, guarantee completion, or change
statistical semantics. Lower system reserve leaves less room for other apps;
use external memory monitoring for large workloads. Concurrent reservation
accounting remains conservative because RSS cannot be attributed exactly to
individual workers.

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

## Automatic inner thread calibration

Automatic policy provenance survives numeric admission, nested nonlinear calls and worker slices. Small HDFE problems use one thread; large problems use the smallest measured candidate within 5% of the best synthetic memory-bandwidth timing. Explicit integer requests remain explicit and are capped by effective runtime capacity. BLAS retains its enclosing allocation; independent outer tasks do not inherit the inner kernel's saturation limit.

Cold calibration occurs at an unreserved planning boundary, before constructor or block-output reservations. Probe arrays must fit both the shared budget and live headroom; they do not consume user data. During reserved work or parallel worker execution, only an existing calibration is used. Otherwise the kernel temporarily uses one thread and retries calibration at a later safe boundary. A temporary fallback is never saved as a successful measurement. The runtime fingerprint includes affinity and the Numba threading configuration. Kernel/JIT overhead remains outside exact array accounting.

Thread scaling for HDFE is usually memory-bandwidth limited. The maximum logical CPU count is therefore not a reliable default. At the same time, microbenchmarks are noisy and cannot guarantee the exact optimum for every topology/operator mix. The auto policy is consequently designed to find a **stable near-saturation region**, not to promise the globally fastest thread count for every regression.

The canonical validation is an alternating multi-round comparison of the calibrated `auto` choice against explicit candidate thread counts on an unchanged synthetic HDFE workload. A machine-specific anomaly is reportable through the planner developer-report mechanism rather than hidden by a benchmark-specific hard-coded threshold.

Calibration state is disposable. Removing the local planner cache can only cause a future recalibration; it cannot change econometric semantics.

## Reclaimable in-memory caches

Linear sessions and reusable PPML/IVPPML execution contexts bound disposable cache storage to at most one quarter of the computation envelope, tightened by current safe headroom. Arrays sharing the same backing storage count once, and caller-owned input storage is excluded. Least-recently-used entries are removed first; a linear entry actively needed by the current fit is protected until the call boundary. A batch fit reconstructs an evicted prewarmed entry when needed.

Reclamation runs before root admission, at cache/workspace preparation, and at call exit. `cache="on"` still obeys the safety budget. `cache="off"` drops retained references; persistent on-disk stores and source data are not deleted. Linear `cache_info()` reports retained array bytes, the current cache allowance and eviction counters. Those bytes are estimates of cache-held backing arrays, not proof of an RSS decrease: active results and the allocator can keep storage alive.

The policy operates at safe boundaries rather than interrupting a running numerical kernel. It does not continuously tune OS scheduling, coordinate independent processes, or promise a globally optimal thread count.

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

`planner.resources` is the single runtime-resource contract; `econhdfe.compute` re-exports `runtime_resources` and `configure_numba_runtime` from it.

## Non-goals

- no ML planner;
- no remote telemetry or automatic report upload;
- no persistent database of user workload performance;
- no estimator-specific planner logic;
- no dynamic changes to the econometric specification;
- no default change to dense/block representation thresholds based only on byte savings.

## Resource policy propagation

An explicit caller memory budget is inherited by group-only fits, nonlinear
rank checks after separation, block-local projectors and internal standard-FE
absorbers. Fused RHS scratch inherits the projection budget by default; an
explicit `fused_rhs_memory_mb` may tighten it, but cannot enlarge it. The former
independent 128 MiB scratch default and 64 MiB group-incidence floor do not
supersede the caller's policy. The public estimator/Session default is now automatic safe headroom when no
budget is supplied. Standalone low-level helpers retain their compatibility defaults. This is still a workspace/representation policy, not an RSS cap;
available runtime headroom may further tighten execution.

Block-local linear projectors inherit `HDFEConfig.max_iter`. Explicit integer
threads are bounded by current effective CPU capacity as well as the initialized
Numba pool. Session `cache="off"` allows temporary reuse inside a single public
call, discards it on return or failure, and bypasses an enabled persistent store.


## Whole-call resource admission

`planner.scheduler` owns the process-local execution scope. A root call resolves
policy after admission, so calls waiting for another estimate cannot reuse stale
headroom. Unrelated root calls in the same process are queued serially: this
avoids simultaneously assigning the entire host budget and prevents races in
process-global BLAS settings. Parallelism within a call uses worker slices.

The memory envelope is the lesser of an explicit user cap (when present) and
current safe headroom. Automatic policy uses all safe headroom; it does not
preallocate that amount. Subsequent planning subtracts positive process RSS
growth since admission and rechecks live host/container headroom. Resident input
and pre-existing Session cache are already reflected in host availability.
Nested estimator stages inherit the admitted envelope, not a new host-sized
budget. Resampling divides CPU and memory between workers; their inner calls
cannot expand the slice. BLAS limits are restored on success and failure.

Known dense/block designs and in-memory data conversions are checked before
large allocations. Weighted nonlinear projection uses the same byte inventory
and pooled-column planner as linear absorption. Session policy is resolved on
every call; resource changes rebuild execution objects without invalidating
mathematically reusable within values or persistent results. Cache off still
removes cross-call estimation caches.

The policy is **cooperative**, not an OS RSS or GPU-memory limit. NumPy/SciPy/JIT
can allocate unmodeled temporary memory, other processes can consume headroom
between checks, and file sources lacking row-count/schema metadata still need a
single materialization before their true size is known. It does not coordinate
unrelated processes. These limits preclude a promise that the host can never run
out of memory. No observation, regressor, FE or convergence rule is changed to
make a computation fit: unsafe known allocations fail explicitly.

Resource metadata is in linear results' `absorb_info["resource_schedule"]` and
nonlinear results' `diagnostics["resource_schedule"]`. It records admitted CPU,
memory envelope, reserve, remaining budget and worker share. This is separate
from econometric diagnostics. The older projection-only memory field is not a
claim of whole-process enforcement.

## Atomic workspace reservations

Known projection construction/output/scratch allocations use
`reserve_memory(bytes, label=...)` before allocation. A root-shared ledger and
lock make admission atomic across workers; claims are released in `finally`.
Nested claims are additive unless `reuse_parent=True` explicitly states that a
substep is covered by the enclosing peak estimate. Worker scopes do not inherit
the dispatcher's ownership of a claim. Planning existing claimed work may use
its own envelope; planning new scratch uses the unreserved balance. This lets
weighted projection reduce its pool rather than reject a feasible narrower run.

Live RSS growth and outstanding claims from other work are charged
conservatively because RSS cannot identify which claim has already materialized.
This can reject work that would fit under exact accounting, but never treats
unmaterialized reservations as measured allocations. Reservations are
cooperative claims, not allocated arrays or OS enforcement.

Wild-bootstrap worker estimates account for sample/design dimensions, FE
workspace, multiplier matrices, batch width, studentization and retained output.
Concurrency is bounded by these estimates as well as CPU capacity. Explicit
batch size and SeedSequence grouping are preserved; a single batch that cannot
fit raises `resource.memory` before expensive setup. Generic replicate workers
may supply `estimated_worker_bytes`; without an estimate, the 32 MiB worker
floor remains a heuristic, not a proof of peak usage. Resource failures always
propagate and are not counted as skippable statistical failures. Shared FE
resident state/output can be conservatively charged per worker, so this policy
may admit fewer workers than an exact ownership model would.
