# Configuration and execution policy

## Contents

- Public configuration objects
- Solver and degrees-of-freedom policy
- Runtime-aware parameter tuning
- Repeated-specification caching
- Weights, covariance, clustering and failure handling


Use this guide when choosing public configuration objects, inference settings, resource limits or numerical execution options.

## Prefer public configuration objects

For advanced control prefer:

```python
from econhdfe import HDFEConfig, InferenceConfig, ExecutionConfig

hdfe = HDFEConfig(
    solver="auto",
    tolerance=1e-8,
    max_iter=16_000,
    dof_method="pairwise",
    canonicalize=True,
)

inference = InferenceConfig(
    vce=None,
    confidence_level=0.95,
    diagnostics="off",   # off | publication | full
    iv_advanced_statistics=True,  # False skips advanced linear-IV calculations
)

execution = ExecutionConfig(
    threads="auto",
    memory_budget_mb="auto",
    profile="off",       # off | summary | full
    cache="auto",        # auto | on | off
    cache_validation="signature",
)
```

These are strategy-level contracts. Avoid depending on private solver constants or internal thresholds.

## Linear-IV diagnostic execution and staged results

`InferenceConfig(iv_advanced_statistics=False)` retains ordinary first-stage F tests
(including the chosen robust/cluster covariance), partial R², Kleibergen–Paap and
overidentification tests. It skips the actual calculations of Shea partial R²,
AP/SW, Cragg–Donald and Stock–Yogo. The default is `True`. Coefficients, covariance,
sample selection and identification/rank checks are unchanged. `diagnostics`
continues to control display only; `"off"` does not disable computation.

Pass `on_stage=callback` to `ivhdfe` or `IVHDFESession` to receive synchronous
independent snapshots: `estimation`, `first_stage`, `diagnostics`, then `complete`.
The first snapshot has coefficients/covariance/SEs in original units. Only
`complete` confirms that the full fit finished; a later failure raises normally.
Snapshots exclude observation-level fitted values and residuals. Callback time
counts toward fitting time, so keep the callback lightweight. A persistent
result-cache hit emits only `complete` with `cached=True`.

See [the full contract and example](../../../docs/development/iv-staged-diagnostics.md)
for event fields, omitted-statistic status and repeated-specification behavior.

## Solver policy

Start with `method="auto"` / `HDFEConfig(solver="auto")` unless a replication or benchmark requires a fixed solver.

- Exactly two effective pure categorical intercept FEs may use the specialized two-way path after canonicalization.
- General ordinary HDFE starts from the package's MAP/CG planning rules.
- `lsmr` is a useful fallback for difficult/poorly connected systems and specialized group+individual structures.
- Do not weaken tolerance solely to make a failing specification return a number.

For heterogeneous slopes use `FixedEffect(..., slopes=...)` rather than materializing large interaction-dummy matrices.

## Direct absorber advanced solver controls

Most empirical users should keep estimator-level `method="auto"` and public strategy configs. Advanced users who construct `HDFEAbsorber` directly can additionally select:

```python
from econhdfe import HDFEAbsorber

a = HDFEAbsorber(
    groups,
    acceleration="auto",        # two-sweep contraction probe, then MAP or CG
    core_reduction="auto",      # auto | on | off
    core_min_peel_fraction=0.05,
    fused_rhs_memory_mb=128,
)
```

The direct absorber default remains `acceleration="cg"`; opting into `auto` is an execution choice. Numerical core reduction is eligible only for NumPy, pure-intercept, 3+ categorical FE systems with strictly positive weights. Positive weight changes can reuse the core topology; any zero weight disables the core fast path because the effective numerical topology changes. `auto_plain_limit` and `auto_polish_limit` are advanced numerical limits and should not be tuned in ordinary empirical work without benchmark/accuracy evidence.

Do not interpret solver-core reduction as dropping requested fixed effects for inference. Requested FE topology remains the object used for DoF/nesting/reporting; canonicalization and numerical core reduction are execution transformations only.

## Degrees of freedom

- `dof_method="pairwise"` is the default compatibility-oriented choice.
- `dof_method="exact"` requests exact structural absorbed DoF for supported multiway intercept-only categorical FE designs.
- Exact DoF changes inference accounting, not the coefficient projection equation. Hard exact-rank cores can be materially more expensive.

## Runtime-aware parameter tuning

Before overriding automatic parameters, inspect the actual host and actual design. Use effective CPU affinity/cgroup limits and available memory headroom rather than blindly using `os.cpu_count()` or total installed RAM.

Default policy:

```python
res = olshdfe(
    ...,
    method="auto",
    projection_backend="auto",
    absorb_threads="auto",
    pool_size="auto",
    memory_budget_mb="auto",
)
```

Tune only after measurement:

1. `absorb_threads`: do not exceed effective CPU capacity. Memory-bandwidth-bound FE projection often stops scaling before the core count.
2. `memory_budget_mb`: base it on free headroom and leave a safety reserve. It is a scratch-workspace budget, not permission to exhaust RAM.
3. `pool_size`: leave `auto` unless profiling shows repeated FE traversals dominate and extra workspace is safe.
4. `projection_backend`: `indexed` can be strong for large reusable categorical groups when index memory is affordable; `fused` is the lower-index-memory fallback.
5. GPU: use only after confirming compatible CuPy/CUDA and measuring transfer + solve cost on the target device.

Use `ExecutionConfig(profile="summary")` or the corresponding public profile controls when a performance diagnosis is requested.

## Repeated specifications and cache policy

For repeated OLS/IV tables on the same DataFrame use `OLSHDFESession` / `IVHDFESession` so FE encoding and within-transformed columns can be reused.

Keep `cache_validation="signature"` unless the caller guarantees the relevant data are immutable and measured profiling justifies disabling validation. Treat cache correctness as part of the econometric result, not merely an optimization detail.

In-memory Session and reusable nonlinear-model caches are bounded by the execution budget and live headroom. Cold entries can be evicted and rebuilt; `cache="on"` does not bypass the limit. `Session.cache_info()` exposes retained array bytes, the current cache allowance and eviction counts. Eviction removes reusable references, not user data or the persistent on-disk store, and does not guarantee an immediate RSS decrease.

## Weights

Use an explicit `weight_type` when Stata-compatible semantics matter:

```python
weight_type="fweight"  # frequency weights
weight_type="aweight"  # analytic weights
weight_type="pweight"  # probability weights; robust inference enforced where required
weight_type="generic"  # generic WLS behavior
```

Never silently reinterpret one weight type as another. Some estimator/VCE combinations are intentionally rejected when the package cannot support the requested semantics faithfully.

## Covariance and clustering

Use the public `vce`, `cluster`, `time`, `panel`, `bandwidth` and `kernel` controls supported by the chosen estimator. One- and multi-way clustering are distinct inference specifications; do not change them for speed.

Standard one-/multi-way CRV1 is part of estimator covariance. `cluster_diagnostics()` and `wild_cluster_test_ols()` are post-estimation tools: the latter is currently one-way OLS only and must not be substituted for a substantively required multi-way cluster specification.

Diagnostics policy:

- `off`: normal paper-facing path;
- `publication`: reportable diagnostic subset;
- `full`: solver/identification detail for validation or debugging.

## Failure handling

If MAP/iterative absorption fails:

1. inspect FE connectivity/canonicalization and omissions;
2. confirm the specification is identified;
3. increase iteration budget if convergence is credible but incomplete;
4. try the documented alternative solver path;
5. only then consider tolerance changes, with a numerical justification.

OOM: reduce pool/memory pressure or choose a lower-index-memory projection path. CPU scaling stalls: reduce thread count and benchmark. GPU import/device failure: verify the installed CuPy build and actual CUDA runtime/device.
## PPML / IV-PPML execution resources

For v0.4.5+, `ExecutionConfig` is enforced by PPML and IV-PPML weighted-HDFE projectors rather than merely validated. For a controlled run:

```python
execution = ExecutionConfig(threads=4, memory_budget_mb=2048, cache="auto")
r = ppmlhdfe(..., execution_config=execution)
print(r.diagnostics["execution"])
print(r.diagnostics["projection_resources"])
print(r.diagnostics["separation_seconds"])
print(r.diagnostics["separation_solvers"])
```

Use `projection_resources["requested_threads"]` and `projection_resources["actual_threads"]` to audit the HDFE projector actually used by the final estimation sample. `memory_budget_mb` controls projector/workspace planning and is not a whole-process RSS cap.

`PPMLConfig(engine="optimized")` now routes simplex separation through the optimized FE plan and ReLU through an optimized MAP/HDFE projector. `engine="replica"` retains the replica-oriented paths. Do not disable `simplex` or `relu` solely because a previous dataset/run found zero additional separated observations: that is an empirical observation, not a general safety certificate. Use FE-only separation only as an explicitly labelled sensitivity/performance experiment.



## Automatic thread calibration

Public estimator and Session calls now use one shared resource scope. `ExecutionConfig(threads="auto", memory_budget_mb="auto")` uses effective CPU capacity and available host/container memory after a 12% total-memory reserve. An explicit memory or thread value caps that scope. Nested stages inherit it; resampling workers divide it. Independent root calls in the same process queue to avoid overlapping full-host budgets and BLAS-setting races.

Automatic inner HDFE kernels now retain that policy through the managed scope: small problems use one thread, while large problems select the smallest measured thread count within 5% of the best synthetic bandwidth timing. The root CPU capacity stays available to independent outer tasks. Calibration runs before workspace reservations and fits available memory; reserved/parallel cold paths temporarily use one thread until a safe calibration boundary. Explicit integer threads bypass calibration. The shared scope is cooperative planning, not an OS RSS or GPU-memory limit; unknown-size file reads and third-party temporaries cannot all be bounded in advance. Inspect `resource_schedule` in linear `absorb_info` or nonlinear `diagnostics` for the admitted envelope.

## Singleton policy and typed Poisson weights

The default `singleton_mode="stata_compat"` is a source-audited sample profile,
not a blanket promise of Stata parity. For frequency-expansion-invariant rules,
construct `HDFEConfig(singleton_mode="unified")` for linear fits/sessions or
`PPMLConfig(singleton_mode="unified")` / `IVPPMLConfig(singleton_mode="unified")`.
Use explicit `weight_type="fw"` or `"pw"` for Poisson weights; do not infer weight
type from values. All typed weights must be strictly positive; zero weights
require explicit prefiltering. Plain Poisson does not support `aw`.

Unified Poisson keeps ordinary singleton and all-zero FE counts separate;
`drop_singletons=False` does not disable `fe` separation. Unified IV-PPML defaults
to `separation=("fe",)` and rejects PPML simplex/ReLU/mu choices; they are not
IV existence certificates. Construct a fresh config when changing modes.
Grouped/individual FE unified mode is explicitly unsupported. This version
retains all statistical rows in the work sample; no extra computational
elimination should be inferred from the policy name. Compare sample, N, DF and
VCE, not only coefficients, when checking modes or frequency expansion.
