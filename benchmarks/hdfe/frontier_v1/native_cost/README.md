# Native cost diagnostics: pinned research build

**Recorded results:** [2026-10-29 internal cost diagnosis](RESULTS_2026-10-29.md).
The main study contains 144 passed projections; the separate one-trial budget
probe adds two. It directly observes the joint-factor route and its fill-budget
boundary. Complete raw observations are archived as lossless JSON gzip files.

This extension of the FE frontier experiment observes the actual native work
behind an iteration count. It patches the **same upstream revisions** used by
the v0.1 measurements, builds separately named `+cost1` wheels, and runs them in
a separate environment. The patches change instrumentation and diagnostic
output; they do not change the solver tolerances, numerical algorithms, default
preconditioner policies, or production EconHDFE routing.

The archived v0.1 wheels and results are unchanged. Their inferred fereg route
must be distinguished from a route directly observed in a new instrumented
build. A new build is identified by its patches, resolved lockfiles and wheel
hashes, not just the original upstream revision string.

## Reproduce

Use Python 3.12, Rust **1.99.1**, maturin **1.15.0**, NumPy **2.3.5**, SciPy
**1.17.0**, and pytest **9.2.0** for the recorded environment. The Rust compiler
and maturin versions match the metadata recovered from the original wheels;
this does not imply byte-identical binaries on another operating system.

```bash
python3.12 -m venv ../fe-native-cost-env
../fe-native-cost-env/bin/python -m pip install \
  numpy==2.3.5 scipy==1.17.0 pytest==9.2.0 maturin==1.15.0
# Activate the pinned Rust toolchain before building.
../fe-native-cost-env/bin/python benchmarks/hdfe/frontier_v1/native_cost/prepare.py \
  --work-dir ../fe-native-cost-build --archives ../fe-native-cost-archives --build
../fe-native-cost-env/bin/python -m pip install --no-deps \
  ../fe-native-cost-build/wheelhouse/*.whl
ECONHDFE_REQUIRE_NATIVE_DIAGNOSTICS=1 ../fe-native-cost-env/bin/python -m pytest \
  benchmarks/hdfe/frontier_v1/tests -q
../fe-native-cost-env/bin/python benchmarks/hdfe/frontier_v1/native_cost/study.py \
  --families rigidity_grid --sizes 32 64 128 256 \
  --backends within native_auto fereg --diagnostics on off \
  --calls cold warm --replicates 3 --threads 2 \
  --output ../fe-native-cost-results/rigidity.json
../fe-native-cost-env/bin/python benchmarks/hdfe/frontier_v1/native_cost/study.py \
  --families rigidity_grid --sizes 320 --backends fereg --diagnostics on \
  --calls cold warm --replicates 1 --threads 2 \
  --output ../fe-native-cost-results/fereg_budget_probe.json
```

`prepare.py` requires a new empty work directory. It verifies the SHA256 of
three fixed source archives, applies each patch with `git apply --check`, copies
the committed lockfiles, and uses `maturin build --locked`. It writes source,
patch, dependency and wheel identities. It never installs wheels itself. The
`dual` wheel carries the wrapper, within and fereg license texts; upstream
source archives preserve their original notices. These are build and source
identities, not a claim about complete license applicability of every linked
file.

The [native CI workflow](../../../../.github/workflows/fe-native-cost.yml)
builds both wheels and requires every native backend. An unavailable old wheel
cannot become a passing diagnostic test. Ordinary benchmark CI can still use
the old wheel API and explicitly report diagnostics unavailable.

### Re-analyze the archived observations

Both summary and plotting tools read plain `.json` or lossless `.json.gz`.
The single-trial budget probe must retain `--expected-replicates 1`; its
timing variability cannot be inferred from the three-repeat main study.

```bash
python benchmarks/hdfe/frontier_v1/native_cost/summarize.py \
  benchmarks/hdfe/frontier_v1/native_cost/results/2026-10-29/rigidity_cost_study.json.gz \
  --expected-replicates 3 --output-json /tmp/rigidity_summary.json \
  --output-markdown /tmp/rigidity_summary.md
python benchmarks/hdfe/frontier_v1/native_cost/summarize.py \
  benchmarks/hdfe/frontier_v1/native_cost/results/2026-10-29/fereg_budget_probe.json.gz \
  --expected-replicates 1 --output-json /tmp/fereg_budget_summary.json \
  --output-markdown /tmp/fereg_budget_summary.md
# Plotting additionally requires matplotlib.
MPLCONFIGDIR=/tmp/fe-frontier-mpl python benchmarks/hdfe/frontier_v1/native_cost/plot.py \
  benchmarks/hdfe/frontier_v1/native_cost/results/2026-10-29/rigidity_cost_study.json.gz \
  --budget-probe benchmarks/hdfe/frontier_v1/native_cost/results/2026-10-29/fereg_budget_probe.json.gz \
  --output /tmp/fe_native_cost_scaling
```

## Data contract and accounting

The Python adapter exposes `native_cost_diagnostics` with `schema_version=1`,
`scope="call"`, `enabled`, and an `attempts` array. Unsupported and malformed
payloads are explicit. Legacy `actual_matvec_counts` remains null for native
backends: the new operator boundaries differ between the cores and are not
interchangeable with SciPy's rectangular operator counts.

| Item | Meaning |
|---|---|
| Outer `setup_s + solve_s` | Input checks/layout, fresh constructor, complete projection and output copy. Deferred native builds remain inside the solve. |
| `diagnostics_collection_s` | Optional getter access and Python diagnostic validation after the outer solve clock. Dual's existing JSON return decoding remains inside the call. |
| `constructor` | Once-per-object metadata, repeated for context on warm calls. Never add it again to each call's cost. |
| `core_before`, `core_after` | Cumulative per-solver snapshots. They are not per-call work. |
| Attempt `delta` | Differences of cumulative counters/timers for that invocation; constructor metadata is excluded. |
| `kernel_builds` | Dual adapter's builds during this call. A fresh object defers its first kernel build; a reused object need not rebuild. |
| `warm` | Second projection on the same solver, weights and RHS. It reuses factor state, with no coefficient warm-start seed or cached output. |
| Worker peak RSS | Whole process including fixture, oracle and both projection calls. It is not peak solver memory or separate cold/warm RSS. |

**Timings are nested or concurrent, not an additive cost partition.** A joint
factor's assembly/analysis/numeric timings are nested inside joint build,
inside PCG, inside residualization. Within's per-vector operator times can
overlap across parallel RHS and exceed the corresponding elapsed wall time
when summed. Always use the outer wall time for end-to-end comparisons.

The on/off experiment disables hot-loop counters and timers. Both variants
still execute the research wrapper, emit metadata and snapshots, and retain
coarse construction timings. Thus it estimates the incremental hot-loop
instrumentation cost, **not the entire cost of the patch relative to the old
wheel**. Repeated measurements alternate the on/off order; timing noise remains
visible in the raw observations.

### Within and native auto

The patch wraps the operators actually passed to LSMR/MLSMR and records
`design_apply`, `design_adjoint`, `preconditioner_apply`, and
`preconditioner_adjoint`. Each call is for one vector. Counts include the
iterative algorithm's residual checks and restarts; they exclude reconstruction
and the wrapper's separate group-mean certificate. They are summed across RHS
and corrective projections with atomics owned by that solver, never global
process counters. Failed entries retain their count and elapsed time.

The snapshot also records constructor and initial preconditioner time, deferred
build calls/time, actual preconditioner variant, escalation state, domain
assembly time, local-factor construction wall time, and a histogram of local
factor dimensions and cross-tab nonzeros. The histogram does not claim that
cross-tab nonzeros equal factor nonzeros. Approximate-Cholesky factor storage,
its symbolic/numeric split, and instrumented flop counts remain unavailable.

Native auto additionally records wrapper projections, certificates and cyclic
polish. Its small-Schur path is named explicitly; it has no LSMR operator work.
Diagonal fallback solvers have separate snapshots so their work is retained
even when the main solver's cumulative state is unchanged. The fallback's
constructor appears as constructor metadata in its own snapshot.

### Fereg

`matvec_calls` and `precond_calls` count actual `CgSystem` entry calls on the
full row-major RHS batch. `*_rhs_columns` sums the allocated batch width,
including frozen columns. Neither value is derived from `iterations`. The
wrapper's repeated `[i, i]` still denotes one shared lockstep counter.

The diagnostics distinguish PCG calls, probes, actual upgrades, closed/direct
factor routes, cached factors and coefficient warm seeds. `exact` is the
kernel's direct-factor route flag; it is not proof of zero numerical error.
Every measured residual must pass the independent forward-error oracle.

| Factor diagnostic | Precise interpretation |
|---|---|
| `joint_aggregation_*`, `joint_pattern_*` | Cell aggregation and candidate sparsity-pattern work, retained even for rejected candidates. |
| `joint_assembly_*` | Explicit candidate matrix assembly. |
| `joint_analyze_*` | Lower-triangle extraction, ordering/AMD and symbolic analysis together; not pure symbolic time. |
| `joint_numeric_*` | Initial numerical factorization. |
| `joint_refactor_*` | Numerical refresh including scaling and pattern compatibility checks. |
| `joint_factor_entries` | Actual allocated factor-value length (`len_val`), potentially including supernodal padding; not exact numerical `nnz(L)`. |
| `joint_matrix_nnz` | Actual full-CSR nonzeros in the last assembled candidate. A rejected candidate can retain this value without a live factor. |
| `joint_symbolic_flops` | Symbolic work estimate, not measured floating-point operations. |

Per-block and small-factor build costs are retained separately. Weight-update
and joint-refresh counters are implemented and have targeted Rust tests; the
main cold/warm experiment keeps weights fixed and does not claim to benchmark
IRLS or a full estimator.

## Validation and interpretation

The tests compare weighted projections against independent dense SVD, permute
observations, FE dimensions and RHS columns, compare diagnostics on/off, check
counter deltas across reuse, and recover after a failed call. Fereg's targeted
Rust tests distinguish an exact one-step solve with zero matvecs from an
adaptive restart requiring an extra residual matvec, and retain failure costs.
No speed threshold is used as a correctness test.

These observations can establish which algorithms ran and where time/storage
went on the tested families. They do not turn a structured-family measurement
into a universal complexity bound, or invalidate a near-quadratic upper bound.
See the [dated evidence report](RESULTS_2026-10-29.md) for measurements and their
reproduction identities.
