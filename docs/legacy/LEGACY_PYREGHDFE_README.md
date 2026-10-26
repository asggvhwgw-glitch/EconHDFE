> **Historical document:** the `pyreghdfe` compatibility package was removed in econhdfe 0.9.0. Import from `econhdfe` instead; see [docs/release/migration.md](../release/migration.md).

# pyreghdfe 0.9.0a1

`pyreghdfe` is a clean-room Python implementation of the econometric behavior
needed for high-dimensional fixed-effects OLS, linear IV, and PPML. It targets the
`reghdfe` / `ivreghdfe` / `ppmlhdfe` workflow while using Python-native numerical kernels,
reusable absorption, bounded-memory execution and parallel resampling rather
than line-by-line Mata translation.

## Highlights

- HDFE OLS, IV, and PPML without dense dummy matrices or `N x N` projection matrices.
- Integrated PPML IRLS with FE/simplex/ReLU separation and model/robust/multiway-cluster inference.
- One shared FE runtime for linear and nonlinear estimators: encoding, canonicalization, DoF, MAP/indexed projection, two-way Schur/PCG, runtime planning and VCE.
- Intercept FE plus heterogeneous slopes (`i.g#c.x` / `i.g##c.x`-style semantics).
- **Group-level outcomes with individual FEs** (`group()` / `individual()`-style data), including `aggregation="mean"` and `"sum"`.
- Matrix-free LSMR/LSQR incidence operators plus an automatic sparse-CSR fast path under a user memory budget.
- MAP, symmetric Kaczmarz + CG acceleration and matrix-free SciPy LSMR for ordinary HDFE.
- Optional CuPy GPU MAP backend for ordinary HDFE.
- OLS, 2SLS, LIML, Fuller/k-class and two-step GMM.
- IID, HC1, 1–10 way cluster, HAC/Newey-West and Driscoll–Kraay covariance.
- First-stage diagnostics, Cragg–Donald, Kleibergen–Paap rk LM/Wald,
  Sanderson–Windmeijer diagnostics and Stock–Yogo table lookup.
- Stata-style `fweight`, `aweight` and `pweight` semantics on the core paths.
- Optional FE recovery with an explicit minimum-norm normalization.
- Recursive singleton pruning, including the group–individual bipartite 1-core rule.
- Batched shared-memory parallel bootstrap with reusable FE encoding.
- Multi-core indexed FE projection with compact O(N+G) group indexes and Numba `prange`.
- Parallel low-memory MAP convergence scans and bootstrap-aware thread budgeting.
- `pool_size="auto"` and memory budgeting for million-/ten-million-row workloads.

## Basic usage

```python
from pyreghdfe import reghdfe, ivreghdfe, FixedEffect

ols = reghdfe(
    df,
    y="y",
    x=["x1", "x2"],
    absorb=["firm", "year"],
    cluster=["firm", "year"],
    vce="cluster",
)

iv = ivreghdfe(
    df,
    y="y",
    exog=["control"],
    endog=["price"],
    instruments=["z1", "z2"],
    absorb=["firm", "year"],
    vce="robust",
    estimator="liml",
)
```

## PPML-HDFE (0.9.0a1)

PPML is now part of the same package and shares the compiled FE topology and numerical backends:

```python
from pyreghdfe import ppmlhdfe, PPMLConfig

pp = ppmlhdfe(
    y_count,
    X,
    absorb=[firm, year],
    clusters=[firm, year],
    vce="cluster",
    config=PPMLConfig(engine="optimized"),
)
```

For repeated specifications on one DataFrame, compile the FE topology once:

```python
from pyreghdfe import PPMLHDFE

model = PPMLHDFE(df, absorb=["firm", "year"])
r1 = model.fit("trade", ["distance", "fta"], vce="robust")
r2 = model.fit("trade", ["distance", "fta", "tariff"], vce="robust")
```

`engine="replica"` keeps the conservative ppmlhdfe-style numerical path;
`engine="optimized"` additionally enables exact FE canonicalization and the shared
two-way weighted Schur/PCG backend where the effective FE structure permits it.
Both engines solve the same PPML estimating equations.

The integration deliberately keeps Poisson-specific code under `pyreghdfe.ppml`;
FE compilation, weighted projection, DoF, covariance and runtime infrastructure live
in the package core and are reusable by future iterative WLS estimators.

## Group-level outcomes with individual fixed effects (v0.5)

The input remains in long membership format: one patent can appear once per
inventor, but `y`, ordinary regressors, ordinary FEs, clusters, weights and
panel/time variables must be constant within the group. The individual ID must
also appear exactly once in `absorb`, matching the current `reghdfe` interface.

```python
r = reghdfe(
    patents_long,
    y="citations",
    x=["funding", "age"],
    absorb=["year", "inventor_id"],
    group="patent_id",
    individual="inventor_id",
    aggregation="mean",          # or "sum"
    method="lsmr",               # "lsqr" is also available
    vce="robust",
)
```

`aggregation="mean"` gives each membership weight `1 / group_size`; `"sum"`
adds individual FE contributions. Unequal group sizes therefore generally make
the two models different.

Individual-specific slopes are supported without expanding a dense dummy
matrix:

```python
r = reghdfe(
    papers_long,
    y="citations",
    x=["funding"],
    absorb=[
        "year",
        FixedEffect("author_id", slopes=("author_experience",), intercept=True),
    ],
    group="paper_id",
    individual="author_id",
    aggregation="mean",
)
```

`ivreghdfe` accepts the same `group=`, `individual=` and `aggregation=` arguments.
The group-level `[y, X, Z]` block is compressed once and residualized against the
same incidence operator before 2SLS/LIML/GMM2S.

### Incidence backend

The default is:

```python
incidence_backend="auto"
memory_budget_mb=512
```

For common sparse patent–inventor / paper–author graphs, `auto` builds a CSR
incidence operator when the estimated temporary construction footprint fits the
budget. This remains `O(number_of_memberships)` storage. If the graph is too
large, it falls back to `incidence_backend="matrix_free"`, which stores only
codes, slopes and work vectors. Both use LSMR/LSQR and span the same FE design.

You may force either path:

```python
reghdfe(..., group="patent", individual="inventor", incidence_backend="csr")
reghdfe(..., group="patent", individual="inventor", incidence_backend="matrix_free")
```

`result.group_info` reports retained groups, memberships, individuals,
aggregation, solver and selected incidence backend. `result.nobs_raw` is the
number of retained group-level estimation rows, not the number of long-format
membership rows.

### Group singleton pruning

With individual FE, a group is removed when it contains an individual whose
remaining graph degree is one; removals are repeated because they can create
new degree-one individuals. This implements the bipartite 1-core logic used by
current `reghdfe` for individual FEs. Ordinary group-level FE singletons are
pruned jointly in the same loop.

Current upstream `reghdfe` leaves redundancy/DoF adjustment for individual FE
as a TODO. `pyreghdfe` therefore conservatively counts all retained individual
FE coefficients rather than claiming an exact graph-rank correction.

`fweight` is deliberately rejected with `group+individual`: current upstream
source explicitly notes that frequency weights are not meaningful for this
membership representation. `aweight` / `pweight` are supported when constant
within group. Group+individual absorption is currently CPU-only; the ordinary
HDFE CuPy path remains available.

## Heterogeneous slopes

```python
r = reghdfe(
    df,
    y="y",
    x=["x1", "x2"],
    absorb=[FixedEffect("firm", slopes=("tenure",), intercept=True), "year"],
    vce="robust",
)
```

`intercept=True` spans a group intercept plus group-specific slope. Setting
`intercept=False` gives a pure heterogeneous slope.

## Typed Stata weights

```python
fw = reghdfe(
    df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
    weights="frequency", weight_type="fweight", vce="robust",
)
```

For `fweight`, `result.nobs` is the effective replicated sample size while
`result.nobs_raw` is the number of physically stored rows. `aweight` and
`pweight` are normalized internally; unclustered `pweight` forces robust
inference. IV rejects `fweight` with HAC/kernel/Driscoll–Kraay covariance.

## Recover fixed effects

```python
r = reghdfe(df, y="y", x=["x1"], absorb=["firm", "year"], save_fe=True)
combined_fe = r.fixed_effects.fitted
```

With multiple FE dimensions, level coefficients are normalization dependent.
The package returns a joint weighted minimum-norm decomposition and reports a
reconstruction error; the combined FE contribution is the invariant object.
The same behavior is available for group+individual FE, including individual
slopes.

## Large-data CPU absorption (v0.6)

Ordinary categorical FE use dense `int32` codes. For repeated multi-FE MAP, the
default `projection_backend="auto"` can build a compact CSR-like observation
index for each intercept-only FE using O(N+G) counting sort, then parallelize
projection over independent groups with Numba. This avoids atomic writes and
avoids the O(threads x groups x RHS) memory cost of thread-private accumulators.

```python
r = reghdfe(
    df, y="y", x=controls, absorb=["firm", "year", "industry"],
    projection_backend="auto",  # auto | indexed | fused
    absorb_threads=4,           # "auto" uses available Numba threads
    pool_size="auto",
    memory_budget_mb=512,
)
```

`indexed` is most useful when there are multiple FE dimensions, many RHS
columns, high FE cardinality, and enough repeated MAP sweeps to amortize the
index. `fused` remains the low-preprocessing fallback. Mixed heterogeneous-slope
models use indexed projection for eligible intercept-only terms and the existing
group-cross-product projector for slope terms. Weighted intercept projections
are supported by the indexed path as well.

`pool_size` controls the extra previous-iterate scratch used by MAP. Wider pools
reduce repeated scans of FE indexes but consume more memory. On the release
container, the 10M-row / 12-control / 4-FE / two-way-cluster stress model took
about 100.9 s with the conservative auto pool and 79.8 s with all 13 absorbed
RHS columns in one block; see `BENCHMARKS.md`.

GPU projection for ordinary HDFE is optional:

```python
r = reghdfe(..., backend="cupy", method="map", acceleration="cg")
```

The development environment has no CUDA device, so no release GPU timing is
reported.

## Parallel bootstrap

```python
from pyreghdfe import wild_bootstrap
r = reghdfe(..., keep_state=True)
b = wild_bootstrap(r, reps=999, n_jobs=8, batch_size=8, seed=123)
```

The encoded absorber and within-transformed design are reused; thread workers
share large immutable arrays and nested BLAS threading is limited. v0.6 also
partitions the Numba FE-thread budget across Joblib workers so bootstrap-level
and absorption-level parallelism do not oversubscribe the CPU.

## Validation status

The integrated v0.9.0a1 local regression suite contains **152 passing tests**: the
115 linear/IV release tests, 34 PPML replica/optimization tests, and 3 new shared
dynamic-weight/core integration tests.

PPML local coverage includes Statsmodels GLM parity without FE, explicit-dummy FE
parity, two-/three-way FE, robust/cluster VCE, offset/exposure, singleton and
collinearity handling, official simplex internal cases, public nonexistence-primer
examples, ReLU separation, standardization, replica/optimized equivalence, and
compiled projector weight updates. Licensed Stata `ppmlhdfe` golden execution and
the upstream 17-dataset separation corpus remain external certification gates.

The historical v0.8.0 linear/IV suite contains **115 passing tests**. Coverage includes the historical OLS/IV/weights/VCE and group+individual cases plus interaction-FE canonicalization, specialized two-way solving, structural and post-absorption collinearity, explicit omitted-variable controls, event-study references, and runtime/workspace execution behavior.

`validation/` now prepares **23 Stata golden models**. The original 18-model corpus is retained and five v0.8 cases add FE-hierarchy canonicalization, auto two-way routing, explicit user omission, structural factor collinearity, and an event-study reference-period specification. The comparator checks coefficients, standard errors, the `x`/`w` covariance, sample size, absorbed DoF, available weak-identification statistics, and v0.8 Python-side semantic metadata.

The repository intentionally does not ship fabricated `golden_results_stata.csv`. The development container has no Stata license, so licensed Stata execution remains an external release-certification gate.

See `ARCHITECTURE.md`, `BENCHMARKS.md`, `CHANGELOG.md`, `UPSTREAM_REFERENCES.md` and `validation/README.md`.

## Remaining compatibility boundaries

Important remaining work includes externally executed Stata golden parity on a
large corpus, exact saved-FE normalization parity, a sharper individual-FE rank
/DoF algorithm than current upstream's conservative TODO, group+individual GPU
kernels, additional historical `ivreg2` finite-sample/version quirks, CUE, and
some Stata `e(sample)`/zero-weight edge semantics.

## Final validation and agent integration

- `SKILL.md` / `skill/SKILL.md`: agent-readable usage and decision rules.
- `EXTERNAL_VALIDATION.md`: required Stata, cross-platform, CPU/NUMA, CUDA and 10M–100M external validation matrix.
- `TEST_STATUS.md`: what was actually verified in the current environment versus what remains external.
- `scripts/build_release.sh`: run the suite, build and isolated-smoke-test the wheel, assemble the source ZIP/release bundle, generate hashes, and verify bundle consistency.
- `scripts/run_external_matrix.sh`: opt-in external benchmark/Stata harness.

## Interaction fixed effects and solver defaults (0.7 development)

Empirically common interaction fixed effects can be specified without first materializing an interaction column:

```python
from pyreghdfe import reghdfe, interaction

r = reghdfe(
    df,
    y="y",
    x=["x1", "x2"],
    absorb=["firm", "year", interaction("city", "year")],
)
```

The nested tuple form `absorb=["firm", "year", ("city", "year")]` is equivalent.  Pure-intercept fixed effects that are exactly spanned by a finer fixed effect are removed from the numerical absorption system before singleton pruning and DoF calculation.  In the example above, `year` is redundant because every city-year cell belongs to exactly one year, so the effective system is `firm + city#year`.  The requested/effective systems are recorded in `result.absorb_info`.

Ordinary HDFE now defaults to symmetric Kaczmarz MAP with conjugate-gradient acceleration.  Public estimation raises `HDFEConvergenceError` when absorption fails to satisfy tolerance instead of silently returning a non-converged regression.  `allow_nonconverged=True` is available for deliberate diagnostics.

Heterogeneous slope-and-intercept terms are intentionally not removed merely because their intercept is nested in another FE; retaining that intercept can materially improve conditioning.

For canonicalized two-way intercept-only systems with a very large FE dimension crossed with a much smaller interaction FE, an explicit specialized solver is available:

```python
r = reghdfe(
    df, y="y", x=controls,
    absorb=["firm", "year", interaction("city", "year")],
    method="twoway",
)
```

After exact nested-FE canonicalization this solves `firm + city#year` through a Schur complement, analytically eliminates the larger diagonal FE block, and runs preconditioned CG only on the smaller FE side. With `method="auto"` (the public default in dev2), this specialized solver is selected automatically only when canonicalization leaves exactly two pure categorical intercept FEs on the NumPy backend. General HDFE systems resolve to symmetric MAP + CG; explicit `method=` always overrides the planner.

### General FE partition canonicalization

`0.7.0.dev1` treats categorical fixed effects as partitions of the estimation sample. For pure-intercept FE terms, a coarse FE `A` is safely redundant when every level of a finer FE `B` maps to exactly one level of `A`; equivalently `col(D_A) <= col(D_B)`. The canonicalizer therefore handles much more than literal component nesting:

```python
absorb=[
    "firm",
    "year",
    interaction("province", "year"),
    interaction("city", "year"),
]
```

If the estimation data prove `city -> province`, the numerical system becomes only `firm + city#year`. `year <= city#year` is certified from the interaction specification itself, while `province#year <= city#year` is certified by an exact full-data functional-dependency scan. The same machinery applies to industry-code hierarchies and higher-order interactions.

The candidate pipeline is deliberately conservative: specification proof -> cardinality filter -> deterministic sample *rejection* -> exact full-data certification. Sampling is never sufficient to remove an FE. If singleton pruning changes the estimation sample, canonicalization is repeated on the original FE set because new exact nesting can appear after pruning. Heterogeneous-slope terms are not aggressively eliminated as whole terms.

The complete proof trace is available in `result.absorb_info["canonicalization"]`, including `refinement_edges`, `equivalences`, dropped terms and diagnostic counts.



## Automatic solver planning and composite collinearity (0.7.0.dev4)

The public default is now `method="auto"`. Solver selection happens **after**
FE partition canonicalization, so redundant interaction components do not
misclassify the effective numerical problem:

```python
r = reghdfe(
    df, y="y", x=controls,
    absorb=["firm", "year", interaction("city", "year")],
    method="auto",
)

r.absorb_info["solver_selection"]
# requested='auto', resolved='twoway',
# reason='canonical_system_is_two_pure_intercept_fes'
```

The current auto policy is intentionally conservative:

- exactly two effective pure-intercept categorical FEs on NumPy -> `twoway`;
- general ordinary HDFE -> symmetric MAP + CG;
- group + individual FE -> LSMR;
- explicit `method=` -> never overridden.

### Explicit regressor frontend

Factor and composite regressor terms can be supplied without manually making
dummy/interaction columns:

```python
from pyreghdfe import factor, reg_interaction

r = reghdfe(
    df,
    y="y",
    x=[
        "size",
        factor("industry"),
        reg_interaction(factor("city"), "trend", name="city_trend"),
        reg_interaction("capital", "age", name="capital_age"),
    ],
    absorb=["firm", "year"],
)
```

This is a deliberately small explicit frontend, not yet a full Stata
`fvvarlist`/formula parser. Every expanded numeric column retains provenance
(term, kind, components and factor level), which is reused by the
collinearity report.

### Post-absorption collinearity resolution

Collinearity is resolved after HDFE partialling-out. This is essential for
cases where a column is not collinear in raw data but becomes collinear only
conditional on absorbed FEs. The resolver distinguishes:

- `absorbed_or_zero`: the within column is numerically annihilated by the FE
  space;
- `duplicate_or_scaled`: one retained column spans the omitted column;
- `linear_combination`: several retained columns jointly span it.

For example, if `year` is absorbed, factor/interacted instrument columns can
have joint rank deficiencies even though no individual raw column is zero.
The IV path resolves excluded instruments conditional on retained included
exogenous regressors, so redundant excluded instruments are removed before
first-stage and weak-IV diagnostics.

```python
iv = ivreghdfe(
    df,
    y="y",
    exog=["x1", "x2"],
    endog=["price"],
    instruments=[
        "z",
        reg_interaction(
            factor("quarter", drop_base=False),
            factor("year", drop_base=False),
            name="quarter_year",
        ),
    ],
    absorb=["firm", "year"],
)

iv.collinearity_info["excluded_instruments"]
```

Rank resolution is order preserving and uses weighted `K x K` Gram matrices
plus a stable two-pass orthogonalization in Gram coordinates. It therefore
avoids an additional large `N x K` QR workspace. In a local microbenchmark on
1,000,000 rows and 24 candidate columns with three exact composite
collinearities, the resolver took about 0.09 seconds.

`collinearity="warn"` is the default. Requested columns that are absorbed or
collinear are still removed from the estimable basis, but the omission is no
longer silent: an `OmittedVariableWarning` is emitted and full provenance is
stored in the result. Use `collinearity="raise"` for strict specification
validation, or explicitly choose `collinearity="drop"` only when silent
omission is intentionally desired. `collinear_tol=` controls near-collinearity.
`result.omitted_variables` provides a flattened audit view.


## Structural factor/interaction collinearity (0.7.0.dev4)

Standard HDFE models now compile the regressor design **after** FE canonicalization and singleton pruning. Before any dense N-by-K factor block is created, the frontend builds compact categorical partitions and applies exact structural rank rules. Examples include:

```python
reghdfe(
    df, y="y",
    x=[factor("year"), reg_interaction(factor("province"), factor("year"))],
    absorb=[interaction("city", "year")],
)
```

If the final estimation sample proves `city -> province`, both `year` and `province#year` are spanned by the absorbed `city#year` partition and are omitted before dense design materialization. Full interaction blocks are also basis-reduced against absorbed coarse FEs: full `qob#year` instruments with absorbed `year` use only `Q-1` cell directions per year.

The rules are conservative and exact. They use partition refinement certificates and preserve user order. General multi-block dependencies still fall through to the post-absorption Gram-rank resolver. Set `structural_collinearity=False` to disable the symbolic layer for golden/parity diagnostics.

Detailed diagnostics are available in:

```python
res.collinearity_info["structural"]
```

including requested/materialized column counts, dependency edges, proof types, and structural omissions.


## Dependency DAG closure and explicit omission reporting (0.7.0.dev4)

The structural design planner now builds a component-level refinement DAG before
reasoning about large factor/interaction blocks. Exact component mappings are
certified once, then transitive closure is reused. For example, if the final
estimation sample proves `city -> province -> region`, the planner can infer
`city -> region` without another full N-row scan and reuse that proof for terms
such as `city#year` versus `region#year`. Direct and closure-derived edges are
reported separately in `result.collinearity_info["structural"]`.

Omitted variables are no longer silent by default. The public APIs now use
`collinearity="warn"`; warnings summarize omitted names/reasons while the full
metadata remains available through `result.collinearity_info` and
`result.omitted_variables`. `collinearity="raise"` remains the strict preflight
mode and `collinearity="drop"` is an explicit quiet opt-in.

## Explicit references / omitted variables and event studies (0.8.0)

Omitted columns are now controllable as part of the model specification instead of being determined only by column order. The public default remains `collinearity="warn"`; all automatic omissions are reported. Deliberate references are separately tagged as user selections.

```python
from pyreghdfe import factor, reg_interaction, omit_level

event_study = reg_interaction(
    factor("event_time", drop_base=False, name="event_time"),
    "ever_treated",
    name="event_study",
)

res = reghdfe(
    df, y="y",
    x=["x1", "x2", event_study],
    absorb=["firm", "year"],
    omit=[omit_level("event_time", -1, term="event_study")],
)
```

Use `omit_column(name)`, `omit_term(term)`, or `omit_level(component, level, term=...)`. A selector that matches no requested expanded column raises an error. Results distinguish `res.user_omitted_variables` from `res.automatic_omitted_variables`, which is particularly useful for TWFE/event-study specifications where the researcher must know which coefficient is the reference and which coefficients disappeared for FE/rank reasons. IV has `omit_exog=`, `omit_endog=` and `omit_instruments=`.

The large-interaction execution path also now caches component encodings, uses multi-RHS PCG in the specialized two-way solver, and streams cluster-score aggregation to reduce peak memory. See `BENCHMARKS.md` for the 10M-row complex event-study stress test.



## v0.8.0 runtime-aware execution

Version 0.8.0 separates econometric solvers from runtime resource planning. `runtime.py` discovers CPU affinity/cgroup quota and memory limits; `execution.py` plans a reusable RHS workspace without changing the requested statistical model. `method="auto"`, `pool_size="auto"`, `absorb_threads="auto"`, and `projection_backend="auto"` remain the recommended defaults.

Local release validation: 115 tests pass. A 10,000,000-row hierarchical interaction-FE/event-study stress model completed in about 19.60 s with ~2.82 GiB peak RSS, canonicalizing six requested FE terms to `firm + city#year`, retaining an explicit event-time reference and structurally pruning 1,904 redundant columns. A 1,000,000-row generic four-FE / 12-control / two-way-cluster sanity model completed in about 5.93 s. The generic 10M four-FE sustained-memory benchmark is **not certified on the 4 GiB release cgroup in this run**; production 10M+ generic multi-FE performance must be validated on target hardware as described in `EXTERNAL_VALIDATION.md`.
