# Weighted singleton policies

Status: 0.9.2 candidate (developed under the untagged number 0.10.0), based on
main `33e8efe` (0.9.1).
This is a sample-policy change, not a new FE solver or a Stata certification.
The default is `stata_compat`; `unified` is explicit opt-in.

## Calling the policy

```python
from econhdfe import HDFEConfig, PPMLConfig, IVPPMLConfig

linear = HDFEConfig(singleton_mode="unified")  # olshdfe / ivhdfe / linear sessions
poisson = PPMLConfig(singleton_mode="unified")
iv_poisson = IVPPMLConfig(singleton_mode="unified")
```

Pass `hdfe_config=linear` to linear estimators/sessions and `config=...` to
Poisson estimators/reusable models. Use `singleton_mode="stata_compat"` for
compatibility. `ppmlhdfe` and `PPMLHDFE.fit` now accept `weight_type="fw"` or
`"pw"`, including full aliases; `aw` is rejected, matching the supported
Poisson interface. Leaving `weight_type` unset preserves generic-weight
semantics and does **not** infer frequencies from integer-looking values.

Linear `drop_singletons=False` remains a scalar option. Poisson families use
`PPMLConfig(drop_singletons=False)` / `IVPPMLConfig(drop_singletons=False)`.
Never interpret this as disabling all separation methods.

## Counting and stage matrix

Let `r_g` be physical row count and `f_i` validated positive integer frequency.
For non-frequency weights, statistical multiplicity is one, independently of
weight magnitude. `pw` sums are not sample counts.

| Stage | `stata_compat` | `unified` |
|---|---|---|
| Linear OLS/IV, including recursive updates | `fw`: sum(f)=1; otherwise r_g=1 | Same counting convention |
| PPML/IV-PPML initial clearing | r_g=1 even with `fw` | `fw`: sum(f)=1; otherwise r_g=1 |
| PPML/IV-PPML after simplex/ReLU trimming | `fw`: sum(f)=1; otherwise r_g=1 | Same multiplicity as initial stage, including zero-level rechecks |
| Initial all-zero FE levels | Combined with singleton removal/reporting | Separate `separated` reason |
| `drop_singletons=False` and initial FE-zero clearing | Coupled: initial FE clearing disabled | Independent: FE-zero clearing still enabled when `fe` requested |
| Ordinary PPML simplex/ReLU | Existing methods | Existing methods; validated numerically, not a new formal certificate |
| IV-PPML default separation | `fe`, `simplex`, `relu` | `fe` only |
| Threshold-only `mu` deletion | Existing estimator-specific behavior | Rejected |

For IV-PPML, `separation=None` (the default) means "the policy default" and is
stored as given; `IVPPMLConfig.separation_methods` resolves it where it is used
(`('fe','simplex','relu')` for compatibility, `('fe',)` for unified). A default
config therefore follows `dataclasses.replace(cfg, singleton_mode=...)` in both
directions. An explicit tuple is kept and validated against the policy.
Code queries `config.sample_policy` (`econhdfe.config.SamplePolicy`) rather than
comparing mode names.

The compatibility label identifies a **sample-policy profile**, not complete
Stata syntax, missing-value handling, arbitrary VCE parity, or a promise of
bitwise numerical agreement. Its upstream source manifest is
[`validation/singleton_policy/upstream.json`](../../../validation/singleton_policy/upstream.json).
No upstream source code is vendored. The new peeler is independently written.

## Mathematics and the boundary of the claim

Assume categorical intercept FEs, positive finite user weights, and (for `fw`)
positive integer multiplicities. A group's frequency sum is one exactly when
its physical degree is one and its sole surviving record has frequency one.
The implementation uses this equivalent predicate, avoiding floating-point
sums of large frequencies. Repeatedly removing eligible rows reaches the
unique fixed point of this monotone deletion rule, independently of FE and row
order. Positive outcome support can only disappear after deletion, so zero-FE
peeling is also monotone. Interleaving both operations reaches their common
fixed point.

For ordinary positive-weight linear equations, if `D c=e_i` and `D' W r=0`,
then `0=c'D'Wr=w_i r_i`: the row is fitted by FE independently of the slope.
This proves the usual singleton result. It does not equate graph leaves with
all independently fitted rows: bridges and more general rank phenomena are
not exhaustively detected here.

For a pure heterogeneous-slope term, a single physical row with all slope
values exactly zero does not provide such a direction. Unified linear mode
therefore requires an intercept or at least one nonzero slope in that row
before that term can trigger a singleton. This is a sufficient local check,
not a general local-rank saturation algorithm. `group()/individual()` unified
mode is explicitly rejected until its observation unit is implemented and
validated; compatibility behavior remains available.

For nonnegative outcomes, an all-zero intercept FE level imposes
`sum(w_i mu_i)=0`, possible only at the exponential-mean boundary `mu_i -> 0`.
Use the exact boolean support `y_i>0`, not a tolerance on a weighted sum.
A positive value as small as `1e-300` is still positive. `fe` in this version
means recursive all-zero **levels**; it does not implement an exhaustive
search over arbitrary FE-only recession directions.

For IV-PPML, ordinary PPML separation on `[D,X]` is not an existence test for
`Q'W(y-mu)=0`. The regression counterexample `x=(0,0,1,1)`, `y=(1,3,0,0)`,
`z=(0,4,3,5)` has a finite IV root at intercept=slope=0 although ordinary PPML
has a diverging slope. Unified IV-PPML accordingly rejects explicit
`simplex`, `relu`, and `mu`; it does not promise finite IV roots in all remaining
samples. Nonconvergence and rank failures still surface normally.

## Statistical versus working sample

A positive single-record FE group with `fw=5` is retained statistically in
unified mode. It represents five copies, not one observation. Its slope score
can vanish while its N/FE-rank/DF contribution still matters. Deleting it from
N would generally change finite-sample inference.

This PR does **not** add optional computational rank/bridge elimination.
`diagnostics['sample_policy']` reports `statistical_rows`,
`statistical_observations`, `working_rows`, and `computational_elimination=False`.
The two row sets currently coincide. Retaining the row in the existing solver
is the conservative implementation of the statistical/work-sample boundary;
no computational shortcut silently shrinks N, DF, clusters or log likelihood.
Poisson `sample_mask` and `separation_mask` refer to physical input records.
N and `nobs_full` (where supplied) use frequency totals for typed `fw`;
`n_singletons`/`n_separated` count removed records, not frequency copies.

Unified removal masks are disjoint. If simultaneous reasons apply, zero-FE
separation has precedence when the row leaves the queue. The union/retained
sample is permutation invariant; a row's first-trigger reason can depend on
queue ordering. Do not interpret per-reason counts as a unique causal
attribution for a deletion cascade. Compatibility retains historical combined
initial counts; in IV-PPML it also retains the established combined post-trim
reporting convention. These labels must not be used as effective N.

## Weight lifecycle and covariance

User multiplicities, user estimation weights, and IRLS weights `w_i*mu_i`
are separate. Only multiplicities decide frequency singletons. Working weights
are never interpreted as sample counts and are never used to identify tiny-
weight observations as singletons.

Frequency-robust meat is `sum(f_i psi_i psi_i')`, not
`sum(f_i^2 psi_i psi_i')`; clustered scores are aggregated as
`sum_{i in c} f_i psi_i` before their outer product. Plain PPML now forwards
both frequency effective N and the `1/sqrt(f_i)` robust-score adjustment to the
existing covariance layer. The IV-PPML finite-sample convention is unchanged
(N/(N-1) for robust; subset G/(G-1) for clustering). Unified mode does not
silently replace these estimator-specific VCE conventions with one universal
formula. Dense/structured, pooled/reusable, and compressed/expanded paths are
covered by tests, including an independent dense moment-root/sandwich oracle.

## Validation and unsupported input

Typed weights already require strictly positive values. This PR preserves
that boundary; unified linear mode also rejects zero generic weights. Filter
zero-weight records explicitly before the call. Negative/nonfinite/fractional
frequency weights raise rather than being silently dropped. Automatic
Stata `marksample` missing/zero-weight replication is **not** part of this PR.

The model-agnostic queue lives in `econhdfe.hdfe.sample`; model policy and
separation routing remain in `models/`. Linear session keys include the policy;
persistent fingerprints already include complete configs. No estimator
selection is pushed into the numeric peeler or the execution planner.

Each categorical incidence is initialized once and scanned only when its node
first becomes a singleton or loses its last positive row: work and storage
are O(q*N + total FE levels), with no physical frequency expansion. A Numba
kernel handles the queue. This complexity claim concerns the peeler only,
not factorization, FE absorption, rank or separation solvers.

See [`validation/singleton_policy/README.md`](../../../validation/singleton_policy/README.md)
for executed checks and external limitations. Million-row runs exercise sample
peeling, not end-to-end million-row PPML estimation or speed versus Stata.
