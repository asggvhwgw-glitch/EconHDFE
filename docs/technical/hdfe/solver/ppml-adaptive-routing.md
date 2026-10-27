# Opt-in adaptive routing for nonlinear weighted FE projection

Use `PPMLConfig(engine="adaptive")` or `IVPPMLConfig(engine="adaptive")`.
Defaults remain `replica`; `optimized` and explicit `within` are unchanged.
The policy sits above existing numerical solvers, not inside Rust. It does not
modify outcome, sample, singleton/frequency-weight policy, rank, moments, outer
convergence, or covariance formulas. Ordinary OLS/IV automatic routing is not
changed by this work.

## Narrow eligibility, not a universal speed guarantee

The first version considers native only for **three** categorical intercept FE
partitions on at least **100,000** retained observations. At least one native
trial iteration and a compatible optional ABI-1 extension must be available.

* A small-Schur candidate has `0 < q <= 64`, where
  `q = sum(levels) - max(levels) - number_of_dimensions + 1`.
  The native `schur` strategy must pass its existing rank/conditioning gate;
  failure does not silently construct an unrelated native iterative backend.
* Otherwise, repeated-cell compression requires an **exact** distinct tuple
  count at most one quarter of the retained rows. Overflow-safe signed int64
  mixed-radix keys are used only when the product of the level counts fits.
  The unique count of a prefix is a lower bound, never a sample extrapolation.
  Counting all tuples confirms admission. Row permutation may change probe
  work, but cannot turn an ineligible tuple count into an eligible one.
* Cases outside these gates retain optimized MAP/CG or the established two-way
  solver. Small samples, missing native installations and encoding overflow do
  not trigger native compilation or a build-and-time race.

These constants are conservative engineering gates, not a learned cost model,
proof of optimality or a claim that all admitted data will be faster. In
particular, Schur width alone cannot guarantee a construction-time advantage.
The selector includes observed size and known workspace admission but does not
predict exact factorization time, global conditioning or machine bandwidth.

## Resource policy and transactional failure

Tuple probing reserves `40*n` bytes of known scratch and is declined if it
would consume more than one eighth of available/configured capacity. The
native precheck includes the adapter's guarded resident/RHS estimates, Schur
state estimates where applicable, and an input-sized output buffer. Existing
scheduler reservations and native checks still apply. These are cooperative
inventories, **not a hard bound on native peak RSS or upstream factor fill**.

Native attempts always project into a separate buffer, even when the caller
uses `copy=False`. The original input is committed only after success. An
unsuccessful or nonfinite certificate, a recognized Schur rejection or a
recognized workspace failure releases native references and retries from the
original input with the optimized backend. A failed retry raises the existing
`convergence.weighted_fe` error; outer deviance convergence cannot override it.
Unexpected programming exceptions are not swallowed.

A native trial gets at most `min(256, max_iter // 4)` iterations **per RHS**.
On failure its complete allowance is conservatively charged against the same
per-RHS budget before fallback. Reported fallback iterations include that
budget charge, not just actual Krylov iterations. A later call gets the full
fallback budget. Native failure/decline due to live workspace or zero weights
is sticky for that projector: IRLS does not repeat an expensive failed trial.
Zero-weight calls use LSMR; subsequent positive-weight calls use MAP rather
than re-entering native. Changed samples receive newly keyed projectors.

Every adaptive inner solve requires a successful finite certificate. Native
weights update on every call. No numerical weight vector or residualized design
is treated as fixed across IRLS. Simplex/ReLU separation explicitly keeps the
optimized engine's operators and is outside adaptive native selection.

## Lifecycle and diagnostics

The shared internal factory covers initial rank screening, direct FEPlan
projection, iterative PPML/IV-PPML solves, final inference, reusable-model
caches and certified block-separable component projectors. Caches include the
requested engine and original FE fingerprint. Signature validation invalidates
topology changes; sample trimming creates a new local plan.

`result.diagnostics["projection_resources"]` records the effective method and
backend, selection reason, dimensions, exact tuple count when computed, probe
seconds/known bytes, trial cap and sticky fallback events. An active native
backend additionally exposes its actual route and weight generation. These
fields contain aggregate metadata, not raw observation identifiers.

## Reproduction

The benchmark retains the default native comparison and adds an opt-in
candidate selector:

```bash
python benchmarks/hdfe/ppml_end_to_end_profile.py \
  --candidate adaptive --case repeated --model ppml --rows 1000000 \
  --repeat 3 --seed 20261029 --budget-mb 2048 --output repeated-ppml.json
```

The complete public fit is timed, including probing, construction, changing
weights, input-protecting buffers, rank screening and inference. Generation,
JIT warmup and post-fit audits are excluded. Comparisons use identical inputs,
outer tolerance 1e-9, target inner tolerance 1e-10, robust VCE and FE-only
separation. Each repetition alternates engine order and saves certificates,
input hashes, the actual extension binary hash and output differences.

Existing first-/second-seed **explicit native** results are documented in
`ppml-native-within.md`; they must not be relabeled as adaptive measurements.
Current tests add transactional failure injection at rank/IRLS/final stages,
iteration-budget checks, exact-count/order/overflow/resource gates, cache
invalidation and block-component parity. Large-scale near-separation and weak-IV
performance are not claimed.

## Local million-row validation (seed 20261029)

Linux x86_64, one thread, three alternating repetitions per engine.
No runtime or Rust algorithm changes were made during these measurements.

| Structure | Model | optimized (s) | adaptive (s) | Ratio | Final backend |
|---|---|---:|---:|---:|---|
| repeated | ppml | 4.3940 | 2.1452 | 2.048x | within |
| repeated | ivppml | 7.8246 | 4.5983 | 1.702x | within |
| crossed | ppml | 3.6558 | 3.6693 | 0.996x | generic |
| crossed | ivppml | 10.3175 | 10.5404 | 0.979x | generic |
| weak | ppml | 9.6051 | 9.7045 | 0.990x | generic |
| weak | ivppml | 20.9185 | 21.0395 | 0.994x | generic |
| schur | ppml | 3.2943 | 1.9637 | 1.678x | within |
| schur | ivppml | 7.1378 | 4.2294 | 1.688x | within |

All 48 timed fits and 24 comparisons pass. Maximum absolute differences: coef_max=1.693e-15, vcov_max=7.93e-20, mu_max=4.181e-11. Sample/separation masks, retained columns and absorbed DoF match. Repeated-cell and small-Schur candidates are selected; general crossings and the weak-community fixture remain on the existing solver.

Ratios near one include probe and transactional-input overhead and timing variation; no universal no-regression or speedup guarantee is asserted. The metadata and source/raw hashes are in `benchmarks/hdfe/ppml_adaptive_profile_1m_20261027.json`.
