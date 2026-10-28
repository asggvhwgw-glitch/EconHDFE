# Optional structured within backend (experimental)

`method="within"` / `HDFEConfig(solver="within")` selects the optional Rust backend
for CPU pure-intercept standard OLS and linear IV. PPML and IV-PPML select it
with their family config `engine="within"` (see below). Defaults are unchanged.
Install the base package, then `python -m pip install ./native/within-accel`.
This source install requires Rust and network access; no published wheel is implied.

## Responsibilities

`econhdfe/hdfe/within_backend.py` is a residualizer and weight-update adapter.
Sample selection, statistical singleton deletion, canonicalization, DoF, weight
semantics, omissions, covariance and FE identification remain in econhdfe on
original rows. `native/within-accel` calls the unmodified Rust within crate pinned
to fc44282cea6124f5f18bb142908605a656d35a7c (MIT). This is a downstream companion,
not an upstream merge or a second implementation of LSMR/Schwarz.

## Numerical routes

A small global Schur complement eliminates the largest categorical term. The
reference-coded remainder is limited to 128 coordinates, admitted against known
scratch, diagonally scaled and checked for positive pivots and an inverse-based
condition estimate. Extra nullity / unsafe conditioning decline this route. No
ridge or eigenvalue truncation. Original-row stationarity and refinement can
trigger fallback to original within.

Stable LSD radix passes find exact FE tuples without mixed-radix overflow or
hash collisions. Current-weight cell means are solved by within and original
residuals restored as y_i - mean_cell + residual_cell. Automatic compression needs
at least 30% fewer numerical rows; this is an initial policy, not a calibrated
hardware crossover. Otherwise original within is used.

Positive weights and RHS columns are reversibly scaled. Weight updates are
transactional. Tuple maps are reusable but the numerical preconditioner is rebuilt
on a weight update; cross-IRLS preconditioner reuse is not implemented.

## Certification and bounded continuation

Each nonzero RHS is divided by its original maximum absolute value. The
acceptance metric is the largest absolute weighted group mean of the returned
residual, recomputed on **original rows** for every categorical term. Its gate
remains `max(tol, 64*float64_epsilon)`. An upstream convergence flag alone does
not certify a result; nonfinite means fail the gate. This remains a backward
stationarity check, not a condition-independent forward-error bound.

The initial inner solve uses `0.01*tol`, as before. If it reports convergence
but misses the original-row gate, the wrapper projects the current residual
again. Removing another vector in the same FE span preserves the original
FWL problem. Continuation is driven by the measured certificate and remaining
iteration budget, rather than a fixed two-replacement limit:

- A replacement that at least halves the metric permits another continuation.
- A stalled replacement permits one stricter inner target: the current inner
  tolerance is multiplied by `clip(gate/error, 0.1, 0.5)`. If that retry still
  does not halve the metric, the route stops. A successful contraction resets
  this stagnation check.
- A failed upstream solve, nonfinite metric, zero-iteration replacement or
  exhausted total iteration budget stops continuation. A regressing replacement
  does not overwrite the best residual. None of these stops implies acceptance.

If the upstream solve succeeded but continuation stops above the finite gate,
one weighted cyclic projection on the original rows is allowed when at least
one iteration remains. It subtracts each term's weighted group means in turn,
using compensated sums for both numerators and group masses. Each subtraction
belongs to the original FE span. This can clear a Krylov/preconditioner roundoff
floor without constructing another solver. The full sweep counts as one
iteration; it does not reserve or remove any iteration from the initial solve.
The wrapper retains the better residual and recomputes the same acceptance
metric. A sweep that fails that gate does not certify the result.

When an uncertified route has remaining iteration and admitted workspace, one
fresh original-row within solver is tried with a static **diagonal**
preconditioner. It starts from the original normalized RHS and uses the same
continuation, bounded cyclic projection and acceptance rules. This avoids
repeatedly reusing a stalled adaptive preconditioner. The fallback changes
neither the FE operator nor weights, sample, rank policy or tolerance; it
introduces no ridge or truncation.
All initial, continuation, cyclic and fallback iterations share `max_iter`.
If no route passes, the bridge reports nonconvergence and the estimator's
existing failure handling remains in force.

`resource_info["certification_fallbacks"]` counts these attempts over the
absorber's lifetime. The existing `schur_fallbacks` retains its narrower meaning:
attempts originating from the Schur route. The reported `route` still describes
the primary selected route. A fallback is an accuracy recovery path, not a claim
that the original route completed successfully or was faster.

## Resource and accuracy boundaries

Only one RHS Krylov solve is active, with inner Rayon parallelism in a private
explicitly sized pool. The bridge avoids upstream per-RHS result concatenation.
Python pools wide RHS blocks. No shared block Krylov basis is implemented.

workspace_bytes caps known reduction scratch / Schur admission. Python accounts
conservatively for observation and coefficient buffers, but upstream factor fill
and Schwarz subdomain construction are NOT hard-memory-bounded. This is not an
out-of-core implementation. Stationarity is a normalized backward check, not a
condition-independent forward-error bound. Validation requires independent
explicit-dummy or known-residual references.

Fallback admission includes the primary state that remains live. With `N` rows,
`K` terms and `L` summed FE levels, the existing original-row buffer estimates
are `R = 8*N*K + 24*N + 32*L` resident bytes and `P = 64*N + 96*L` one-RHS bytes.
Let `S` be the exact retained Schur struct and vector-capacity bytes, or zero for
an iterative primary route. A native call buffering `b` RHS columns admits a
fallback only when `2*(R+P) + S + 8*N*(b-1)` fits `workspace_bytes`. This
conservatively charges primary scratch as well as the fresh diagonal solver;
the large retained Schur cross matrix is not omitted.

The cyclic projection needs `8*N + 32*max(levels)` scratch bytes after the
upstream solve's temporary vectors have been released. This fits within the
existing `P` reservation, including when a second solver is not admitted. The
actual fallback constructor is guarded by checked arithmetic for all known
buffers and receives only the remaining iteration allowance.

Python charges `R+S` resident bytes. If a one-column fallback fits, each pooled
column reserves `R+2*P`; otherwise it reserves `P` and the native guard disables
fallback. Pool width uses the budget after subtracting resident state. Thus a
tight budget can still run the primary route and honestly report failure if
certification needs an unavailable fallback. These known-buffer estimates do
not become a whole-process RSS or opaque upstream allocation guarantee.
Older ABI-1 wheels remain loadable: the adapter conservatively estimates their
Schur storage and always reserves their unguarded Schur fallback. Rebuild the
optional extension to obtain the new conditional admission and continuation.

## Scope exclusions

No default switch, GPU, slopes, group+individual or zero-weight native route.
WeightedFEProjector uses existing LSMR for zero weights. PPML/IV-PPML estimation
can opt into native absorption; separation stays on the replica solvers.
Reported FE normalizations stay on econhdfe's established recovery path.

## PPML and IV-PPML integration

After installing the optional extension from this checkout, select it explicitly:

```python
from econhdfe.models.ppml import PPMLConfig, ppmlhdfe
from econhdfe.models.ppml_iv import IVPPMLConfig, ivppmlhdfe

ppml_result = ppmlhdfe(
    y=y, x=X, absorb=[firm_id, year_id],
    config=PPMLConfig(engine="within"),
)
ivppml_result = ivppmlhdfe(
    y=y, exog=C, endog=E, instruments=Z, absorb=[firm_id, year_id],
    config=IVPPMLConfig(engine="within"),
)
```

The same configs apply to DataFrame calls and reusable `PPMLHDFE` /
`IVPPMLHDFE` models. `ExecutionConfig` continues to set threads and projection
workspace budgets. Neither default `engine="replica"` nor `engine="optimized"`
is changed, and no automatic native selection or speed guarantee is implied.
PPML with no FEs has nothing to absorb; IV-PPML's existing noabsorb convention
uses one constant FE and therefore invokes the requested absorber.

`hdfe.factory.projection_route` resolves this engine once, in the shared
`WeightedFEProjector` constructor, to `method="within"` with certified solves.
It covers rank checks, every IRLS weighted projection, post-separation sample
rebuilds and final high-accuracy inference projection. Block-local projectors
and the one-shot `FEPlan.residualize` entry honor the same selection. Original
FE plans and reported DoF remain unchanged; this mode does not implicitly
request the separate optimized canonicalization policy.

At iteration t the numerical weight is the estimator's existing working weight
based on `true_w * mu_t`; IV-PPML retains its pre-existing lower bound. The
adapter copies the weight state, so in-place reuse of the outer IRLS buffer
cannot silently leave stale weights in the native engine. The topology and
exact-tuple map are reusable, but numerical preconditioners are rebuilt on each
weight update. Cross-IRLS factorization/Krylov recycling is not implemented.
Existing fast-partial, inner tolerance scheduling, standardization, step control
and warm-start semantics remain at the estimator layer.

Separation is deliberately separate: `detect_separation` runs on
`config.separation_engine`, which is `"replica"` for `engine="within"`. Simplex
and ReLU therefore keep the established auxiliary solvers, including exact
zero weights. A shared projector directly given zero weights uses its separate
LSMR cache; it can return to its native cache after strictly positive weights
resume. This is not a zero-weight native implementation or a changed sample rule.

Every solve on the `engine="within"` projector (including its zero-support
LSMR path) raises `HDFEConvergenceError` (`code="convergence.hdfe"`,
`stage="hdfe"`) for unsuccessful/nonfinite inner certificates. An explicit
`method="within"` request on another engine keeps returning `AbsorbInfo`
unchecked, as before. This check occurs before rank filtering, an IRLS update or final
covariance construction can consume a failed result, even when the caller
would discard `AbsorbInfo`. The raw `WithinFEAbsorber` information-returning
contract is unchanged. Missing native extensions fail explicitly instead of
silently changing the requested estimation backend.

For dense fits, inspect `result.diagnostics["projection_resources"]`: `backend`
and `method` identify selection, and `within` (the absorber's
`absorb_info_extras()`) contains route, original-row and
cell counts, `weight_generation` and certification-fallback counts. Block fits
report the corresponding entries per component. Counters on a reused model
are lifetime counters, not timings or a per-fit iteration count. Inner backward
stationarity and outer deviance convergence are different tests; neither alone
is a condition-independent forward-error guarantee.

The native test module `native/within-accel/tests/test_poisson.py` includes
independent full-dummy nonlinear roots for PPML and exactly identified additive
IV moments, replica parity for coefficients/means/covariances/sample/DoF, units
and permutations, frequency-weight expansion, separation, cached warm starts,
weight-support changes, and failure propagation. Overidentified IV is tested
against the established estimator, not an inappropriate just-identified root.

A reproducible end-to-end large-sample audit is available with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
python native/within-accel/validate_poisson.py --rows 1000000 --threads 2 --repeat 1
```

At one million rows it uses three FE partitions (20,000 / 4,000 / 50 levels),
90% partial nesting between the first two partitions, unequal positive weights
and an offset. It checks both estimators against replica, plus independently
computed raw FE and regressor/instrument scores. The IV corpus is exactly
identified; its raw instrument-score gate is not asserted for general
weighted overidentification. Positive continuous outcomes avoid separation in
this corpus; zero outcomes and separation are covered by the smaller tests.
The JSON includes inputs/source/binary hashes and all timings. One repetition
establishes accuracy for this corpus, not a stable acceleration factor; request
at least three alternating repetitions for same-machine performance study.
The native CI workflow runs this audit at 100,000 rows on pull requests and at
one million rows on manual dispatch.

## Validation

```
cargo test --release --manifest-path native/within-accel/Cargo.toml
python -m pip install ./native/within-accel
REQUIRE_WITHIN_NATIVE=1 python -m pytest native/within-accel/tests -q
python scripts/run_tests.py --suite full -- -q
```

REQUIRE_WITHIN_NATIVE=1 makes a missing extension fail rather than skip. The
benchmark uses known residuals, nonadjacent repeated tuples, positive weights,
fresh RHS reuse, fixed threads and an external 1e-8 relative weighted residual
gate. First timing includes constructor and solve, excludes DGP and validation.
The plain route is original within under identical wrapper policies, not the
upstream solve_batch default. No speedup is claimed before actual CI results.
No publication/version bump or mathematical novelty claim is part of this PR.
