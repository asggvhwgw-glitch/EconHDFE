# Optional structured within backend (experimental)

`method="within"` / `HDFEConfig(solver="within")` selects the optional Rust backend
for CPU pure-intercept standard OLS and linear IV. Defaults are unchanged.
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

When an uncertified route has remaining iteration and admitted workspace, one
fresh original-row within solver is tried with a static **diagonal**
preconditioner. It starts from the original normalized RHS and uses the same
continuation and acceptance rules. This avoids repeatedly reusing a stalled
adaptive preconditioner. The fallback changes neither the FE operator nor
weights, sample, rank policy or tolerance; it introduces no ridge or truncation.
All initial, continuation and fallback iterations share `max_iter`. If no route
passes, the bridge reports nonconvergence and the estimator's existing failure
handling remains in force.

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
WeightedFEProjector uses existing LSMR for zero weights. High-level PPML/IV-PPML
routing is not switched: the reusable weighted primitive is only infrastructure.
Reported FE normalizations stay on econhdfe's established recovery path.

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
