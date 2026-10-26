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
