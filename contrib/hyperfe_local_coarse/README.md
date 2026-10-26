# Hypergraph local/coarse FE solver — research prototype

**Draft research contribution, not an estimator backend.** Nothing here is
imported by `econhdfe`, registered as a method, or included in its wheel. The
existing estimator defaults, dependencies, sample rules, DoF and inference
contracts are unchanged. This is separate from the native-within adapter PR #25.

This freezes the 2026-10-26 prototype before developing an automatic partition
selector. It profiles out one categorical FE, builds a star-Schur approximation,
partitions its normalized strong-edge graph into bounded local blocks, and adds
an optional balanced coarse correction. Rectangular LSMR still solves the
original weighted projection. The threshold affects the preconditioner only;
it never drops observations, changes weights or adds a ridge to the target.

## Run locally

Use a separate Python 3.13 environment on Linux for the tested benchmark setup:

```bash
cd contrib/hyperfe_local_coarse
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 python -m pytest tests -q
python benchmark.py --case smooth_sparse --mode balanced --tau 0.04 --seed 20261028 --n 1000000 --rhs 3 --out results/local.json
python benchmark.py --case smooth_sparse --mode native_adaptive --seed 20261028 --n 1000000 --rhs 3 --out results/native.json
```

The native comparator is the real `within-py==0.4.0` extension. Nothing is copied
from its Rust implementation, and no wheel is vendored. Benchmark commands run
one isolated process each and force one thread. Small tests establish algebraic
correctness, not performance. Performance runs below one million rows are rejected.

For independent all-column checking, first run the same case/seed/RHS count in
`--mode reference --tol 1e-13 --save results/reference.npy`, then pass
`--reference results/reference.npy` to the other runs. `--reuse` changes the RHS
on the same design; `--relabel` permutes FE level labels. Output JSON, including
`status`, `passed` and rejection reasons, is authoritative: the diagnostic script
records failures rather than using its process exit code as a success signal.

## Boundaries that must remain visible

- `tau` is manual. Automatic threshold/cost selection is **not implemented**.
- `equal_sparse` and `smooth_sparse` remove the weight split, but retain the
  planted additive relation from `joint4`; they are not unrelated hard designs.
- The first two seeds were evaluated at tau=0.03. Equal-weight seed 20261027
  exceeded the 256-level block budget; tau=0.04 was then tested on seed 20261028.
  This is sequential research/tuning, not a fully untouched holdout experiment.
- Irregular communities can become small blocks yet select no coarse vectors;
  the block-only path was much slower than native within. The nonadditive
  high-dimensional design was rejected. Rejections and misses are not wins.
- `preflight()` is the old conservative weight-support heuristic; it bypasses
  uniform weights and is **not** an adaptive threshold selector.
- Positive finite weights, pure categorical intercept FE and observed dense
  nonnegative integer codes only. Input arrays must not be mutated during a
  solve. The low-level routines are research internals, not validated public APIs.
- Pairwise reference removal does not certify every higher-order nullspace.
  Extra singularities can fail factorization; no ridge silently repairs them.
- Block/coarse limits are structural guards, not a hard RSS bound. Sparse Gram
  assembly and observation-space coarse columns can allocate before a guard.
- No real-data generality, full PPML weight lifecycle, varying-slope, zero-weight,
  GPU, estimator integration or external DoF/small-sample convention claims.

`src/scalable_hyperfe.py` includes older comparison kernels; the formal benchmark
uses `src/validation.py`, whose oracle error is normalized by the true residual,
not by the input. IC(0), ILUT and full-star modes are retained as research controls,
not recommended defaults. New estimator integration needs a separate PR.

## Evidence and reproducibility

`evidence/run_ledger_compact.json` retains all 54 historical recorded run outcomes,
including errors/rejections, with rounded timings. Its zero timing entries mean
no completed total was recorded, not zero-cost failure. Source archive hashes and
unchanged numerical-kernel hashes are in `evidence/provenance.json`. Million-row
inputs and residual arrays are generated from seeds, not committed.

`evidence/submission_checks.json` is separate: 38 tests and four one-million-row
packaging checks (prototype, native, independent reference, expected rejection).
These single runs do not replace the historical repeated timing experiments.
The full EconHDFE suite, version gate and architecture-map check were not run
locally because cloning the complete repository was blocked by DNS. This draft
must not be described as having passed those checks or as release-ready.

## Next research step

Freeze this snapshot. Develop automatic threshold selection with block/coarse
budgets, construction/application cost and untouched structural holdouts. Keep
native within as the default comparison and explicit fallback candidate; do not
change the main package's default on the strength of these synthetic results.

Two-level Schwarz, Schur complements and spectral coarse spaces are established
techniques. No historical-priority claim is made. For the algebra and scope, see
`../../../docs/technical/hdfe/solver/hypergraph-local-coarse-prototype.md`.
