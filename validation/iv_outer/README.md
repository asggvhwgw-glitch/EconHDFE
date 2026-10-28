# IV-Poisson outer-estimator validation

Candidate `0.10.1.dev0`, based on `main@2ab8cddd9a6e367291f3152591e09c1270418148`.
The source baseline was obtained from the successful main CI run 40478222473,
artifact 12720269934. See `provenance.json` for the artifact and runtime source hashes.
This directory contains candidate evidence, **not** a public release certification.

## Scope

The nonlinear outer algorithm and weighted IV step change only when
`IVPPMLConfig(outer_solver="moment")` is selected. Ordinary PPML, FE solvers,
linear IV and the legacy outer default are unchanged. The new path supports
dense explanatory/instrument matrices and categorical-intercept FE. It
explicitly rejects BlockDesign and uncertified PPML regressor-separation routes.

The theory note is `docs/technical/ivppml/outer-moment-estimator.md`.
In exact identification the step is a damped inexact Newton step on additive
moments. In overidentification it preserves the legacy projected-IV equation;
raw overidentifying moments need not vanish. The frozen-instrument merit test
is **not** a global convergence theorem for the changing projected equation.

## Numerical tests

`tests/numerics/test_iv_outer_moment.py` includes independent dense-dummy
root and sandwich calculations, an analytic counterexample to Poisson-deviance
acceptance, frequency replication, changed-specification warm starts, scale
invariance, stable fallback, failure paths and trial-step safeguards.
The final targeted and full-suite outputs are recorded separately in this
directory. Historical development logs, where retained, are not the final gate.

## Benchmarks

`formal/` contains every successful and failed run's JSON/log plus the sequential
ledger. The timed harness is `benchmarks/ivppml_outer.py`. A portable runner is
`benchmarks/run_ivppml_outer_suite.py`; results are summarized and independently
compared by `benchmarks/summarize_ivppml_outer.py`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
  python benchmarks/run_ivppml_outer_suite.py --output-dir validation/iv_outer/reproduction
OPENBLAS_NUM_THREADS=1 python benchmarks/summarize_ivppml_outer.py \
  --directory validation/iv_outer/reproduction \
  --output validation/iv_outer/reproduction-summary.json
```

Each mode runs in a fresh process and gets the same-case 100,000-row warmup
outside the timer. Data generation, JIT compilation, checksumming and result
serialization are excluded; the full estimation transaction, including
sample preparation and robust inference, is included. Both modes use the same
`optimized` FE backend, same thread policy, same tolerances and identical input
hashes. Alternating mode order and three repeats are used for five primary
cases; weak-first-stage stress has one run per mode and is not a repeated
performance estimate. A three-run stable-linear-solver ablation isolates the
sufficient-statistic route within the new outer solver.

Peak RSS is the fresh-process high-water mark (MiB); it includes the surrounding
benchmark process and is not a theoretical fit-only allocation bound. The
4 GiB process-container memory limit is separate from the 2048 MiB projector
planning budget. Full saved arrays are used for comparisons; large `.npz`
outputs are omitted from the deliverable archive and can be regenerated from
the seeded harness. No run is discarded on performance grounds.

The DGP is for numerical/performance validation, not a Monte Carlo proof of
causal identification or valid inference under endogeneity. Conditioning guards
are numerical, not weak-IV-robust statistical diagnostics.

## External boundaries

No licensed Stata comparison was run for the new mode. No new Windows/macOS,
dependency-floor, GPU or Rust build matrix was executed. The pinned baseline's
old CI success is not used to certify this candidate. The default remains
legacy. Empirical speedup applies only to the recorded host/workloads; no
unconditional runtime, universal global convergence, or arbitrary-data speedup
is promised. The fast-path scratch bound and accepted-step certificates are
stated with their assumptions in the theory note.

## Final executed outcome

Full source suite: **561 passed, 1 skipped**; the skip requires the optional
`_econhdfe_within` native extension, which was not installed. New numerical
suite: **36 passed**. Installed-wheel numerical suite: **36 passed** from an
independent target directory. Text transcripts use `.txt` so source-distribution
packaging retains the hashed evidence. `checks.json`, `summary.json`, and
`candidate_acceptance.json` separate executed outcomes from review status.
