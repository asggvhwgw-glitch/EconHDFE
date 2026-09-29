# Data layer phase 3 checkpoint — persistent empirical sessions

Status: development checkpoint on top of `0.4.10.1`; not a release/version bump.

## Scope

This checkpoint tests whether repeated empirical work remains reusable after the Python process/session ends. It adds an explicit `PersistentSessionStore` used through `OLSHDFESession.enable_persistent_cache()` and `IVHDFESession.enable_persistent_cache()`.

Implemented:

- versioned, atomic, pickle-free local cache files;
- cross-process encoded-column reuse for file-backed DataSources;
- incremental source expansion when a new y/x/IV variable is requested;
- FE/sample/weight-signature-scoped linear within-column reuse;
- completed OLS/linear-IV result resume;
- `fit_many()` partial-suite resume: completed specifications are skipped before unfinished specifications are prewarmed;
- strict full-content source validation and opt-in fast metadata validation;
- automatic invalidation after normal source-file rewrites;
- source-role separation: identifier-only codes are never reused as explicit factor/regressor values;
- safe cache deletion semantics.

Not implemented:

- persistent PPML/IV-PPML IRLS states;
- iteration-level numerical checkpoints;
- serialized HDFE absorber/topology objects;
- distributed/shared-network cache;
- automatic cache eviction policy.

## Development benchmark boundary

`persistent_session_resume_strict.json`: 150k rows, 32 raw CSV columns. Strict complete-file hashing made changed-specification reuse approximately break-even, while exact completed-result resume remained useful.

`persistent_session_resume_metadata.json`: 300k rows, 16 raw CSV columns. Metadata validation produced approximately 1.21x changed-y and 1.22x add-control estimator-call speedups, with smaller full-process gains; exact result resume bypassed the estimator and delivered much larger estimator-side savings.

The evidence does **not** justify making persistent within-column caching automatic, nor does it justify persistent FE-topology engineering. The feature remains an opt-in research-workflow accelerator and interruption-recovery mechanism.
