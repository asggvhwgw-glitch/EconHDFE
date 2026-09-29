# Benchmark layout

Benchmark evidence is grouped when a stable domain boundary exists:

- `hdfe/`: HDFE numerical-solver and exact-rank evidence.
- `ppml/`: PPML benchmark programs and results.
- `repeated/`: repeated-specification/session benchmarks.
- `real_world/`: externally executed, user-supplied real-data benchmark records; preserve original runs immutably and append follow-ups.

Some inherited linear/HDFE benchmark scripts remain at this directory level to avoid path churn in historical tooling. New benchmark artifacts should be placed in the appropriate domain subdirectory rather than adding more root-level files.

## Real-machine benchmark return format

Use `real_world/BENCHMARK_REPORT_TEMPLATE.md` for new user-authorized real-data benchmarks. Preserve prior dated records; never overwrite historical benchmark evidence. A benchmark report must separate parity from speed, state timing scope/cold-vs-warm behavior, record machine/package versions and resource settings, and avoid embedding raw/private data.

## Linear-IV diagnostic decomposition

`bench_iv.py` runs a synthetic IV workload with all existing diagnostics computed,
even when their display mode is `off`. It separates unprofiled warm timings from
a separate stage/call-count profile. For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
PYTHONPATH=. python benchmarks/bench_iv.py --n 96000 --groups 12 --reps 5 \
  --representation structured --vce robust --output /tmp/iv-structured.json
```

Run the same script in separate processes against both source revisions and
repeat with `--representation dense`; do not compare only an old dense run with
a new structured run. `--vce cluster` and `--estimator liml|gmm2s` select other
workloads. `--save-numerics` writes a companion synthetic-results NPZ for
coefficient, VCE, fitted-value, DoF and diagnostic comparisons. Timed dense fits
include role-design construction, as in the heterogeneous-spec benchmark.
Cumulative stage times overlap and must not be added together. Peak RSS is
process-wide (unavailable on Windows), not workspace allocation. These timings
are development evidence, not a universal speed claim or external Stata parity.

### 2026-09-29 workspace check

Baseline: `22e176b771cb9554ec26cc60338a92776c42031d`. On one Linux host,
Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 / Numba 0.65.1, BLAS=1 and
Numba=4, five unprofiled warm repetitions in separate before/after processes gave:

| Synthetic 2SLS workload | Before (s) | After (s) | Before/after |
| --- | ---: | ---: | ---: |
| 96k rows, 12 endogenous slopes, structured, robust | 1.738855 | 0.835278 | 2.08 |
| Same roles, dense, robust | 1.760017 | 0.938512 | 1.88 |
| 96k rows, 12 endogenous slopes, structured, clustered | 1.790198 | 0.890690 | 2.01 |
| 300k rows, one endogenous regressor, dense, robust | 0.224270 | 0.178506 | 1.26 |

All existing diagnostics were computed. Structured robust GELSY calls fell
from 125 to 22. The optimized dense/structured ratio is only 1.12, not 2.08:
the latter is the same structured path before versus after this patch.
Process peak RSS was 570.04 to 571.43 MiB for structured robust and 549.46 to
559.89 MiB for clustered; this is not a memory-reduction claim.

The four before/after pairs had identical coefficients, VCE, first-stage
coefficients/fits and DoF. Diagnostic differences were within rtol=2e-9,
atol=1e-9 (largest absolute difference 3.03e-9 on large F statistics).
The diagnostic test module passed 60 tests; a 132-specification old/new sweep
passed 528 family comparisons; 137 selected historical regression tests passed.
These local checks are not the current full repository suite, cross-platform
CI, or licensed-Stata certification. Public-contract capture for the root and
effects namespaces had zero differences. No version or default changes.
