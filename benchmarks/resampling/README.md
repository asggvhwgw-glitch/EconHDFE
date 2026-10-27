# Coefficient-only wild bootstrap scores

`wild_bootstrap_scores.py` compares the production score path with the public
function's preserved per-batch path. A delegating custom projector selects
the latter, with the same underlying absorber, data, fit and RNG batches.
This is an implementation comparison; the separate dense dummy-SVD tests
provide the independent numerical reference.

## Recorded run

`wild_bootstrap_scores_1m_20261027.json` records three alternating warm runs
per route on Linux, Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and Numba 0.68.0.
The JSON includes exact versions, source hashes and all timings.

The synthetic design has 1,000,000 observations, three crossed FEs with
20,000 / 4,000 / 50 levels, three controls, 1,000 clusters, 64 Rademacher
draws in batches of eight, tolerance `1e-11`, and one CPU/BLAS thread.

| Weights | Legacy median | Production score median | Ratio | Max coefficient-draw difference |
| --- | ---: | ---: | ---: | ---: |
| Unit | 5.5652 s | 0.1716 s | 32.43× | 3.39e-15 |
| Lognormal, log SD 0.5 | 7.2433 s | 0.1516 s | 47.78× | 7.44e-15 |

Each score call includes cluster recoding, Gram pseudoinverse, condition and
memory checks, the one-time design projection, score aggregation, multipliers
and coefficient draws. The original fit, data generation, imports and JIT
warmup are excluded from both routes. All measured absorptions converged.
The legacy route made eight absorption calls per bootstrap; the score route
made one. Peak RSS is cumulative process RSS, not a per-route memory claim.

These are coefficient-bootstrap timings for this fixture. They do not claim
the same ratio for initial estimation, studentized WCR11/WCU11 tests, weak
designs that use the fallback, or arbitrary real datasets. In particular the
two-way Schur absorber retains its original path in this change.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python benchmarks/resampling/wild_bootstrap_scores.py --output /tmp/wild-scores.json
```

Use the functional launcher with `NUMBA_NUM_THREADS=4` for tests; the benchmark
thread cap must not be carried over to the full test suite.
