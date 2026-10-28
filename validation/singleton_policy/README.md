# Weighted sample-policy validation

Baseline: main `33e8efe2b4cfe694c496cd2fb8da63260ae41d53`, tree
`6abce6d45afb08d75fc6263fb96fe4adffdd7495`. Source was recovered from the CI
sdist built at merge test commit `b1a09d3e952d074ff2bd403d4d200bd0b963587b`;
GitHub reports the same tree hash as main. The unmodified baseline passed all
468 core/full-suite cases locally before development.

Local final result: **509 passed** (including **41 new targeted checks**), four
expected omission warnings; version gate, architecture check and offline wheel
installation/policy smoke passed. See `local_checks.json`. Remote CI is separate.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 python -m pytest tests/behavior/test_singleton_policy.py tests/numerics/test_sample_peeling.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 python benchmarks/singleton_policy_million.py --output /tmp/singleton-million.json
python scripts/run_tests.py --suite full -- -q
python scripts/version.py gate
python scripts/generate_architecture_map.py --check
```

The targeted tests cover frequency expansion, linear/IV sessions, dense and
block-structured Poisson paths, separate initial deletion reasons, the
post-separation frequency recheck, ordinary-weight scale invariance,
zero-valued slope-only terms, no-singleton mode, invalid weights, and the
IV-PPML counterexample. Independent references include a brute-force integer
fixed-point implementation, explicit linear dummy regression, and dense
full-design nonlinear moment roots with independently assembled sandwiches.

`million.json` is actual execution of four one-million-row sample-peeling
checks. Two use four FE dimensions with 500000/500001/1000/53 levels and an
analytic long-chain fixed point. Two use 190000/40/5000/200 levels and known
masks under the two policies. Timings are single-run scope observations, not a
benchmark speedup claim. Generation and JIT warmup are excluded; public API
validation and peeling allocations are included.

`upstream.json` pins the source-audited profile. A licensed Stata environment
was **not** available: no new golden Stata output, cross-platform run, or
external certification is asserted. The frozen source headers/blob hashes
are audit provenance, not substitutes for running the estimators. Repository
CI results must be checked on the final PR head independently of local results.

No rank/bridge computational elimination was added. Statistical and working
rows are equal. Arbitrary FE-recession certificates, grouped individual FE
unified semantics, automatic zero/missing-weight sample selection, and new
HC2/HC3/CR2 guarantees are outside scope. Candidate version 0.9.2 (developed as untagged 0.10.0) is not a tag,
published package, or authorized release; acceptance records remain pending.
