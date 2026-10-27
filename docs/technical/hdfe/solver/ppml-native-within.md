# Native within for weighted Poisson-family estimators

PPML and IV-PPML support `PPMLConfig(engine="within")` and
`IVPPMLConfig(engine="within")`. The default `engine="replica"`
and existing `engine="optimized"` remain unchanged. This adds an
engine choice without changing published public dataclass signatures
or the release-0.9.1 contract snapshot.

For the within engine, dense and block-separable Poisson estimators
use the existing `WeightedFEProjector(method="within")`.
The selection applies to post-FE rank screening, IRLS weighted solves,
separation-induced sample rebuilding, and the final high-precision
FWL/inference projection. Separation-specific simplex and ReLU
operators remain unchanged.

Native within requires strictly positive finite numerical weights.
The FE topology is reusable, but numerical weights are updated on each
IRLS iteration. For zero-weight states the existing generic LSMR
backend is selected, returning to native when weights are positive.
Absent the optional `_econhdfe_within` extension, an explicitly
requested within solve raises a configuration error.

Numerical tolerances, rank policy, moment conditions, separation,
sample semantics and VCE remain unchanged. Inspect
`result.diagnostics["projection_resources"]` to see the method
and effective backend. Speedup requires end-to-end validation.

## Million-row end-to-end profile (2026-10-27)

Single-thread Linux x86_64, 1,000,000 observations, three repetitions with alternating engine order. Timings include complete fits, not just residualization; generation/JIT and post-fit audits are excluded. `separation=("fe",)` is used and both engines receive the same tolerances.

| Structure | Estimator | optimized (s) | within (s) | Speed ratio | Update share of native fit |
| --- | --- | ---: | ---: | ---: | ---: |
| repeated | ppml | 4.317 | 1.882 | 2.29x | 6.8% |
| repeated | ivppml | 7.751 | 4.299 | 1.80x | 2.7% |
| crossed | ppml | 3.688 | 4.426 | 0.83x | 7.0% |
| crossed | ivppml | 9.563 | 12.022 | 0.80x | 2.8% |
| weak | ppml | 10.192 | 6.742 | 1.51x | 5.0% |
| weak | ivppml | 22.420 | 17.841 | 1.26x | 1.5% |
| schur | ppml | 3.424 | 2.165 | 1.58x | 22.5% |
| schur | ivppml | 6.149 | 3.780 | 1.63x | 9.1% |

See `benchmarks/hdfe/ppml_native_profile_1m_20261027.json` for measured-source identity, recorded import-entrypoint and raw-record hashes (see correction below), exact medians, and limitations. Reproduce with `benchmarks/hdfe/ppml_end_to_end_profile.py`.

Native is faster on repeated cells and small-Schur designs, but slower on the general crossing in this corpus. Do not change the default engine on this evidence. Eliminating numerical weight updates altogether would save only the measured update share, not a factor of ten. The weak-community design also compresses repeated cells; no pure preconditioner attribution is made.

## Certification propagation

The opt-in native nonlinear projector now requires a successful finite inner certificate at rank filtering, IRLS and final inference. A failure raises `ConvergenceError(code="convergence.weighted_fe")`; it is not hidden behind outer deviance convergence. The raw projector and linear paths retain their existing status-returning contract, and legacy replica/optimized behavior is unchanged. Tests include an independent full-dummy nonlinear moment root, six failure-injection locations, an actually exhausted native iteration budget, and zero-support/positive-weight transitions.

The original dynamic-weight microbenchmark generated non-dense labels and was rejected before timing native at 100k rows. It now uses `FEPlan.from_arrays`, like formal estimator inputs. Failed runs are not reported as speedups.


## Follow-up: final-source replication and binary provenance

The benchmark previously computed `native_sha256` from `native.__file__`.
For the maturin package layout this is `__init__.py`, not the loaded extension.
Consequently the historical `native_library_sha256` in the original summary
identifies the import wrapper only. Those historical measurements are preserved;
the current binary hash is **not** retroactively assigned to them.

New records use `native_identity_schema=2`: `native_sha256` hashes the loaded
extension (`.so`/`.pyd`), while `native_import_entrypoint_sha256` separately hashes
the import entrypoint. An unresolved extension path fails explicitly. Three
regression cases cover a direct extension, a package/submodule, and an unresolved
Python wrapper; changing the binary under an unchanged wrapper changes its hash.

A second-seed replication uses the final certified runtime at
`5da5990113a5c2413e4755c96852fabdfa40f6eb` (CI merge
`b9ade49e9acf52e8a52dd527ea3a0297cbd05486`, identical source tree
`0c494631ef72975efc6cb1d0421a66a64908e1f6`). No estimator or Rust numerical
algorithm was changed in this follow-up. The original Linux native CI artifact
`12716039905` supplies both the source and ABI3 wheel; no fresh local Rust build
is claimed.

- Seed: `20261028`; rows: 1,000,000; one thread; three alternating-order repetitions.
- Outer tolerance: `1e-9`; target inner tolerance: `1e-10`; robust VCE;
  `separation=('fe',)`; 2,048 MiB configured budget.
- Environment: Linux x86_64, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Numba 0.65.1.
- Actual native extension SHA-256:
  `a3e30e39c493da8226e18c1bd84fba83e2aa5a0a1c90924410ff213ea56b28a8`.
- Harness SHA-256:
  `d532b98c409393941ab7bc5a5c347250a3c0f2c4106a08b947ba520dbe6a81f0`.

| Structure | Model | optimized (s) | within (s) | Ratio | Native update share |
| --- | --- | ---: | ---: | ---: | ---: |
| repeated | ppml | 4.3072 | 1.9551 | 2.203x | 6.45% |
| repeated | ivppml | 10.3636 | 5.1070 | 2.029x | 2.84% |
| crossed | ppml | 3.8264 | 4.7696 | 0.802x | 6.21% |
| crossed | ivppml | 11.0982 | 11.3160 | 0.981x | 2.64% |
| weak | ppml | 10.3215 | 6.7308 | 1.533x | 3.07% |
| weak | ivppml | 21.8371 | 14.6186 | 1.494x | 1.63% |
| schur | ppml | 3.5540 | 1.9478 | 1.825x | 19.60% |
| schur | ivppml | 8.4442 | 5.3630 | 1.575x | 10.22% |

The local full Python suite passed 506 tests (four existing omission warnings).
The earlier focused native/integration/resource run passed 87 tests.

All 48 timed fits and all 24 matched comparisons passed. The largest absolute
differences were 7.38e-13 for coefficients, 1.30e-17 for covariance entries,
1.70e-7 for fitted means and 4.48e-8 for deviance. Every timed inner solve returned
a successful finite certificate. Sample masks, separation masks, retained names
and absorbed degrees of freedom matched. These are paired-backend comparisons;
the independent full-dummy moment-root tests are separate correctness checks.

Warm preparation (weight validation/dispatch plus numerical rebuilding) accounts
for median shares of 1.63%-19.60% of native total time in this second-seed corpus.
Holding other costs fixed, eliminating it entirely gives at most about 1.244x
end-to-end speedup; even eliminating cold preparation too gives about 1.310x.
This is an Amdahl upper bound for these measured workloads, not a universal bound
on algorithms that also change iteration counts or matrix operations.

The crossing IV case is essentially tied on this machine; crossing PPML is
slower natively. Repeated-cell and small-Schur cases are the robust opportunities
in this corpus. The prior macOS weak-community IV-PPML result reversed the Linux
advantage. Preserve explicit opt-in selection; do not infer a universal router
or a tenfold gain from these measurements. The weak case also compresses repeated
cells. Weak instruments, near-separation, and large-scale simplex/ReLU costs
remain outside this performance claim. The recorded RSS high-water mark is
process-lifetime cumulative, not an isolated per-engine memory comparison.

Reproduce each row with, for example:

```bash
python benchmarks/hdfe/ppml_end_to_end_profile.py --rows 1000000 \
  --case weak --model ivppml --repeat 3 --seed 20261028 \
  --budget-mb 2048 --output weak-ivppml.json
```

The nonlinear CI profile now explicitly uses this second seed, with one repeat
per engine on each runner. Do not pool those single-run CI timings with the
three-repetition local medians above.
