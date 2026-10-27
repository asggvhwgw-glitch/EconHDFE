# Main integration acceptance — 2026-10-27

Reviewed PR30 at `b469dbfcf9488b0809aad906a92b1e73258b625d` against
main `54402cc4184fa15a9c7a53f957ca76bc8708609c`, including merged PR28/29.
Integrated main; regenerated the architecture maps to resolve their conflicts.
The within backend and coefficient-bootstrap production files match main.

## Functional findings and correction

The shared peeler implements integer frequency multiplicity without expanding
rows, preserves explicit `stata_compat` defaults, and separates unified
singleton/separation reasons. User weights remain distinct from IRLS weights;
cache keys include the policy. Unsupported unified group/individual and
IV-PPML regressor-separation combinations remain explicit errors.

Additional frequency-expansion acceptance exposed a real PPML multiway
covariance discrepancy: coefficients and fitted means agreed, but the
covariance depended on solver standardization. Explicit weighted dummy
projection and independently aggregated CGM scores identified an indefinite
raw sandwich (minimum eigenvalue about -1.71945e-5). Repairing it in standardized
coordinates disagreed with the original-unit oracle by 1.25507e-7.

Dense and block PPML/IV-PPML now defer eigenvalue clipping until covariance is
in reported coefficient units. Existing finite-sample corrections are unchanged.
Internal helpers keep their default PSD repair for other callers.

The four new regression cases compare against an independent dummy/CGM oracle,
exercise genuinely negative eigenvalues and cross-component clusters, and
check standardization and frequency expansion. All four fail on the original
PR30 code and pass after the correction.

## Executed validation

Environment: research Python 3.12.15, macOS arm64, NumPy 2.5.4, Numba 0.68.0;
local native extension previously built and validated from the merged PR29.

- `python scripts/run_tests.py --suite full -- -q`: **526 passed**, 4 expected warnings.
- `REQUIRE_WITHIN_NATIVE=1 python -m pytest native/within-accel/tests -q`:
  **53 passed**, no skips.
- `python scripts/version.py gate`: **0.10.0**, passed.
- `python scripts/generate_architecture_map.py --check`: passed.
- Four 1,000,000-row sample-peeling cases: all known retained sets reproduced.
- Sixteen default-call comparisons with current main: exactly equal for
  coefficients, covariance, sample/DF and applicable fitted outputs.
- Nineteen additional combinations: all passed after correction, covering
  two seeds, four model families, robust/two-way cluster covariance with offset,
  frequency expansion, native within, coefficient bootstrap and 160 randomized
  peeler comparisons with an independent integer fixed-point oracle.

Local task archives retain the initial failure, before/after regressions,
independent scripts, JSON outputs and JUnit records. Million-row checks cover
sample selection only; no new million-row end-to-end speed claim or licensed
Stata numerical certification is made. This remains a 0.10.0 source candidate;
no package or release tag is published by this acceptance.
