# Linear IV correctness closeout (2026-09-30)

Selected base: PR #14, `a9384e084a8bdb596bf6960a58b3951dfbd320dc`.
Keep its 4 MiB call-local covariance reuse. Do not layer the alternate 8 MiB
workspace/content-hash cache onto this implementation.

## Numerical and output policy

Normalize X'WZ by weighted column norms. Singular values below
`eps * max(n, k, l) * max(1, smax)` denote numerical underidentification;
values at or below `sqrt(eps) * max(1, smax)` fail with NumericalError.
The latter is conservative numerical admission, not a weak-IV test or a
universal accuracy guarantee. A further solve-rank check rejects silently
truncated coefficients. Tests retain column-unit invariance.

Sensitive 2SLS systems (normalized moment ratio < 1e-3) solve the whitened
instrument moments by QR/SVD, rather than forming X'PzX. Cholesky failure
is explicit. This does not recover precision lost in original moments or
promise reliable inference arbitrarily close to nonidentification.

`first_stage["fitted_endog"]` uses the unweighted, FE-residualized coordinate
system even in weighted estimation. Fixed k-class covariance uses
`H=(1-kappa)X+kappa PzX`: IID uses sigma² bread H'H bread', and robust/cluster
use H_i u_i scores. kappa=0 recovers OLS. LIML's conventional covariance
path remains unchanged. This is conditional fixed-kappa inference, not an
additional consistency claim for arbitrary endogenous-regressor estimators.

## Validation and limits

The recovered cloud audit was rerun on synthetic observations. At delta=1e-6,
structured/dense coefficient disagreement fell from 0.00639749 to 4.26e-9;
delta=1e-8 now raises NumericalError and delta=0 raises UnderidentifiedError.
Weighted first-stage disagreement fell from 1.90171649 to 6.22e-15.
kappa=0 robust endogenous SE is 0.00982514900281733, matching OLS.
The regression suite contains independent observation-space sandwich and
exact Fraction moment oracles. No licensed-Stata or real-data benchmark
was run. Full CI must validate the new commit independently of old PR CI.

Remaining resource issues: `diagnostics="off"` still computes established
results; first-stage fitted arrays and compatibility diagnostics remain dense;
KP fallback may allocate full scores; session memory has no global eviction.
The exactly identified overidentification fix removes one allocation only.
