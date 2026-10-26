# Validation policy

Scope: standalone positive-weight categorical FE projection. Never infer a
production estimator guarantee from these microbenchmarks.

1. Performance cases have at least 1,000,000 observations. Record FE levels,
   unique full tuples, weight distribution, eliminated dimension, local block
   sizes, coarse dimension, factor nnz and RSS; rows alone are not difficulty.
2. Pin the native within version. Compare real Adaptive, Diagonal and Additive
   configurations where relevant. A local reimplementation is not native within.
3. Count input conversion, structure detection, preparation, factorization,
   coarse setup, solve and reconstruction. Exclude generation/JIT/audit explicitly.
   Run processes sequentially; report setup and reuse separately.
4. Check a planted exact residual and all columns' normalized normal equations
   at 1e-8. Add an independent full rectangular reference for all RHS where
   feasible. Solver flags/tolerance numbers are not forward-error certificates.
5. Preserve errors, resource refusals, slow cases and stricter-baseline reruns.
   A rejected specialized path plus native fallback is not a specialized win.
6. Freeze parameters before holdout tests. Record every development threshold;
   changing a seed is not the same as changing a data-generating mechanism.
7. Three isolated repetitions are required for timing comparisons. PR packaging
   smoke checks may run once, but cannot create new speed claims.
8. No assertions about Stata/R DoF conventions from the shared CR0 calculation.
   Coefficients, covariance, nearly absorbed X, weight updates and actual estimator
   integration need their own independent acceptance before production use.
