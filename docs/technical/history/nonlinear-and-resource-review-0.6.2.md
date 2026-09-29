# Nonlinear equations, numerical contracts and resource boundaries — 0.6.2

This is a local implementation review, not an originality certificate, third-party
peer review, Stata certification, or proof of floating-point accuracy for all inputs.
The three 0.6.1 mathematical manuscripts are retained unchanged; this follow-up
adds explicit nonlinear references and exercises their implementation boundaries.

## Independent IV-PPML reference

On a small sample, form full dummy designs `T=[C,E,D]` and `Q=[C,Z,D]`, with
strictly positive observation weights `w` and offset `o`. Define

```
mu(theta) = exp(T theta + o)
Omega(theta) = diag(w * mu(theta))
Gamma(theta) = (Q' Omega Q)^(-1) Q' Omega T
H(theta) = Q Gamma(theta).
```

For exactly identified, full-rank systems, the reference solves
`Q' diag(w) (y-mu(theta)) = 0` directly. Its Jacobian is
`-Q' Omega(theta) T`; central finite differences independently check this
formula. No econhdfe projection, IRLS, IV solver or covariance routine is used.

For overidentified systems, the implemented IRLS-IV fixed point instead solves
`Gamma(theta)' Q' diag(w) (y-mu(theta)) = 0`. One must NOT require each component
of the overidentified raw sample moment vector to be zero. The reference solves
this square projected-score system independently using SciPy `root` [S1].

The reported covariance uses the existing frozen-IRLS/upstream convention:
`A=H' Omega T`, with bread `A^(-1)` and scores `h_i*w_i*(y_i-mu_i)`.
Frequency-weight robust meat contains one power of the replication count;
probability-weight robust meat contains squared weighted scores. The existing
finite-sample factors are `N_eff/(N_eff-1)` for robust and `G/(G-1)` for cluster
subsets. These are IV-PPML conventions, NOT a substitution of OLS small-sample DoF.
For exactly identified systems this agrees algebraically with the usual moment
sandwich. For overidentified systems this is not a claim that the finite-sample
Jacobian of `Gamma(theta)'M(theta)` has no `dGamma'M` term. Under the identifying
population moments, that term vanishes in the limiting Jacobian; the tests check
the implemented convention, not finite-sample exact coverage or weak-IV validity.

Twelve cases cover dense/block execution, no/probability/frequency weights,
and exact/overidentification. Separate tests cover both engines, clustered
covariance, physical frequency replication, probability-weight rescaling,
standardization, first-stage units, and constant-offset shifts. Ordinary PPML is
compared with an independent explicit-dummy GLM and separately formed Hessian
and sandwich; PPML's positive observation weights are not advertised as a new
`weight_type` API. Agreement in these cases is numerical evidence, not global
existence or uniqueness of a nonlinear root.

## Confirmed implementation repairs

1. **IV-PPML covariance and units.** The point-estimation solver already retained
   the appropriate equilibrated directions, but VCE rebuilt an unscaled Gram
   pseudoinverse and could discard one. Dense and block final VCE now reuse the
   solver bread. Fallback Gram inversion is equilibrated. Changing an endogenous
   unit by 4e7 previously reduced the other slope's standard error from about
   0.0308 to 4.1e-18 on the saved deterministic probe.
2. **Cluster validity and labels.** No dimension with fewer than two observed
   groups can silently become robust inference or be dropped from CGM
   inclusion-exclusion. Labels are factorized on the actual estimation sample
   before DoF and VCE. Strings, shifted integers and fractional labels represent
   categories, not array offsets. This is shared by dense/block PPML and IV-PPML.
3. **Offset-consistent initialization.** The initial positive `mu` is a guess
   of the total mean, so initialize total `eta=log(mu)`, not `log(mu)+offset`.
   The working response still subtracts offset as required. Previous dense and
   block initialization violated `mu=exp(eta)` and could diverge after an offset
   constant was compensated by the intercept.
4. **Divergence and coordinates.** A large raw coefficient can be harmless after
   changing regressor units. The existing large-coefficient guard now measures
   `X beta`; finite eta/mean and the other stopping guards remain. This is not a
   promise of convergence or permission to ignore genuinely explosive predictors.
5. **Original-unit first stages.** With `X*=X/Sx`, `Q*=Q/Sq`, the reported
   first-stage matrix is `Gamma = Sq^(-1) Gamma* Sx`. Both dense and block
   diagnostics now apply this map and label their units explicitly.
6. **Truthful result semantics.** Historical IV-PPML aliases `None/model/iid/
   unadjusted/homoskedastic` compute robust VCE and now report `vce='robust'`.
   The original request remains in diagnostics. They do NOT introduce a new
   model-based IV-Poisson covariance. The normalized `_cons` retains its stored
   zero SE for compatibility, but has no reported CI or p-value.
7. **Block numerical consistency.** PPML block VCE uses equilibrated Gram bread.
   Block 2SLS reports numerical rank from the same scaled eigen-directions and
   threshold as its bread, not an unrelated raw-unit `matrix_rank` call.
8. **Input/resource contracts.** Iteration budgets must be integers in their
   declared ranges. The dense-integer fast path only calls `bincount` when
   `max(label)<N`, a necessary condition for contiguous observed codes starting
   at zero. Sparse labels such as 1e9 no longer request a billion-slot array.

## Mathematical assumption and fallback checks

| Mathematical object | Required boundary | New evidence |
|---|---|---|
| Exact categorical rank | Finite categorical intercept design, observed levels only; `rank_Fp <= rank_Q <= U` | Actual GF(2) XOR budgets 0/1/2 plus bad prime 2 force integer fallback on the parity example; result remains exact rank 4 |
| Optional exact backend | A modular shortfall is not the characteristic-zero result | Installed SymPy is exercised with prime 2, where modular rank is 3 but integer rank is 4 |
| Resource failure | Failure must not be reported as an exact result | Injected `MemoryError`/`RuntimeError` escape the native exact backend; FLINT cell guard rejects before importing the absent dependency |
| Weighted residual projection | Strictly positive weights, proper FE support, actual solver accuracy | Core on/off, MAP plain/CG and LSMR; explicit weighted least-squares, orthogonality, idempotence, exact-span annihilation and weight update |
| QR/SVD compression | Scaled coordinates and consistent numerical rank; rank-deficient coordinates require a declared solution rule | Tall/wide/empty/zero-column/rank-deficient inputs, chunk sizes 1/9/17/32768; reference scaled pseudoinverse and unchanged input buffers |
| Partition/refinement identities | Active/reference columns and identical multipliers | Prior 0.6.1 independent tests remain in the full suite and installed-wheel verification; no new broader theorem is claimed |

The existing 4,095-support exact enumeration is retained; it is not counted as
4,095 newly added pytest cases. The new projection tests use a finite weight
ratio of 1,000, not arbitrary conditioning. A failed exact backend is not silently
converted to floating-point rank, but there is still no universal time/memory
budget for every native rational-elimination workload. Optional FLINT execution
is not certified merely because its pre-import budget guard was tested.

## Stable performance change

The dense equilibrated least-squares routine still compresses `[X/scale,y]` by
orthogonal QR and uses one SVD to define beta, bread and rank. It now allocates
one Fortran-ordered buffer and asks LAPACK/SciPy for raw reflectors plus compact
R, avoiding `block + vstack + full-height R` allocations. SciPy documents the
raw compressed R shape [S2]. No unscaled normal-equation shortcut is restored.
The Gram helper uses a common eigen-decomposition and the existing 1e-15
relative pseudoinverse cutoff [S3], so its numerical rank and inverse agree.
Neither optimization can recover information already lost by forming a Gram
matrix or remove genuine weak identification/near-collinearity.

## Sources and historical boundary

- [S1] SciPy, `scipy.optimize.root`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
- [S2] SciPy, `scipy.linalg.qr`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html
- [S3] NumPy, `numpy.linalg.pinv`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
- [S4] Upstream IV-PPML implementation, consulted for algorithm/convention context, not a licensed execution: https://github.com/ekwonomist/ivppmlhdfe/blob/main/ivppmlhdfe.ado

Official web documentation was checked on 2026-09-14; actual installed versions
are separately recorded in validation evidence. Previously reviewed mathematical
references and originality qualifications remain in `mathematical-review-0.6.1.md`.
This follow-up does not complete a new exhaustive literature-priority search.
