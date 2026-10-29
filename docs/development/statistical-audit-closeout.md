# Statistical audit closeout — 0.9.4 candidate

This change follows the 2026-10-29 audit of main
`44dae3f41f978a3ce78e0d780feab42b49a352f3` (0.9.2) and the first PR #38 head
`23d0d6f125ff197641f68d35f5a5645540ab7086` (0.9.3).
The 0.9.3 contract snapshot is preserved; the follow-up is a new 0.9.4 candidate
on the same PR. A candidate version does not publish a release.

## What was corrected

| Finding | Previous behavior | Corrected contract |
| --- | --- | --- |
| Exact IV nonidentification | A roundoff-sized first-stage fitted column could be rescaled and reported as rank one, with a successful zero-SE estimate | Check dimensionless instrument/regressor cross-space rank before rescaling the fitted design; raise `UnderidentifiedError` when the current weighted system loses coefficient identification |
| IV sample selection | Default compatibility mode applied ordinary PPML regressor separation even when additive IV moments had a unique finite solution | Default to FE-zero checks in both modes; explicit legacy PPML separation warns in compatibility mode and remains rejected in unified mode |
| Default PPML outcome units | The outcome-SD floor at 1e-3 and absolute convergence safeguards produced different slopes after changing outcome units | Use a positive weighted-mean outcome unit in both standardization modes, including structured paths |
| PPML intercept direction | Rank pruning could remove the generated intercept when supplied columns already spanned it, so reported coefficients missed the outcome-unit intercept shift | Preserve the generated constant direction, omit a redundant user column, and use its numerical index for unit restoration, Wald and PSD handling |
| Final IV inference units/state | Converged means were restored to user units before a final weight floor and coefficient-only re-solve | Keep normalized units through final projection, VCE, EE and J; report the jointly accepted IRLS coefficient/mean state |
| Full multiway covariance | Inserting the separately clipped slope block into a full spectral clip could leave a negative full-matrix eigenvalue | Preserve the repaired slope block and complete the constant block using the PSD Schur-complement condition |
| PPML finite-sample convention | A residual-DF correction was implicitly used despite upstream-compatibility claims | Explicit `InferenceConfig.ppml_small_sample`: upstream asymptotic factors by default, earlier residual-DF factors available by request |
| Inference scope claims | Broad statements could imply EE or ordinary PPML sandwich inference resolves all misspecification or many-FE bias | State identification, valid-moment/pseudo-true-value and growing-FE boundaries explicitly |

The new regression files are
`tests/numerics/test_iv_audit_fixes.py`,
`tests/numerics/test_ppml_audit_fixes.py` and
`tests/numerics/test_covariance_audit_fixes.py`.
Existing independent references remain in the full suite.

## 1. Identification has to concern the cross moments

Consider four observations with
$$
x=(-1,-1,1,1),\qquad z=(-1,1,-1,1),\qquad y=(1,1,1,1).
$$
For every slope `b`, the intercept `a=-log(cosh(b))` makes
`[1,z]' (y-exp(a+b*x))=0`. There is a continuum of solutions, even though
`x` and `z` separately have full column rank.

The old solver could project `x` onto `z`, obtain machine-roundoff noise,
then normalize that noise into a nonzero fitted column. Both sample modes
returned convergence, a slope of zero and a slope SE of zero. A rank check
only on the rescaled fitted design cannot prevent this failure.

The dense fix checks the first-stage projection against the original weighted
regressor norms, which is equivalent to checking its coordinates in an
orthonormal instrument basis. The structured path uses bounded-row QR
compression of the weighted augmented instrument/regressor/response design.
Both reject numerically absent coefficient directions before fitted-column equilibration
and retain the final solve's rank check.

Cross-review also exposed a concrete near-collinear block-solve failure:
with self-instrumentation `X=Z=[a,a+2e-7*b]`, orthogonal equal-length `a,b`
and response `a+1e-7*b`, the coefficients must be `(0.5,0.5)`. A normal-Gram
solve could err by several percentage points. Applying an observation-space
rank threshold to `Z'WX` additionally squared the conditioning and falsely
rejected the block design. Bounded QR compression avoids both problems while
preserving the original weighted least-squares geometry. Dense relevance
checks reuse the first-stage projection, normalized by the original regressor
norms rather than its own potentially roundoff-sized fitted norms.

This is a numerical identification guard. It does not establish strong
identification, global uniqueness of nonlinear roots, or weak-IV-robust
confidence sets. A singular intermediate weighted system is reported as a
failure of that solve, not proof about every possible nonlinear root.

## 2. Ordinary PPML separation is not IV existence

For
$$
x=(0,0,1,1),\quad z=(0,2,1,3),\quad y=(1,3,0,0),
$$
write the two fitted means as `mu0` and `mu1`. The intercept and instrument
moments give
$$
2\mu_0+2\mu_1=4,\qquad 2\mu_0+4\mu_1=6.
$$
Their unique positive solution is `mu0=mu1=1`, or intercept=slope=0.
Ordinary PPML nevertheless has a separating direction that sends the
`x=1` rows' means to zero. Dropping those observations changes the IV problem.

Accordingly `IVPPMLConfig(separation=None)` now resolves to `("fe",)` in
both sample modes. All-zero intercept-FE levels retain their valid boundary
interpretation. This is not an exhaustive existence test for nonlinear IV.
Explicit `simplex`, `relu` or `mu` requests in compatibility mode emit a
warning that their PPML rule does not certify IV existence; unified mode
continues to reject them. Ordinary PPML's separation defaults are unchanged.

## 3. Outcome units and one consistent fitted state

With an intercept or intercept FE, multiplying `y` by a positive constant
`c` must leave slopes unchanged and shift the index by `log(c)`. The mean
scales by `c`. A robust/cluster sandwich remains unchanged because its
bread and scores compensate for the unit change. A model-based Poisson
inverse Hessian instead scales by `1/c`; the regression tests keep these
two covariance contracts separate.

The seed-294 ordinary-PPML example in the audit had a baseline slope near
0.60170993. The first PR #38 head reported approximately 0.38564944 after
multiplication by `exp(-36)` and 0.01167697 after `exp(-40)`, while claiming
convergence. Fixing only `standardize=False` was insufficient.

Both PPML modes now use the weighted mean of the retained outcome as a
positive unit, with no absolute lower bound. Mean computation uses bounded
outcome and weight coordinates to avoid unnecessary intermediate overflow.
Regressor standardization remains a separate choice.

In a no-FE fit, an explicit user constant or a complete set of categorical
dummies can already span the intercept. Rank pruning now preserves the
known generated constant and omits a redundant user column instead. Otherwise
the former code could drop `_cons`, omit the `log(y_scale)` coefficient
restoration, and return coefficients that did not reconstruct `eta`.
The actual constant index, rather than a user's column label, now governs
unit restoration, the model Wald test and full-covariance completion. The
retained model column space is unchanged; the omitted redundant column can
change. Regression checks cover both standardization settings and clustered
covariance.

IV-PPML also retains these normalized units through its final projection,
standard covariance, EE covariance and overidentification calculation.
Previously a final `max(w*mu,1e-20)` in restored user units distorted relative
weights at small units. On the seed-1107 audit data the slope changed from
about 0.46673037 to 0.19601258 at `exp(-50)`, even though normalized returned
means had not changed.

The last projection is now used for inference at the accepted IRLS state.
It cannot replace the slope while leaving the previous `eta` and `mu`.
Scores use the structural residual at that same state. Public-unit conversion
occurs only after inference. Tests cover the original extreme-scale cases,
offsets, weights, dense/structured paths and the within-FE identity of
`eta-X*beta-offset`.

Finite floating-point range still bounds representable inputs and outputs.
Neither this repair nor numerical convergence certifies a model's economic
identification assumptions.

## 4. A coherent full multiway covariance

A signed Cameron–Gelbach–Miller covariance can be indefinite in a finite
sample. The issue here was that an advertised PSD repair still returned an
indefinite full matrix.

For example, starting from `[[1,2],[2,1]]`, a full spectral clip is
`1.5*ones((2,2))`. Replacing only the slope variance by its separately
repaired value `1` gives determinant `-0.75`.

The new repair works in the existing regressor-stdev coordinates. Let `F`
be the full spectral clip and `A` the independently repaired slope block.
Set
$$
b=P_{\operatorname{range}(A)}F_{s,c},
\qquad d=\max\{F_{c,c},b'A^+b\}.
$$
A small upward rounding guard is applied to the bound. Then
`[[A,b],[b',d]]` is PSD: the cross-covariance lies in the range of `A` and
the Schur complement is nonnegative. Unsupported numerical-null directions
are projected out. Existing post-estimation checks stay active; a singular
contrast system can still be rejected.

The actual seed-41 two-way PPML example formerly returned variance
`-0.001418325094` for the meaningful index contrast `_cons-1.75*x`, with
`-1.75` inside the observed regressor range. The new public regression checks
PSD, a usable positive contrast variance, and preservation of the positive
raw slope variance using independently aggregated cluster scores.

This is a declared completion convention, not a claim of a nearest-matrix
optimum, exact full-matrix Stata agreement, or corrected finite-sample
coverage. The upstream regressor-stdev floor at `1e-3` is retained; arbitrary
regressor rescaling across that floor is outside the invariance claim.
The new full covariance invalidates persisted linear result caches.

## 5. Make the finite-sample convention explicit

For ordinary PPML:

| Option | Robust multiplier | Cluster multiplier |
| --- | --- | --- |
| `ppml_small_sample="asymptotic"` (default) | `N/(N-1)` | `G_min/(G_min-1)` on the combined multiway meat |
| `ppml_small_sample="residual"` | `N/max(N-K,1)` | The previous residual-DF/nesting multiplier |

`N` is frequency effective N for fweights. `K` in the residual convention
includes retained slopes and counted absorbed DoF; the existing nested
adjustment is preserved. The model-based inverse-Hessian path is unaffected.
`diagnostics["finite_sample_correction"]` records the choice and applied
multiplier. This choice is independent of singleton policy.

The audit's n=120, 40-FE example differed by `sqrt(119/79)`, about 22.7% in
SE, entirely because of these conventions. The residual correction is not
universally statistically wrong; it was wrong to imply it reproduced the
current ppmlhdfe convention without qualification.

IV-PPML retains its distinct upstream per-subset `G/(G-1)` cluster
corrections. Do not substitute ordinary PPML's common minimum-cluster factor
for that implementation.

## 6. EE, moment tests and growing fixed effects

The IV estimator's profiled equation has the form
`F(theta)=Pi(theta)' m(theta)`. Its derivative includes
`(dPi)'m + Pi'dm`. Under valid population moment restrictions the first
term vanishes at the true parameter; under misspecification it need not
vanish at the pseudo-true parameter. This is why the first PR's optional
stacked-EE covariance remains useful and is retained.

An independent nine-parameter reference estimates the slope, three FE
parameters and five first-stage projection coefficients, then differentiates
the full stacked scores. That reference already validated the EE construction
in the audit. It is not a universal coverage theorem.

`ivppml_misspecification_robust=True` requires regular identification,
appropriate moments, dependence assumptions and an applicable asymptotic
sequence. It targets uncertainty around this estimator's pseudo-true value,
not a causal coefficient made valid by repairing invalid instruments.
The dense J diagnostic is an efficient-step moment Wald/Hansen construction.
Raw overidentifying sample moments need not all vanish at the fitted
projected-IV equation; their nonzero values alone do not prove nonconvergence.
Direct BlockDesign execution still does not provide J/EE; requesting EE
through the structured DataFrame frontend uses the supported dense path.

Ordinary PPML's sandwich does not omit an IV first-stage term. That does not
mean all PPML inference biases disappear. In growing-FE three-way gravity
models, consistent slopes can coexist with biased centering and cluster-VCE
bias. The relevant correction depends on the panel design and assumptions.
The IV-PPML SPJ/bootstrap tools also have design-specific conditions. See
[Weidner and Zylkin, Bias and Consistency in Three-way Gravity Models](https://arxiv.org/abs/1909.01327)
and the [authors' correction implementation](https://github.com/tomzylkin/ppml_fe_bias).

## Executed candidate validation

The final integrated suite passed **588 tests**, with **one optional native
extension skip**. The three new audit files contribute 76 cases. Version,
compatibility, architecture-layer and generated-map checks passed. Evidence
and environment details are recorded in
[`statistical-audit.json`](../release/evidence/0.9.4/statistical-audit.json).

Re-running the original audit scripts gives:

- PPML slope 0.60170993287074 at all original tested outcome units, with five
  iterations in both standardization modes.
- IV-PPML maximum slope change below 2e-16 through the original scaling range
  down to exp(-60), with seven iterations in every case.
- The independently differentiated nine-parameter EE reference agrees in SE
  to about 1.8e-15 relative error in the retained fixture.
- The previously negative PPML index-contrast variance is now
  0.01285244910686 under the new default factor. Its full covariance is PSD;
  a joint restriction involving its null variance direction is still rejected.
- Default PPML SEs match the independent upstream-convention finite-sample
  references to floating-point precision; residual-DF opt-in has separate
  independent dummy/sandwich tests.

These checks describe the cases executed, not universal guarantees. Source
candidate execution is distinct from CI support-matrix validation and formal
release acceptance.

## Upstream provenance and validation boundary

The audit consulted pinned primary sources:

- [reghdfe 6.14.1, 29b2b203](https://github.com/sergiocorreia/reghdfe/tree/29b2b203f534413369d675d0c0f674d61291f536).
- [ppmlhdfe 2.3.4, b85665d8](https://github.com/sergiocorreia/ppmlhdfe/tree/b85665d8f674e93c1e07446a9a2b29be7f797910).
- [ivppmlhdfe 0.9.4, 4d8ac171](https://github.com/ekwonomist/ivppmlhdfe/tree/4d8ac171adf71d137c101d8d7387d047bf0ad5aa).
- [reghdfe issue #225](https://github.com/sergiocorreia/reghdfe/issues/225),
  which explains the slope/constant repair lineage.

Current ppmlhdfe uses reghdfe5 and therefore does not automatically inherit
reghdfe v6 fixes. Historical Stata factor-variable loading and prediction/
margins fixes are not new Python runtime defects. Unsupported Python margins
cannot be presented as repaired by this PR.

No licensed Stata executable was run for this follow-up. Validation uses
analytical moment systems, independent explicit-dummy/sandwich constructions,
metamorphic unit checks, the full repository gate and CI attached to the new
head. Historical Monte Carlo figures in the original PR are not new runs.
No solver novelty, universal bias correction or blanket Stata parity is claimed.

