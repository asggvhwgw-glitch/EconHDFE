# Weak identification and growing fixed effects in additive IV–Poisson inference

Research checkpoint: 28 October 2026. Parent: the local overidentified `0.10.1.dev1` source artifact, not current `main`. All new inference is opt-in. No estimator default, existing variance estimator, or released version is changed. This note gives conditional derivations and reproducible experiments; it does not claim peer review or general finite-sample validity.

## 1. The estimand and two distinct problems

Write

\[
Y_i=\exp(o_i+E_i'\gamma_0+U_i'\delta_0)+u_i,
\qquad E[a_iU_i u_i]=0,\quad E[a_iZ_i u_i]=0.
\]

`E` contains every potentially weakly identified target slope; `U=[D,C]` contains categorical FE and regular included controls; `Z` contains L excluded instruments. `a` denotes positive base weights. The distribution need not be Poisson. Endogeneity is permitted within an observation. This is the additive-error moment model, not the multiplicative-error model and not the misspecified pseudo-true root of the parent projected-IV algorithm.

Weak target identification and incidental-parameter bias are logically different. A test avoiding inversion of the target Jacobian can still fail because its moments are biased after estimating many FE. A bias-corrected point estimate can still have a nonnormal weak-IV distribution. Consequently we correct/test null moments, rather than run a point-estimate SPJ followed by a Wald interval.

The literature gives the appropriate context, not a blanket certification of the code: Stock and Wright (2000) develop weak-identification-robust GMM confidence sets; Jochmans (2022) establishes inconsistency and long-panel inference bias for FE IV count estimators; Kwon et al. (2026) implement coefficient SPJ and bootstrap for IV-PPML-HDFE. Ordinary three-way PPML consistency results of Weidner and Zylkin (2021) cannot simply be transferred to IV moments. The moment-SPJ, cross-product, and balanced-circuit constructions below are derived for the stated models and assumptions; no novelty claim is made.

## 2. Null-restricted S test with regular nuisance parameters

For H0: gamma=gamma0, fit only delta by

\[
U'\operatorname{diag}(a)(Y-\widehat\mu_0)=0,
\quad \widehat\mu_0=\exp(o+E\gamma_0+U\widehat\delta_0).
\]

There is no unrestricted IV estimate or first stage in this computation. At the true null let

\[
A_U=E[a\mu UU'],\quad \Pi=A_U^{-1}E[a\mu UZ'],\quad
Z^\perp=Z-\Pi'U.
\]

In a normalized identified nuisance space, suppose A_U is nonsingular, delta is regular, Taylor remainders are negligible, and a moment CLT holds. Expanding the restricted nuisance score and then the instrument moments gives

\[
\sqrt n\,\widehat m(\gamma_0)
= n^{-1/2}\sum_i a_i Z_i^\perp u_i+o_p(1).
\]

Indeed delta-hat minus delta equals A_U inverse times the nuisance score to first order; substituting this into m subtracts the nuisance-score projection. In the sample the W-weighted projection of Z onto U, W=diag(a mu-hat), estimates this projection. Its score totals equal the raw Z moment totals up to the certified numerical nuisance-score error.

Let s=sum_i a_i Z_i(Y_i-mu-hat_i), and let V_s estimate the covariance of this sum using scores psi_i=a_i Z_i-perp(Y_i-mu-hat_i). Then

\[
S(\gamma_0)=s'\widehat V_s^{-1}s\ \Rightarrow\ \chi^2_L.
\]

For independent clusters sum the observation scores within each cluster before forming their covariance. Arbitrary within-cluster dependence is allowed only insofar as the nuisance expansion and cluster CLT are valid. There must be enough independent clusters; this implementation does not claim a few-cluster correction. For frequency weights the independent-observation meat reproduces explicitly replicated data, rather than squaring multiplicity. Probability-weight validity still requires a defensible weighted sampling/moment model.

**Why weak identification is accommodated.** No inverse of E[a mu Z-perp E'] appears. With uniform nuisance regularity, moment bounds, positive covariance, negligible remainders and a uniform CLT, the result extends to sequences of weak or zero target relevance. A singular moment covariance is different from a singular target Jacobian: the former prevents this chi-square calibration and is explicitly rejected.

**Degrees of freedom.** The target is fixed, while the regular included controls are partialled out; the statistic uses L excluded moments. It is not an overidentification J test evaluated at an estimated K-dimensional target and must not use L-K degrees of freedom. The deliberately incorrect L-K comparison in the experiments is a negative control, not a claim that the parent's separate overidentification test has this bug.

**Nuisance caveat.** Unknown additional endogenous slopes cannot be plugged in as though they were strongly identified controls. They must be included in the joint null, or a properly constructed joint confidence set must be projected. The current implementation supplies point-null tests and scalar grid inversion, not a certified multidimensional projection optimizer or a CLR/KLM procedure.

`RestrictedPoissonS` freezes the sample, checks the null nuisance scores, and does not silently drop observations as beta changes. Infeasible/nonconverged null fits, all-zero FE levels, and singular covariance produce an error or no p-value; confidence-grid inversion labels these points **unknown**, not rejected.

## 3. Confidence sets, not automatically bounded intervals

The continuum S confidence set is {gamma:S(gamma)<=chi-square(L,1-alpha)}. It may be disconnected, unbounded, empty, or uninformative. Weak-ID validity does not imply power. A zero-relevance model should not be forced to produce a short confidence interval.

`invert_scalar_grid` evaluates only an explicitly supplied finite grid. It preserves disconnected accepted grid runs, failed evaluations, and acceptance at the search boundaries. It does not interpolate certified endpoints, fill gaps, claim that boundary acceptance proves infinity, or interpret an empty grid as proof of an empty continuum set. Simulation of point-null size is not a simulation of a numerical grid's coverage when the true value need not be on the grid.

## 4. What growing FE require beyond the fixed-nuisance argument

For population nuisance eta and projection Pi, the score

\[
\psi=(Z-\Pi'U)a\{Y-\exp(o+E\gamma+\eta)\}
\]

is Neyman-orthogonal at the true null: the derivative in eta is killed by the W projection, and the derivative in Pi is killed by E[aUu]=0. This eliminates a first-order *population* nuisance bias, not all sample overfitting or nonlinear incidental bias.

A useful sufficient-condition statement is the following. Suppose the oracle score satisfies a uniform CLT with well-conditioned covariance; exponential Taylor expansions and moments are uniformly controlled; the score's empirical-process remainder is negligible, either by a valid independence-respecting cross-fit or a separate empirical-process bound. Let r_eta and r_Pi be appropriate prediction-norm errors. Then

\[
\sqrt n\{r_\eta^2+r_\eta r_\Pi\}=o_p(1)
\]

is a sufficient second-order remainder condition. Proof: expand the exponential to second order; orthogonality removes the mean-linear term; bound the remaining quadratic eta term and bilinear eta–projection term by their prediction norms, with the stipulated moment/envelope bounds. Add the controlled empirical-process remainder and apply Slutsky to the oracle CLT.

For the illustrative rates r_eta,r_Pi=O_p(sqrt(d_n/n)), this requires d_n=o(sqrt(n)). This is a conservative sufficient rate, **not a necessity, not a theorem for every FE graph, and not a diagnostic automatically certified by small d_n/n**. Weakly connected designs and little support per FE can invalidate the rate premise. With clusters of increasing size the standardized remainder must be compared to the actual moment covariance; mechanically replacing n with a cluster count is not justified.

Cross-fitting is not a free cure here. Leaving a complete unit out can leave its own FE unidentifiable, while splitting dependent observations of one unit into folds does not yield independent training/evaluation errors. The present code does not claim to implement a universal FE cross-fit.

## 5. One-way panels: bias correction at the null-moment level

Consider independent units g=1,...,G, each observed for T periods for this rate discussion. Fix **all slopes** at beta0; any controls not tested must have known coefficients included in the offset. Write r_gt=exp(o_gt+X_gt'beta0), Y_gt=a_g r_gt+u_gt. The extended one-way profiled FE is a-hat_g=Y_g/R_g, including a-hat_g=0 for all-zero outcome units. The cluster-average moment is

\[
s_g=\overline{ZY}_g-\frac{\bar Y_g}{\bar r_g}\overline{Zr}_g
=\overline{Zu}_g-\frac{\overline{Zr}_g}{\bar r_g}\bar u_g.
\]

Under a stable within-unit iid or suitably controlled mixing expansion, bounded-away denominators and sufficient uniform moments,

\[
E[s_g]=b_g/T+c_g/T^2+o(T^{-2}).
\]

For conditional iid observations the leading coefficient is

\[
b_g=-\frac{E[(Z-\zeta_g)r u\mid g]}{E[r\mid g]},
\qquad \zeta_g=E[Zr\mid g]/E[r\mid g].
\]

This follows by expanding the ratio Zr-bar/r-bar around zeta_g; the leading ratio fluctuation is avg((Z-zeta_g)r)/E[r|g], whose product with u-bar has mean E[(Z-zeta_g)ru|g]/T. Endogeneity generally makes it nonzero.

With a predetermined, equal-sized split into two halves (at least two observations in each), form

\[
s_g^{bc}=2s_g-\tfrac12(s_{g,1}+s_{g,2}).
\]

If the same b_g describes both half panels, the 1/T bias cancels, leaving O(T^-2). The covariance must be computed from these **joint combined unit scores**. Full and half samples overlap; treating their estimates or scores as independent is wrong. The leading random term remains the full-sample average under the regular expansion, so Var(s_g^{bc}) is of order 1/T in the weak-dependence balanced case.

Consequently the raw score's standardized bias has order sqrt(G/T), while the corrected score's remaining standardized bias has order sqrt(G/T^3). Useful sufficient rates are

\[
G/T\to0\quad\text{(raw profile)},\qquad
G/T^3\to0\quad\text{(moment SPJ)},
\]

along with the uniform expansion, remainder, covariance and CLT assumptions. More generally one must require sqrt(G) Sigma_T^-1/2 E[s_g^{bc}] -> 0, without assuming Var(score)~1/T under arbitrary persistence. Nonstationary halves, an unstable split, or fixed T can defeat the cancellation/calibration.

**Fixed T is not repaired by SPJ.** With G increasing and T fixed, even a smaller nonzero bias becomes increasingly detectable. The simulations intentionally include this failure. Moment SPJ does not invert a target-slope Jacobian and is therefore compatible with weak-target inference *when its separate nuisance-bias conditions hold*. It is not the coefficient-SPJ routine in the existing estimator.

## 6. Fixed short one-way panels: an additional-assumption exact moment

The following alternative removes the FE rather than estimating it. For each unit define

\[
q_g(\beta)=\frac{R_g (ZY)_g-Y_g(Zr)_g}{T_g(T_g-1)}.
\]

Equivalently its numerator is

\[
\sum_{t<s}(Z_{gt}-Z_{gs})\{Y_{gt}r_{gs}-Y_{gs}r_{gt}\}.
\]

Substituting Y=a_g r+u eliminates a_g algebraically. A sufficient condition for E[q_g(beta0)|g]=0 is independence across periods conditional on g, together with E[u_gt|g]=0 and E[Z_gt u_gt|g]=0 for each period. For t!=s, independence gives

\[
E[Z_t u_t r_s\mid g]=E[Z_tu_t\mid g]E[r_s\mid g]=0,
\quad E[Z_t r_t u_s\mid g]=E[Z_t r_t\mid g]E[u_s\mid g]=0.
\]

More general off-diagonal orthogonality can replace independence, but is not inferred from the data. The usual contemporaneous additive IV conditions alone do **not** imply these cross-period restrictions. Serially dependent endogenous regressors can violate them even when contemporaneous IV moments are valid. Clustering the final covariance by unit does not correct an invalid mean restriction.

If the units are independent, L is fixed, the q_g covariance is nondegenerate, and a no-dominant-unit/Lindeberg CLT and covariance consistency hold, studentizing their sum yields chi-square(L) with **fixed T_g>=2 and G increasing**, including weak target relevance. There are no estimated FE in these moments. All-zero outcome units are retained with zero contribution. The moment is unbiased under its assumptions; the chi-square test is **asymptotic**, not finite-sample exact.

The code computes four group sufficient statistics (Y,R,ZY,ZR), avoiding quadratic pair enumeration. It uses one common log-scale factor across all observations: this changes every score by the same factor and leaves the statistic exactly invariant. Random unit-specific rescaling could change the zero-mean restriction and is not used.

This is an additional family of moment restrictions/weighting, not a numerically faster implementation of the exact same projected-IV point estimate. Unknown slopes must all be included in beta, not plugged in from a weak first step. Known offsets are supported; generic analytic/frequency weights are not yet added to this panel interface.

## 7. Multiple FE: balanced product circuits under stronger assumptions

Let D_i be the dummy row for any number of categorical FE partitions and a_i=exp(D_i'alpha). Choose two disjoint sets P and M, of equal size, satisfying

\[
\sum_{i\in P}D_i=\sum_{i\in M}D_i.
\]

For r_i(beta)=exp(o_i+X_i'beta), define

\[
\Delta_c(\beta)=\prod_{i\in P}Y_i\prod_{j\in M}r_j(\beta)
-\prod_{i\in M}Y_i\prod_{j\in P}r_j(\beta).
\]

Suppose, conditional on **all instruments and FE**, the row pairs (Y_i,X_i) are independent, and E[Y_i|Z,FE]=a_i E[r_i(beta0)|Z,FE]. Conditional independence factors each product expectation; signed dummy balance implies product_P a_i=product_M a_i. The two expectations are therefore identical, proving E[Delta_c(beta0)|Z,FE]=0. Multiplying by an instrument-only contrast h_c(Z) preserves the zero mean. The implementation uses h_c=sum_P Z_i-sum_M Z_i.

Select observation-disjoint circuits using the design/topology, not outcomes. Conditional on the FE, the resulting circuit scores are independent under the stipulated row independence. A fixed-L triangular-array CLT, covariance consistency and no dominant circuit then give an S statistic calibrated by chi-square(L). No coefficient-derivative inverse or estimated FE appears, so growing FE and fixed within-FE support do not themselves introduce incidental-estimation bias into this construction.

For gravity FE (i,j), (i,t), (j,t), a 2x2x2 cube with even-parity cells in P and odd-parity cells in M cancels every pair effect. Each side has four cells. The implemented experiment tiles a balanced 64x64x4 panel into 2048 disjoint cubes. The code verifies level-count cancellation in every supplied FE partition and refuses repeated observations across circuits. It does **not** automatically search every sparse or unbalanced graph for suitable circuits.

**This is not a general solution for three-way gravity errors.** Dyadic/serial dependence can connect otherwise observation-disjoint cubes; unconditional E[Zu]=0 alone does not justify multiplying by arbitrary instrument contrasts; non-disjoint circuits require an appropriate dependence covariance; missing cells and sample selection need separate analysis. Higher-order products can have heavy tails and poor power. The current simulator's median largest score-energy shares are about 12%–21%, a material warning despite reasonable null rejection frequencies. Do not choose circuits by outcomes or normalize each circuit by a random score magnitude.

The exact algebra establishes a mean restriction under explicit assumptions, not a proof of optimality, novelty, useful power in all settings, or universal finite-sample normality.

## 8. Numerical correctness and failure handling

A full-dummy oracle detected a projection-range problem in the **local dev1 parent's** fused MAP/CG route on a near-orthogonal right-hand side: nominal convergence and small normal-equation residual did not imply that the fitted increment lay in the FE span. The reproducer records a fitted-range discrepancy of about 4.9e-5. The same case with the existing LSMR route has range error near machine precision. This is not an audit of current main.

The new restricted-null fit therefore explicitly selects LSMR and uses RHS equilibration plus a convergence check. The scalar adapter is exposed at `hdfe/scaled_projection.py` so the inference layer does not import a model-layer implementation. Its numerical body is identical to the pre-existing dev1 adapter. The FE solver algorithm itself is not rewritten.

Covariance evaluation equilibrates the small moment covariance and checks positive definiteness/rank. It does not rescue a singular test by a pseudoinverse while retaining the old chi-square degrees of freedom. Underflow/overflow and nonconvergence remain visible. Passing an `assume_*` argument acknowledges a model condition; it is not an empirical certificate.

## 9. Experiments and what was actually measured

The scripts in `benchmarks/p1/` and raw JSON in `validation/iv_p1/` are authoritative. Each formal Monte Carlo cell has 1000 seeded datasets. There are 3000 fixed-nuisance, 8000 one-way panel and 3000 gravity-circuit datasets: **14000 total**. Multiple tests on the same dataset do not increase that dataset count. All tests fix the true null, so reported percentages are nominal-5% rejection rates, not full confidence-set coverage experiments. Failure counts and Wilson simulation intervals are retained.

The endogenous DGP has X=pi(Z1+.4 Z2)+.7v (with an extra lagged-v term only for the deliberate serial counterexample), and Y conditional on primitives is Poisson with intensity mu+.3 exp(alpha)(1+.8 Z1)v. Z and v are independent Rademacher draws. Hence contemporaneous conditional additive IV moments hold while X is genuinely endogenous; positivity of the intensity is checked. A bounded control is estimated in the fixed-nuisance experiment and has a **known** coefficient included in the offset in panel experiments. Gravity experiments generate independent rows conditional on deterministic FE and instruments.

The three fixed-nuisance strength cases reject 4.7%,5.1%,4.9%. For G=1000,T=4,strong relevance, raw profile/moment-SPJ/cross-product reject 37.9%,13.6%,4.0%; at G=4000,T=4 they reject 95.8%,49.0%,6.4%. At G=1000,T=64 they reject 6.4%,5.2%,5.1%. The 6.4% cross-product result is slightly above nominal even with its assumptions, illustrating finite-sample/calibration uncertainty, not exact testing. The serial violation raises cross-product rejection to 11.1%.

Gravity-circuit null rejection is 4.0%,5.3%,5.6% for strong/local-weak/zero relevance. Against beta=0 when true beta=.5, rejection is 100%,5.2%,5.2% respectively. Lack of power with weak or absent instruments is expected and must not be hidden. The explicit finite-state tests separately verify the algebraic mean cancellations and their failure under serial dependence.

## 10. Complexity and timing scope

`PreparedPanelS` copies the design once, encodes groups and caches Y/ZY. Per-null cost is O(N(K+L)+G L^2) and additional workspace is O(N+GL), for fixed K,L; initial `numpy.unique` encoding may be O(N log N). No N-by-N pair array is constructed. The exact preprocessing costs are recorded separately.

Serial same-host benchmarks used four Numba threads and one BLAS thread, excluded data generation/warmup, and used three repeats. Median cross-product test times are .0478s at one million rows and .3571s at five million, after preparation. Preparations take .2516s and 1.6748s respectively in those runs. Moment-SPJ null evaluation takes .0721s and .4810s; its recorded preparation is .5104s and 10.8364s. The larger construction cost and timing noise are retained, not omitted.

The general restricted-null three-FE test with regular controls takes 5.7562s at one million rows, excluding 1.5345s preparation. This uses LSMR and actually estimates null nuisance parameters. These are **one-null test timings**, not complete IV point-estimation speedups or entire confidence-grid timings. Process peak RSS is a process high-water mark, not a clean per-method allocation measurement or a hard memory guarantee. No Rust speedup claim is made.

## 11. Integration and usage contract

New imports live under `econhdfe.inference.iv_poisson`. They do not replace the parent `standard` or `estimating_equation` covariance, its overidentification test, or its outer solvers. `RestrictedPoissonS` is the closest addition to the existing additive-IV model under regular nuisance conditions. `PreparedPanelS(method="spj")` is a long-panel moment-bias extension. `cross_product` and `circuit_s_test` are separate extra-assumption moment families.

The source package is a local follow-on to the supplied dev1 artifact; PR36/main integration and version reconciliation remain separate. There is no new licensed-Stata or cross-platform certification. Documentation and validation distinguish mathematical identities, conditional asymptotic statements, finite-sample simulation, computational speed, and a future package-default decision.

## References

- Stock, J. H., and J. H. Wright (2000). GMM with Weak Identification. Econometrica 68(5), 1055–1096. DOI: 10.1111/1468-0262.00151.
- Jochmans, K. (2022). Bias in instrumental-variable estimators of fixed-effect models for count data. Economics Letters 212, 110318. DOI: 10.1016/j.econlet.2022.110318.
- Kwon, O., M. Larch, J. Yoon, and Y. V. Yotov (2026). Instrumental-Variable Poisson PML with High-Dimensional Fixed Effects. CESifo Working Paper 12641. DOI: 10.65864/2zo2fdcdpz. Working-paper status and methods checked on 28 October 2026.
- Weidner, M., and T. Zylkin (2021). Bias and consistency in three-way gravity models. IFS/cemmap working paper. DOI: 10.47004/wp.cem.2021.1121.
- Chernozhukov, V., et al. Double/Debiased Machine Learning for Treatment and Structural Parameters. arXiv:1608.00060. Cited for orthogonal-score/cross-fitting methodology, not as an FE-specific validation of this implementation.
