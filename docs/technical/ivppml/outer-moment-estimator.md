# IV-Poisson outer iteration: correction, implementation, and guarantees

Development candidate: **0.10.1.dev0**. Base: EconHDFE `main@2ab8cddd9a6e367291f3152591e09c1270418148`.
This is an opt-in rewrite of the **nonlinear outer estimator**, not a rewrite of the FE absorber.
The legacy default, weight semantics, and public estimator entry points remain available.
No release, tag, licensed-Stata certification, or universal speedup is implied.

## 1. What is corrected

The previous implementation's Poisson-deviance stopping and step-halving rules are not a general merit-function argument for IV estimation. An IV step can be correct even when Poisson deviance increases. The new mode instead checks the appropriate estimating equation, verifies the linearized moment residual, and uses an Armijo search on a moment merit function.

A second distinction is essential: in an overidentified model, the sample instrument moments generally do not all vanish. Requiring them all to be zero would reject legitimate estimates. Conversely, replacing the old iteration by an arbitrary GMM minimization changes the finite-sample estimating equation. This implementation deliberately preserves the legacy projected-IV equation in that case. It does **not** silently relabel it as efficient GMM or continuously updated GMM.

A third correction concerns the returned state. The legacy estimator performs a final coefficient-only weighted solve after the accepted `eta` and `mu` have been stored. The new mode returns one internally consistent `beta/eta/mu` and reuses its certified weighted system for inference. It does not append an unchecked final coefficient update.

## 2. Statistical and numerical contract

Let

\[
\eta=o+X\beta+D\alpha=o+A\theta,\qquad
\mu=\exp(\eta),\qquad r=y-\mu.
\]

`o` is the offset, `D` is the categorical-intercept FE design, and `X` contains included exogenous and endogenous regressors. Let `Z` contain included exogenous regressors and excluded instruments. User weights form \(W_0=\operatorname{diag}(w_{0i})\), whereas the changing Newton weights are

\[
W=W_0\operatorname{diag}(\mu).
\]

These weights have different roles and are never interchanged. Positive user weights are required by the existing public API. Redundant FE coordinates are interpreted on an identified quotient; the observation-level predictor, not an arbitrary dummy normalization, is the numerical object of interest.

The opt-in mode is selected with:

```python
from econhdfe import IVPPMLConfig, ExecutionConfig, ivppmlhdfe

cfg = IVPPMLConfig(
    outer_solver="moment",
    outer_linear_solver="auto",   # guarded sufficient statistics, stable fallback
    singleton_mode="unified",
    engine="optimized",
    tolerance=1e-9,
    target_inner_tol=1e-10,
)
result = ivppmlhdfe(
    y=y, exog=C, endog=E, instruments=Z_excluded,
    absorb=[firm, year], config=cfg,
    execution_config=ExecutionConfig(threads=4),
    vce="robust",
)
print(result.diagnostics["outer"])
```

The default remains `outer_solver="legacy"`. `outer_solver="moment"` defaults to FE-only separation and rejects ordinary-PPML regressor-separation methods. FE-only checks are not a complete general IV existence test. Structured `BlockDesign` execution is explicitly rejected in the new outer path rather than silently using a different algorithm. Dense designs with categorical FEs are supported; the new mode does not add heterogeneous-slope or group-individual FE support.

## 3. Derivation of the outer direction

At the current mean, let \(M_W\) be weighted residualization against `D`, and define

\[
\widetilde X=M_WX,\quad\widetilde Z=M_WZ,\quad
G=\widetilde Z'W\widetilde Z,\quad
C=\widetilde Z'W\widetilde X,\quad
\nu=\widetilde Z'W_0r.
\]

The weighted IV increment is

\[
\Delta\beta=(C'G^{-1}C)^{-1}C'G^{-1}\nu.
\]

The FE part is recovered implicitly from the working increment
\(v=(y-\mu)/\mu\): residualize `[v, X, Z_excluded]`, solve its weighted IV system, and set

\[
\Delta\eta=v-\left(\widetilde v-\widetilde X\Delta\beta\right).
\]

Thus \(\Delta\eta\in\operatorname{col}(A)\), up to numerical error, without constructing all FE coefficients. An offset-aware constant-FE initialization is feasible. An arbitrary warm start is first projected into the **new** specification, so a changed regressor specification cannot inject an off-model predictor into the line search.

### 3.1 Exactly identified models

When the non-FE instrument count equals the regressor count and `C` is nonsingular,

\[
\Delta\beta=C^{-1}\nu.
\]

Together with the FE update this solves

\[
Q'WA\Delta\theta=Q'W_0r,\qquad Q=[D,Z].
\]

For \(F(\theta)=Q'W_0[y-\exp(o+A\theta)]\), the Jacobian is
\(J_F=-Q'WA\). Therefore the weighted IV step is exactly a Newton step on the additive IV moments in exact arithmetic. This is a derivation for the specified estimator, not a claim that the algorithm solves every possible nonlinear IV definition.

### 3.2 Overidentified models

The preserved estimating equation is

\[
\Psi(\theta)=
\begin{bmatrix}
D'W_0r\\
C'G^{-1}\widetilde Z'W_0r
\end{bmatrix}=0.
\]

Its raw overidentifying moments need not be zero. Define \(\Pi=G^{-1}C\) and the effective instruments \(B_t=[D,\widetilde Z_t\Pi_t]\). During a trial step, these instruments are **frozen**. The direction solves the corresponding linearized frozen-instrument moments, and the line search decreases their scaled squared norm. At the next accepted point, instruments and the projected equation are recomputed before convergence can be certified.

This preserves the old fixed-point equation where all relevant ranks are full. It does not guarantee the same root when multiple roots exist, because initialization and globalization differ. It is not a full Newton method on \(\Psi\): the derivative of \(\Pi(\theta)\) and of weighted residualization is not included in the direction. Nor is it the derivative of a scalar CUE objective. For example,

\[
d\{m'\Omega^{-1}m\}
=2\,dm'\Omega^{-1}m-m'\Omega^{-1}(d\Omega)\Omega^{-1}m,
\]

so changing weights cannot simply be ignored in a purported CUE gradient.

**There is no unconditional global-convergence or quadratic-convergence guarantee for the overidentified mode.** Its guarantees are narrower: each accepted step satisfies the frozen-merit test, and a returned success satisfies the freshly recomputed projected estimating equation and a step-size check. Failure to find such a step raises an error; it is not converted to success using a flat deviance.

## 4. Why Poisson deviance is the wrong IV merit function

Take

\[
y=(3,1,2),\quad x=(-1,0,1),\quad z=(0,-1,1),\quad
(\alpha,\beta)=(\log2,0).
\]

Then \(\mu=(2,2,2)\). The intercept moment is zero and the instrument moment is
\(z'(y-\mu)=1\). The exact IV Newton step is \(\Delta\alpha=0\), \(\Delta\beta=1/2\).
But the derivative of the negative Poisson log likelihood along that direction is

\[
\sum_i(\mu_i-y_i)x_i\Delta\beta=1/2>0.
\]

Even a sufficiently small step in the correct IV Newton direction increases that likelihood objective. Therefore monotone Poisson-deviance acceptance is not generally compatible with IV Newton progress.

An analytic solution is obtained by writing \(t=e^\beta\):

\[
5t^2-7t-1=0,\quad
\widehat\beta=\log\left(\frac{7+\sqrt{69}}{10}\right),\quad
 e^{\widehat\alpha}=\frac1{t-1}.
\]

The regression test checks this solution independently and verifies that the final Poisson objective is worse than at the initial mean, while the IV moments vanish. This is an algorithmic counterexample, not an empirical endogeneity simulation.

## 5. What can actually be proved

### 5.1 Descent certificate, exact identification

Let `S` be the fixed positive normalization based on observed outcome and user-weight mass. In particular, it is independent of an arbitrary warm start. Set \(f=SF\) and \(\phi=\tfrac12\|f\|_2^2\). The implementation explicitly evaluates the linearized residual

\[
e=f+J_fd
\]

and requires \(\|e\|\le\eta\|f\|\), with default \(\eta=0.25<1\). Hence

\[
\phi'(0)=f'J_fd=f'(e-f)
\le-(1-\eta)\|f\|^2<0
\]

away from a root. This is the actual forcing certificate, not merely a heuristic mapping from an outer tolerance to an FE tolerance. If it fails, the inner solve is tightened, then the stable IV linear solver is tried; otherwise the algorithm stops with an explicit error.

Given differentiability and an exact descent direction, sufficiently short Armijo steps exist in exact arithmetic. Standard inexact-Newton convergence results require additional assumptions such as a suitable compact level set, uniform nonsingularity on the identified quotient, and regularity of the Jacobian. Local superlinear convergence requires forcing terms tending to zero; quadratic convergence requires a correspondingly stronger forcing rate. This implementation has finite refinement/backtracking budgets and floating-point tolerance floors. It therefore does **not** claim unconditional termination or asymptotic quadratic convergence at arbitrary precision.

In particular, weak identification, nonexistence of a finite root, multiple roots, and floating-point underflow cannot be resolved by a programming language change.

### 5.2 Acceptance and termination checks implemented

Each nonfinal step passes the linearized-moment forcing bound and an Armijo test. A trial that overflows or underflows the representable positive mean is rejected. The new path does not clip `eta`, impose the historical `eta >= -10` step-halving floor, or reject an economically valid finite predictor merely because `|eta| > 30`.

Success requires the normalized current equation residual to be at most `tolerance`, a relative predictor/coefficient correction at most `sqrt(tolerance)`, and the target FE accuracy (or a single analytically residualized FE). For overidentification the current effective instruments are rebuilt before this check. The step-size condition is important: moments can become numerically small along a divergent/separated sequence while the Newton correction remains nonzero.

`diagnostics["outer"]["moment_error"]` is the relevant certificate. `raw_moment_error` is also reported, but is not the stopping target in overidentified models. In modern mode `epsilon` reports the equation residual, not a relative change in Poisson deviance. `iterations` includes final certification passes and should not be confused with the count of accepted nonzero nonlinear steps.

### 5.3 Inference boundary

At termination, projected regressors, bread and the structural residual multiplier \((y-\mu)/\mu\) are evaluated at the returned mean. The sandwich retains the package's existing robust/cluster and frequency-weight conventions. Independent small dense-dummy tests check this numerical formula. This is not a new general proof of many-incidental-parameter asymptotics, weak-IV-valid inference, or absence of finite-sample bias.

## 6. Performance-oriented implementation

`econhdfe/iv/step.py` streams the sufficient statistics. It scales columns and weights, checks the condition of the dimensionless instrument Gram, applies a Cholesky whitening, and solves a **small** `L x K` SVD. It avoids full-height weighted `X`, weighted `Z`, and first-stage fitted `X` during ordinary outer iterations. Full fitted instruments are formed once for inference. Ambiguous instrument Gram conditioning or slope rank triggers the established stable tall-design solver.

This guard matters because forming a Gram matrix squares the instrument-design condition number. The fast path is not a universal replacement for QR/SVD. The independent moment forcing check adds a second, equation-based guard; it does not constitute an arbitrary-precision forward-error bound on coefficients.

For `N` rows, `K` regressors, `L` non-FE instruments, and block size `b`, fast linear-step work is

\[
O\{N(L^2+LK)+L^3+LK^2\},
\]

and extra matrix workspace is

\[
O\{b(K+L)+L^2+LK+K^2\},
\]

plus the `N`-vector residual/output. Existing persistent design arrays, FE workspaces and inference storage are outside this *linear-step* scratch bound. The entire estimator is not a streaming/out-of-core implementation, and `outer_chunk_rows` is not a global RSS cap.

Weighted residualized `X/Z` are reused between outer iterations. The identity
\(M_{W_{t+1}}M_{W_t}X=M_{W_{t+1}}X\) holds because the difference lies in the FE column space. The changing working increment is reset, so no old response is inadvertently treated as current. Every line-search trial evaluates exponentials, instrument moments, and FE group sums; it does **not** perform another FE residualization or weighted 2SLS solve.

The arithmetic-heavy paths already execute in BLAS/LAPACK, NumPy reductions and Numba-compiled loops. This version does not add Rust or modify the native within backend. Any future Rust work should be justified by a remaining measured reduction/copy bottleneck, not by assuming that Python control flow is the dominant cost.

## 7. Reproduction and performance claim boundary

Run new numerical tests with:

```bash
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
python -m pytest -q tests/numerics/test_iv_outer_moment.py
```

Run fresh-process full-estimator timings with:

```bash
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
python benchmarks/ivppml_outer.py --n 1000000 --case crossed3 \
  --mode legacy --output legacy.json
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
python benchmarks/ivppml_outer.py --n 1000000 --case crossed3 \
  --mode moment --output moment.json
```

The harness performs a 100,000-row same-case warm-up outside the measured interval. Input generation, JIT warm-up, hashing and result serialization are excluded; sample preparation, FE absorption, nonlinear iteration and robust inference are included. Input hashes must match. Peak RSS is fresh-process high-water memory, not a theoretical allocation bound. Timings compare the existing `optimized` FE backend in both modes; they do not mix in the unmerged native-PPML backend work.

Measured results, individual runs, errors, stage timings and environmental provenance are recorded under `validation/iv_outer/`. A performance claim applies to those configurations and that host, not every sample or architecture. There is no promised universal multiplier, and the new solver is not made the default merely because selected benchmarks improve.

## References and prior-art boundary

1. Eisenstat, S. C. and H. F. Walker (1996), *Choosing the Forcing Terms in an Inexact Newton Method*, SIAM Journal on Scientific Computing 17(1), 16–32, DOI: 10.1137/0917003. https://epubs.siam.org/doi/10.1137/0917003
2. Eisenstat, S. C. and H. F. Walker (1994), *Globally Convergent Inexact Newton Methods*, SIAM Journal on Optimization 4(2), 393–422, DOI: 10.1137/0804022. https://epubs.siam.org/doi/10.1137/0804022
3. StataCorp, *Poisson regression with endogenous variables*, official discussion of additive/multiplicative specifications and GMM. https://www.stata.com/features/overview/poisson-regression-with-endogenous-variables/
4. EconHDFE source at the pinned base commit, especially `models/ppml_iv/irls.py`, `estimator.py`, and `iv/solve.py`.

Newton globalization, FWL residualization, Cholesky whitening, QR/SVD fallback, and sufficient-statistic IV are established techniques. The contribution here is their estimator-specific integration, explicit correction of convergence semantics, independently checked state consistency, and measured implementation—not a claim of a new general Newton or GMM theorem.
