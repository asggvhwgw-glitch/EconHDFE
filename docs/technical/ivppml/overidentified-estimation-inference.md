# Overidentified IV-Poisson: profile Newton and estimating-equation inference

Development continuation of PR #36, 2026-10-28. No change to the default
`outer_solver="legacy"`. No efficient-GMM, CUE, arbitrary-HDFE asymptotic or
weak-IV-robust inference claim is made.

## 1. The estimator is explicitly fixed

Let the log mean be `eta = offset + X beta + D alpha`, `mu = exp(eta)`,
`u = y-mu`, base weights `w0 > 0`, and working weights `W = diag(w0*mu)`.
D is any full-rank coordinate representation of the categorical FE space;
all formulas are invariant to FE normalization. Z includes the exogenous
regressors and excluded instruments. K=cols(X), L=cols(Z), L >= K.

The legacy iterated weighted-IV update has the estimating equations

    D' diag(w0) u = 0,
    H' diag(w0) u = 0,

where, using the CURRENT W,

    P = D (D' W D)^(-1) D' W,    M = I-P,
    Xt = M X,  Zt = M Z,
    C = Zt' W Zt,  B = Zt' W Xt,
    Pi = C^(-1) B,   H = Zt Pi.

These are not all L raw sample moments being set to zero. They are also not
necessarily efficient GMM: C uses working-mean weights, not the covariance
of instrument residual scores. Continuously updating a weight inside a scalar
GMM objective introduces another derivative and defines a different estimator.
This continuation keeps the same root equations rather than silently making
that change.

The standard general references are Hansen (1982), *Econometrica* 50, 1029-1054,
and, for the distinction between correct-specification and misspecification
inference, Hall and Inoue (2003), *Journal of Econometrics* 114, 361-394, and
Hansen and Lee (2021), *Econometrica* 89, 1419-1447. The specific equations below
are derived for this implementation; the references do not certify its code.

## 2. Exact profiled derivative

For each beta, solve the FE moment equation for alpha(beta). Suppose this
finite profile exists locally, is differentiable on the identified FE space,
and the relevant matrices have full rank. Implicit differentiation gives

    d eta = Xt d beta.

Define

    m = Zt' diag(w0) u,
    R = Xt-H,
    h = Zt C^(-1) m,
    F(beta) = H' diag(w0) u = Pi' m.

At a profiled point D'diag(w0)u=0. The derivative of the residualized Z term
against u vanishes there. Weighted projection orthogonality gives

    d C = Zt' diag(W_diag * d eta) Zt,
    d B = Zt' diag(W_diag * d eta) Xt,
    d Pi = C^(-1) Zt' diag(W_diag * d eta) R.

Consequently

    J_profile = d F / d beta'
              = -H' W Xt + R' diag(W_diag*h) Xt.

Write

    A0 = H' W Xt,
    T  = R' diag(W_diag*h) Xt,
    A  = -J_profile = A0-T.

The correction T is generally nonsymmetric. It must NOT be symmetrized or
passed to a positive-definite inverse routine as though it were a Hessian.
The Newton increment solves A delta_beta = F.

Exactly identified, C^(-1)B is square and nonsingular. At a root all raw
instrument moments vanish, hence h=0 and T=0. In an overidentified finite
sample, m generally remains nonzero at a root, so freezing Pi omits T.
Under correct specification and fixed dimensions, m=O_p(sqrt(N)), h=O_p(N^-1/2)
and T/N vanishes; this explains why a standard frozen-bread sandwich can still
be first-order valid. A finite-sample derivative mismatch alone is NOT proof
that the old first-order variance is statistically inconsistent.

## 3. Actual-merit safeguarded profile Newton

`outer_solver="profile"` profiles the categorical FE equations at each trial,
recomputes W, residualized X/Z and Pi, and tests a fixed-scaled merit

    phi(beta) = 0.5 || S F(beta) ||^2.

S is fixed throughout a fit, with diagonal positive normalizations. An exact
Newton direction satisfies phi'(0)=-||S F||^2. With controlled profile and
projection errors, the usual inexact-Newton residual conditions apply. Local
quadratic convergence requires a sufficiently regular profile, a nonsingular
Jacobian and exact enough nuisance solves. Numerical tolerances ultimately
limit that rate. No unconditional global root-existence, uniqueness or
convergence guarantee follows. A merit stationary point is not automatically a
root. The implementation only returns success after both FE and slope residual
certificates and a small slope correction hold.

There is also a precise local obstruction for the frozen-profile iteration.
For beta_new = beta + A0(beta)^(-1) F(beta), at a root its derivative is

    d beta_new / d beta' = A0^(-1) T.

A spectral radius strictly below one gives local contraction under the usual
smoothness conditions; a radius above one gives an unstable local linearization.
Thus overidentification alone does not make the old algorithm wrong or slow.
The relevant issue is this derivative correction relative to A0. Under correct
specification it is often small in large samples; under misspecification it
need not be. Full profile Newton removes this first-order fixed-point term,
subject to a nonsingular exact Jacobian and accurate nuisance solves.

The method costs more per trial than `moment`: every accepted or rejected beta
trial may need nonlinear FE profiling and a new weighted design projection.
The old `moment` mode remains the cheaper frozen-instrument safeguard. Its merit
is frozen per step and is not silently re-described as the true profile merit.

The extra nuisance solve uses the existing FE projector, not a new FE linear
solver. A grouped-moment merit avoids cancellation in `wr' delta_FE` near the
FE profile root, where individual structural residuals need not become small.
Projection RHS are equilibrated and failed projection convergence is rejected;
this protects the near-zero included-exogenous residual columns occurring in
inference. No FE sample trimming or estimator change is introduced by that guard.

## 4. Why replacing only the bread is not enough

At an overidentified misspecified population root, Pi is itself estimated from
the empirical distribution. Replacing A0 by A while retaining the old score
`H_i*w0_i*u_i` misses that sampling variation. It also misses the indirect
variation of the profiled FE solution.

Consider a multiplicative infinitesimal perturbation to observation i's base
weight. At fixed theta, differentiation of the weighted first stage produces
`w0_i*mu_i*h_i*R_i`. Solving the perturbed FE equation adds the profile term
`P(hR)_i*w0_i*u_i`. Thus the profiled estimating-equation score is

    psi_i = [H_i + P(hR)_i] w0_i u_i + w0_i mu_i h_i R_i,

where hR is the N-by-K rowwise product. At a sample root, sum(psi_i)=0 in exact
arithmetic: H'wr=0; R'Wh=0 by first-stage orthogonality; and the FE projection
term sums to zero because D'wr=0.

The empirical first-order response of beta to observation i's log-weight is

    IF_i = A^(-1) psi_i.

This identity was checked against central finite differences of the COMPLETE
full-dummy estimating function, including re-estimation of its first stage for
each empirical perturbation. It is not checked against another implementation
of the same symbolic formula alone.

Under a regular, strongly identified, fixed-dimensional random-design
M/Z-estimation asymptotic sequence, it gives the sandwich

    V = A^(-1) [sum_i psi_i psi_i'] A^(-T),

with the existing finite-sample scalar corrections. For independent clusters,
replace individual scores by cluster sums; the existing CGM subset algebra is
available for multiway clusters. Cluster-count/asymptotic assumptions remain
necessary. The implementation is not a proof for arbitrary growing numbers of
FE, short panels with incidental-parameter bias, or few clusters.

For frequency weights, the robust meat uses psi_i/sqrt(f_i), so compressed
frequency observations reproduce literal row replication. Cluster sums retain
full frequency weights. The numerical residual scores and weights are never
used to reinterpret frequency multiplicity.

`outer_inference="estimating_equation"` enables this complete sandwich for a
certified `moment` or `profile` result. `standard` remains the default. Under
correct specification h converges to zero and the correction vanishes to first
order; under fixed misspecification it generally need not. Neither option is
weak-IV-robust. The `_cons` display remains an FE normalization with zero posted
variance; it is not being certified as an estimated intercept with zero sampling
uncertainty.

## 5. Overidentification is not the arbitrary-weight objective

At a consistent root under the null, let N0 be an L-by-(L-K) full-column-rank
basis satisfying N0'B=0 (implemented after instrument equilibration). Then
parameter-estimation error cancels to first order in the orthogonalized moments:

    c = N0' m,
    Omega_c = sum_i [N0' Zt_i w0_i u_i][...]',
    T_oi = c' Omega_c^(-1) c  -> chi-square(L-K).

The profiled-FE influence is represented by Zt, not raw Z without nuisance
adjustment. This test can be used with a regular consistent non-efficient
estimator; it is NOT the raw `N*Q` from a nonoptimal weighting matrix re-labelled
as a Hansen J. Equivalently, its finite-sample quadratic form is the minimized
linearized moment criterion

    min_delta (m-B delta)' Omega^(-1) (m-B delta).

Under exactly identification there is no test and no p-value. Rank-deficient
moment derivatives and singular/indefinite restriction covariance matrices
produce a diagnostic with no p-value. The test does not silently PSD-repair or
use a pseudoinverse while retaining nominal degrees of freedom. A numerically
full-rank Jacobian is NOT evidence against economically weak instruments.
Nonrejection is not proof that individual instruments are valid.

`outer_overid_test=True` enables this optional diagnostic. The null theory still
requires strong identification, suitable nuisance rates and a CLT. The test is
not a replacement for weak-IV-robust confidence sets or many-instrument theory.

## 6. Performance and validation contract

All derivative corrections are block reductions. For fixed K,L the algebra is
linear in N, with additional work of order N*(L^2+LK+K^2) and small L/K matrix
solves. There is no numerical derivative loop over FE coefficients. The full
sandwich adds ONE K-RHS FE projection at final inference, not per observation
and not one projection per FE coefficient. The overidentification test stores
N*(L-K) restriction scores and uses the shared cluster aggregation machinery.
These statements do not bound all estimator memory or eliminate the cost of
FE projections. The profile solver can be slower overall despite fewer outer
iterations; report measured whole-estimator timings, not just iteration counts.

Tests cover independent reprofiled derivatives away from a root, full-dummy
empirical influence derivatives, variance matrices, both certified outer modes,
1/2/3 FE dimensions, frequency replication, cluster aggregation, regressor units,
instrument basis changes, zero restrictions, rank-deficient restriction meat,
projection failure, and near-zero RHS. The benchmark script includes a genuinely
endogenous Gaussian-control DGP with valid additive moments and a misspecified
variant. Fixed-dimensional Monte Carlo is diagnostic, not a coverage guarantee.

The parent source candidate remains PR #36 at 76523437. Current main has moved
to 85e5ff90 (0.9.2); integration of #35 and final version reconciliation remain
separate gates. The branch-local dev1 label does not establish a new release
line and no tag or publication is authorized here.
