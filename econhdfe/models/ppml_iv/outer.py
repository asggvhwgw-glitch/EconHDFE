"""Moment-certified IV-Poisson outer iteration (opt-in).

Exactly identified: damped inexact Newton on [D,Z]' w0 (y-mu) = 0.
Overidentified: retain the legacy iterated-IV estimating equation.  Each line
search freezes the current effective instruments; this is NOT fixed-weight
GMM, CUE, or an unconditional global-convergence claim.

Categorical-intercept FE only.  The existing FE projector is not modified.
"""
from __future__ import annotations
from time import perf_counter
import math
import numpy as np
from numba import njit, prange
from scipy.special import logsumexp
from ...errors import ConvergenceError, NumericalError, ShapeError
from ...iv.step import solve_iv_step
from ...compute.wls import weighted_lstsq
from ..poisson import poisson_deviance


@njit(cache=True, nogil=True, parallel=True)
def _increment_state(y, mu, w0, weights, work, wr):
    invalid = 0
    for i in prange(len(y)):
        m = mu[i]
        if not math.isfinite(m) or m <= 0.0:
            invalid += 1
            weights[i] = work[i] = wr[i] = 0.0
        else:
            weights[i] = w0[i] * m
            work[i] = y[i] / m - 1.0
            wr[i] = w0[i] * (y[i] - m)
            if (not math.isfinite(weights[i]) or weights[i] <= 0.0
                    or not math.isfinite(work[i]) or not math.isfinite(wr[i])):
                invalid += 1
    return invalid


@njit(cache=True, nogil=True, parallel=True)
def _trial(eta, direction, step, y, w0, eta_out, mu_out, wr_out):
    invalid = 0
    for i in prange(len(eta)):
        value = eta[i] + step * direction[i]
        # Reject nonrepresentable trials; never clip the predictor or the mean.
        if (not math.isfinite(value) or value > 709.782712893384
                or value < -744.4400719213812):
            invalid += 1
            eta_out[i] = value
            mu_out[i] = wr_out[i] = 0.0
        else:
            m = math.exp(value)
            eta_out[i], mu_out[i] = value, m
            wr_out[i] = w0[i] * (y[i] - m)
            if not math.isfinite(wr_out[i]):
                invalid += 1
    return invalid


class _MomentMetric:
    """Fixed exact-ID metric; frozen-per-step projected metric for over-ID."""
    def __init__(self, raw_Z, plan, reference_mass, chunk_rows):
        self.Z, self.plan = raw_Z, plan
        self.fe_scale = [np.bincount(g, weights=reference_mass, minlength=L)
                         for g, L in zip(plan.groups, plan.levels)]
        if any(np.any(s <= 0) or not np.all(np.isfinite(s)) for s in self.fe_scale):
            raise NumericalError("invalid FE moment normalization")
        ss = np.zeros(raw_Z.shape[1])
        for lo in range(0, len(raw_Z), chunk_rows):
            hi = min(len(raw_Z), lo + chunk_rows)
            z = raw_Z[lo:hi]
            ss += np.einsum('ij,i,ij->j', z, reference_mass[lo:hi], z)
        self.z_scale = np.sqrt(ss) * np.sqrt(float(np.sum(reference_mass)))
        if np.any(self.z_scale <= 0) or not np.all(np.isfinite(self.z_scale)):
            raise NumericalError("invalid instrument moment normalization; rescale X/Z")

    def evaluate(self, wr, *, Z_dm=None, first_stage=None):
        parts = [np.bincount(g, weights=wr, minlength=L) / scale
                 for g, L, scale in zip(self.plan.groups, self.plan.levels, self.fe_scale)]
        if first_stage is None:
            slope = self.Z.T @ wr / self.z_scale
        else:
            scale = np.abs(first_stage).T @ self.z_scale
            scale = np.maximum(scale, np.finfo(float).tiny)
            slope = first_stage.T @ (Z_dm.T @ wr) / scale
        parts.append(slope)
        return np.concatenate(parts)


def fit_moment_irls(y, exog, endog, excluded, plan, true_w, offset, config,
                    *, projector=None, initial_eta=None):
    """Fit with a verified linear residual and moment-based acceptance.

    The state returned for inference is evaluated at the SAME beta/eta/mu;
    callers must not append an unchecked last coefficient-only Newton update.
    """
    from .irls import IVIRLSState
    t_start = perf_counter()
    y, C, E, I, w0, offset = [np.asarray(a, dtype=np.float64)
                             for a in (y, exog, endog, excluded, true_w, offset)]
    X = np.column_stack([C, E]); raw_Z = np.column_stack([C, I])
    n, k = X.shape; kc, ki = C.shape[1], I.shape[1]
    exact = raw_Z.shape[1] == k
    target = max(1e-12, min(config.target_inner_tol, 0.1 * config.tolerance))
    inner_tol = max(target, config.start_inner_tol)
    projector = projector if projector is not None else (
        plan.projector(engine=config.engine) if plan.groups else None)
    beta = np.zeros(k)
    # Feasible, offset-aware initialization; unlike the legacy y-dependent guess.
    if plan.groups:
        total_y = float(w0 @ y)
        if not total_y > 0:
            raise NumericalError("no positive outcome mass for IV-Poisson initialization")
        intercept = math.log(total_y) - float(logsumexp(offset + np.log(w0)))
        eta = offset + intercept
    else:
        eta = offset.copy()
    timings = dict(projection=0.0, linear_solve=0.0, moments=0.0, line_search=0.0)
    absorb_iterations = 0; projections = 0; linear_solves = 0
    if initial_eta is not None:
        warm = np.asarray(initial_eta, dtype=np.float64)
        if warm.shape != y.shape or not np.all(np.isfinite(warm)):
            raise ShapeError("initial_eta must be finite with one value per observation")
        # Project arbitrary/changed-specification warm starts onto the NEW model.
        initial_block = np.column_stack([warm - offset, X])
        if projector is not None:
            tic = perf_counter()
            initial_block, info = projector.residualize(
                initial_block, w0, tol=target, return_info=True, copy=False)
            timings['projection'] += perf_counter() - tic
            absorb_iterations += int(info.iterations); projections += 1
        beta, residual = weighted_lstsq(initial_block[:, 1:], initial_block[:, 0], w0)
        eta = warm - residual
        del initial_block, residual
    mu = np.empty(n); wr = np.empty(n)
    if _trial(eta, np.zeros(n), 0., y, w0, eta, mu, wr):
        raise NumericalError("initial IV-Poisson mean is not representable in float64")
    # Normalize by observed mass, not an arbitrary warm start.  This fixed
    # metric is identical across cold/warm starts and frequency replication.
    metric = _MomentMetric(raw_Z, plan, w0 * (y + float(np.average(y, weights=w0))),
                           config.outer_chunk_rows)
    block = np.empty((n, 1 + k + ki))
    Z_dm = np.empty_like(raw_Z)
    weights = np.empty(n); work = np.empty(n); resid = np.empty(n)
    direction = np.empty(n); linear_wr = np.empty(n)
    eta_trial = np.empty(n); mu_trial = np.empty(n); wr_trial = np.empty(n)
    history = []; backtracks = 0; refinements = 0; backend_counts = {}
    last_error = math.inf; max_forcing = 0.0; good_count = 0

    for iteration in range(1, config.max_iter + 1):
        if _increment_state(y, mu, w0, weights, work, wr):
            raise NumericalError("invalid IV-Poisson working state; no clipping/floor applied")
        if not np.isfinite(np.sum(weights)):
            raise NumericalError("IV-Poisson weights overflow; rescale the model")
        if iteration == 1 or not config.fast_partial:
            block[:, 1:1+k] = X
            if ki: block[:, 1+k:] = I
        force_stable = False
        # Refinement changes only the Newton solve, not the nonlinear iterate.
        for refinement in range(9):
            block[:, 0] = work
            if projector is not None:
                tic = perf_counter()
                block, info = projector.residualize(
                    block, weights, tol=inner_tol, return_info=True, copy=False)
                timings['projection'] += perf_counter() - tic
                absorb_iterations += int(info.iterations); projections += 1
            X_dm = block[:, 1:1+k]
            if kc: Z_dm[:, :kc] = X_dm[:, :kc]
            if ki: Z_dm[:, kc:] = block[:, 1+k:]
            tic = perf_counter()
            solved = solve_iv_step(block[:, 0], X_dm, Z_dm, weights,
                                  mode="stable" if force_stable else config.outer_linear_solver,
                                  chunk_rows=config.outer_chunk_rows,
                                  gram_rcond=config.outer_gram_rcond, resid_out=resid)
            timings['linear_solve'] += perf_counter() - tic
            linear_solves += 1
            backend_counts[solved.backend] = backend_counts.get(solved.backend, 0) + 1
            np.subtract(work, resid, out=direction)
            tic = perf_counter()
            kw = {} if exact else dict(Z_dm=Z_dm, first_stage=solved.first_stage)
            f0 = metric.evaluate(wr, **kw)
            error = float(np.max(np.abs(f0), initial=0.0))
            step_error = max(float(np.max(np.abs(direction))) / (1. + float(np.max(np.abs(eta-offset)))),
                             float(np.max(np.abs(solved.beta))) / (1. + float(np.max(np.abs(beta)))))
            norm0 = float(np.linalg.norm(f0))
            np.multiply(weights, direction, out=linear_wr)
            np.subtract(wr, linear_wr, out=linear_wr)
            flin = metric.evaluate(linear_wr, **kw)
            forcing = float(np.linalg.norm(flin)) / max(norm0, np.finfo(float).tiny)
            derivative = float(f0 @ (flin - f0))
            timings['moments'] += perf_counter() - tic
            if not (np.isfinite(error) and np.isfinite(forcing) and np.isfinite(derivative)):
                raise NumericalError("non-finite IV moment residual")
            accurate = (len(plan.groups) <= 1) or inner_tol <= 1.01 * target
            ready = error <= config.tolerance and step_error <= math.sqrt(config.tolerance)
            if ready and not accurate:
                inner_tol = target; refinements += 1
                continue
            if ready and accurate:
                good_count += 1
                if good_count >= config.min_ok:
                    projected_X = Z_dm @ solved.first_stage
                    # Use structural residual/mean for inference at accepted state.
                    residual = work.copy()
                    raw_error = float(np.max(np.abs(metric.evaluate(wr)), initial=0.0))
                    diagnostics = dict(
                        solver="moment", target="exact_additive_moments" if exact else "legacy_projected_iv_equation",
                        exact_identification=exact, moment_error=error,
                        raw_moment_error=raw_error, step_error=step_error,
                        inner_tolerance=inner_tol, line_search_backtracks=backtracks,
                        inner_refinements=refinements, projections=projections,
                        linear_solves=linear_solves, linear_backends=backend_counts,
                        max_accepted_forcing=max_forcing, history=history,
                        timings={**timings, "total": perf_counter()-t_start},
                        global_guarantee="conditional_inexact_newton" if exact else "none_frozen_instrument_local_safeguard",
                        final_state_consistent=True,
                    )
                    return IVIRLSState(beta.copy(), mu, eta, True, iteration,
                        poisson_deviance(y, mu, w0, eta), error, absorb_iterations,
                        np.zeros(n, dtype=bool), block, X_dm, Z_dm, residual,
                        projected_X, weights, diagnostics, solved)
                # Confirm again at the identical accepted point, with a fresh solve.
                break
            good_count = 0
            if forcing <= config.outer_forcing and derivative < 0.0:
                break
            refinements += 1
            if inner_tol > target:
                inner_tol = max(target, inner_tol * .01)
            elif not force_stable and config.outer_linear_solver == "auto":
                force_stable = True
            else:
                raise ConvergenceError(
                    "IV Newton direction failed the linear-moment forcing certificate",
                    code="convergence.ivppml_forcing",
                    details={"iteration": iteration, "forcing": forcing,
                             "moment_error": error, "step_error": step_error})
        else:
            raise ConvergenceError("IV Newton direction exhausted inner refinements")
        if ready and accurate:
            continue
        phi0 = .5 * norm0 * norm0
        step = 1.0; accepted = False
        tic = perf_counter()
        for trial in range(config.outer_max_backtracks + 1):
            invalid = _trial(eta, direction, step, y, w0, eta_trial, mu_trial, wr_trial)
            phi = math.inf
            if not invalid:
                ft = metric.evaluate(wr_trial, **kw)
                phi = .5 * float(ft @ ft)
            if np.isfinite(phi) and phi <= phi0 + config.outer_armijo * step * derivative:
                accepted = True
                break
            step *= .5
        timings['line_search'] += perf_counter() - tic
        if not accepted:
            raise ConvergenceError(
                "IV moment line search failed; no unverified step was accepted",
                code="convergence.ivppml_line_search",
                details={"iteration": iteration, "moment_error": error,
                         "backtracks": config.outer_max_backtracks})
        backtracks += trial
        max_forcing = max(max_forcing, forcing)
        if config.outer_record_history:
            history.append(dict(iteration=iteration, moment_error=error,
                step_error=step_error, step=step, backtracks=trial,
                merit_before=phi0, merit_after=phi, directional_derivative=derivative,
                forcing=forcing, inner_tolerance=inner_tol,
                linear_backend=solved.backend,
                instrument_rcond=solved.instrument_rcond, slope_rcond=solved.slope_rcond))
        beta += step * solved.beta
        eta, eta_trial = eta_trial, eta
        mu, mu_trial = mu_trial, mu
        last_error = error
        # Accuracy tied to equation residual; final certification always uses target.
        inner_tol = max(target, min(inner_tol, .01 * max(error, config.tolerance)))
    raise ConvergenceError("IV moment solver exhausted outer iterations",
        code="convergence.ivppml_moment", details={"iterations": config.max_iter,
                                                   "moment_error": last_error})
