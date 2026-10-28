from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from ...errors import ShapeError, ConvergenceError, DivergenceError, NumericalError
from ...hdfe.plan import FEPlan
from ...iv.solve import weighted_2sls
from ..poisson import exp_clip_into, working_state, poisson_deviance
from .config import IVPPMLConfig


@dataclass(slots=True)
class IVIRLSState:
    beta: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    converged: bool
    iterations: int
    deviance: float
    epsilon: float
    absorb_iterations: int
    separated: np.ndarray
    transformed: np.ndarray
    X_dm: np.ndarray
    Z_dm: np.ndarray
    residual: np.ndarray
    projected_X: np.ndarray
    irls_w: np.ndarray
    outer_diagnostics: dict = field(default_factory=dict)
    linear_result: object = None


def fit_iv_irls(
    y,
    exog,
    endog,
    excluded,
    plan: FEPlan,
    true_w,
    offset,
    config: IVPPMLConfig,
    *,
    projector=None,
    initial_eta=None,
    detect_mu=False,
) -> IVIRLSState:
    """IRLS-IV core; legacy by default, with opt-in moment globalization."""
    if config.outer_solver == "moment":
        from .outer import fit_moment_irls
        return fit_moment_irls(y, exog, endog, excluded, plan, true_w, offset,
                               config, projector=projector, initial_eta=initial_eta)
    y = np.asarray(y, dtype=np.float64)
    C = np.asarray(exog, dtype=np.float64)
    E = np.asarray(endog, dtype=np.float64)
    I = np.asarray(excluded, dtype=np.float64)
    w0 = np.asarray(true_w, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    X = np.column_stack([C, E])
    n, k = X.shape
    kc, ki = C.shape[1], I.shape[1]

    mu = np.empty(n, dtype=np.float64)
    if initial_eta is not None:
        eta = np.asarray(initial_eta, dtype=np.float64).copy()
        if eta.shape != y.shape or not np.all(np.isfinite(eta)):
            raise ShapeError("initial_eta must be finite with one value per observation")
        exp_clip_into(eta, mu)
    else:
        mean_y = float(np.average(y, weights=w0))
        mu[:] = 0.5 * (y + mean_y)
        np.maximum(mu, 0.05 * y, out=mu)
        np.maximum(mu, 1e-3, out=mu)
        # mu guesses the TOTAL mean. Offset enters the working response;
        # adding it again here would violate mu == exp(eta).
        eta = np.log(mu)

    target_tol = max(1e-12, min(config.target_inner_tol, 0.1 * config.tolerance))
    hdfe_tol = max(config.start_inner_tol, config.tolerance)
    alt_tol = config.start_inner_tol
    projector = projector if projector is not None else (plan.projector(engine=config.engine) if plan.groups else None)

    irls_w = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    block = np.empty((n, 1 + k + ki), dtype=np.float64)
    eta_candidate = np.empty(n, dtype=np.float64)
    mu_candidate = np.empty(n, dtype=np.float64)
    beta = np.zeros(k, dtype=np.float64)
    anchor_eta = eta.copy()
    in_step_halving = False
    previous_deviance = None
    deviance = math.inf
    epsilon = math.inf
    ok = 0
    step_halving = 0
    absorb_iterations = 0
    separated = np.zeros(n, dtype=bool)
    final_solved = None

    for it in range(1, config.max_iter + 1):
        if not in_step_halving:
            anchor_eta[:] = eta
        if not np.all(np.isfinite(mu)) or not np.isfinite(np.sum(mu, dtype=np.float64)):
            raise DivergenceError(
                f"IV-PPML IRLS produced non-finite mu at iteration {it}",
                code="numerical.non_finite_mu", details={"iteration": int(it)},
                suggestion="Try separation=('fe','simplex','relu','mu'), standardize=True, or a simpler FE specification.",
            )
        if it > 10 and (np.max(np.abs(X @ beta), initial=0.0) > 1e6 or np.max(np.abs(eta), initial=0.0) > 30):
            raise DivergenceError(f"IV-PPML IRLS diverged at iteration {it}", details={"iteration": int(it)})

        working_state(y, mu, eta, offset, w0, irls_w, z)
        np.maximum(irls_w, 1e-20, out=irls_w)
        block[:, 0] = z
        block[:, 1:1 + k] = X
        if ki:
            block[:, 1 + k:] = I
        info = None
        if projector is not None:
            block, info = projector.residualize(block, irls_w, tol=hdfe_tol, return_info=True, copy=False)
            absorb_iterations += 0 if info is None else int(info.iterations)

        z_dm = block[:, 0]
        X_dm = block[:, 1:1 + k]
        if kc and ki:
            Z_dm = np.column_stack([X_dm[:, :kc], block[:, 1 + k:]])
        elif kc:
            Z_dm = X_dm[:, :kc]
        else:
            Z_dm = block[:, 1 + k:]

        solved = weighted_2sls(z_dm, X_dm, Z_dm, weights=irls_w)
        beta = solved.beta
        np.subtract(z, solved.residuals, out=eta_candidate)
        eta_candidate += offset

        if detect_mu and it > 1:
            threshold = math.log(config.mu_tol)
            if np.any(y > 0):
                threshold += min(float(np.min(eta_candidate[y > 0])) + 5.0, 0.0)
            separated |= (y == 0) & (eta_candidate <= threshold)

        eta_candidate[:] = np.clip(eta_candidate, -745.0, 709.0)
        exp_clip_into(eta_candidate, mu_candidate)
        if np.any(separated):
            mu_candidate[separated] = np.finfo(float).eps * 100.0
        dev_candidate = poisson_deviance(y, mu_candidate, w0, eta_candidate)
        if not np.isfinite(dev_candidate):
            raise NumericalError(f"IV-PPML IRLS produced non-finite deviance at iteration {it}", code="numerical.non_finite_deviance", details={"iteration": int(it)})

        if previous_deviance is not None:
            delta = previous_deviance - dev_candidate
            if dev_candidate < 0.1 * delta:
                delta = dev_candidate
            denom = max(min(dev_candidate, previous_deviance), 0.1)
            epsilon = abs(delta) / denom
            if epsilon < config.tolerance and (len(plan.groups) <= 1 or hdfe_tol <= 1.1 * target_tol):
                ok += 1
            elif delta < 0 and step_halving < config.max_step_halving:
                eta_candidate[:] = (
                    config.step_halving_memory * anchor_eta
                    + (1.0 - config.step_halving_memory) * eta_candidate
                )
                if step_halving > 0:
                    np.maximum(eta_candidate, -10.0, out=eta_candidate)
                exp_clip_into(eta_candidate, mu_candidate)
                eta, eta_candidate = eta_candidate, eta
                mu, mu_candidate = mu_candidate, mu
                step_halving += 1
                in_step_halving = True
                ok = 0
                continue
            else:
                step_halving = 0
                in_step_halving = False
                ok = 0

        eta, eta_candidate = eta_candidate, eta
        mu, mu_candidate = mu_candidate, mu
        previous_deviance = dev_candidate
        deviance = dev_candidate
        final_solved = solved

        if ok >= config.min_ok:
            return IVIRLSState(
                beta, mu, eta, True, it, deviance, float(epsilon), absorb_iterations,
                separated.copy(), block, X_dm, Z_dm, solved.residuals,
                solved.projected_X, irls_w.copy(),
            )
        if np.isfinite(epsilon) and epsilon < hdfe_tol:
            hdfe_tol = max(min(0.1 * hdfe_tol, alt_tol), target_tol)
            scaled = max(0.1 * epsilon, np.finfo(float).eps)
            alt_tol = 10.0 ** (-math.ceil(math.log10(1.0 / scaled)))

    if final_solved is None:
        raise ConvergenceError("IV-PPML IRLS did not complete an iteration", code="convergence.ivppml_no_iteration")
    return IVIRLSState(
        beta, mu, eta, False, config.max_iter, deviance, float(epsilon), absorb_iterations,
        separated.copy(), block, X_dm, Z_dm, final_solved.residuals,
        final_solved.projected_X, irls_w.copy(),
    )
