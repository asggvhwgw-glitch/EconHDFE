from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from ...errors import ShapeError, ConvergenceError, DivergenceError
from .config import PPMLConfig
from ...hdfe.plan import FEPlan
from ...compute.linalg import weighted_lstsq
from ..poisson import exp_clip_into, working_state, poisson_deviance


@dataclass(slots=True)
class IRLSState:
    beta: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    z: np.ndarray
    transformed: np.ndarray
    converged: bool
    iterations: int
    deviance: float
    epsilon: float
    absorb_iterations: int
    separated: np.ndarray



def _predict_eps(history: list[float], eps: float | None) -> float:
    """Three-point log-epsilon extrapolation used by ppmlhdfe."""
    if eps is not None and np.isfinite(eps) and eps > 0:
        history.append(math.log(eps))
        if len(history) > 3:
            del history[0]
    if len(history) < 3:
        return math.inf
    x = np.arange(1.0, 4.0)
    A = np.column_stack([x, x*x, x*x*x])
    coef = np.linalg.solve(A, np.asarray(history))
    xx = 4.0
    forecast = np.array([xx, xx*xx, xx*xx*xx]) @ coef
    return float(np.exp(np.clip(forecast, -745.0, 709.0)))


def fit_irls(
    y,
    X,
    plan: FEPlan,
    true_w,
    offset,
    config: PPMLConfig,
    *,
    detect_mu=False,
    projector=None,
    initial_eta=None,
) -> IRLSState:
    """Fit PPML by IRLS with persistent workspaces and a reusable FE projector."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    w0 = np.asarray(true_w, dtype=np.float64)
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=np.float64)
    n, k = X.shape

    mu = np.empty(n, dtype=np.float64)
    if initial_eta is not None:
        eta = np.asarray(initial_eta, dtype=np.float64).copy()
        if eta.shape != y.shape or not np.all(np.isfinite(eta)):
            raise ShapeError("initial_eta must be finite with one value per observation")
        exp_clip_into(eta, mu)
    else:
        ybar = float(np.average(y, weights=w0))
        mu[:] = 0.5 * (y + ybar)
        np.maximum(mu, 0.05 * y, out=mu)
        np.maximum(mu, 1e-3, out=mu)
        eta = np.log(mu)
    prev_dev = poisson_deviance(y, mu, w0, eta)

    highest_tol = max(1e-12, min(config.target_inner_tol, 0.1 * config.tolerance))
    hdfe_tol = max(config.start_inner_tol, config.tolerance)
    alt_tol = math.inf
    eps = None
    eps_history: list[float] = []
    transformed = None
    ok = 0
    absorb_iters = 0
    beta = np.zeros(k)
    separation_mask = np.zeros(n, dtype=bool)
    zero = y == 0
    positive = ~zero
    log_sep_tol = math.log(config.mu_tol)
    if projector is None and plan.groups:
        projector = plan.projector(engine=config.engine, method=("within" if config.engine == "within" else "map"), require_convergence=(config.engine == "within"))

    # Persistent N-sized workspaces.  The fast-partial block is residualized
    # in place and reused across outer iterations.
    irls_w = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    z_prev = np.empty(n, dtype=np.float64)
    z_prev.fill(0.0)
    eta_next = np.empty(n, dtype=np.float64)
    mu_next = np.empty(n, dtype=np.float64)
    resid = np.empty(n, dtype=np.float64)
    block = np.empty((n, k + 1), dtype=np.float64)

    for it in range(1, config.max_iter + 1):
        predicted_eps = _predict_eps(eps_history, eps)
        working_state(y, mu, eta, offset, w0, irls_w, z)

        fast_partial = config.fast_partial and it > 1 and transformed is not None
        if fast_partial:
            # ``block`` already contains the previous residualized X.  FWL
            # invariance lets the new weighted projection start from it; only
            # the working-response delta must be injected.
            block[:, 0] += z - z_prev
        else:
            block[:, 0] = z
            if k:
                block[:, 1:] = X

        info = None
        if projector is not None:
            block, info = projector.residualize(
                block, irls_w, tol=hdfe_tol, return_info=True, copy=False
            )
            absorb_iters += 0 if info is None else int(info.iterations)
        transformed = block
        zt, Xt = block[:, 0], block[:, 1:]

        # Upstream uses normal equations only while both the HDFE tolerance and
        # predicted outer error are comfortably above the final tolerance.
        fast = bool(
            config.fast_solver and k
            and hdfe_tol > config.tolerance * 11
            and predicted_eps > config.tolerance
        )
        beta, _ = weighted_lstsq(Xt, zt, irls_w, fast=fast, resid_out=resid)
        np.subtract(z, resid, out=eta_next)
        eta_next += offset
        if detect_mu and np.any(zero):
            positive_eta = eta_next[positive]
            adjusted = log_sep_tol + min(float(np.min(positive_eta)) + 5.0, 0.0)
            separation_mask |= zero & (eta_next <= adjusted)
        exp_clip_into(eta_next, mu_next)
        if np.any(separation_mask):
            mu_next[separation_mask] = np.finfo(np.float64).eps * 100.0
        dev = poisson_deviance(y, mu_next, w0, eta_next)
        delta = prev_dev - dev
        if dev < 0.1 * delta:
            delta = dev
        denom = max(min(dev, prev_dev), 0.1)
        eps = abs(delta) / denom

        accurate_inner = (len(plan.groups) <= 1) or (hdfe_tol <= 1.1 * highest_tol)
        if eps < config.tolerance and (not fast) and accurate_inner:
            ok += 1
        else:
            ok = 0

        z_prev[:] = z
        eta, eta_next = eta_next, eta
        mu, mu_next = mu_next, mu
        prev_dev = dev
        if ok >= config.min_ok or (ok >= 1 and dev == 0.0):
            return IRLSState(
                beta, mu, eta, z, transformed, True, it, dev, float(eps),
                absorb_iters, separation_mask.copy(),
            )

        if eps < hdfe_tol:
            hdfe_tol = max(min(0.1 * hdfe_tol, alt_tol), highest_tol)
            scaled = max(0.1 * eps, np.finfo(float).eps)
            alt_tol = 10.0 ** (-math.ceil(math.log10(1.0 / scaled)))

    return IRLSState(
        beta, mu, eta, z_prev, transformed, False, config.max_iter,
        prev_dev, float(eps or np.inf), absorb_iters, separation_mask.copy(),
    )
