"""Centered, smoothed semismooth-Newton GPU proposals; no novelty claim.

The final estimator is still the finite nonsmoothed QR LP checked elsewhere.
This module only proposes a beta/active set. Smoothing, Newton continuation,
GPU batching, and active Hessians are established ingredients. Failure and
line-search costs are retained. It intentionally leaves gpu_candidate.py intact.

The optimization variable is delta=beta-beta0. Original FP64 residuals at
beta0 are formed once per block. Objective changes use cancellation-resistant
rowwise differences, including the finite pseudo observation. This matters
when FP32 total losses dwarf the improvement available from a good one-step fit.
"""
import time
import numpy as np
try:
    from .gpu_candidate import _prepare_windows_import_paths, _original_diagnostics
except ImportError:
    from .gpu_pdhg import _prepare_windows_import_paths, _original_diagnostics


def solve_batch(X, y, tau, W, beta0, *, batch_size=16, max_iter=5,
                mu_factors=(1., .4, .15, .05, .015, .004),
                initial_mu=None, max_seconds=3.0, ridge_relative=1e-6,
                return_dual=False):
    """Return B-by-p proposals and diagnostics for dense X and p-by-B W.

    max_iter is the maximum Newton iterations at EACH continuation parameter.
    The Moreau envelope has dual a=clip(r/mu,tau-1,tau), gradient -Z'a,
    Hessian Z_J'Z_J/mu on strictly interior dual coordinates. Each replicate
    has its own active Hessian and backtracking step, while residual/gradient
    GEMMs share X. A recorded ridge modifies only the Newton direction.
    All candidates retain the pseudo row; smoothing never certifies a fit.
    continuation_complete means all scheduled stages were attempted, not that
    they converged. Inspect line_failures and per-stage stationarity explicitly.
    """
    wall_start = time.perf_counter()
    if hasattr(X, "tocsr"):
        raise TypeError("v2 accepts dense X only.")
    x, y, w = (np.asarray(a, dtype=np.float64) for a in (X, y, W))
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("Expected nonempty dense X(n,p).")
    n, p = x.shape
    if y.shape != (n,) or w.ndim != 2 or w.shape[0] != p or w.shape[1] == 0:
        raise ValueError("Expected y(n) and W(p,B).")
    if not 0 < tau < 1 or batch_size < 1 or max_iter < 1:
        raise ValueError("Invalid tau, batch size, or iteration budget.")
    if not mu_factors or min(mu_factors) <= 0 or ridge_relative < 0:
        raise ValueError("Positive smoothing factors and nonnegative ridge required.")
    b = w.shape[1]
    initial = np.asarray(beta0, dtype=np.float64)
    if initial.shape == (p,):
        initial = np.broadcast_to(initial, (b, p)).copy()
    if initial.shape != (b, p):
        raise ValueError("beta0 must be p or B-by-p.")
    if not all(np.isfinite(a).all() for a in (x, y, w, initial)):
        raise ValueError("Nonfinite input.")
    m = float(n * np.max(np.abs(y)))
    _prepare_windows_import_paths()
    import cupy as cp
    stream = cp.cuda.get_current_stream()
    setup_start = time.perf_counter()
    gx = cp.asarray(x, dtype=cp.float32)
    stream.synchronize()
    result = initial.copy()
    info = dict(method="centered_Moreau_Newton_proposal", certificate=False,
                novelty_claim=False, shape=[n, p, b], tau=float(tau),
                max_iter_per_mu=int(max_iter), mu_factors=list(mu_factors),
                ridge_relative=ridge_relative, blocks=[], replicates=[], timings={})
    info["timings"]["common_upload_seconds"] = time.perf_counter() - setup_start
    clock = time.perf_counter()
    returned_duals = []
    for first in range(0, b, batch_size):
        last = min(b, first + batch_size)
        count = last - first
        block_start = time.perf_counter()
        # Center BEFORE rounding: X32 @ beta0_32 would reintroduce large-fit
        # cancellation and alter the local residual ordering unnecessarily.
        residual0 = y[:, None] - x @ initial[first:last].T
        pseudo0 = m - np.sum((w[:, first:last] / tau) * initial[first:last].T, axis=0)
        width = (2 * np.median(np.abs(residual0)) * np.sqrt(p / n)
                 if initial_mu is None else float(initial_mu))
        width = max(float(width), 1e-5)
        gr0 = cp.asarray(residual0, dtype=cp.float32)
        gr_pseudo = cp.asarray(pseudo0, dtype=cp.float32)
        gd = cp.asarray(w[:, first:last] / tau, dtype=cp.float32)
        delta = cp.zeros((p, count), dtype=cp.float32)
        best = delta.copy()
        best_value = cp.zeros(count, dtype=cp.float64)
        best_mu = cp.full(count, width, dtype=cp.float64)
        endpoint0 = cp.where(gr0 >= 0, tau, tau - 1)
        endpoint_pseudo0 = cp.where(gr_pseudo >= 0, tau, tau - 1)
        eye = cp.eye(p, dtype=cp.float32)[None, :, :]
        history = []
        status = "continuation_complete"
        line_failures = 0

        def changes(coeff, mu=None):
            shift = gx @ coeff
            shift0 = cp.sum(gd * coeff, axis=0)
            r = gr0 - shift
            r0 = gr_pseudo - shift0
            if mu is None:
                # Exact nonsmooth change: outside a sign crossing it is linear.
                term = -endpoint0 * shift + cp.where((gr0 >= 0) != (r >= 0), cp.abs(r), 0)
                term0 = -endpoint_pseudo0 * shift0 + cp.where((gr_pseudo >= 0) != (r0 >= 0), cp.abs(r0), 0)
            else:
                aa = cp.clip(r / mu, tau - 1, tau)
                a0 = cp.clip(gr0 / mu, tau - 1, tau)
                ss = cp.clip(r0 / mu, tau - 1, tau)
                s0 = cp.clip(gr_pseudo / mu, tau - 1, tau)
                term = -aa * shift + (aa - a0) * (gr0 - .5 * mu * (aa + a0))
                term0 = -ss * shift0 + (ss - s0) * (gr_pseudo - .5 * mu * (ss + s0))
            return cp.sum(term, axis=0, dtype=cp.float64) + term0

        for factor in mu_factors:
            mu = width * factor
            for iteration in range(1, max_iter + 1):
                residual = gr0 - gx @ delta
                rp = gr_pseudo - cp.sum(gd * delta, axis=0)
                dual = cp.clip(residual / mu, tau - 1, tau)
                pseudo_dual = cp.clip(rp / mu, tau - 1, tau)
                stationarity = gx.T @ dual + gd * pseudo_dual[None, :]
                hessians = []
                sizes = []
                for j in range(count):
                    indices = cp.flatnonzero((residual[:, j] > mu * (tau - 1)) &
                                             (residual[:, j] < mu * tau))
                    local = gx[indices]
                    hessian = (local.T @ local) / mu
                    pseudo_interior = (rp[j] > mu * (tau - 1)) & (rp[j] < mu * tau)
                    hessian += cp.outer(gd[:, j], gd[:, j]) * pseudo_interior / mu
                    hessians.append(hessian)
                    sizes.append(int(indices.size))
                hessian = cp.stack(hessians)
                ridge = cp.maximum(cp.trace(hessian, axis1=1, axis2=2) / p,
                                   cp.float32(1.0)) * max(ridge_relative, 1e-8)
                hessian += ridge[:, None, None] * eye
                try:
                    direction = cp.linalg.solve(hessian, stationarity.T[:, :, None])[:, :, 0].T
                except cp.linalg.LinAlgError:
                    status = "linear_solve_failure"
                    break
                slope = cp.sum(stationarity * direction, axis=0, dtype=cp.float64)
                current_value = changes(delta, mu)
                alpha = cp.ones(count, dtype=cp.float32)
                accepted = cp.zeros(count, dtype=cp.bool_)
                candidate = delta.copy()
                for backtrack in range(14):
                    trial = delta + direction * alpha[None, :]
                    trial_value = changes(trial, mu)
                    ok = cp.isfinite(trial_value) & (trial_value <= current_value - 1e-4 * alpha * slope + 1e-9)
                    take = ok & ~accepted
                    candidate = cp.where(take[None, :], trial, candidate)
                    accepted |= ok
                    if bool(cp.all(accepted)):
                        break
                    alpha = cp.where(accepted, alpha, .5 * alpha)
                failed = int(cp.count_nonzero(~accepted))
                line_failures += failed
                delta = candidate
                value = changes(delta)
                take = value < best_value
                best = cp.where(take[None, :], delta, best)
                best_value = cp.minimum(value, best_value)
                best_mu = cp.where(take, mu, best_mu)
                eq_l2 = cp.asnumpy(cp.linalg.norm(stationarity, axis=0))
                history.append(dict(mu=float(mu), iteration=iteration, active_sizes=sizes,
                                    max_stationarity_l2=float(np.max(eq_l2)),
                                    min_step=float(cp.min(alpha)), backtracks=backtrack,
                                    line_failures=failed,
                                    best_objective_change=cp.asnumpy(best_value).tolist()))
                if not np.isfinite(eq_l2).all():
                    status = "nonfinite_iteration"
                    break
                if max_seconds is not None and time.perf_counter() - clock >= max_seconds:
                    status = "time_limit"
                    break
                if np.max(eq_l2) < 1e-3 or failed == count:
                    break
            if status != "continuation_complete":
                break
        stream.synchronize()
        block = dict(first=first, last=last, status=status, initial_mu=width,
                     history=history, line_failures=line_failures,
                     total_seconds=time.perf_counter() - block_start)
        delta_cpu = cp.asnumpy(best).T.astype(np.float64)
        result[first:last] = initial[first:last] + delta_cpu
        # Do not combine an earlier selected candidate with an arbitrarily
        # smaller failed stage's mu: retain that candidate's smoothing scale.
        best_mu_cpu = cp.asnumpy(best_mu)
        dual_cpu = np.clip((residual0 - x @ delta_cpu.T) / best_mu_cpu, tau - 1, tau)
        pseudo_cpu = np.clip((pseudo0 - np.sum((w[:, first:last] / tau) * delta_cpu.T,
                                               axis=0)) / best_mu_cpu, tau - 1, tau)
        for j in range(count):
            diag = _original_diagnostics(x, y, tau, w[:, first+j], result[first+j],
                                         dual_cpu[:, j], pseudo_cpu[j], m)
            diag.update(replicate=first+j, status=status,
                        coefficient_move_inf=float(np.max(np.abs(delta_cpu[j]))),
                        smoothing_at_proposal=float(best_mu_cpu[j]))
            info["replicates"].append(diag)
            if return_dual:
                returned_duals.append(np.r_[dual_cpu[:, j], pseudo_cpu[j]])
        info["blocks"].append(block)
        if status != "continuation_complete":
            for j in range(last, b):
                info["replicates"].append(dict(replicate=j, status="not_run_after_"+status,
                                                certificate=False))
                if return_dual:
                    returned_duals.append(None)
            break
    info["timings"]["all_blocks_seconds"] = time.perf_counter() - clock
    info["timings"]["total_seconds"] = time.perf_counter() - wall_start
    if return_dual:
        info["dual_batch"] = returned_duals
    return result, info
