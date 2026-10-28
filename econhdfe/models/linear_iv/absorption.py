"""Certified refinement of weak control directions after dense IV absorption."""
from __future__ import annotations

from copy import copy
from time import perf_counter
import numpy as np
import scipy.linalg as la
from ...hdfe.absorber import AbsorbInfo
from ...hdfe.protocol import capability


_WEAK_DIRECTION_RATIO = 1e-3


def _scaled_control_block(controls, scale, weights, lo, hi):
    block = np.asarray(controls[lo:hi] / scale, order="F")
    if weights is not None:
        block *= np.sqrt(weights[lo:hi, None])
    return block


def _control_qr(controls, scale, weights, chunk):
    """Compress only the screened difficult design; no tall Q is retained."""
    k = controls.shape[1]
    factor = np.empty((0, k))
    for lo in range(0, len(controls), chunk):
        hi = min(len(controls), lo + chunk)
        block = _scaled_control_block(controls, scale, weights, lo, hi)
        factor = la.qr(np.vstack([factor, block]), mode="raw",
                       overwrite_a=True, check_finite=False)[1]
    return factor


def _certified_direction_projection(absorber, work, thread_kwargs):
    """Bound slow MAP tails and certify the final Krylov residual independently.

    A shallow local absorber reuses the prepared FE operator and execution lock,
    but changing its iteration allowance cannot affect the caller's settings.
    All phases together stay within the original per-direction iteration limit.
    """
    local = copy(absorber)
    budget = int(absorber.max_iter)
    local.max_iter = min(8, budget)
    projected, info = local.residualize_fixed_map(
        work, 1, return_info=True, certify=True, batch_iterations=3, **thread_kwargs,
    )
    phases = {"map_iterations": int(info.iterations), "lsmr_iterations": 0,
              "polish_iterations": 0, "initial_map_certificate": float(info.max_update)}
    used = int(info.iterations)
    if info.converged:
        phases["criterion"] = "iv_control_map_certificate"
        return projected, AbsorbInfo(True, used, info.max_update, phases["criterion"]), phases
    remaining = budget - used
    if remaining > 0:
        # Reserve at most eight inexpensive cleanup sweeps. LSMR's normwise
        # stopping tolerance does not directly bound the maximum FE mean, so
        # a failed independent certificate can tighten its internal tolerance.
        reserve = min(8, max(remaining - 1, 0))
        rounds = []
        internal_tol = float(absorber.tol)
        for _ in range(3):
            allowance = budget - used - reserve
            if allowance <= 0:
                break
            local.max_iter = allowance
            local.tol = internal_tol
            result, krylov = local._residualize_lsmr(projected[:, None])
            projected = result[:, 0]
            phases["lsmr_iterations"] += int(krylov.iterations)
            phases["lsmr_solver_converged"] = bool(krylov.converged)
            used += int(krylov.iterations)
            error = float(local.fe_orthogonality_error(projected))
            rounds.append({"internal_tolerance": internal_tol,
                           "iterations": int(krylov.iterations),
                           "solver_converged": bool(krylov.converged),
                           "certificate": error})
            if error <= absorber.tol or internal_tol <= np.finfo(float).eps:
                break
            # Reprojecting the current residual preserves the original within
            # component. A safety factor targets a certificate below its limit,
            # while both the round count and total iteration count stay bounded.
            internal_tol = max(np.finfo(float).eps,
                               internal_tol * 0.1 * absorber.tol / error)
        phases["lsmr_rounds"] = rounds
        phases["lsmr_certificate"] = error
        local.tol = absorber.tol
        if error > absorber.tol and used < budget:
            local.max_iter = min(8, budget - used)
            projected, polish = local.residualize_fixed_map(
                projected, 1, return_info=True, certify=True, batch_iterations=3,
                **thread_kwargs,
            )
            phases["polish_iterations"] = int(polish.iterations)
            used += int(polish.iterations)
    # LSMR's stopping code is not an FE-orthogonality certificate. The same
    # check and tolerance govern success whether it stopped normally or early.
    error = float(local.fe_orthogonality_error(projected))
    phases["criterion"] = "iv_control_map_lsmr_certificate"
    phases["final_certificate"] = error
    return projected, AbsorbInfo(error <= absorber.tol, used, error, phases["criterion"]), phases


def refine_iv_control_absorption(absorber, original_controls, within_controls, *,
                                 weights=None, chunk_rows=16384):
    """Refine weak control directions in place, preserving every model column.

    Ordinary designs need only a bounded normalized-Gram screen. A difficult
    design gets QR/SVD confirmation; rank-deficient designs are returned to the
    caller's existing omission policy. For each retained weak right direction,
    reproject its original-data combination at unit within scale, using bounded
    MAP/LSMR phases and the original FE certificate. Mapping it back is exactly
    a reversible control-basis change, shared by structural and instrument roles.

    Returns (within_controls, refinement_info_or_None, metadata). The caller
    must apply its usual non-convergence policy to refinement_info. Only pure
    categorical NumPy MAP has the certificate required by this helper.
    """
    controls = np.asarray(within_controls)
    original = np.asarray(original_controls)
    n, k = controls.shape
    timings = {"screen_seconds": 0., "qr_seconds": 0., "construct_seconds": 0.,
               "projection_seconds": 0., "mapback_seconds": 0.}
    metadata = {"applied": False, "weak_directions": 0, "iterations": 0,
                "timings": timings}
    if k < 2 or not getattr(absorber, "groups", ()):
        return controls, None, metadata
    if not capability(absorber, "supports_fe_orthogonality_certificate"):
        metadata["reason"] = "unsupported_absorption_certificate"
        return controls, None, metadata
    started = perf_counter()
    chunk = max(k, min(chunk_rows, max(k, 1_000_000 // k)))
    scale = np.zeros(k)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        np.maximum(scale, np.max(np.abs(controls[lo:hi]), axis=0), out=scale)
    if np.any(scale == 0):
        timings["screen_seconds"] = perf_counter() - started
        metadata["reason"] = "defer_collinearity"
        return controls, None, metadata
    gram = np.zeros((k, k))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        block = _scaled_control_block(controls, scale, weights, lo, hi)
        gram += block.T @ block
    # A Gram eigenvalue selects execution only; it never chooses model rank.
    eigenvalues = la.eigvalsh(gram, check_finite=False)
    timings["screen_seconds"] = perf_counter() - started
    if eigenvalues[0] > _WEAK_DIRECTION_RATIO**2 * eigenvalues[-1]:
        return controls, None, metadata
    started = perf_counter()
    factor = _control_qr(controls, scale, weights, chunk)
    _, singular, vh = la.svd(factor, full_matrices=False, check_finite=False)
    timings["qr_seconds"] = perf_counter() - started
    cutoff = np.finfo(float).eps * max(n, k) * singular[0]
    if len(singular) != k or singular[-1] <= cutoff:
        metadata["reason"] = "defer_collinearity"
        return controls, None, metadata
    weak = np.flatnonzero(singular < _WEAK_DIRECTION_RATIO * singular[0])
    if not len(weak):
        return controls, None, metadata
    metadata["weak_directions"] = int(len(weak))
    metadata["condition_ratio"] = float(singular[-1] / singular[0])
    total_weight = float(n if weights is None else np.sum(weights))
    plan = getattr(absorber, "execution_plan", None)
    thread_kwargs = {} if plan is None else {"absorb_threads": plan.threads}
    infos = []
    direction_phases = []
    max_correction = 0.
    # One N-vector at a time bounds memory even with several weak directions.
    # Extended accumulation is restricted to each small block of the flagged
    # directions, where subtracting shared FE loadings is precision-sensitive.
    for index in weak:
        started = perf_counter()
        direction = vh[index]
        within_scale = singular[index] / np.sqrt(total_weight)
        loading = (direction / scale / within_scale).astype(np.longdouble)
        inverse_loading = direction * scale * within_scale
        work = np.empty(n)
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            work[lo:hi] = original[lo:hi].astype(np.longdouble) @ loading
        timings["construct_seconds"] += perf_counter() - started
        started = perf_counter()
        projected, info, phases = _certified_direction_projection(absorber, work, thread_kwargs)
        timings["projection_seconds"] += perf_counter() - started
        infos.append(info)
        direction_phases.append(phases)
        if not info.converged:
            # Leave this direction unchanged; report failure for the caller's
            # explicit allow_nonconverged policy rather than hide it as success.
            break
        started = perf_counter()
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            current = controls[lo:hi].astype(np.longdouble) @ loading
            delta = projected[lo:hi].astype(np.longdouble) - current
            correction = np.asarray(delta[:, None] * inverse_loading, dtype=float)
            max_correction = max(max_correction, float(np.max(np.abs(correction))))
            controls[lo:hi] += correction
        timings["mapback_seconds"] += perf_counter() - started
        metadata["applied"] = True
    worst = max(infos, key=lambda info: (not info.converged, info.iterations, info.max_update))
    metadata.update(
        iterations=int(max(info.iterations for info in infos)),
        total_direction_iterations=int(sum(info.iterations for info in infos)),
        certified=bool(all(info.converged for info in infos)),
        max_update=float(max(info.max_update for info in infos)),
        max_control_correction=max_correction,
        criterion=worst.criterion,
        direction_phases=direction_phases,
    )
    return controls, worst, metadata
