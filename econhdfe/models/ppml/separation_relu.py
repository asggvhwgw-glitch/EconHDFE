from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ...hdfe.plan import FEPlan
from ...compute.linalg import weighted_lstsq


@dataclass(slots=True)
class ReLUInfo:
    """ReLU outcome.

    ``converged`` means the full problem was certified, so ``separated`` is
    the complete ReLU separation set. Otherwise ``status`` says why the loop
    stopped and ``separated`` holds only the candidates that a restricted
    re-check certified (possibly none); further separation is not ruled out.
    """

    separated: np.ndarray
    iterations: int
    converged: bool
    status: str = "certified"  # certified | maxiter | stalled
    candidates: int = 0
    verified: int = 0


def _relu_iterate(boundary, X, plan, projector, engine, *, tol, zero_tol, max_iter, stall_rounds):
    """CGZ ReLU loop with ``boundary`` as the inequality set.

    Rows outside ``boundary`` receive the large weight M, which enforces
    z = Xb + Da = 0 there. Returns (separated, iterations, status, u).
    """
    n = len(boundary)
    u = boundary.astype(np.float64)
    M = float(np.ceil(np.finfo(float).eps ** -0.5))
    # Upstream create_mask(N, M, boundary, 1): equality constraints
    # receive the large weight, boundary observations weight one.
    weights = np.where(boundary, 1.0, M)
    norm_b = np.linalg.norm(X[boundary], 1) if X.size else 0.0
    norm_i = np.linalg.norm(X[~boundary], 1) if X.size else 0.0
    ratio = max(norm_b / norm_i, 1.0) if norm_i > 0 else 1.0

    if plan.groups:
        method = "map" if engine == "optimized" else "lsmr"
        if projector is None:
            from ...hdfe.weighted_projection import WeightedFEProjector
            projector = WeightedFEProjector(plan.groups, engine=engine, method=method)
        absorber = projector.prepare(weights, tol=max(tol * tol, 1e-13))
    else:
        absorber = None
    xtilde = None
    utilde = None
    u_last = None
    support_last = None
    unchanged = 0
    for it in range(1, max_iter + 1):
        if it == 1:
            utilde_input = u
        else:
            utilde_input = u + utilde - u_last
        u_last = u.copy()

        if absorber is not None:
            utilde = absorber.residualize(utilde_input)
            if it == 1:
                xtilde = absorber.residualize(X) if X.shape[1] else X
        else:
            utilde = utilde_input.copy()
            if it == 1:
                xtilde = X

        if X.shape[1]:
            _, resid = weighted_lstsq(xtilde, utilde, weights, fast=False)
        else:
            resid = utilde.copy()
        epsilon = max(float(np.dot(u, u) / M * ratio * ratio), 1e-8)
        delta = epsilon + tol
        xbd = u - resid
        xbd[~boundary] = 0.0
        near = boundary & (xbd >= -0.1 * delta) & (xbd <= delta)
        xbd[near] = 0.0
        if np.all(xbd[boundary] >= 0.0):
            return boundary & (xbd > 0.0), it, "certified", u

        resid[np.abs(resid) <= zero_tol] = 0.0
        if np.min(resid[boundary]) >= 0.0:
            xbd[boundary & (resid > delta)] = 0.0
            return boundary & (xbd > 0.0), it, "certified", u
        u[boundary] = np.maximum(xbd[boundary], 0.0)

        # Stall: neither the candidate support nor the violating rows move.
        # The loop is then a fixed point of the thresholded update, and more
        # rounds only repeat the same full FE solve.
        support = np.flatnonzero(boundary & ((u > 0.0) | (xbd < 0.0)))
        if support_last is not None and np.array_equal(support, support_last):
            unchanged += 1
        else:
            unchanged = 0
        support_last = support
        if stall_rounds and unchanged >= stall_rounds:
            return np.zeros(n, dtype=bool), it, "stalled", u

    return np.zeros(n, dtype=bool), max_iter, "maxiter", u


def relu_separation(
    y, X, plan: FEPlan, *, tol=1e-4, zero_tol=1e-8, max_iter=100,
    stall_rounds=20, engine="replica", projector=None,
):
    """IR/ReLU separation certificate following Correia-Guimaraes-Zylkin.

    When the loop does not certify the full problem, the rows still positive
    in the last iterate are re-checked by the same loop with only those rows
    as the inequality set; every other row, including the remaining zeros, is
    held at z = 0. A certificate of that restricted problem is a certificate
    of the original one (z = 0 is admissible on y = 0), so the rows it flags
    are separated. Rows it does not flag are left in the sample, and the
    caller must treat the result as uncertified.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = len(y)
    boundary = y == 0
    if not np.any(boundary):
        return ReLUInfo(np.zeros(n, dtype=bool), 0, True)

    opts = dict(tol=tol, zero_tol=zero_tol, max_iter=max_iter, stall_rounds=stall_rounds)
    separated, iterations, status, u = _relu_iterate(boundary, X, plan, projector, engine, **opts)
    if status == "certified":
        return ReLUInfo(separated, iterations, True)

    candidates = boundary & (u > 0.0)
    verified = np.zeros(n, dtype=bool)
    if np.any(candidates):
        sub, sub_it, sub_status, _ = _relu_iterate(candidates, X, plan, projector, engine, **opts)
        iterations += sub_it
        if sub_status == "certified":
            verified = sub
    return ReLUInfo(
        verified, iterations, False, status,
        candidates=int(np.sum(candidates)), verified=int(np.sum(verified)),
    )
