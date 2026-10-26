from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from ...errors import InputError, SpecificationError
from ...config import ExecutionConfig
from .config import PPMLConfig
from ...hdfe.plan import FEPlan, fe_separated
from .separation_simplex import mixed_simplex_separation
from .separation_relu import relu_separation
from .execution import get_projector


@dataclass(slots=True)
class SeparationResult:
    separated: np.ndarray
    by_method: dict[str, int]
    timings: dict[str, float]
    iterations: dict[str, int]
    solvers: dict[str, str]
    # Rows that became singletons after simplex/ReLU trimming and were then
    # pruned, as upstream's ``trim_separated_obs`` -> ``HDFE.reload()`` does.
    singletons: np.ndarray | None = None


# Upstream order: ppmlhdfe runs FE separation with the singleton stage, then
# simplex, then ReLU; mu is detected inside IRLS.
_UPSTREAM_ORDER = ("fe", "simplex", "relu", "mu")


def initial_sample_mask(y, plan: FEPlan, config: PPMLConfig) -> tuple[np.ndarray, int]:
    """ppmlhdfe first stage: rows dropped before any regressor-based check.

    Singletons are counted by rows (user weights do not enter this stage).
    With ``"fe"`` in ``config.separation``, levels whose outcome sums to zero
    are removed in the same fixed point (``fixed_effects(..., "iweight",
    depvar)``), so FE separation can create singletons and vice versa.
    Returns the dropped-row mask and, for diagnostics, how many of those rows
    pure singleton pruning would have kept.
    """
    n = len(y)
    if not plan.groups:
        return np.zeros(n, dtype=bool), 0
    single = plan.singleton_mask()
    y = np.asarray(y, dtype=np.float64)
    if "fe" not in config.separation or np.all(y > 0):
        return single, 0
    joint = plan.singleton_mask(positive=y > 0)
    n_fe = int(np.sum(joint) - np.sum(single))
    if n_fe and np.all(joint):
        raise InputError(
            "PPML has no finite estimable sample after separation",
            code="identification.no_finite_estimate", stage="separation",
            details={
                "nobs_after_singletons": int(n - np.sum(single)),
                "n_separated": n_fe,
                "separation_by_method": {"fe": n_fe},
                "reason": "ppml_separation_no_finite_mle",
            },
            suggestion="Inspect all-zero FE levels and separated support; changing normalization cannot create a finite PPML estimate.",
        )
    return joint, n_fe


def detect_separation(
    y, X, plan: FEPlan, weights, config: PPMLConfig,
    execution_config: ExecutionConfig | None = None, *, frequency_weights: bool = False,
) -> SeparationResult:
    """Run separation methods in upstream order with explicit stage diagnostics.

    After simplex or ReLU drops rows, singletons created by the trim are
    pruned before the next method (row counts, or fweight mass when
    ``frequency_weights``), and are reported in ``singletons`` rather than
    ``separated``.

    Separation projectors are deliberately local to this call.  In
    particular, simplex uses zero observation weights; sharing that absorber
    with the positive-weight IRLS projector can disable topology-dependent
    optimized paths and increase retained memory.
    """
    execution = ExecutionConfig() if execution_config is None else execution_config
    execution.validate()
    y0 = np.asarray(y, dtype=np.float64)
    X0 = np.asarray(X, dtype=np.float64)
    w0 = np.ones(len(y0)) if weights is None else np.asarray(weights, dtype=np.float64)
    full_sep = np.zeros(len(y0), dtype=bool)
    full_single = np.zeros(len(y0), dtype=bool)
    counts: dict[str, int] = {}
    timings: dict[str, float] = {}
    iterations: dict[str, int] = {}
    solvers: dict[str, str] = {}

    for method in [m for m in _UPSTREAM_ORDER if m in config.separation]:
        t0 = perf_counter()
        if method == "mu":
            counts[method] = 0
            timings[method] = 0.0
            iterations[method] = 0
            solvers[method] = "post-irls"
            continue
        removed = full_sep | full_single
        if bool(np.any(removed)):
            active = ~removed
            if not np.any(active) or np.all(y0[active] > 0):
                counts[method] = 0
                timings[method] = perf_counter() - t0
                iterations[method] = 0
                solvers[method] = "skipped"
                continue
            ya, Xa, wa = y0[active], X0[active], w0[active]
            pa = plan.subset(active) if plan.groups else plan
        else:
            if np.all(y0 > 0):
                counts[method] = 0
                timings[method] = perf_counter() - t0
                iterations[method] = 0
                solvers[method] = "skipped"
                continue
            active = None
            ya, Xa, wa, pa = y0, X0, w0, plan

        if method == "fe":
            local = fe_separated(ya, pa)
            nit = 0
            solver = "structural"
        elif method == "simplex":
            solver_plan, _ = pa.for_engine(config.engine)
            projector = get_projector(
                solver_plan, engine=config.engine, method="map",
                execution=execution, context=None, cache=False,
            )
            info = mixed_simplex_separation(
                ya, Xa, solver_plan, wa, tol=config.simplex_tol,
                hdfe_tol=config.target_inner_tol, max_iter=config.simplex_max_iter,
                engine=config.engine, projector=projector,
            )
            local = info.separated
            nit = int(info.iterations)
            solver = f"{config.engine}:map"
        elif method == "relu":
            solver_plan, _ = pa.for_engine(config.engine)
            relu_method = "map" if config.engine == "optimized" else "lsmr"
            projector = get_projector(
                solver_plan, engine=config.engine, method=relu_method,
                execution=execution, context=None, cache=False,
            )
            info = relu_separation(
                ya, Xa, solver_plan, tol=config.relu_tol, zero_tol=config.relu_zero_tol,
                hdfe_tol=config.target_inner_tol, max_iter=config.relu_max_iter,
                engine=config.engine, projector=projector,
            )
            local = info.separated
            nit = int(info.iterations)
            solver = f"{config.engine}:{relu_method}"
        else:
            raise SpecificationError(f"unknown separation method: {method}", code="specification.separation")

        nlocal = int(np.sum(local))
        if nlocal:
            if active is None:
                full_sep |= local
            else:
                idx = np.flatnonzero(active)
                full_sep[idx[local]] = True
            alive = ~(full_sep | full_single)
            if method != "fe" and plan.groups and np.any(alive):
                fw = w0[alive] if frequency_weights else None
                new = plan.subset(alive).singleton_mask(frequency_weights=fw)
                full_single[np.flatnonzero(alive)[new]] = True
        counts[method] = nlocal
        timings[method] = perf_counter() - t0
        iterations[method] = nit
        solvers[method] = solver
    return SeparationResult(full_sep, counts, timings, iterations, solvers, full_single)
