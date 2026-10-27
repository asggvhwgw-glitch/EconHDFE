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


@dataclass(slots=True)
class InitialSampleSelection:
    dropped: np.ndarray
    singletons: np.ndarray
    separated: np.ndarray
    n_fe: int


def initial_sample_selection(y, plan, config, *, weights=None, weight_kind="none"):
    """Versioned sample semantics, separate from the numerical FE peeler.

    Compatibility preserves upstream's combined initial singleton/zero-level
    count. Unified mode counts frequencies at every stage and keeps the two
    removal reasons disjoint. Both operate only on validated positive weights.
    """
    from ...hdfe.sample import peel_sample
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    unified = config.singleton_mode == "unified"
    fw = weights if unified and weight_kind == "fweight" else None
    fe_on = "fe" in config.separation and (unified or config.drop_singletons)
    selected = peel_sample(
        plan.groups, nobs=n, frequency_weights=fw,
        positive=y > 0 if fe_on else None, drop_singletons=config.drop_singletons,
    )
    dropped = ~selected.keep
    if unified:
        singles, separated = selected.singletons, selected.separated
        n_fe = int(np.sum(separated))
    else:
        singles, separated = dropped, np.zeros(n, dtype=bool)
        pure = peel_sample(plan.groups, nobs=n, drop_singletons=config.drop_singletons)
        n_fe = int(np.sum(dropped) - np.sum(~pure.keep))
    if n_fe and np.all(dropped):
        raise InputError(
            "PPML has no finite estimable sample after separation",
            code="identification.no_finite_estimate", stage="separation",
            details={"nobs_after_singletons": int(n - np.sum(singles)),
                     "n_separated": n_fe, "separation_by_method": {"fe": n_fe},
                     "reason": "ppml_separation_no_finite_mle"},
            suggestion="Inspect all-zero FE levels and separated support.",
        )
    return InitialSampleSelection(dropped, singles, separated, n_fe)


def initial_sample_mask(y, plan, config, *, weights=None, weight_kind="none"):
    """Backward-compatible internal tuple adapter."""
    selected = initial_sample_selection(y, plan, config, weights=weights, weight_kind=weight_kind)
    return selected.dropped, selected.n_fe


def sample_policy_diagnostics(config, sample, weights, weight_kind):
    """Counts only: never place row-level identifiers in diagnostic reports."""
    rows = int(np.sum(sample))
    observations = float(np.sum(np.asarray(weights)[sample])) if weight_kind == "fweight" else float(rows)
    return {
        "mode": config.singleton_mode,
        "profile": "stata-source-20261027" if config.singleton_mode == "stata_compat" else "frequency-invariant-v1",
        "initial_count": "frequency" if weight_kind == "fweight" and config.singleton_mode == "unified" else "rows",
        "post_separation_count": "frequency" if weight_kind == "fweight" else "rows",
        "drop_singletons": config.drop_singletons,
        "statistical_rows": rows, "statistical_observations": observations,
        "working_rows": rows, "computational_elimination": False,
        "separation_methods": tuple(config.separation),
    }


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
        if method == "fe" and not config.drop_singletons and config.singleton_mode == "stata_compat":
            counts[method], timings[method], iterations[method], solvers[method] = 0, 0.0, 0, "disabled-with-singletons"
            continue
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
            if plan.groups and np.any(alive):
                fw = w0[alive] if frequency_weights else None
                if config.singleton_mode == "unified":
                    from ...hdfe.sample import peel_sample
                    new = peel_sample(
                        plan.subset(alive).groups, frequency_weights=fw,
                        positive=y0[alive] > 0 if "fe" in config.separation else None,
                        drop_singletons=config.drop_singletons,
                    )
                    idx = np.flatnonzero(alive)
                    full_single[idx[new.singletons]] = True
                    full_sep[idx[new.separated]] = True
                    counts["fe"] = counts.get("fe", 0) + int(np.sum(new.separated))
                elif method != "fe" and config.drop_singletons:
                    new = plan.subset(alive).singleton_mask(frequency_weights=fw)
                    full_single[np.flatnonzero(alive)[new]] = True
        counts[method] = counts.get(method, 0) + nlocal
        timings[method] = perf_counter() - t0
        iterations[method] = nit
        solvers[method] = solver
    return SeparationResult(full_sep, counts, timings, iterations, solvers, full_single)
