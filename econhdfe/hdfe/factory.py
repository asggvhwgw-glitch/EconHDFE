"""Single construction registry for standard FE absorbers.

Estimators, ``FEPlan`` and ``WeightedFEProjector`` construct absorbers only
here. Kinds without a registered builder use ``HDFEAbsorber``, whose
constructor rejects unknown methods.
"""
from __future__ import annotations
import numpy as np
from .absorber import HDFEAbsorber
from .two_way import TwoWayFEAbsorber
from .within_backend import WithinFEAbsorber
from .projection import resolve_threads
from .workspace import absorption_build_cost, slope_build_bytes
from ..planner import scheduler

_DEFAULT = object()


def _passed(**settings):
    return {name: value for name, value in settings.items() if value is not _DEFAULT}


def _build_within(groups, weights, tol, max_iter, absorb_threads, memory_budget_mb, options):
    # Constructing the adapter imports the optional native module.
    return WithinFEAbsorber(groups, weights=weights, tol=tol, **_passed(
        max_iter=max_iter, absorb_threads=absorb_threads,
        projection_memory_budget_mb=memory_budget_mb,
    ))


def _build_twoway(groups, weights, tol, max_iter, absorb_threads, memory_budget_mb, options):
    absorber = TwoWayFEAbsorber(groups, weights=weights, tol=tol, **_passed(max_iter=max_iter))
    if absorb_threads is _DEFAULT:
        return absorber
    return absorber._configure_execution(absorb_threads=absorb_threads)


def _build_generic(groups, weights, tol, max_iter, absorb_threads, memory_budget_mb, options):
    return HDFEAbsorber(groups, weights=weights, tol=tol, **options, **_passed(
        max_iter=max_iter, absorb_threads=absorb_threads,
        projection_memory_budget_mb=memory_budget_mb,
    ))


_BUILDERS = {"within": _build_within, "twoway": _build_twoway}


def build_absorber(kind, groups, *, weights, tol, max_iter=_DEFAULT, absorb_threads=_DEFAULT,
                   memory_budget_mb=_DEFAULT, **options):
    """Construct the absorber registered for ``kind``.

    Omitted settings keep each constructor's default; in particular an
    unconfigured two-way absorber stays single-threaded. ``options`` are
    HDFEAbsorber settings and are ignored by the specialized solvers.
    """
    builder = _BUILDERS.get(kind, _build_generic)
    return builder(groups, weights, tol, max_iter, absorb_threads, memory_budget_mb, options)


def projection_kind(method, *, engine, ngroups, weights=None):
    """``(kind, generic method)`` for FEPlan and WeightedFEProjector.

    With ``weights``, the specialized solvers also require strictly positive
    weights; native within then falls back to exact generic LSMR.
    """
    if method == "within":
        if weights is None or np.all(weights > 0):
            return "within", method
        return "generic", "lsmr"
    if (engine == "optimized" and ngroups == 2 and method != "lsmr"
            and (weights is None or np.all(weights > 0))):
        return "twoway", method
    return "generic", method


def make_standard_absorber(
    groups, slopes, intercepts, *, weights, tol, max_iter, backend, method,
    transform, acceleration, preconditioner, projection_backend, absorb_threads,
    memory_budget_mb,
):
    if method == "within":
        if backend != "numpy" or any(s is not None for s in slopes) or not all(intercepts):
            raise ValueError("method='within' requires CPU pure-intercept categorical FEs")
    if method == "twoway":
        if backend != "numpy":
            raise ValueError("method='twoway' currently requires backend='numpy'")
        if len(groups) != 2 or any(s is not None for s in slopes) or not all(intercepts):
            raise ValueError(
                "method='twoway' requires exactly two intercept-only fixed effects "
                "after FE canonicalization"
            )
    build_bytes = absorption_build_cost(groups, "twoway" if method == "twoway" else "generic")
    build_bytes += slope_build_bytes(groups, slopes)
    if backend == "numpy" and groups:
        # Warm the saturation cache before construction reserves workspace.
        # Small samples/explicit requests do not run a calibration probe.
        resolve_threads(absorb_threads, nobs=len(groups[0]))
    with scheduler.reserve_memory(build_bytes, label="linear_fe_build"):
        return build_absorber(
            method, groups, weights=weights, tol=tol, max_iter=max_iter,
            absorb_threads=absorb_threads, memory_budget_mb=memory_budget_mb,
            slopes=slopes, intercepts=intercepts, backend=backend, method=method,
            transform=transform, acceleration=acceleration, preconditioner=preconditioner,
            projection_backend=projection_backend,
        )
