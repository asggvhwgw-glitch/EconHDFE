"""Read-only projection adapters for the FE complexity-frontier experiment.

The four backend names are deliberately explicit. ``within`` and ``fereg``
call the pinned upstream kernels through the same CI-built dual adapter, with
cross-kernel fallback disabled. ``native_auto`` is EconHDFE's native-within
adapter, including its existing SmallSchur / exact-tuple / within routing.
These are projection calls, never complete regression estimates.

Only SciPy's operator applications are instrumented here. Native iteration
counts are *not* substituted for unavailable matrix-vector-product counts.
Independent error, feasibility, rank and spectrum oracles belong to the runner.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


BACKEND_NAMES = ("scipy_lsmr", "within", "native_auto", "fereg")
_MODULES = {
    "scipy_lsmr": "scipy.sparse.linalg",
    "within": "econhdfe_dual",
    "native_auto": "_econhdfe_within",
    "fereg": "econhdfe_dual",
}
_DISTRIBUTIONS = {
    "scipy_lsmr": "scipy", "within": "econhdfe-dual-prototype",
    "native_auto": "econhdfe-within", "fereg": "econhdfe-dual-prototype",
}


class BackendUnavailable(RuntimeError):
    """An optional backend could not be imported; this is not a passing run."""


def _field(case: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(case, Mapping) and name in case:
            return case[name]
        if hasattr(case, name):
            return getattr(case, name)
    return default


def _module(name: str):
    if name not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")
    try:
        return importlib.import_module(_MODULES[name])
    except (ImportError, OSError) as exc:
        raise BackendUnavailable(f"{name}: cannot import {_MODULES[name]}: {exc}") from exc


def probe_backend(name: str) -> dict[str, Any]:
    """Inspect availability without silently replacing an unavailable backend."""
    try:
        module = _module(name)
    except BackendUnavailable as exc:
        return {"name": name, "available": False, "reason": str(exc)}
    return {"name": name, "available": True, "provenance": _provenance(name, module)}


def available_backends() -> list[str]:
    return [name for name in BACKEND_NAMES if probe_backend(name)["available"]]


@lru_cache(maxsize=16)
def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _provenance(name: str, module: Any) -> dict[str, Any]:
    distribution = _DISTRIBUTIONS[name]
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    result: dict[str, Any] = {
        "distribution": distribution, "distribution_version": version,
        "numpy_version": np.__version__,
    }
    if name == "scipy_lsmr":
        result.update(kernel="scipy.sparse.linalg.lsmr", operator="matrix_free_weighted_categorical",
                      preconditioner="right_diagonal_column_norm", precision="float64",
                      normal_equations_formed=False, rank_gauge_removed=False)
    else:
        result["precision"] = "float64"
        result["upstream_within_revision"] = getattr(module, "WITHIN_REV", getattr(module, "UPSTREAM_REV", None))
        if name in {"within", "fereg"}:
            result.update(adapter="econhdfe_dual.DualAbsorber", requested_kernel=name,
                          upstream_fereg_revision=getattr(module, "FEREG_REV", None),
                          kernel_entry="within::Solver::solve_batch" if name == "within" else "fereg_core::FeSolver::residualize",
                          cross_kernel_fallback=False, correction_attempts_allowed=True,
                          thread_control="private_native_pool")
        else:
            result.update(adapter="_econhdfe_within.NativeSolver", strategy="auto",
                          thread_control="private_native_pool")
        parent = Path(module.__file__).resolve().parent
        binaries = sorted(parent.glob("*.so")) + sorted(parent.glob("*.pyd"))
        result["loaded_extension_files"] = [
            {"name": p.name, "sha256": _file_hash(str(p))} for p in binaries
        ]
    return result


def _inputs(case: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], bool]:
    raw = _field(case, "codes", "groups")
    rhs = _field(case, "rhs", "y", "Y")
    if raw is None or rhs is None:
        raise ValueError("case must supply codes/groups and rhs/y")
    if isinstance(raw, (list, tuple)) and raw and all(np.asarray(g).ndim == 1 for g in raw):
        raw = np.column_stack(raw)
    codes = np.asarray(raw)
    y = np.asarray(rhs, dtype=np.float64)
    vector = y.ndim == 1
    if vector:
        y = y[:, None]
    if codes.ndim != 2 or not codes.shape[0] or not codes.shape[1] or codes.dtype.kind not in "iu":
        raise ValueError("codes must be a nonempty N x K integer array")
    n, k = codes.shape
    if y.ndim != 2 or y.shape[0] != n or not y.shape[1] or not np.isfinite(y).all():
        raise ValueError("rhs must be a finite nonempty N x P array or N vector")
    if np.any(codes < 0) or np.any(codes > np.iinfo(np.uint32).max):
        raise ValueError("categorical labels must fit nonnegative uint32")
    compact = np.empty((n, k), dtype=np.uint32, order="F")
    levels = []
    for j in range(k):
        column = codes[:, j]
        maximum = int(column.max())
        # Avoid a huge allocation for sparse labels. Dense inputs take a linear
        # check and copy; sparse labels use an exact, timed relabelling.
        dense = maximum < n and np.all(np.bincount(column.astype(np.int64), minlength=maximum + 1) > 0)
        if dense:
            compact[:, j] = column
            levels.append(maximum + 1)
        else:
            unique, inverse = np.unique(column, return_inverse=True)
            compact[:, j] = inverse
            levels.append(len(unique))
    raw_weights = _field(case, "weights", "w")
    w = np.ones(n, dtype=np.float64) if raw_weights is None else np.asarray(raw_weights, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("the common benchmark contract requires strictly positive finite weights")
    # Global weight scaling does not change the weighted FE projection. Native
    # adapters also normalize numerical weights; make the common input explicit.
    w = np.ascontiguousarray(w / np.max(w))
    if np.any(w == 0):
        raise ValueError("weight normalization underflows; float64 contract cannot represent this case")
    return compact, np.ascontiguousarray(y), w, levels, vector


def _lsmr_solver(module: Any, codes: np.ndarray, w: np.ndarray, levels: list[int]):
    n, k = codes.shape
    offsets = np.concatenate(([0], np.cumsum(levels))).astype(np.int64)
    g = int(offsets[-1])
    sqrtw = np.sqrt(w)
    diagonal = np.concatenate([np.bincount(codes[:, j], weights=w, minlength=levels[j]) for j in range(k)])
    scale = 1.0 / np.sqrt(diagonal)
    counts = {"matvec": 0, "rmatvec": 0}

    def fitted(theta: np.ndarray) -> np.ndarray:
        result = np.zeros(n, dtype=np.float64)
        for j in range(k):
            result += theta[offsets[j] + codes[:, j]]
        return result

    def matvec(x: np.ndarray) -> np.ndarray:
        counts["matvec"] += 1
        return sqrtw * fitted(scale * np.asarray(x).reshape(-1))

    def rmatvec(u: np.ndarray) -> np.ndarray:
        counts["rmatvec"] += 1
        weighted = sqrtw * np.asarray(u).reshape(-1)
        result = np.empty(g, dtype=np.float64)
        for j in range(k):
            result[offsets[j]:offsets[j + 1]] = np.bincount(codes[:, j], weights=weighted, minlength=levels[j])
        return scale * result

    operator = module.LinearOperator((n, g), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
    return operator, scale, sqrtw, fitted, counts


def run_backend(name: str, case: Any, tol: float = 1e-10, max_iter: int = 16_000,
                threads: int = 1, *, scratch_bytes: int = 1 << 30) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(residual, metrics)`` for one cold projection.

    Dependency import/provenance inspection is outside the numerical timings.
    ``setup_s`` includes input validation/layout conversion and construction;
    ``solve_s`` includes the complete projection call and output copy/layout.
    Deferred preconditioner construction and native certification can therefore
    occur inside ``solve_s``. ``total_s`` includes both; external oracles do not.
    Native failures that raise are intentionally left for the runner to record.
    A returned nonconvergence flag is retained, never converted into success.
    """
    if not np.isfinite(tol) or not 0 < tol <= 1:
        raise ValueError("tol must lie in (0, 1]")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("threads must be a positive integer")
    module = _module(name)
    provenance = _provenance(name, module)
    start = perf_counter()
    codes, y, w, levels, vector = _inputs(case)
    n, k = codes.shape
    metrics: dict[str, Any] = {
        "backend": name, "provenance": provenance, "nobs": n, "n_factors": k,
        "levels": levels, "n_rhs": y.shape[1], "threads_requested": int(threads),
        "tolerance_requested": float(tol), "max_iter_requested": int(max_iter),
        "input_weight_normalization": "divide_by_max_positive_weight",
        "actual_matvec_counts": None, "actual_preconditioner_apply_counts": None,
        "count_kind": "unavailable_native", "reported_iterations": None,
        "timing_scope": "setup=validation_layout_constructor; solve=full_call_and_output_copy; deferred_setup_in_solve; excludes_import_provenance_and_external_oracles",
    }
    if name == "scipy_lsmr":
        operator, scale, sqrtw, fitted, counts = _lsmr_solver(module, codes, w, levels)
        setup_end = perf_counter()
        out = np.empty_like(y)
        details = []
        for j in range(y.shape[1]):
            before = counts.copy()
            sol = module.lsmr(operator, sqrtw * y[:, j], atol=tol, btol=tol,
                              conlim=0.0, maxiter=int(max_iter))
            # Reconstruction is accounted separately, not misreported as an
            # operator call made by the iterative solver.
            out[:, j] = y[:, j] - fitted(scale * sol[0])
            details.append({"istop": int(sol[1]), "iterations": int(sol[2]),
                            "normr": float(sol[3]), "normar": float(sol[4]),
                            "condition_estimate": float(sol[6]),
                            "matvec": counts["matvec"] - before["matvec"],
                            "rmatvec": counts["rmatvec"] - before["rmatvec"]})
        flags = [d["istop"] in {0, 1, 2, 4, 5} for d in details]
        metrics.update(reported_iterations=[d["iterations"] for d in details], flags=flags,
                       converged=bool(all(flags)), solver_details=details,
                       actual_matvec_counts={**counts, "total": sum(counts.values())},
                       count_kind="instrumented_matrix_free_operator_calls",
                       reconstructed_fitted_vectors=y.shape[1],
                       operator_incidence_visits=sum(counts.values()) * n * k,
                       iteration_budget_scope="per_rhs_lsmr", conlim=0.0,
                       solver_reported_setup_s=None, threads_effective=1,
                       thread_note="NumPy categorical bincount baseline; requested native threads do not parallelize this operator")
    elif name in {"within", "fereg"}:
        solver = module.DualAbsorber(codes, weights=w, backend=name, threads=int(threads),
                                    scratch_bytes=int(scratch_bytes), allow_fallback=False)
        setup_end = perf_counter()
        out, info = solver.residualize(y, tol=float(tol), max_iter=int(max_iter), return_info=True)
        out = np.array(out, dtype=np.float64, copy=True, order="C")
        metrics.update(reported_iterations=info.get("iterations"),
                       flags=[bool(info.get("accepted", False))] * y.shape[1],
                       converged=bool(info.get("accepted", False)), solver_details=info,
                       solver_reported_setup_s=info.get("setup_seconds"),
                       iteration_budget_scope=info.get("max_iter_scope"), threads_effective=int(threads))
    else:
        solver = module.NativeSolver(np.asfortranarray(codes), weights=w, strategy="auto",
                                     threads=int(threads), workspace_bytes=int(scratch_bytes))
        setup_end = perf_counter()
        out, flags, iterations, errors = solver.solve(y, tol=float(tol), max_iter=int(max_iter))
        out = np.array(out, dtype=np.float64, copy=True, order="C")
        flags = [bool(x) for x in flags]
        metrics.update(reported_iterations=[int(x) for x in iterations], flags=flags,
                       converged=bool(all(flags)), solver_reported_setup_s=None,
                       iteration_budget_scope="per_rhs_total_initial_continuation_polish_fallback",
                       threads_effective=int(threads),
                       solver_details={"route": solver.route, "n_cells": int(solver.n_cells),
                                       "certificate": [float(x) for x in errors],
                                       "fallback_count": int(solver.fallback_count),
                                       "schur_condition": solver.schur_condition})
    end = perf_counter()
    metrics.update(setup_s=setup_end - start, solve_s=end - setup_end, total_s=end - start,
                   output_finite=bool(np.isfinite(out).all()))
    return (out[:, 0] if vector else out), metrics
