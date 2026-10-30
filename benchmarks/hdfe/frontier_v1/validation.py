"""Independent numerical checks for the FE complexity-frontier benchmark.

No check here reads a backend's ``converged`` flag.  In particular, a small
normal-equation residual is recorded separately from error against the supplied
projection oracle.  The latter is what determines forward accuracy.

The dense diagnostics are *floating-point estimates*, with explicit size and
rank thresholds.  They are never presented as exact rank or condition proofs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_columns(value: Any, name: str, n: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or (n is not None and out.shape[0] != n):
        raise ValueError(f"{name} must have shape (n, rhs)")
    return out


def _groups(groups: Any) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    labels = np.asarray(groups)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("groups must be a nonempty (n, dimensions) array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("groups must contain integer category labels")
    encoded: list[np.ndarray] = []
    levels: list[int] = []
    for j in range(labels.shape[1]):
        # np.unique is independent of a solver's encoding/graph bookkeeping and
        # avoids allocating arrays indexed by an arbitrarily large sparse label.
        _, inverse = np.unique(labels[:, j], return_inverse=True)
        inverse = np.asarray(inverse, dtype=np.intp)
        encoded.append(inverse)
        levels.append(int(inverse.max()) + 1)
    return labels, encoded, levels


def _weights(weights: Any, n: int) -> tuple[np.ndarray, float]:
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be finite and strictly positive, with shape (n,)")
    # All reported ratios and the normalized Gram are invariant to this scaling.
    scale = float(w.max())
    w = w / scale
    if np.any(w == 0):
        raise ValueError("weight dynamic range underflows after normalization")
    return w, scale


def _relative_weighted_norm(error: np.ndarray, reference: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Columnwise ||error||_W / ||reference||_W without squaring huge inputs."""
    out = np.empty(reference.shape[1], dtype=np.float64)
    for j in range(reference.shape[1]):
        scale = max(float(np.max(np.abs(reference[:, j]))), float(np.max(np.abs(error[:, j]))))
        if scale == 0:
            out[j] = 0.0
            continue
        e = error[:, j] / scale
        y = reference[:, j] / scale
        numerator = float(np.sqrt(np.dot(w, e * e)))
        denominator = float(np.sqrt(np.dot(w, y * y)))
        out[j] = numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else np.inf)
    return out


def _stationarity(
    encoded: list[np.ndarray], levels: list[int], residual: np.ndarray,
    reference: np.ndarray, w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """max_k ||P_k residual||_W / ||reference||_W, plus each dimension.

    In dimension k the numerator squared is sum_g (sum_{i in g} w_i r_i)^2
    / sum_{i in g} w_i.  This is only stationarity; it is not a forward bound.
    """
    by_dimension = np.empty((len(encoded), residual.shape[1]), dtype=np.float64)
    mass = [np.bincount(code, weights=w, minlength=g) for code, g in zip(encoded, levels)]
    for col in range(residual.shape[1]):
        scale = max(float(np.max(np.abs(reference[:, col]))), float(np.max(np.abs(residual[:, col]))))
        if scale == 0:
            by_dimension[:, col] = 0.0
            continue
        y = reference[:, col] / scale
        r = residual[:, col] / scale
        denominator = float(np.sqrt(np.dot(w, y * y)))
        for j, (code, g, group_mass) in enumerate(zip(encoded, levels, mass)):
            sums = np.bincount(code, weights=w * r, minlength=g)
            numerator = float(np.sqrt(np.sum((sums / np.sqrt(group_mass)) ** 2)))
            by_dimension[j, col] = numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else np.inf)
    return np.max(by_dimension, axis=0), by_dimension


def _number(value: float) -> float | None:
    """Use null for a nonfinite diagnostic so JSON can set allow_nan=False."""
    return float(value) if np.isfinite(value) else None


def _maximum(values: np.ndarray) -> float | None:
    return _number(float(np.max(values))) if values.size else 0.0


def certification(case: Any, residual: Any, forward_tol: float = 2e-8, *,
                  stationarity_tol: float | None = None) -> dict[str, Any]:
    """Certify a returned projection against ``case.truth`` on original rows.

    Validate the oracle separately with :func:`verify_case`, once per case.
    ``aggregate_pass`` requires finite arrays, forward error and stationarity.
    Backend convergence/status flags are intentionally not accepted as inputs.
    """
    if not np.isfinite(forward_tol) or forward_tol <= 0:
        raise ValueError("forward_tol must be finite and positive")
    stationarity_tol = forward_tol if stationarity_tol is None else stationarity_tol
    if not np.isfinite(stationarity_tol) or stationarity_tol <= 0:
        raise ValueError("stationarity_tol must be finite and positive")
    labels, encoded, levels = _groups(case.groups)
    n = labels.shape[0]
    y = _as_columns(case.y, "y", n)
    truth = _as_columns(case.truth, "truth", n)
    r = _as_columns(residual, "residual", n)
    if y.shape != truth.shape or y.shape != r.shape:
        raise ValueError("y, truth and residual must have the same shape")
    w, weight_scale = _weights(getattr(case, "weights", None), n)
    finite_columns = np.all(np.isfinite(y) & np.isfinite(truth) & np.isfinite(r), axis=0)
    forward = np.full(y.shape[1], np.inf)
    moments = np.full(y.shape[1], np.inf)
    by_dimension = np.full((len(encoded), y.shape[1]), np.inf)
    if np.any(finite_columns):
        index = np.flatnonzero(finite_columns)
        forward[index] = _relative_weighted_norm(r[:, index] - truth[:, index], y[:, index], w)
        moments[index], by_dimension[:, index] = _stationarity(encoded, levels, r[:, index], y[:, index], w)
    forward_pass = finite_columns & np.isfinite(forward) & (forward <= forward_tol)
    stationarity_pass = finite_columns & np.isfinite(moments) & (moments <= stationarity_tol)
    per_rhs = [
        {"rhs": j, "finite": bool(finite_columns[j]),
         "forward_relative_error": _number(forward[j]),
         "normalized_fe_stationarity": _number(moments[j]),
         "normalized_fe_stationarity_by_dimension": [_number(v) for v in by_dimension[:, j]],
         "forward_pass": bool(forward_pass[j]), "stationarity_pass": bool(stationarity_pass[j]),
         "pass": bool(forward_pass[j] and stationarity_pass[j])}
        for j in range(y.shape[1])
    ]
    return {
        "finite": bool(np.all(finite_columns)),
        "forward_pass": bool(np.all(forward_pass)),
        "stationarity_pass": bool(np.all(stationarity_pass)),
        "aggregate_pass": bool(np.all(forward_pass & stationarity_pass)),
        "per_rhs": per_rhs,
        "max_forward_relative_error": _maximum(forward),
        "max_normalized_fe_stationarity": _maximum(moments),
        "forward_tolerance": float(forward_tol), "stationarity_tolerance": float(stationarity_tol),
        "norm": "||returned-truth||_W / ||y||_W",
        "stationarity_definition": "max_k ||P_k returned||_W / ||y||_W",
        "reference": "case.truth; oracle validation is reported separately by verify_case",
        "weight_normalization_scale": weight_scale,
        "nonfinite_diagnostics": "null means unavailable or infinite; it never passes",
    }


def _dense_budget(n: int, levels: list[int], *, max_categories: int, max_rows: int,
                  max_dense_entries: int) -> dict[str, Any]:
    g = sum(levels)
    reason = []
    if g > max_categories:
        reason.append(f"categories {g} exceed max_categories={max_categories}")
    if n > max_rows:
        reason.append(f"rows {n} exceed max_rows={max_rows}")
    if n * g > max_dense_entries:
        reason.append(f"dense entries {n*g} exceed max_dense_entries={max_dense_entries}")
    return {"allowed": not reason, "reason": "; ".join(reason) or None,
            "n": n, "categories": g, "levels": levels,
            "estimated_dense_entries": n * g,
            "budget": {"max_categories": max_categories, "max_rows": max_rows,
                       "max_dense_entries": max_dense_entries}}


def _normalized_design(encoded: list[np.ndarray], levels: list[int], w: np.ndarray) -> np.ndarray:
    n, g = w.size, sum(levels)
    a = np.zeros((n, g), dtype=np.float64)
    rows = np.arange(n)
    offset = 0
    for code, count in zip(encoded, levels):
        mass = np.bincount(code, weights=w, minlength=count)
        a[rows, offset + code] = np.sqrt(w / mass[code])
        offset += count
    return a


def _component_count(encoded: list[np.ndarray], levels: list[int]) -> int:
    """Small-design incidence components; independent of backend graph metadata."""
    parent = list(range(sum(levels)))
    offsets = np.cumsum([0] + levels[:-1]).tolist()

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in range(encoded[0].size):
        root = find(int(encoded[0][row]))
        for j in range(1, len(encoded)):
            other = find(offsets[j] + int(encoded[j][row]))
            if other != root:
                parent[other] = root
    return len({find(i) for i in range(sum(levels))})


def normalized_gram_spectrum(groups: Any, weights: Any = None, *,
                             max_categories: int = 1024, max_rows: int = 8192,
                             max_dense_entries: int = 4_000_000,
                             rank_rtol: float | None = None) -> dict[str, Any]:
    """Numerical positive spectrum of diag(D'WD)^(-1/2) D'WD diag(...)^(-1/2).

    Singular values of the normalized rectangular design are squared rather
    than diagonalizing its formed Gram, avoiding gratuitous precision loss.
    Positivity/rank still use a floating-point threshold and are not proofs.
    """
    labels, encoded, levels = _groups(groups)
    w, weight_scale = _weights(weights, labels.shape[0])
    budget = _dense_budget(labels.shape[0], levels, max_categories=max_categories,
                           max_rows=max_rows, max_dense_entries=max_dense_entries)
    result: dict[str, Any] = {
        **budget, "numerical_estimate_only": True, "structural_rank_proof": False,
        "matrix": "diag(D'WD)^(-1/2) D'WD diag(D'WD)^(-1/2)",
        "method": "float64 SVD of weighted, column-normalized incidence matrix",
        "weight_normalization_scale": weight_scale,
    }
    if not budget["allowed"]:
        result.update(status="skipped_budget", warnings=["No dense matrix was allocated; spectrum and numerical rank are unavailable."])
        return result
    a = _normalized_design(encoded, levels, w)
    singular = np.linalg.svd(a, compute_uv=False)
    if rank_rtol is None:
        rank_rtol = max(a.shape) * np.finfo(np.float64).eps
    if not np.isfinite(rank_rtol) or rank_rtol <= 0:
        raise ValueError("rank_rtol must be finite and positive")
    cutoff = float(rank_rtol * singular[0])
    positive = singular[singular > cutoff]
    rank = positive.size
    components = _component_count(encoded, levels)
    normal_nullity = (len(levels) - 1) * components
    numerical_nullity = sum(levels) - rank
    extra_nullity = numerical_nullity - normal_nullity
    largest = float(positive[0] ** 2) if rank else 0.0
    smallest = float(positive[-1] ** 2) if rank else None
    kappa = float((positive[0] / positive[-1]) ** 2) if rank else None
    warnings = ["Finite precision and the declared singular-value threshold determine numerical rank; small unresolved modes can inflate apparent nullity."]
    if extra_nullity > 0:
        warnings.append("Numerical nullity exceeds component-wise FE gauge nullity; this may be additional structural rank deficiency or unresolved small positive modes.")
    result.update(
        status="ok", numerical_rank=int(rank), numerical_nullity=int(numerical_nullity),
        components=components, normal_nullity=int(normal_nullity),
        expected_normal_rank=int(sum(levels) - normal_nullity),
        extra_nullity=int(extra_nullity), has_extra_nullity=bool(extra_nullity > 0),
        lambda_max=largest, lambda_min_positive=smallest,
        kappa_positive_gram=kappa, kappa_plus=kappa,
        singular_value_rank_rtol=float(rank_rtol), singular_value_cutoff=cutoff,
        positive_eigenvalue_cutoff=cutoff ** 2,
        positive_eigenvalues=[float(v * v) for v in positive],
        warnings=warnings,
    )
    return result


def verify_case(case: Any, tolerance: float = 1e-10, *,
                oracle_forward_tol: float = 1e-9,
                max_categories: int = 1024, max_rows: int = 8192,
                max_dense_entries: int = 4_000_000,
                rank_rtol: float | None = None) -> dict[str, Any]:
    """Check an oracle's W-orthogonality and the feasibility of y - truth.

    Supplied FE coefficients are an independently evaluated constructive
    witness, including on large inputs.  A metadata-only algebraic claim is
    recorded but does not pass.  Small inputs also receive a separate dense
    least-squares check.  Neither test is a large-scale structural rank proof.

    When the generator declares an analytic upper bound on the condition
    number of the frequency-normalized Gram, also require the implied forward
    error bound to pass ``oracle_forward_tol``.  This diagnostic uses rounded
    float64 moments and feasibility checks, not interval-certified arithmetic.
    Without an analytic spectral bound, passing the constructive/dense checks
    is explicitly qualified and is not called a forward-error bound.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(oracle_forward_tol) or oracle_forward_tol <= 0:
        raise ValueError("oracle_forward_tol must be finite and positive")
    labels, encoded, levels = _groups(case.groups)
    n = labels.shape[0]
    y = _as_columns(case.y, "y", n)
    truth = _as_columns(case.truth, "truth", n)
    if y.shape != truth.shape:
        raise ValueError("y and truth must have the same shape")
    w, weight_scale = _weights(getattr(case, "weights", None), n)
    finite = bool(np.all(np.isfinite(y)) and np.all(np.isfinite(truth)))
    budget = _dense_budget(n, levels, max_categories=max_categories, max_rows=max_rows,
                           max_dense_entries=max_dense_entries)
    result: dict[str, Any] = {
        "finite": finite, "tolerance": float(tolerance),
        "structural_rank_proof": False, "weight_normalization_scale": weight_scale,
        "algebraic_metadata": dict(getattr(case, "metadata", {}).get("oracle", {})),
        "dense_budget": budget, "warnings": [],
        "oracle_forward_error_bound": None,
        "oracle_forward_tolerance": float(oracle_forward_tol),
        "oracle_forward_bound_pass": None,
        "oracle_forward_bound_available": False,
        "depends_on_declared_analytic_spectral_bound": False,
        "computed_in_float64_not_interval_certified": True,
        "oracle_basis": "checked constructive/dense feasibility and orthogonality; no analytic forward bound",
    }
    if not finite:
        result.update(aggregate_pass=False, oracle_valid=False,
                      orthogonality={"pass": False, "status": "nonfinite"},
                      feasibility={"pass": False, "status": "nonfinite"})
        return result
    moments, by_dimension = _stationarity(encoded, levels, truth, y, w)
    orthogonal = bool(np.all(np.isfinite(moments)) and np.all(moments <= tolerance))
    result["orthogonality"] = {
        "pass": orthogonal, "method": "independent original-row weighted group sums",
        "per_rhs": [_number(v) for v in moments], "maximum": _maximum(moments),
        "by_dimension": [[_number(v) for v in row] for row in by_dimension],
    }
    target = y - truth
    witnesses: list[dict[str, Any]] = []
    coefficients = getattr(case, "coefficients", None)
    if coefficients is not None:
        if len(coefficients) != labels.shape[1]:
            raise ValueError("coefficient witness must have one array per FE dimension")
        fitted = np.zeros_like(y)
        for j, coeff in enumerate(coefficients):
            alpha = _as_columns(coeff, f"coefficients[{j}]")
            if np.any(labels[:, j] < 0) or int(labels[:, j].max()) >= alpha.shape[0] or alpha.shape[1] != y.shape[1]:
                raise ValueError("coefficient witness must be indexed by the original nonnegative category codes")
            fitted += alpha[labels[:, j]]
        witness_finite = bool(np.all(np.isfinite(fitted)))
        error = _relative_weighted_norm(fitted - target, y, w) if witness_finite else np.full(y.shape[1], np.inf)
        witnesses.append({"method": "independently evaluated supplied FE coefficients",
                          "status": "checked", "pass": bool(witness_finite and np.all(error <= tolerance)),
                          "per_rhs_relative_error": [_number(v) for v in error], "maximum": _maximum(error)})
    if budget["allowed"]:
        a = _normalized_design(encoded, levels, w)
        beta, _, rank, singular = np.linalg.lstsq(a, np.sqrt(w)[:, None] * target, rcond=rank_rtol)
        weighted_error = a @ beta - np.sqrt(w)[:, None] * target
        error = _relative_weighted_norm(weighted_error, np.sqrt(w)[:, None] * y, np.ones(n))
        witnesses.append({"method": "independent dense normalized-design least squares",
                          "status": "checked", "pass": bool(np.all(np.isfinite(error)) and np.all(error <= tolerance)),
                          "per_rhs_relative_error": [_number(v) for v in error], "maximum": _maximum(error),
                          "numerical_rank": int(rank), "rank_rtol": rank_rtol,
                          "numerical_estimate_only": True})
    else:
        result["warnings"].append("Dense feasibility and numerical-rank checks skipped: " + str(budget["reason"]))
    # A failed independent check is never silently outvoted by a successful one.
    feasible = bool(witnesses and all(v["pass"] for v in witnesses))
    result["feasibility"] = {
        "pass": feasible, "status": "checked" if witnesses else "unverified",
        "witnesses": witnesses,
        "metadata_alone_is_not_a_certificate": True,
    }
    if not witnesses:
        result["warnings"].append("No checked feasibility witness: a generator's algebraic metadata is not an independently verified oracle.")

    # Let B=W^(1/2) D diag(D'WD)^(-1/2).  Its largest Gram eigenvalue is K.
    # If q=max_k ||P_k truth||_W/||y||_W, then ||B' W^(1/2) truth||/||y||_W
    # is at most sqrt(K)*q.  A declared kappa upper bound gives
    # lambda_min^+(B'B) >= K/kappa_upper, so the column-space component of truth
    # is bounded by sqrt(kappa_upper)*q.  With f=y-truth-D alpha, subtracting
    # the exact projection adds at most ||f||_W/||y||_W.  Numerical SVD estimates
    # must never be substituted for the declared analytic spectral bound here.
    condition = getattr(case, "metadata", {}).get("condition_number", {})
    declared_kappa = condition.get("analytic_upper_bound")
    bound_gate = True
    if declared_kappa is not None:
        result["depends_on_declared_analytic_spectral_bound"] = True
        valid_kappa = (not isinstance(declared_kappa, (bool, np.bool_))
                       and isinstance(declared_kappa, (int, float, np.integer, np.floating))
                       and np.isfinite(declared_kappa) and declared_kappa >= 1)
        if not valid_kappa:
            result["oracle_forward_bound_pass"] = False
            result["oracle_basis"] = "invalid declared analytic spectral bound"
            result["warnings"].append("Declared analytic kappa upper bound must be finite and at least one; the oracle gate fails.")
            bound_gate = False
        elif witnesses:
            witness_errors = [v["maximum"] for v in witnesses]
            feasibility_error = max(float(v) if v is not None else np.inf for v in witness_errors)
            with np.errstate(over="ignore", invalid="ignore"):
                forward_bound = float(np.sqrt(declared_kappa) * np.max(moments) + feasibility_error)
            bound_gate = bool(np.isfinite(forward_bound) and forward_bound <= oracle_forward_tol)
            result.update(
                oracle_forward_error_bound=_number(forward_bound),
                oracle_forward_bound_available=True,
                oracle_forward_bound_pass=bound_gate,
                oracle_basis="declared analytic spectral bound plus independently evaluated float64 witnesses",
                oracle_forward_bound_details={
                    "kappa_upper": float(declared_kappa),
                    "truth_stationarity_max": _maximum(moments),
                    "feasibility_relative_error_max": _number(feasibility_error),
                    "formula": "sqrt(kappa_upper) * max_truth_fe_stationarity + max_feasibility_relative_error",
                    "lambda_max_normalized_gram": int(labels.shape[1]),
                    "source": "case.metadata.condition_number.analytic_upper_bound",
                    "numerical_spectrum_estimates_used": False,
                },
            )
            if not bound_gate:
                result["warnings"].append("The declared-spectrum float64 oracle forward-error bound exceeds the requested oracle tolerance or is nonfinite.")
    else:
        result["warnings"].append("No analytic spectral bound supplied: the feasibility and orthogonality checks do not by themselves provide a general forward-error guarantee.")
    passed = bool(orthogonal and feasible and bound_gate)
    result.update(aggregate_pass=passed, oracle_valid=passed)
    return result
