"""Versioned local-file bridge for statistical-software callers.

All binary arrays are IEEE float64, little endian and column major (R order).
Only completed runs publish an ok response. Nonfinite JSON values become null;
the corresponding binary arrays preserve Inf, -Inf and NaN without ambiguity.
No solver, residual-score, or finite-pseudo-observation semantics change here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .core import solve_perturbations
from .inference import summarize_process


def _strict(value):
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _strict(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(v) for v in value]
    return value


def _json_write(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(_strict(value), allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_array(root, spec, expected_shape):
    if not isinstance(spec, dict) or tuple(spec.get("shape", ())) != tuple(expected_shape):
        raise ValueError(f"Array descriptor shape must be {expected_shape}")
    relative = Path(spec["file"])
    if relative.is_absolute():
        raise ValueError("Input file names must be relative to the request directory")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input file must stay inside the request directory")
    count = math.prod(expected_shape)
    if path.stat().st_size != 8 * count:
        raise ValueError(f"Wrong binary byte count for {relative}; expected {8 * count}")
    array = np.fromfile(path, dtype="<f8", count=count).reshape(expected_shape, order="F")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite input in {relative}")
    return array


def _write_array(root, name, value):
    array = np.asarray(value, dtype="<f8")
    path = root / name
    array.ravel(order="F").tofile(path)
    return dict(file=name, shape=list(array.shape), dtype="float64", endian="little", order="F")


def run_request(request_path, response_path):
    started = time.perf_counter()
    request_path, response_path = Path(request_path).resolve(), Path(response_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if request.get("protocol_version") != 1:
        raise ValueError("Expected protocol_version=1")
    if (request.get("dtype"), request.get("endian"), request.get("order")) != ("float64", "little", "F"):
        raise ValueError("Protocol requires little-endian float64 column-major arrays")
    dimensions = [request.get(key) for key in ("n", "p", "reps")]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in dimensions):
        raise ValueError("n, p and reps must be positive integers")
    n, p, reps = dimensions
    if n < p or reps < 2:
        raise ValueError("Require n >= p and reps >= 2")
    jobs = request.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list of quantile jobs")
    root = request_path.parent
    x = _read_array(root, request["x"], (n, p))
    y = _read_array(root, request["y"], (n,))
    options = request.get("solver", {})
    allowed = {"backend", "batch_size", "verify_radius", "cpu_factor", "threads", "audit_backend", "repair",
               "positive_pseudo"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError(f"Unsupported solver options; allowed keys: {sorted(allowed)}")
    if "positive_pseudo" in options and not isinstance(options["positive_pseudo"], bool):
        raise ValueError("positive_pseudo must be a JSON boolean (true or false)")
    response_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = response_path.parent
    q = len(jobs)
    coefficients, draws = np.empty((q, p)), np.empty((reps, q, p))
    radii = np.full((reps, q), np.inf)
    records, output_jobs = [], []
    input_seconds = time.perf_counter() - started
    for j, job in enumerate(jobs):
        read_started = time.perf_counter()
        tau = float(job["tau"])
        if not np.isfinite(tau) or not 0 < tau < 1:
            raise ValueError("Every tau must be finite and strictly inside (0,1)")
        w = _read_array(root, job["W"], (p, reps))
        beta0 = _read_array(root, job["beta0"], (p,))
        input_seconds += time.perf_counter() - read_started
        values, diagnostic = solve_perturbations(x, y, tau, w, beta0, **options)
        coefficients[j], draws[:, j] = beta0, values
        if options.get("verify_radius", True):
            seen = set()
            for index, fit in enumerate(diagnostic["fits"]):
                at = int(fit.get("replicate", index))
                if at < 0 or at >= reps or at in seen:
                    raise ValueError("Solver diagnostic replicate indices are invalid")
                radii[at, j] = fit["radius_certificate"]["radius"]
                seen.add(at)
            if len(seen) != reps:
                raise ValueError("Solver did not return every replicate certificate")
        output_jobs.append(dict(tau=tau,
                                coefficients=_write_array(output_root, f"coefficients_{j}.bin", values),
                                radii=_write_array(output_root, f"radii_{j}.bin", radii[:, j])))
        records.append(dict(tau=tau, solver=diagnostic))
    summary_started = time.perf_counter()
    inference = summarize_process(coefficients, draws, level=float(request.get("level", .95)),
                                  radii=radii if options.get("verify_radius", True) else None)
    inference_arrays, inference_scalars = {}, {}
    for key, value in inference.items():
        if isinstance(value, np.ndarray):
            inference_arrays[key] = _write_array(output_root, f"inference_{key}.bin", value)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            # Strict JSON cannot distinguish Inf/-Inf/NaN from null. Preserve
            # nonfinite scalar summaries through the same binary channel.
            inference_arrays[key] = _write_array(output_root, f"inference_{key}.bin", [value])
            inference_scalars[key] = value
        else:
            inference_scalars[key] = value
    response = dict(ok=True, protocol_version=1, dtype="float64", endian="little", order="F",
                    n=n, p=p, reps=reps, jobs=output_jobs,
                    inference_arrays=inference_arrays, inference=inference_scalars,
                    diagnostics=records,
                    numerical_scope="Fixed FP64 X/y/W and quantreg residual scores; model-qualified enclosures, not a formal interval proof",
                    nonfinite_json="Nonfinite JSON numbers are null; binary output arrays preserve IEEE nonfinite values",
                    timing=dict(python_input_seconds=input_seconds,
                                python_summary_output_seconds=time.perf_counter()-summary_started,
                                python_request_seconds=time.perf_counter()-started))
    _json_write(response_path, response)
    return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        response = run_request(args.request, args.response)
    except Exception as exc:
        args.response.parent.mkdir(parents=True, exist_ok=True)
        _json_write(args.response, dict(ok=False, protocol_version=1,
                                       error_type=type(exc).__name__, message=str(exc)))
        traceback.print_exc(file=sys.stderr)
        return 1
    print(json.dumps(dict(ok=True, protocol_version=1, reps=response["reps"],
                          quantiles=len(response["jobs"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
