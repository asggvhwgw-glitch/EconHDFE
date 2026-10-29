#!/usr/bin/env python3
"""Isolated cold/reuse and diagnostics-on/off experiments for research wheels.

Each request gets a fresh process and solver. Its second call reuses that solver
on identical data/weights, with no coefficient warm-start seed. Timings exclude
fixture/oracle work; native nested timers are retained without addition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from frontier_v1 import backends, run

NATIVE_BACKENDS = ("within", "native_auto", "fereg")


class DiagnosticsUnsupported(RuntimeError):
    """The installed wheel cannot perform this controlled diagnostic study."""


def create_solver(name, case, *, diagnostics=True, threads=1, scratch_bytes=1 << 30):
    """Construct through the research-only API; never retry without its flag."""
    module = backends._module(name)
    if getattr(module, "COST_DIAGNOSTICS_SCHEMA", None) != 1:
        raise DiagnosticsUnsupported(f"{name}: research diagnostics schema 1 wheel required")
    provenance = backends._provenance(name, module)
    started = time.perf_counter()
    codes, y, weights, _, _ = backends._inputs(case)
    options = dict(weights=weights, threads=threads, diagnostics=diagnostics)
    if name == "native_auto":
        solver = module.NativeSolver(codes, strategy="auto", workspace_bytes=scratch_bytes, **options)
    else:
        solver = module.DualAbsorber(codes, backend=name, scratch_bytes=scratch_bytes,
                                    allow_fallback=False, **options)
    return solver, y, provenance, time.perf_counter() - started


def read_diagnostics(name, solver, info=None):
    if name != "native_auto" and info is None:
        try:
            info = solver.diagnostics
        except Exception as error:
            return {"available": False, "status": "invalid", "reason": str(error)}
    return backends._native_cost_diagnostics(name, solver, info)


def solve_once(name, solver, y, *, tol=1e-10, max_iter=2000):
    """One full call and output copy; collect optional getters after its timer."""
    import numpy as np
    started = time.perf_counter()
    if name == "native_auto":
        residual, flags, iterations, errors = solver.solve(y, tol=tol, max_iter=max_iter)
        info = {"converged": bool(all(flags)), "iterations": list(iterations),
                "certificate": list(errors), "route": solver.route}
    else:
        residual, info = solver.residualize(y, tol=tol, max_iter=max_iter, return_info=True)
        info = dict(info, converged=bool(info.get("accepted", False)))
    residual = np.array(residual, dtype=np.float64, order="C", copy=True)
    elapsed = time.perf_counter() - started
    started = time.perf_counter()
    cost = read_diagnostics(name, solver, info)
    return residual, {"solve_s": elapsed, "native_converged": info["converged"],
                      "solver_details": info, "native_cost_diagnostics": cost,
                      "diagnostics_collection_s": time.perf_counter() - started}


def _error(error):
    return {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}


def worker(request_path, output_path):
    from frontier_v1.families import make_case
    from frontier_v1.validation import certification, verify_case
    started = time.perf_counter()
    request = json.loads(Path(request_path).read_text())
    result = {"request": request, "status": "started", "calls": []}
    try:
        tick = time.perf_counter()
        case = make_case(request["family"], request["size"], rhs=request["rhs"],
                         seed=request["seed"], rhs_kind="broadband")
        result.update(generation_s=time.perf_counter() - tick,
                      case={"name": case.name, "n": len(case.groups), "rhs": case.y.shape[1],
                            "metadata": case.metadata})
        tick = time.perf_counter()
        result["oracle"] = verify_case(case, max_categories=512, max_rows=8192,
                                       oracle_forward_tol=min(1e-9, request["forward_tol"] / 10))
        result["oracle_verification_s"] = time.perf_counter() - tick
        if not result["oracle"]["aggregate_pass"]:
            result["status"] = "invalid_fixture"
            return
        name = request["backend"]
        solver, y, provenance, setup_s = create_solver(
            name, case, diagnostics=request["diagnostics"], threads=request["threads"])
        result.update(provenance=provenance, constructor_and_layout_s=setup_s)
        run.write_json(output_path, result)
        previous_succeeded = False
        for call_index, mode in enumerate(request["calls"]):
            record = {"mode": mode, "call_index": call_index, "coefficient_warm_start": False,
                      "same_fixture_and_weights": True, "prior_call_succeeded": previous_succeeded,
                      "setup_s": setup_s if call_index == 0 else 0.0}
            call_started = time.perf_counter()
            try:
                residual, metrics = solve_once(name, solver, y, tol=request["tol"],
                                               max_iter=request["max_iter"])
                record.update(metrics)
                record["total_s"] = record["setup_s"] + record["solve_s"]
                tick = time.perf_counter()
                record["accuracy"] = certification(case, residual, forward_tol=request["forward_tol"])
                record["verification_s"] = time.perf_counter() - tick
                diagnostic = record["native_cost_diagnostics"]
                if not record["accuracy"]["aggregate_pass"]:
                    record["status"] = "accuracy_failed"
                elif not record["native_converged"]:
                    record["status"] = "accuracy_pass_native_failed"
                elif diagnostic.get("available") is not True or diagnostic.get("enabled") is not request["diagnostics"]:
                    record["status"] = "diagnostics_failed"
                else:
                    record["status"] = "passed"
            except Exception as error:
                record.update(status="solve_error", error=_error(error),
                              failed_call_elapsed_s=time.perf_counter() - call_started,
                              native_cost_diagnostics=read_diagnostics(name, solver))
            previous_succeeded = record["status"] == "passed"
            result["calls"].append(record)
            run.write_json(output_path, result)
        result["status"] = "passed" if all(c["status"] == "passed" for c in result["calls"]) else "failed"
    except Exception as error:
        status = "unsupported_native_diagnostics" if isinstance(error, DiagnosticsUnsupported) else "worker_error"
        result.update(status=status, error=_error(error))
    finally:
        result["worker_wall_s"] = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["worker_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        result["memory_scope"] = "whole worker including both calls and both external checks; not per-call solver memory"
        run.write_json(output_path, result)


def run_study(requests, output, timeout):
    output = Path(output).resolve()
    work = output.parent / (output.stem + "_workers_" + uuid.uuid4().hex[:12])
    work.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "study": "native_cost_v1", "environment": run.environment(),
              "study_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "timing_contract": {
                  "cold": "layout + fresh constructor + first full projection; lazy factors remain in solve_s",
                  "warm": "second projection on same solver, fixture and weights; no coefficient seed",
                  "excluded": "imports, provenance, fixture, oracle, external forward check, optional getter processing",
                  "on_off": "separate fresh workers; toggle measures hot-loop counters/timers, not total patch overhead; both modes retain metadata/snapshot output",
                  "replicates": "fresh process and solver for every replicate; identical fixture seed; on/off order alternates by replicate",
                  "native_timers": "nested or concurrent elapsed times; never add to outer wall time",
              }, "cases": []}
    for index, request in enumerate(requests):
        tag = f"{index:03d}_{request['family']}_{request['size']}_{request['backend']}_{request['diagnostics']}"
        req, out, log = (work / (tag + suffix) for suffix in (".request.json", ".json", ".log"))
        run.write_json(req, request)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   RAYON_NUM_THREADS=str(request["threads"]))
        started = time.perf_counter()
        print(f"[{index + 1}/{len(requests)}] {tag}", flush=True)
        timed_out = False
        with log.open("w") as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-request", str(req),
                                      "--worker-output", str(out)], cwd=run.REPO, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        try:
            record = json.loads(out.read_text()) if out.exists() else None
        except (ValueError, OSError):
            record = None
        if timed_out or child.returncode != 0 or not record or record.get("request") != request:
            record = {"request": request, "status": "timeout" if timed_out else "process_failed",
                      "partial_result": record, "log_tail": log.read_text()[-6000:]}
        record.update(process_exit_code=child.returncode, process_wall_s=time.perf_counter() - started,
                      worker_log=str(log.relative_to(output.parent)))
        report["cases"].append(record)
        report["summary"] = run.summarize([call for case in report["cases"] for call in (case.get("calls") or [case])])
        run.write_json(output, report)
        print(f"  {record['status']}", flush=True)
    # Finish after every child has exited, then verify the aggregate rather than
    # assuming that a printed in-memory summary is the persisted deliverable.
    run.write_json(output, report)
    persisted = output.read_bytes()
    if json.loads(persisted) != run.json_safe(report):
        raise RuntimeError("persisted study report differs from completed in-memory records")
    print(f"persisted_sha256={hashlib.sha256(persisted).hexdigest()} cases={len(report['cases'])}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--families", nargs="+", default=["rigidity_grid"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--backends", nargs="+", choices=NATIVE_BACKENDS, default=list(NATIVE_BACKENDS))
    parser.add_argument("--calls", nargs="+", choices=["cold", "warm"], default=["cold", "warm"])
    parser.add_argument("--diagnostics", nargs="+", choices=["on", "off"], default=["on", "off"])
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--rhs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--forward-tol", type=float, default=2e-8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "study.json")
    args = parser.parse_args(argv)
    if args.worker_request:
        if not args.worker_output:
            parser.error("--worker-output is required")
        worker(args.worker_request, args.worker_output)
        return 0
    if args.calls not in (["cold"], ["cold", "warm"]):
        parser.error("--calls must be 'cold' or 'cold warm'; warm needs the first call")
    values = [*args.sizes, args.rhs, args.threads, args.max_iter, args.replicates,
              args.timeout, args.tol, args.forward_tol]
    if not all(math.isfinite(v) and v > 0 for v in values):
        parser.error("sizes, counts, tolerances and timeout must be positive")
    requests = []
    for replicate in range(args.replicates):
        mode_order = args.diagnostics if replicate % 2 == 0 else list(reversed(args.diagnostics))
        for family in args.families:
            for size in args.sizes:
                for backend in args.backends:
                    for mode in mode_order:
                        requests.append(dict(family=family, size=size, backend=backend, diagnostics=mode == "on",
                                             replicate=replicate, mode_order=mode_order,
                                             calls=args.calls, rhs=args.rhs, seed=args.seed, threads=args.threads,
                                             tol=args.tol, forward_tol=args.forward_tol, max_iter=args.max_iter))
    report = run_study(requests, args.output, args.timeout)
    print(json.dumps(report["summary"], indent=2))
    return 0 if all(c["status"] == "passed" for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
