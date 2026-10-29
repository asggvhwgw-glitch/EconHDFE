#!/usr/bin/env python3
"""Run each FE family/backend in an isolated process and retain every outcome.

Example (from the repository root)::

    python benchmarks/hdfe/frontier_v1/run.py --families rigidity_grid \
      --sizes 16 32 64 --backends scipy_lsmr within native_auto fereg \
      --output benchmarks/hdfe/frontier_v1/results/pilot.json

The timeout applies to a complete worker, including generation and verification.
Solver timings exclude those phases but include adapter preparation and recovery.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as im
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))


def json_safe(value):
    """Strict JSON; nonfinite diagnostics never become permissive NaN literals."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def environment():
    versions = {}
    for name in ("numpy", "scipy", "pytest", "within-py", "econhdfe-within", "econhdfe-dual-prototype"):
        try:
            versions[name] = im.version(name)
        except im.PackageNotFoundError:
            versions[name] = None
    cpu_model = None
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    caps = {}
    for name in ("memory.max", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            caps[name] = path.read_text().strip()
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        source_commit = None
    hashes = {}
    for pattern in ("*.py", "tests/*.py"):
        for path in sorted(HERE.glob(pattern)):
            hashes[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus_visible": os.cpu_count(),
        "cgroup_limits": caps,
        "versions": versions,
        "repository_base_commit": source_commit,
        "benchmark_source_sha256": hashes,
        "clock": "time.perf_counter",
    }


def worker(request_path, output_path):
    started = time.perf_counter()
    request = json.loads(Path(request_path).read_text())
    result = {"request": request, "status": "started"}
    phase_path = str(output_path) + ".phase.json"
    current_phase = "initialization"

    def phase(name):
        nonlocal current_phase
        current_phase = name
        write_json(phase_path, {
            "phase": name, "elapsed_s": time.perf_counter() - started,
            "case": result.get("case"),
            "oracle_valid": result.get("oracle", {}).get("oracle_valid"),
        })

    try:
        phase("imports")
        import numpy as np
        from frontier_v1.families import make_case
        from frontier_v1.backends import run_backend
        from frontier_v1.validation import certification, verify_case, normalized_gram_spectrum

        phase("generation")
        t = time.perf_counter()
        case = make_case(
            request["family"], request["size"], rhs=request["rhs"],
            seed=request["seed"], repeat=request["repeat"],
            weight_ratio=request["weight_ratio"], rhs_kind=request["rhs_kind"],
        )
        result["generation_s"] = time.perf_counter() - t
        result["case"] = {
            "name": case.name,
            "n": int(case.groups.shape[0]),
            "k": int(case.groups.shape[1]),
            "rhs": int(case.y.shape[1]),
            "group_levels": [int(np.unique(case.groups[:, j]).size) for j in range(case.groups.shape[1])],
            "metadata": case.metadata,
        }

        phase("oracle_verification")
        t = time.perf_counter()
        result["oracle"] = verify_case(
            case, max_categories=request["spectrum_max_categories"], max_rows=8192,
            oracle_forward_tol=min(1e-9, request["forward_tol"] / 10),
        )
        result["oracle_verification_s"] = time.perf_counter() - t
        oracle_ok = result["oracle"].get("aggregate_pass", result["oracle"].get("oracle_valid"))
        if oracle_ok is False:
            result["status"] = "invalid_fixture"
            return

        phase("small_spectrum")
        t = time.perf_counter()
        result["spectrum"] = normalized_gram_spectrum(
            case.groups, case.weights, max_categories=request["spectrum_max_categories"],
            max_rows=8192,
        )
        result["spectrum_diagnostic_s"] = time.perf_counter() - t

        phase("backend")
        residual, metrics = run_backend(
            request["backend"], case, tol=request["tol"],
            max_iter=request["max_iter"], threads=request["threads"],
        )
        result["solver"] = metrics

        phase("independent_forward_check")
        t = time.perf_counter()
        result["accuracy"] = certification(case, residual, forward_tol=request["forward_tol"])
        result["verification_s"] = time.perf_counter() - t
        native_converged = metrics.get("converged")
        if isinstance(native_converged, (list, tuple, np.ndarray)):
            native_converged = bool(np.all(native_converged))
        if result["accuracy"]["aggregate_pass"]:
            result["status"] = "passed" if native_converged is not False else "accuracy_pass_native_failed"
        else:
            result["status"] = "accuracy_failed"
    except Exception as error:
        result["status"] = ("unavailable" if type(error).__name__ == "BackendUnavailable" else
                            "backend_error" if current_phase == "backend" else "verification_error"
                            if current_phase in {"oracle_verification", "small_spectrum", "independent_forward_check"}
                            else "fixture_error")
        result["error"] = {"type": type(error).__name__, "phase": current_phase,
                           "message": str(error), "traceback": traceback.format_exc()}
    finally:
        result["worker_wall_s"] = time.perf_counter() - started
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["worker_peak_rss_bytes"] = int(maxrss if sys.platform == "darwin" else maxrss * 1024)
        result["memory_scope"] = "whole isolated worker: imports, fixture, oracle, solver, verification; not solver-only"
        write_json(output_path, result)
        phase("finished")


def summarize(records):
    by_status = {}
    for record in records:
        status = record["status"]
        by_status[status] = by_status.get(status, 0) + 1
    errors = [r.get("accuracy", {}).get("max_forward_relative_error") for r in records]
    finite_errors = [x for x in errors if isinstance(x, (int, float))]
    return {"records": len(records), "by_status": by_status,
            "max_returned_forward_relative_error": max(finite_errors) if finite_errors else None}


def run_requests(requests, output, timeout_seconds):
    output = Path(output).resolve()
    run_id = uuid.uuid4().hex[:12]
    work = output.with_suffix("").with_name(output.stem + "_workers_" + run_id)
    work.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "prototype_version": "0.1.0", "run_id": run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": environment(),
        "timing_contract": {
            "solver_total": "adapter setup/layout + projection + output recovery; lazy factors may occur in solve_s",
            "excluded": "DGP, independent fixture verification, spectrum diagnostic and output forward verification",
            "process_wall": "includes interpreter/imports, all phases and process exit",
            "timeout": "complete child process deadline; retained as failure, never a successful solve time",
            "matvec_counts": "only actually observed counts; native opaque counters remain null",
            "execution_order": "sequential children; backend order alternates with workload index",
        },
        "records": [],
    }
    for index, request in enumerate(requests):
        tag = f"{index:03d}_{request['family']}_{request['size']}_{request['backend']}"
        reqpath, outpath, logpath = (work / (tag + suffix) for suffix in (".request.json", ".json", ".log"))
        write_json(reqpath, request)
        env = os.environ.copy()
        env.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": str(request["threads"]), "RAYON_NUM_THREADS": str(request["threads"])})
        print(f"[{index+1}/{len(requests)}] {tag}", flush=True)
        t = time.perf_counter()
        timed_out = False
        with logpath.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker-request", str(reqpath),
                 "--worker-output", str(outpath)], cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if timed_out:
            record = {"request": request, "status": "timeout", "timeout_seconds": timeout_seconds}
            last_phase = Path(str(outpath) + ".phase.json")
            if last_phase.exists():
                record["last_phase"] = json.loads(last_phase.read_text())
                record["case"] = record["last_phase"].get("case")
        elif outpath.exists():
            record = json.loads(outpath.read_text())
            if record.get("request") != request or process.returncode != 0:
                record = {
                    "request": request, "status": "process_failed",
                    "error": {"type": "InvalidWorkerResult", "message":
                              "Worker request mismatch or nonzero process exit; partial output is not accepted."},
                    "partial_result": record,
                }
        else:
            record = {"request": request, "status": "process_failed", "log_tail": logpath.read_text()[-6000:]}
        record["process_exit_code"] = process.returncode
        record["process_wall_s"] = time.perf_counter() - t
        record["worker_log"] = str(logpath.relative_to(output.parent))
        report["records"].append(record)
        report["summary"] = summarize(report["records"])
        write_json(output, report)
        elapsed = record.get("solver", {}).get("total_s")
        forward = record.get("accuracy", {}).get("max_forward_relative_error")
        print(f"  {record['status']} solver_s={elapsed} forward={forward} wall_s={record['process_wall_s']:.3f}", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--families", nargs="+", default=["rigidity_grid", "kernel_grid"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--backends", nargs="+", default=["scipy_lsmr", "within", "native_auto", "fereg"])
    parser.add_argument("--rhs", type=int, default=2)
    parser.add_argument("--rhs-kind", default="mixed")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--weight-ratio", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--forward-tol", type=float, default=2e-8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--spectrum-max-categories", type=int, default=512)
    parser.add_argument("--timing-repeats", type=int, default=1,
                        help="Independent fresh-process timing trials; distinct from row replication --repeat")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "pilot.json")
    args = parser.parse_args(argv)
    if args.worker_request:
        if not args.worker_output:
            parser.error("--worker-output is required for worker mode")
        worker(args.worker_request, args.worker_output)
        return 0
    if min(args.rhs, args.threads, args.max_iter, args.repeat, args.timing_repeats) < 1:
        parser.error("positive rhs/threads/max-iter/repeat/timing-repeats required")
    if args.timeout <= 0 or args.forward_tol <= 0 or args.tol <= 0 or args.weight_ratio < 1:
        parser.error("positive timeout/tolerances and weight-ratio >= 1 required")
    requests = []
    workload_index = 0
    for trial in range(args.timing_repeats):
        for family in args.families:
            for size in args.sizes:
                order = args.backends if workload_index % 2 == 0 else list(reversed(args.backends))
                for backend in order:
                    requests.append({
                        "family": family, "size": size, "backend": backend,
                        "rhs": args.rhs, "rhs_kind": args.rhs_kind, "seed": args.seed,
                        "repeat": args.repeat, "weight_ratio": args.weight_ratio,
                        "threads": args.threads, "tol": args.tol, "forward_tol": args.forward_tol,
                        "max_iter": args.max_iter, "spectrum_max_categories": args.spectrum_max_categories,
                        "timing_trial": trial,
                    })
                workload_index += 1
    report = run_requests(requests, args.output, args.timeout)
    return 0 if all(r["status"] == "passed" for r in report["records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
