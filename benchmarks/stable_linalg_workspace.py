"""Isolated-process PERF-03 comparison against a supplied pre-change module.

Usage: python benchmarks/stable_linalg_workspace.py --baseline /path/to/stable_linalg.py \
         --output /path/to/result.json --rounds 3
Times exclude profiling. One additional process per case/backend records NumPy
allocation peak with tracemalloc. RSS is process high-water, not incremental
workspace; do not generalize these same-host measurements to other machines.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
CASES = {"small_C": (2000, 8, "C"), "tall_C": (240000, 32, "C"),
         "tall_F": (240000, 32, "F"), "sliced": (120000, 32, "sliced"),
         "wide_C": (1000, 128, "C")}


def worker(args):
    import numpy as np
    import scipy
    spec = importlib.util.spec_from_file_location("measured_linalg", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n, k, layout = CASES[args.case]
    rng = np.random.default_rng(6403)
    X = rng.normal(size=(n, k))
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "sliced":
        storage = np.zeros((2 * n, 2 * k))
        storage[::2, ::2] = X
        X = storage[::2, ::2]
    y = rng.normal(size=n)
    # Warm library dispatch on a small problem, not the measured data.
    module.equilibrated_lstsq(np.eye(4), np.ones(4))
    if args.profile:
        tracemalloc.start()
    start = time.perf_counter()
    beta, bread, rank = module.equilibrated_lstsq(X, y)
    elapsed = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1] if args.profile else None
    if args.profile:
        tracemalloc.stop()
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        rss = None
    np.savez(args.arrays, beta=beta, bread=bread, rank=rank)
    print(json.dumps({"seconds": elapsed, "traced_peak_bytes": peak,
                      "process_peak_rss_bytes": rss, "rank": int(rank),
                      "numpy": np.__version__, "scipy": scipy.__version__}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--module", type=Path)
    p.add_argument("--case", choices=CASES)
    p.add_argument("--arrays", type=Path)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.module:
        worker(args)
        return
    if args.baseline is None or not args.baseline.is_file() or args.output is None or args.rounds < 2:
        p.error("require --baseline file, --output and --rounds >= 2")
    import numpy as np
    candidate = ROOT / "econhdfe/compute/stable_linalg.py"
    modules = {"baseline": args.baseline.resolve(), "candidate": candidate}
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    result = {"schema_version": 1, "environment": {"python": platform.python_version(),
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "threads": 1}, "rounds": args.rounds,
              "module_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in modules.items()},
              "cases": {}}
    with tempfile.TemporaryDirectory() as temporary:
        for case, (n, k, layout) in CASES.items():
            record = {"n": n, "k": k, "layout": layout, "runs": [], "parity": True,
                      "max_beta_abs_diff": 0., "max_bread_abs_diff": 0.}
            reference = None
            for cycle in range(args.rounds + 1):
                profile = cycle == args.rounds
                order = ("baseline", "candidate") if cycle % 2 == 0 else ("candidate", "baseline")
                for kind in order:
                    arrays = Path(temporary) / f"{kind}.npz"
                    command = [sys.executable, str(Path(__file__).resolve()), "--module", str(modules[kind]),
                               "--case", case, "--arrays", str(arrays)]
                    if profile:
                        command.append("--profile")
                    completed = subprocess.run(command, env=env, check=True, capture_output=True,
                                               text=True, timeout=180)
                    run = json.loads(completed.stdout)
                    run.update(kind=kind, cycle=cycle, profiling=profile)
                    record["runs"].append(run)
                    with np.load(arrays) as values:
                        current = {name: values[name].copy() for name in values.files}
                    if reference is None:
                        reference = current
                    for name in ("beta", "bread"):
                        np.testing.assert_allclose(current[name], reference[name], rtol=1e-10, atol=1e-12)
                        diff = float(np.max(np.abs(current[name] - reference[name])))
                        record[f"max_{name}_abs_diff"] = max(record[f"max_{name}_abs_diff"], diff)
                    assert int(current["rank"]) == int(reference["rank"])
            record["median_seconds"] = {
                kind: statistics.median(r["seconds"] for r in record["runs"]
                                        if r["kind"] == kind and not r["profiling"])
                for kind in modules}
            record["candidate_over_baseline"] = record["median_seconds"]["candidate"] / record["median_seconds"]["baseline"]
            result["cases"][case] = record
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(case, record["candidate_over_baseline"], flush=True)


if __name__ == "__main__":
    main()
