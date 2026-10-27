"""Standalone million-row v0.5 selector vs native within benchmark.

Benchmarks pure categorical FE projections, not complete regressions.
Use separate processes for each method; the reported time includes
preparation, route selection, solve, and numerical audit, not data generation.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from cases import make_case, metadata
from validation import audit
from feselection import FESelector, SelectionConfig, ResourceBudget

CASES = ("joint4", "equal_sparse", "smooth_sparse", "irregular", "nonadditive", "mobility3", "hub4", "weighted4", "high5")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=CASES, default="smooth_sparse")
    p.add_argument("--method", choices=("selector", "native_adaptive", "native_diagonal", "native_additive"), required=True)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=93517)
    p.add_argument("--rhs", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-13)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.n < 1_000_000 or a.rhs < 3:
        p.error("Performance benchmark requires N >= 1,000,000 and RHS >= 3")
    codes, weights, Y, truth = make_case(a.case, a.n, a.seed)
    if a.rhs > Y.shape[1]:
        rng = np.random.default_rng(a.seed + 913)
        Y = np.column_stack((Y[:, :2], rng.normal(size=(a.n, a.rhs-3)), Y[:, -1]))
    source_hash = hashlib.sha256(codes.tobytes() + weights.tobytes() + Y.tobytes()).hexdigest()
    t0 = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        if a.method == "selector":
            cfg = SelectionConfig(mode="auto", tol=a.tol)
            selector = FESelector(cfg).prepare(codes, weights, expected_rhs=Y.shape[1])
            res = selector.residualize(Y)
            residuals = res.residuals
            trace = res.diagnostics
            success = True
        else:
            import within
            mode = a.method.split("_", 1)[1]
            pc = {"adaptive": within.PreconditionerConfig.Adaptive,
                  "diagonal": within.PreconditionerConfig.Diagonal,
                  "additive": within.PreconditionerConfig.Additive}[mode]()
            design = within.Design(np.asfortranarray(codes, dtype=np.uint32))
            solver = within.Solver(design, weights=weights, preconditioner=pc)
            result = solver.solve_batch(np.asfortranarray(Y, dtype=np.float64),
                                        options=within.LsmrOptions(tol=a.tol, maxiter=1200))
            residuals = np.asarray(result.demeaned)
            trace = {"iterations": list(result.iterations), "converged": list(result.converged)}
            success = bool(all(result.converged))
        elapsed = time.perf_counter() - t0
        checked = audit(codes, weights, Y, residuals, truth)
        pools = threadpool_info()
    record = {"n": a.n, "rhs": a.rhs, "seed": a.seed, "case": a.case,
              "method": a.method, "tol": a.tol, "elapsed_seconds": elapsed,
              "input_sha256": source_hash, "successful_solve": success,
              "audit": checked, "diagnostics": trace,
              "threadpools": [{"api": x.get("user_api"), "threads": x.get("num_threads")}
                              for x in pools if x.get("user_api") == "blas"]}
    record["passed"] = bool(success and checked.get("projection_gate_passed", False))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    print(json.dumps({"passed": record["passed"], "method": a.method,
                      "case": a.case, "elapsed_seconds": elapsed, "out": str(a.out)}))
    if not record["passed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
