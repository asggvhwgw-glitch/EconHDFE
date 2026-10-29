"""Benchmark EconHDFE NativeSolver routes against upstream within-py kernel.

This benchmark does not involve fereg. See kernel-pair/ for direct fereg-core.
All routes use exactly same RHS, weights, precision check, thread count, DGP.
Reports cold/rehit/new weight setup and solve separately, including memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import time
import traceback

import numpy as np
from fereg_within_topologies import generate, new_weights


def cert(codes, rhs, weights, residual):
    n, k = rhs.shape
    if not np.all(np.isfinite(residual)):
        return float("inf")
    sw = float(weights.sum())
    worst = 0.0
    for j in range(k):
        scale = np.sqrt(np.dot(weights, rhs[:, j] ** 2) / sw)
        if scale <= 1e-25:
            continue
        wr = weights * residual[:, j]
        for q in range(codes.shape[1]):
            numer = np.bincount(codes[:, q], weights=wr)
            mass = np.bincount(codes[:, q], weights=weights)
            good = mass > 0
            score = np.sqrt(np.dot(numer[good], numer[good] / mass[good]) / sw) / scale
            worst = max(worst, float(score))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["one_big", "two_big", "three_big", "weak_links", "hub", "repeated_cells", "reweight"], required=True)
    p.add_argument("--method", choices=["within", "native_plain", "native_auto", "native_compressed"], required=True)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--rhs", type=int, default=2)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--seed", type=int, default=10292026)
    args = p.parse_args()
    for env in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        os.environ[env] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    codes, x, y, weights = generate(args.rows, args.case, args.seed)
    rng = np.random.default_rng(args.seed + 900)
    ys = [y, x] + [rng.normal(size=len(x)) for _ in range(max(0, args.rhs - 2))]
    Y = np.column_stack(ys)
    del ys
    gc.collect()
    result = dict(case=args.case, method=args.method, rows=len(x), rhs=args.rhs,
                  threads=args.threads, tol=args.tol, reps=args.reps,
                  platform=platform.platform(), python=platform.python_version(),
                  dimensions=[int(codes[:, q].max()) + 1 for q in range(codes.shape[1])])
    rounds = []
    try:
        weighted=args.case=="reweight"
        if args.method=="within":
            from within import Design, Solver, LsmrOptions
            t0=time.perf_counter()
            design=Design(codes)
            source_setup=time.perf_counter()-t0
            options=LsmrOptions(tol=args.tol,maxiter=6000)
            solver=None
            for step in range(args.reps):
                ww=new_weights(weights,step) if weighted else weights
                t0=time.perf_counter()
                if step==0 or weighted:
                    solver=Solver(design,weights=ww)
                state_setup=time.perf_counter()-t0
                t0=time.perf_counter()
                out=solver.solve_batch(Y,options=options)
                solve=time.perf_counter()-t0
                residual=np.asarray(out.demeaned)
                error=cert(codes,Y,ww,residual)
                rounds.append(dict(step=step,source_setup_s=source_setup if step==0 else 0.0,
                                   setup_s=state_setup,solve_s=solve,route="within_base",
                                   iters=[int(k) for k in out.iterations],
                                   converged=[bool(k) for k in out.converged],error=error,
                                   valid=error<5e-8 and all(out.converged)))
        else:
            from _econhdfe_within import NativeSolver, UPSTREAM_REV
            strategy=args.method.removeprefix("native_")
            t0=time.perf_counter()
            solver=NativeSolver(codes,weights=weights,strategy=strategy,threads=args.threads)
            source_setup=time.perf_counter()-t0
            result["upstream_rev"]=UPSTREAM_REV
            for step in range(args.reps):
                ww=new_weights(weights,step) if weighted else weights
                t0=time.perf_counter()
                if step>0 and weighted:
                    solver.update_weights(ww)
                setup=time.perf_counter()-t0
                t0=time.perf_counter()
                residual,conv,iters,internal_errors=solver.solve(Y,tol=args.tol,max_iter=6000)
                solve=time.perf_counter()-t0
                error=cert(codes,Y,ww,np.asarray(residual))
                rounds.append(dict(step=step,source_setup_s=source_setup if step==0 else 0.0,
                                   setup_s=setup,solve_s=solve,route=solver.route,
                                   n_cells=solver.n_cells, iters=[int(k) for k in iters],
                                   converged=[bool(k) for k in conv],error=error,
                                   valid=error<5e-8 and all(conv)))
        result.update(rounds=rounds,valid=all(q["valid"] for q in rounds),
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    except Exception as e:
        result.update(valid=False,error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc()[-1800:])
    print("BENCH_ROUTER "+json.dumps(result,allow_nan=False),flush=True)
    if not result["valid"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
