#!/usr/bin/env python3
"""Topology-controlled fixed-effect benchmark: fereg vs within 0.4.0.

One Python process per (case, method, n), no cross-method caches.
within: native residualization + minimal FWL OLS (no variance estimation).
fereg: complete feols() estimator (includes FE handling and variance).
Consequently fereg total and within total are NOT pure kernel-to-kernel times.
Structured JSON lines include independent normal-equation certificates.
All timing excludes synthetic-data generation and import time.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import time
import traceback

import numpy as np


def balanced(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """All IDs occur at least floor(n/k) times, avoiding singleton artifacts."""
    g = np.arange(n, dtype=np.uint32) % np.uint32(k)
    rng.shuffle(g)
    return g


def generate(n: int, case: str, seed: int):
    rng = np.random.default_rng(seed)
    if case == "one_big":
        dims = [max(2, n // 4), 48, 16]
        codes = np.column_stack([balanced(n, g, rng) for g in dims])
    elif case in ("two_big", "reweight"):
        dims = [max(2, n // 8), max(2, n // 12), 16]
        codes = np.column_stack([balanced(n, g, rng) for g in dims])
    elif case == "three_big":
        dims = [max(2, n // 10), max(2, n // 12), max(2, n // 16)]
        codes = np.column_stack([balanced(n, g, rng) for g in dims])
    elif case == "weak_links":
        # Locally well connected bipartite components with rare inter-block edges.
        communities = 32
        a_per = max(2, n // 12 // communities)
        b_per = max(2, n // 16 // communities)
        i = np.arange(n, dtype=np.uint32)
        community = i % communities
        a = community * a_per + (i // communities) % a_per
        b = community * b_per + rng.integers(b_per, size=n, dtype=np.uint32)
        bridge = rng.random(n) < 0.0003
        b[bridge] = rng.integers(0, b_per * communities, size=int(bridge.sum()), dtype=np.uint32)
        order = rng.permutation(n)
        codes = np.column_stack([a[order], b[order]])
    elif case == "hub":
        g0, g1 = max(2, n // 10), max(2, n // 24)
        a = balanced(n, g0, rng)
        i = np.arange(n, dtype=np.uint32)
        b = i % g1
        # Each nonhub B level retains >= 5 observations, 75%+ visits go to 1% hubs.
        b[i // g1 >= 5] = rng.integers(0, max(2, g1 // 100), size=int((i // g1 >= 5).sum()), dtype=np.uint32)
        rng.shuffle(b)
        codes = np.column_stack([a, b, balanced(n, 16, rng)])
    elif case == "repeated_cells":
        repeat = 8
        u = n // repeat
        n = u * repeat
        base = np.column_stack([
            balanced(u, max(2, u // 3), rng),
            balanced(u, max(2, u // 6), rng),
            balanced(u, 32, rng),
        ])
        codes = np.repeat(base, repeat, axis=0)
        order = rng.permutation(n)
        codes = codes[order]
    else:
        raise ValueError(case)

    codes = np.asfortranarray(codes, dtype=np.uint32)
    x = rng.normal(size=n)
    y = 0.83 * x + rng.normal(scale=0.9, size=n)
    for j in range(codes.shape[1]):
        g = int(codes[:, j].max()) + 1
        effects = rng.normal(scale=0.4 / (j + 1) ** 0.5, size=g)
        y += effects[codes[:, j]]
    w = np.exp(np.clip(rng.normal(scale=0.6, size=n), -2.0, 2.0))
    return codes, np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(w)


def new_weights(w: np.ndarray, iteration: int) -> np.ndarray:
    if iteration == 0:
        return w
    i = np.arange(len(w), dtype=np.float64)
    return w * np.exp(0.18 * np.sin(i * (0.011 + 0.002 * iteration)))


def certificate(codes, x, y, w, resid, beta):
    """Independent orthogonality and scalar slope normal-equation tests."""
    if resid is None or len(resid) != len(x):
        return {"valid": False, "error": "missing or shortened residual"}
    resid = np.asarray(resid, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(resid)):
        return {"valid": False, "error": "nonfinite residual"}
    sumw = float(w.sum())
    scale_y = np.sqrt(np.dot(w, y * y) / sumw)
    score = []
    wr = w * resid
    for j in range(codes.shape[1]):
        col = np.ascontiguousarray(codes[:, j])
        numer = np.bincount(col, weights=wr)
        mass = np.bincount(col, weights=w)
        where = mass > 0
        mean = np.zeros_like(numer)
        mean[where] = numer[where] / mass[where]
        score.append(float(np.sqrt(np.dot(mass, mean * mean) / sumw) / max(scale_y, 1e-30)))
    slope = abs(np.dot(w * x, resid)) / max(np.sqrt(np.dot(w, x * x) * np.dot(w, y * y)), 1e-30)
    return {
        "valid": bool(np.isfinite(beta) and max(score) <= 3e-6 and slope <= 3e-6),
        "fe_moment_max": float(max(score)),
        "fe_moments": score,
        "x_moment": float(slope),
        "beta": float(beta),
        "resid_rms": float(np.sqrt(np.dot(w, resid * resid) / sumw)),
    }


def rss_mb():
    # Linux ru_maxrss is KB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def within_run(method, codes, x, y, w0, n_rounds, tol, maxiter, do_reweight):
    from within import Design, Solver, LsmrOptions, PreconditionerConfig

    t0 = time.perf_counter()
    design = Design(codes)
    t_design = time.perf_counter() - t0
    opts = LsmrOptions(tol=tol, maxiter=maxiter)
    pre = PreconditionerConfig.Additive() if method == "within_additive" else None
    Y = np.column_stack((y, x))
    rounds = []
    solver = None
    for iteration in range(n_rounds):
        weights = new_weights(w0, iteration) if do_reweight else None
        gc.collect()
        t0 = time.perf_counter()
        if iteration == 0 or do_reweight:
            solver = Solver(design, weights=weights, preconditioner=pre)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = solver.solve_batch(Y, options=opts)
        projection = time.perf_counter() - t0
        rx, xx = result.demeaned[:, 0], result.demeaned[:, 1]
        ww = w0 if weights is None else weights
        beta = float(np.dot(ww * xx, rx) / np.dot(ww * xx, xx))
        residual = rx - beta * xx
        c = certificate(codes, x, y, ww, residual, beta)
        c.update({
            "iteration": iteration,
            "design_seconds": t_design if iteration == 0 else 0,
            "build_seconds": build,
            "projection_seconds": projection,
            "seconds": build + projection + (t_design if iteration == 0 else 0),
            "iters": [int(i) for i in result.iterations],
            "converged": [bool(v) for v in result.converged],
            "peak_rss_mb": rss_mb(),
        })
        c["valid"] = bool(c["valid"] and all(c["converged"]))
        rounds.append(c)
        del result, rx, xx, residual
    return rounds


def fereg_run(codes, x, y, w0, n_rounds, tol, maxiter, do_reweight):
    import fereg
    import pandas as pd

    cols = {f"f{j}": codes[:, j].astype(np.int64, copy=True) for j in range(codes.shape[1])}
    cols.update(x=x, y=y)
    if do_reweight:
        cols["w"] = w0.copy()
    df = pd.DataFrame(cols)
    formula = "y ~ x | " + " + ".join(f"f{j}" for j in range(codes.shape[1]))
    rounds = []
    for iteration in range(n_rounds):
        wi = new_weights(w0, iteration) if do_reweight else None
        if do_reweight:
            df["w"] = wi
        gc.collect()
        t0 = time.perf_counter()
        fit = fereg.feols(
            formula, data=df, weights="w" if do_reweight else None,
            fixef_rm="none", vcov="iid", fixef_tol=tol, fixef_maxiter=maxiter,
            nthreads=int(os.environ.get("RAYON_NUM_THREADS", "2")),
        )
        elapsed = time.perf_counter() - t0
        beta = float(fit.coef()["x"])
        residual = np.asarray(fit.resid())
        ww = w0 if wi is None else wi
        c = certificate(codes, x, y, ww, residual, beta)
        c.update({
            "iteration": iteration,
            "seconds": elapsed,
            "peak_rss_mb": rss_mb(),
            "nobs_fit": int(fit._N),
            "scope": "full_fereg_ols_including_statistics",
        })
        c["valid"] = bool(c["valid"] and c["nobs_fit"] == len(x))
        rounds.append(c)
        del fit, residual
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["one_big", "two_big", "three_big", "weak_links", "hub", "repeated_cells", "reweight"])
    ap.add_argument("--method", required=True, choices=["fereg", "within", "within_additive"])
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2, help="Number of repeats following the first solve")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=10292026)
    a = ap.parse_args()
    for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # keep numpy single threaded; solver uses RAYON
    t0 = time.perf_counter()
    codes, x, y, w = generate(a.rows, a.case, a.seed)
    generated = time.perf_counter() - t0
    gc.collect()
    dims = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
    meta = dict(
        case=a.case, method=a.method, n=int(len(x)), dimensions=dims,
        threads=a.threads, tol=a.tol, maxiter=a.maxiter, seed=a.seed,
        platform=platform.platform(), python=platform.python_version(),
        rss_after_dgp_mb=rss_mb(), dgp_seconds=generated,
        scope="fereg full estimator vs within minimal FWL OLS; not raw-kernel parity",
    )
    try:
        rounds = (fereg_run(codes, x, y, w, 1 + a.reps, a.tol, a.maxiter, a.case == "reweight")
                  if a.method == "fereg"
                  else within_run(a.method, codes, x, y, w, 1 + a.reps, a.tol, a.maxiter, a.case == "reweight"))
        times = [q["seconds"] for q in rounds]
        meta.update(status="ok" if all(q["valid"] for q in rounds) else "invalid",
                    rounds=rounds, median_seconds=float(statistics.median(times)),
                    cold_seconds=times[0], peak_rss_mb=rss_mb())
    except Exception as ex:
        meta.update(status="error", error=f"{type(ex).__name__}: {ex}", traceback=traceback.format_exc()[-3000:])
    print("BENCH_RESULT " + json.dumps(meta, allow_nan=False), flush=True)
    if meta["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
