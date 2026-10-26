from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econhdfe import PPMLConfig, ppmlhdfe


def _draw_y(rng, eta):
    mu = np.exp(np.clip(eta, -3.0, 3.0))
    return rng.poisson(mu).astype(np.float64)


def two_way(n, seed):
    rng = np.random.default_rng(seed)
    k = 4
    X = rng.normal(size=(n, k))
    g1 = rng.integers(0, max(200, n // 120), n)
    g2 = rng.integers(0, max(150, n // 180), n)
    a = rng.normal(scale=.25, size=g1.max() + 1)
    b = rng.normal(scale=.2, size=g2.max() + 1)
    beta = np.array([.16, -.11, .08, .04])
    y = _draw_y(rng, X @ beta + a[g1] + b[g2])
    return y, X, [g1, g2], [f"x{i+1}" for i in range(k)]


def gravity(_, seed):
    rng = np.random.default_rng(seed)
    ne, ni, nt = 120, 120, 10
    e, i, t = np.meshgrid(np.arange(ne), np.arange(ni), np.arange(nt), indexing="ij")
    e, i, t = e.ravel(), i.ravel(), t.ravel()
    pair = e * ni + i
    ey = e * nt + t
    iy = i * nt + t
    X = rng.normal(size=(len(e), 3))
    beta = np.array([.12, -.07, .05])
    ap = rng.normal(scale=.18, size=ne * ni)
    ae = rng.normal(scale=.12, size=ne * nt)
    ai = rng.normal(scale=.12, size=ni * nt)
    y = _draw_y(rng, X @ beta + ap[pair] + ae[ey] + ai[iy])
    return y, X, [pair, ey, iy], ["x1", "x2", "x3"]


def hierarchy(_, seed):
    rng = np.random.default_rng(seed)
    nf, nt, reps = 10_000, 12, 2
    firm = np.repeat(np.arange(nf), nt * reps)
    year = np.tile(np.repeat(np.arange(nt), reps), nf)
    city_by_firm = rng.integers(0, 600, nf)
    city = city_by_firm[firm]
    province = city // 20
    cy = city * nt + year
    py = province * nt + year
    X = rng.normal(size=(len(firm), 3))
    beta = np.array([.10, -.08, .06])
    af = rng.normal(scale=.18, size=nf)
    acy = rng.normal(scale=.16, size=600 * nt)
    y = _draw_y(rng, X @ beta + af[firm] + acy[cy])
    return y, X, [firm, year, py, cy], ["x1", "x2", "x3"]


SCENARIOS = {"two_way": two_way, "gravity": gravity, "hierarchy": hierarchy}


def run_one(scenario, n, engine, seed):
    y, X, absorb, names = SCENARIOS[scenario](n, seed)
    cfg = PPMLConfig(engine=engine, separation=(), max_iter=200)
    t0 = time.perf_counter()
    result = ppmlhdfe(y=y, x=X, absorb=absorb, names=names, config=cfg, vce="robust")
    elapsed = time.perf_counter() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "scenario": scenario,
        "engine": engine,
        "nobs_input": int(len(y)),
        "nobs_estimation": int(result.nobs),
        "seconds": elapsed,
        "iterations": int(result.iterations),
        "peak_rss_mb_process": float(rss_kb / 1024.0),
        "coef": result.params.tolist(),
        "canonicalization": result.diagnostics.get("canonicalization", {}),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=SCENARIOS, default="two_way")
    p.add_argument("--n", type=int, default=200_000, help="used only by two_way")
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    results = [run_one(args.scenario, args.n, e, args.seed) for e in ("replica", "optimized")]
    results.append({
        "speedup": results[0]["seconds"] / results[1]["seconds"],
        "max_abs_coef_diff": float(np.max(np.abs(np.asarray(results[0]["coef"]) - np.asarray(results[1]["coef"])))),
    })
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
