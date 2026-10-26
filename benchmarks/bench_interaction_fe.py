"""Benchmark the empirically common firm + year + city×year FE pattern.

Example:
    python benchmarks/bench_interaction_fe.py --n 2000000 --firms 300000 --threads 8
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from econhdfe import reghdfe, interaction


def make_data(n, n_firms, n_cities, n_years, mobility, seed):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, n, dtype=np.int32)
    year = rng.integers(0, n_years, n, dtype=np.int16)
    home = rng.integers(0, n_cities, n_firms, dtype=np.int32)
    city = home[firm].copy()
    move = rng.random(n) < mobility
    city[move] = rng.integers(0, n_cities, int(move.sum()), dtype=np.int32)
    city_year = city * n_years + year.astype(np.int32)
    X = rng.normal(size=(n, 3))
    beta = np.array([0.8, -0.4, 0.25])
    y = (
        X @ beta
        + rng.normal(size=n_firms)[firm]
        + rng.normal(size=n_cities * n_years)[city_year]
        + rng.normal(scale=0.5, size=n)
    )
    return y, X, firm, year, city, beta


def run(label, y, X, firm, year, city, beta, *, threads, acceleration, canonicalize, max_iter, allow):
    t0 = time.perf_counter()
    r = reghdfe(
        None, y=y, x=X,
        absorb=[firm, year, interaction(city, year)],
        acceleration=acceleration, canonicalize_fe=canonicalize,
        projection_backend="indexed", absorb_threads=threads,
        max_iter=max_iter, tol=1e-8, allow_nonconverged=allow,
    )
    elapsed = time.perf_counter() - t0
    print({
        "case": label,
        "seconds": round(elapsed, 3),
        "iterations": r.iterations,
        "converged": r.converged,
        "beta_max_abs_error": float(np.max(np.abs(r.params-beta))),
        "effective_fe": r.absorb_info["canonicalization"]["effective"],
        "dropped_fe": r.absorb_info["canonicalization"]["dropped"],
        "criterion": r.absorb_info["convergence_criterion"],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--firms", type=int, default=80_000)
    ap.add_argument("--cities", type=int, default=347)
    ap.add_argument("--years", type=int, default=16)
    ap.add_argument("--mobility", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()
    data = make_data(args.n, args.firms, args.cities, args.years, args.mobility, args.seed)
    y, X, firm, year, city, beta = data
    run("plain_map_raw", y, X, firm, year, city, beta, threads=args.threads,
        acceleration="none", canonicalize=False, max_iter=250, allow=True)
    run("cg_raw", y, X, firm, year, city, beta, threads=args.threads,
        acceleration="cg", canonicalize=False, max_iter=16_000, allow=False)
    run("cg_canonicalized", y, X, firm, year, city, beta, threads=args.threads,
        acceleration="cg", canonicalize=True, max_iter=16_000, allow=False)


if __name__ == "__main__":
    main()
