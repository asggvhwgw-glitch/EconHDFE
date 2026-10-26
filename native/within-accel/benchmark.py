"""Known-residual benchmark; same wrapper policies, not package-default comparisons."""
import argparse
import json
import platform
import resource
import statistics
import time
import numpy as np
from _econhdfe_within import NativeSolver, UPSTREAM_REV


def data(case, n, rhs, seed=271):
    rng = np.random.default_rng(seed)
    if case == "schur":
        n -= n % 32; u = n // 2
        firm = np.arange(u, dtype=np.uint32) // 8
        city = firm % 64
        moves = rng.random(u) < 0.01
        city[moves] = rng.integers(64, size=moves.sum(), dtype=np.uint32)
        year = np.arange(u, dtype=np.uint32) % 16
        codes = np.repeat(np.column_stack((firm, city, year)), 2, axis=0)
    else:
        n -= n % 16; u = n // 16
        base = np.column_stack([rng.integers(max(2, min(l, u//4)), size=u, dtype=np.uint32)
                                for l in (20000, 2000, 80, 16)])
        codes = np.repeat(base, 16, axis=0)
    codes = np.column_stack([np.unique(g, return_inverse=True)[1] for g in codes.T]).astype(np.uint32)
    pair_error = rng.normal(size=(len(codes)//2, rhs))
    error = np.empty((len(codes), rhs)); error[::2] = pair_error; error[1::2] = -pair_error
    weight = np.repeat(np.exp(rng.normal(size=len(codes)//2)), 2) if case == "weighted" else np.ones(len(codes))
    y = error.copy()
    for g in codes.T:
        y += rng.normal(size=(int(g.max())+1, rhs))[g]
    order = rng.permutation(len(codes))
    return codes[order], y[order], weight[order], error[order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["schur", "repeated", "weighted"], required=True)
    p.add_argument("--strategy", choices=["plain", "auto", "compressed"], required=True)
    p.add_argument("--rows", type=int, default=1000000); p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--rhs", type=int, default=4); p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(); codes, y, w, truth = data(a.case, a.rows, a.rhs)
    records = []
    for repeat in range(a.repeat):
        t = time.perf_counter()
        solver = NativeSolver(codes, weights=w, strategy=a.strategy, threads=a.threads)
        setup = time.perf_counter()-t
        t = time.perf_counter(); out, ok, it, e = solver.solve(y, tol=1e-11)
        solve = time.perf_counter()-t
        errors = np.sqrt(np.sum(w[:, None]*(out-truth)**2, axis=0)/np.sum(w[:, None]*truth**2, axis=0))
        assert all(ok) and errors.max() <= 1e-8, (ok, errors, e)
        y2 = 2*y + 3.0
        t = time.perf_counter(); out2, ok2, _, _ = solver.solve(y2, tol=1e-11); reuse = time.perf_counter()-t
        errors2 = np.sqrt(np.sum(w[:, None]*(out2-2*truth)**2, axis=0)/np.sum(w[:, None]*(2*truth)**2, axis=0))
        assert all(ok2) and errors2.max() <= 1e-8
        records.append(dict(setup=setup, solve=solve, total=setup+solve, reuse=reuse,
                            error=float(max(errors.max(), errors2.max())), iterations=it, route=solver.route,
                            cells=solver.n_cells, schur_condition=solver.schur_condition))
        del solver, out, out2
    result = dict(case=a.case, strategy=a.strategy, n=len(y), rhs=a.rhs, threads=a.threads,
                  upstream=UPSTREAM_REV, platform=platform.platform(), python=platform.python_version(),
                  scope="same native wrapper; plain pinned-within single-RHS solves, no package-default claim",
                  runs=records, median_first=statistics.median(r["total"] for r in records),
                  median_reuse=statistics.median(r["reuse"] for r in records),
                  peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
