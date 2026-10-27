"""Matched weighted-absorber benchmark for nonlinear IRLS workloads.

Times include numerical weight updates and projection. This is a kernel
benchmark, not a claim about end-to-end PPML or IV-PPML speedup.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from econhdfe.hdfe.weighted_projection import WeightedFEProjector


def sample(n, rhs, seed):
    rng = np.random.default_rng(seed)
    # Dense crossing, nontrivial overlap, many repeated cells.
    base_n = (n + 7) // 8
    base = np.column_stack([
        rng.integers(0, 12000, base_n, dtype=np.int32),
        rng.integers(0, 2000, base_n, dtype=np.int32),
        rng.integers(0, 48, base_n, dtype=np.int32),
    ])
    groups = tuple(np.repeat(base[:, i], 8)[:n] for i in range(3))
    x = rng.standard_normal((n, rhs))
    latent = rng.standard_normal(n)
    w0 = np.exp(np.clip(0.4 * latent, -2, 2))
    weights = [np.ascontiguousarray(w0 * np.exp(0.16 * np.sin(latent + t * 0.7)))
               for t in range(5)]
    return groups, x, weights


def run(groups, x, weights, method, budget, tol, threads):
    p = WeightedFEProjector(
        groups, engine="optimized", method=method, absorb_threads=threads,
        projection_memory_budget_mb=budget,
    )
    records = []
    outputs = []
    for i, w in enumerate(weights):
        t0 = time.perf_counter()
        absorber = p.prepare(w, tol=tol)
        update = time.perf_counter() - t0
        t1 = time.perf_counter()
        out, info = absorber.residualize(x, return_info=True, copy=True,
                                         absorb_threads=threads)
        solve = time.perf_counter() - t1
        if not info.converged:
            raise RuntimeError(f"{method} did not converge on round {i}: {info}")
        # Weighted FE orthogonality on the original observations.
        max_group_error = 0.0
        for g in groups:
            denom = np.bincount(g, weights=w)
            for j in range(out.shape[1]):
                means = np.bincount(g, weights=w * out[:, j]) / np.maximum(denom, 1e-300)
                max_group_error = max(max_group_error, float(np.max(np.abs(means))))
        records.append(dict(round=i, update_seconds=update, solve_seconds=solve,
                            total_seconds=update + solve,
                            iterations=int(info.iterations), orthogonality=max_group_error))
        outputs.append(out)
    return records, outputs, p.resource_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000000)
    ap.add_argument("--rhs", type=int, default=3)
    ap.add_argument("--budget-mb", type=float, default=1536)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261027)
    a = ap.parse_args()
    groups, x, weights = sample(a.rows, a.rhs, a.seed)
    results = {}
    ref = None
    for method in ("map", "within"):
        rec, outs, resource = run(groups, x, weights, method, a.budget_mb, a.tol, a.threads)
        results[method] = dict(rounds=rec, resources=resource,
                               median_update=statistics.median(v["update_seconds"] for v in rec[1:]),
                               median_solve=statistics.median(v["solve_seconds"] for v in rec[1:]),
                               median_total=statistics.median(v["total_seconds"] for v in rec[1:]))
        if ref is None:
            ref = outs
        else:
            errors = [float(np.linalg.norm((r - v).ravel()) / max(np.linalg.norm(r.ravel()), 1e-30))
                      for r, v in zip(ref, outs)]
            results["relative_errors"] = errors
            if max(errors) > 1e-6:
                raise AssertionError(f"weighted within parity failed: {errors}")
    m, w = results["map"], results["within"]
    results["speedup_update"] = m["median_update"] / max(w["median_update"], 1e-12)
    results["speedup_solve"] = m["median_solve"] / max(w["median_solve"], 1e-12)
    results["speedup_total"] = m["median_total"] / max(w["median_total"], 1e-12)
    print(json.dumps(dict(rows=a.rows, rhs=a.rhs, tol=a.tol, threads=a.threads,
                          platform=platform.platform(), pid=os.getpid(), results=results),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
