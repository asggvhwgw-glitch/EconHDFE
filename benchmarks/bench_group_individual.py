"""Stress benchmark for group-level outcomes with individual fixed effects."""
from __future__ import annotations
import argparse
import resource
import time
import numpy as np
from econhdfe import reghdfe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=400_000)
    p.add_argument("--members-per-group", type=int, default=3)
    p.add_argument("--individuals", type=int, default=200_000)
    p.add_argument("--incidence-backend", choices=["auto", "csr", "matrix_free"], default="auto")
    args = p.parse_args()
    if args.members_per_group > 5:
        raise SystemExit("benchmark fixture currently supports at most five deterministic offsets")

    rng = np.random.default_rng(20260910)
    G, m, I = args.groups, args.members_per_group, args.individuals
    M = G*m
    membership_group = np.repeat(np.arange(G, dtype=np.int32), m)
    base = rng.integers(0, I, size=G, dtype=np.int32)
    offsets = np.array([0, 7919, 15401, 23339, 31847], dtype=np.int32)[:m]
    individual = ((base[:, None] + offsets[None, :]) % I).astype(np.int32).ravel()
    year_g = np.arange(G, dtype=np.int32) % 1000
    year = np.repeat(year_g, m)
    Xg = rng.normal(size=(G, 4))
    beta = np.array([1.0, -0.5, 0.25, 2.0])
    alpha = rng.normal(size=I)
    indiv_fe = np.bincount(membership_group, weights=alpha[individual]/m, minlength=G)
    yg = Xg @ beta + indiv_fe + rng.normal(size=1000)[year_g] + rng.normal(scale=.5, size=G)
    X = np.repeat(Xg, m, axis=0)
    y = np.repeat(yg, m)

    t0 = time.perf_counter()
    r = reghdfe(
        None, y=y, x=X, absorb=[year, individual], group=membership_group,
        individual=individual, aggregation="mean", method="lsmr", vce="robust",
        incidence_backend=args.incidence_backend, drop_singletons=False, tol=1e-8,
        max_iter=3000,
    )
    seconds = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print({
        "groups": G, "memberships": M, "individuals": I,
        "incidence_backend": r.group_info.incidence_backend,
        "seconds": seconds, "iterations": r.iterations,
        "max_rss_mib": rss, "params": r.params.tolist(),
    })


if __name__ == "__main__":
    main()
