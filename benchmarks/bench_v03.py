"""Release-style v0.3 stress benchmark. Timings depend on hardware and BLAS."""
from __future__ import annotations
import resource
import time
import numpy as np
from econhdfe import reghdfe, ivreghdfe


def mib_rss():
    # Linux ru_maxrss is KiB; this benchmark is intended for the release Linux container.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    rng = np.random.default_rng(20260910)
    n0 = 3000
    f0 = rng.integers(0, 300, n0); t0 = rng.integers(0, 30, n0)
    X0 = rng.normal(size=(n0, 2)); y0 = X0 @ np.array([1.0, -0.5]) + rng.normal(size=n0)
    reghdfe(None, y=y0, x=X0, absorb=[f0, t0], vce="robust")

    n = 1_000_000
    firm = rng.integers(0, 100_000, n, dtype=np.int32)
    year = rng.integers(0, 2_000, n, dtype=np.int32)
    X = rng.normal(size=(n, 4)); beta = np.array([1.0, -0.5, 0.25, 2.0])
    y = X @ beta + rng.normal(size=100_000)[firm] + rng.normal(size=2_000)[year] + rng.normal(size=n)
    clusters = [(firm % 20_000).astype(np.int32), (year % 500).astype(np.int32), ((firm + year) % 1000).astype(np.int32)]
    tic = time.perf_counter()
    r = reghdfe(None, y=y, x=X, absorb=[firm, year], cluster=clusters, vce="cluster")
    print(f"OLS 1m: {time.perf_counter()-tic:.3f}s; sweeps={r.iterations}; RSS={mib_rss():.1f} MiB; b={r.params}")

    n = 300_000
    firm = rng.integers(0, 30_000, n, dtype=np.int32); year = rng.integers(0, 1000, n, dtype=np.int32)
    w = rng.normal(size=n); z = rng.normal(size=(n, 3)); v1 = rng.normal(size=n); v2 = rng.normal(size=n)
    x1 = .55*z[:, 0] + .25*z[:, 1] + .15*w + v1
    x2 = .20*z[:, 0] + .50*z[:, 2] - .10*w + v2
    y = 1.2*x1 - .7*x2 + .2*w + .35*v1 + .25*v2 + rng.normal(size=n) + rng.normal(size=30_000)[firm] + rng.normal(size=1000)[year]
    tic = time.perf_counter()
    r = ivreghdfe(None, y=y, exog=w, endog=np.c_[x1, x2], instruments=z, absorb=[firm, year], vce="robust")
    print(f"IV 300k: {time.perf_counter()-tic:.3f}s; sweeps={r.iterations}; RSS={mib_rss():.1f} MiB; b={r.params}; KP-F={r.diagnostics['kleibergen_paap']['rk_wald_f']:.3f}")


if __name__ == "__main__":
    main()
