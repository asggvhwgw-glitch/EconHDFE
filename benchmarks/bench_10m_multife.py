from __future__ import annotations
import argparse, resource, time
import numpy as np
from econhdfe import reghdfe

p = argparse.ArgumentParser(description="Large ordinary-HDFE stress benchmark")
p.add_argument("--n", type=int, default=10_000_000)
p.add_argument("--controls", type=int, default=12)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--pool-size", default="auto", help="integer or auto")
p.add_argument("--memory-budget-mb", type=float, default=512)
a = p.parse_args()
pool = a.pool_size if a.pool_size == "auto" else int(a.pool_size)
levels = (300_000, 350_000, 400_000, 450_000)
rng = np.random.default_rng(20260910)
groups = [rng.integers(0, g, a.n, dtype=np.int32) for g in levels]
X = rng.normal(size=(a.n, a.controls))
beta = np.linspace(-0.8, 1.0, a.controls)
y = X @ beta
for codes, g in zip(groups, levels, strict=False):
    y += rng.normal(scale=0.8, size=g)[codes]
y += rng.normal(scale=0.5, size=a.n)
t = time.perf_counter()
r = reghdfe(
    None, y=y, x=X, absorb=groups, cluster=groups[:2], vce="cluster",
    projection_backend="auto", absorb_threads=a.threads, pool_size=pool,
    memory_budget_mb=a.memory_budget_mb, tol=1e-8,
)
dt = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"seconds={dt:.6f} iterations={r.iterations} max_rss_MiB={rss:.1f}")
print("max_abs_beta_error=", float(np.max(np.abs(r.params - beta))))
