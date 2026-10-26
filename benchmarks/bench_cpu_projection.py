from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from econhdfe.hdfe.projection import build_group_index, project_indexed_inplace

p = argparse.ArgumentParser(description="CPU scaling for one indexed FE projection")
p.add_argument("--n", type=int, default=10_000_000)
p.add_argument("--levels", type=int, default=450_000)
p.add_argument("--rhs", type=int, default=6)
p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
p.add_argument("--quick", action="store_true", help="small smoke-sized CPU projection benchmark")
a = p.parse_args()
if a.quick:
    if a.n == 10_000_000:
        a.n = 500_000
    if a.levels == 450_000:
        a.levels = 50_000
    if a.rhs == 6:
        a.rhs = 4
    if a.threads == [1, 2, 4]:
        a.threads = [1, 2]
rng = np.random.default_rng(606)
codes = rng.integers(0, a.levels, a.n, dtype=np.int32)
base = rng.normal(size=(a.n, a.rhs))
# Compile counting-sort/index and projection signatures on a small problem.
small = codes[: min(20_000, a.n)]
idx0 = build_group_index(small, a.levels)
project_indexed_inplace(base[: len(small)].copy(), idx0, threads=max(a.threads))
t = time.perf_counter(); idx = build_group_index(codes, a.levels); build_s = time.perf_counter() - t
print(f"index_build_s={build_s:.6f} index_MiB={idx.nbytes / 2**20:.2f}")
for threads in a.threads:
    x = base.copy()
    t = time.perf_counter(); project_indexed_inplace(x, idx, threads=threads); dt = time.perf_counter() - t
    print(f"threads={threads} projection_s={dt:.6f}")
