from __future__ import annotations
import argparse, time
import numpy as np
from econhdfe import reghdfe, FixedEffect

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=1_000_000)
a = p.parse_args()
rng = np.random.default_rng(321)
n = a.n
g_firm = rng.integers(0, max(n//10, 2), n, dtype=np.int32)
g_time = rng.integers(0, 2000, n, dtype=np.int32)
tenure = rng.normal(size=n)
X = rng.normal(size=(n, 4))
beta = np.array([1.0, -0.5, 0.25, 2.0])
G = int(g_firm.max()) + 1
T = int(g_time.max()) + 1
firm_fe = rng.normal(scale=.5, size=G)
firm_slope = rng.normal(scale=.2, size=G)
time_fe = rng.normal(scale=.3, size=T)
y = X @ beta + firm_fe[g_firm] + firm_slope[g_firm]*tenure + time_fe[g_time] + rng.normal(size=n)

t0 = time.perf_counter()
r = reghdfe(
    y=y, x=X,
    absorb=[FixedEffect(g_firm, slopes=(tenure,), intercept=True), g_time],
    drop_singletons=False,
)
print({
    'seconds': time.perf_counter()-t0,
    'params': r.params.tolist(),
    'iters': r.iterations,
    'df_absorbed': r.df_absorbed,
    'n': r.nobs,
})
