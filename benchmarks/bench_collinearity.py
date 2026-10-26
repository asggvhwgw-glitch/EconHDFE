from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from econhdfe.collinearity import resolve_collinearity


def main(n=1_000_000, k=20, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    X = np.column_stack([
        X,
        X[:, 0] + X[:, 1],
        2.0 * X[:, 3],
        X[:, 4] - 3.0 * X[:, 5] + 0.5 * X[:, 6],
        X[:, 7] + 1e-5 * rng.normal(size=n),
    ])
    names = [f"x{i}" for i in range(X.shape[1])]
    t0 = time.perf_counter()
    plan = resolve_collinearity(X, names=names, original=X, tolerance=1e-8)
    dt = time.perf_counter() - t0
    print({
        "shape": X.shape,
        "seconds": dt,
        "active": len(plan.active_indices),
        "omitted": [(o.name, o.reason) for o in plan.omitted],
    })


if __name__ == "__main__":
    main()
