import numpy as np
import pandas as pd
from econhdfe import reghdfe


def test_singleton_pruning_can_reveal_new_exact_nesting():
    # Before pruning, B=0 maps to both A=0 and A=1, so A is not nested in B.
    # The sole violating row is a singleton in C.  On the final estimation
    # sample, B refines A exactly and the second canonicalization can remove A.
    n_groups = 10
    reps = 6
    b = np.repeat(np.arange(n_groups), reps)
    a = b // 2
    a = a.copy()
    a[0] = 1
    c = np.tile(np.arange(reps), n_groups)
    c = c.copy()
    c[0] = 99
    rng = np.random.default_rng(805)
    x = rng.normal(size=b.size)
    y = .8*x + rng.normal(size=n_groups)[b] + rng.normal(size=6)[np.minimum(c, 5)] + rng.normal(scale=.1, size=b.size)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "x": x, "y": y})
    r = reghdfe(df, y="y", x=["x"], absorb=["a", "b", "c"], tol=1e-9)
    plan = r.absorb_info["canonicalization"]
    assert r.dropped_singletons >= 1
    assert plan["passes"] == 2
    assert "a" not in plan["effective"]
    assert "b" in plan["effective"]
