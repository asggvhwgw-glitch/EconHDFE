"""Numerical and execution regressions for pooled warm-start CG."""

import numpy as np
from econhdfe.hdfe.absorber import HDFEAbsorber


def cycle(m=80):
    i = np.arange(m, dtype=np.int32)
    rows = np.vstack([np.column_stack([i, i, i]),
                      np.column_stack([(i + 1) % m, i, i])])
    return [rows[:, j] for j in range(3)]


def test_auto_respects_total_iteration_budget():
    groups = cycle()
    rhs = np.random.default_rng(1844).normal(size=(160, 3))
    absorber = HDFEAbsorber(
        groups, tol=1e-14, max_iter=3, acceleration="auto",
        core_reduction="off", absorb_threads=2,
    )
    absorber._profile_enabled = True
    _, info = absorber.residualize(rhs, return_info=True)
    assert not info.converged
    assert info.iterations == 3
    assert absorber._iteration_profile["cg_iterations"] == 1
