"""Regression: an internal Krylov residual flag need not bound max group means."""
from pathlib import Path
import importlib.util
import numpy as np
from econhdfe_dual import DualAbsorber


def test_sparse_group_certificate_tightens_inner_correction():
    path = Path(__file__).resolve().parents[1] / 'benchmark.py'
    spec = importlib.util.spec_from_file_location('dual_oracle_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g, y, w, truth = module.corpus('two_big', 1_000_000, 2)
    solver = DualAbsorber(g, weights=w, backend='within', threads=2, allow_fallback=False)
    out, info = solver.residualize(y, tol=1e-10, return_info=True)
    assert info['accepted'] and info['backend'] == 'within'
    assert max(info['certificate']) <= 1e-10
    errors = np.sqrt(np.sum(w[:, None]*(out-truth)**2, axis=0)/np.sum(w[:, None]*truth**2, axis=0))
    assert errors.max() <= 2e-8
    attempts = info['attempts']
    for i in range(1, len(attempts)):
        if attempts[i]['correction'] > 0:
            assert attempts[i]['inner_tolerance'] < attempts[i-1]['inner_tolerance']
    # The user-requested gate must not be loosened to force a pass.
    assert info['certificate_gate'] == 1e-10
