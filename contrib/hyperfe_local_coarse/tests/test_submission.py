"""Submission checks; small cases test correctness, never performance."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scalable_hyperfe import ProfiledSystem, dummy
from budget_preconditioner import ComponentRight, build_star, solve_rect


@pytest.mark.parametrize('permutation', [(0, 1, 2), (2, 0, 1)])
def test_projection_against_dense_weighted_reference(permutation):
    rng = np.random.default_rng(2901)
    codes = np.column_stack([rng.integers(0, k, 600) for k in (10, 8, 6)])
    weights = np.exp(rng.normal(size=600))
    rhs = rng.normal(size=(600, 3))
    design = dummy(codes, [10, 8, 6]).toarray()
    sw = np.sqrt(weights)
    coef = np.linalg.lstsq(sw[:, None] * design, sw[:, None] * rhs, rcond=None)[0]
    expected = rhs - design @ coef
    rows = rng.permutation(600)
    reordered = codes[rows][:, permutation]
    system = ProfiledSystem.build(reordered, weights[rows])
    matrix, scale, _ = build_star(system)
    root = ComponentRight(matrix, scale, tau=0, max_block=64, eig_cut=0)
    actual, info = solve_rect(system, rhs[rows], root, tol=1e-13)
    assert all(r['status'] in (1, 2, 4, 5) for r in info['rhs'])
    np.testing.assert_allclose(actual, expected[rows], atol=2e-10, rtol=2e-10)


def test_native_within_reference_contract():
    import within
    rng = np.random.default_rng(9201)
    codes = np.column_stack([rng.integers(0, k, 300) for k in (7, 6, 5)])
    weights = np.exp(rng.normal(size=300))
    rhs = rng.normal(size=(300, 3))
    d = dummy(codes, [7, 6, 5]).toarray()
    sw = np.sqrt(weights)
    coef = np.linalg.lstsq(sw[:, None] * d, sw[:, None] * rhs, rcond=None)[0]
    solver = within.Solver(within.Design(np.asfortranarray(codes, dtype=np.uint32)),
                           weights=weights, preconditioner=within.PreconditionerConfig.Adaptive())
    result = solver.solve_batch(np.asfortranarray(rhs), options=within.LsmrOptions(tol=1e-13, maxiter=1000))
    assert all(result.converged)
    np.testing.assert_allclose(result.demeaned, rhs - d @ coef, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('args', [['--case', 'typo'], ['--n', '100']])
def test_benchmark_rejects_wrong_case_or_small_performance_run(tmp_path, args):
    run = subprocess.run([sys.executable, str(ROOT / 'benchmark.py'),
                          '--out', str(tmp_path / 'never.json'), *args],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode != 0
    assert not (tmp_path / 'never.json').exists()
