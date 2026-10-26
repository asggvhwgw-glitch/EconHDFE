"""Exercise the benchmark's actual result gate with small fault-injection cases.

The CLI's million-row performance minimum stays intact. Only the data factory
is replaced here; these are control-flow checks, never timing evidence.
"""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import budget_preconditioner as bp
import cases
from scalable_hyperfe import dummy


@pytest.mark.parametrize('mode', ['block', 'native_adaptive'])
@pytest.mark.parametrize('fault', ['none', 'nonconvergence', 'audit', 'linearity'])
def test_reuse_controls_overall_passed(tmp_path, monkeypatch, mode, fault):
    import within

    rng = np.random.default_rng(9201)
    codes = np.column_stack([rng.integers(0, k, 300) for k in (7, 6, 5)])
    weights = np.exp(rng.normal(size=300))
    rhs = rng.normal(size=(300, 3))
    d = dummy(codes, [7, 6, 5]).toarray()
    sw = np.sqrt(weights)
    coef = np.linalg.lstsq(sw[:, None] * d, sw[:, None] * rhs, rcond=None)[0]
    truth = (rhs - d @ coef)[:, -1]
    monkeypatch.setattr(cases, 'make_case', lambda *args: (codes, weights, rhs, truth))
    calls = 0

    def inject(residual, converged):
        nonlocal calls
        calls += 1
        residual = np.array(residual, copy=True)
        converged = list(converged)
        if calls == 2:
            if fault == 'nonconvergence':
                converged[0] = False
            elif fault == 'audit':
                residual[:, -1] += 1.0
            elif fault == 'linearity':
                # Remains FE-orthogonal and leaves the oracle column correct,
                # but is the wrong projection for the requested first RHS.
                residual[:, 0] += 0.1 * truth
        return residual, converged

    if mode == 'block':
        solve = bp.solve_rect

        def checked_solve(*args, **kwargs):
            residual, info = solve(*args, **kwargs)
            residual, converged = inject(residual, [r['status'] in (1, 2, 4, 5) for r in info['rhs']])
            for row, ok in zip(info['rhs'], converged):
                if not ok:
                    row['status'] = 7
            return residual, info

        monkeypatch.setattr(bp, 'solve_rect', checked_solve)
    else:
        solver_type = within.Solver

        class CheckedSolver:
            def __init__(self, *args, **kwargs):
                self.solver = solver_type(*args, **kwargs)
                self.preconditioner = self.solver.preconditioner

            def solve_batch(self, *args, **kwargs):
                result = self.solver.solve_batch(*args, **kwargs)
                residual, converged = inject(result.demeaned, result.converged)
                return SimpleNamespace(demeaned=residual, converged=converged, iterations=result.iterations)

        monkeypatch.setattr(within, 'Solver', CheckedSolver)

    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', [str(ROOT / 'benchmark.py'), '--case', 'smooth_sparse',
                                    '--mode', mode, '--tau', '0', '--n', '1000000',
                                    '--tol', '1e-13', '--reuse', '--out', str(output)])
    runpy.run_path(str(ROOT / 'benchmark.py'), run_name='__main__')
    record = json.loads(output.read_text())
    assert calls == 2
    assert record['status'] == 'completed'
    assert all(record['converged']) and record['audit']['projection_gate_passed']
    assert record['passed'] is (fault == 'none')
    assert record['reuse']['passed'] is (fault == 'none')
    if fault == 'nonconvergence':
        assert not all(record['reuse']['converged'])
        assert record['reuse']['audit']['projection_gate_passed']
    elif fault == 'audit':
        assert all(record['reuse']['converged'])
        assert not record['reuse']['audit']['projection_gate_passed']
    elif fault == 'linearity':
        assert all(record['reuse']['converged'])
        assert record['reuse']['audit']['projection_gate_passed']
        assert record['reuse']['linearity_error'][0] > 1e-8
