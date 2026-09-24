"""Finite-target guards and bounded/reset candidate-phase regression tests."""
import os
from fractions import Fraction

import numpy as np
import pytest

from boundedqr import bootstrap, solve_perturbations
from boundedqr.preprocess import CFMPreprocessor
from boundedqr.batched_repair import iter_repair_batches
from boundedqr.positive_pseudo import pseudo_residual_lower


@pytest.mark.parametrize('tau,w,expected,pseudo_sign', [(.5, -1., -.5, 0), (.75, 1.5, 1., -1)])
@pytest.mark.parametrize('engine', ['scalar', 'batched'])
def test_finite_fallback_when_positive_face_is_infeasible(tau, w, expected, pseudo_sign, engine):
    x, y, ww = np.ones((1, 1)), np.ones(1), np.array([[w]])
    pre = CFMPreprocessor(x, y, tau, np.zeros(1), positive_pseudo=True)
    if engine == 'scalar':
        beta, fit = pre._fit_one(ww[:, 0], np.zeros(1), np.ones(1))
    else:
        values, dual, block = next(iter_repair_batches(pre, ww, np.zeros((1, 1)), positive_pseudo=True))
        beta, fit = values[0], block['fits'][0]
        assert np.isfinite(dual).all()
    np.testing.assert_allclose(beta, [expected], atol=1e-9)
    assert fit['numerically_checked'] and fit['positive_pseudo']['free_restart']
    assert not fit['positive_pseudo']['accepted']
    assert fit['positive_pseudo']['lp_failures'] > 0
    assert abs(fit['pseudo_residual']) < 1e-9 if pseudo_sign==0 else fit['pseudo_residual'] < 0


def test_zero_pseudo_uses_free_branch_and_bootstrap_forwards_option():
    result = bootstrap(np.ones((9, 1)), np.zeros(9), np.arange(9)%3,
                       reps=3, base_coefficients=np.zeros(1), positive_pseudo=True, threads=1)
    np.testing.assert_array_equal(result.bootstrap_coefficients, np.zeros((3, 1, 1)))
    records = result.diagnostics['quantiles'][0]['solver']['fits']
    assert all(r['positive_pseudo']['free_restart'] for r in records)
    assert all(r['positive_pseudo']['fallback_reason']=='pseudo_residual_not_strictly_positive' for r in records)


def test_face_optimum_with_negative_pseudo_is_rejected_and_restarted(monkeypatch):
    from boundedqr import batched_repair
    original = batched_repair._lp
    calls = []

    def candidate(matrix, response, tau, rhs, **kwargs):
        calls.append((matrix.copy(), response.copy(), rhs.copy()))
        if len(matrix)==1:  # beta=2 is a valid optimum of this tilted face.
            return np.array([2.]), np.array([-.5])
        return original(matrix, response, tau, rhs, **kwargs)

    monkeypatch.setattr(batched_repair, '_lp', candidate)
    pre = CFMPreprocessor(np.ones((1, 1)), np.ones(1), .5, np.zeros(1))
    beta, _, block = next(iter_repair_batches(pre, np.array([[.5]]), np.zeros((1, 1)), positive_pseudo=True))
    np.testing.assert_allclose(beta, [[1.]])
    fit = block['fits'][0]
    assert fit['positive_pseudo']['residual_lower_bound'] < 0
    assert fit['positive_pseudo']['free_restart'] and not fit['positive_pseudo']['accepted']
    assert [len(call[0]) for call in calls] == [1, 2]


@pytest.mark.parametrize('engine', ['scalar', 'batched'])
def test_recoverable_face_infeasibility_doubles_and_is_checked(engine):
    x, y, w = np.ones((101, 1)), np.arange(-50., 51.), np.array([[15.]])
    pre = CFMPreprocessor(x, y, .5, np.zeros(1), factor=.01, positive_pseudo=True)
    if engine=='scalar':
        beta, fit = pre._fit_one(w[:, 0], np.zeros(1), np.ones(101))
    else:
        values, _, block = next(iter_repair_batches(pre, w, np.zeros((1, 1)), positive_pseudo=True))
        beta, fit = values[0], block['fits'][0]
    info = fit['positive_pseudo']
    assert fit['numerically_checked'] and info['accepted']
    assert info['lp_failures'] > 0 and info['residual_lower_bound'] > 0
    assert all(b <= 2*a for a, b in zip(info['active_sizes'], info['active_sizes'][1:]))
    reference, _ = solve_perturbations(x, y, .5, w, np.zeros(1), threads=1)
    np.testing.assert_allclose(beta, reference[0], atol=1e-8)


def test_nonbinary_tau_uses_stored_row_product_in_face_rhs(monkeypatch):
    from boundedqr import batched_repair
    tau, w = .1, .9726197088202265
    d = w/tau
    assert d*tau != w
    assert Fraction.from_float(d)*Fraction.from_float(tau) != Fraction.from_float(w)
    original = batched_repair._lp
    observed = []

    def observe(matrix, response, tt, rhs, **kwargs):
        if len(matrix)==1:
            observed.append(rhs.copy())
        return original(matrix, response, tt, rhs, **kwargs)

    monkeypatch.setattr(batched_repair, '_lp', observe)
    pre = CFMPreprocessor(np.ones((1, 1)), np.ones(1), tau, np.zeros(1))
    _, _, block = next(iter_repair_batches(pre, np.array([[w]]), np.zeros((1, 1)), positive_pseudo=True))
    assert block['all_checked']
    np.testing.assert_array_equal(observed[0], [-d*tau])


def test_huge_wrong_set_is_capped_and_free_restart_restores_initial_problem(monkeypatch):
    from boundedqr import batched_repair
    original = batched_repair._lp
    positive_calls, free_calls = [], []

    def candidate(matrix, response, tau, rhs, **kwargs):
        if np.all(matrix[-1]==0):  # The free problem's actual d=0 row.
            free_calls.append((matrix.copy(), response.copy(), rhs.copy()))
            return original(matrix, response, tau, rhs, **kwargs)
        positive_calls.append((matrix.copy(), response.copy(), rhs.copy()))
        return np.array([1000.]), np.zeros(len(matrix))

    monkeypatch.setattr(batched_repair, '_lp', candidate)
    pre = CFMPreprocessor(np.ones((101, 1)), np.arange(-50., 51.), .5, np.zeros(1))
    beta, _, block = next(iter_repair_batches(pre, np.zeros((1, 1)), np.zeros((1, 1)),
                                             positive_pseudo=True, max_rounds=2))
    assert block['all_checked'] and abs(beta[0, 0]) < 1e-9
    fit = block['fits'][0]
    assert fit['sign_repairs'][0] > 4 and fit['positive_pseudo']['active_sizes'] == [4, 8]
    assert fit['positive_pseudo']['free_restart']
    assert fit['positive_pseudo']['fallback_reason'] == 'positive_round_limit'
    np.testing.assert_array_equal(free_calls[0][0][:-1], positive_calls[0][0])
    np.testing.assert_array_equal(free_calls[0][1][:-1], positive_calls[0][1])
    np.testing.assert_array_equal(free_calls[0][2], positive_calls[0][2])


def test_strict_guard_does_not_accept_rounded_zero_or_underflow():
    tiny = np.nextafter(0., 1.)
    assert pseudo_residual_lower(np.ones(1), 1., np.ones(1)) < 0
    assert pseudo_residual_lower(np.zeros(1), 0., np.zeros(1)) < 0
    assert pseudo_residual_lower(np.zeros(1), tiny, np.zeros(1)) <= 0
    assert pseudo_residual_lower(np.array([.1]), 10., np.array([2.])) > 0


@pytest.mark.skipif(os.environ.get('BOUNDEDQR_RUN_GPU_TESTS')!='1', reason='Opt-in tiny GPU test')
@pytest.mark.parametrize('tau', [.1, .5, .9])
def test_guarded_gpu_and_cpu_match_fixed_finite_problems(tau):
    rng = np.random.default_rng(610)
    x = np.column_stack((np.ones(121), rng.normal(size=(121, 2))))
    y = x@np.array([.3, -.5, 1.]) + rng.normal(size=121)
    w = rng.normal(size=(3, 3))*.25
    reference, _ = solve_perturbations(x, y, tau, w, np.zeros(3), threads=1)
    cpu, cdiag = solve_perturbations(x, y, tau, w, np.zeros(3), threads=1, positive_pseudo=True)
    gpu, diag = solve_perturbations(x, y, tau, w, np.zeros(3), backend='gpu',
                                   audit_backend='gpu_fused', positive_pseudo=True,
                                   batch_size=2, threads=1)
    np.testing.assert_allclose(cpu, reference, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(gpu, reference, atol=2e-8, rtol=2e-8)
    for result in (cdiag, diag):
        assert result['all_checked'] and result['positive_pseudo']
        assert all(np.isfinite(f['radius_certificate']['radius']) for f in result['fits'])
