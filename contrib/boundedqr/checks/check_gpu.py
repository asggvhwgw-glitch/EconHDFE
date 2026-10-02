"""Opt-in, tiny GPU integration checks; no timing or scaling benchmarks.

Run with BOUNDEDQR_RUN_GPU_TESTS=1. The default test run neither imports CuPy
nor initializes CUDA. These checks exercise n<=256 and p<=4 only.
"""
import os
import numpy as np
import pytest

from boundedqr import solve_perturbations

pytestmark = pytest.mark.skipif(
    os.environ.get('BOUNDEDQR_RUN_GPU_TESTS') != '1',
    reason='Set BOUNDEDQR_RUN_GPU_TESTS=1 to run tiny CUDA integration tests',
)


@pytest.fixture(scope='module', autouse=True)
def cuda_device():
    from boundedqr.gpu_pdhg import _prepare_windows_import_paths
    _prepare_windows_import_paths()
    cp = pytest.importorskip('cupy')
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f'CUDA device unavailable: {error}')
    if available < 1:
        pytest.skip('CUDA device unavailable')
    yield
    cp.get_default_memory_pool().free_all_blocks()


def fixture_arrays(n=127, p=4, draws=5):
    rng = np.random.default_rng(72015)
    x = np.column_stack((np.ones(n), rng.normal(size=(n, p-1))))
    y = x @ np.linspace(-.4, .8, p) + rng.normal(size=n)
    w = rng.normal(size=(p, draws)) * .5
    return x, y, w


def finite_loss(x, y, tau, w, beta):
    residual = y - x @ beta
    pseudo = len(y) * np.max(np.abs(y)) - (w / tau) @ beta
    return (np.maximum(tau * residual, (tau-1) * residual).sum() +
            max(tau * pseudo, (tau-1) * pseudo))


@pytest.mark.parametrize('tau', [.1, .5, .9])
@pytest.mark.parametrize('audit_backend', ['cpu', 'gpu'])
def test_gpu_matches_fixed_cpu_problems_in_nondivisible_blocks(tau, audit_backend):
    x, y, w = fixture_arrays()
    cpu, cdiag = solve_perturbations(x, y, tau, w, np.zeros(x.shape[1]),
                                    threads=1, batch_size=2)
    gpu, gdiag = solve_perturbations(x, y, tau, w, np.zeros(x.shape[1]),
                                    backend='gpu', audit_backend=audit_backend,
                                    threads=1, batch_size=2)
    assert cdiag['all_checked'] and gdiag['all_checked']
    np.testing.assert_allclose(gpu, cpu, atol=2e-8, rtol=2e-8)
    assert [b['actual_batch_size'] for b in gdiag['repair_blocks']] == [2, 2, 1]
    assert [f['replicate'] for f in gdiag['fits']] == list(range(w.shape[1]))
    for block in gdiag['repair_blocks']:
        assert block['acceptance_dtype'] == 'float64'
        assert all(product['dtype'] == 'float64' for product in block['products'])
    for j, fit in enumerate(gdiag['fits']):
        assert abs(finite_loss(x, y, tau, w[:, j], gpu[j]) -
                   finite_loss(x, y, tau, w[:, j], cpu[j])) < 1e-7
        assert np.isfinite(fit['radius_certificate']['radius'])
        assert fit['stationarity_coordinates'] == 'column-normalized'


@pytest.mark.parametrize('tau', [.1, .5, .9])
def test_gpu_retains_active_finite_pseudo_observation(tau):
    x, y, w = fixture_arrays(n=101, p=3, draws=3)
    w *= 20000.
    cpu, _ = solve_perturbations(x, y, tau, w, np.zeros(3), threads=1)
    gpu, diag = solve_perturbations(x, y, tau, w, np.zeros(3), backend='gpu',
                                   audit_backend='gpu', batch_size=2, threads=1)
    assert diag['all_checked']
    np.testing.assert_allclose(gpu, cpu, atol=2e-7, rtol=2e-7)
    for fit in diag['fits']:
        assert abs(fit['pseudo_residual']) < 2e-5
        # An ideal-tilt-only implementation would force the pseudo dual to tau.
        assert abs(fit['pseudo_dual']-tau) > 1e-3
        assert np.isfinite(fit['radius_certificate']['radius'])


def test_gpu_mixed_column_units_map_coefficients_and_radii_back():
    x, y, w = fixture_arrays(n=109, p=3, draws=3)
    reference, _ = solve_perturbations(x, y, .5, w, np.zeros(3), threads=1)
    scales = np.ldexp(np.ones(3), [-44, 33, 12])
    physical, diag = solve_perturbations(x*scales, y, .5, w*scales[:, None], np.zeros(3),
                                        backend='gpu', audit_backend='cpu',
                                        batch_size=2, threads=1)
    np.testing.assert_allclose(physical*scales, reference, atol=2e-8, rtol=2e-8)
    assert diag['all_checked']
    normalization = diag['column_normalization']
    assert normalization['exact_fp64_roundtrip_checked']
    assert normalization['output_coefficient_and_radius_units'] == 'original'
    for fit in diag['fits']:
        certificate = fit['radius_certificate']
        assert np.isfinite(certificate['radius'])
        expected = np.ldexp(certificate['normalized_radius'],
                            certificate['radius_multiplier_power2'])
        assert certificate['radius'] >= expected
        assert certificate['radius_coordinates'] == 'original Euclidean coefficient units'


def test_gpu_scalar_and_batched_repair_agree():
    x, y, w = fixture_arrays(n=103, p=3, draws=3)
    scalar, sdiag = solve_perturbations(x, y, .5, w, np.zeros(3), backend='gpu',
                                       repair='scalar', threads=1, batch_size=2)
    batched, bdiag = solve_perturbations(x, y, .5, w, np.zeros(3), backend='gpu',
                                        repair='batched', threads=1, batch_size=2)
    assert sdiag['all_checked'] and bdiag['all_checked']
    np.testing.assert_allclose(scalar, batched, atol=2e-8, rtol=2e-8)
    assert all(np.isfinite(f['radius_certificate']['radius']) for f in sdiag['fits'])


def test_auditor_abs_cache_is_bitwise_exact_with_one_host_design_transfer(monkeypatch):
    import cupy as cp
    from boundedqr.batched_radius import RadiusAuditor

    tiny, normal, largest = np.nextafter(0., 1.), np.finfo(float).tiny, np.finfo(float).max
    edge = np.array([0., -0., tiny, -tiny, 2*tiny, -2*tiny, normal, -normal,
                     np.nextafter(normal, 0.), -np.nextafter(normal, 0.),
                     1., -1., .1, -.9, largest, -largest])
    random_bits = np.random.default_rng(215).integers(0, 2**64, 496, dtype=np.uint64)
    random_values = random_bits.view(np.float64)
    random_values[~np.isfinite(random_values)] = 0.
    x = np.r_[edge, random_values].reshape(256, 2)
    expected_x = x.copy()
    expected_abs = np.abs(x)
    host_transfers = []
    original = cp.asarray

    def observe_transfer(value, *args, **kwargs):
        if isinstance(value, np.ndarray):
            host_transfers.append((value.shape, value.nbytes))
        return original(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', observe_transfer)
        auditor = RadiusAuditor(x, np.zeros(len(x)), .5, backend='gpu')
    # Validate the bounded-upload contract that addresses the observed failing
    # allocation path, not just numeric equality after two duplicate uploads.
    assert host_transfers == [(x.shape, x.nbytes)]
    x[:] = 3.  # Caller mutation must not alter either private snapshot/cache.
    np.testing.assert_array_equal(cp.asnumpy(auditor.device_x).view(np.uint64),
                                  expected_x.view(np.uint64))
    np.testing.assert_array_equal(cp.asnumpy(auditor.device_abs_x).view(np.uint64),
                                  expected_abs.view(np.uint64))
    np.testing.assert_array_equal(auditor.abs_x.view(np.uint64), expected_abs.view(np.uint64))
    assert auditor.device_abs_x.dtype == cp.float64
    assert auditor.diagnostics['persistent_device_array_bytes'] == 2*expected_x.nbytes
