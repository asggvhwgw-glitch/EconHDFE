import numpy as np
import pytest
from pyreghdfe.encoding import factorize_interaction
from pyreghdfe.kernels import hac_meat, driscoll_kraay_meat
from pyreghdfe.vcov import score_covariance


def _meat(scores, codes):
    g = int(np.max(codes)) + 1
    sums = np.column_stack([
        np.bincount(codes, weights=scores[:, j], minlength=g)
        for j in range(scores.shape[1])
    ])
    return sums.T @ sums


def test_three_way_cluster_meat_matches_inclusion_exclusion():
    rng = np.random.default_rng(301)
    n = 3000
    scores = rng.normal(size=(n, 3))
    cs = [rng.integers(0, 40, n).astype(np.int32),
          rng.integers(0, 25, n).astype(np.int32),
          rng.integers(0, 15, n).astype(np.int32)]
    got = score_covariance(scores, kind="cluster", clusters=cs, small_sample=False)
    expected = np.zeros((3, 3))
    for i in range(3):
        expected += _meat(scores, cs[i])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        code, _ = factorize_interaction([cs[i], cs[j]])
        expected -= _meat(scores, code)
    code, _ = factorize_interaction(cs)
    expected += _meat(scores, code)
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-9)


def test_hac_bartlett_matches_direct_lag_formula():
    rng = np.random.default_rng(302)
    s = rng.normal(size=(500, 2))
    got, bw = hac_meat(s, bandwidth=3, kernel="bartlett")
    expected = s.T @ s
    for lag in (1, 2):
        cross = s[lag:].T @ s[:-lag]
        expected += (1 - lag / 3) * (cross + cross.T)
    assert bw == 3
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_dkraay_aggregates_scores_by_time():
    rng = np.random.default_rng(303)
    units, periods = 30, 20
    time = np.tile(np.arange(periods), units)
    s = rng.normal(size=(units * periods, 2))
    got, bw, t = driscoll_kraay_meat(s, time, bandwidth=3)
    sums = np.zeros((periods, 2))
    for tt in range(periods):
        sums[tt] = s[time == tt].sum(axis=0)
    expected = sums.T @ sums
    for lag in (1, 2):
        cross = sums[lag:].T @ sums[:-lag]
        expected += (1 - lag / 3) * (cross + cross.T)
    assert (bw, t) == (3, periods)
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_all_ivreg2_kernel_weights_and_spectral_horizon():
    from pyreghdfe.kernels import kernel_weight, hac_meat
    assert kernel_weight(1, 4, "thann") == pytest.approx(0.5 + 0.5*np.cos(np.pi/4))
    assert kernel_weight(1, 4, "thamm") == pytest.approx(0.54 + 0.46*np.cos(np.pi/4))
    assert np.isfinite(kernel_weight(1, 4, "qs"))
    assert kernel_weight(1, 4, "daniell") == pytest.approx(np.sin(np.pi/4)/(np.pi/4))
    assert np.isfinite(kernel_weight(1, 4, "tent"))
    # Spectral kernels retain nonzero lags beyond bandwidth; lag-window
    # Bartlett does not.  A score process engineered to correlate only at
    # lag 5 therefore distinguishes the horizons when bw=3.
    s = np.zeros((12, 1))
    s[[0, 5], 0] = 1.0
    bart, _ = hac_meat(s, bandwidth=3, kernel="bartlett")
    qs, _ = hac_meat(s, bandwidth=3, kernel="qs")
    assert not np.allclose(bart, qs)
