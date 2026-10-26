"""Independent FFT/Toeplitz comparator for Bartlett HAC meat.

Prior art: Heberle and Sattarhoff (2017), A Fast Algorithm for the Computation
of HAC Covariance Matrix Estimators, Econometrics 5(1), 9.
https://doi.org/10.3390/econometrics5010009

We implement a real-FFT circulant embedding independently, with no dense
Toeplitz matrix and one transform per score column, not per column pair.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def hac_fft(x, nlags):
    """Return unnormalized Bartlett HAC meat without centering the scores."""
    lag = operator.index(nlags)
    if lag < 0:
        raise ValueError("nlags must be nonnegative")
    array = np.asarray(x)
    if np.iscomplexobj(array):
        raise TypeError("complex scores are not supported")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("x must contain at least one row and one score column")
    if not np.isfinite(array).all():
        raise ValueError("x must contain finite values")
    if lag == 0:
        return array.T @ array

    nobs = len(array)
    # Embedding has distinct slots for +d and -d, d=1,...,N-1.
    transform_length = next_fast_len(2 * nobs - 1, real=True)
    column = np.zeros(transform_length, dtype=np.float64)
    column[0] = 1.0
    active_lags = min(lag, nobs - 1)
    if active_lags:
        weights = 1.0 - np.arange(1, active_lags + 1) / (lag + 1)
        column[1:active_lags + 1] = weights
        column[-active_lags:] = weights[::-1]
    spectrum = rfft(column, workers=1)
    transformed = rfft(array, n=transform_length, axis=0, workers=1)
    transformed *= spectrum[:, None]
    product = irfft(transformed, n=transform_length, axis=0, workers=1)[:nobs]
    meat = array.T @ product
    return (meat + meat.T) * 0.5


def verify_small_cases():
    import statsmodels
    from statsmodels.stats.sandwich_covariance import S_hac_simple

    generator = np.random.default_rng(33588)
    cases = 0
    max_relative = 0.0
    for nobs in (1, 2, 3, 7, 19, 33):
        for nvar in (1, 3):
            arrays = [generator.normal(size=(nobs, nvar)),
                      np.ones((nobs, nvar)),
                      generator.normal(size=(nobs, nvar * 2))[:, ::2]]
            if nvar == 1:
                arrays.append(generator.normal(size=nobs))
            for array in arrays:
                for lag in sorted({0, 1, nobs-1, nobs, 2*nobs+3}):
                    actual = hac_fft(array, lag)
                    expected = S_hac_simple(array, nlags=lag)
                    matrix = array[:, None] if array.ndim == 1 else array
                    distance = np.abs(np.arange(nobs)[:, None] - np.arange(nobs))
                    dense = matrix.T @ np.maximum(1-distance/(lag+1), 0) @ matrix
                    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
                    np.testing.assert_allclose(actual, dense, rtol=3e-13, atol=3e-13)
                    max_relative = max(max_relative, np.linalg.norm(actual-expected) /
                                       max(np.linalg.norm(expected), np.finfo(float).tiny))
                    cases += 1
    return {"statsmodels": statsmodels.__version__, "cases": cases,
            "max_relative_frobenius_error": max_relative}


if __name__ == "__main__":
    import json
    print(json.dumps(verify_small_cases(), indent=2))
