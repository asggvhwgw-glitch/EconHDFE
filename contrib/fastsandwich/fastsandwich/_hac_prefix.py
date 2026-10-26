"""Independent exact Bartlett HAC prototype using global prefix sums.

This is an experiment, not a universal statsmodels replacement. Inputs are
finite, real, one- or two-dimensional numeric arrays; accumulation is float64.
No demeaning or finite-sample correction is applied. Global prefix subtraction
can suffer cancellation on long, badly scaled inputs; the production version
should consider locally reset or compensated sums. Algebraic equality does not
promise bitwise agreement with the direct lagged-cross-product implementation.
"""

from __future__ import annotations

import operator

import numpy as np


def hac_prefix(x, nlags):
    """Return unnormalized Bartlett HAC meat, including every boundary window.

    The usual lag L weights are 1-lag/(L+1). Zero-extended moving sums
    z[t] = sum(x[t-j] for j in range(L+1)) satisfy S = z.T @ z / (L+1).
    For L >= N-1, identical all-sample windows are collapsed analytically so
    storage and runtime do not grow with an arbitrarily large requested lag.
    """
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

    nobs, nvar = array.shape
    width = lag + 1
    cumulative = np.empty((nobs + 1, nvar), dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(array, axis=0, out=cumulative[1:])

    if width >= nobs:
        # Partial left and right boundary windows each have N-1 rows. The
        # remaining width-N+1 windows contain the entire original series.
        left = cumulative[1:nobs]
        right = cumulative[-1] - cumulative[1:nobs]
        all_sample = cumulative[-1]
        result = (left.T @ left + right.T @ right) / width
        result += ((width - nobs + 1) / width) * np.outer(all_sample, all_sample)
        return result

    moving = np.empty((nobs + lag, nvar), dtype=np.float64)
    moving[:lag] = cumulative[1:width]
    moving[lag:nobs] = cumulative[width:] - cumulative[:-width]
    moving[nobs:] = cumulative[-1] - cumulative[nobs - lag:nobs]
    return (moving.T @ moving) / width


def verify_small_cases():
    """Compare tiny boundary cases to independent dense and released oracles."""
    import statsmodels
    from statsmodels.stats.sandwich_covariance import S_hac_simple

    generator = np.random.default_rng(93522)
    cases = 0
    worst_relative = 0.0
    for nobs in (1, 2, 3, 7, 19):
        for nvar in (1, 3):
            arrays = [
                generator.normal(size=(nobs, nvar)),
                np.ones((nobs, nvar)),
                generator.normal(size=(nobs, nvar * 2))[:, ::2],
            ]
            if nvar == 1:
                arrays.append(generator.normal(size=nobs))
            for array in arrays:
                for lag in sorted({0, 1, nobs - 1, nobs, 2 * nobs + 3}):
                    got = hac_prefix(array, lag)
                    baseline = S_hac_simple(array, nlags=lag)
                    matrix = array[:, None] if array.ndim == 1 else array
                    distance = np.abs(np.arange(nobs)[:, None] - np.arange(nobs))
                    weight = np.maximum(1.0 - distance / (lag + 1), 0.0)
                    dense = matrix.T @ weight @ matrix
                    np.testing.assert_allclose(got, baseline, rtol=2e-13, atol=2e-13)
                    np.testing.assert_allclose(got, dense, rtol=2e-13, atol=2e-13)
                    denominator = max(np.linalg.norm(baseline), np.finfo(float).tiny)
                    worst_relative = max(worst_relative, np.linalg.norm(got-baseline)/denominator)
                    cases += 1
    # L far exceeds N: verify the collapsed plateau without calling the
    # baseline's million-iteration Python loop.
    array = generator.normal(size=(7, 3))
    lag = 1_000_000
    distance = np.abs(np.arange(7)[:, None] - np.arange(7))
    dense = array.T @ (1.0 - distance / (lag + 1)) @ array
    np.testing.assert_allclose(hac_prefix(array, lag), dense, rtol=2e-13, atol=2e-13)
    return {"statsmodels": statsmodels.__version__, "ordinary_cases": cases,
            "huge_lag_case": "passed", "max_relative_frobenius_error": worst_relative}


if __name__ == "__main__":
    import json
    print(json.dumps(verify_small_cases(), indent=2))
