"""Exact Bartlett HAC covariance with locally accumulated moving windows.

The returned meat is UNNORMALIZED, matching statsmodels S_hac_simple.
The rows must be consecutive equally spaced observations. No demeaning,
prewhitening, bandwidth selection beyond the legacy default, or sorting occurs.
"""
from __future__ import annotations

import operator
import numpy as np


def _scores(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise TypeError("complex scores are not supported")
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("x must be a nonempty one- or two-dimensional array")
    if not np.isfinite(a).all():
        raise ValueError("scores must be finite")
    return a


def _lag(n, nlags):
    if nlags is None:
        return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    if isinstance(nlags, (bool, np.bool_)):
        raise TypeError("nlags must be a nonnegative integer, not bool")
    lag = operator.index(nlags)
    if lag < 0:
        raise ValueError("nlags must be nonnegative")
    return lag


def _direct(x, lag):
    result = x.T @ x
    for h in range(1, min(lag, len(x) - 1) + 1):
        cross = x[h:].T @ x[:-h]
        result += (1.0 - h / (lag + 1)) * (cross + cross.T)
    return result


def _blocked(x, lag, target_rows=65536):
    """Window Gram using local forward/reverse sums, never global subtraction.

    Each width-b window crosses at most two b-row blocks. A forward sum in
    the right block plus a reverse sum in the left block gives the window.
    Processing block batches caps workspace at O((target_rows+b)*k + k²).
    """
    n, k = x.shape
    b = lag + 1
    if lag == 0:
        return x.T @ x
    if b >= n:
        left = np.cumsum(x, axis=0)[:-1]
        right = np.cumsum(x[::-1], axis=0)[:-1]
        total = np.sum(x, axis=0)
        result = (left.T @ left + right.T @ right) / b
        result += ((b - n + 1) / b) * np.outer(total, total)
        return result
    out_rows = n + b - 1
    out_blocks = (out_rows + b - 1) // b
    batch = max(1, target_rows // b)
    result = np.zeros((k, k))
    for start in range(0, out_blocks, batch):
        count = min(batch, out_blocks - start)
        # First block is a read-only halo from the preceding batch.
        a = np.zeros((count + 1, b, k))
        input_start = max(0, (start - 1) * b)
        input_end = min(n, (start + count) * b)
        offset = b if start == 0 else 0
        if input_end > input_start:
            a.reshape(-1, k)[offset:offset + input_end - input_start] = x[input_start:input_end]
        windows = np.cumsum(a[1:], axis=1)
        suffix = np.cumsum(a[:-1, ::-1], axis=1)[:, ::-1]
        windows[:, :-1] += suffix[:, 1:]
        windows = windows.reshape(-1, k)[:min(count * b, out_rows - start * b)]
        result += windows.T @ windows
    return result / b


def hac_meat(x, nlags=None, method="auto"):
    """Return the unnormalized Bartlett HAC score covariance.

    Parameters
    ----------
    x : array_like, (n,) or (n,k)
        Finite real scores, converted to float64 without centering.
    nlags : int or None
        Lag L has weight 1-L/(nlags+1). None uses statsmodels' legacy rule.
        Lags beyond n-1 have zero cross-products but retain this denominator.
    method : {'auto', 'direct', 'blocked', 'prefix', 'fft'}
        Auto keeps short lags and small, short-lag samples on the direct path.
        Other inputs use stable blocked sums.
        Prefix is the round-1 experimental comparator, sensitive to a large
        accumulated offset. FFT is the established Toeplitz comparator.
    """
    a = _scores(x)
    lag = _lag(len(a), nlags)
    if method not in {"auto", "direct", "blocked", "prefix", "fft"}:
        raise ValueError("unknown HAC method")
    if method == "auto":
        method = "direct" if lag <= 4 or (len(a) <= 2048 and lag <= 16) else "blocked"
    if method == "direct":
        return _direct(a, lag)
    if method == "blocked":
        return _blocked(a, lag)
    if method == "prefix":
        from ._hac_prefix import hac_prefix
        return hac_prefix(a, lag)
    from ._hac_fft import hac_fft
    return hac_fft(a, lag)


def cov_hac(results, nlags=None, use_correction=True, method="auto"):
    """HAC parameter covariance for OLS/WLS or explicit (scores, bread).

    Matches statsmodels cov_hac_simple for these supported inputs. Other
    fitted model families require explicitly supplied scores and bread.
    Returns a matrix; never changes the fitted result or package globals.
    """
    if isinstance(results, tuple) and len(results) == 2:
        x, bread = results
        x = _scores(x)
    else:
        from statsmodels.regression.linear_model import OLS, WLS
        model = getattr(results, "model", None)
        if not isinstance(model, (OLS, WLS)):
            raise TypeError("supported inputs are OLS/WLS results or (scores, bread)")
        x = _scores(model.wexog * np.asarray(results.wresid)[:, None])
        bread = results.normalized_cov_params
    bread = np.asarray(bread, dtype=np.float64)
    n, k = x.shape
    if bread.shape != (k, k) or not np.isfinite(bread).all():
        raise ValueError("bread must be a finite k by k matrix")
    if use_correction and n <= k:
        raise ValueError("small-sample correction requires n > k")
    result = bread @ hac_meat(x, nlags, method=method) @ bread.T
    if use_correction:
        result *= n / (n - k)
    return result
