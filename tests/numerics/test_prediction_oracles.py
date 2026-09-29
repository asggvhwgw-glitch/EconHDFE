"""Independent dummy oracle for prediction levels and beta covariance."""
import numpy as np
import pandas as pd
import pytest
from econhdfe import olshdfe


@pytest.mark.parametrize('weighted', [False, True])
def test_new_rows_and_beta_standard_error_match_dense_dummy_wls(weighted):
    rng = np.random.default_rng(992)
    n = 500
    g = np.arange(n) % 10
    t = np.arange(n) // 10 % 5
    X = rng.normal(size=(n, 2))
    D = np.column_stack([np.eye(10)[g], np.eye(5)[t, 1:]])
    A = np.column_stack([X, D])
    y = A @ rng.normal(size=A.shape[1]) + rng.normal(size=n)
    weights = rng.uniform(.5, 1.5, size=n) if weighted else np.ones(n)
    sw = np.sqrt(weights)
    b = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    e = y - A @ b
    sigma2 = np.dot(weights * e, e) / (n - A.shape[1])
    V = np.linalg.inv((A * sw[:, None]).T @ (A * sw[:, None])) * sigma2
    df = pd.DataFrame({'y': y, 'x': X[:, 0], 'w': X[:, 1], 'g': g, 't': t})
    fit = olshdfe(df, y='y', x=['x', 'w'], absorb=['g', 't'], vce='iid',
                  weights=weights if weighted else None, save_fe=True, tol=1e-11)
    new = df.iloc[:23].copy()
    new[['x', 'w']] = rng.normal(size=(len(new), 2))
    new['g'] = rng.integers(0, 10, len(new))
    new['t'] = rng.integers(0, 5, len(new))
    raw = new[['x', 'w']].to_numpy()
    newD = np.column_stack([np.eye(10)[new.g], np.eye(5)[new.t, 1:]])
    expected = np.column_stack([raw, newD]) @ b
    se_beta = np.sqrt(np.einsum('ij,jk,ik->i', raw, V[:2, :2], raw))
    np.testing.assert_allclose(fit.predict(new, chunk_size=7), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fit.predict(new, kind='stdp', chunk_size=3), se_beta, rtol=1e-9, atol=1e-9)
