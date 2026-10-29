"""Test-only common outer IRLS: no production estimator integration or inference."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from econhdfe_dual import DualAbsorber


def run_irls(g, x, y, backend):
    s = DualAbsorber(g, backend=backend, allow_fallback=False)
    eta = np.full(len(y), np.log(y.mean()))
    beta = np.zeros(x.shape[1])
    history = []
    def loss(e):
        return float(np.sum(np.exp(e)-y*e))
    for t in range(80):
        mu = np.exp(eta)
        s.update_weights(mu)
        z = eta + (y-mu)/mu
        transformed = s.residualize(np.column_stack([z, x]), tol=1e-11)
        zt, xt = transformed[:, 0], transformed[:, 1:]
        b = np.linalg.solve(xt.T @ (mu[:, None]*xt), xt.T @ (mu*zt))
        candidate = z-(zt-xt@b)
        old_loss = loss(eta)
        step = 1.0
        while step > 2**-24:
            trial = eta + step*(candidate-eta)
            if loss(trial) <= old_loss + 1e-10*max(1, abs(old_loss)):
                break
            step *= .5
        eta = trial
        beta = beta + step*(b-beta)
        score = y-np.exp(eta)
        err = max(abs(float(np.mean(score))), float(np.max(np.abs(x.T@score)))/len(y))
        for col in g.T:
            sums = np.bincount(col, weights=score)
            mass = np.bincount(col)
            good = mass > 0
            err = max(err, float(np.max(np.abs(sums[good]/mass[good]))))
        history.append(err)
        if err < 2e-9:
            return beta, np.exp(eta), s.diagnostics, history
    raise AssertionError(f'IRLS did not converge: {history}')


@pytest.mark.parametrize('backend', ['within', 'fereg', 'auto'])
def test_common_irls_score_and_dense_dummy_parity(backend):
    rng = np.random.default_rng(133)
    n = 1200
    g = np.column_stack([rng.integers(l, size=n) for l in [40, 15, 8]]).astype(np.uint32)
    x = rng.normal(size=(n, 2))
    eta = .4 + x@np.array([.16, -.21])
    for col in g.T:
        eta += rng.normal(scale=.2, size=int(col.max())+1)[col]
    y = np.exp(eta + .15*rng.normal(size=n) - .15**2/2)
    b, mu, diagnostics, history = run_irls(g, x, y, backend)
    # Independent full dummy Newton/WLS oracle; singular gauge handled by SVD.
    d = np.column_stack([x] + [np.eye(int(c.max())+1)[c] for c in g.T])
    a = np.zeros(d.shape[1])
    e = np.full(n, np.log(y.mean()))
    for _ in range(40):
        m = np.exp(e)
        z = e + (y-m)/m
        a = np.linalg.lstsq(np.sqrt(m)[:, None]*d, np.sqrt(m)*z, rcond=1e-13)[0]
        new = d@a
        if np.max(np.abs(new-e)) < 1e-11:
            e = new
            break
        e = new
    assert_allclose(b, a[:2], atol=2e-8, rtol=2e-8)
    assert_allclose(mu, np.exp(e), atol=2e-7, rtol=2e-8)
    assert max(history[-1:]) < 2e-9
    assert diagnostics['weight_generation'] >= 2
    assert diagnostics['kernel_builds'] == 1
