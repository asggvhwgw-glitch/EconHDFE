"""Unified CGM repair uses frequency-expanded original-regressor stdev coordinates in every storage path."""
import numpy as np
import pytest

from econhdfe import PPMLConfig, IVPPMLConfig, ExecutionConfig
from econhdfe.compute.block_design import compile_block_design
from econhdfe.hdfe.plan import FEPlan


def _fixture(family):
    rng = np.random.default_rng(801 if family == 'ppml' else 906)
    n = 324
    g = np.r_[np.repeat(np.arange(40), 8), 40, 41, 42, 42]
    h = np.r_[np.tile(np.arange(8), 40), 0, 1, 2, 3]
    x, z, noise = rng.normal(size=(3, n))
    d = .9*z + .2*noise
    offset = .15*rng.normal(size=n)
    y = rng.poisson(np.exp(.8 + .12*x + .16*d + .01*(g % 7) + offset)).astype(float)
    y[np.arange(40)*8] += 1
    y[-4:] = [3, 2, 0, 0]
    fw = rng.integers(1, 5, n)
    fw[-4:] = [4, 1, 2, 3]
    # Two independent FE components, with cluster ids shared across them.
    # Both dense and certified block storage see the same four coefficients.
    X = np.zeros((2*n, 4))
    X[:n, 0], X[:n, 2] = x, d
    X[n:, 1], X[n:, 3] = 3*x, .4*d
    return (np.tile(y, 2), X, np.r_[g, g+43], np.r_[h, h+8],
            np.tile(g, 2), np.tile(h, 2), np.tile(fw, 2), np.tile(offset, 2))


def _fit(family, block, standardize, expanded):
    y, X, g, h, cg, ch, fw, offset = _fixture(family)
    if expanded:
        ix = np.repeat(np.arange(len(y)), fw)
        y, X, g, h, cg, ch, offset = (a[ix] for a in (y, X, g, h, cg, ch, offset))
        fw = np.ones(len(y))
    plan = FEPlan.from_arrays([g, h])
    config_type = PPMLConfig if family == 'ppml' else IVPPMLConfig
    kw = dict(offset=offset, true_w=fw, weight_kind='fweight' if not expanded else 'none',
              vce='cluster', clusters=[cg, ch],
              config=config_type(singleton_mode='unified', engine='optimized', standardize=standardize,
                                 tolerance=1e-11, target_inner_tol=1e-12, dof_method='exact'),
              execution_config=ExecutionConfig(threads=2, memory_budget_mb=512))
    def design(A):
        return compile_block_design(A, groups=plan.groups)[0] if block else A
    if family == 'ppml':
        if block:
            from econhdfe.models.ppml.heterogeneous import fit_block_arrays as fit
        else:
            from econhdfe.models.ppml.estimator import fit_arrays as fit
        return fit(y, design(X), plan, names=('x0', 'x1', 'd0', 'd1'), **kw)
    if block:
        from econhdfe.models.ppml_iv.heterogeneous import fit_block_arrays as fit
    else:
        from econhdfe.models.ppml_iv.estimator import fit_arrays as fit
    # Self-instrumentation gives a known first stage, allowing an independent
    # IV sandwich oracle without reusing the implementation's projection.
    return fit(y, design(X[:, :2]), design(X[:, 2:]), design(X[:, 2:]), plan,
               exog_names=('x0', 'x1'), endog_names=('d0', 'd1'), instrument_names=('d0', 'd1'), **kw)


def _oracle(result, family):
    y, X, g, h, cg, ch, fw, _ = _fixture(family)
    mask = np.ones(len(y), bool)
    mask[[321, 322, 323, 645, 646, 647]] = False
    np.testing.assert_array_equal(result.sample_mask, mask)
    y, X, g, h, cg, ch, fw = (a[mask] for a in (y, X, g, h, cg, ch, fw))
    mu = result.mu[mask]
    D = np.column_stack([g == v for v in np.unique(g)] + [h == v for v in np.unique(h)])
    sw = np.sqrt(fw * mu)
    Xdm = X - D @ np.linalg.lstsq(D * sw[:, None], X * sw[:, None], rcond=None)[0]
    bread = np.linalg.inv(Xdm.T @ ((fw * mu)[:, None] * Xdm))
    scores = Xdm * (fw * (y - mu))[:, None]
    meat = np.zeros((4, 4))
    for sign, codes in ((1, cg), (1, ch), (-1, cg*8+ch)):
        levels = np.unique(codes)
        sums = np.vstack([scores[codes == v].sum(axis=0) for v in levels])
        correction = len(levels)/(len(levels)-1) if family == 'ivppml' else 1
        meat += sign * correction * (sums.T @ sums)
    raw = bread @ meat @ bread
    if family == 'ppml':
        assert result.df_absorbed == 0  # Both FEs nested in cluster dimensions.
        # Ordinary PPML's default is upstream's asymptotic factor, independent
        # of absorbed/nested FE counts. Residual corrections remain opt-in.
        raw *= 8/7
    # Unified mode: the stdev of the frequency-expanded original regressors.
    n = fw.sum()
    s = np.sqrt(((fw[:, None] * X**2).sum(0) - (fw @ X)**2 / n) / (n - 1))
    std = (raw + raw.T)/2 * np.outer(s, s)
    eig, vec = np.linalg.eigh(std)
    assert eig[0] < -1e-7 * np.max(eig)  # Exercise repair, not an already positive covariance.
    return ((vec * np.maximum(eig, 0)) @ vec.T) / np.outer(s, s)


@pytest.mark.parametrize('family', ['ppml', 'ivppml'])
@pytest.mark.parametrize('block', [False, True], ids=['dense', 'block'])
def test_multiway_cgm_matches_stdev_coordinate_oracle_under_standardization_and_replication(family, block):
    base = _fit(family, block, False, False)
    expected = _oracle(base, family)
    np.testing.assert_allclose(base.vcov[:4, :4], expected, rtol=2e-7, atol=2e-10)
    for expanded in (False, True):
        got = _fit(family, block, True, expanded)
        assert got.converged and got.nobs == base.nobs
        np.testing.assert_allclose(got.params, base.params, rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(got.vcov[:4, :4], expected, rtol=2e-7, atol=2e-10)
