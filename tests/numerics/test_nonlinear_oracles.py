"""Independent nonlinear moment and covariance references.
Explicit dummy scipy.root calculations do not use production absorption/IRLS."""

from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import root
from econhdfe import IVPPMLConfig, PPMLConfig, ivppmlhdfe
from econhdfe.errors import InferenceError, SpecificationError
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml_iv.heterogeneous import fit_block_arrays


def fixture(*, n=480, overidentified=False):
    rng = np.random.default_rng(61402)
    c = rng.normal(size=(n, 1))
    z = rng.normal(size=(n, 2 if overidentified else 1))
    e = .9*z[:, :1] + .2*c + rng.normal(scale=.4, size=(n, 1))
    g = np.arange(n) % 8
    off = .12 * rng.normal(size=n)
    y = rng.poisson(np.exp(.5 + .12*c[:, 0] + .24*e[:, 0] + .1*g/8 + off))
    return dict(y=y, exog=c, endog=e, instruments=z, g=g, offset=off)


def as_block(a, g):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    blocks = []
    for j in (0, 1):
        rows = np.flatnonzero(g % 2 == j)
        blocks.append(DenseDesignBlock(j, rows, np.arange(a.shape[1]), a[rows].copy()))
    return BlockDesign(len(g), a.shape[1], tuple(blocks))


def fit(data, *, path='dense', engine='replica', standardize=False, kind='robust',
        clusters=None, weights=None, weight_type=None, min_ok=2, **controls):
    cfg = IVPPMLConfig(separation=(), tolerance=1e-11, target_inner_tol=1e-12,
                       max_iter=300, min_ok=min_ok, standardize=standardize,
                       engine=engine, **controls)
    cfg.validate()
    names = dict(exog_names=('c',), endog_names=('e',),
                 instrument_names=tuple(f'z{i}' for i in range(data['instruments'].shape[1])))
    n = len(data['y'])
    if path == 'dense':
        return ivppmlhdfe(data['y'], exog=data['exog'], endog=data['endog'],
                         instruments=data['instruments'], absorb=[data['g']],
                         offset=data['offset'], weights=weights, weight_type=weight_type,
                         vce=kind, clusters=clusters, config=cfg, **names)
    return fit_block_arrays(data['y'], as_block(data['exog'], data['g']),
                            as_block(data['endog'], data['g']),
                            as_block(data['instruments'], data['g']),
                            FEPlan.from_arrays([data['g']]), offset=data['offset'],
                            true_w=np.ones(n) if weights is None else weights,
                            weight_kind=weight_type or 'none', vce=kind,
                            clusters=clusters, config=cfg, **names)


def reference(data, w, *, overidentified=False, fweight=False, clusters=None):
    """Solve explicit moment equations and compute the frozen-IRLS sandwich.

    Overidentification targets projected score, not zero for each raw moment.
    """
    D = np.eye(8)[data['g']]
    T = np.column_stack([data['exog'], data['endog'], D])
    Q = np.column_stack([data['exog'], data['instruments'], D])
    y, off = data['y'], data['offset']
    def moments(theta):
        mu = np.exp(T @ theta + off)
        u = w * (y - mu)
        if not overidentified:
            return Q.T @ u / len(y)
        omega = w*mu
        gamma = np.linalg.solve(Q.T @ (omega[:, None]*Q), Q.T @ (omega[:, None]*T))
        return gamma.T @ Q.T @ u / len(y)
    def jac(theta):
        mu = np.exp(T @ theta + off)
        return -Q.T @ ((w*mu)[:, None]*T) / len(y)
    start = np.r_[0., 0., np.full(8, np.log(np.average(y, weights=w)))]
    out = root(moments, start, jac=None if overidentified else jac, tol=1e-11)
    assert np.max(np.abs(moments(out.x))) < 2e-10, out.message
    mu = np.exp(T @ out.x + off)
    omega = w*mu
    gamma = np.linalg.solve(Q.T @ (omega[:, None]*Q), Q.T @ (omega[:, None]*T))
    H = Q @ gamma
    bread = np.linalg.inv(H.T @ (omega[:, None]*T))
    scalar = w*(y-mu)
    if clusters is None:
        if fweight:
            scalar = scalar / np.sqrt(w)
        S = H * scalar[:, None]
        neff = float(w.sum() if fweight else len(y))
        meat = S.T @ S * neff/(neff-1)
    else:
        _, ids = np.unique(clusters, return_inverse=True)
        S = np.zeros((ids.max()+1, T.shape[1]))
        np.add.at(S, ids, H*scalar[:, None])
        meat = S.T @ S * len(S)/(len(S)-1)
    V = bread @ meat @ bread.T
    return out.x, mu, V, moments, jac, T, Q


@pytest.mark.parametrize('path,weight_type,overidentified', [
    ('dense', None, False),
    ('block', 'pweight', True),
    ('dense', 'fweight', True),
])
def test_ivppml_explicit_nonlinear_reference(path, weight_type, overidentified):
    d = fixture(overidentified=overidentified)
    n = len(d['y'])
    w = np.ones(n) if weight_type is None else (1 + np.arange(n)%3).astype(float)
    oracle = reference(d, w, overidentified=overidentified, fweight=weight_type=='fweight')
    r = fit(d, path=path, weights=w if weight_type else None, weight_type=weight_type)
    assert_allclose(r.coef[:2], oracle[0][:2], rtol=2e-7, atol=2e-8)
    assert_allclose(r.mu, oracle[1], rtol=2e-7, atol=2e-8)
    assert_allclose(r.vcov[:2, :2], oracle[2][:2, :2], rtol=3e-6, atol=2e-8)
    assert np.max(np.abs(oracle[3](np.r_[r.coef[:2], oracle[0][2:]]))) < 2e-7


def test_ivppml_reference_cluster_and_jacobian():
    engine = 'optimized'
    d = fixture()
    w = 1. + np.arange(len(d['y']))%3
    oracle = reference(d, w, clusters=d['g'])
    r = fit(d, engine=engine, kind='cluster', clusters=[d['g']], weights=w, weight_type='pweight')
    assert_allclose(r.vcov[:2, :2], oracle[2][:2, :2], rtol=2e-6, atol=1e-8)
    theta = oracle[0] + .03
    J = oracle[4](theta)
    h = 1e-5
    eye = np.eye(len(theta))
    Jnum = np.column_stack([(oracle[3](theta+h*v)-oracle[3](theta-h*v))/(2*h) for v in eye])
    assert_allclose(J, Jnum, rtol=2e-8, atol=2e-9)


def test_ivppml_units_preserve_covariance():
    path,kind,role,scale = ('block', 'cluster', 'instruments', 1e-07)
    d = fixture()
    cls = [d['g']] if kind == 'cluster' else None
    r = fit(d, path=path, kind=kind, clusters=cls)
    ds = dict(d); ds[role] = d[role]*scale
    s = fit(ds, path=path, kind=kind, clusters=cls)
    units = np.array([scale if role=='exog' else 1, scale if role=='endog' else 1, 1.])
    assert_allclose(s.coef*units, r.coef, rtol=2e-6, atol=2e-8)
    assert_allclose(s.vcov*units[:, None]*units[None, :], r.vcov, rtol=2e-5, atol=1e-10)


def test_ivppml_rejects_every_degenerate_cluster_dimension():
    path,mixed = ('block', True)
    d = fixture()
    cls = [np.zeros(len(d['y']), dtype=int)]
    if mixed:
        cls.insert(0, d['g'])
    with pytest.raises(InferenceError):
        fit(d, path=path, kind='cluster', clusters=cls)


@pytest.mark.parametrize('weight_type', ['pweight', 'fweight'])
def test_ivppml_frequency_replication_and_probability_rescaling(weight_type):
    d = fixture(n=240); n=len(d['y']); w=(1+np.arange(n)%3).astype(float)
    r = fit(d, weights=w, weight_type=weight_type)
    if weight_type=='pweight':
        s=fit(d, weights=100*w, weight_type=weight_type)
    else:
        ids=np.repeat(np.arange(n),w.astype(int))
        dd={k:v[ids] for k,v in d.items()}
        s=fit(dd)
    assert r.nobs == (int(w.sum()) if weight_type == 'fweight' else n)
    assert r.diagnostics['n_rows'] == n
    assert_allclose(s.coef,r.coef,rtol=2e-7,atol=2e-8)
    assert_allclose(s.vcov,r.vcov,rtol=2e-6,atol=2e-9)


def test_iteration_controls_require_finite_integers():
    field,value = ('max_iter', float('nan'))
    with pytest.raises(SpecificationError):
        PPMLConfig(**{field:value}).validate()
