"""Independent nonlinear moment and covariance references.
Explicit dummy scipy.root calculations do not use production absorption/IRLS."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import root

from econhdfe import IVPPMLConfig, PPMLConfig, ivppmlhdfe, ppmlhdfe
from econhdfe.errors import InferenceError, DivergenceError, SpecificationError
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml_iv.heterogeneous import fit_block_arrays
from econhdfe.models.poisson import working_state


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


@pytest.mark.parametrize('path', ['dense', 'block'])
@pytest.mark.parametrize('weight_type', [None, 'pweight', 'fweight'])
@pytest.mark.parametrize('overidentified', [False, True])
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


@pytest.mark.parametrize('engine', ['replica', 'optimized'])
def test_ivppml_reference_cluster_and_jacobian(engine):
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


@pytest.mark.parametrize('path,kind,role,scale', [('dense','robust','exog',4e7), ('dense','cluster','endog',4e7), ('dense','robust','instruments',1e-7), ('block','cluster','exog',4e7), ('block','robust','endog',4e7), ('block','cluster','instruments',1e-7)])
def test_ivppml_units_preserve_covariance(path, kind, role, scale):
    d = fixture()
    cls = [d['g']] if kind == 'cluster' else None
    r = fit(d, path=path, kind=kind, clusters=cls)
    ds = dict(d); ds[role] = d[role]*scale
    s = fit(ds, path=path, kind=kind, clusters=cls)
    units = np.array([scale if role=='exog' else 1, scale if role=='endog' else 1, 1.])
    assert_allclose(s.coef*units, r.coef, rtol=2e-6, atol=2e-8)
    assert_allclose(s.vcov*units[:, None]*units[None, :], r.vcov, rtol=2e-5, atol=1e-10)


@pytest.mark.parametrize('path', ['dense', 'block'])
@pytest.mark.parametrize('mixed', [False, True])
def test_ivppml_rejects_every_degenerate_cluster_dimension(path, mixed):
    d = fixture()
    cls = [np.zeros(len(d['y']), dtype=int)]
    if mixed:
        cls.insert(0, d['g'])
    with pytest.raises(InferenceError):
        fit(d, path=path, kind='cluster', clusters=cls)


@pytest.mark.parametrize('path', ['dense', 'block'])
def test_ivppml_first_stage_is_in_original_units(path):
    d = fixture(); d['exog'] *= 3; d['endog'] *= 4; d['instruments'] *= .07
    r = fit(d, path=path, standardize=False)
    s = fit(d, path=path, standardize=True)
    assert_allclose(s.diagnostics['first_stage'], r.diagnostics['first_stage'], rtol=2e-6, atol=2e-8)
    D = np.eye(8)[d['g']]; X = np.column_stack([d['exog'], d['endog']]); Q = np.column_stack([d['exog'], d['instruments']])
    omega = r.mu
    proj = np.linalg.solve(D.T @ (omega[:, None]*D), D.T @ (omega[:, None]*np.column_stack([X, Q])))
    within = np.column_stack([X, Q])-D@proj
    expected = np.linalg.lstsq(within[:, 2:]*np.sqrt(omega)[:, None], within[:, :2]*np.sqrt(omega)[:, None], rcond=None)[0]
    assert_allclose(s.diagnostics['first_stage'], expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('path', ['dense', 'block'])
@pytest.mark.parametrize('shift', [-40., 40.])
def test_ivppml_constant_offset_shift(path, shift):
    d = fixture(); r = fit(d, path=path)
    ds = dict(d); ds['offset'] = d['offset']+shift
    s = fit(ds, path=path)
    assert_allclose(s.coef[:2], r.coef[:2], rtol=3e-7, atol=2e-8)
    assert_allclose(s.mu, r.mu, rtol=3e-7, atol=2e-8)
    assert_allclose(s.coef[-1]+shift, r.coef[-1], rtol=3e-7, atol=2e-8)


@pytest.mark.parametrize('path', ['dense', 'block'])
def test_ivppml_large_coefficient_is_not_divergence_when_units_are_small(path):
    d = fixture(); r = fit(d, path=path, min_ok=12)
    ds = dict(d); ds['endog'] = d['endog']*1e-8
    s = fit(ds, path=path, min_ok=12)
    assert_allclose(s.coef*np.array([1., 1e-8, 1.]), r.coef, rtol=3e-6, atol=2e-8)
    assert_allclose(s.mu, r.mu, rtol=3e-6, atol=2e-8)


@pytest.mark.parametrize('path', ['dense', 'block'])
@pytest.mark.parametrize('kind', [None, 'model', 'iid', 'unadjusted', 'robust'])
def test_ivppml_reports_actual_variance_semantics(path, kind):
    r = fit(fixture(), path=path, kind=kind)
    assert r.vce == 'robust'
    assert r.model_stats()['vce'] == 'robust'


@pytest.mark.parametrize('path', ['dense', 'block'])
def test_ivppml_normalized_constant_does_not_report_a_confidence_interval(path):
    r = fit(fixture(), path=path)
    assert r.stderr[-1] == 0.  # retain the documented normalized storage convention
    row = r.coef_table().loc['_cons']
    assert np.isnan(row.ci_low) and np.isnan(row.ci_high)
    assert np.isnan(row.p_value) and row.stars == ''


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
    assert_allclose(s.coef,r.coef,rtol=2e-7,atol=2e-8)
    assert_allclose(s.vcov,r.vcov,rtol=2e-6,atol=2e-9)


@pytest.mark.parametrize('engine', ['replica', 'optimized'])
@pytest.mark.parametrize('kind', ['model', 'robust', 'cluster'])
def test_ppml_explicit_glm_with_offset_and_general_weights(engine, kind):
    import statsmodels.api as sm
    d=fixture(); w=(1+np.arange(len(d['y']))%3).astype(float)
    X=np.column_stack([d['exog'],d['endog']]); D=np.eye(8)[d['g']]; A=np.column_stack([X,D])
    reference_fit=sm.GLM(d['y'],A,family=sm.families.Poisson(),offset=d['offset'],freq_weights=w).fit(tol=1e-12,maxiter=100)
    omega=w*reference_fit.mu; bread=np.linalg.inv(A.T@(omega[:,None]*A))
    if kind=='model':
        V=bread
    elif kind=='robust':
        S=A*(w*(d['y']-reference_fit.mu))[:,None]
        V=bread@(S.T@S)@bread * len(w)/(len(w)-A.shape[1])
    else:
        S=np.zeros((8,A.shape[1]));np.add.at(S,d['g'],A*(w*(d['y']-reference_fit.mu))[:,None])
        V=bread@(S.T@S)@bread * ((len(w)-1)/(len(w)-X.shape[1]-1)) * 8/7
    # General positive PPML weights use squared scores, physical N, and the
    # package's documented finite-sample convention; there is no fweight API.
    r=ppmlhdfe(d['y'],X,absorb=[d['g']],weights=w,offset=d['offset'],vce=kind,
               clusters=[d['g']] if kind=='cluster' else None,
               config=PPMLConfig(engine=engine,separation=(),tolerance=1e-11,target_inner_tol=1e-12))
    assert_allclose(r.coef[:2],reference_fit.params[:2],rtol=2e-7,atol=2e-8)
    assert_allclose(r.vcov[:2,:2],V[:2,:2],rtol=3e-6,atol=2e-9)


def test_working_response_is_newton_linearization_with_offset():
    rng=np.random.default_rng(123);n=40;k=3
    A=rng.normal(size=(n,k));theta=rng.normal(scale=.1,size=k);off=rng.normal(scale=.2,size=n)
    eta=A@theta+off;mu=np.exp(eta);y=rng.poisson(mu);w=rng.uniform(.2,3,n)
    ww=np.empty(n);z=np.empty(n);working_state(y,mu,eta,off,w,ww,z)
    step=np.linalg.solve(A.T@(ww[:,None]*A),A.T@(ww*z))-theta
    newton=np.linalg.solve(A.T@((w*mu)[:,None]*A),A.T@(w*(y-mu)))
    assert_allclose(step,newton,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('field', ['max_iter', 'min_ok', 'simplex_max_iter', 'relu_max_iter'])
@pytest.mark.parametrize('value', [1.5, True, float('nan'), float('inf')])
def test_iteration_controls_require_finite_integers(field, value):
    with pytest.raises(SpecificationError):
        PPMLConfig(**{field:value}).validate()

@pytest.mark.parametrize('model', ['ppml', 'iv_dense', 'iv_block'])
@pytest.mark.parametrize('labels', ['shifted', 'strings', 'fractional'])
def test_cluster_relabeling_cannot_change_inference(model, labels):
    d=fixture();g=d['g']
    relabeled=g+100 if labels=='shifted' else (np.array([f'group-{i}' for i in g]) if labels=='strings' else g*.1-.7)
    def run(c):
        if model=='ppml':
            return ppmlhdfe(d['y'],np.column_stack([d['exog'],d['endog']]),absorb=[g],offset=d['offset'],
                            vce='cluster',clusters=[c],config=PPMLConfig(separation=(),tolerance=1e-11,target_inner_tol=1e-12))
        return fit(d,path='block' if model=='iv_block' else 'dense',kind='cluster',clusters=[c])
    r=run(g);s=run(relabeled)
    assert s.cluster_counts==r.cluster_counts==(8,)
    assert_allclose(s.coef,r.coef,rtol=1e-8,atol=1e-10)
    assert_allclose(s.vcov,r.vcov,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('model', ['ppml', 'iv_dense', 'iv_block'])
def test_constant_nonzero_cluster_label_is_still_one_cluster(model):
    d=fixture();g=np.full(len(d['y']),42)
    with pytest.raises(InferenceError):
        if model=='ppml':
            ppmlhdfe(d['y'],d['exog'],absorb=[d['g']],vce='cluster',clusters=[g],config=PPMLConfig(separation=()))
        else:
            fit(d,path='block' if model=='iv_block' else 'dense',kind='cluster',clusters=[g])
