"""Independent estimator oracles: explicit dummy WLS/2SLS, GLM and covariance.
Historical audit provenance is recorded in the test migration document."""

from __future__ import annotations
import numpy as np
import pytest
import scipy.linalg as la
import statsmodels.api as sm
import econhdfe as e
from econhdfe.effects import recover_fixed_effects


def dummy(groups):
    return np.column_stack([np.eye(int(np.max(g))+1)[g] for g in groups])


def independent_basis(D):
    _, R, p = la.qr(D, mode='economic', pivoting=True)
    rank = np.count_nonzero(np.abs(np.diag(R)) > 1e-10)
    return D[:, p[:rank]], int(rank)


def sample(seed=20260914, n=600):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0,17,n), rng.integers(0,7,n)
    X = rng.normal(size=(n,2))
    y = X @ np.array([1.,-.4]) + rng.normal(size=17)[a] + rng.normal(size=7)[b] + rng.normal(size=n)
    cluster = np.arange(n)%13
    return rng, y, X, [a,b], cluster


def full_cov(U, y, w, beta, *, kind, n_eff, freq, clusters):
    bread = np.linalg.inv(U.T @ (w[:,None]*U))
    err = y-U@beta
    if kind == 'iid':
        return bread*(np.dot(w*err,err)/(n_eff-U.shape[1]))
    scores = U*(w*err)[:,None]
    if kind == 'robust':
        if freq: scores /= np.sqrt(w)[:,None]
        meat=scores.T@scores
        scale=n_eff/(n_eff-U.shape[1])
    else:
        unq,inv=np.unique(clusters,return_inverse=True)
        agg=np.zeros((len(unq),U.shape[1]));np.add.at(agg,inv,scores)
        meat=agg.T@agg
        scale=(n_eff-1)/(n_eff-U.shape[1])*len(unq)/(len(unq)-1)
    return scale*bread@meat@bread


@pytest.mark.parametrize('kind,weight_type', [
    ('iid', 'pweight'),
    ('robust', 'fweight'),
    ('cluster', 'aweight'),
])
def test_ols_against_explicit_dummy_wls(kind,weight_type):
    rng,y,X,groups,cluster=sample()
    raw = None if weight_type is None else (rng.integers(1,4,len(y)).astype(float) if weight_type=='fweight' else rng.uniform(.4,2.0,len(y)))
    w=np.ones(len(y)) if raw is None else raw.copy()
    if weight_type in ('aweight','pweight'):w*=len(y)/w.sum()
    actual_kind='robust' if weight_type=='pweight' and kind=='iid' else kind
    D,rank=independent_basis(dummy(groups));U=np.column_stack([X,D])
    ref=la.lstsq(np.sqrt(w)[:,None]*U,np.sqrt(w)*y,lapack_driver='gelsy')[0]
    n_eff=float(w.sum()) if weight_type=='fweight' else len(y)
    V=full_cov(U,y,w,ref,kind=actual_kind,n_eff=n_eff,freq=weight_type=='fweight',clusters=cluster)
    got=e.olshdfe(y=y,x=X,absorb=groups,weights=raw,weight_type=weight_type,vce=kind,
                  cluster=[cluster] if kind=='cluster' else None,dof_method='exact',drop_singletons=False,tol=1e-11)
    assert got.df_absorbed == rank
    np.testing.assert_allclose(got.params,ref[:2],rtol=2e-8,atol=2e-9)
    np.testing.assert_allclose(got.vcov,V[:2,:2],rtol=2e-7,atol=2e-10)


@pytest.mark.parametrize('kind', [
    'iid',
    'cluster',
])
def test_linear_iv_dense_oracle_and_frequency_replication(kind):
    rng,y,X,groups,cluster=sample(8237)
    I=rng.normal(size=(len(y),2));C=X[:,:1];E=(I@np.array([.7,.4])+X[:,1])[:,None]
    y=1.2*E[:,0]+.3*C[:,0]+.2*X[:,1]+y*.2
    raw=rng.integers(1,4,len(y)).astype(float)
    w=raw
    D,_=independent_basis(dummy(groups));U=np.column_stack([C,E,D]);Q=np.column_stack([C,I,D]);sw=np.sqrt(w)
    Uw=U*sw[:,None];Qw=Q*sw[:,None];yw=y*sw
    Qorth,_=la.qr(Qw,mode='economic');ref=la.lstsq(Qorth.T@Uw,Qorth.T@yw,lapack_driver='gelsy')[0]
    kw=dict(y=y,exog=C,endog=E,instruments=I,absorb=groups,vce=kind,dof_method='exact',drop_singletons=False,tol=1e-11)
    if kind=='cluster':kw['cluster']=[cluster]
    got=e.ivhdfe(**kw,weights=raw,weight_type='fweight')
    np.testing.assert_allclose(got.params,ref[:2],rtol=2e-8,atol=2e-9)
    ix=np.repeat(np.arange(len(y)),raw.astype(int))
    clone=e.ivhdfe(y=y[ix],exog=C[ix],endog=E[ix],instruments=I[ix],absorb=[g[ix] for g in groups],
                   vce=kind,cluster=[cluster[ix]] if kind=='cluster' else None,dof_method='exact',drop_singletons=False,tol=1e-11)
    np.testing.assert_allclose(got.vcov,clone.vcov,rtol=2e-7,atol=2e-10)


@pytest.mark.parametrize('engine,kind', [
    ('replica', 'robust'),
    ('optimized', 'cluster'),
])
def test_ppml_against_independent_glm(engine,kind):
    rng,_,X,groups,cluster=sample(6729)
    off=rng.normal(scale=.15,size=len(X));weights=rng.uniform(.7,1.4,len(X))
    eta=.2*X[:,0]-.15*X[:,1]+rng.normal(scale=.15,size=17)[groups[0]]+rng.normal(scale=.15,size=7)[groups[1]]+off
    y=rng.poisson(np.exp(eta)).astype(float)
    D,_=independent_basis(dummy(groups));U=np.column_stack([X,D])
    ref=sm.GLM(y,U,family=sm.families.Poisson(),offset=off,var_weights=weights).fit(tol=1e-12,maxiter=100)
    got=e.ppmlhdfe(y=y,x=X,absorb=groups,offset=off,weights=weights,vce=kind,
                   cluster=[cluster] if kind=='cluster' else None,
                   config=e.PPMLConfig(engine=engine,tolerance=1e-11,target_inner_tol=1e-12,dof_method='exact'))
    assert np.all(got.sample_mask)
    bread=np.linalg.inv(U.T@((weights*ref.mu)[:,None]*U))
    if kind=='iid':V=bread
    else:
        scores=U*(weights*(y-ref.mu))[:,None]
        if kind=='robust':meat=scores.T@scores;scale=len(y)/(len(y)-U.shape[1])
        else:
            unq,inv=np.unique(cluster,return_inverse=True);agg=np.zeros((len(unq),U.shape[1]));np.add.at(agg,inv,scores)
            meat=agg.T@agg;scale=(len(y)-1)/(len(y)-U.shape[1])*len(unq)/(len(unq)-1)
        V=scale*bread@meat@bread
    np.testing.assert_allclose(got.params,ref.params[:2],rtol=2e-6,atol=2e-8)
    np.testing.assert_allclose(got.mu,ref.mu,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(got.vcov,V[:2,:2],rtol=2e-5,atol=2e-8)


def scale_sample():
    rng=np.random.default_rng(20260914);n=600;g=np.arange(n)%20
    X=rng.normal(size=(n,2));y=X@[1.,.5]+rng.normal(size=20)[g]+rng.normal(size=n)
    return y,X,[g]


def test_regression_column_scaling_must_not_delete_independent_regressor():
    y,X,g=scale_sample()
    ref=e.olshdfe(y=y,x=X,absorb=g,vce='robust',drop_singletons=False)
    got=e.olshdfe(y=y,x=X*[1.,1e8],absorb=g,vce='robust',drop_singletons=False)
    assert got.names==ref.names,'Changing a measurement unit deleted an independent variable'
    np.testing.assert_allclose(got.params*[1.,1e8],ref.params,rtol=1e-7)


def test_regression_covariance_must_be_scale_equivariant():
    y,X,g=scale_sample()
    ref=e.olshdfe(y=y,x=X,absorb=g,vce='robust',drop_singletons=False)
    got=e.olshdfe(y=y,x=X*[1.,4e7],absorb=g,vce='robust',drop_singletons=False)
    np.testing.assert_allclose(got.stderr*[1.,4e7],ref.stderr,rtol=1e-6)


def test_regression_fe_recovery_must_reject_nonfinite_inputs():
    case = 'weights_inf'
    a=np.array([0,0,1,1,2,2]);b=np.array([0,1,0,1,0,1]);target=np.array([1.,2,2,3,3,4]);kw={}
    if case=='target_nan':target[0]=np.nan
    else:
        w=np.ones(6);w[0]=np.nan if case=='weights_nan' else np.inf;kw['weights']=w
    try:got=recover_fixed_effects(target,[a,b],**kw)
    except (e.InputError,ValueError):return
    pytest.fail(f'{case}: invalid input returned a result with converged={got.diagnostics.converged}')
