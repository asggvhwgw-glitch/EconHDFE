"""Independent dense derivatives, empirical perturbations and invariance oracles."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import root
from econhdfe import ivppmlhdfe,IVPPMLConfig,ExecutionConfig
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml_iv.profile import FEProfiler
from econhdfe.models.ppml_iv.overidentified import (
    profile_geometry,inverse_sensitivity,estimating_equation_scores,
    overidentification_test,residualize_scaled,
)


def cfg(**kwargs):
    defaults=dict(outer_solver='profile',outer_inference='estimating_equation',
        outer_overid_test=True,engine='optimized',tolerance=1e-10,
        target_inner_tol=1e-11,max_iter=100,singleton_mode='unified')
    defaults.update(kwargs);return IVPPMLConfig(**defaults)


def data(seed=831,n=180,q=2,exact=False):
    rng=np.random.default_rng(seed)
    groups=[rng.integers(0,3 if j==0 else 2,n) for j in range(q)]
    D=np.column_stack([np.ones(n)]+[(g==j).astype(float) for g in groups
                                   for j in range(1,int(g.max())+1)])
    c=rng.normal(size=n);z=rng.normal(size=(n,1 if exact else 3))
    e=.8*z[:,0]+.15*c+.55*rng.normal(size=n)
    off=.1*rng.normal(size=n);w=np.exp(.2*rng.normal(size=n))
    # Explicitly misspecified over-ID moments, not just residual sampling noise.
    eta=.2+.2*e+.1*c+off+(0 if exact else .35*z[:,1])
    y=rng.poisson(np.exp(eta)).astype(float)+.05
    return dict(y=y,exog=c,endog=e,instruments=z,absorb=groups,offset=off,weights=w),D


def fit(d,**kwargs):
    return ivppmlhdfe(**d,config=cfg(**kwargs),execution_config=ExecutionConfig(threads=2),vce='robust')


def dense(d,D):
    X=np.column_stack([d['exog'],d['endog']]);Z=np.column_stack([d['exog'],d['instruments']])
    A=np.column_stack([D,X]);Q=np.column_stack([D,Z]);y=d['y'];w0=d['weights'];off=d['offset']
    def eq(theta,weights=w0):
        mu=np.exp(off+A@theta);w=weights*mu;r=weights*(y-mu)
        Pi=np.linalg.solve(Q.T@(w[:,None]*Q),Q.T@(w[:,None]*A))
        return Pi.T@(Q.T@r)
    ref=root(eq,np.zeros(A.shape[1]),tol=1e-11)
    assert np.max(np.abs(eq(ref.x)))<1e-7
    eta=off+A@ref.x;mu=np.exp(eta);w=w0*mu;wr=w0*(y-mu)
    P=D@np.linalg.solve(D.T@(w[:,None]*D),D.T*w)
    Xt=X-P@X;Zt=Z-P@Z
    H=Zt@np.linalg.solve(Zt.T@(w[:,None]*Zt),Zt.T@(w[:,None]*Xt))
    return SimpleNamespace(theta=ref.x,eq=eq,A=A,Q=Q,eta=eta,mu=mu,w=w,wr=wr,P=P,X=X,Z=Z,Xt=Xt,Zt=Zt,H=H)


@pytest.mark.parametrize('q',[1,2,3])
def test_profile_jacobian_against_reprofiled_finite_difference(q):
    d,D=data(seed=850+q,q=q);n=len(d['y']);w0=d['weights'];off=d['offset'];X=np.column_stack([d['exog'],d['endog']]);Z=np.column_stack([d['exog'],d['instruments']])
    beta=np.array([.12,-.04])
    def profile(b):
        alpha=root(lambda a:D.T@(w0*(d['y']-np.exp(off+X@b+D@a))),np.zeros(D.shape[1]),tol=1e-11).x
        mu=np.exp(off+X@b+D@alpha);w=w0*mu;wr=w0*(d['y']-mu)
        M=lambda B:B-D@np.linalg.solve(D.T@(w[:,None]*D),D.T@(w[:,None]*B))
        Xt,Zt=M(X),M(Z)
        H=Zt@np.linalg.solve(Zt.T@(w[:,None]*Zt),Zt.T@(w[:,None]*Xt))
        return H.T@wr,profile_geometry(Xt,Zt,H,w,wr)
    f,g=profile(beta);h=1e-5
    J=np.column_stack([(profile(beta+h*np.eye(2)[j])[0]-profile(beta-h*np.eye(2)[j])[0])/(2*h) for j in range(2)])
    np.testing.assert_allclose(-g.sensitivity,J,rtol=2e-7,atol=2e-7)
    assert np.linalg.norm(J+g.frozen_sensitivity)>1e-3


@pytest.mark.parametrize('mode',['moment','profile'])
@pytest.mark.parametrize('q',[1,2,3])
def test_empirical_influence_and_vcov_against_full_dummy_numeric_derivatives(mode,q):
    d,D=data(seed=900+q,n=110,q=q);s=dense(d,D);h=2e-5;p=len(s.theta);n=len(d['y'])
    # Full estimating function: re-estimate projection under each data perturbation.
    J=np.column_stack([(s.eq(s.theta+h*np.eye(p)[j])-s.eq(s.theta-h*np.eye(p)[j]))/(2*h) for j in range(p)])
    scores=np.column_stack([(s.eq(s.theta,d['weights']*np.exp(h*np.eye(n)[i]))-
                             s.eq(s.theta,d['weights']*np.exp(-h*np.eye(n)[i])))/(2*h) for i in range(n)])
    IF=-np.linalg.solve(J,scores)[-2:]
    V=IF@IF.T*n/(n-1)
    got=fit(d,outer_solver=mode)
    np.testing.assert_allclose(got.params[:2],s.theta[-2:],rtol=2e-7,atol=2e-9)
    np.testing.assert_allclose(got.vcov[:2,:2],V,rtol=3e-6,atol=2e-9)
    g=profile_geometry(s.Xt,s.Zt,s.H,s.w,s.wr)
    class Projection:
        def residualize(self,B,w,**kwargs):return B-s.P@B,None
    psi=estimating_equation_scores(s.Xt,s.H,g,s.w,s.wr,projector=Projection())
    inv,_=inverse_sensitivity(g.sensitivity)
    np.testing.assert_allclose(inv@psi.T,IF,rtol=3e-6,atol=2e-9)
    # A bread-only modification is generally NOT the empirical influence function.
    wrong=inv@(s.H*s.wr[:,None]).T
    assert np.max(np.abs(wrong-IF))>1e-6


def test_profile_accepts_only_actual_merit_decreases():
    d,_=data();r=fit(d)
    for h in r.diagnostics['outer']['history']:
        assert h['merit_after']<=h['merit_before']+1e-4*h['step']*h['directional_derivative']
    assert r.diagnostics['outer']['actual_profile_merit']


def test_overid_statistic_matches_optimal_linearized_residual_not_raw_objective():
    d,D=data();s=dense(d,D);r=fit(d);n=len(d['y'])
    Z=s.Zt;B=Z.T@(s.w[:,None]*s.Xt);m=Z.T@s.wr;psi=Z*s.wr[:,None];S=psi.T@psi*n/(n-1)
    Si=np.linalg.inv(S);delta=np.linalg.solve(B.T@Si@B,B.T@Si@m);e=m-B@delta
    want=e@Si@e
    got=r.diagnostics['overidentification']
    assert got['status']=='ok' and got['df']==2
    np.testing.assert_allclose(got['statistic'],want,rtol=1e-8,atol=1e-9)
    assert abs(m@Si@m-want)>1e-5


@pytest.mark.parametrize('vce',['robust','cluster'])
def test_frequency_replication_full_inference_and_overid(vce):
    d,_=data(n=180);rng=np.random.default_rng(18);fw=rng.integers(1,4,len(d['y']));ix=np.repeat(np.arange(len(fw)),fw)
    a={**d,'weights':fw,'weight_type':'fw'}
    b={k:([x[ix] for x in v] if k=='absorb' else v[ix]) for k,v in d.items() if k!='weights'}
    if vce=='cluster':a['cluster']=[d['absorb'][0]];b['cluster']=[d['absorb'][0][ix]]
    ra=ivppmlhdfe(**a,config=cfg(),execution_config=ExecutionConfig(threads=2),vce=vce)
    rb=ivppmlhdfe(**b,config=cfg(),execution_config=ExecutionConfig(threads=2),vce=vce)
    np.testing.assert_allclose(ra.params,rb.params,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(ra.vcov,rb.vcov,rtol=2e-7,atol=1e-10)
    np.testing.assert_allclose(ra.diagnostics['overidentification']['statistic'],rb.diagnostics['overidentification']['statistic'],rtol=2e-7,atol=1e-8)


def test_instrument_basis_and_regressor_unit_invariance():
    d,_=data(n=500);a=fit(d)
    transform=np.array([[2.,.4,-.1],[.2,.7,.2],[-.2,.2,1.3]])
    b=fit({**d,'instruments':d['instruments']@transform})
    np.testing.assert_allclose(a.params,b.params,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(a.vcov,b.vcov,rtol=2e-7,atol=1e-10)
    np.testing.assert_allclose(a.diagnostics['overidentification']['statistic'],b.diagnostics['overidentification']['statistic'],rtol=1e-8)
    c=fit({**d,'exog':d['exog']*1e4,'endog':d['endog']*1e-3},standardize=True)
    scale=np.array([1e4,1e-3,1.])
    np.testing.assert_allclose(c.params*scale,a.params,rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(c.vcov*scale[:,None]*scale[None,:],a.vcov,rtol=2e-7,atol=1e-9)


def test_exact_id_reduces_to_standard_sandwich_and_has_no_overid_pvalue():
    d,_=data(exact=True);a=fit(d);b=fit(d,outer_inference='standard')
    np.testing.assert_allclose(a.vcov,b.vcov,rtol=1e-8,atol=1e-10)
    oi=a.diagnostics['overidentification'];assert oi['status']=='exactly_identified' and oi['pvalue'] is None and oi['df']==0


def test_too_few_clusters_suppresses_overid_pvalue():
    d,D=data(n=150);s=dense(d,D)
    Z=np.column_stack([s.Zt,np.sin(s.Zt[:,1])])
    Z-=s.P@Z
    H=Z@np.linalg.solve(Z.T@(s.w[:,None]*Z),Z.T@(s.w[:,None]*s.Xt))
    g=profile_geometry(s.Xt,Z,H,s.w,s.wr)
    oi=overidentification_test(Z,g,s.wr,kind='cluster',clusters=[np.arange(len(Z))%2])
    assert oi['status']=='singular_or_indefinite_restriction_covariance'
    assert oi['pvalue'] is None and oi['df']==3


def test_bad_projection_is_not_silently_accepted():
    class Bad:
        def residualize(self,x,w,**kw):return x,SimpleNamespace(converged=False)
    with pytest.raises(RuntimeError,match='certification'):
        residualize_scaled(Bad(),np.ones((5,2)),np.ones(5),tol=1e-10)


def test_zero_and_tiny_rhs_projection_regression():
    d,D=data();w=d['weights'];rng=np.random.default_rng(7)
    B=np.column_stack([rng.normal(size=len(w))*1e-20,rng.normal(size=len(w))])
    P=D@np.linalg.solve(D.T@(w[:,None]*D),D.T*w)
    pr=FEPlan.from_arrays(d['absorb']).projector(engine='optimized')
    r,info=residualize_scaled(pr,B.copy(),w,tol=1e-11)
    np.testing.assert_allclose(r[:,1],(B-P@B)[:,1],rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('kw',[{'outer_inference':'bad'},{'outer_profile_max_iter':0},
 {'outer_overid_test':1},{'outer_solver':'legacy','outer_inference':'estimating_equation'},
 {'outer_solver':'legacy','outer_overid_test':True}])
def test_inference_config_validation(kw):
    with pytest.raises(ValueError):cfg(**kw).validate()


def test_profile_nonconvergence_and_warm_start():
    d,_=data(n=400);a=fit(d)
    b=ivppmlhdfe(**d,config=cfg(),warm_start=a,execution_config=ExecutionConfig(threads=2))
    np.testing.assert_allclose(a.params,b.params,rtol=1e-8,atol=1e-9)
    with pytest.raises(RuntimeError,match='exhausted'):fit(d,max_iter=1)

@pytest.mark.parametrize('multiway',[False,True])
def test_cluster_inference_against_independent_empirical_influence(multiway):
    d,D=data(seed=973,n=120,q=2);s=dense(d,D);h=2e-5;p=len(s.theta);n=len(d['y'])
    J=np.column_stack([(s.eq(s.theta+h*np.eye(p)[j])-s.eq(s.theta-h*np.eye(p)[j]))/(2*h) for j in range(p)])
    S=np.column_stack([(s.eq(s.theta,d['weights']*np.exp(h*np.eye(n)[i]))-
                        s.eq(s.theta,d['weights']*np.exp(-h*np.eye(n)[i])))/(2*h) for i in range(n)])
    IF=-np.linalg.solve(J,S)[-2:].T
    c1=np.arange(n)%12;c2=(np.arange(n)//5)%9
    def meat(c):
        _,codes=np.unique(c,return_inverse=True,axis=0);G=int(codes.max())+1
        agg=np.zeros((G,2));np.add.at(agg,codes,IF)
        return G/(G-1.)*(agg.T@agg)
    V=meat(c1)
    if multiway:
        V+=meat(c2)-meat(np.column_stack([c1,c2]));eig,U=np.linalg.eigh(V);V=(U*np.maximum(eig,0))@U.T
    got=ivppmlhdfe(**d,cluster=[c1,c2] if multiway else [c1],config=cfg(),execution_config=ExecutionConfig(threads=2),vce='cluster')
    np.testing.assert_allclose(got.vcov[:2,:2],V,rtol=4e-6,atol=2e-9)


def test_profile_finite_predictor_above_old_guard():
    rng=np.random.default_rng(871);n=300;z=rng.normal(size=(n,2));x=.8*z[:,0]+.2*rng.normal(size=n)
    y=np.exp(45.+.12*x)
    r=ivppmlhdfe(y=y,endog=x,instruments=z,config=cfg(),execution_config=ExecutionConfig(threads=2))
    np.testing.assert_allclose(r.params[0],.12,rtol=1e-8,atol=1e-9)
    assert np.min(r.eta)>30


def test_projection_exception_restores_rhs_units_for_next_trial():
    original=np.array([[1e-12,1e3],[2e-12,-2e3]])
    data=original.copy()
    class Raises:
        def residualize(self,x,w,**kw):raise RuntimeError('trial failed')
    with pytest.raises(RuntimeError,match='trial failed'):
        residualize_scaled(Raises(),data,np.ones(2),tol=1e-10)
    np.testing.assert_allclose(data,original,rtol=1e-15,atol=0)
