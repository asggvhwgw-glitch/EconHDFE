"""Independent numerical references for the IV-Poisson outer rewrite."""
import math
import numpy as np
import pytest
from scipy.optimize import root
from econhdfe import ivppmlhdfe, IVPPMLConfig, ExecutionConfig
from econhdfe.iv.step import solve_iv_step
from econhdfe.iv.solve import weighted_2sls


def config(**kw):
    opts = dict(outer_solver="moment", separation=(), engine="optimized",
                tolerance=1e-10, target_inner_tol=1e-11, max_iter=100)
    opts.update(kw)
    return IVPPMLConfig(**opts)


def fit(y, e, z, *, c=None, groups=None, **kw):
    return ivppmlhdfe(y=y, exog=c, endog=e, instruments=z, absorb=groups,
                      config=kw.pop('config', config()),
                      execution_config=ExecutionConfig(threads=2), **kw)


@pytest.mark.parametrize('l', [2, 5, 12])
@pytest.mark.parametrize('weighted', [False, True])
def test_stats_step_against_independent_tall_solver(l, weighted):
    rng = np.random.default_rng(100+l)
    n, k = 1003, 2
    Z = rng.normal(size=(n, l)); X = Z[:, :k] + .2*rng.normal(size=(n, k))
    y = X @ [.4, -.2] + rng.normal(size=n)
    w = np.exp(rng.normal(size=n)) if weighted else np.ones(n)
    got = solve_iv_step(y, X, Z, w, chunk_rows=71)
    want = weighted_2sls(y, X, Z, weights=w)
    assert got.backend == 'sufficient_statistics'
    for key in ('beta', 'bread', 'first_stage', 'residuals'):
        np.testing.assert_allclose(getattr(got, key), getattr(want, key), rtol=1e-10, atol=1e-12)


def test_stats_step_scale_and_weight_invariance():
    rng=np.random.default_rng(27); Z=rng.normal(size=(1001,3))
    X=Z[:,:2]+rng.normal(size=(1001,2));y=rng.normal(size=1001);w=np.exp(rng.normal(size=1001))
    a=solve_iv_step(y,X,Z,w)
    sx=np.array([1e-8,1e8]);sz=np.array([1e7,1e-7,1e4])
    b=solve_iv_step(y,X*sx,Z*sz,w*1e4)
    np.testing.assert_allclose(b.beta*sx,a.beta,rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(b.bread*sx[:,None]*sx[None,:]*1e4,a.bread,rtol=1e-9,atol=1e-12)


def test_stats_step_falls_back_on_nearly_duplicate_instruments():
    rng=np.random.default_rng(11);z=rng.normal(size=900)
    Z=np.column_stack([z,z+1e-7*rng.normal(size=900)])
    X=(z+.1*rng.normal(size=900))[:,None];y=rng.normal(size=900)
    a=solve_iv_step(y,X,Z,np.ones(900));b=weighted_2sls(y,X,Z,weights=np.ones(900))
    assert a.backend=='stable_fallback'
    np.testing.assert_allclose(a.beta,b.beta,rtol=1e-11,atol=1e-12)


def test_deviance_is_not_an_iv_merit_function():
    y=np.array([3.,1.,2.]);x=np.array([-1.,0.,1.]);z=np.array([0.,-1.,1.])
    got=fit(y,x,z)
    exact_beta=math.log((7.+math.sqrt(69.))/10.)
    np.testing.assert_allclose(got.params[0],exact_beta,rtol=0,atol=1e-9)
    mu0=np.full(3,2.)
    def loss(mu): return float(np.sum(mu-y*np.log(mu)))
    assert loss(got.mu)>loss(mu0)
    assert max(abs(np.sum(y-got.mu)),abs(z@(y-got.mu)))<1e-9
    assert got.diagnostics['outer']['moment_error']<1e-10
    for h in got.diagnostics['outer']['history']:
        assert h['merit_after'] <= h['merit_before'] + 1e-4*h['step']*h['directional_derivative']
        assert h['forcing'] <= .25


@pytest.mark.parametrize('over', [False, True])
@pytest.mark.parametrize('q', [1, 2, 3])
def test_dense_dummy_equation_reference(over,q):
    rng=np.random.default_rng(204+q+10*over);n=220
    groups=[rng.integers(0,3,n) for _ in range(q)]
    D=np.column_stack([np.ones(n)]+[(g==j).astype(float) for g in groups for j in range(1,3)])
    c=rng.normal(size=n);z=rng.normal(size=(n,2 if over else 1))
    e=.9*z[:,0]+.1*c+.25*rng.normal(size=n)
    X=np.column_stack([c,e]);Q=np.column_stack([D,c,z]);A=np.column_stack([D,X])
    w=np.exp(.25*rng.normal(size=n));off=.15*rng.normal(size=n)
    theta=np.r_[rng.normal(scale=.08,size=D.shape[1]),.1,.25]
    y=rng.poisson(np.exp(off+A@theta)).astype(float)+.05
    def eq(b):
        mu=np.exp(off+A@b);W=w*mu;r=w*(y-mu)
        if not over: return Q.T@r
        P=np.linalg.solve(Q.T@(W[:,None]*Q),Q.T@(W[:,None]*A))
        return P.T@(Q.T@r)
    ref=root(eq,np.zeros(A.shape[1]),tol=1e-11)
    assert np.max(abs(eq(ref.x)))<1e-7
    got=fit(y,e,z,c=c,groups=groups,weights=w,offset=off)
    np.testing.assert_allclose(got.params[:2],ref.x[-2:],rtol=2e-8,atol=2e-9)
    np.testing.assert_allclose(got.eta,off+A@ref.x,rtol=2e-8,atol=2e-8)
    assert got.diagnostics['outer']['moment_error']<1e-10
    # Independent full-dummy sandwich, not the package VCE implementation.
    mu=np.exp(off+A@ref.x);W=w*mu
    P=np.linalg.solve(Q.T@(W[:,None]*Q),Q.T@(W[:,None]*A))
    B=Q@P;J=B.T@(W[:,None]*A);Jinv=np.linalg.inv(J)
    scores=B*(w*(y-mu))[:,None]
    Vref=Jinv@(scores.T@scores*(n/(n-1)))@Jinv.T
    np.testing.assert_allclose(got.vcov[:2,:2],Vref[-2:,-2:],rtol=2e-7,atol=2e-9)
    assert got.diagnostics['outer']['exact_identification']==(not over)
    if over:
        assert got.diagnostics['outer']['raw_moment_error']>1e-5
        assert got.diagnostics['outer']['global_guarantee'].startswith('none_')


@pytest.mark.parametrize('fast',[False,True])
def test_fast_partial_and_stable_linear_solver_agree(fast):
    rng=np.random.default_rng(35);n=1500;g=[rng.integers(0,15,n),rng.integers(0,8,n)]
    z=rng.normal(size=(n,3));e=.7*z[:,0]+.4*rng.normal(size=n)
    y=rng.poisson(np.exp(.1+.3*e)).astype(float)
    a=fit(y,e,z,groups=g,config=config(fast_partial=fast))
    b=fit(y,e,z,groups=g,config=config(fast_partial=fast,outer_linear_solver='stable'))
    np.testing.assert_allclose(a.params,b.params,atol=2e-9,rtol=2e-8)
    np.testing.assert_allclose(a.vcov,b.vcov,atol=2e-10,rtol=2e-8)


def test_weight_replication_and_cluster_inference():
    rng=np.random.default_rng(36);n=300;g=rng.integers(0,8,n)
    z=rng.normal(size=n);e=z+.2*rng.normal(size=n);y=rng.poisson(np.exp(.1+.2*e))
    fw=rng.integers(1,4,n);idx=np.repeat(np.arange(n),fw)
    for vce in ['robust','cluster']:
        kwargs={'cluster':[g]} if vce=='cluster' else {}
        a=fit(y,e,z,groups=[g],weights=fw,weight_type='fw',vce=vce,
              config=config(singleton_mode='unified'),**kwargs)
        b=fit(y[idx],e[idx],z[idx],groups=[g[idx]],vce=vce,
              config=config(singleton_mode='unified'),**({'cluster':[g[idx]]} if vce=='cluster' else {}))
        np.testing.assert_allclose(a.params,b.params,atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(a.vcov,b.vcov,atol=1e-10,rtol=1e-8)


def test_warm_start_projected_to_changed_design():
    rng=np.random.default_rng(38);n=900;g=rng.integers(0,8,n)
    z=rng.normal(size=n);e=z+.2*rng.normal(size=n);c=rng.normal(size=n)
    y=rng.poisson(np.exp(.1+.2*e+.1*c))
    old=fit(y,e,z,c=c,groups=[g]);cold=fit(y,e,z,groups=[g])
    warm=fit(y,e,z,groups=[g],warm_start=old)
    np.testing.assert_allclose(warm.params,cold.params,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(warm.mu,cold.mu,rtol=1e-8,atol=1e-8)


def test_standardization_and_exposure_equivalence():
    rng=np.random.default_rng(40);n=2000;g=rng.integers(0,10,n)
    z=rng.normal(size=n);e=.7*z+.3*rng.normal(size=n);off=rng.normal(size=n)*.4
    y=rng.poisson(np.exp(off+.2+.2*e))
    a=fit(y,e*1e5,z*1e-5,groups=[g],offset=off,config=config(standardize=True))
    b=fit(y,e,z,groups=[g],exposure=np.exp(off))
    np.testing.assert_allclose(a.params[0]*1e5,b.params[0],rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(a.mu,b.mu,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(a.vcov[0,0]*1e10,b.vcov[0,0],rtol=1e-7)


def test_failure_is_not_silent_convergence():
    with pytest.raises(RuntimeError,match='exhausted'):
        fit(np.array([3.,1.,2.]),np.array([-1.,0.,1.]),np.array([0.,-1.,1.]),
            config=config(max_iter=1))
    # Constant-FE zero group has no finite solution when separation is disabled.
    with pytest.raises(RuntimeError):
        fit(np.array([0.,0.,1.,2.]),np.array([-1.,1.,-1.,1.]),np.array([-1.,1.,-1.,1.]),
            groups=[np.array([0,0,1,1])],config=config(max_iter=30))


@pytest.mark.parametrize('kwargs',[{'outer_solver':'bad'}, {'outer_linear_solver':'bad'},
    {'outer_chunk_rows':0},{'outer_forcing':1.},{'outer_armijo':0.},
    {'outer_gram_rcond':0.},{'outer_max_backtracks':-1},{'outer_record_history':3}])
def test_new_config_validation(kwargs):
    with pytest.raises(ValueError): config(**kwargs).validate()


def test_modern_default_does_not_use_ppml_regressor_separation():
    c=IVPPMLConfig(outer_solver='moment');assert c.separation==('fe',);c.validate()
    with pytest.raises(ValueError): IVPPMLConfig(outer_solver='moment',separation=('relu',)).validate()
    assert IVPPMLConfig().outer_solver=='legacy'


def test_large_finite_predictor_not_artificially_rejected():
    rng=np.random.default_rng(41);n=400
    z=rng.normal(size=n);e=.8*z+.2*rng.normal(size=n)
    y=np.exp(45.+.12*e)
    r=fit(y,e,z)
    np.testing.assert_allclose(r.params[0],.12,rtol=1e-8,atol=1e-9)
    assert np.min(r.eta)>30
    assert r.diagnostics['outer']['moment_error']<1e-10


def test_min_ok_is_a_confirmation_not_a_coefficient_update():
    y=np.array([3.,1.,2.]);e=np.array([-1.,0.,1.]);z=np.array([0.,-1.,1.])
    a=fit(y,e,z);b=fit(y,e,z,config=config(min_ok=2))
    assert b.iterations==a.iterations+1
    np.testing.assert_allclose(a.params,b.params,atol=1e-12)
    np.testing.assert_allclose(a.eta,b.eta,atol=1e-12)


def test_history_can_be_disabled():
    r=fit(np.array([3.,1.,2.]),np.array([-1.,0.,1.]),np.array([0.,-1.,1.]),
          config=config(outer_record_history=False))
    assert r.diagnostics['outer']['history']==[]
    assert r.diagnostics['outer']['final_state_consistent']


def test_block_kernel_rejects_uncertified_modern_routing():
    from econhdfe.models.ppml_iv.heterogeneous import _fit_iv_irls_block
    with pytest.raises(ValueError,match='BlockDesign'):
        _fit_iv_irls_block(None,None,None,None,None,None,None,config(),projector=None)


def test_bad_but_feasible_warm_start_uses_backtracking():
    from types import SimpleNamespace
    rng=np.random.default_rng(904);n=300
    z=rng.normal(size=n);e=.8*z+.3*rng.normal(size=n)
    y=np.exp(.2+.5*e)
    warm=SimpleNamespace(eta=-5.-e)
    # Deliberately rejected early trials can overflow the squared merit.
    with np.errstate(over='ignore'):
        r=fit(y,e,z,warm_start=warm)
    np.testing.assert_allclose(r.params[0],.5,atol=1e-8,rtol=1e-8)
    assert r.diagnostics['outer']['line_search_backtracks']>0
    for h in r.diagnostics['outer']['history']:
        assert h['merit_after'] <= h['merit_before']+1e-4*h['step']*h['directional_derivative']


def test_newton_direction_matches_full_dummy_linear_system():
    rng=np.random.default_rng(905);n=110
    g=rng.integers(0,4,n)
    D=np.column_stack([np.ones(n)]+[(g==j).astype(float) for j in range(1,4)])
    z=rng.normal(size=n);e=.7*z+.4*rng.normal(size=n)
    Q=np.column_stack([D,z]);A=np.column_stack([D,e])
    eta=A@rng.normal(scale=.1,size=A.shape[1]);mu=np.exp(eta)
    y=rng.poisson(mu);w0=np.exp(rng.normal(scale=.3,size=n));w=w0*mu
    v=(y-mu)/mu
    # Independent dense FWL, then the new IV linear kernel.
    M=lambda x:x-D@np.linalg.solve(D.T@(w[:,None]*D),D.T@(w[:,None]*x))
    vt=M(v[:,None])[:,0];xt=M(e[:,None]);zt=M(z[:,None])
    step=solve_iv_step(vt,xt,zt,w)
    direction=v-step.residuals
    exact=A@np.linalg.solve(Q.T@(w[:,None]*A),Q.T@(w0*(y-mu)))
    np.testing.assert_allclose(direction,exact,rtol=1e-10,atol=1e-11)
