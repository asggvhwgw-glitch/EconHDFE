import numpy as np
import pandas as pd
from econhdfe import reghdfe, ivreghdfe


def test_parallel_wild_bootstrap_shape_and_reproducibility():
    from econhdfe import wild_bootstrap
    rng=np.random.default_rng(42)
    n=1800; f=rng.integers(0,60,n); t=rng.integers(0,9,n); c=rng.integers(0,30,n)
    X=rng.normal(size=(n,2)); y=X@np.array([.5,-.25])+rng.normal(size=60)[f]+rng.normal(size=9)[t]+rng.normal(size=n)
    r=reghdfe(None,y=y,x=X,absorb=[f,t],cluster=c,vce="cluster",keep_state=True)
    a=wild_bootstrap(r,reps=12,n_jobs=2,batch_size=3,seed=9)
    b=wild_bootstrap(r,reps=12,n_jobs=1,batch_size=3,seed=9)
    assert a.shape==(12,2)
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)


def test_heterogeneous_intercept_slope_matches_explicit_dummy():
    from econhdfe import FixedEffect
    rng = np.random.default_rng(101)
    n = 5000
    f = rng.integers(0, 90, n); t = rng.integers(0, 12, n)
    z = rng.normal(size=n); x = rng.normal(size=n)
    af = rng.normal(size=90); bf = rng.normal(scale=.4, size=90); at = rng.normal(size=12)
    y = 1.35*x + af[f] + bf[f]*z + at[t] + rng.normal(scale=.3, size=n)
    got = reghdfe(None, y=y, x=x, absorb=[FixedEffect(f, slopes=(z,), intercept=True), t], tol=1e-10)
    Df = pd.get_dummies(pd.Series(f).astype('category'), drop_first=False, dtype=float).to_numpy()
    Dt = pd.get_dummies(pd.Series(t).astype('category'), drop_first=False, dtype=float).to_numpy()
    A = np.column_stack([x, Df, Df*z[:,None], Dt])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    np.testing.assert_allclose(got.params, b[:1], atol=5e-8, rtol=5e-8)
    assert got.converged


def test_heterogeneous_slope_only_matches_explicit_dummy():
    from econhdfe import FixedEffect
    rng = np.random.default_rng(102)
    n = 3500
    f = rng.integers(0, 70, n); t = rng.integers(0, 10, n)
    z = rng.normal(size=n); x = rng.normal(size=n)
    bf = rng.normal(scale=.5, size=70); at = rng.normal(size=10)
    y = -.8*x + bf[f]*z + at[t] + rng.normal(scale=.2, size=n)
    got = reghdfe(None, y=y, x=x, absorb=[FixedEffect(f, slopes=(z,), intercept=False), t], tol=1e-10)
    Df = pd.get_dummies(pd.Series(f).astype('category'), drop_first=False, dtype=float).to_numpy()
    Dt = pd.get_dummies(pd.Series(t).astype('category'), drop_first=False, dtype=float).to_numpy()
    A = np.column_stack([x, Df*z[:,None], Dt])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    np.testing.assert_allclose(got.params, b[:1], atol=5e-8, rtol=5e-8)


def _explicit_within(A, groups):
    D = pd.get_dummies(pd.DataFrame({f'f{i}':g for i,g in enumerate(groups)}).astype('category'), drop_first=False, dtype=float).to_numpy()
    coef, *_ = np.linalg.lstsq(D, A, rcond=None)
    return A - D@coef


def test_liml_matches_direct_kclass_formula_after_explicit_absorption():
    rng = np.random.default_rng(111)
    n=2200
    f1=rng.integers(0,45,n); f2=rng.integers(0,9,n)
    z1=rng.normal(size=n); z2=rng.normal(size=n); w=rng.normal(size=n); v=rng.normal(size=n)
    x=.45*z1+.30*z2+.2*w+v
    y=1.6*x+.3*w+rng.normal(size=45)[f1]+rng.normal(size=9)[f2]+.6*v+rng.normal(size=n)
    got=ivreghdfe(None,y=y,exog=w,endog=x,instruments=np.column_stack([z1,z2]),absorb=[f1,f2],estimator='liml',vce='iid',tol=1e-10)
    A=_explicit_within(np.column_stack([y,w,x,z1,z2]),[f1,f2])
    yy=A[:,0]; C=A[:,1:2]; E=A[:,2:3]; I=A[:,3:]
    X=np.column_stack([C,E]); Z=np.column_stack([C,I]); ee=np.column_stack([yy,E])
    ez=ee-Z@(np.linalg.pinv(Z)@ee)
    ex1=ee-C@(np.linalg.pinv(C)@ee)
    vals,vecs=np.linalg.eigh(ez.T@ez)
    invroot=(vecs/np.sqrt(vals))@vecs.T
    kap=np.min(np.linalg.eigvalsh(invroot@(ex1.T@ex1)@invroot))
    pinvz=np.linalg.pinv(Z)
    P1=(X.T@X)*(1-kap)+kap*((X.T@Z)@(pinvz@X))
    P2=(X.T@yy)*(1-kap)+kap*((X.T@Z)@(pinvz@yy))
    bref=np.linalg.solve(P1,P2)
    np.testing.assert_allclose(got.params,bref,atol=2e-7,rtol=2e-7)
    assert abs(got.kappa-kap) < 2e-7
