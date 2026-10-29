"""Common test-only outer IRLS; no production estimator or inference change."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from econhdfe_planner import BudgetedAbsorber

@pytest.mark.parametrize('backend',['auto','fereg','within','compressed_within'])
def test_budgeted_ppml_outer_reference(backend):
    r=np.random.default_rng(19883);n=600
    g=r.integers(0,10,size=(n,3),dtype=np.uint32)
    x=r.normal(size=(n,2))
    d=np.column_stack([x]+[np.eye(10)[c] for c in g.T])
    y=np.exp(.2+x@np.array([.13,-.11])+.1*r.normal(size=n))
    def fit(s):
        eta=np.full(n,np.log(y.mean()))
        for step in range(40):
            mu=np.exp(eta);z=eta+(y-mu)/mu
            if s is None:
                coef=np.linalg.lstsq(np.sqrt(mu)[:,None]*d,np.sqrt(mu)*z,rcond=1e-13)[0]
                new=d@coef;beta=coef[:2]
            else:
                s.update_weights(mu)
                q=s.residualize(np.column_stack([z,x]),tol=1e-11)
                beta=np.linalg.solve(q[:,1:].T@(mu[:,None]*q[:,1:]),q[:,1:].T@(mu*q[:,0]))
                new=z-q[:,0]+q[:,1:]@beta
            if np.max(np.abs(new-eta))<1e-10:return beta,np.exp(new)
            eta=new
        raise AssertionError('IRLS did not converge')
    with BudgetedAbsorber(g,backend=backend,expected_calls=12,workload='reweight') as s:
        beta,mu=fit(s)
    br,mr=fit(None)
    assert_allclose(beta,br,atol=2e-8)
    assert_allclose(mu,mr,atol=2e-8)
    assert np.max(np.abs(d.T@(y-mu)))/n<1e-8
