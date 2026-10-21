"""Numerical regressions for role-aware solving and shared cluster aggregates."""
import numpy as np
import pytest
from econhdfe.compute.wls import weighted_arrays
from econhdfe.iv.solve import weighted_2sls
from econhdfe.models.linear_iv import solve, diagnostics, estimators
from econhdfe.compute.vcov import _cluster_kron_meat, cluster_subset_meats_xe


def design(n=503):
    rng=np.random.default_rng(121)
    C=rng.normal(size=(n,4));I=rng.normal(size=(n,4))
    E=I[:,:2]+rng.normal(size=(n,2));y=C.sum(axis=1)+E.sum(axis=1)+rng.normal(size=n)
    return y,C,E,I


@pytest.mark.parametrize("weighted",[False,True])
def test_role_solve_matches_generic_and_preserves_inputs(weighted):
    y,C,E,I=design();X=np.column_stack([C,E]);Z=np.column_stack([C,I])
    w=1.+np.arange(len(y))%3 if weighted else None
    prepared=weighted_arrays(y,X,Z,w)
    before=[None if a is None else a.copy() for a in prepared]
    expected=weighted_2sls(y,X,Z,weights=w)
    got, _ = solve._solve_2sls_roles(y, X, Z, weights=w, n_exog=4, weighted=prepared)
    for name in expected.__dataclass_fields__:
        np.testing.assert_allclose(getattr(got,name),getattr(expected,name),rtol=1e-10,atol=1e-11)
    for a,b in zip(prepared,before):
        if a is not None:np.testing.assert_array_equal(a,b)


def explicit_meat(scores,codes):
    sums=np.zeros((int(codes.max())+1,scores.shape[1]))
    np.add.at(sums,codes,scores)
    return sums.T@sums


@pytest.mark.parametrize("singleton",[False,True])
def test_streamed_cluster_kron_and_singleton_intersections(singleton):
    y,C,E,I=design(300)
    c1=np.arange(len(y))%20
    c2=np.arange(len(y))//20 if singleton else np.arange(len(y))%15
    codes=np.unique(np.column_stack([c1,c2]),axis=0,return_inverse=True)[1]
    scores=(E[:,:,None]*I[:,None,:]).reshape(len(y),-1)
    expected=explicit_meat(scores,c1)+explicit_meat(scores,c2)-explicit_meat(scores,codes)
    got=_cluster_kron_meat(E,I,[c1,c2])
    np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-10)
    pieces=list(cluster_subset_meats_xe(C,y,[c1,c2]))
    explicit=explicit_meat(C*y[:,None],codes)
    np.testing.assert_allclose(pieces[-1][1],explicit,rtol=1e-12,atol=1e-10)


def test_near_collinear_overid_matches_direct_moments():
    rng=np.random.default_rng(172);n=1201
    C=rng.normal(size=(n,5));I=rng.normal(size=(n,4))
    E=I[:,:2]+.4*rng.normal(size=(n,2))
    y=C.sum(axis=1)+E.sum(axis=1)+rng.normal(size=n)
    C[:,-1]=C[:,0]+1e-5*C[:,-1]
    result = estimators.fit_iv_kclass(y, C, E, I, vce="cluster", clusters=[np.arange(n)%23])
    cs,es,ins,*_=estimators._equilibrate_iv_roles(C,E,I)
    expected=diagnostics.overid_test(result[2],np.column_stack([cs,ins]),X=np.column_stack([cs,es]),
                      kind="cluster",clusters=[np.arange(n)%23])
    assert result[6]["overidentification"]==expected
