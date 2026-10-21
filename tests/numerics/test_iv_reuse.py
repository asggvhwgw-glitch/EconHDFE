"""Numerical oracles and execution boundaries for call-local IV reuse."""
import numpy as np
import pytest
from unittest.mock import patch
from econhdfe.compute.wls import weighted_arrays
from econhdfe.iv.solve import weighted_2sls
from econhdfe.models.linear_iv import solve, diagnostics, estimators
from econhdfe.compute.vcov import _cluster_kron_meat, cluster_subset_meats_xe
from econhdfe.errors import InferenceError


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
    with patch("econhdfe.compute.wls.weighted_arrays",side_effect=AssertionError("duplicate weighting")):
        got,qr=solve._solve_2sls_roles(y,X,Z,weights=w,n_exog=4,weighted=prepared)
    assert qr is not None and qr.R.shape[0] <= Z.shape[1]+E.shape[1]+1
    for name in expected.__dataclass_fields__:
        np.testing.assert_allclose(getattr(got,name),getattr(expected,name),rtol=1e-10,atol=1e-11)
    for a,b in zip(prepared,before):
        if a is not None:np.testing.assert_array_equal(a,b)


def test_role_qr_fallback_and_diagnostic_projection_guard():
    y,C,E,I=design()
    I[:,-1]=I[:,0]+1e-10*I[:,-1]
    X,Z=np.column_stack([C,E]),np.column_stack([C,I])
    with patch.object(solve,"equilibrated_lstsq",wraps=solve.equilibrated_lstsq) as fallback:
        _,qr=solve._solve_2sls_roles(y,X,Z,weights=None,n_exog=4,weighted=weighted_arrays(y,X,Z))
    assert qr is None and fallback.call_count==1
    with patch.object(diagnostics,"_residualize_small",wraps=diagnostics._residualize_small) as projection:
        diagnostics._IVDiagnosticWorkspace(E,C,I,_qr=qr)
    assert projection.call_count==1


@pytest.mark.parametrize("weighted", [False, True])
def test_sensitive_role_solve_preserves_qr_layout_without_reweighting(weighted):
    y, C, E, I = design()
    I[:, -1] = I[:, 0] + 1e-10 * I[:, -1]
    X, Z = np.column_stack([C, E]), np.column_stack([C, I])
    w = 1. + np.arange(len(y)) % 3 if weighted else None
    prepared = weighted_arrays(y, X, Z, w)
    with patch.object(solve, "_role_qr", wraps=solve._role_qr) as qr_call, patch(
            "econhdfe.compute.wls.weighted_arrays", side_effect=AssertionError("duplicate weighting")):
        got, qr = solve._solve_2sls_roles(y, X, Z, weights=w, n_exog=4, weighted=prepared)
    assert qr is None and qr_call.call_count == 2
    assert "yw" not in qr_call.call_args_list[-1].kwargs
    xhat = prepared[1].copy()
    xhat[:, 4:] = prepared[2] @ got.first_stage[:, 4:]
    beta, bread, rank = solve.equilibrated_lstsq(xhat, prepared[0])
    np.testing.assert_array_equal(got.beta, beta)
    np.testing.assert_array_equal(got.bread, bread)
    assert got.rank == rank


@pytest.mark.parametrize("near_perfect",[False,True])
def test_control_projection_reuse_and_cancellation_fallback(near_perfect):
    y,C,E,I=design()
    if near_perfect:E=I[:,:2]+1e-10*E
    X,Z=np.column_stack([C,E]),np.column_stack([C,I])
    _,qr=solve._solve_2sls_roles(y,X,Z,weights=None,n_exog=4,weighted=weighted_arrays(y,X,Z))
    expected=diagnostics._IVDiagnosticWorkspace(E,C,I)
    with patch.object(diagnostics,"_residualize_small",wraps=diagnostics._residualize_small) as projection:
        got=diagnostics._IVDiagnosticWorkspace(E,C,I,_qr=qr)
    assert projection.call_count==int(near_perfect)
    np.testing.assert_allclose(got.Ep,expected.Ep,rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(got.Zp,expected.Zp,rtol=1e-10,atol=1e-11)


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


def test_raw_shared_cluster_meat_matches_score_covariance():
    y,C,E,I=design();X=np.column_stack([C,E]);Z=np.column_stack([C,I])
    result=weighted_2sls(y,X,Z)
    groups=np.arange(len(y))%23
    scores=Z*result.residuals[:,None]
    meat=explicit_meat(scores,groups)
    expected=diagnostics.overid_test(result.residuals,Z,X=X,kind="cluster",clusters=[groups])
    with patch.object(diagnostics,"_cluster_meat_xe",side_effect=AssertionError("duplicate aggregation")):
        got=diagnostics.overid_test(result.residuals,Z,X=X,kind="cluster",clusters=[groups],
                                   _xz=X.T@Z,_meat=meat)
    assert got["df"]==expected["df"]
    np.testing.assert_allclose([got["stat"],got["pvalue"]],[expected["stat"],expected["pvalue"]],rtol=1e-10,atol=1e-11)


def test_residual_reused_and_basic_skips_shea_but_retains_kp():
    y,C,E,I=design();saved=[]
    original=estimators._solve_2sls_roles
    def record(*args,**kw):
        result,qr=original(*args,**kw);saved.append(result.residuals);return result,qr
    with patch.object(estimators,"_solve_2sls_roles",record), patch.object(
            diagnostics,"_shea_from_moments",side_effect=AssertionError("advanced disabled")):
        got=estimators.fit_iv_kclass(y,C,E,I,advanced_statistics=False)
    assert got[2] is saved[0]
    assert "kleibergen_paap" in got[6] and "cragg_donald_f" not in got[6]


@pytest.mark.parametrize("weighted", [False, True])
def test_shared_cluster_vcov_preserves_symmetry(weighted):
    y, C, E, I = design()
    w = 1. + np.arange(len(y)) % 3 if weighted else None
    def check(callback, beta, covariance, scale, **kwargs):
        np.testing.assert_array_equal(covariance, covariance.T)
    with patch.object(estimators, "estimation_ready", check):
        estimators.fit_iv_kclass(y, C, E, I, weights=w, vce="cluster",
                                clusters=[np.arange(len(y)) % 23])


def test_streamed_scores_reject_one_cluster():
    _,_,E,I=design()
    with pytest.raises(InferenceError,match="at least two clusters"):
        _cluster_kron_meat(E,I,[np.zeros(len(E),dtype=int)])


def test_group_tiled_kron_meat_does_not_expand_full_scores(monkeypatch):
    from econhdfe.compute import vcov
    _, _, E, I = design(300)
    c1 = np.arange(len(E)) % 31
    c2 = np.arange(len(E)) % 23
    scores = (E[:, :, None] * I[:, None, :]).reshape(len(E), -1)
    joint = np.unique(np.column_stack([c1, c2]), axis=0, return_inverse=True)[1]
    expected = explicit_meat(scores, c1) + explicit_meat(scores, c2) - explicit_meat(scores, joint)
    monkeypatch.setattr(vcov, "_KRON_GROUP_BYTES", 5 * scores.shape[1] * 8)
    original_empty = np.empty
    def bounded(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and shape[1] == scores.shape[1]:
            assert shape[0] <= 5
        return original_empty(shape, *args, **kwargs)
    with patch.object(vcov.np, "empty", bounded):
        got = vcov._cluster_kron_meat(E, I, [c1, c2])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-10)


def test_near_collinear_overid_retains_original_cross_product_order():
    rng=np.random.default_rng(172);n=1201
    C=rng.normal(size=(n,5));I=rng.normal(size=(n,4))
    E=I[:,:2]+.4*rng.normal(size=(n,2))
    y=C.sum(axis=1)+E.sum(axis=1)+rng.normal(size=n)
    C[:,-1]=C[:,0]+1e-5*C[:,-1]
    original=estimators.overid_test
    def checked(*args,**kw):
        assert kw["_xz"] is None and kw["_meat"] is None
        return original(*args,**kw)
    with patch.object(estimators,"overid_test",checked):
        result=estimators.fit_iv_kclass(y,C,E,I,vce="cluster",clusters=[np.arange(n)%23])
    cs,es,ins,*_=estimators._equilibrate_iv_roles(C,E,I)
    expected=original(result[2],np.column_stack([cs,ins]),X=np.column_stack([cs,es]),
                      kind="cluster",clusters=[np.arange(n)%23])
    assert result[6]["overidentification"]==expected
