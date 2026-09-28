"""Inference boundaries: units, rank, invalid references and solver termination."""
import importlib
from unittest.mock import patch
import numpy as np
import pytest
import econhdfe as e
from econhdfe.models.linear_iv.estimators import fit_iv_kclass, fit_iv_gmm2s
from econhdfe.compute.stable_linalg import equilibrated_lstsq
from econhdfe.reporting import inference_table

@pytest.mark.parametrize('estimator,scale,vce', [
    (model, 1e-10 if (i+j)%2 else 1e10, vce)
    for i, model in enumerate(("2sls", "liml", "kclass", "gmm2s"))
    for j, vce in enumerate(("iid", "robust", "cluster"))
])
def test_iv_role_scaling_preserves_beta_vcov_and_first_stage(estimator,scale,vce):
    rng=np.random.default_rng(2016);n=600
    C=rng.normal(size=(n,1));I=rng.normal(size=(n,2));E=(I@np.array([.8,.4])+rng.normal(size=n))[:,None]
    y=1.2*C[:,0]+1.5*E[:,0]+rng.normal(size=n)
    f=fit_iv_gmm2s if estimator=='gmm2s' else fit_iv_kclass
    kw={} if estimator=='gmm2s' else {'estimator':estimator}
    if estimator=='kclass':kw['kappa']=.98
    kw.update(vce=vce,clusters=[np.arange(n)%23] if vce=='cluster' else None,weights=rng.uniform(.5,2,n))
    base=f(y,C,E,I,**kw)
    # Distinct column unit changes, including opposite instrument units.
    xs=np.array([1/scale,scale]); zs=np.array([1/scale,scale,1/scale])
    got=f(y,C*xs[0],E*xs[1],I*zs[1:],**kw)
    np.testing.assert_allclose(got[0]*xs,base[0],rtol=2e-8,atol=2e-9)
    np.testing.assert_allclose(got[1]*xs[:,None]*xs[None,:],base[1],rtol=2e-7,atol=1e-10)
    np.testing.assert_allclose(got[2],base[2],atol=2e-8,rtol=1e-8)
    np.testing.assert_allclose(got[5]['coefficients']*zs[:,None]/xs[None,:],base[5]['coefficients'],atol=2e-8,rtol=1e-8)
    np.testing.assert_allclose(got[5]['fitted_endog']/xs[1],base[5]['fitted_endog'],atol=2e-8,rtol=1e-8)
    if estimator=='gmm2s':
        np.testing.assert_allclose(got[7]['first_step_params']*xs,base[7]['first_step_params'],atol=2e-8)
        np.testing.assert_allclose(got[7]['weight_matrix']*zs[:,None]*zs[None,:],base[7]['weight_matrix'],rtol=1e-7,atol=1e-8)

@pytest.mark.parametrize('df',[None,0,-1,np.nan,np.inf])
def test_invalid_t_reference_is_not_normal_fallback(df):
    table=inference_table(['x'],[1.],[.01],df=df,distribution='t')
    assert np.isnan(table['p_value'].iloc[0])
    assert table['stars'].iloc[0]==''

@pytest.mark.parametrize('stop',range(8))
def test_lsmr_stop_code_mapping_and_raw_diagnostics(stop):
    m=importlib.import_module('econhdfe.effects.recover'); details={}
    with patch.object(m,'lsmr',return_value=(np.zeros(2),stop,3,2.,1.,4.,1e9,0.)):
        args = ((np.array([0,0,1,1]),), np.ones(4), np.ones(4), 1e-10, 20)
        with_diagnostics = m._generic_lsmr(*args, diagnostics=details)
        without_diagnostics = m._generic_lsmr(*args)
        np.testing.assert_array_equal(with_diagnostics[0], without_diagnostics[0])
        assert with_diagnostics[1:] == without_diagnostics[1:]
        _, ok, _, _ = with_diagnostics
    assert ok == (stop in {0,1,2,4,5})
    assert details['stop_code']==stop and details['condition_estimate']==1e9
    assert details['stop_reason']

@pytest.mark.parametrize('chunk',[1,7,1000])
def test_stable_lstsq_common_beta_covariance_rank_under_extreme_units(chunk):
    rng=np.random.default_rng(29);X=rng.normal(size=(100,4));y=rng.normal(size=100)
    scale=np.array([1e-12,1e12,1e-7,1e7])
    b,V,r=equilibrated_lstsq(X*scale,y,chunk_rows=chunk)
    np.testing.assert_allclose(b*scale,np.linalg.lstsq(X,y,rcond=None)[0],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(V*scale[:,None]*scale[None,:],np.linalg.inv(X.T@X),atol=1e-12,rtol=1e-10)
    assert r==4

@pytest.mark.parametrize('small_sample',[False,True])
def test_one_cluster_invalid_even_without_small_sample_scale(small_sample):
    from econhdfe.compute.vcov import score_covariance
    from econhdfe.errors import InferenceError
    with pytest.raises(InferenceError):
        score_covariance(np.ones((5,2)),kind='cluster',clusters=[np.zeros(5,dtype=int)],small_sample=small_sample)

@pytest.mark.parametrize('stop',[3,6,7])
def test_generic_absorber_lsmr_rejects_condition_and_iteration_failure(stop):
    from econhdfe.hdfe.absorber import HDFEAbsorber
    m=importlib.import_module('econhdfe.hdfe.absorber')
    a=HDFEAbsorber([np.array([0,0,1,1])],method='lsmr')
    with patch.object(m,'lsmr',return_value=(np.zeros(2),stop,3,1.,1.,1.,1e9,0.)):
        _,info=a.residualize(np.ones(4),return_info=True)
        assert not info.converged

@pytest.mark.parametrize('scale',[1e-10,1e10])
def test_liml_outcome_units_do_not_change_kappa(scale):
    rng=np.random.default_rng(127);n=400
    C=rng.normal(size=(n,1));I=rng.normal(size=(n,2));E=I[:,:1]+rng.normal(size=(n,1))
    y=C[:,0]+2*E[:,0]+rng.normal(size=n)
    a=fit_iv_kclass(y,C,E,I,estimator='liml')
    b=fit_iv_kclass(y*scale,C,E,I,estimator='liml')
    np.testing.assert_allclose(b[0]/scale,a[0],rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(b[1]/scale**2,a[1],rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(b[7]['kappa'],a[7]['kappa'],rtol=1e-9)
