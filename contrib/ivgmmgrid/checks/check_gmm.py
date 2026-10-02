"""Numerical and sample contracts, including an external frozen Stata oracle."""
from pathlib import Path
import json
import numpy as np
import pytest
from ivgmmgrid import fit_grid, IdentificationError, distributed_lag, select_water_grid

FIXTURES = Path(__file__).parent/'fixtures'


@pytest.fixture
def data():
    return dict(np.load(FIXTURES/'native_contract.npz'))


def args(d):
    return [d['y'],d['w'],np.column_stack([d['x1'],d['x2']]),
            np.column_stack([d['z1'],d['z2']]),d['t'],d['bsid']]


@pytest.mark.parametrize('engine',['cached','reference'])
@pytest.mark.parametrize('case',['base','nested','omitted'])
def test_external_native_oracle(data,engine,case):
    a=args(data);a[2]=data['w'] if case=='omitted' else data['x1']
    if case=='nested': a[4]=data['bsid']
    m=fit_grid(*a,engine=engine).models[0]
    r=next(r for r in json.loads((FIXTURES/'native_contract.json').read_text()) if r['model']==case)
    np.testing.assert_allclose(m.coefficients,[r['b1'],r['b2']],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(m.covariance,[[r['v11'],r['v12']],[r['v12'],r['v22']]],rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose([m.rss,m.rmse,m.hansen_j],[r['rss'],r['rmse'],r['j']],rtol=1e-9,atol=1e-11)
    assert (m.nobs,m.nclusters,m.df_resid,m.absorbed_df)==(r['n'],r['g'],r['df'],r['sdof'])
    assert m.omitted.tolist()==[False,case=='omitted']


def test_cache_reference_and_permutation(data):
    a=args(data); result=fit_grid(*a,verify=True)
    perm=np.random.default_rng(13).permutation(len(a[0]))
    reordered=fit_grid(*[x[perm] for x in a],verify=True)
    np.testing.assert_allclose(result.coefficients,reordered.coefficients,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result.covariance,reordered.covariance,rtol=1e-10,atol=1e-12)


def test_same_n_different_rows_are_not_pooled(data):
    a=args(data);a[2]=a[2].copy();a[2][0,0]=np.nan;a[2][1,1]=np.nan
    result=fit_grid(*a,verify=True)
    assert result.sample_groups==2
    assert result.models[0].nobs==result.models[1].nobs
    assert not np.array_equal(result.models[0].sample_indices,result.models[1].sample_indices)
    for j in range(2):
        solo=fit_grid(a[0],a[1],a[2][:,j],*a[3:]).models[0]
        np.testing.assert_allclose(result.models[j].coefficients,solo.coefficients)


def test_units(data):
    a=args(data); ref=fit_grid(*a)
    a[1]=a[1].astype(float)*1e8;a[2]=a[2].astype(float)*1e-5;a[3]=a[3]*np.array([1e7,1e-4])
    actual=fit_grid(*a)
    scale=np.array([1e-5,1e8])
    np.testing.assert_allclose(actual.coefficients*scale,ref.coefficients,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(actual.covariance*scale[:,None]*scale[None,:],ref.covariance,rtol=1e-8,atol=1e-10)


def test_mask_missing_and_singletons(data):
    a=args(data);a[4]=a[4].astype(np.int64);a[4][0]=99999
    a[1]=a[1].copy();a[1][10]=np.nan
    mask=np.arange(len(a[0]))>=5
    result=fit_grid(*a,mask=mask)
    for m in result.models:
        assert not np.isin([0,1,2,3,4,10],m.sample_indices).any()
    result=fit_grid(*a)
    assert 0 not in result.models[0].sample_indices


@pytest.mark.parametrize('failure',['empty','absorbed','clusters','mask','unidentified'])
def test_failures(data,failure):
    a=args(data);kwargs={}
    if failure=='empty': a[2]=np.full_like(a[2],np.nan)
    if failure=='absorbed': a[2]=np.ones_like(a[2])
    if failure=='clusters': a[5]=np.zeros_like(a[5])
    if failure=='mask': kwargs['mask']=np.ones(len(a[0]))
    if failure=='unidentified': a[3]=a[1][:,None]
    with pytest.raises((ValueError,IdentificationError)):
        fit_grid(*a,**kwargs)


def test_float_replacement_and_ties(data):
    lags=np.column_stack([data['x1'],np.zeros((len(data['x1']),3))])
    # All gamma/order fits identical: within-order mean, between-order maximum order.
    gammas=np.array([.05,.06,.07])
    r=select_water_grid(data['y'],data['w'],lags,np.column_stack([data['z1'],data['z2']]),
                        data['t'],data['bsid'],gammas)
    assert r.order==4
    assert r.gamma==float(np.float32(np.mean(gammas.astype(np.float32).astype(float))))
    np.testing.assert_array_equal(distributed_lag(lags,4,.2),lags[:,0].astype(np.float32).astype(float))
