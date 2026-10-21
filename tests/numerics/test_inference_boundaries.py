"""Inference boundaries: units, rank, invalid references and solver termination."""

import importlib
from unittest.mock import patch
import numpy as np
import pytest
from econhdfe.models.linear_iv.estimators import fit_iv_kclass
from econhdfe.reporting import inference_table


def test_invalid_t_reference_is_not_normal_fallback():
    df = None
    table=inference_table(['x'],[1.],[.01],df=df,distribution='t')
    assert np.isnan(table['p_value'].iloc[0])
    assert table['stars'].iloc[0]==''


@pytest.mark.parametrize('stop', [
    1,
    7,
])
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


def test_liml_outcome_units_do_not_change_kappa():
    scale = 1e-10
    rng=np.random.default_rng(127);n=400
    C=rng.normal(size=(n,1));I=rng.normal(size=(n,2));E=I[:,:1]+rng.normal(size=(n,1))
    y=C[:,0]+2*E[:,0]+rng.normal(size=n)
    a=fit_iv_kclass(y,C,E,I,estimator='liml')
    b=fit_iv_kclass(y*scale,C,E,I,estimator='liml')
    np.testing.assert_allclose(b[0]/scale,a[0],rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(b[1]/scale**2,a[1],rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(b[7]['kappa'],a[7]['kappa'],rtol=1e-9)
