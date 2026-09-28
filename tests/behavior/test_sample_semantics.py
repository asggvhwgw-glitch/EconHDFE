"""Observable sample semantics: pruning, row identity and separation are distinct."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import econhdfe as e


def test_recursive_singletons_match_explicit_final_sample():
    # Last row is singleton in a, removing it exposes a singleton in b.
    a = np.array([0,0,1,1,1,2]); b = np.array([0,1,0,1,2,2])
    x = np.array([0.,1.,2.,0.,7.,9.]); y = np.array([1.,3.,2.,1.,8.,10.])
    r = e.olshdfe(y=y, x=x, absorb=[a,b])
    keep = np.array([True,True,True,True,False,False])
    ref = e.olshdfe(y=y[keep], x=x[keep], absorb=[a[keep],b[keep]], drop_singletons=False)
    assert r.nobs == 4 and r.dropped_singletons == 2
    assert_allclose(r.params, ref.params, atol=1e-12, rtol=1e-12)
    assert_allclose(r.vcov, ref.vcov, atol=1e-12, rtol=1e-12)
    assert_allclose(r.fitted+r.residuals,y[keep],atol=1e-12,rtol=0)


@pytest.mark.parametrize('family', ['ols', 'iv'])
def test_dataframe_index_does_not_change_positional_sample_identity(family, panel_data):
    original = panel_data.copy(deep=True)
    changed = panel_data.copy(deep=True)
    changed.index = pd.Index(np.arange(len(changed))[::-1]*17+1000, name='external_row_id')
    fit = e.olshdfe if family == 'ols' else e.ivhdfe
    roles = dict(x=['c','endog']) if family == 'ols' else dict(exog=['c'],endog=['endog'],instruments=['z'])
    a = fit(original,y='y',absorb=['firm','period'],cluster='firm',vce='cluster',**roles)
    b = fit(changed,y='y',absorb=['firm','period'],cluster='firm',vce='cluster',**roles)
    assert_allclose(a.params,b.params,atol=0,rtol=0)
    assert_allclose(a.vcov,b.vcov,atol=0,rtol=0)
    pd.testing.assert_frame_equal(original,panel_data)
    assert_array_equal(changed.index,np.arange(len(changed))[::-1]*17+1000)
