import numpy as np
import pandas as pd
import pytest

from pyreghdfe import (
    factor, reg_interaction, omit_column, omit_level,
    reghdfe, ivreghdfe,
)


def _event_fixture(seed=7501, n=40000):
    rng = np.random.default_rng(seed)
    nfirm, nyear = 3000, 10
    firm = rng.integers(0, nfirm, n)
    year = rng.integers(0, nyear, n)
    cohort = rng.integers(2, 8, nfirm)
    ever = (rng.random(nfirm) > 0.2).astype(float)
    et = np.clip(year - cohort[firm], -3, 3)
    # Never-treated observations get a sentinel category whose interacted
    # monomial is exactly zero and is automatically omitted.
    et = np.where(ever[firm] > 0, et, 99)
    x = rng.normal(size=n)
    dyn = {-3:-0.1, -2:-0.05, -1:0.0, 0:0.2, 1:0.35, 2:0.5, 3:0.6}
    eff = np.array([dyn.get(int(k), 0.0) for k in et]) * ever[firm]
    y = 0.4*x + eff + rng.normal(size=nfirm)[firm] + rng.normal(size=nyear)[year] + rng.normal(scale=.5,size=n)
    return pd.DataFrame({'firm':firm,'year':year,'event_time':et,'ever':ever[firm],'x':x,'y':y})


def test_event_study_user_reference_matches_factor_base_parameterization():
    df = _event_fixture()
    full = reg_interaction(factor('event_time', drop_base=False, name='event_time'), 'ever', name='event_study')
    based = reg_interaction(factor('event_time', base=-1, drop_base=True, name='event_time'), 'ever', name='event_study')
    with pytest.warns(UserWarning):
        a = reghdfe(df, y='y', x=['x', full], absorb=['firm','year'], omit=[omit_level('event_time', -1, term='event_study')])
    with pytest.warns(UserWarning):
        b = reghdfe(df, y='y', x=['x', based], absorb=['firm','year'])
    assert 'event_time[-1]#ever' not in a.names
    assert tuple(a.names) == tuple(b.names)
    np.testing.assert_allclose(a.params, b.params, atol=2e-10, rtol=2e-10)
    u = a.user_omitted_variables
    assert len(u) == 1
    assert u[0]['name'] == 'event_time[-1]#ever'
    assert u[0]['reason'] == 'user_reference'
    assert u[0]['selected_by_user'] is True


def test_user_can_choose_basis_member_to_omit_in_composite_collinearity():
    rng = np.random.default_rng(7502); n=12000
    x1=rng.normal(size=n); x2=rng.normal(size=n); x3=x1+x2
    g=rng.integers(0,500,n); y=.3*x1-.2*x2+rng.normal(size=500)[g]+rng.normal(size=n)
    df = pd.DataFrame({'y':y,'x1':x1,'x2':x2,'x3':x3,'g':g})
    r = reghdfe(df, y='y', x=['x1','x2','x3'], absorb='g', omit=[omit_column('x1')], collinearity='warn')
    assert r.names == ('x2','x3')
    assert [o['name'] for o in r.user_omitted_variables] == ['x1']
    assert not any(o['name']=='x3' for o in r.automatic_omitted_variables)


def test_omit_selector_typo_is_not_silent():
    rng=np.random.default_rng(7503); n=2000
    with pytest.raises(ValueError, match='matched no requested columns'):
        reghdfe(None, y=rng.normal(size=n), x=rng.normal(size=n), absorb=rng.integers(0,100,n), omit=['does_not_exist'])


def test_iv_allows_explicit_instrument_omission():
    rng=np.random.default_rng(7504); n=15000
    g=rng.integers(0,800,n); z1=rng.normal(size=n); z2=rng.normal(size=n); z3=z1+z2
    w=rng.normal(size=n); x=.7*z1+.4*z2+.2*w+rng.normal(size=n)
    y=1.2*x+.3*w+rng.normal(size=800)[g]+rng.normal(size=n)
    r=ivreghdfe(None,y=y,exog=w,endog=x,instruments=np.column_stack([z1,z2,z3]),absorb=g,
                 omit_instruments=[omit_column('z1')],collinearity='warn')
    assert [o['name'] for o in r.user_omitted_variables] == ['z1']
    assert 'z1' not in r.collinearity_info['excluded_instruments']['active']
