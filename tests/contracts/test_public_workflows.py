"""Current public contracts: real fits across four families, not version smoke tests."""
import importlib
import inspect
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import econhdfe as e
import econhdfe.api as api
import pyreghdfe as legacy


@pytest.fixture(params=['ols', 'iv', 'ppml', 'ivppml'])
def workflow(request, panel_data):
    family, df = request.param, panel_data
    groups = [df.firm.to_numpy(), df.period.to_numpy()]
    cluster_names = ['firm', 'period'] if family == 'ivppml' else ['firm']
    clusters = [df[name].to_numpy() for name in cluster_names]
    xnames = ('c', 'endog')
    common = dict(vce='cluster')
    if family in {'ols', 'iv'}:
        fit = e.olshdfe if family == 'ols' else e.ivhdfe
        roles = dict(x=list(xnames)) if family == 'ols' else dict(exog=['c'], endog=['endog'], instruments=['z'])
        named = fit(df, y='y', absorb=['firm', 'period'], cluster=cluster_names, **roles, **common)
        arrays = {name: df[columns].to_numpy() for name, columns in roles.items()}
        array = fit(y=df.y.to_numpy(), absorb=groups, cluster=clusters, **arrays, **common)
        cls = e.OLSHDFESession if family == 'ols' else e.IVHDFESession
        model = cls(df, cluster=cluster_names, **common)
        reused = model.fit(y='y', absorb=['firm', 'period'], **roles)
        assert model.cache_info()['fe_entries'] == 1
    else:
        cfg = (e.PPMLConfig if family == 'ppml' else e.IVPPMLConfig)(
            engine='optimized', separation=(), tolerance=1e-10, target_inner_tol=1e-11)
        cls = e.PPMLHDFE if family == 'ppml' else e.IVPPMLHDFE
        model = cls(df, absorb=['firm', 'period'], config=cfg)
        identity = id(model.plan)
        roles = dict(X=list(xnames)) if family == 'ppml' else dict(exog=['c'], endog=['endog'], instruments=['z'])
        named = model.fit('count', offset='offset', clusters=cluster_names, **roles, **common)
        reused = model.fit('count', exposure='exposure', clusters=cluster_names, **roles, **common)
        assert id(model.plan) == identity
        if family == 'ppml':
            array = e.ppmlhdfe(df['count'].to_numpy(), df[list(xnames)].to_numpy(), names=xnames,
                              absorb=groups, offset=df.offset.to_numpy(), clusters=clusters, config=cfg, **common)
        else:
            array = e.ivppmlhdfe(df['count'].to_numpy(), exog=df[['c']].to_numpy(),
                                endog=df[['endog']].to_numpy(), instruments=df[['z']].to_numpy(),
                                exog_names=['c'], endog_names=['endog'], instrument_names=['z'],
                                absorb=groups, offset=df.offset.to_numpy(), clusters=clusters, config=cfg, **common)
    return family, df, named, array, reused


def coefficients(result):
    return result.params if isinstance(result, e.RegressionResult) else result.coef


def test_named_array_and_reusable_routes_align(workflow):
    family, df, named, array, reused = workflow
    assert named.names == reused.names
    assert named.names[:2] == ('c', 'endog')
    assert isinstance(named, {'ols': e.RegressionResult, 'iv': e.RegressionResult,
                              'ppml': e.PPMLResult, 'ivppml': e.IVPPMLResult}[family])
    for result in (array, reused):
        assert result.converged and result.nobs == named.nobs == len(df)
        assert result.cluster_counts == named.cluster_counts
        assert_allclose(coefficients(result), coefficients(named), rtol=2e-9, atol=2e-10)
        assert_allclose(result.vcov, named.vcov, rtol=2e-9, atol=2e-10)
        if family in {'ppml', 'ivppml'}:
            assert_array_equal(result.sample_mask, named.sample_mask)
            assert_array_equal(result.separation_mask, named.separation_mask)
            assert_allclose(result.mu, named.mu, rtol=2e-9, atol=2e-10)
            assert_allclose(result.eta, named.eta, rtol=2e-9, atol=2e-10)
        else:
            assert result.df_absorbed == named.df_absorbed
            assert result.df_resid == named.df_resid
            assert_allclose(result.fitted + result.residuals, df.y, rtol=0, atol=1e-12)


def test_publication_protocol_preserves_model_specific_semantics(workflow):
    family, _, result, _, _ = workflow
    table = result.coef_table()
    assert list(table.columns) == ['estimate', 'std_error', 'statistic', 'p_value', 'ci_low', 'ci_high', 'stars']
    assert tuple(table.index) == result.names
    assert_allclose(table['estimate'], coefficients(result))
    out = result.publication_output()
    assert {'coefficients', 'model', 'reproducibility'} <= set(out)
    assert 'diagnostics' not in out and 'profile' not in out
    assert {'nobs', 'df_absorbed', 'vce', 'fixed_effects', 'converged'} <= set(out['model'])
    assert out['model']['vce'] == 'cluster'
    if family == 'iv':
        assert 'diagnostics' in out['first_stage']
        assert 'fitted_endog' not in out['first_stage']
        assert 'diagnostics' in result.publication_output(include_diagnostics=True)
        assert {'kleibergen_paap', 'overidentification'} <= set(out['identification_tests'])
    elif family == 'ols':
        assert {'r2', 'r2_adjusted', 'r2_within', 'r2_adjusted_within'} <= set(out['model'])
    elif family == 'ppml':
        assert {'loglike', 'deviance', 'pseudo_r2'} <= set(out['model'])
    else:
        assert out['model']['endogenous'] == ('endog',)
        assert out['model']['excluded_instruments'] == ('z',)
        normalized = table.loc['_cons']
        assert np.isnan(normalized.p_value) and np.isnan(normalized.ci_low)
        assert np.isnan(normalized.ci_high) and normalized.stars == ''


def test_every_current_public_alias_has_one_implementation():
    assert e.reghdfe is e.olshdfe and e.ivreghdfe is e.ivhdfe
    for name in e.__all__:
        assert getattr(legacy, name) is getattr(e, name), name
    for name in api.__all__:
        assert getattr(api, name) is getattr(e, name), name
    effects = importlib.import_module('econhdfe.effects')
    assert len(effects.__all__) == len(set(effects.__all__))
    assert all(hasattr(effects, name) for name in effects.__all__)


@pytest.mark.parametrize('name', ['olshdfe', 'ivhdfe', 'ppmlhdfe', 'ivppmlhdfe'])
def test_estimator_specific_keyword_boundaries_are_explicit(name):
    # Deliberately DO NOT pretend the four API signatures are interchangeable.
    signature = inspect.signature(getattr(e, name))
    params = signature.parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    assert ('cluster' in params) == (name in {'olshdfe', 'ivhdfe'})
    assert ('clusters' in params) == (name in {'ppmlhdfe', 'ivppmlhdfe'})
    assert ('weight_type' in params) == (name != 'ppmlhdfe')
    required = {key: None for key, p in params.items()
                if p.default is inspect.Parameter.empty}
    # Supply required arguments: failure must identify the unknown keyword,
    # not pass accidentally because a required outcome argument was missing.
    with pytest.raises(TypeError, match="unknown_keyword"):
        signature.bind(**required, unknown_keyword=True)
    with pytest.raises(TypeError, match="unknown_keyword"):
        getattr(e, name)(**required, unknown_keyword=True)
