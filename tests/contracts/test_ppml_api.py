"""PPML/IV-PPML public entry points share the OLS/IV calling convention."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import econhdfe as e
from econhdfe import (ExecutionConfig, InferenceConfig, IVPPMLConfig, IVPPMLHDFE, PPMLConfig,
                      PPMLHDFE, ivppmlhdfe, ppmlhdfe)
from econhdfe.errors import DataTypeError, SpecificationError


FIELDS = ('params', 'vcov', 'stderr', 'mu', 'eta', 'sample_mask', 'separation_mask')


def _same(a, b):
    assert a.names == b.names
    assert a.nobs == b.nobs and a.iterations == b.iterations and a.vce == b.vce
    assert a.cluster_counts == b.cluster_counts and a.df_absorbed == b.df_absorbed
    for name in FIELDS:
        assert_array_equal(getattr(a, name), getattr(b, name))


def _pcfg():
    return PPMLConfig(engine='optimized', tolerance=1e-10, target_inner_tol=1e-11)


def _icfg():
    return IVPPMLConfig(engine='optimized', tolerance=1e-10, target_inner_tol=1e-11)


def _iv_arrays(df):
    return dict(exog=df[['c']].to_numpy(), endog=df[['endog']].to_numpy(),
                instruments=df[['z']].to_numpy(), exog_names=['c'], endog_names=['endog'],
                instrument_names=['z'])


def test_ppml_dataframe_and_array_forms_are_bitwise_identical(panel_data):
    df = panel_data
    framed = ppmlhdfe(df, y='count', x=['c', 'endog'], absorb=['firm', 'period'],
                      offset='offset', vce='cluster', cluster='firm', config=_pcfg())
    model = PPMLHDFE(df, absorb=['firm', 'period'], config=_pcfg())
    reused = model.fit('count', ['c', 'endog'], offset='offset', vce='cluster', cluster='firm')
    array = ppmlhdfe(y=df['count'].to_numpy(), x=df[['c', 'endog']].to_numpy(),
                     names=('c', 'endog'), absorb=[df.firm.to_numpy(), df.period.to_numpy()],
                     offset=df.offset.to_numpy(), vce='cluster', cluster=[df.firm.to_numpy()],
                     config=_pcfg())
    assert isinstance(framed, e.PPMLResult)
    _same(framed, reused)
    _same(framed, array)


def test_ppml_dataframe_form_accepts_heterogeneous_and_single_terms(panel_data):
    df = panel_data.assign(cat=np.arange(len(panel_data)) % 3)
    spec = ['c', e.factor('cat')]
    framed = ppmlhdfe(df, y='count', x=spec, absorb='firm', exposure='exposure', config=_pcfg())
    reused = PPMLHDFE(df, absorb=['firm'], config=_pcfg()).fit('count', spec, exposure='exposure')
    _same(framed, reused)
    single = ppmlhdfe(df, y='count', x='c', absorb='firm', config=_pcfg())
    _same(single, PPMLHDFE(df, absorb=['firm'], config=_pcfg()).fit('count', ['c']))


def test_ivppml_dataframe_and_array_forms_are_bitwise_identical(panel_data):
    df = panel_data
    roles = dict(exog=['c'], endog='endog', instruments=['z'])
    framed = ivppmlhdfe(df, y='count', absorb=['firm', 'period'], offset='offset',
                        vce='cluster', cluster=['firm', 'period'], config=_icfg(), **roles)
    reused = IVPPMLHDFE(df, absorb=['firm', 'period'], config=_icfg()).fit(
        'count', offset='offset', vce='cluster', cluster=['firm', 'period'], **roles)
    array = ivppmlhdfe(y=df['count'].to_numpy(), absorb=[df.firm.to_numpy(), df.period.to_numpy()],
                       offset=df.offset.to_numpy(), vce='cluster',
                       cluster=[df.firm.to_numpy(), df.period.to_numpy()], config=_icfg(),
                       **_iv_arrays(df))
    assert isinstance(framed, e.IVPPMLResult)
    _same(framed, reused)
    _same(framed, array)


def test_cluster_is_the_only_cluster_keyword(panel_data):
    df = panel_data
    y, x = df['count'].to_numpy(), df[['c']].to_numpy()
    with pytest.raises(TypeError):
        ppmlhdfe(y=y, x=x, vce='cluster', clusters=[df.firm.to_numpy()])
    with pytest.raises(TypeError):
        ivppmlhdfe(y=y, vce='cluster', clusters=[df.firm.to_numpy()], **_iv_arrays(df))
    with pytest.raises(TypeError):
        PPMLHDFE(df, absorb=['firm']).fit('count', ['c'], vce='cluster', clusters='firm')
    with pytest.raises(TypeError):
        IVPPMLHDFE(df, absorb=['firm']).fit('count', endog=['endog'], instruments=['z'],
                                            vce='cluster', clusters='firm')
    with pytest.raises(TypeError):
        ppmlhdfe(y, x)  # the old positional (y, X) form is gone
    assert e.SPJPanel.class_b().cluster == ('pair',)
    assert not hasattr(e.SPJPanel.class_b(), 'clusters')


def test_params_is_the_coefficient_array(panel_data):
    df = panel_data
    r = ppmlhdfe(y=df['count'].to_numpy(), x=df[['c', 'endog']].to_numpy(), config=_pcfg())
    iv = ivppmlhdfe(y=df['count'].to_numpy(), config=_icfg(), **_iv_arrays(df))
    for result in (r, iv):
        assert isinstance(result.params, np.ndarray) and result.params.dtype == np.float64
        assert len(result.params) == len(result.names)
        assert not hasattr(result, 'coef')
        assert_array_equal(result.coef_table()['estimate'].to_numpy(), result.params)
    for cls in (e.PPMLResult, e.IVPPMLResult, e.SPJResult):
        assert [f.name for f in cls.__dataclass_fields__.values()][0] == 'params'


def test_vce_and_inference_config_conflict_or_apply_one_sided(panel_data):
    df = panel_data
    arrays = dict(y=df['count'].to_numpy(), x=df[['c']].to_numpy(), config=_pcfg())
    with pytest.raises(SpecificationError) as err:
        ppmlhdfe(**arrays, vce='model', inference_config=InferenceConfig(vce='robust'))
    assert err.value.code == 'specification.config_conflict'
    with pytest.raises(SpecificationError) as err:
        ivppmlhdfe(y=arrays['y'], vce='model', inference_config=InferenceConfig(vce='cluster'),
                   config=_icfg(), **_iv_arrays(df))
    assert err.value.code == 'specification.config_conflict'
    model = PPMLHDFE(df, absorb=['firm'], config=_pcfg(), inference_config=InferenceConfig(vce='model'))
    with pytest.raises(SpecificationError) as err:
        model.fit('count', ['c'], vce='cluster', cluster='firm')
    assert err.value.code == 'specification.config_conflict'
    ivmodel = IVPPMLHDFE(df, absorb=['firm'], config=_icfg(), inference_config=InferenceConfig(vce='model'))
    with pytest.raises(SpecificationError) as err:
        ivmodel.fit('count', endog=['endog'], instruments=['z'], vce='cluster', cluster='firm')
    assert err.value.code == 'specification.config_conflict'

    by_arg = ppmlhdfe(**arrays, vce='model')
    by_cfg = ppmlhdfe(**arrays, inference_config=InferenceConfig(vce='model'))
    both = ppmlhdfe(**arrays, vce='model', inference_config=InferenceConfig(vce='model'))
    default_arg = ppmlhdfe(**arrays, vce='robust', inference_config=InferenceConfig(vce='model'))
    for result in (by_arg, by_cfg, both, default_arg):
        _same(result, by_arg)
        assert result.vce == by_arg.vce
    assert by_arg.vce != ppmlhdfe(**arrays).vce
    _same(model.fit('count', ['c']), PPMLHDFE(df, absorb=['firm'], config=_pcfg()).fit('count', ['c'], vce='model'))
    _same(ivmodel.fit('count', endog=['endog'], instruments=['z']),
          IVPPMLHDFE(df, absorb=['firm'], config=_icfg()).fit('count', endog=['endog'], instruments=['z'], vce='model'))


@pytest.mark.parametrize('family', ['ppml', 'ivppml'])
def test_full_diagnostics_include_profile_by_default(panel_data, family):
    df = panel_data
    execution = ExecutionConfig(profile='summary')

    def fit(mode):
        inference = InferenceConfig(diagnostics=mode)
        if family == 'ppml':
            return ppmlhdfe(df, y='count', x=['c'], absorb=['firm'], config=_pcfg(),
                            inference_config=inference, execution_config=execution)
        return ivppmlhdfe(df, y='count', endog='endog', instruments='z', absorb=['firm'],
                          config=_icfg(), inference_config=inference, execution_config=execution)

    full, publication = fit('full'), fit('publication')
    assert full.profile is not None and publication.profile is not None
    out = full.publication_output()
    assert out['profile'] == full.profile and 'diagnostics' in out
    assert 'profile' not in full.publication_output(include_profile=False)
    assert 'profile' not in publication.publication_output()
    assert 'profile' in publication.publication_output(include_profile=True)


def test_dataframe_form_rejects_array_only_labels_and_non_frames(panel_data):
    df = panel_data
    with pytest.raises(SpecificationError) as err:
        ppmlhdfe(df, y='count', x=['c'], names=['a'])
    assert err.value.code == 'specification.array_only_option'
    with pytest.raises(SpecificationError):
        ivppmlhdfe(df, y='count', endog='endog', instruments='z', endog_names=['e'])
    with pytest.raises(DataTypeError) as err:
        ppmlhdfe(df.to_numpy(), y='count', x=['c'])
    assert err.value.code == 'input.data_type'


@pytest.mark.parametrize('config_type', [PPMLConfig, IVPPMLConfig])
def test_engine_routing_contract(config_type):
    from econhdfe.hdfe.factory import projection_route
    with pytest.raises(SpecificationError, match='engine'):
        config_type(engine='unknown').validate()
    assert config_type(engine='within').separation_engine == 'replica'
    assert config_type(engine='optimized').separation_engine == 'optimized'
    assert projection_route('within', 'map') == ('within', True)
    assert projection_route('within', 'lsmr') == ('lsmr', True)
    assert projection_route('replica', 'within') == ('within', False)
    assert projection_route('optimized', 'map') == ('map', False)


@pytest.mark.parametrize('family', ['ppml', 'ivppml'])
def test_default_engine_keeps_the_map_projector(family):
    rng = np.random.default_rng(26027)
    n = 550
    groups = [rng.integers(0, L, size=n) for L in (45, 25, 10)]
    z, c = rng.normal(size=n), rng.normal(size=n)
    x = 0.75 * z + 0.4 * c + rng.normal(scale=0.6, size=n)
    y = rng.poisson(np.exp(0.2 + 0.11 * x + 0.07 * c)).astype(float)
    common = dict(y=y, absorb=groups, weights=np.exp(rng.normal(scale=0.3, size=n)),
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=192))
    if family == 'ppml':
        result = ppmlhdfe(x=np.column_stack([x, c]), config=PPMLConfig(separation=()), **common)
    else:
        result = ivppmlhdfe(exog=c, endog=x, instruments=z, weight_type='pweight',
                            config=IVPPMLConfig(separation=()), **common)
    resources = result.diagnostics['projection_resources']
    assert result.converged and resources['method'] == 'map' and 'within' not in resources
