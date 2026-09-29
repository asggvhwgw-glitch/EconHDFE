"""Prediction behavior; failures must not silently change the fitted model."""
import numpy as np
import pandas as pd
import pytest

from econhdfe import olshdfe, ivhdfe, factor, fv, FixedEffect, OLSHDFESession, IVHDFESession
from econhdfe.data import CSVSource
from econhdfe.design import build_design
from econhdfe.errors import InputError, SpecificationError, InferenceError


def frame(seed=88):
    rng = np.random.default_rng(seed)
    n = 480
    df = pd.DataFrame({'g': np.repeat(np.arange(24), 20),
                       't': np.tile(np.arange(5), n // 5),
                       'category': np.tile(['a', 'b', 'c', 'b'], n // 4),
                       'x': rng.normal(size=n), 'z': rng.normal(size=n)})
    df['e'] = .8 * df.z + rng.normal(size=n)
    df['y'] = .7 * df.x + 1.2 * df.e + .3 * df.g - .2 * df.t + rng.normal(size=n)
    return df


@pytest.mark.parametrize('family', ['ols', 'iv'])
def test_predict_saved_fe_reconstructs_sample_and_permuted_rows(family):
    df = frame()
    r = (olshdfe(df, y='y', x=['x', 'e'], absorb=['g', 't'], save_fe=True) if family == 'ols'
         else ivhdfe(df, y='y', exog=['x'], endog=['e'], instruments=['z'],
                     absorb=['g', 't'], save_fe=True))
    new = df.sample(frac=1, random_state=3)
    np.testing.assert_allclose(r.predict(new, chunk_size=37), r.fitted[new.index], atol=2e-8)
    np.testing.assert_allclose(r.predict(df, kind='xb'), df[list(r.names)] @ r.params, atol=1e-12)
    np.testing.assert_allclose(r.predict(df, kind='fe') + r.predict(df, kind='xb'), r.fitted, atol=2e-8)
    np.testing.assert_allclose(r.predict(kind='xb'), r.predict(df, kind='xb'), atol=2e-8)
    copy = r.predict()
    copy[:] = 0
    assert not np.all(r.fitted == 0)
    np.testing.assert_array_equal(r.prediction_state.fixed_effects.categorical.beta, r.params)
    np.testing.assert_allclose(r.predict(df.drop(columns='z')), r.fitted, atol=2e-8)


def test_frozen_factor_base_polynomial_and_interaction_match_manual_design():
    df = frame()
    spec = [fv('i(category,base="b")##c(x)'), fv('c(x)#c(x)')]
    r = olshdfe(df, y='y', x=spec, absorb=['g', 't'], save_fe=True)
    new = df.iloc[[15, 1, 89, 12]].copy()
    new.x += 1.5
    columns = {'category[a]': (new.category == 'a').astype(float),
               'category[c]': (new.category == 'c').astype(float), 'x': new.x,
               'category[a]#x': (new.category == 'a') * new.x,
               'category[c]#x': (new.category == 'c') * new.x, 'x#x': new.x ** 2}
    X = np.column_stack([columns[name] for name in r.names])
    np.testing.assert_allclose(r.predict(new, kind='xb', chunk_size=1), X @ r.params, atol=1e-12)
    np.testing.assert_allclose(r.predict(X=X, kind='stdp'), np.sqrt(np.einsum('ij,jk,ik->i', X, r.vcov, X)), atol=1e-12)
    raw = build_design(df, spec, len(df), structural=False)
    np.testing.assert_allclose(r.predict(df, kind='xb'), raw.values @ r.params, atol=1e-12)
    assert r.prediction_state.design('regressor').terms[0].categorical[0].base_levels == ('b',)


@pytest.mark.parametrize('kind', ['xb', 'stdp', 'response'])
def test_unknown_and_nonfinite_rows_raise_or_produce_nan(kind):
    df = frame()
    r = olshdfe(df, y='y', x=[factor('category', base='b'), 'x'], absorb=['g', 't'], save_fe=True)
    new = df.iloc[:4].copy()
    new.loc[0, 'category'] = 'unseen'
    new.loc[1, 'x'] = np.inf
    new.loc[2, 'x'] = np.nan
    with pytest.raises(InputError):
        r.predict(new, kind=kind)
    pred = r.predict(new, kind=kind, unknown='nan', chunk_size=2)
    assert np.isnan(pred[:3]).all() and np.isfinite(pred[3])
    np.testing.assert_allclose(pred[3:], r.predict(new.iloc[3:], kind=kind))


def test_unobserved_interaction_cell_does_not_turn_into_base_zero():
    rng = np.random.default_rng(33)
    cells = np.tile([[0, 0], [0, 1], [1, 0]], (80, 1))
    df = pd.DataFrame({'a': cells[:, 0], 'b': cells[:, 1], 'g': np.arange(len(cells)) % 8,
                       'x': rng.normal(size=len(cells)), 'y': rng.normal(size=len(cells))})
    r = olshdfe(df, y='y', x=[fv('i(a,base=none)#i(b,base=none)#c(x)')], absorb=['g'])
    new = pd.DataFrame({'a': [0, 1], 'b': [1, 1], 'x': [2., 2.]})
    with pytest.raises(InputError) as e:
        r.predict(new, kind='xb')
    assert e.value.code == 'prediction.unobserved_cell'
    out = r.predict(new, kind='xb', unknown='nan')
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_disconnected_fe_cross_component_combinations_rejected():
    df = frame()
    df['t'] = df.t + 5 * (df.g >= 12)
    r = olshdfe(df, y='y', x=['x'], absorb=['g', 't'], save_fe=True)
    new = df.iloc[:2].copy()
    new.loc[0, 't'] = 7
    new.loc[1, 'g'] = 999
    with pytest.raises(InputError):
        r.predict(new)
    assert np.isnan(r.predict(new, unknown='nan')).all()
    np.testing.assert_allclose(r.predict(df), r.fitted, atol=1e-8)


def test_canonicalized_nested_fe_must_keep_fitted_mapping():
    df = frame()
    df['province'] = df.g // 6
    r = olshdfe(df, y='y', x=['x'], absorb=['g', 'province', 't'], save_fe=True)
    assert r.prediction_state.fixed_effects.categorical.nesting
    np.testing.assert_allclose(r.predict(df), r.fitted, atol=1e-8)
    new = df.iloc[:2].copy()
    new.loc[0, 'province'] = 3
    result = r.predict(new, unknown='nan')
    assert np.isnan(result[0]) and np.isfinite(result[1])


def test_three_way_extra_nullity_refuses_only_unidentified_component():
    good = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T
    bad = np.array([[2, 2, 2], [2, 2, 3], [3, 3, 2]])
    edges = np.repeat(np.vstack([good, bad]), 20, axis=0)
    rng = np.random.default_rng(51)
    df = pd.DataFrame(edges, columns=['a', 'b', 'c'])
    df['x'] = rng.normal(size=len(edges))
    df['y'] = .8 * df.x + .2 * df.a - .3 * df.b + .7 * df.c + rng.normal(size=len(edges))
    r = olshdfe(df, y='y', x=['x'], absorb=['a', 'b', 'c'], save_fe=True, canonicalize_fe=False)
    out = r.predict(df, unknown='nan')
    good_rows = df.a < 2
    np.testing.assert_allclose(out[good_rows], r.fitted[good_rows], atol=1e-8)
    assert np.isnan(out[~good_rows]).all()
    np.testing.assert_array_equal(r.predict(), r.fitted)


def test_interaction_fe_labels_and_file_backed_labels_are_preserved(tmp_path):
    df = frame()
    df['g'] = 'firm-' + df.g.astype(str)
    df['t'] = 'year-' + df.t.astype(str)
    path = tmp_path / 'data.csv'
    df.to_csv(path, index=False)
    r = olshdfe(CSVSource(path), y='y', x=['x', factor('category')],
                absorb=[('g', 't')], save_fe=True)
    assert r.prediction_state.can_include_fixed_effects
    np.testing.assert_allclose(r.predict(df), r.fitted, atol=2e-8)
    assert r.prediction_state.fixed_effects.categorical.terms[0].sources == ('g', 't')
    term = r.prediction_state.design('regressor').terms[1]
    assert all(str(v) in {'a', 'b', 'c'} for v in term.categorical[0].levels)
    frozen = term.categorical[0].levels.copy()
    df.loc[:, 'category'] = 'changed'
    np.testing.assert_array_equal(term.categorical[0].levels, frozen)


@pytest.mark.parametrize('family', ['ols', 'iv'])
def test_session_persistent_predictions_match_direct_without_refit(tmp_path, family):
    df = frame()
    path = tmp_path / 'data.csv'
    df.to_csv(path, index=False)
    cls = OLSHDFESession if family == 'ols' else IVHDFESession
    roles = {'x': ['x', 'e']} if family == 'ols' else {'exog': ['x'], 'endog': ['e'], 'instruments': ['z']}
    kwargs = {'y': 'y', 'absorb': ['g', 't'], **roles}
    a = cls(CSVSource(path)).enable_persistent_cache(tmp_path / 'cache')
    first = a.fit(**kwargs)
    b = cls(CSVSource(path)).enable_persistent_cache(tmp_path / 'cache')
    second = b.fit(**kwargs)
    assert b.persistent_cache_info()['result_hits'] == 1
    assert b.cache_info()['residualize_calls'] == 0
    np.testing.assert_allclose(second.predict(df, kind='xb'), first.predict(df, kind='xb'))
    np.testing.assert_allclose(second.predict(df, kind='stdp'), first.predict(df, kind='stdp'))
    with pytest.raises(SpecificationError, match='save_fe'):
        second.predict(df)


def test_sample_restore_uses_positions_not_index_or_fweight_count():
    df = frame().iloc[:40].copy()
    df.loc[0, 'g'] = 999
    df['weight'] = 1
    df.loc[2, 'weight'] = 3
    df.index = np.repeat('duplicate-index', len(df))
    r = olshdfe(df, y='y', x=['x'], absorb=['g'], weights='weight', weight_type='fweight', save_fe=True)
    assert r.nobs != r.prediction_state.sample.nobs_final
    restored = r.predict(restore_sample=True)
    assert len(restored) == len(df) and np.isnan(restored[0])
    np.testing.assert_allclose(restored[1:], r.fitted)
    with pytest.raises(SpecificationError):
        r.predict(df, restore_sample=True)
    assert np.isnan(r.predict(df.iloc[[0]], unknown='nan')).all()


def test_varying_slopes_never_fall_back_silently_to_xb():
    df = frame()
    r = olshdfe(df, y='y', x=['e'], absorb=[FixedEffect('g', slopes=('x',))], save_fe=True)
    assert not r.prediction_state.can_include_fixed_effects
    with pytest.raises(SpecificationError):
        r.predict(df)
    np.testing.assert_allclose(r.predict(df, kind='xb'), df[['e']] @ r.params)


def test_explicit_matrix_and_argument_failures():
    df = frame()
    X = df[['x', 'e']].to_numpy()
    r = olshdfe(y=df.y.to_numpy(), x=X, absorb=df.g.to_numpy())
    np.testing.assert_allclose(r.predict(X=X, kind='xb'), X @ r.params)
    for kwargs in ({'data': df, 'X': X}, {'X': X[:, :1], 'kind': 'xb'},
                   {'data': df, 'kind': 'unknown'}, {'data': df, 'unknown': 'zero'},
                   {'data': df, 'chunk_size': 0}, {'data': df, 'chunk_size': True},
                   {'data': df, 'kind': 'xb'}, {'X': X}):
        with pytest.raises(ValueError):
            r.predict(**kwargs)
    with pytest.raises(SpecificationError):
        r.predict(kind='stdp')
    named = olshdfe(df, y='y', x=['x', 'e'], absorb=['g'])
    assert named.predict(df.iloc[:0], kind='xb').shape == (0,)
    with pytest.raises(SpecificationError):
        named.predict(df.drop(columns='x'), kind='xb')
    original = X.copy()
    X.flags.writeable = False
    named.predict(X=X, kind='stdp', chunk_size=3)
    np.testing.assert_array_equal(X, original)


def test_automatic_omissions_need_estimability_but_explicit_xb_available():
    df = frame()
    df['xdup'] = df.x * 2
    r = olshdfe(df, y='y', x=['x', 'xdup'], absorb=['g'], save_fe=True, collinearity='drop')
    with pytest.raises(SpecificationError) as e:
        r.predict(df)
    assert e.value.code == 'prediction.omitted_estimability'
    np.testing.assert_allclose(r.predict(df, kind='xb'), df[list(r.names)] @ r.params)


def test_negative_variance_and_mutated_coefficients_fail():
    df = frame()
    r = olshdfe(df, y='y', x=['x'], absorb=['g'], save_fe=True)
    r.vcov[:] = -1
    with pytest.raises(InferenceError, match='negative'):
        r.predict(df, kind='stdp')
    r.params[:] += 1
    with pytest.raises(SpecificationError, match='changed'):
        r.predict(df)


def test_resource_failure_preserves_fit_disables_new_fe_predictions(monkeypatch):
    from econhdfe.effects import prediction
    from econhdfe.hdfe.rank import _ExactRankResourceError
    def refused(*args, **kwargs):
        raise _ExactRankResourceError('test', 'work', 2, 1)
    monkeypatch.setattr(prediction, 'identification_structure', refused)
    df = frame()
    r = olshdfe(df, y='y', x=['x'], absorb=['g', 't'], save_fe=True)
    assert r.converged
    assert r.prediction_state.fixed_effects.unavailable_reason == 'identification_resource_budget'
    with pytest.raises(SpecificationError):
        r.predict(df)
    np.testing.assert_array_equal(r.predict(), r.fitted)


def test_categorical_fe_result_cache_roundtrip_remains_read_only(tmp_path):
    from econhdfe.data.persistent import PersistentSessionStore
    df = frame()
    df['g'] = 'firm-' + df.g.astype(str)
    r = olshdfe(df, y='y', x=[factor('category'), 'x'], absorb=['g', 't'], save_fe=True)
    store = PersistentSessionStore(tmp_path)
    key = store.result_key({'case': 'saved-categorical'})
    assert store.save_result(key, r)
    restored = store.load_result(key)
    assert restored is not None
    np.testing.assert_allclose(restored.predict(df), r.predict(df), atol=0, rtol=0)
    for array in restored.prediction_state.fixed_effects.categorical.coefficients:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    np.testing.assert_allclose(restored.predict(kind='fe'), r.predict(kind='fe'))


@pytest.mark.parametrize('family', ['ols', 'iv'])
def test_block_native_predicts_from_raw_rows(family):
    from econhdfe import reg_interaction
    df = frame()
    slopes = [reg_interaction(factor('g', drop_base=False), 'e')]
    if family == 'ols':
        specs = slopes + ['x']
        r = olshdfe(df, y='y', x=specs, absorb=['g'], collinearity='drop')
        assert r.absorb_info.get('heterogeneous_spec') is not None
    else:
        r = ivhdfe(df, y='y', exog=['x'], endog=slopes,
                   instruments=[reg_interaction(factor('g', drop_base=False), 'z')],
                   absorb=['g'], collinearity='drop')
        assert r.diagnostics.get('heterogeneous_spec_path') is True
        specs = ['x'] + slopes
    X = build_design(df, specs, len(df), structural=False).values
    np.testing.assert_allclose(r.predict(df, kind='xb', chunk_size=17), X @ r.params, atol=1e-12)
    np.testing.assert_allclose(r.predict(df, kind='stdp', chunk_size=17),
                               np.sqrt(np.einsum('ij,jk,ik->i', X, r.vcov, X)), atol=1e-12)


def test_user_omission_and_nullable_numeric_predictions():
    df = frame()
    r = olshdfe(df, y='y', x=['x', 'e'], omit=['e'], absorb=['g'], save_fe=True)
    np.testing.assert_allclose(r.predict(df), r.fitted, atol=1e-8)
    new = df.iloc[:3].copy()
    new['x'] = pd.array([1., pd.NA, 2.], dtype='Float64')
    out = r.predict(new, unknown='nan')
    assert np.isfinite(out[[0, 2]]).all() and np.isnan(out[1])
    with pytest.raises(SpecificationError):
        r.predict(pd.DataFrame(), kind='xb')


def test_predict_chunks_and_default_fit_do_not_run_fe_recovery(monkeypatch):
    from econhdfe.effects import prediction as effects_prediction
    from econhdfe import prediction_api
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected FE identification during ordinary fit/prediction')
    monkeypatch.setattr(effects_prediction, 'identification_structure', forbidden)
    df = frame()
    r = olshdfe(df, y='y', x=[factor('category'), 'x'], absorb=['g'])
    shapes = []
    original = prediction_api._design_chunk
    def measured(*args, **kwargs):
        X, bad = original(*args, **kwargs)
        shapes.append(X.shape)
        return X, bad
    monkeypatch.setattr(prediction_api, '_design_chunk', measured)
    r.predict(df, kind='stdp', chunk_size=17)
    assert len(shapes) == (len(df) + 16) // 17
    assert max(shape[0] for shape in shapes) == 17
    assert all(shape[1] == len(r.params) for shape in shapes)


def test_fitted_result_does_not_retain_dataframe():
    import weakref
    import gc
    df = frame()
    new = df.iloc[:3].copy()
    reference = weakref.ref(df)
    r = olshdfe(df, y='y', x=[factor('category'), 'x'], absorb=['g'], save_fe=True)
    expected = r.predict(new)
    del df
    gc.collect()
    assert reference() is None
    np.testing.assert_allclose(r.predict(new), expected)
