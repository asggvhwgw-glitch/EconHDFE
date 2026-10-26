"""Prediction behavior; failures must not silently change the fitted model."""

import numpy as np
import pandas as pd
import pytest
from econhdfe import olshdfe, factor, fv, OLSHDFESession, IVHDFESession
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


def test_unknown_and_nonfinite_rows_raise_or_produce_nan():
    kind = 'response'
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


def test_negative_variance_and_mutated_coefficients_fail():
    df = frame()
    r = olshdfe(df, y='y', x=['x'], absorb=['g'], save_fe=True)
    r.vcov[:] = -1
    with pytest.raises(InferenceError, match='negative'):
        r.predict(df, kind='stdp')
    r.params[:] += 1
    with pytest.raises(SpecificationError, match='changed'):
        r.predict(df)
