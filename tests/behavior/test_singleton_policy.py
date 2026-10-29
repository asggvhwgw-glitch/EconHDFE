"""Mode/weight/stage contracts; expanded data are the frequency oracle."""
import numpy as np
import pytest
from econhdfe import (
    ExecutionConfig, HDFEConfig, PPMLConfig, IVPPMLConfig, ppmlhdfe, ivppmlhdfe, olshdfe, ivhdfe,
)
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.separation import initial_sample_selection


def _fixture():
    rng = np.random.default_rng(270026)
    n = 164
    g = np.r_[np.repeat(np.arange(20), 8), 20, 21, 22, 22]
    h = np.r_[np.tile(np.arange(4), 40), 0, 1, 2, 3]
    c, z, v = rng.normal(size=(3, n))
    d = .85 * z + .2 * v
    y = rng.poisson(np.exp(1 + .12*c + .18*d + .01*g)).astype(float)
    y[np.arange(20)*8] += 1
    y[-4:] = [3, 2, 0, 0]
    fw = rng.integers(1, 4, n)
    fw[-4:] = [5, 1, 2, 3]
    return y, c, d, z, g, h, fw


@pytest.mark.parametrize('family,vce', [('ppml','robust'), ('ppml','cluster'), ('ppml','model'),
                                        ('ivppml','robust'), ('ivppml','cluster')])
def test_unified_frequency_expansion_including_saturated_and_zero_levels(family, vce):
    y, c, d, z, g, h, fw = _fixture()
    ix = np.repeat(np.arange(len(y)), fw)
    config_type = PPMLConfig if family == 'ppml' else IVPPMLConfig
    cfg = config_type(singleton_mode='unified', tolerance=1e-10, target_inner_tol=1e-11)
    def fit(rows, weights=None):
        kwargs = dict(y=y[rows], absorb=[g[rows], h[rows]], weights=weights,
                      weight_type='fw' if weights is not None else None,
                      vce=vce, cluster=[g[rows]] if vce == 'cluster' else None, config=cfg)
        return (ppmlhdfe(x=np.column_stack([c[rows], d[rows]]), **kwargs) if family == 'ppml'
                else ivppmlhdfe(exog=c[rows], endog=d[rows], instruments=z[rows], **kwargs))
    compact, expanded = fit(np.arange(len(y)), fw), fit(ix)
    np.testing.assert_allclose(compact.params, expanded.params, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(compact.vcov, expanded.vcov, rtol=2e-6, atol=2e-9)
    np.testing.assert_array_equal(compact.sample_mask[ix], expanded.sample_mask)
    np.testing.assert_array_equal(compact.separation_mask[ix], expanded.separation_mask)
    np.testing.assert_allclose(compact.mu[ix], expanded.mu, rtol=2e-7, atol=2e-8, equal_nan=True)
    assert compact.nobs == expanded.nobs == fw[compact.sample_mask].sum()
    assert compact.df_absorbed == expanded.df_absorbed
    assert compact.df_resid == expanded.df_resid
    assert compact.sample_mask[-4] and not compact.sample_mask[-3]
    assert compact.n_singletons == expanded.n_singletons == 1
    assert compact.n_separated == 2 and expanded.n_separated == 5
    assert compact.weight_type == 'fweight'
    assert compact.loglike == pytest.approx(expanded.loglike, rel=2e-8)
    audit = compact.diagnostics['sample_policy']
    assert audit['mode'] == 'unified'
    assert audit['statistical_rows'] == audit['working_rows'] == 161
    assert not audit['computational_elimination']


@pytest.mark.parametrize('family', ['ppml','ivppml'])
def test_compatibility_initial_row_count_and_combined_reporting(family):
    y, c, d, z, g, h, fw = _fixture()
    cfg = PPMLConfig() if family == 'ppml' else IVPPMLConfig()
    kwargs = dict(y=y, absorb=[g,h], weights=fw, weight_type='fw', config=cfg)
    r = (ppmlhdfe(x=c, **kwargs) if family == 'ppml' else
         ivppmlhdfe(exog=c, endog=d, instruments=z, **kwargs))
    assert not r.sample_mask[-4:].any()
    assert r.n_singletons == 4 and r.n_separated == 0
    audit = r.diagnostics['sample_policy']
    assert audit['initial_count'] == 'rows' and audit['post_separation_count'] == 'frequency'


@pytest.mark.parametrize('mode', ['stata_compat','unified'])
def test_post_separation_recheck_uses_user_frequency_not_irls(monkeypatch, mode):
    from types import SimpleNamespace
    import econhdfe.models.ppml.separation as sep
    y = np.array([2.,0.,3.,0.,1.,2.])
    fw = np.array([5.,1.,1.,1.,1.,1.])
    plan = FEPlan.from_arrays([np.repeat(np.arange(3),2)])
    cfg = PPMLConfig(singleton_mode=mode, separation=('simplex',))
    assert not initial_sample_selection(y,plan,cfg,weights=fw,weight_kind='fweight').dropped.any()
    monkeypatch.setattr(sep,'mixed_simplex_separation',lambda *a,**k:
                        SimpleNamespace(separated=y==0,iterations=1))
    result = sep.detect_separation(y,np.arange(6.)[:,None],plan,fw,cfg,frequency_weights=True,
                                   execution_config=ExecutionConfig(memory_budget_mb=512))
    assert np.flatnonzero(result.separated).tolist() == [1,3]
    assert np.flatnonzero(result.singletons).tolist() == [2]


def test_singleton_switch_and_zero_fe_switch_are_independent_only_in_unified():
    y = np.array([3.,0.,0.,1.,2.])
    plan = FEPlan.from_arrays([np.array([0,1,1,2,2])])
    for mode, expected in [('stata_compat', []), ('unified', [1,2])]:
        r = initial_sample_selection(y,plan,PPMLConfig(singleton_mode=mode,drop_singletons=False))
        assert np.flatnonzero(r.dropped).tolist() == expected
        assert not r.singletons.any()


def test_iv_unified_does_not_apply_ppml_regressor_separation():
    y = np.array([1.,3.,0.,0.])
    x = np.array([0.,0.,1.,1.])
    z = np.array([0.,4.,3.,5.])
    cfg = IVPPMLConfig(singleton_mode='unified', tolerance=1e-11)
    assert cfg.separation is None and cfg.separation_methods == ('fe',)
    r = ivppmlhdfe(y=y,endog=x,instruments=z,config=cfg)
    assert r.sample_mask.all() and r.n_separated == 0
    np.testing.assert_allclose(r.params,0,atol=1e-9)
    np.testing.assert_allclose(r.mu,1,atol=1e-9)
    np.testing.assert_allclose(np.array([np.sum(y-r.mu), z@(y-r.mu)]),0,atol=1e-9)
    for methods in [('simplex',), ('relu',), ('mu',)]:
        with pytest.raises(ValueError,match='separation|threshold'):
            IVPPMLConfig(singleton_mode='unified',separation=methods).validate()
    # Replacing sample policy never enables an unsafe IV existence heuristic.
    from dataclasses import replace
    assert replace(cfg, singleton_mode='stata_compat').separation_methods == ('fe',)
    assert replace(IVPPMLConfig(), singleton_mode='unified').separation_methods == ('fe',)
    with pytest.raises(ValueError, match='separation'):
        replace(IVPPMLConfig(separation=('fe','relu')), singleton_mode='unified').validate()


@pytest.mark.parametrize('mode', ['stata_compat','unified'])
def test_probability_weights_are_not_counts_and_are_scale_invariant(mode):
    y, c, _, _, g, h, _ = _fixture()
    w = np.linspace(.2,2.,len(y))
    cfg = PPMLConfig(singleton_mode=mode)
    a = ppmlhdfe(y=y,x=c,absorb=[g,h],weights=w,weight_type='pw',config=cfg)
    b = ppmlhdfe(y=y,x=c,absorb=[g,h],weights=100*w,weight_type='pw',config=cfg)
    np.testing.assert_array_equal(a.sample_mask,b.sample_mask)
    np.testing.assert_allclose(a.params,b.params,rtol=2e-7,atol=1e-9)
    np.testing.assert_allclose(a.vcov,b.vcov,rtol=2e-7,atol=1e-9)
    assert a.nobs == b.nobs == 160
    assert not a.sample_mask[-4]


def test_unified_slope_only_zero_row_is_not_a_singleton_direction():
    import pandas as pd
    from econhdfe import FixedEffect
    frame = pd.DataFrame(dict(y=[1.,2.,2.,4.,3.],x=[.2,.1,.3,.2,.7],
                              g=[0,0,1,1,2],t=[1.,2.,1.,2.,0.]))
    r = olshdfe(frame,y='y',x=['x'],absorb=[FixedEffect('g',slopes=('t',),intercept=False)],
                hdfe_config=HDFEConfig(singleton_mode='unified'),method='lsmr',dof_method='exact')
    D = np.column_stack([frame.x,(frame.g==0)*frame.t,(frame.g==1)*frame.t])
    expected = np.linalg.lstsq(D,frame.y,rcond=None)[0][0]
    assert r.nobs == 5 and r.dropped_singletons == 0
    np.testing.assert_allclose(r.params,[expected],rtol=1e-7,atol=1e-8)
    assert r.estimator_info['singleton_mode'] == 'unified'


@pytest.mark.parametrize('kind', ['fw','aw','pw'])
def test_linear_unified_retains_frequency_semantics(kind):
    y,c,d,z,g,h,fw = _fixture()
    r = olshdfe(y=y,x=c,absorb=[g,h],weights=fw,weight_type=kind,
                hdfe_config=HDFEConfig(singleton_mode='unified'))
    assert r.dropped_singletons == (1 if kind=='fw' else 2)
    assert r.estimator_info['singleton_mode']=='unified'


def test_policy_rejects_unknown_mode_and_uncertified_mu():
    for cfg in [HDFEConfig(singleton_mode='typo'),PPMLConfig(singleton_mode='typo'),
                PPMLConfig(singleton_mode='unified',separation=('mu',))]:
        with pytest.raises(ValueError): cfg.validate()


def test_ppml_frequency_validation_precedes_dropping_singletons():
    y,c,_,_,g,h,fw=_fixture()
    fw=fw.astype(float);fw[-4]=1.5
    with pytest.raises(ValueError,match='integer-valued'):
        ppmlhdfe(y=y,x=c,absorb=[g,h],weights=fw,weight_type='fw')
    with pytest.raises(ValueError,match='not aweight'):
        ppmlhdfe(y=y,x=c,absorb=[g,h],weights=np.ones(len(y)),weight_type='aw')


@pytest.mark.parametrize('family,engine,vce', [
    ('ppml','replica','robust'), ('ppml','optimized','cluster'),
    ('ivppml','replica','cluster'), ('ivppml','optimized','robust'),
])
def test_typed_weights_and_policy_reach_structured_reusable_models(family, engine, vce):
    import pandas as pd
    from econhdfe import PPMLHDFE, IVPPMLHDFE, factor, reg_interaction
    y,c,d,z,g,h,fw = _fixture()
    block = np.minimum(g // 5, 3)
    h_local = h + 4*block
    frame = pd.DataFrame(dict(y=y,c=c,d=d,z=z,g=g,h=h_local,b=block,w=fw,cl=h))
    cfg_type = PPMLConfig if family == 'ppml' else IVPPMLConfig
    cfg = cfg_type(singleton_mode='unified',engine=engine,tolerance=1e-10,target_inner_tol=1e-11)
    def interaction(name):
        return reg_interaction(factor('b',drop_base=False),name)
    def dense(a):
        return np.column_stack([a*(block==i) for i in range(4)])
    kwargs = dict(y=y,absorb=[g,h_local],weights=fw,weight_type='fw',config=cfg,
                  vce=vce,cluster=[h] if vce=='cluster' else None)
    model_type = PPMLHDFE if family == 'ppml' else IVPPMLHDFE
    model = model_type(frame,absorb=['g','h'],config=cfg)
    fitkw = dict(weights='w',weight_type='fw',vce=vce,cluster='cl' if vce=='cluster' else None)
    if family == 'ppml':
        expected = ppmlhdfe(x=dense(c),**kwargs)
        actual = model.fit('y',[interaction('c')],**fitkw)
    else:
        expected = ivppmlhdfe(exog=dense(c),endog=dense(d),instruments=dense(z),**kwargs)
        actual = model.fit('y',exog=[interaction('c')],endog=[interaction('d')],
                           instruments=[interaction('z')],**fitkw)
    np.testing.assert_allclose(actual.params,expected.params,rtol=3e-7,atol=3e-8)
    np.testing.assert_allclose(actual.vcov,expected.vcov,rtol=4e-6,atol=3e-8)
    np.testing.assert_array_equal(actual.sample_mask,expected.sample_mask)
    assert actual.diagnostics['heterogeneous_spec_path']
    assert actual.nobs == expected.nobs
    assert actual.n_singletons == expected.n_singletons == 1
    assert actual.n_separated == expected.n_separated == 2


@pytest.mark.parametrize('family,vce', [('ols','unadjusted'), ('ols','robust'),
                                       ('iv','robust'), ('iv','cluster')])
def test_linear_frequency_expansion_and_session_use_same_policy(family,vce):
    import pandas as pd
    from econhdfe import OLSHDFESession, IVHDFESession
    y,c,d,z,g,h,fw = _fixture()
    frame = pd.DataFrame(dict(y=y,c=c,d=d,z=z,g=g,h=h,fw=fw))
    ix = np.repeat(np.arange(len(frame)),fw)
    cfg = HDFEConfig(singleton_mode='unified')
    estimator = olshdfe if family=='ols' else ivhdfe
    spec = dict(y='y',absorb=['g','h'])
    spec.update(dict(x=['c','d']) if family=='ols' else dict(exog=['c'],endog=['d'],instruments=['z']))
    common = dict(vce=vce,cluster='g' if vce=='cluster' else None,hdfe_config=cfg)
    compact = estimator(frame,weights='fw',weight_type='fw',**spec,**common)
    expanded = estimator(frame.iloc[ix].reset_index(drop=True),**spec,**common)
    session_type = OLSHDFESession if family=='ols' else IVHDFESession
    cached = session_type(frame,weights='fw',weight_type='fw',**common).fit(**spec)
    for actual in (expanded,cached):
        np.testing.assert_allclose(actual.params,compact.params,rtol=2e-8,atol=2e-9)
        np.testing.assert_allclose(actual.vcov,compact.vcov,rtol=2e-7,atol=2e-9)
        assert actual.nobs == compact.nobs
        assert actual.df_absorbed == compact.df_absorbed
        assert actual.df_resid == compact.df_resid
        assert actual.estimator_info['singleton_mode']=='unified'


def test_session_policy_change_partitions_memory_and_disk_cache(tmp_path):
    import pandas as pd
    from econhdfe import OLSHDFESession
    y,c,d,z,g,h,fw = _fixture()
    frame = pd.DataFrame(dict(y=y,c=c,g=g,h=h,w=fw))
    session = OLSHDFESession(frame,weights='w',weight_type='fw').enable_persistent_cache(tmp_path/'cache')
    old = session.fit(y='y',x=['c'],absorb=['g','h'])
    old_key = session._entry_key(('g','h'))
    entry = session._entries[old_key]
    hits = session.persistent_cache_info()['result_hits']
    session.hdfe = HDFEConfig(singleton_mode='unified')
    new = session.fit(y='y',x=['c'],absorb=['g','h'])
    reference = olshdfe(frame,y='y',x=['c'],absorb=['g','h'],weights='w',weight_type='fw',
                       hdfe_config=session.hdfe)
    np.testing.assert_allclose(new.params,reference.params,rtol=1e-8,atol=1e-9)
    assert old.estimator_info['singleton_mode']=='stata_compat'
    assert new.estimator_info['singleton_mode']=='unified'
    assert session.persistent_cache_info()['result_hits']==hits
    assert session.cache_info()['fe_entries']==2
    assert session._entries[old_key] is entry
    assert session._entries[session._entry_key(('g','h'))] is not entry


def test_unified_group_individual_scope_is_explicit():
    with pytest.raises(ValueError,match='group'):
        olshdfe(y=np.arange(4.),x=np.arange(4.),absorb=[],group='group',
                hdfe_config=HDFEConfig(singleton_mode='unified'))


@pytest.mark.parametrize('family',['ppml','ivppml'])
def test_unified_frequency_against_independent_dense_moment_root_and_sandwich(family):
    from scipy.optimize import root
    y,c,d,z,g,h,fw = _fixture()
    # Hand-specified support, not the production sample peeler.
    mask = np.ones(len(y),dtype=bool)
    mask[-3:] = False
    ys, cs, ds, zs, gs, hs, fs = [a[mask] for a in (y,c,d,z,g,h,fw)]
    D = np.column_stack([gs==j for j in np.unique(gs)] +
                        [hs==j for j in np.unique(hs)[1:]]).astype(float)
    A = np.column_stack([cs,ds,D])
    Q = A if family=='ppml' else np.column_stack([cs,zs,D])
    assert np.linalg.matrix_rank(A)==A.shape[1]==Q.shape[1]
    def moments(theta):
        return Q.T@(fs*(ys-np.exp(A@theta)))
    def jacobian(theta):
        return -Q.T@((fs*np.exp(A@theta))[:,None]*A)
    start = np.zeros(A.shape[1])
    start[2:2+len(np.unique(gs))] = np.log(np.average(ys,weights=fs))
    solved = root(moments,start,jac=jacobian,tol=1e-11)
    assert solved.success and np.max(np.abs(moments(solved.x)))<1e-8
    mu = np.exp(A@solved.x)
    Jinv = np.linalg.inv(-jacobian(solved.x))
    S = Q*(np.sqrt(fs)*(ys-mu))[:,None]
    N = fs.sum()
    # Both Poisson families default to the upstream asymptotic N/(N-1)
    # convention; PPML's residual-DoF correction is now an explicit option.
    correction = N/(N-1)
    oracle_vcov = correction*Jinv@(S.T@S)@Jinv.T
    cfgtype = PPMLConfig if family=='ppml' else IVPPMLConfig
    kw = dict(y=y,absorb=[g,h],weights=fw,weight_type='fw',vce='robust',
              config=cfgtype(singleton_mode='unified',tolerance=1e-11,target_inner_tol=1e-12,
                             dof_method='exact'))
    actual = (ppmlhdfe(x=np.column_stack([c,d]),**kw) if family=='ppml' else
              ivppmlhdfe(exog=c,endog=d,instruments=z,**kw))
    np.testing.assert_allclose(actual.params[:2],solved.x[:2],rtol=3e-7,atol=2e-8)
    np.testing.assert_allclose(actual.vcov[:2,:2],oracle_vcov[:2,:2],rtol=3e-6,atol=2e-9)
    np.testing.assert_allclose(actual.mu[mask],mu,rtol=3e-7,atol=2e-8)


def test_sample_policy_truth_table():
    from econhdfe.config import SamplePolicy
    compat, unified = HDFEConfig().sample_policy, PPMLConfig(singleton_mode='unified').sample_policy
    assert compat == SamplePolicy('stata_compat') and unified == SamplePolicy('unified')
    for policy, u in ((compat, False), (unified, True)):
        assert policy.unified is u
        assert policy.requires_positive_user_weights is u and policy.slope_only_eligibility is u
        assert policy.disjoint_reasons is u and policy.allows_group_individual is (not u)
        assert policy.profile == ('frequency-invariant-v1' if u else 'stata-source-20261027')
        assert policy.initial_frequency_count('fweight') is u
        assert policy.initial_frequency_count('pweight') is False
        assert policy.psd_frequency_scale('fweight') is u
        assert policy.psd_frequency_scale('pweight') is False and policy.psd_frequency_scale('aweight') is False
        for drop in (True, False):
            assert policy.fe_zero_peeling(('fe', 'relu'), drop) is (u or drop)
            assert policy.fe_zero_peeling(('relu',), drop) is False


@pytest.mark.parametrize('change', [dict(dof_method='none'), dict(solver='lsmr')])
def test_session_config_change_never_reuses_stale_fe_entry(change):
    import pandas as pd
    from dataclasses import replace
    from econhdfe import OLSHDFESession
    y, c, d, z, g, h, fw = _fixture()
    frame = pd.DataFrame(dict(y=y, c=c, g=g, h=h, gh=g * 10 + h % 2))
    absorb = ['g', 'h', 'gh']
    session = OLSHDFESession(frame)
    session.fit(y='y', x=['c'], absorb=absorb)
    entry = session._entries[session._entry_key(tuple(absorb))]
    session.hdfe = replace(session.hdfe, **change)
    got = session.fit(y='y', x=['c'], absorb=absorb)
    assert session._entries[session._entry_key(tuple(absorb))] is not entry
    reference = olshdfe(frame, y='y', x=['c'], absorb=absorb, hdfe_config=session.hdfe)
    assert got.df_absorbed == reference.df_absorbed
    assert got.absorb_info['solver_selection'] == reference.absorb_info['solver_selection']
    np.testing.assert_allclose(got.params, reference.params, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(got.vcov, reference.vcov, rtol=1e-7, atol=1e-12)
