from __future__ import annotations
import numpy as np
import pandas as pd
from econhdfe import IVPPMLConfig, SPJPanel, ivppml_spj, ivppml_spj_bootstrap
from econhdfe.models.ppml_iv.bias import _combine_a, _combine_b, _combine_c


def cfg():
    return IVPPMLConfig(engine="optimized", separation=(), tolerance=1e-8, target_inner_tol=1e-9, max_iter=300)


def class_a_data(seed=42, N=28, T=6):
    rng=np.random.default_rng(seed)
    ids=np.repeat(np.arange(N),T); year=np.tile(np.arange(T),N); n=len(ids)
    z=rng.normal(size=n); x2=.5*rng.normal(size=n); e=rng.normal(size=n)
    x1=.85*z+e
    ai=rng.normal(scale=.15,size=N); gt=rng.normal(scale=.08,size=T)
    y=rng.poisson(np.exp(ai[ids]+gt[year]+.18*x1+.08*x2))
    return pd.DataFrame(dict(id=ids,year=year,y=y,x1=x1,x2=x2,z=z))


def test_spj_combination_formulas_vectorize():
    f=np.array([1.,2.]); a=np.array([.8,1.7]); b=np.array([.9,1.8]); c=np.array([.7,1.6])
    np.testing.assert_allclose(_combine_a(f,a,b),3*f-a-b)
    np.testing.assert_allclose(_combine_b(f,a),2*f-a)
    np.testing.assert_allclose(_combine_c(f,a,b,c),4*f-2*a-2*b+c)


def test_class_a_spj_runs_and_exposes_bias_vector():
    d=class_a_data()
    r=ivppml_spj(d,panel=SPJPanel.class_a(),y="y",exog=("x2",),endog=("x1",),instruments=("z",),config=cfg(),seed=11)
    assert r.panel_class=="A" and r.fit_count==5
    assert r.names==("x2","x1")
    np.testing.assert_allclose(r.bias, r.full_coef-r.coef)
    assert np.all(np.isfinite(r.coef))


def test_class_a_bootstrap_is_deterministic_and_tracks_failures():
    d=class_a_data(N=24,T=6)
    kw=dict(data=d,panel=SPJPanel.class_a(),y="y",exog=("x2",),endog=("x1",),instruments=("z",),config=cfg(),reps=4,seed=99,n_jobs=1)
    a=ivppml_spj_bootstrap(**kw); b=ivppml_spj_bootstrap(**kw)
    assert a.requested==4 and a.completed+a.failed==4 and a.completed>=2
    np.testing.assert_allclose(a.draws,b.draws)
    np.testing.assert_allclose(a.stderr,np.std(a.draws,axis=0,ddof=1))
    assert np.all(a.ci_low<=a.ci_high)


def gravity_data(seed=40, Nc=18, T=8):
    rng = np.random.default_rng(seed)
    rows = [(e, i, t) for e in range(Nc) for i in range(Nc) if e != i for t in range(T)]
    exp = np.array([r[0] for r in rows]); imp = np.array([r[1] for r in rows]); year = np.array([r[2] for r in rows])
    pair = exp * Nc + imp; n = len(rows)
    z = rng.normal(size=n); x2 = .3 * rng.normal(size=n)
    x1 = .85 * z + .08 * x2 + rng.normal(scale=.4, size=n)
    ey = rng.normal(scale=.04, size=Nc*T); iy = rng.normal(scale=.04, size=Nc*T)
    pfe = rng.normal(scale=.03, size=Nc*Nc)
    eta = .05 + ey[exp*T+year] + iy[imp*T+year] + pfe[pair] + .04*x2 + .12*x1
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame(dict(exp=exp, imp=imp, year=year, pair=pair, y=y, x1=x1, x2=x2, z=z))


def test_class_b_and_c_spj_run_with_expected_fit_counts():
    d = gravity_data()
    b = ivppml_spj(d, panel=SPJPanel.class_b(), y="y", exog=("x2",), endog=("x1",), instruments=("z",), config=cfg(), seed=12)
    c = ivppml_spj(d, panel=SPJPanel.class_c(), y="y", exog=("x2",), endog=("x1",), instruments=("z",), config=cfg(), seed=12)
    assert b.fit_count == 5 and c.fit_count == 15
    assert b.names == c.names == ("x2", "x1")
    assert np.all(np.isfinite(b.coef)) and np.all(np.isfinite(c.coef))


def test_bootstrap_reports_ci_implied_standard_error():
    from statistics import NormalDist
    d = class_a_data(N=24, T=6)
    r = ivppml_spj_bootstrap(
        d, panel=SPJPanel.class_a(), y="y", exog=("x2",), endog=("x1",), instruments=("z",),
        config=cfg(), reps=5, seed=123, n_jobs=1,
    )
    zcrit = NormalDist().inv_cdf(.975)
    expected = (r.ci_high - r.ci_low) / (2.0 * zcrit)
    np.testing.assert_allclose(r.stderr_ci_implied, expected)
    np.testing.assert_allclose(r.se_ci_implied, expected)
