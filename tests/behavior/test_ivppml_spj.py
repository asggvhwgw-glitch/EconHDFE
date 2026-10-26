from __future__ import annotations
import numpy as np
import pandas as pd
from econhdfe import IVPPMLConfig, SPJPanel, ivppml_spj, ivppml_spj_bootstrap


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
    assert np.all(np.isfinite(b.params)) and np.all(np.isfinite(c.params))
