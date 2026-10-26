import numpy as np
import statsmodels.api as sm
from econhdfe import ppmlhdfe, PPMLConfig


def cfg(**kw):
    d = dict(standardize=True, separation=(), tolerance=1e-9, target_inner_tol=1e-10)
    d.update(kw)
    return PPMLConfig(**d)


def test_no_fe_matches_statsmodels_poisson():
    rng=np.random.default_rng(1); n=1500
    X=rng.normal(size=(n,2)); b=np.array([0.35,-0.22]); c=0.4
    mu=np.exp(c+X@b); y=rng.poisson(mu)
    ours=ppmlhdfe(y=y,x=X,names=('x1','x2'),vce='model',config=cfg())
    ref=sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit()
    # ours orders explicit x then _cons
    assert np.allclose(ours.params[:2], ref.params[1:], atol=2e-7, rtol=2e-7)
    assert np.allclose(ours.params[2], ref.params[0], atol=2e-7, rtol=2e-7)


def test_default_separation_pipeline_matches_relu_primer():
    y=np.array([0,1,0,0,1.]); id1=np.array([1,1,2,2,2]); id2=np.array([1,1,1,2,2])
    r=ppmlhdfe(y=y,absorb=[id1,id2],vce='model',config=PPMLConfig(tolerance=1e-8,target_inner_tol=1e-10))
    assert not np.any(r.separation_mask[y>0])
    assert r.n_separated >= 1


def test_mu_separation_is_explicit_post_irls_path():
    y = np.array([1., 2., 3., 2., 1., 0., 0., 0., 0.])
    x = np.r_[np.zeros(5), np.ones(4)][:, None]
    r = ppmlhdfe(
        y=y, x=x, names=["x"],
        config=PPMLConfig(separation=("mu",), max_iter=1000),
    )
    assert r.n_separated == 4
    assert r.nobs == 5
    assert r.diagnostics["separation_by_method"]["mu"] == 4
    assert "x" not in r.names


def test_ppml_warm_start_preserves_solution_and_does_not_add_iterations():
    import pandas as pd
    from econhdfe import PPMLHDFE
    rng = np.random.default_rng(731)
    n = 5000
    df = pd.DataFrame({
        "g1": rng.integers(0, 140, n),
        "g2": rng.integers(0, 70, n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    a = rng.normal(scale=.2, size=140)
    b = rng.normal(scale=.15, size=70)
    eta = a[df.g1.to_numpy()] + b[df.g2.to_numpy()] + .18*df.x1.to_numpy() - .09*df.x2.to_numpy()
    df["y"] = rng.poisson(np.exp(eta))
    model = PPMLHDFE(df, absorb=["g1", "g2"], config=cfg(engine="optimized"))
    base = model.fit("y", ["x1"], vce="model")
    cold = model.fit("y", ["x1", "x2"], vce="model")
    warm = model.fit("y", ["x1", "x2"], vce="model", warm_start=base)
    assert np.allclose(cold.params, warm.params, atol=2e-8, rtol=2e-8)
    assert warm.iterations <= cold.iterations
    assert warm.diagnostics["warm_started"]
