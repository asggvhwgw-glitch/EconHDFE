import numpy as np
import statsmodels.api as sm
from pyreghdfe import ppmlhdfe, PPMLConfig
from pyreghdfe.fe_plan import FEPlan, fe_separated
from pyreghdfe.ppml.separation_relu import relu_separation


def cfg(**kw):
    d = dict(standardize=True, separation=(), tolerance=1e-9, target_inner_tol=1e-10)
    d.update(kw)
    return PPMLConfig(**d)


def test_no_fe_matches_statsmodels_poisson():
    rng=np.random.default_rng(1); n=1500
    X=rng.normal(size=(n,2)); b=np.array([0.35,-0.22]); c=0.4
    mu=np.exp(c+X@b); y=rng.poisson(mu)
    ours=ppmlhdfe(y,X,names=('x1','x2'),vce='model',config=cfg())
    ref=sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit()
    # ours orders explicit x then _cons
    assert np.allclose(ours.coef[:2], ref.params[1:], atol=2e-7, rtol=2e-7)
    assert np.allclose(ours.coef[2], ref.params[0], atol=2e-7, rtol=2e-7)


def test_one_fe_matches_explicit_dummy_slopes():
    rng=np.random.default_rng(2); G=35; m=35; n=G*m
    g=np.repeat(np.arange(G),m); X=rng.normal(size=(n,2))
    a=rng.normal(scale=.35,size=G); b=np.array([.25,-.15])
    y=rng.poisson(np.exp(a[g]+X@b))
    ours=ppmlhdfe(y,X,absorb=[g],names=('x1','x2'),vce='model',config=cfg())
    D=np.eye(G)[g][:,1:]
    ref=sm.GLM(y,np.column_stack([np.ones(n),X,D]),family=sm.families.Poisson()).fit(maxiter=200)
    assert np.allclose(ours.coef, ref.params[1:3], atol=2e-6, rtol=2e-6)


def test_two_way_optimized_matches_replica():
    rng=np.random.default_rng(3); n=5000
    g1=rng.integers(0,120,n); g2=rng.integers(0,45,n); X=rng.normal(size=(n,3))
    a=rng.normal(scale=.2,size=120); d=rng.normal(scale=.2,size=45); b=np.array([.12,-.18,.08])
    y=rng.poisson(np.exp(a[g1]+d[g2]+X@b))
    r=ppmlhdfe(y,X,absorb=[g1,g2],config=cfg(engine='replica'),vce='model')
    o=ppmlhdfe(y,X,absorb=[g1,g2],config=cfg(engine='optimized'),vce='model')
    assert np.allclose(r.coef,o.coef,atol=5e-8,rtol=5e-8)


def test_fe_only_separation():
    y=np.array([0,0,0,1,2,3.],float); g=np.array([0,0,1,1,2,2])
    s=fe_separated(y,FEPlan.from_arrays([g]))
    assert s.tolist()==[True,True,False,False,False,False]


def test_relu_primer_detects_only_zero_outcomes():
    y=np.array([0,1,0,0,1.]); id1=np.array([1,1,2,2,2]); id2=np.array([1,1,1,2,2])
    info=relu_separation(y,np.empty((5,0)),FEPlan.from_arrays([id1,id2]),tol=1e-5,max_iter=100)
    assert info.converged
    assert not np.any(info.separated[y>0])
    assert np.any(info.separated[y==0])


def test_default_separation_pipeline_matches_relu_primer():
    y=np.array([0,1,0,0,1.]); id1=np.array([1,1,2,2,2]); id2=np.array([1,1,1,2,2])
    r=ppmlhdfe(y,None,absorb=[id1,id2],vce='model',config=PPMLConfig(tolerance=1e-8,target_inner_tol=1e-10))
    assert not np.any(r.separation_mask[y>0])
    assert r.n_separated >= 1


def test_recursive_singletons_are_removed_not_labeled_separated():
    # group 2 is a singleton in g1; removing it makes another level singleton in g2.
    y=np.array([1.,2.,1.,3.]); x=np.array([[0.],[1.],[2.],[3.]])
    g1=np.array([0,0,1,2]); g2=np.array([0,1,1,2])
    plan=FEPlan.from_arrays([g1,g2])
    m=plan.singleton_mask()
    assert m.tolist()==[True,True,True,True]


def test_optimized_canonicalization_preserves_replica_results_and_dof():
    rng=np.random.default_rng(44)
    ncity, years, reps=40, 6, 8
    city=np.repeat(np.arange(ncity),years*reps)
    year=np.tile(np.repeat(np.arange(years),reps),ncity)
    province=city//5
    firm=np.arange(len(city))%(ncity*3)
    cy=city*years+year; py=province*years+year
    X=rng.normal(size=(len(city),2))
    a=rng.normal(scale=.15,size=firm.max()+1); d=rng.normal(scale=.15,size=cy.max()+1)
    y=rng.poisson(np.exp(a[firm]+d[cy]+X@np.array([.13,-.08])))
    rr=ppmlhdfe(y,X,absorb=[firm,year,py,cy],vce='robust',config=cfg(engine='replica'))
    oo=ppmlhdfe(y,X,absorb=[firm,year,py,cy],vce='robust',config=cfg(engine='optimized'))
    assert np.allclose(rr.coef,oo.coef,atol=2e-7,rtol=2e-7)
    assert np.allclose(rr.stderr,oo.stderr,atol=2e-7,rtol=2e-7)
    assert rr.df_absorbed==oo.df_absorbed
    assert oo.diagnostics['canonicalization']['changed']
    assert len(oo.diagnostics['canonicalization']['effective'])==2


def test_cluster_vce_nested_fe_is_finite_and_engine_invariant():
    rng=np.random.default_rng(55); G=50; m=25; n=G*m
    g=np.repeat(np.arange(G),m); h=rng.integers(0,20,n); X=rng.normal(size=(n,2))
    a=rng.normal(scale=.2,size=G); d=rng.normal(scale=.1,size=20)
    y=rng.poisson(np.exp(a[g]+d[h]+X@np.array([.1,-.06])))
    r=ppmlhdfe(y,X,absorb=[g,h],clusters=[g],vce='cluster',config=cfg(engine='replica'))
    o=ppmlhdfe(y,X,absorb=[g,h],clusters=[g],vce='cluster',config=cfg(engine='optimized'))
    assert np.all(np.isfinite(r.stderr))
    assert np.allclose(r.stderr,o.stderr,atol=2e-7,rtol=2e-7)
    assert r.diagnostics['dof_nested']>0


def test_mu_separation_is_explicit_post_irls_path():
    y = np.array([1., 2., 3., 2., 1., 0., 0., 0., 0.])
    x = np.r_[np.zeros(5), np.ones(4)][:, None]
    r = ppmlhdfe(
        y, x, names=["x"],
        config=PPMLConfig(separation=("mu",), max_iter=1000),
    )
    assert r.n_separated == 4
    assert r.nobs == 5
    assert r.diagnostics["separation_by_method"]["mu"] == 4
    assert "x" not in r.names

def test_compiled_projector_matches_fresh_projectors_under_changing_weights():
    rng = np.random.default_rng(71)
    n = 2000
    g1 = rng.integers(0, 90, n)
    g2 = rng.integers(0, 70, n)
    A = rng.normal(size=(n, 3))
    plan = FEPlan.from_arrays([g1, g2])
    for engine in ("replica", "optimized"):
        compiled = plan.projector(engine=engine)
        for _ in range(3):
            w = np.exp(rng.normal(scale=.4, size=n))
            got, info = compiled.residualize(A, w, tol=1e-10, return_info=True)
            expected, fresh_info = plan.residualize(A, w, tol=1e-10, engine=engine)
            assert info.converged and fresh_info.converged
            assert np.allclose(got, expected, atol=2e-9, rtol=2e-9)


def test_compiled_three_way_map_projector_matches_fresh():
    rng = np.random.default_rng(72)
    n = 2500
    groups = [rng.integers(0, m, n) for m in (80, 60, 40)]
    A = rng.normal(size=(n, 2))
    plan = FEPlan.from_arrays(groups)
    compiled = plan.projector(engine="optimized")
    for scale in (.2, .8):
        w = np.exp(rng.normal(scale=scale, size=n))
        got, _ = compiled.residualize(A, w, tol=1e-9, return_info=True)
        expected, _ = plan.residualize(A, w, tol=1e-9, engine="optimized")
        assert np.allclose(got, expected, atol=2e-8, rtol=2e-8)


def test_fe_plus_simplex_does_not_replace_relu_on_upstream_primer():
    # Upstream nonexistence primer: FE + simplex alone deliberately fails to
    # identify this joint-FE separation; ReLU is required.
    y = np.array([0., 1., 0., 0., 1.])
    id1 = np.array([1, 1, 2, 2, 2])
    id2 = np.array([1, 1, 1, 2, 2])
    r = ppmlhdfe(
        y, None, absorb=[id1, id2], vce="model",
        config=PPMLConfig(
            separation=("fe", "simplex"), tolerance=1e-8, target_inner_tol=1e-10
        ),
    )
    assert r.n_separated == 0


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
    assert np.allclose(cold.coef, warm.coef, atol=2e-8, rtol=2e-8)
    assert warm.iterations <= cold.iterations
    assert warm.diagnostics["warm_started"]


def test_ppml_streamed_vce_matches_explicit_score_formula():
    from econhdfe.models.ppml.vce import ppml_vcov
    from econhdfe.compute.vcov import score_covariance
    rng = np.random.default_rng(732)
    n, k = 1600, 3
    X = rng.normal(size=(n, k))
    mu = np.exp(rng.normal(scale=.3, size=n))
    y = rng.poisson(mu).astype(float)
    tw = np.exp(rng.normal(scale=.2, size=n))
    H = X.T @ ((tw*mu)[:, None] * X)
    bread = np.linalg.pinv(H, hermitian=True)
    score = X * (tw*(y-mu))[:, None]
    meat = score_covariance(score, kind="robust", k_total=k+5, effective_n=n)
    expected = bread @ meat @ bread
    got = ppml_vcov(X, y, mu, tw, kind="robust", df_absorbed=5)
    assert np.allclose(got, expected, atol=2e-11, rtol=2e-11)
