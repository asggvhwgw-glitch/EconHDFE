import numpy as np
import pytest

from econhdfe.compute.block_design import BlockDesign, compile_block_design
from econhdfe.compute.design_plan import analyze_design_structure
from econhdfe.hdfe.block_projection import BlockWeightedFEProjector
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.heterogeneous import detect_separation_block, fit_irls_block
from econhdfe.models.ppml.config import PPMLConfig
from econhdfe.models.ppml.irls import fit_irls
from econhdfe.models.ppml.separation import detect_separation
from econhdfe.models.ppml.standardize import Standardization


def _block_dgp(seed=9001, nblock=4, n_per=700, p=3):
    rng = np.random.default_rng(seed)
    n = nblock * n_per
    block = np.repeat(np.arange(nblock), n_per)
    X = np.zeros((n, nblock * p), dtype=float)
    g1 = np.empty(n, dtype=int)
    g2 = np.empty(n, dtype=int)
    eta = np.empty(n, dtype=float)
    for b in range(nblock):
        rows = np.flatnonzero(block == b)
        local = rng.normal(size=(n_per, p))
        X[np.ix_(rows, np.arange(b*p, (b+1)*p))] = local
        a = rng.integers(0, 35, n_per)
        d = rng.integers(0, 17, n_per)
        g1[rows] = b * 35 + a
        g2[rows] = b * 17 + d
        ae = rng.normal(scale=.16, size=35)
        de = rng.normal(scale=.12, size=17)
        beta = rng.normal(scale=.08, size=p)
        eta[rows] = ae[a] + de[d] + local @ beta
    y = rng.poisson(np.exp(eta)).astype(float)
    w = np.exp(rng.normal(scale=.12, size=n))
    off = rng.normal(scale=.03, size=n)
    return y, X, g1, g2, w, off


def _cfg(**kw):
    opts = dict(
        engine="optimized", separation=(), standardize=True,
        tolerance=1e-9, target_inner_tol=1e-10,
        fast_partial=False, max_iter=500,
    )
    opts.update(kw)
    return PPMLConfig(**opts)


def test_block_irls_matches_pooled_two_way_global_convergence():
    y, X, g1, g2, w, off = _block_dgp()
    plan = FEPlan.from_arrays([g1, g2])
    std = Standardization.fit(y, X)
    ys, Xs = std.transform(y, X)
    structure = analyze_design_structure(Xs, groups=plan.groups)
    assert structure.certified_block_separable and structure.block_count == 4
    BX, _ = compile_block_design(Xs, structure=structure)
    cfg = _cfg()

    pooled_projector = plan.projector(engine="optimized")
    pooled = fit_irls(ys, Xs, plan, w, off, cfg, projector=pooled_projector)
    block_projector = BlockWeightedFEProjector(structure, plan, engine="optimized")
    blocked = fit_irls_block(
        ys, BX, plan, w, off, cfg, structure=structure, projector=block_projector
    )
    assert pooled.converged and blocked.converged
    np.testing.assert_allclose(blocked.beta, pooled.beta, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(blocked.mu, pooled.mu, atol=3e-8, rtol=3e-8)
    np.testing.assert_allclose(blocked.eta, pooled.eta, atol=3e-8, rtol=3e-8)
    np.testing.assert_allclose(blocked.deviance, pooled.deviance, atol=1e-8, rtol=1e-9)


def test_block_irls_matches_pooled_with_mu_detection():
    y, X, g1, g2, w, off = _block_dgp(seed=9002, nblock=3, n_per=500, p=2)
    # Add an exactly unsupported positive-direction regressor in the final
    # component so zero outcomes there can be driven toward mu=0.
    rows = np.arange(1000, 1500)
    y[rows[:120]] = 0.0
    X[rows[:120], -1] = 1.0
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg(separation=("mu",), max_iter=800)
    pooled = fit_irls(y, X, plan, w, off, cfg, detect_mu=True, projector=plan.projector(engine="optimized"))
    blocked = fit_irls_block(y, BX, plan, w, off, cfg, structure=structure, detect_mu=True)
    assert pooled.converged == blocked.converged
    np.testing.assert_array_equal(blocked.separated, pooled.separated)
    np.testing.assert_allclose(blocked.mu, pooled.mu, atol=2e-7, rtol=2e-7)


def test_component_separation_matches_pooled_fe_simplex_relu():
    # Repeat the upstream ReLU primer in disconnected components and add a
    # local regressor so every component is represented in the BlockDesign.
    y0 = np.array([0., 1., 0., 0., 1.])
    a0 = np.array([1, 1, 2, 2, 2])
    b0 = np.array([1, 1, 1, 2, 2])
    copies = 3
    y = np.tile(y0, copies)
    n = len(y)
    X = np.zeros((n, copies), dtype=float)
    g1 = np.empty(n, dtype=int)
    g2 = np.empty(n, dtype=int)
    for c in range(copies):
        rows = np.arange(c*5, (c+1)*5)
        X[rows, c] = np.linspace(.5, 1.0, 5)
        g1[rows] = a0 + c*10
        g2[rows] = b0 + c*10
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg(separation=("fe", "simplex", "relu"), relu_max_iter=200)
    pooled = detect_separation(y, X, plan, np.ones(n), cfg)
    blocked = detect_separation_block(y, BX, plan, np.ones(n), cfg)
    np.testing.assert_array_equal(blocked.separated, pooled.separated)
    assert blocked.by_method == pooled.by_method


def test_component_separation_skips_positive_component_without_affecting_other_blocks():
    rng = np.random.default_rng(9003)
    y = np.r_[np.ones(8), np.array([0., 0., 1., 2., 0., 1., 2., 1.])]
    X = np.zeros((16, 2), dtype=float)
    X[:8, 0] = rng.normal(size=8)
    X[8:, 1] = rng.normal(size=8)
    g = np.r_[np.repeat([0, 1], 4), np.repeat([10, 11], 4)]
    plan = FEPlan.from_arrays([g])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg(separation=("fe", "simplex"))
    pooled = detect_separation(y, X, plan, np.ones(16), cfg)
    blocked = detect_separation_block(y, BX, plan, np.ones(16), cfg)
    np.testing.assert_array_equal(blocked.separated, pooled.separated)
    assert blocked.by_method == pooled.by_method


def test_block_final_wls_and_robust_vce_match_pooled():
    from econhdfe.compute.linalg import weighted_lstsq
    from econhdfe.models.ppml.heterogeneous import final_wls_block, ppml_vcov_block
    from econhdfe.models.ppml.vce import ppml_vcov
    y, X, g1, g2, w, off = _block_dgp(seed=9010, nblock=4, n_per=650, p=3)
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg()
    pooled_projector = plan.projector(engine="optimized")
    state = fit_irls(y, X, plan, w, off, cfg, projector=pooled_projector)

    # Established pooled final solve.
    wf = w * state.mu
    z = state.eta - off - 1.0
    pos = y > 0
    z = z.copy(); z[pos] += y[pos] / state.mu[pos]
    dense = np.column_stack([z, X])
    dense, _ = pooled_projector.residualize(dense, wf, tol=cfg.target_inner_tol, return_info=True, copy=False)
    beta_dense, _ = weighted_lstsq(dense[:, 1:], dense[:, 0], wf, fast=False)

    beta_block, _, Xt_block, _ = final_wls_block(
        y, BX, plan, w, off, state, cfg, structure=structure
    )
    np.testing.assert_allclose(beta_block, beta_dense, atol=2e-9, rtol=2e-9)
    V_dense = ppml_vcov(dense[:, 1:], y, state.mu, w, kind="robust", df_absorbed=7)
    V_block = ppml_vcov_block(Xt_block, y, state.mu, w, kind="robust", df_absorbed=7)
    np.testing.assert_allclose(V_block, V_dense, atol=3e-9, rtol=3e-9)


def test_block_cluster_vce_keeps_cross_component_cluster_covariance():
    from econhdfe.models.ppml.heterogeneous import final_wls_block, ppml_vcov_block
    from econhdfe.models.ppml.vce import ppml_vcov
    y, X, g1, g2, w, off = _block_dgp(seed=9011, nblock=4, n_per=600, p=2)
    n = len(y)
    # Deliberately reuse the same cluster ids in every structural component.
    # This forces nonzero off-block cluster meat terms.
    cluster = np.arange(n) % 47
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg()
    state = fit_irls_block(y, BX, plan, w, off, cfg, structure=structure)
    _, _, Xt_block, wf = final_wls_block(y, BX, plan, w, off, state, cfg, structure=structure)
    Xt_dense = Xt_block.materialize()
    V_dense = ppml_vcov(
        Xt_dense, y, state.mu, w, kind="cluster", clusters=[cluster],
        df_absorbed=9, nested_adj=0,
    )
    V_block = ppml_vcov_block(
        Xt_block, y, state.mu, w, kind="cluster", clusters=[cluster],
        df_absorbed=9, nested_adj=0,
    )
    np.testing.assert_allclose(V_block, V_dense, atol=5e-10, rtol=5e-10)
    # Ensure this test really exercises an off-block term.
    assert np.max(np.abs(V_dense[:2, 2:])) > 1e-10


def test_block_two_way_cluster_vce_matches_dense_with_spanning_clusters():
    from econhdfe.models.ppml.heterogeneous import final_wls_block, ppml_vcov_block
    from econhdfe.models.ppml.vce import ppml_vcov
    y, X, g1, g2, w, off = _block_dgp(seed=9012, nblock=3, n_per=550, p=2)
    n = len(y)
    c1 = np.arange(n) % 31
    c2 = (np.arange(n) // 5) % 23
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg()
    state = fit_irls_block(y, BX, plan, w, off, cfg, structure=structure)
    _, _, Xt_block, _ = final_wls_block(y, BX, plan, w, off, state, cfg, structure=structure)
    V_dense = ppml_vcov(
        Xt_block.materialize(), y, state.mu, w, kind="cluster", clusters=[c1, c2],
        df_absorbed=8, nested_adj=0,
    )
    V_block = ppml_vcov_block(
        Xt_block, y, state.mu, w, kind="cluster", clusters=[c1, c2],
        df_absorbed=8, nested_adj=0,
    )
    np.testing.assert_allclose(V_block, V_dense, atol=8e-10, rtol=8e-10)


def test_block_rank_filter_can_remove_entire_component_without_losing_rows():
    from econhdfe.models.ppml.heterogeneous import filter_columns_block
    rng = np.random.default_rng(9013)
    m = 240
    n = 2*m
    X = np.zeros((n, 2))
    # Block 0 survives FE projection.
    X[:m, 0] = rng.normal(size=m)
    # Block 1 is exactly an FE-level value and is fully absorbed.
    local_g = np.repeat(np.arange(12), m//12)
    X[m:, 1] = (local_g + 1).astype(float)
    g = np.r_[np.repeat(np.arange(12), m//12), 100 + local_g]
    plan = FEPlan.from_arrays([g])
    structure = analyze_design_structure(X, groups=plan.groups)
    assert structure.certified_block_separable and structure.block_count == 2
    BX, _ = compile_block_design(X, structure=structure)
    filtered, names, original = filter_columns_block(
        BX, ["survive", "absorbed"], plan, np.ones(n), 1e-10, engine="replica"
    )
    assert names == ("survive",)
    np.testing.assert_array_equal(original, [0])
    assert filtered.ncols == 1 and len(filtered.blocks) == 2
    assert filtered.blocks[1].values.shape == (m, 0)
    # The response-only component must still be projected and returned.
    projector = BlockWeightedFEProjector.from_design(filtered, plan, engine="replica")
    out = projector.residualize_response_design(
        rng.normal(size=n), filtered, np.ones(n), tol=1e-10
    )
    assert out.response.shape == (n,)
    assert out.design.blocks[1].values.shape == (m, 0)


def test_block_filtered_irls_matches_dense_when_one_component_has_no_surviving_x():
    from econhdfe.models.ppml.heterogeneous import filter_columns_block
    from econhdfe.models.ppml.estimator import _filter_columns
    rng = np.random.default_rng(9014)
    m = 360; n = 2*m
    g0 = np.repeat(np.arange(18), m//18)
    g = np.r_[g0, 100 + g0]
    X = np.zeros((n, 2))
    X[:m, 0] = rng.normal(size=m)
    X[m:, 1] = (g0 + 1).astype(float)
    fe = rng.normal(scale=.15, size=18)
    eta = np.r_[fe[g0] + .12*X[:m, 0], fe[g0]]
    y = rng.poisson(np.exp(eta)).astype(float)
    w = np.ones(n); off = np.zeros(n)
    plan = FEPlan.from_arrays([g])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg(engine="replica")
    Bd, bnames, _ = filter_columns_block(BX, ["x0", "x1"], plan, w, cfg.target_inner_tol, engine="replica")
    Xd, dnames, _ = _filter_columns(X, ["x0", "x1"], plan, w, cfg.target_inner_tol, engine="replica")
    assert bnames == dnames == ("x0",)
    dense = fit_irls(y, Xd, plan, w, off, cfg, projector=plan.projector(engine="replica"))
    blocked = fit_irls_block(y, Bd, plan, w, off, cfg, structure=structure)
    assert dense.converged and blocked.converged
    np.testing.assert_allclose(blocked.beta, dense.beta, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(blocked.mu, dense.mu, atol=2e-8, rtol=2e-8)


def test_component_separation_trim_rebuilds_block_design_without_dense_global_x():
    from econhdfe.models.ppml.heterogeneous import filter_columns_block
    rng = np.random.default_rng(9015)
    m = 180; n = 3*m
    X = np.zeros((n, 3))
    g = np.empty(n, dtype=int)
    y = np.empty(n, dtype=float)
    for b in range(3):
        rows = np.arange(b*m, (b+1)*m)
        X[rows, b] = rng.normal(size=m)
        loc = np.repeat(np.arange(12), m//12)
        g[rows] = b*100 + loc
        if b == 1:
            # Entire middle FE component has zero outcome and is FE-separated.
            y[rows] = 0.0
        else:
            a = rng.normal(scale=.12, size=12)
            y[rows] = rng.poisson(np.exp(a[loc] + .1*X[rows, b])).astype(float)
    plan = FEPlan.from_arrays([g])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    cfg = _cfg(engine="replica", separation=("fe",))
    sep = detect_separation_block(y, BX, plan, np.ones(n), cfg)
    assert np.all(sep.separated[m:2*m])
    survivor = ~sep.separated
    Bs = BX.subset_rows(survivor)
    ps = plan.subset(survivor)
    # The removed component's now-zero coefficient column is dropped by the
    # ordinary rank contract; remaining component rows are retained.
    Bf, names, keep = filter_columns_block(
        Bs, ["b0", "b1", "b2"], ps, np.ones(np.sum(survivor)), 1e-10,
        engine="replica",
    )
    assert names == ("b0", "b2")
    np.testing.assert_array_equal(keep, [0, 2])
    assert Bf.nobs == 2*m and Bf.ncols == 2
    out = fit_irls_block(
        y[survivor], Bf, ps, np.ones(2*m), np.zeros(2*m),
        _cfg(engine="replica"), structure=structure,
    )
    assert out.converged and len(out.mu) == 2*m


def _assert_result_parity(a, b, *, coef_tol=3e-8, se_tol=5e-8):
    assert a.names == b.names
    np.testing.assert_allclose(a.coef, b.coef, atol=coef_tol, rtol=coef_tol)
    np.testing.assert_allclose(a.stderr, b.stderr, atol=se_tol, rtol=se_tol)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=se_tol**2 * 10, rtol=2e-7)
    np.testing.assert_array_equal(a.sample_mask, b.sample_mask)
    np.testing.assert_array_equal(a.separation_mask, b.separation_mask)
    np.testing.assert_allclose(a.mu, b.mu, atol=2e-7, rtol=2e-7, equal_nan=True)
    np.testing.assert_allclose(a.eta, b.eta, atol=2e-7, rtol=2e-7, equal_nan=True)
    assert a.nobs == b.nobs and a.n_separated == b.n_separated
    assert a.df_absorbed == b.df_absorbed and a.df_resid == b.df_resid
    np.testing.assert_allclose(a.loglike, b.loglike, atol=2e-7, rtol=2e-9)
    np.testing.assert_allclose(a.loglike_null, b.loglike_null, atol=2e-7, rtol=2e-9)
    np.testing.assert_allclose(a.pseudo_r2, b.pseudo_r2, atol=2e-9, rtol=2e-9)


def test_end_to_end_private_block_model_vce_matches_pooled_fit_arrays():
    from econhdfe.models.ppml.heterogeneous import fit_block_arrays
    from econhdfe.models.ppml.estimator import fit_arrays
    y, X, g1, g2, w, off = _block_dgp(seed=9020, nblock=4, n_per=650, p=3)
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    config = _cfg(engine="optimized", separation=())
    names = tuple(f"x{i}" for i in range(X.shape[1]))
    dense = fit_arrays(y, X, plan, offset=off, true_w=w, vce="model", clusters=None,
                       names=names, config=config)
    block = fit_block_arrays(
        y, BX, plan, offset=off, true_w=w, vce="model", clusters=None,
        names=names, config=config,
    )
    _assert_result_parity(block, dense)


def test_end_to_end_private_block_cluster_cross_component_matches_pooled():
    from econhdfe.models.ppml.heterogeneous import fit_block_arrays
    from econhdfe.models.ppml.estimator import fit_arrays
    y, X, g1, g2, w, off = _block_dgp(seed=9021, nblock=3, n_per=700, p=2)
    n = len(y)
    c1 = np.arange(n) % 41
    c2 = (np.arange(n)//7) % 29
    plan = FEPlan.from_arrays([g1, g2])
    structure = analyze_design_structure(X, groups=plan.groups)
    BX, _ = compile_block_design(X, structure=structure)
    config = _cfg(engine="replica", separation=())
    names = tuple(f"x{i}" for i in range(X.shape[1]))
    dense = fit_arrays(y, X, plan, offset=off, true_w=w, vce="cluster", clusters=[c1,c2],
                       names=names, config=config)
    block = fit_block_arrays(
        y, BX, plan, offset=off, true_w=w, vce="cluster", clusters=[c1,c2],
        names=names, config=config,
    )
    _assert_result_parity(block, dense, se_tol=1e-7)


def test_end_to_end_private_block_fe_separation_removes_whole_component():
    from econhdfe.models.ppml.heterogeneous import fit_block_arrays
    from econhdfe.models.ppml.estimator import fit_arrays
    rng = np.random.default_rng(9022)
    B=3; m=300; n=B*m
    X=np.zeros((n,B)); g=np.empty(n,dtype=int); y=np.empty(n)
    for b in range(B):
        rows=np.arange(b*m,(b+1)*m); loc=np.repeat(np.arange(15),m//15)
        g[rows]=b*100+loc; X[rows,b]=rng.normal(size=m)
        if b==1:
            y[rows]=0.0
        else:
            a=rng.normal(scale=.1,size=15); y[rows]=rng.poisson(np.exp(a[loc]+.08*X[rows,b]))
    w=np.ones(n); off=np.zeros(n); plan=FEPlan.from_arrays([g])
    structure=analyze_design_structure(X,groups=plan.groups); BX,_=compile_block_design(X,structure=structure)
    config=_cfg(engine="replica", separation=("fe",))
    names=("b0","b1","b2")
    dense=fit_arrays(y,X,plan,offset=off,true_w=w,vce="robust",clusters=None,names=names,config=config)
    block=fit_block_arrays(y,BX,plan,offset=off,true_w=w,vce="robust",clusters=None,names=names,config=config)
    _assert_result_parity(block,dense,se_tol=1e-7)
    assert block.names == ("b0","b2") and block.n_separated == m


def test_structured_ppml_frontend_avoids_global_dense_design_and_matches_pooled(monkeypatch):
    import pandas as pd
    import econhdfe.design as design_module
    from econhdfe.design import build_design, factor, reg_interaction
    from econhdfe.models.ppml.heterogeneous import fit_structured_dataframe
    from econhdfe.models.ppml.estimator import fit_arrays
    from econhdfe.hdfe.plan import FEPlan

    rng = np.random.default_rng(6631)
    G, per, p = 4, 90, 3
    n = G * per
    g = np.repeat(np.arange(G), per)
    # Both FE partitions are local to g, so the full design/FE graph separates.
    within = np.tile(np.arange(per), G)
    fe1 = g * 10_000 + within // 6
    fe2 = g * 1_000 + within % 6
    data = {"g": g, "fe1": fe1, "fe2": fe2}
    for j in range(p):
        data[f"x{j}"] = rng.normal(size=n)
    df = pd.DataFrame(data)
    specs = [reg_interaction(factor("g", drop_base=False), f"x{j}") for j in range(p)]
    plan = FEPlan.from_arrays([fe1, fe2])

    # Generate a finite PPML problem with block-specific slopes.
    eta = np.zeros(n)
    for j in range(p):
        beta = np.linspace(-0.15, 0.18, G) * (j + 1) / p
        eta += beta[g] * df[f"x{j}"].to_numpy()
    mu = np.exp(np.clip(eta, -1.0, 1.0))
    y = rng.poisson(mu).astype(float)
    y[g == 0][0:1] if False else None
    df["y"] = y

    cfg = PPMLConfig(separation=("fe",), engine="optimized", max_iter=100)
    dense = build_design(df, specs, n, structural=False)
    pooled = fit_arrays(
        y, dense.values, plan, offset=np.zeros(n), true_w=np.ones(n),
        vce="robust", clusters=None, names=dense.names, config=cfg,
    )

    # After the pooled reference exists, forbid the global dense materializer.
    def forbidden(*args, **kwargs):
        raise AssertionError("structured PPML must not materialize global dense X")
    monkeypatch.setattr(design_module, "_materialize_compiled_dense", forbidden)

    got, exec_design = fit_structured_dataframe(
        df, y="y", x=specs, plan=plan, vce="robust", config=cfg,
        structural_collinearity=False,
    )
    assert isinstance(exec_design.values, BlockDesign)
    assert exec_design.storage_plan.representation == "block_dense"
    np.testing.assert_allclose(got.coef, pooled.coef, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(got.stderr, pooled.stderr, rtol=2e-7, atol=2e-8)
    np.testing.assert_array_equal(got.sample_mask, pooled.sample_mask)
    assert got.loglike == pytest.approx(pooled.loglike, rel=2e-9, abs=2e-9)


def test_partitioned_shared_control_irls_and_final_wls_match_pooled():
    from econhdfe.compute.block_design import DenseDesignBlock
    from econhdfe.compute.design_plan import analyze_structural_partition
    from econhdfe.models.ppml.heterogeneous import final_wls_block

    rng = np.random.default_rng(9053)
    B, m, p = 4, 420, 2
    n = B*m
    block_id = np.repeat(np.arange(B), m)
    within = np.tile(np.arange(m), B)
    g1 = block_id * 10_000 + within // 12
    g2 = block_id * 1_000 + within % 12
    local = rng.normal(size=(n, p))
    z = rng.normal(size=n)
    X = np.zeros((n, B*p + 1))
    blocks = []
    for b in range(B):
        rows = np.flatnonzero(block_id == b).astype(np.int64)
        cols = np.r_[np.arange(b*p, (b+1)*p), B*p].astype(np.int64)
        vals = np.column_stack([local[rows], z[rows]])
        X[np.ix_(rows, cols)] = vals
        blocks.append(DenseDesignBlock(b, rows, cols, vals.copy()))
    BX = BlockDesign(n, X.shape[1], tuple(blocks))
    plan = FEPlan.from_arrays([g1, g2])
    eta = .12*z
    for b in range(B):
        rows = block_id == b
        eta[rows] += local[rows] @ np.array([.08 + .01*b, -.05 + .015*b])
    y = rng.poisson(np.exp(np.clip(eta, -1.2, 1.2))).astype(float)
    w = np.exp(rng.normal(scale=.08, size=n)); off = np.zeros(n)
    cfg = _cfg(engine="optimized", fast_partial=False)

    pooled = fit_irls(y, X, plan, w, off, cfg, projector=plan.projector(engine="optimized"))
    blocked = fit_irls_block(y, BX, plan, w, off, cfg, structure=None)
    assert pooled.converged and blocked.converged
    np.testing.assert_allclose(blocked.beta, pooled.beta, rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(blocked.mu, pooled.mu, rtol=3e-8, atol=3e-8)

    beta_dense, _, _, _ = final_wls_block(
        y, BX, plan, w, off, blocked, cfg, structure=None
    )
    # Pooled final WLS computed from the same converged state.
    wf = w * blocked.mu
    zwork = blocked.eta - off - 1.0
    pos = y > 0
    zwork = zwork.copy(); zwork[pos] += y[pos] / blocked.mu[pos]
    aug = np.column_stack([zwork, X])
    aug, _ = plan.projector(engine="optimized").residualize(
        aug, wf, tol=cfg.target_inner_tol, return_info=True, copy=False,
    )
    from econhdfe.compute.linalg import weighted_lstsq
    ref, _ = weighted_lstsq(aug[:, 1:], aug[:, 0], wf, fast=False)
    np.testing.assert_allclose(beta_dense, ref, rtol=3e-8, atol=3e-9)


def test_structured_ppml_shared_global_control_uses_partitioned_design_without_dense_x(monkeypatch):
    import pandas as pd
    import econhdfe.design as design_module
    from econhdfe.design import build_design, factor, reg_interaction
    from econhdfe.models.ppml.heterogeneous import fit_structured_dataframe
    from econhdfe.models.ppml.estimator import fit_arrays

    rng = np.random.default_rng(9054)
    B, m = 5, 110
    n = B*m
    g = np.repeat(np.arange(B), m)
    within = np.tile(np.arange(m), B)
    fe1 = g*10_000 + within//10
    fe2 = g*1_000 + within%10
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    df = pd.DataFrame({"g": g, "x": x, "z": z, "fe1": fe1, "fe2": fe2})
    specs = [reg_interaction(factor("g", drop_base=False), "x"), "z"]
    eta = .11*z + np.linspace(-.12, .16, B)[g] * x
    y = rng.poisson(np.exp(np.clip(eta, -1.0, 1.0))).astype(float)
    df["y"] = y
    plan = FEPlan.from_arrays([fe1, fe2])
    cfg = PPMLConfig(separation=("fe",), engine="optimized", max_iter=120)

    dense = build_design(df, specs, n, structural=False)
    pooled = fit_arrays(
        y, dense.values, plan, offset=np.zeros(n), true_w=np.ones(n),
        vce="robust", clusters=None, names=dense.names, config=cfg,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("partitioned structured PPML must not materialize global dense X")
    monkeypatch.setattr(design_module, "_materialize_compiled_dense", forbidden)
    got, exec_design = fit_structured_dataframe(
        df, y="y", x=specs, plan=plan, vce="robust", config=cfg,
        structural_collinearity=False,
    )
    assert isinstance(exec_design.values, BlockDesign)
    assert not exec_design.values.columns_disjoint
    assert exec_design.execution_structure.reason == "exact_fe_partition_shared_columns"
    np.testing.assert_allclose(got.coef, pooled.coef, rtol=5e-8, atol=5e-9)
    np.testing.assert_allclose(got.stderr, pooled.stderr, rtol=6e-7, atol=6e-8)
    assert got.loglike == pytest.approx(pooled.loglike, rel=5e-9, abs=5e-9)
