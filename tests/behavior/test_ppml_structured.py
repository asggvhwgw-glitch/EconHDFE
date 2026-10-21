import numpy as np
import pytest
from econhdfe.compute.block_design import BlockDesign, compile_block_design
from econhdfe.compute.design_plan import analyze_design_structure
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.heterogeneous import detect_separation_block, fit_irls_block
from econhdfe.models.ppml.config import PPMLConfig


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
