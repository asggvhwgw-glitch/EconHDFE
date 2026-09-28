import numpy as np
import pandas as pd

from pyreghdfe import FEPlan
from pyreghdfe.absorber import HDFEAbsorber
from pyreghdfe.two_way import TwoWayFEAbsorber


def test_public_absorber_weight_update_matches_fresh_map():
    rng = np.random.default_rng(9001)
    n = 3000
    groups = [rng.integers(0, 80, n), rng.integers(0, 50, n), rng.integers(0, 20, n)]
    a = rng.normal(size=(n, 3))
    w0 = np.exp(rng.normal(scale=.2, size=n))
    w1 = np.exp(rng.normal(scale=.6, size=n))
    reused = HDFEAbsorber(groups, weights=w0, tol=1e-10)
    reused.update_weights(w1, tol=1e-9)
    got = reused.residualize(a)
    fresh = HDFEAbsorber(groups, weights=w1, tol=1e-9).residualize(a)
    assert np.allclose(got, fresh, atol=2e-8, rtol=2e-8)


def test_public_two_way_weight_update_matches_fresh():
    rng = np.random.default_rng(9002)
    n = 3500
    groups = [rng.integers(0, 100, n), rng.integers(0, 60, n)]
    a = rng.normal(size=(n, 2))
    w0 = np.exp(rng.normal(scale=.2, size=n))
    w1 = np.exp(rng.normal(scale=.5, size=n))
    reused = TwoWayFEAbsorber(groups, weights=w0, tol=1e-10)
    reused.update_weights(w1, tol=1e-9)
    got = reused.residualize(a)
    fresh = TwoWayFEAbsorber(groups, weights=w1, tol=1e-9).residualize(a)
    assert np.allclose(got, fresh, atol=2e-8, rtol=2e-8)


def test_dataframe_fe_plan_carries_structural_metadata():
    city = np.repeat(np.arange(20), 5)
    year = np.tile(np.arange(5), 20)
    province = city // 4
    df = pd.DataFrame({"city": city, "year": year, "province": province})
    plan = FEPlan.from_dataframe(df, ["year", ("province", "year"), ("city", "year")])
    effective, meta = plan.canonicalized()
    assert meta["changed"]
    assert len(effective.groups) == 1
    assert effective.metadata[0].components == ("city", "year")


def test_streaming_fast_wls_matches_explicit_normal_equations():
    from econhdfe.compute.linalg import weighted_lstsq
    rng = np.random.default_rng(733)
    n, k = 4000, 5
    X = rng.normal(size=(n, k)); y = rng.normal(size=n); w = np.exp(rng.normal(scale=.4,size=n))
    b, r = weighted_lstsq(X, y, w, fast=True, chunk_rows=317)
    G = X.T @ (w[:,None]*X); rhs = X.T @ (w*y)
    ref = np.linalg.pinv(G, hermitian=True) @ rhs
    assert np.allclose(b, ref, atol=2e-11, rtol=2e-11)
    assert np.allclose(r, y-X@b, atol=2e-12, rtol=2e-12)
