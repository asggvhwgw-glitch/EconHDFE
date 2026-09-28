import numpy as np
import pandas as pd

from pyreghdfe import interaction, reghdfe


def _hierarchy_fixture(seed=801, n=30000):
    rng = np.random.default_rng(seed)
    year = rng.integers(0, 8, n)
    city = rng.integers(0, 100, n)
    province = city // 10
    firm = rng.integers(0, 2500, n)
    x = rng.normal(size=n)
    cy = city * 8 + year
    y = 0.7 * x + rng.normal(size=2500)[firm] + rng.normal(size=800)[cy] + rng.normal(scale=.2, size=n)
    return pd.DataFrame({
        "firm": firm, "year": year, "province": province, "city": city,
        "x": x, "y": y,
    })


def test_refinement_dag_collapses_complex_interaction_hierarchy():
    df = _hierarchy_fixture()
    r = reghdfe(
        df, y="y", x=["x"],
        absorb=[
            "firm",
            "year",
            interaction("province", "year"),
            interaction("city", "year"),
        ],
        tol=1e-9,
    )
    c = r.absorb_info["canonicalization"]
    assert c["effective"] == ("firm", "city#year")
    dropped = {d["name"]: d for d in c["dropped"]}
    assert dropped["year"]["spanned_by"] == "city#year"
    assert dropped["province#year"]["spanned_by"] == "city#year"
    # year <= city#year follows directly from interaction components; the
    # province hierarchy is learned from the estimation data (city -> province).
    edges = {(e["coarse"], e["fine"]): e["proof_type"] for e in c["refinement_edges"]}
    assert edges[("year", "city#year")] == "component_containment"
    assert edges[("province#year", "city#year")] == "exact_mapping"


def test_complex_canonicalization_is_numerically_equivalent_to_raw_system():
    df = _hierarchy_fixture(seed=802, n=18000)
    kw = dict(
        data=df, y="y", x=["x"],
        absorb=["firm", "year", ("province", "year"), ("city", "year")],
        tol=1e-9,
    )
    a = reghdfe(**kw, canonicalize_fe=True)
    b = reghdfe(**kw, canonicalize_fe=False)
    np.testing.assert_allclose(a.params, b.params, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(a.stderr, b.stderr, atol=2e-7, rtol=2e-7)


def test_observed_bijections_are_reported_as_equivalent_partitions():
    rng = np.random.default_rng(803)
    n = 12000
    city = rng.integers(0, 400, n)
    city_alias = city * 37 + 11
    x = rng.normal(size=n)
    y = .4 * x + rng.normal(size=400)[city] + rng.normal(size=n)
    df = pd.DataFrame({"city": city, "city_alias": city_alias, "x": x, "y": y})
    r = reghdfe(df, y="y", x=["x"], absorb=["city", "city_alias"], tol=1e-9)
    c = r.absorb_info["canonicalization"]
    assert c["effective"] == ("city",)
    assert c["dropped"][0]["reason"] == "equivalent_intercept"
    assert c["equivalences"]


def test_unrelated_high_dimensional_partitions_are_not_dropped():
    rng = np.random.default_rng(804)
    n = 50000
    a = rng.integers(0, 700, n)
    b = rng.integers(0, 900, n)
    c = rng.integers(0, 1000, n)
    x = rng.normal(size=n)
    y = .3*x + rng.normal(size=700)[a] + rng.normal(size=900)[b] + rng.normal(size=n)
    r = reghdfe(None, y=y, x=x, absorb=[a, b, c], tol=1e-8)
    plan = r.absorb_info["canonicalization"]
    assert len(plan["effective"]) == 3
    assert not plan["dropped"]
    # The cheap rejection stage should prevent at least some full-data scans.
    assert plan["diagnostics"]["sample_rejections"] > 0


def test_singleton_pruning_can_reveal_new_exact_nesting():
    # Before pruning, B=0 maps to both A=0 and A=1, so A is not nested in B.
    # The sole violating row is a singleton in C.  On the final estimation
    # sample, B refines A exactly and the second canonicalization can remove A.
    n_groups = 10
    reps = 6
    b = np.repeat(np.arange(n_groups), reps)
    a = b // 2
    a = a.copy()
    a[0] = 1
    c = np.tile(np.arange(reps), n_groups)
    c = c.copy()
    c[0] = 99
    rng = np.random.default_rng(805)
    x = rng.normal(size=b.size)
    y = .8*x + rng.normal(size=n_groups)[b] + rng.normal(size=6)[np.minimum(c, 5)] + rng.normal(scale=.1, size=b.size)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "x": x, "y": y})
    r = reghdfe(df, y="y", x=["x"], absorb=["a", "b", "c"], tol=1e-9)
    plan = r.absorb_info["canonicalization"]
    assert r.dropped_singletons >= 1
    assert plan["passes"] == 2
    assert "a" not in plan["effective"]
    assert "b" in plan["effective"]


def test_three_way_interaction_structurally_spans_each_named_component():
    rng = np.random.default_rng(806)
    n = 16000
    year = rng.integers(0, 6, n)
    city = rng.integers(0, 40, n)
    sector = rng.integers(0, 12, n)
    x = rng.normal(size=n)
    key = (city * 12 + sector) * 6 + year
    y = .5*x + rng.normal(size=40*12*6)[key] + rng.normal(size=n)
    df = pd.DataFrame({"year":year,"city":city,"sector":sector,"x":x,"y":y})
    r = reghdfe(
        df, y="y", x=["x"],
        absorb=["year", "sector", interaction("city", "sector", "year")],
        tol=1e-9,
    )
    plan = r.absorb_info["canonicalization"]
    assert plan["effective"] == ("city#sector#year",)
    proofs = {d["name"]: d["proof_type"] for d in plan["dropped"]}
    assert proofs["year"] == "component_containment"
    assert proofs["sector"] == "component_containment"
