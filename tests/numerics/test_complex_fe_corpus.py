from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from econhdfe import interaction, reghdfe
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.rank import categorical_rank
from econhdfe.pipeline import _prepare_standard_fe_structure


def _factorize_interaction(*arrays):
    mi = pd.MultiIndex.from_arrays([np.asarray(a) for a in arrays])
    return pd.factorize(mi, sort=False)[0].astype(np.int32)


def _complex_panel(*, mobility: bool, seed: int = 6501, n_individual: int = 180, T: int = 7):
    rng = np.random.default_rng(seed + int(mobility))
    individual = np.repeat(np.arange(n_individual, dtype=np.int32), T)
    year = np.tile(np.arange(T, dtype=np.int32), n_individual)
    n = len(individual)
    n_city = max(10, n_individual // 12)
    n_industry = max(7, n_individual // 20)

    base_city = rng.integers(0, n_city, n_individual, dtype=np.int32)
    base_industry = rng.integers(0, n_industry, n_individual, dtype=np.int32)
    city = base_city[individual].copy()
    industry = base_industry[individual].copy()
    if mobility:
        # Persistent but non-nested mobility: roughly one-third of person-years
        # move city and one-quarter switch industry.  The destination depends on
        # individual and year so the resulting interactions are not simple nests.
        move_city = rng.random(n) < 0.34
        move_industry = rng.random(n) < 0.27
        city[move_city] = (
            city[move_city] + 1 + (individual[move_city] + 3 * year[move_city]) % max(n_city - 1, 1)
        ) % n_city
        industry[move_industry] = (
            industry[move_industry] + 1 + (2 * individual[move_industry] + year[move_industry]) % max(n_industry - 1, 1)
        ) % n_industry

    cy = _factorize_interaction(city, year)
    iy = _factorize_interaction(industry, year)
    ci = _factorize_interaction(city, industry)
    x = rng.normal(size=n)
    person_fe = rng.normal(scale=.35, size=n_individual)
    cy_fe = rng.normal(scale=.12, size=int(cy.max()) + 1)
    iy_fe = rng.normal(scale=.10, size=int(iy.max()) + 1)
    y = 0.7 * x + person_fe[individual] + cy_fe[cy] + iy_fe[iy] + rng.normal(scale=.4, size=n)
    return pd.DataFrame({
        "y": y, "x": x, "individual": individual, "year": year,
        "city": city, "industry": industry,
    })


def _specs():
    return {
        "raw4": ["individual", "year", "city", "industry"],
        "year_interactions": ["individual", interaction("city", "year"), interaction("industry", "year")],
        "cross4": ["individual", interaction("city", "year"), interaction("industry", "year"), interaction("city", "industry")],
        "full7": [
            "individual", "year", "city", "industry",
            interaction("city", "year"), interaction("industry", "year"), interaction("city", "industry"),
        ],
    }


def _requested_groups(df, absorb):
    # Reuse the public specification/compiler path but explicitly request no
    # canonicalization so this represents the user's inference topology.
    groups, slopes, intercepts, names, plan, mask, dropped, inference = _prepare_standard_fe_structure(
        df, absorb, canonicalize_fe=False, drop_singletons=False,
        weights=None, weight_kind="none", canonical_sample_size=1000,
    )
    assert all(s is None for s in inference.slopes)
    assert all(inference.intercepts)
    return inference.groups


@pytest.mark.parametrize("mobility", [False, True], ids=["stable", "mobility"])
@pytest.mark.parametrize("spec_name", ["raw4", "year_interactions", "cross4", "full7"])
def test_complex_fe_auto_solver_matches_cg_reference(mobility, spec_name):
    df = _complex_panel(mobility=mobility, n_individual=120, T=6)
    groups = _requested_groups(df, _specs()[spec_name])
    rng = np.random.default_rng(6600 + int(mobility))
    rhs = rng.normal(size=(len(df), 4))
    ref_abs = HDFEAbsorber(
        groups, tol=1e-10, max_iter=20_000, acceleration="cg",
        core_reduction="off", projection_backend="fused",
    )
    got_abs = HDFEAbsorber(
        groups, tol=1e-10, max_iter=20_000, acceleration="auto",
        core_reduction="auto", projection_backend="auto",
    )
    ref = ref_abs.residualize(rhs)
    got = got_abs.residualize(rhs)
    np.testing.assert_allclose(got, ref, atol=2e-8, rtol=2e-8)
    assert got_abs.fe_orthogonality_error(got) < 5e-8


@pytest.mark.parametrize("mobility", [False, True], ids=["stable", "mobility"])
def test_full7_canonicalization_preserves_exact_requested_inference(mobility):
    df = _complex_panel(mobility=mobility, n_individual=90, T=6)
    absorb = _specs()["full7"]
    common = dict(
        data=df, y="y", x=["x"], absorb=absorb, vce="robust",
        drop_singletons=False, tol=1e-10, dof_method="exact",
    )
    canonical = reghdfe(**common, canonicalize_fe=True)
    raw = reghdfe(**common, canonicalize_fe=False)
    np.testing.assert_allclose(canonical.params, raw.params, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(canonical.vcov, raw.vcov, atol=3e-9, rtol=3e-9)
    assert canonical.df_absorbed == raw.df_absorbed

    requested = _requested_groups(df, absorb)
    assert canonical.df_absorbed == categorical_rank(requested, backend="auto")
    if mobility:
        # Mobility destroys the simple individual->city/industry nesting and
        # leaves more of the requested topology numerically relevant.
        assert len(canonical.absorb_info["canonicalization"]["effective"]) >= 3
    else:
        assert len(canonical.absorb_info["canonicalization"]["effective"]) < len(absorb)


def test_stable_and_mobility_full7_have_distinct_topology():
    stable = _complex_panel(mobility=False, n_individual=100, T=6)
    moving = _complex_panel(mobility=True, n_individual=100, T=6)
    absorb = _specs()["full7"]
    gs = _requested_groups(stable, absorb)
    gm = _requested_groups(moving, absorb)
    rank_stable = categorical_rank(gs, backend="auto")
    rank_moving = categorical_rank(gm, backend="auto")
    # This is the corpus invariant we care about: the mobility DGP really does
    # break nesting rather than just relabel categories.
    assert rank_moving > rank_stable
