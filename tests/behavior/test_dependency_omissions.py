import warnings
import numpy as np
import pandas as pd
import pytest

from pyreghdfe import (
    reghdfe, ivreghdfe, factor, reg_interaction, OmittedVariableWarning,
)
from pyreghdfe.design import build_design


def _hierarchy_df(seed=740, n=12000):
    rng = np.random.default_rng(seed)
    city = rng.integers(0, 300, n)
    province = city // 10
    region = province // 6
    year = rng.integers(0, 8, n)
    firm = rng.integers(0, 700, n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 0.8 * z + 0.3 * u + rng.normal(size=n)
    y = 1.1 * x + 1.5 * endog + rng.normal(size=700)[firm] + rng.normal(size=n)
    return pd.DataFrame(dict(
        y=y, x=x, z=z, endog=endog, firm=firm, year=year,
        city=city, province=province, region=region,
    ))


def test_component_dependency_dag_uses_transitive_closure():
    df = _hierarchy_df(1)
    specs = [
        factor("region", drop_base=False),
        factor("province", drop_base=False),
        factor("city", drop_base=False),
    ]
    d = build_design(df, specs, len(df), structural=True)
    plan = d.structural_plan
    direct = {(x.coarse, x.fine) for x in plan.component_dependencies}
    closure = {(x.coarse, x.fine) for x in plan.closure_dependencies}
    assert ("column:region", "column:province") in direct
    assert ("column:province", "column:city") in direct
    assert ("column:region", "column:city") in closure
    # The transitive region->city relation should not require a third full scan.
    assert plan.component_checks == 2
    assert plan.closure_inferences >= 1


def test_interaction_dependency_can_use_component_closure_proof():
    df = _hierarchy_df(2)
    city_year = reg_interaction(
        factor("city", drop_base=False), factor("year", drop_base=False), name="city_year"
    )
    # Include province as a separate block so the component DAG can establish
    # city -> province -> region once and reuse it for interaction reasoning.
    province = factor("province", drop_base=False)
    region_year = reg_interaction(
        factor("region", drop_base=False), factor("year", drop_base=False), name="region_year"
    )
    d = build_design(df, [city_year, province, region_year], len(df), structural=True)
    hits = [x for x in d.structural_plan.dependencies if x.coarse == "region_year" and x.fine == "city_year"]
    assert hits
    assert hits[0].proof_type == "component_dependency_closure"


def test_default_collinearity_warns_instead_of_silent_drop():
    df = _hierarchy_df(3)
    with pytest.warns(OmittedVariableWarning, match="omitted"):
        r = reghdfe(
            df, y="y", x=["x", factor("year")],
            absorb=["firm", ("city", "year")],
        )
    assert r.has_omitted_variables
    assert any(v["name"].startswith("year[") for v in r.omitted_variables)


def test_explicit_drop_remains_quiet_opt_in():
    df = _hierarchy_df(4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = reghdfe(
            df, y="y", x=["x", factor("year")],
            absorb=["firm", ("city", "year")], collinearity="drop",
        )
    assert not [w for w in caught if issubclass(w.category, OmittedVariableWarning)]
    assert r.has_omitted_variables


def test_raise_mode_still_blocks_omitted_columns():
    df = _hierarchy_df(5)
    with pytest.raises(np.linalg.LinAlgError, match="collinear/absorbed"):
        reghdfe(
            df, y="y", x=["x", factor("year")],
            absorb=["firm", ("city", "year")], collinearity="raise",
        )


def test_iv_default_warns_for_redundant_instrument_interactions():
    df = _hierarchy_df(6)
    inst = reg_interaction(
        factor("region", drop_base=False), factor("year", drop_base=False), name="region_year"
    )
    with pytest.warns(OmittedVariableWarning):
        r = ivreghdfe(
            df, y="y", exog=["x"], endog=["endog"], instruments=["z", inst],
            absorb=["firm", ("city", "year")],
        )
    assert r.has_omitted_variables
    assert any(v.get("role") == "excluded_instrument" for v in r.omitted_variables)
