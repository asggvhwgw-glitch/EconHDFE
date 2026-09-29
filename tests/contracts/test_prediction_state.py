from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from econhdfe import factor, ivhdfe, olshdfe


def _frame(seed=91, n=600):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n // 6), 6)
    g = np.tile(np.array(["a", "b", "c", "a", "b", "c"], dtype=object), n // 6)
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    z = rng.normal(size=n)
    zcat = np.tile(np.array([0, 1, 2, 0, 1, 2]), n // 6)
    endog = 0.7 * z + 0.2 * w + rng.normal(size=n)
    y = 1.2 * x + 0.4 * w + 0.8 * endog + rng.normal(size=n // 6)[firm] + rng.normal(size=n)
    return pd.DataFrame(
        {"y": y, "x": x, "w": w, "z": z, "zcat": zcat, "endog": endog, "g": g, "firm": firm}
    )


def test_ols_prediction_state_freezes_design_order_and_factor_levels():
    df = _frame()
    result = olshdfe(
        df,
        y="y",
        x=[factor("g", base="b"), "x"],
        absorb=["firm"],
        collinearity="drop",
    )

    state = result.prediction_state
    assert state is not None
    assert state.estimator == "ols"
    assert state.coefficient_names == result.names
    assert state.coefficient_roles == ("regressor",)
    design = state.design("regressor")
    assert design.active_names == result.names
    assert design.reconstructable
    assert state.can_rebuild_linear_predictor

    term = next(item for item in design.terms if item.name == "g")
    encoding = term.categorical[0]
    assert encoding.source == "g"
    assert encoding.levels == ("a", "b", "c")
    assert encoding.base_levels == ("b",)
    assert term.cells == (("a",), ("b",), ("c",))

    assert state.fixed_effects.requested_names == ("firm",)
    assert state.fixed_effects.effective_names == ("firm",)
    assert not state.fixed_effects.level_maps_available
    assert not state.can_include_fixed_effects

    with pytest.raises(FrozenInstanceError):
        state.estimator = "changed"


def test_prediction_sample_snapshot_preserves_singleton_pruning_compactly():
    rng = np.random.default_rng(92)
    firm = np.repeat(np.arange(10), 2)
    firm = np.r_[firm, 10]
    x = rng.normal(size=len(firm))
    y = 0.5 * x + rng.normal(size=11)[firm] + rng.normal(size=len(firm))
    result = olshdfe(y=y, x=x, absorb=firm, collinearity="drop")

    sample = result.prediction_state.sample
    assert sample.nobs_raw == 21
    assert sample.nobs_final == 20
    assert sample.packed_mask is not None
    mask = sample.mask()
    assert mask.dtype == bool
    assert mask.shape == (21,)
    assert mask.sum() == 20
    assert not mask[-1]
    assert not mask.flags.writeable
    assert sample.history[-1].reason == "singleton_pruning"


def test_iv_prediction_state_keeps_role_specific_designs():
    df = _frame(seed=93)
    result = ivhdfe(
        df,
        y="y",
        exog=["w"],
        endog=["endog"],
        instruments=[factor("zcat"), "z"],
        absorb=["firm"],
        vce="robust",
        collinearity="drop",
    )

    state = result.prediction_state
    assert state is not None
    assert state.coefficient_roles == ("exogenous", "endogenous")
    exog = state.design("exogenous")
    endog = state.design("endogenous")
    instruments = state.design("excluded_instrument")
    assert exog.active_names + endog.active_names == result.names
    assert instruments.active_names
    assert exog.reconstructable and endog.reconstructable and instruments.reconstructable
    assert state.can_rebuild_linear_predictor


def test_array_design_is_recorded_but_not_claimed_reconstructable():
    rng = np.random.default_rng(94)
    n = 300
    g = np.repeat(np.arange(60), 5)
    X = rng.normal(size=(n, 2))
    y = X @ np.array([0.7, -0.2]) + rng.normal(size=60)[g] + rng.normal(size=n)
    result = olshdfe(y=y, x=X, absorb=g, collinearity="drop")

    state = result.prediction_state
    design = state.design("regressor")
    assert design.active_names == result.names
    assert not design.reconstructable
    assert not state.can_rebuild_linear_predictor
    assert all(term.continuous[0].source is None for term in design.terms)
