import numpy as np


def test_public_nonexistence_example_flags_obs_1_and_5():
    from econhdfe import ppmlhdfe, PPMLConfig
    y = np.array([0,0,0,0,0,2,3,5,7,10.])
    X = np.array([
        [0,1,0], [0,0,0], [0,0,0], [0,0,0], [1,9,0],
        [21,21,21], [0,0,0], [0,0,0], [0,0,0], [-18,-18,0],
    ], dtype=float)
    result = ppmlhdfe(
        y=y, x=X, vce="model",
        config=PPMLConfig(
            separation=("simplex",),
            tolerance=1e-9,
            target_inner_tol=1e-10,
        ),
    )
    assert (np.flatnonzero(result.separation_mask) + 1).tolist() == [1, 5]
    # The third original regressor survives; upstream reports about 0.066.
    assert abs(result.params[result.names.index("x2")] - 0.066) < 1e-3


# A 3x5 panel on which the ReLU fixed point stalls: the full loop never
# certifies, while eight zero rows are separated (exact LP below).
_STALL_Y = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0], dtype=float)
_STALL_X = np.array([
    [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 1],
    [1, 0, 1], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 0],
], dtype=float)
_STALL_G = [np.repeat(np.arange(3), 5), np.tile(np.arange(5), 3)]


def _lp_separated(y, X, groups):
    """Row i is separated iff max z_i > 0 s.t. z = Xb + Da, z >= 0 on y = 0, z = 0 on y > 0."""
    from scipy.optimize import linprog
    A = np.column_stack([X] + [np.eye(g.max() + 1)[g] for g in groups])
    zero = y == 0
    out = np.zeros(len(y), dtype=bool)
    for i in np.flatnonzero(zero):
        r = linprog(
            -A[i], A_ub=np.vstack([-A[zero], A[zero]]),
            b_ub=np.r_[np.zeros(zero.sum()), np.ones(zero.sum())],
            A_eq=A[~zero], b_eq=np.zeros((~zero).sum()),
            bounds=[(None, None)] * A.shape[1], method="highs",
        )
        out[i] = r.status == 0 and -r.fun > 1e-7
    return out


def test_uncertified_relu_drops_only_verified_rows_and_is_row_order_invariant():
    from econhdfe import FEPlan
    from econhdfe.models.ppml.separation_relu import relu_separation
    truth = _lp_separated(_STALL_Y, _STALL_X, _STALL_G)
    assert truth.sum() == 8
    info = relu_separation(_STALL_Y, _STALL_X, FEPlan.from_arrays(_STALL_G))
    # The previous loop returned an empty set here after 100 rounds.
    assert not info.converged and info.status == "stalled"
    assert info.iterations < 100
    assert info.verified == info.separated.sum() >= 1
    assert not np.any(info.separated & ~truth)
    perm = np.random.default_rng(5).permutation(len(_STALL_Y))
    again = relu_separation(_STALL_Y[perm], _STALL_X[perm][:, ::-1], FEPlan.from_arrays([g[perm] for g in _STALL_G]))
    assert np.array_equal(again.separated, info.separated[perm])


def test_uncertified_relu_warns_reports_status_and_fails_when_strict():
    import pytest
    from econhdfe import ConvergenceError, PPMLConfig, ppmlhdfe
    from econhdfe.models.ppml.separation import SeparationWarning
    kw = dict(y=_STALL_Y, x=_STALL_X, names=("x1", "x2", "x3"), absorb=_STALL_G, vce="model")
    with pytest.warns(SeparationWarning, match="without a certificate"):
        r = ppmlhdfe(**kw, config=PPMLConfig(separation=("relu",), drop_singletons=False))
    status = r.diagnostics["separation_status"]
    assert status["certified"] is False and status["methods"]["relu"] == "stalled"
    assert status["relu_verified"] == r.diagnostics["separation_by_method"]["relu"] >= 1
    with pytest.raises(ConvergenceError) as err:
        ppmlhdfe(**kw, config=PPMLConfig(separation=("relu",), drop_singletons=False, relu_strict=True))
    assert err.value.code == "convergence.separation_relu"


def test_certified_relu_reports_certified_status():
    from econhdfe import PPMLConfig, ppmlhdfe
    y = np.array([0, 1, 0, 0, 1.]); id1 = np.array([1, 1, 2, 2, 2]); id2 = np.array([1, 1, 1, 2, 2])
    r = ppmlhdfe(y=y, absorb=[id1, id2], vce="model", config=PPMLConfig(tolerance=1e-8, target_inner_tol=1e-10))
    assert r.diagnostics["separation_status"]["certified"] is True


def test_relu_controls_are_validated():
    import pytest
    from econhdfe import PPMLConfig, SpecificationError
    for bad in (dict(relu_stall_rounds=-1), dict(relu_stall_rounds=True), dict(relu_strict=1)):
        with pytest.raises(SpecificationError):
            PPMLConfig(**bad).validate()
