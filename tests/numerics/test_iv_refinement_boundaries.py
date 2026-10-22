"""Explicit-FE IV references when omission precedes control refinement."""

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, ivhdfe
from econhdfe.errors import LinearAlgebraError


@pytest.fixture(scope="module")
def boundary_design():
    rng = np.random.default_rng(231022)
    n = 720
    groups = []
    for levels in (18, 11, 9):
        probability = np.arange(1, levels + 1, dtype=float)
        probability /= probability.sum()
        codes = np.r_[np.repeat(np.arange(levels), 2),
                      rng.choice(levels, n - 2 * levels, p=probability)]
        rng.shuffle(codes)
        groups.append(codes)
    dummy = np.column_stack([np.eye(g.max() + 1)[g] for g in groups])
    qf, singular, _ = np.linalg.svd(dummy, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
    qf = qf[:, :rank]
    raw = rng.normal(size=(n, 6))
    q, _ = np.linalg.qr(raw - qf @ (qf.T @ raw), mode="reduced")
    q *= np.sqrt(n)
    shared = 50 * (dummy @ rng.normal(size=dummy.shape[1])) + q[:, 0]
    c1, c2 = shared, shared + 1e-5 * q[:, 1]
    z = q[:, 2:4] + dummy @ rng.normal(size=(dummy.shape[1], 2))
    e = .9 * q[:, 2] + .4 * q[:, 3] + .5 * q[:, 4]
    e += dummy @ rng.normal(size=dummy.shape[1])
    y = .7 * c1 - .4 * c2 + 1.1 * e + .3 * q[:, 4] + .4 * q[:, 5]
    y += dummy @ rng.normal(size=dummy.shape[1])
    frame = pd.DataFrame(dict(y=y, c1=c1, c2=c2, duplicate=-2*c1,
                              absorbed=2*groups[0]+1, e=e, z1=z[:, 0], z2=z[:, 1],
                              endog_control=c2, instrument_control=c1,
                              cluster=rng.permutation(n) % 24,
                              fe1=groups[0], fe2=groups[1], fe3=groups[2]))
    return frame, qf, rank


def _reference(frame, active, qf, fe_rank, control_units):
    """Project an invertible contrast basis of the actual float input controls.

    Projecting two FE-heavy columns separately and only then subtracting them
    can amplify projection rounding by 1e5. Form their normalized difference
    before the explicit dummy projection. Decimal is confined to this small
    input-basis change, independent of the package's absorption implementation.
    """
    a = frame[["y", *active, "e", "z1", "z2"]].to_numpy(copy=True)
    u0, u1 = control_units
    with localcontext() as context:
        context.prec = 50
        d0, d1 = Decimal.from_float(u0), Decimal.from_float(u1)
        delta = Decimal("0.00001")
        controls = []
        for row in a[:, 1:3]:
            first = Decimal.from_float(float(row[0])) / d0
            second = Decimal.from_float(float(row[1])) / d1
            controls.append((float(first), float((second-first) / delta)))
    a[:, 1:3] = controls
    # X_good = X_original @ transform; restore both coefficients and influence
    # rows after solving in the well-conditioned parameterization.
    transform = np.array([[1/u0, -1e5/u0, 0.], [0., 1e5/u1, 0.], [0., 0., 1.]])
    inverse_transform = np.array([[u0, u1, 0.], [0., 1e-5*u1, 0.], [0., 0., 1.]])
    a = a - qf @ (qf.T @ a)
    a -= qf @ (qf.T @ a)
    x = a[:, 1:4]
    z = np.column_stack((a[:, 1:3], a[:, 4:6]))
    scales = np.linalg.norm(x, axis=0)
    qz, _ = np.linalg.qr(z / np.linalg.norm(z, axis=0), mode="reduced")
    projected = qz @ (qz.T @ (x / scales))
    u, singular, vt = np.linalg.svd(projected, full_matrices=False)
    inverse = ((vt.T / singular) @ u.T) / scales[:, None]
    gamma = inverse @ a[:, 0]
    beta = transform @ gamma
    residual = a[:, 0] - x @ gamma
    influence = (transform @ inverse).T * residual[:, None]
    cluster = frame.cluster.to_numpy()
    sums = np.array([influence[cluster == g].sum(axis=0) for g in np.unique(cluster)])
    n = len(frame)
    correction = 24 / 23 * (n - 1) / (n - 3 - fe_rank)
    units = np.linalg.norm(x @ inverse_transform, axis=0) / np.sqrt(n)
    return beta, sums.T @ sums * correction, residual, units


def _fit(frame, controls, *, mode="drop", redundant_roles=False):
    return ivhdfe(frame, y="y", exog=controls,
                  endog=["e", "endog_control"] if redundant_roles else ["e"],
                  instruments=["z1", "z2", "instrument_control"] if redundant_roles else ["z1", "z2"],
                  absorb=["fe1", "fe2", "fe3"], cluster="cluster", vce="cluster",
                  drop_singletons=False, dof_method="exact", collinearity=mode,
                  tol=1e-12, max_iter=16000,
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=64, cache="off"))


@pytest.mark.parametrize("case", ["duplicate", "absorbed", "reordered_rescaled"])
def test_omission_preserves_refinement_and_iv_role_priority(boundary_design, case):
    original, qf, fe_rank = boundary_design
    frame = original.copy()
    control_units = (1., 1.)
    if case == "duplicate":
        controls, active = ["c1", "duplicate", "c2"], ["c1", "c2"]
        omitted = {"duplicate": "duplicate_or_scaled"}
    elif case == "absorbed":
        controls, active = ["absorbed", "c1", "c2"], ["c1", "c2"]
        omitted = {"absorbed": "absorbed_or_zero"}
    else:
        # Keep the first eligible column, even when a scaled duplicate appears
        # before its original. Included controls retain priority over E and Z.
        frame["c1"] *= 1e4
        frame["c2"] *= 1e-3
        frame["duplicate"] *= 1e2
        control_units = (1e-3, -2e2)
        controls, active = ["c2", "absorbed", "duplicate", "c1"], ["c2", "duplicate"]
        omitted = {"absorbed": "absorbed_or_zero", "c1": "duplicate_or_scaled"}
    beta, covariance, residual, units = _reference(frame, active, qf, fe_rank, control_units)
    # The displayed RMS-scaled e coefficient is 1.1*sqrt(1.22), about 1.21499;
    # its original-unit structural coefficient remains 1.1.
    np.testing.assert_allclose(beta[-1], 1.1, rtol=1e-10, atol=1e-10)
    got = _fit(frame, controls, redundant_roles=case == "reordered_rescaled")
    assert got.names == (*active, "e")
    assert got.converged and got.nobs == len(frame) and got.rank == 3
    assert got.df_absorbed == fe_rank and got.dropped_singletons == 0
    info = got.collinearity_info
    assert info["exogenous"]["active"] == tuple(active)
    assert {item["name"]: item["reason"] for item in info["exogenous"]["omitted"]} == omitted
    assert info["endogenous"]["active"] == ("e",)
    assert info["excluded_instruments"]["active"] == ("z1", "z2")
    if case == "reordered_rescaled":
        assert {item["name"] for item in info["endogenous"]["omitted"]} == {"endog_control"}
        assert {item["name"] for item in info["excluded_instruments"]["omitted"]} == {"instrument_control"}
    np.testing.assert_allclose(got.params * units, beta * units, rtol=1e-5, atol=1e-6)
    se = np.sqrt(np.diag(covariance))
    assert np.max(np.abs(got.vcov - covariance) / np.outer(se, se)) < 5e-4
    assert np.linalg.norm(got.residuals - residual) / np.linalg.norm(residual) < 1e-6


def test_raise_collinearity_policy_is_not_bypassed_by_refinement(boundary_design):
    frame, _, _ = boundary_design
    with pytest.raises(LinearAlgebraError) as caught:
        _fit(frame, ["c1", "duplicate", "c2"], mode="raise")
    assert caught.value.code == "numerical.linalg"
