"""Independent weighted-FE and clustered-inference references for dense IV.

Only the public estimator is imported from econhdfe. Dummy rank, weight
normalization, FWL, IV projection, CGM intersections and finite-sample factors
are calculated independently; no fitted-model DoF enters the oracle.
"""

from decimal import Decimal, localcontext
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, ivhdfe


@pytest.fixture(scope="module")
def inference_design():
    rng = np.random.default_rng(231023)
    n = 720
    fe1, fe2 = rng.permutation(n) % 15, rng.permutation(n) % 12
    dummy = np.column_stack((np.eye(15)[fe1], np.eye(12)[fe2]))
    g1 = np.arange(n) % 24
    g2 = (np.arange(n) // 24) % 15
    raw = rng.normal(size=(n, 7))
    shared = 50 * (dummy @ rng.normal(size=dummy.shape[1])) + raw[:, 0]
    c1, c2 = shared, shared + 1e-5 * raw[:, 1]
    z = raw[:, 2:4] + dummy @ rng.normal(size=(dummy.shape[1], 2))
    e = .9 * raw[:, 2] + .5 * raw[:, 3] + .4 * raw[:, 4]
    e += dummy @ rng.normal(size=dummy.shape[1])
    noise = (.4 * raw[:, 4] + raw[:, 5]
             + .8 * rng.normal(size=24)[g1] + .7 * rng.normal(size=15)[g2])
    y = .7 * c1 - .4 * c2 + 1.1 * e + noise
    y += dummy @ rng.normal(size=dummy.shape[1])
    frame = pd.DataFrame(dict(y=y, c1=c1, c2=c2, e=e, z1=z[:, 0], z2=z[:, 1],
                              fe1=fe1, fe2=fe2, g1=g1, g2=g2,
                              aw=rng.uniform(.3, 2.7, n), fw=rng.integers(1, 4, n)))
    # These configurations intentionally have no nested FE adjustment. Check
    # that fact directly, without using the package's topology or DoF helpers.
    for fe in (fe1, fe2):
        assert np.bincount(fe).min() >= 2
        for groups in (g1, g2):
            assert all(len(np.unique(groups[fe == level])) > 1 for level in np.unique(fe))
    return frame


def _reference(frame, *, weights, clusters, units=(1., 1., 1., 1., 1.)):
    """Weighted dummy FWL followed by observation-space IV and CGM scores."""
    n = len(frame)
    weights = np.asarray(weights, dtype=float)
    sw = np.sqrt(weights)
    dummy = np.column_stack([
        np.eye(len(np.unique(frame[name])))[np.unique(frame[name], return_inverse=True)[1]]
        for name in ("fe1", "fe2")
    ])
    qf, singular, _ = np.linalg.svd(dummy * sw[:, None], full_matrices=False)
    cutoff = np.finfo(float).eps * max(dummy.shape) * singular[0]
    fe_rank = int(np.count_nonzero(singular > cutoff))
    # The deterministic fixture is one connected two-FE graph. The explicit
    # numerical dummy rank also independently certifies its 15+12-1 DoF.
    assert fe_rank == 26
    qf = qf[:, :fe_rank]
    a = frame[["y", "c1", "c2", "e", "z1", "z2"]].to_numpy(copy=True)
    u0, u1, ue, uz0, uz1 = units
    # Form the weak control direction BEFORE projecting away large common FE.
    # Decimal acts only on the actual binary-float inputs and units; it does
    # not replace the observations by their unrounded data-generating values.
    with localcontext() as context:
        context.prec = 50
        d0, d1 = Decimal.from_float(u0), Decimal.from_float(u1)
        delta = Decimal("0.00001")
        controls = []
        for first, second in a[:, 1:3]:
            c0 = Decimal.from_float(float(first)) / d0
            c1 = Decimal.from_float(float(second)) / d1
            controls.append((float(c0), float((c1-c0) / delta)))
    a[:, 1:3] = controls
    a[:, 3:] /= np.array([ue, uz0, uz1])
    transform = np.array([[1/u0, -1e5/u0, 0.],
                          [0., 1e5/u1, 0.], [0., 0., 1/ue]])
    inverse_transform = np.array([[u0, u1, 0.],
                                  [0., 1e-5*u1, 0.], [0., 0., ue]])
    a *= sw[:, None]
    a -= qf @ (qf.T @ a)
    a -= qf @ (qf.T @ a)
    x = a[:, 1:4]
    z = np.column_stack((a[:, 1:3], a[:, 4:6]))
    scale = np.linalg.norm(x, axis=0)
    qz, _ = np.linalg.qr(z / np.linalg.norm(z, axis=0), mode="reduced")
    projected = qz @ (qz.T @ (x / scale))
    u, singular, vt = np.linalg.svd(projected, full_matrices=False)
    assert singular[-1] > singular[0] * 1e-4
    inverse = ((vt.T / singular) @ u.T) / scale[:, None]
    gamma = inverse @ a[:, 0]
    residual_weighted = a[:, 0] - x @ gamma
    scores = inverse.T * residual_weighted[:, None]
    # CGM is evaluated in the well-conditioned control-contrast coordinates.
    # The invertible coefficient transform is applied only after aggregation.
    raw = np.zeros((3, 3))
    groups = [frame[name].to_numpy() for name in clusters]
    for count in range(1, len(groups) + 1):
        for subset in combinations(groups, count):
            _, codes = np.unique(np.column_stack(subset), axis=0, return_inverse=True)
            sums = np.array([scores[codes == g].sum(axis=0) for g in np.unique(codes)])
            raw += (1 if count % 2 else -1) * (sums.T @ sums)
    # Select a fixture with a positive raw CGM covariance, so this reference
    # does not depend on a coefficient-unit-sensitive PSD repair convention.
    sd = np.sqrt(np.diag(raw))
    assert np.linalg.eigvalsh(raw / np.outer(sd, sd))[0] > 1e-6
    ng = min(len(np.unique(g)) for g in groups)
    df_fit = n - 3 - fe_rank
    correction = ng / (ng-1) * (n-1) / df_fit
    covariance = transform @ (raw * correction) @ transform.T
    return dict(beta=transform @ gamma, covariance=covariance,
                residual=residual_weighted / sw,
                units=np.linalg.norm(x @ inverse_transform, axis=0) / np.sqrt(n),
                fe_rank=fe_rank, df_fit=df_fit, df_resid=min(df_fit, ng-1), nobs=n)


def _fit(frame, *, weight, weight_type, clusters):
    return ivhdfe(
        frame, y="y", exog=["c1", "c2"], endog=["e"], instruments=["z1", "z2"],
        absorb=["fe1", "fe2"], weights=weight, weight_type=weight_type,
        vce="cluster", cluster=list(clusters), drop_singletons=False,
        dof_method="exact", tol=1e-12, max_iter=16000,
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=64, cache="off"),
    )


def _assert_reference(got, expected, *, nobs_raw, residual_rows=None):
    assert got.converged and got.rank == 3 and got.names == ("c1", "c2", "e")
    assert got.nobs == expected["nobs"] and got.nobs_raw == nobs_raw
    assert got.df_absorbed == expected["fe_rank"]
    assert got.df_resid_fit == expected["df_fit"]
    assert got.df_resid == expected["df_resid"]
    assert got.dropped_singletons == 0
    units = expected["units"]
    np.testing.assert_allclose(got.params * units, expected["beta"] * units,
                               rtol=1e-5, atol=1e-6)
    se = np.sqrt(np.diag(expected["covariance"]))
    np.testing.assert_allclose(got.stderr, se, rtol=1e-4, atol=0.)
    assert np.max(np.abs(got.vcov - expected["covariance"]) / np.outer(se, se)) < 5e-4
    residual = got.residuals if residual_rows is None else got.residuals[residual_rows]
    assert np.linalg.norm(residual - expected["residual"]) / np.linalg.norm(expected["residual"]) < 1e-6


@pytest.fixture(scope="module")
def aweight_fit(inference_design):
    frame = inference_design
    weights = frame.aw.to_numpy()
    # Analytic weights use physical rows for sample size and a unit-mean
    # estimation normalization. Both are computed here without package helpers.
    expected = _reference(frame, weights=weights / weights.mean(), clusters=("g1", "g2"))
    got = _fit(frame, weight="aw", weight_type="aweight", clusters=("g1", "g2"))
    return got, expected


def test_aweight_fe_two_way_cluster_has_independent_dof_and_vce(inference_design, aweight_fit):
    got, expected = aweight_fit
    _assert_reference(got, expected, nobs_raw=len(inference_design))
    assert got.weight_type == "aweight"
    assert got.sum_weights == pytest.approx(inference_design.aw.sum())


def test_aweight_and_variable_units_preserve_fe_inference(inference_design, aweight_fit):
    baseline, _ = aweight_fit
    frame = inference_design.copy()
    units = (1e4, 1e-3, 1e2, 1e3, 1e-2)
    frame[["c1", "c2", "e", "z1", "z2"]] *= np.array(units)
    frame["aw"] *= 37
    weights = frame.aw.to_numpy()
    expected = _reference(frame, weights=weights / weights.mean(),
                          clusters=("g1", "g2"), units=units)
    got = _fit(frame, weight="aw", weight_type="aweight", clusters=("g1", "g2"))
    _assert_reference(got, expected, nobs_raw=len(frame))
    scale = np.array(units[:3])
    np.testing.assert_allclose(got.params * scale, baseline.params, rtol=1e-5, atol=1e-6)
    assert np.max(np.abs(got.vcov * np.outer(scale, scale) - baseline.vcov)
                  / np.outer(baseline.stderr, baseline.stderr)) < 5e-4
    assert got.nobs == baseline.nobs and got.df_resid == baseline.df_resid
    assert got.sum_weights == pytest.approx(37 * baseline.sum_weights)


def test_fweight_fe_cluster_matches_literal_replication_reference(inference_design):
    frame = inference_design
    rows = np.repeat(np.arange(len(frame)), frame.fw.to_numpy())
    expanded = frame.iloc[rows].reset_index(drop=True)
    # The frequency-weight oracle is an actual expanded unweighted dataset;
    # it never reads effective_n or normalized weights from a fitted result.
    expected = _reference(expanded, weights=np.ones(len(expanded)), clusters=("g1",))
    got = _fit(frame, weight="fw", weight_type="fweight", clusters=("g1",))
    _assert_reference(got, expected, nobs_raw=len(frame), residual_rows=rows)
    assert got.weight_type == "fweight"
    assert got.sum_weights == pytest.approx(len(expanded))
