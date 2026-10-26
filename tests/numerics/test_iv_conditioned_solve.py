"""Observation-space IV references for ill-conditioned included controls.

These small references never form normal equations, import solver helpers, or
reuse a fitted model's degrees of freedom. They exercise the dense 2SLS path.
"""

import numpy as np
import pytest
from scipy.linalg import hadamard
from scipy.stats import chi2

from econhdfe import ExecutionConfig, ivhdfe
from econhdfe.compute.weights import prepare_weights
from econhdfe.errors import InferenceError, UnderidentifiedError
from econhdfe.models.linear_iv.estimators import fit_iv_kclass


@pytest.mark.parametrize("controls", [1, 2])
def test_small_qr_partial_matches_observation_space(controls, monkeypatch):
    from types import SimpleNamespace
    from econhdfe.models.linear_iv import diagnostics

    rng = np.random.default_rng(726)
    c = rng.normal(size=(80, controls))
    z = rng.normal(size=(80, 3)) + .2 * c[:, :1]
    e = rng.normal(size=(80, 2)) + .3 * z[:, :2]
    instruments = np.column_stack((c, z))
    _, r = np.linalg.qr(np.column_stack((instruments, e)), mode="reduced")
    qr = SimpleNamespace(R=r, zscale=np.ones(instruments.shape[1]), escale=np.ones(2))
    a = np.column_stack((e, z))
    expected = a - c @ np.linalg.lstsq(c, a, rcond=None)[0]
    if controls == 1:
        def forbidden(*args, **kwargs):
            pytest.fail("scalar diagnostic solve must not call LAPACK trtrs")
        monkeypatch.setattr(diagnostics.la, "solve_triangular", forbidden)
    np.testing.assert_allclose(diagnostics._partial_from_solver(e, c, z, qr),
                               expected, rtol=1e-12, atol=1e-12)


def _cluster_reference(y, c, e, z, groups, weights=None, df_absorbed=0,
                       vce="cluster", second_groups=None):
    """Define 2SLS in observation space and accumulate coefficient influences."""
    n = len(y)
    sw = np.ones(n) if weights is None else np.sqrt(weights)
    x = np.column_stack((c, e)) * sw[:, None]
    instruments = np.column_stack((c, z)) * sw[:, None]
    # Scaling is an invertible change of units, including for the reference.
    xscale = np.linalg.norm(x, axis=0)
    x = x / xscale
    instruments = instruments / np.linalg.norm(instruments, axis=0)
    qz, _ = np.linalg.qr(instruments, mode="reduced")
    projected = qz @ (qz.T @ x)
    u, singular, vt = np.linalg.svd(projected, full_matrices=False)
    assert singular[-1] > singular[0] * 1e-9
    inverse = (vt.T / singular) @ u.T
    beta_scaled = inverse @ (y * sw)
    residual = y - (x @ beta_scaled) / sw
    # inverse @ diag(weighted residual) is each observation's coefficient
    # influence; outer products of cluster sums give the uncorrected VCE.
    scores = inverse.T * (residual * sw)[:, None]
    denominator = n - x.shape[1] - df_absorbed
    if vce == "iid":
        covariance = inverse @ inverse.T * np.sum((residual * sw) ** 2) / denominator
    elif vce == "robust":
        covariance = scores.T @ scores * n / denominator
    else:
        def group_meat(codes):
            sums = np.array([scores[codes == g].sum(axis=0) for g in np.unique(codes)])
            return sums.T @ sums
        ng = len(np.unique(groups))
        raw = group_meat(groups)
        if second_groups is not None:
            _, intersection = np.unique(np.column_stack((groups, second_groups)),
                                        axis=0, return_inverse=True)
            raw += group_meat(second_groups) - group_meat(intersection)
            ng = min(ng, len(np.unique(second_groups)))
        covariance = raw * ng / (ng - 1) * (n - 1) / denominator
    return beta_scaled / xscale, covariance / xscale[:, None] / xscale[None, :], residual


def _problem(delta=1e-5):
    rng = np.random.default_rng(20261022)
    n = 384
    base = rng.normal(size=(n, 2))
    c = np.column_stack((base[:, 0], base[:, 0] + delta * base[:, 1]))
    z = rng.normal(size=(n, 2))
    shock = rng.normal(size=n)
    e = (z[:, 0] + .6 * z[:, 1] + .2 * base[:, 0] + .35 * shock)[:, None]
    y = c @ np.array([.4, -.2]) + 1.3 * e[:, 0] + .25 * shock + rng.normal(size=n)
    groups = np.arange(n) % 24
    weights = rng.uniform(.3, 2.5, n)
    return y, c, e, z, groups, weights


def _cluster_hansen_reference(y, c, e, z, groups, residual, weights=None):
    """Efficient moment correction in orthonormal observation-space bases.

    QR of X changes its parameterization without changing its column space.
    Whiten instrument moments by the cluster-score covariance, then minimize
    their squared norm by least squares, without forming D' W D.
    """
    sw = np.ones(len(y)) if weights is None else np.sqrt(weights)
    x = np.column_stack((c, e)) * sw[:, None]
    instruments = np.column_stack((c, z)) * sw[:, None]
    qx, _ = np.linalg.qr(x / np.linalg.norm(x, axis=0), mode="reduced")
    qz, _ = np.linalg.qr(instruments / np.linalg.norm(instruments, axis=0),
                         mode="reduced")
    scores = qz * (residual * sw)[:, None]
    sums = np.array([scores[groups == g].sum(axis=0) for g in np.unique(groups)])
    values, vectors = np.linalg.eigh(sums.T @ sums)
    assert values[0] > values[-1] * 1e-10
    whitener = vectors.T / np.sqrt(values)[:, None]
    moments = whitener @ (qz.T @ (residual * sw))
    design = whitener @ (qz.T @ qx)
    correction = np.linalg.lstsq(design, moments, rcond=None)[0]
    efficient_moments = moments - design @ correction
    return float(efficient_moments @ efficient_moments)


def _assert_hansen_matches_reference(got, expected):
    overid = got[6]["overidentification"]
    assert overid["df"] == 1
    np.testing.assert_allclose(overid["stat"], expected, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(overid["pvalue"], chi2.sf(expected, 1),
                               rtol=2e-5, atol=2e-7)


def _assert_fit_matches_reference(got, expected):
    beta, covariance, residual = expected
    # Compare the entire VCE on marginal-SE scales, including off-diagonal
    # entries near zero; elementwise relative tolerances are misleading there.
    se = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(got[0], beta, rtol=2e-6, atol=2e-7)
    assert np.max(np.abs(got[1] - covariance) / np.outer(se, se)) < 2e-5
    assert np.linalg.norm(got[2] - residual) / np.linalg.norm(residual) < 2e-7
    assert got[3] == len(beta)


@pytest.mark.parametrize("delta", [.4, 1e-5])
@pytest.mark.parametrize("weighted", [False, True])
def test_strong_iv_with_correlated_controls_matches_full_cluster_reference(delta, weighted):
    y, c, e, z, groups, weights = _problem(delta)
    weights = weights if weighted else None
    expected = _cluster_reference(y, c, e, z, groups, weights, df_absorbed=3)
    got = fit_iv_kclass(y, c, e, z, weights=weights, vce="cluster",
                       clusters=[groups], df_absorbed=3)
    _assert_fit_matches_reference(got, expected)
    assert got[4] == 23
    _assert_hansen_matches_reference(
        got, _cluster_hansen_reference(y, c, e, z, groups, expected[2], weights),
    )

    # Partial R2 is checked in observation space independently of IV moments.
    sw = np.ones(len(y)) if weights is None else np.sqrt(weights)
    qc, _ = np.linalg.qr(c * sw[:, None], mode="reduced")
    qz, _ = np.linalg.qr(np.column_stack((c, z)) * sw[:, None], mode="reduced")
    ew = e[:, 0] * sw
    restricted = ew - qc @ (qc.T @ ew)
    unrestricted = ew - qz @ (qz.T @ ew)
    partial_r2 = 1 - (unrestricted @ unrestricted) / (restricted @ restricted)
    assert partial_r2 > .8
    np.testing.assert_allclose(got[5]["diagnostics"][0]["partial_r2"],
                               partial_r2, rtol=2e-7, atol=2e-8)


def test_conditioned_iv_preserves_units_and_weighted_cluster_inference():
    y, c, e, z, groups, weights = _problem()
    expected = _cluster_reference(y, c, e, z, groups, weights)
    cscale = np.array([1e8, 1e-7])
    escale = np.array([1e4])
    zscale = np.array([1e6, 1e-5])
    got = list(fit_iv_kclass(y, c * cscale, e * escale, z * zscale,
                            weights=weights, vce="cluster", clusters=[groups]))
    scale = np.concatenate((cscale, escale))
    got[0] = got[0] * scale
    got[1] = got[1] * scale[:, None] * scale[None, :]
    _assert_fit_matches_reference(got, expected)


def test_conditioned_frequency_weights_match_literal_observation_replication():
    y, c, e, z, groups, _ = _problem()
    frequency = np.arange(len(y)) % 3 + 1
    wi = prepare_weights(frequency, len(y), "fweight")
    rows = np.repeat(np.arange(len(y)), frequency)
    yy, cc, ee, zz, gg = y[rows], c[rows], e[rows], z[rows], groups[rows]
    expected = _cluster_reference(yy, cc, ee, zz, gg, df_absorbed=3)
    got = list(fit_iv_kclass(y, c, e, z, weights=wi.estimation, weight_info=wi,
                            vce="cluster", clusters=[groups], df_absorbed=3))
    got[2] = got[2][rows]
    _assert_fit_matches_reference(got, expected)
    _assert_hansen_matches_reference(
        got, _cluster_hansen_reference(yy, cc, ee, zz, gg, expected[2]),
    )


@pytest.mark.parametrize("repetitions", [1, 16])
def test_repeating_conditioned_design_does_not_change_rank_decision(repetitions):
    y, c, e, z, groups, _ = _problem()
    expected = _cluster_reference(y, c, e, z, groups)
    # Existing clusters are also repeated: their uncorrected covariance remains
    # the same; only the analytically known finite-sample correction changes.
    yy = np.tile(y, repetitions)
    cc, ee, zz = (np.tile(a, (repetitions, 1)) for a in (c, e, z))
    gg = np.tile(groups, repetitions)
    got = fit_iv_kclass(yy, cc, ee, zz, vce="cluster", clusters=[gg])
    n, k = len(y), c.shape[1] + e.shape[1]
    factor = ((len(yy) - 1) / (len(yy) - k)) / ((n - 1) / (n - k))
    _assert_fit_matches_reference(got, (expected[0], expected[1] * factor,
                                        np.tile(expected[2], repetitions)))


def test_full_rank_weak_instruments_are_not_declared_underidentified():
    # Exact orthogonal columns ensure weakness is not due to ill-conditioned
    # controls or accidental sample correlations. Partial R2 is about 1e-6.
    q = hadamard(256).astype(float)
    c, z = q[:, :1], q[:, 1:3]
    e = (1e-3 * q[:, 1] + q[:, 3])[:, None]
    y = .4 * c[:, 0] + 1.2 * e[:, 0] + .3 * q[:, 4]
    groups = np.random.default_rng(721).permutation(len(y)) % 16
    got = fit_iv_kclass(y, c, e, z, vce="cluster", clusters=[groups])
    _assert_fit_matches_reference(got, _cluster_reference(y, c, e, z, groups))
    np.testing.assert_allclose(got[0], [.4, 1.2], rtol=0, atol=1e-8)
    np.testing.assert_allclose(got[5]["diagnostics"][0]["partial_r2"],
                               1e-6 / (1 + 1e-6), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("rescaled", [False, True])
def test_full_role_rank_but_missing_iv_direction_is_underidentified(rescaled):
    q = hadamard(256).astype(float)
    c, z = q[:, :1], q[:, 1:3]
    e = np.column_stack((q[:, 1] + q[:, 3], q[:, 1] + q[:, 4]))
    y = .4 * c[:, 0] + e @ np.array([1.2, -.7]) + .3 * q[:, 5]
    # Both X and Z have full column rank, but the excluded instruments project
    # the two endogenous variables onto the same single direction.
    assert np.linalg.matrix_rank(np.column_stack((c, e))) == 3
    assert np.linalg.matrix_rank(np.column_stack((c, z))) == 3
    if rescaled:
        c = c * 1e8
        e = e * np.array([1e-6, 1e7])
        z = z * np.array([1e4, 1e-5])
    with pytest.raises(UnderidentifiedError):
        fit_iv_kclass(y, c, e, z)


@pytest.mark.parametrize("vce", ["iid", "robust", "two_way"])
def test_conditioned_other_covariances_match_observation_space_reference(vce):
    y, c, e, z, groups, weights = _problem()
    second = np.arange(len(y)) // 24 if vce == "two_way" else None
    kind = "cluster" if vce == "two_way" else vce
    expected = _cluster_reference(y, c, e, z, groups, weights, df_absorbed=2,
                                  vce=kind, second_groups=second)
    if second is not None:
        # This oracle compares the raw CGM definition. Ensure the selected DGP
        # does not require an additional, unit-dependent PSD repair convention.
        sd = np.sqrt(np.diag(expected[1]))
        assert np.linalg.eigvalsh(expected[1] / np.outer(sd, sd))[0] > 0
    got = fit_iv_kclass(y, c, e, z, weights=weights, vce=kind, df_absorbed=2,
                       clusters=[groups, second] if second is not None else None)
    assert got[7]["numerical_solver"] == "stable_tsqr"
    _assert_fit_matches_reference(got, expected)


@pytest.mark.parametrize("endogenous_delta", [.4, 1e-6])
def test_multi_endogenous_diagnostics_preserve_equivalent_spaces(endogenous_delta):
    rng = np.random.default_rng(70324)
    n = 600
    raw_controls = rng.normal(size=(n, 2))
    c = np.column_stack((raw_controls[:, 0],
                         raw_controls[:, 0] + 1e-5 * raw_controls[:, 1]))
    # Explicitly transform the actual float inputs, not the latent DGP columns.
    good_c = np.column_stack((c[:, 0], (c[:, 1] - c[:, 0]) / 1e-5))
    z = rng.normal(size=(n, 3))
    shocks = rng.normal(size=(n, 2))
    e0 = z[:, :2] + z[:, 2:] @ np.array([[.3, -.2]]) + .5 * shocks
    e = e0 @ np.array([[1., 1.], [0., endogenous_delta]])
    y = raw_controls @ np.array([.3, -.1]) + e0 @ np.array([1.2, -.7])
    y += .25 * shocks[:, 0] + rng.normal(size=n)
    groups = np.arange(n) % 30
    kwargs = dict(vce="cluster", clusters=[groups], df_absorbed=2)
    got = fit_iv_kclass(y, c, e, z, **kwargs)
    control_reference = fit_iv_kclass(y, good_c, e, z, **kwargs)
    span_reference = fit_iv_kclass(y, good_c, e0, z, **kwargs)
    assert got[7]["numerical_solver"] == "stable_tsqr"

    # KP/CD concern the rank of an endogenous span and are invariant to an
    # invertible mixing of E. Near-collinear E must not corrupt their whitening.
    kp_keys = ("rk_lm", "rk_wald_chi2", "rk_wald_f")
    for key in kp_keys:
        reference = span_reference[6]["kleibergen_paap"][key]
        assert np.isfinite(reference) and reference > 0
        np.testing.assert_allclose(got[6]["kleibergen_paap"][key], reference,
                                   rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(got[6]["cragg_donald_f"],
                               span_reference[6]["cragg_donald_f"], rtol=2e-4, atol=1e-6)

    # Individual-variable diagnostics are not invariant to mixing endogenous
    # columns. Compare the same E with only the control basis changed instead.
    keys = ("partial_r2", "shea_partial_r2", "f_classic", "f_robust")
    for actual, reference in zip(got[5]["diagnostics"], control_reference[5]["diagnostics"]):
        for key in keys:
            assert np.isfinite(actual[key]) and np.isfinite(reference[key])
            np.testing.assert_allclose(actual[key], reference[key], rtol=2e-4, atol=1e-8)
    actual_sw = got[6]["sanderson_windmeijer"]
    reference_sw = control_reference[6]["sanderson_windmeijer"]
    assert len(actual_sw) == len(reference_sw) == 2
    for actual, reference in zip(actual_sw, reference_sw):
        assert actual["df1"] == reference["df1"] == 2
        for statistic in ("ap", "sw"):
            for key in ("partial_r2", "f_classic", "f_robust"):
                assert np.isfinite(actual[statistic][key])
                np.testing.assert_allclose(actual[statistic][key], reference[statistic][key],
                                           rtol=2e-4, atol=1e-8)


@pytest.mark.parametrize("fe_topology", ["unbalanced", "weak_chain"])
def test_hdfe_shared_loading_preserves_weak_control_direction(fe_topology):
    rng = np.random.default_rng(20261023)
    if fe_topology == "weak_chain":
        # Dense diagonal cells connected only by one observation in each
        # direction per adjacent pair have a slow alternating-projection mode.
        levels = 16
        diagonal = np.repeat(np.arange(levels), 40)
        left, right = np.arange(levels - 1), np.arange(1, levels)
        groups = [np.r_[diagonal, left, right], np.r_[diagonal, right, left]]
        n = len(groups[0])
        slow_loading = 30 * np.cos(np.pi * (groups[1] + .5) / levels)
        execution = dict(method="map", acceleration="cg", max_iter=512)
    else:
        n = 720
        groups = []
        for levels in (18, 11, 9):
            probability = np.arange(1, levels + 1, dtype=float)
            probability /= probability.sum()
            # Each FE level occurs at least twice; remaining counts are unequal.
            codes = np.r_[np.repeat(np.arange(levels), 2),
                          rng.choice(levels, n - 2 * levels, p=probability)]
            rng.shuffle(codes)
            groups.append(codes)
        slow_loading = 0.
        execution = dict(max_iter=16000)
    dummy = np.column_stack([np.eye(g.max() + 1)[g] for g in groups])
    qf, singular, _ = np.linalg.svd(dummy, full_matrices=False)
    fe_rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
    qf = qf[:, :fe_rank]

    def residualize(a):
        # Reorthogonalize the small explicit-dummy reference; no N-by-N P_FE.
        within = a - qf @ (qf.T @ a)
        return within - qf @ (qf.T @ within)

    q, _ = np.linalg.qr(residualize(rng.normal(size=(n, 7))), mode="reduced")
    q *= np.sqrt(n)
    loading = 50 * (dummy @ rng.normal(size=dummy.shape[1]))
    c = np.column_stack((loading + q[:, 0],
                         loading + q[:, 0] + 1e-5 * (q[:, 1] + slow_loading)))
    z = q[:, 2:4] + dummy @ rng.normal(size=(dummy.shape[1], 2))
    e = (.9 * q[:, 2] + .4 * q[:, 3] + .5 * q[:, 4]
         + dummy @ rng.normal(size=dummy.shape[1]))[:, None]
    y = c @ np.array([.7, -.4]) + 1.1 * e[:, 0] + .3 * q[:, 4] + .4 * q[:, 5]
    y += dummy @ rng.normal(size=dummy.shape[1])
    cluster = rng.permutation(n) % 24
    # Project actual rounded inputs in well-scaled contrast coordinates. Doing
    # two independent FE subtractions on nearly identical controls amplifies
    # the reference's rounding error by 1e5, obscuring the solver error tested.
    contrast = np.column_stack((c[:, 0], (c[:, 1] - c[:, 0]) * 1e5))
    within = residualize(np.column_stack((y, contrast, e, z)))
    beta, covariance, residual = _cluster_reference(
        within[:, 0], within[:, 1:3], within[:, 3:4], within[:, 4:6],
        cluster, df_absorbed=fe_rank)
    units = np.array([[1., -1e5, 0.], [0., 1e5, 0.], [0., 0., 1.]])
    expected = units @ beta, units @ covariance @ units.T, residual
    got = ivhdfe(y=y, exog=c, endog=e, instruments=z, absorb=groups,
                 cluster=[cluster], vce="cluster", dof_method="exact",
                 drop_singletons=False, tol=1e-12, **execution,
                 execution_config=ExecutionConfig(threads=1, memory_budget_mb=64, cache="off"))
    assert got.converged and got.nobs == n and got.rank == 3
    assert got.df_absorbed == fe_rank and got.dropped_singletons == 0
    # The O(1) control coefficients make absorption error in the 1e-5 contrast
    # visible; random large weak-direction coefficients would hide that error.
    np.testing.assert_allclose(expected[0], [.7, -.4, 1.1], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(got.params, expected[0], rtol=1e-5, atol=1e-6)
    reference_se = np.sqrt(np.diag(expected[1]))
    assert np.max(np.abs(got.vcov - expected[1]) / np.outer(reference_se, reference_se)) < 5e-4
    assert np.linalg.norm(got.residuals - expected[2]) / np.linalg.norm(expected[2]) < 1e-6


def test_singleton_cluster_codes_are_counted_in_accumulator_budget(monkeypatch):
    from econhdfe.models.linear_iv import stable_inference

    # No repeated-cell score sums are needed. Retaining the codes, lookups and
    # HC0 matrices across subsets still exceeds this deliberately tiny budget.
    singleton = np.arange(16)
    monkeypatch.setattr(stable_inference, "_SCORE_BYTES", 400)
    with pytest.raises(InferenceError) as caught:
        stable_inference._cluster_accumulators([singleton, singleton, singleton], p=3)
    assert caught.value.details["required_bytes"] > 400
    assert caught.value.details["budget_bytes"] == 400
