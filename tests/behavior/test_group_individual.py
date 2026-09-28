import numpy as np
import pytest

from pyreghdfe import FixedEffect, ivreghdfe, reghdfe
from pyreghdfe.group_individual import group_individual_singleton_mask


def _membership_fixture(seed=501, n_groups=120, n_ind=30, size=3):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), size)
    members = []
    shifts = (0, 7, 13, 19, 23)[:size]
    for g in range(n_groups):
        members.extend([(g + s) % n_ind for s in shifts])
    individual = np.asarray(members, dtype=np.int64)
    year_g = np.arange(n_groups) % 8
    year = np.repeat(year_g, size)
    return rng, group, individual, year, year_g


def _incidence(group, individual, aggregation="mean", n_groups=None, n_ind=None, slope=None):
    group = np.asarray(group)
    individual = np.asarray(individual)
    if n_groups is None:
        n_groups = int(group.max()) + 1
    if n_ind is None:
        n_ind = int(individual.max()) + 1
    A = np.zeros((n_groups, n_ind), dtype=float)
    scale = np.ones(len(group), dtype=float)
    if aggregation == "mean":
        sizes = np.bincount(group, minlength=n_groups)
        scale = 1.0 / sizes[group]
    vals = scale if slope is None else scale * np.asarray(slope)
    np.add.at(A, (group, individual), vals)
    return A


def _long(group_values, group):
    return np.asarray(group_values)[np.asarray(group)]


def _explicit_ols_beta(y, x, fe_design):
    D = np.column_stack([np.asarray(x).reshape(len(y), -1), fe_design])
    return np.linalg.lstsq(D, y, rcond=None)[0][:np.asarray(x).reshape(len(y), -1).shape[1]]


def test_group_only_matches_one_row_per_group_regression():
    rng, group, _, year, year_g = _membership_fixture(seed=502, size=4)
    ng = len(year_g)
    firm_g = np.arange(ng) % 17
    xg = rng.normal(size=(ng, 2))
    yg = xg @ np.array([0.8, -0.35]) + rng.normal(size=17)[firm_g] + rng.normal(size=8)[year_g] + rng.normal(scale=.2, size=ng)
    a = reghdfe(None, y=_long(yg, group), x=_long(xg, group),
                absorb=[_long(firm_g, group), year], group=group, tol=1e-11)
    b = reghdfe(None, y=yg, x=xg, absorb=[firm_g, year_g], tol=1e-11)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a.stderr, b.stderr, rtol=1e-9, atol=1e-9)
    assert a.nobs_raw == ng
    assert a.group_info.n_groups == ng
    assert a.group_info.n_memberships == len(group)


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
def test_group_individual_ols_matches_explicit_incidence_design(aggregation):
    rng, group, individual, year, year_g = _membership_fixture(seed=503)
    ng, ni = len(year_g), int(individual.max()) + 1
    A = _incidence(group, individual, aggregation, ng, ni)
    alpha = rng.normal(size=ni)
    tau = rng.normal(size=8)
    xg = rng.normal(size=ng)
    yg = 1.35*xg + A @ alpha + tau[year_g] + rng.normal(scale=.15, size=ng)
    y = _long(yg, group); x = _long(xg, group)
    r = reghdfe(None, y=y, x=x,
                absorb=[year, individual], group=group, individual=individual,
                aggregation=aggregation, method="lsmr", drop_singletons=False, tol=1e-12)
    Y = np.eye(8)[year_g]
    ref = _explicit_ols_beta(yg, xg, np.column_stack([Y, A]))
    np.testing.assert_allclose(r.params, ref, rtol=1e-8, atol=1e-8)
    assert r.group_info.aggregation == aggregation
    assert r.group_info.n_individuals == ni
    assert r.group_info.solver == "lsmr"


def test_group_individual_lsqr_matches_lsmr():
    rng, group, individual, year, year_g = _membership_fixture(seed=504)
    A = _incidence(group, individual, "mean")
    xg = rng.normal(size=len(year_g))
    yg = .75*xg + A @ rng.normal(size=A.shape[1]) + rng.normal(size=8)[year_g] + rng.normal(scale=.2, size=len(year_g))
    kw = dict(data=None, y=_long(yg, group), x=_long(xg, group), absorb=[year, individual],
              group=group, individual=individual, aggregation="mean", drop_singletons=False, tol=1e-11)
    a = reghdfe(**kw, method="lsmr")
    b = reghdfe(**kw, method="lsqr")
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8, atol=1e-8)


def test_individual_heterogeneous_slope_matches_explicit_design():
    rng, group, individual, year, year_g = _membership_fixture(seed=505)
    ng, ni = len(year_g), int(individual.max()) + 1
    slope = rng.normal(size=len(group))
    A = _incidence(group, individual, "mean", ng, ni)
    B = _incidence(group, individual, "mean", ng, ni, slope=slope)
    xg = rng.normal(size=ng)
    yg = 1.1*xg + A @ rng.normal(size=ni) + B @ rng.normal(scale=.3, size=ni) + rng.normal(size=8)[year_g] + rng.normal(scale=.1, size=ng)
    r = reghdfe(
        None, y=_long(yg, group), x=_long(xg, group),
        absorb=[year, FixedEffect(individual, slopes=(slope,), intercept=True, name="inventor")],
        group=group, individual=individual, aggregation="mean", method="lsmr",
        drop_singletons=False, tol=1e-12, save_fe=True,
    )
    ref = _explicit_ols_beta(yg, xg, np.column_stack([np.eye(8)[year_g], A, B]))
    np.testing.assert_allclose(r.params, ref, rtol=2e-8, atol=2e-8)
    assert r.fixed_effects.terms[-1].slopes.shape == (ni, 1)
    target = yg - xg*r.params[0] - r.residuals
    np.testing.assert_allclose(r.fixed_effects.fitted, target, rtol=2e-8, atol=2e-8)


def test_recursive_group_individual_singletons_prune_graph_to_core():
    # A three-edge path has leaf individuals at both ends. Removing the two
    # outer groups makes the middle group a leaf too, so the entire graph drops.
    membership_group = np.repeat(np.arange(3), 2).astype(np.int32)
    individual = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
    keep, dropped = group_individual_singleton_mask([], membership_group, individual)
    assert not np.any(keep)
    assert dropped == 3


def test_group_individual_iv_matches_explicit_fwl_2sls():
    rng, group, individual, year, year_g = _membership_fixture(seed=506, n_groups=180, n_ind=40)
    ng, ni = len(year_g), int(individual.max()) + 1
    A = _incidence(group, individual, "mean", ng, ni)
    F = np.column_stack([np.eye(8)[year_g], A])
    z = rng.normal(size=(ng, 2)); w = rng.normal(size=ng); v = rng.normal(size=ng)
    e = .8*z[:, 0] + .25*z[:, 1] + .2*w + A @ rng.normal(scale=.3, size=ni) + v
    y = 1.6*e + .35*w + A @ rng.normal(size=ni) + rng.normal(size=8)[year_g] + .35*v + rng.normal(size=ng)
    r = ivreghdfe(
        None, y=_long(y, group), exog=_long(w, group), endog=_long(e, group),
        instruments=_long(z, group), absorb=[year, individual], group=group,
        individual=individual, aggregation="mean", method="lsmr", drop_singletons=False,
        vce="iid", tol=1e-12,
    )
    def resid(q):
        q2 = np.asarray(q).reshape(ng, -1)
        return q2 - F @ np.linalg.lstsq(F, q2, rcond=None)[0]
    yw, ww, ew, zw = resid(y)[:, 0], resid(w)[:, 0], resid(e)[:, 0], resid(z)
    X = np.column_stack([ww, ew]); Zall = np.column_stack([ww, zw])
    Xhat = Zall @ np.linalg.lstsq(Zall, X, rcond=None)[0]
    ref = np.linalg.lstsq(Xhat, yw, rcond=None)[0]
    np.testing.assert_allclose(r.params, ref, rtol=2e-8, atol=2e-8)


def test_group_mode_requires_group_constant_regressors_and_unique_membership():
    _, group, individual, year, _ = _membership_fixture(seed=507, n_groups=20, n_ind=12)
    y = np.repeat(np.arange(20.0), 3)
    x = y.copy(); x[1] += 1
    with pytest.raises(ValueError, match="constant within group"):
        reghdfe(None, y=y, x=x, absorb=[year, individual], group=group,
                individual=individual, drop_singletons=False)
    individual2 = individual.copy(); individual2[1] = individual2[0]
    x = np.repeat(np.arange(20.0), 3)
    with pytest.raises(ValueError, match="uniquely identify"):
        reghdfe(None, y=y, x=x, absorb=[year, individual2], group=group,
                individual=individual2, drop_singletons=False)


def test_group_individual_rejects_fweights_and_gpu_backend():
    _, group, individual, year, _ = _membership_fixture(seed=508, n_groups=20, n_ind=12)
    y = np.repeat(np.arange(20.0), 3); x = np.repeat(np.linspace(0, 1, 20), 3)
    fw = np.ones(len(group), dtype=int)
    with pytest.raises(ValueError, match="fweights"):
        reghdfe(None, y=y, x=x, absorb=[year, individual], group=group,
                individual=individual, weights=fw, weight_type="fweight", drop_singletons=False)
    with pytest.raises(ValueError, match="backend must be 'numpy'"):
        reghdfe(None, y=y, x=x, absorb=[year, individual], group=group,
                individual=individual, backend="cupy", drop_singletons=False)


def test_csr_and_matrix_free_incidence_backends_agree():
    rng, group, individual, year, year_g = _membership_fixture(seed=509, n_groups=140, n_ind=35)
    A = _incidence(group, individual, "mean")
    xg = rng.normal(size=len(year_g))
    yg = 1.2*xg + A @ rng.normal(size=A.shape[1]) + rng.normal(size=8)[year_g] + rng.normal(scale=.15, size=len(year_g))
    kw = dict(data=None, y=_long(yg, group), x=_long(xg, group), absorb=[year, individual],
              group=group, individual=individual, aggregation="mean", method="lsmr",
              drop_singletons=False, tol=1e-11)
    csr = reghdfe(**kw, incidence_backend="csr")
    mf = reghdfe(**kw, incidence_backend="matrix_free")
    np.testing.assert_allclose(csr.params, mf.params, rtol=1e-8, atol=1e-8)
    assert csr.group_info.incidence_backend == "csr"
    assert mf.group_info.incidence_backend == "matrix_free"


def test_group_individual_aweight_point_estimate_matches_explicit_wls():
    rng, group, individual, year, year_g = _membership_fixture(seed=510, n_groups=150, n_ind=40)
    ng, ni = len(year_g), int(individual.max()) + 1
    A = _incidence(group, individual, "mean", ng, ni)
    F = np.column_stack([np.eye(8)[year_g], A])
    xg = rng.normal(size=ng)
    wg = rng.uniform(.2, 3.0, size=ng)
    yg = .9*xg + A @ rng.normal(size=ni) + rng.normal(size=8)[year_g] + rng.normal(scale=.3, size=ng)
    r = reghdfe(
        None, y=_long(yg, group), x=_long(xg, group), absorb=[year, individual],
        group=group, individual=individual, weights=_long(wg, group), weight_type="aweight",
        aggregation="mean", method="lsmr", drop_singletons=False, tol=1e-11,
    )
    D = np.column_stack([xg, F])
    sw = np.sqrt(wg)
    ref = np.linalg.lstsq(D*sw[:, None], yg*sw, rcond=None)[0][0]
    np.testing.assert_allclose(r.params[0], ref, rtol=2e-8, atol=2e-8)


def test_group_individual_pweight_forces_robust_like_aweight():
    rng, group, individual, year, year_g = _membership_fixture(seed=511, n_groups=140, n_ind=35)
    A = _incidence(group, individual, "mean")
    xg = rng.normal(size=len(year_g)); wg = rng.uniform(.3, 2.5, size=len(year_g))
    yg = 1.05*xg + A @ rng.normal(size=A.shape[1]) + rng.normal(size=8)[year_g] + rng.normal(size=len(year_g))
    common = dict(data=None, y=_long(yg, group), x=_long(xg, group), absorb=[year, individual],
                  group=group, individual=individual, weights=_long(wg, group), aggregation="mean",
                  method="lsmr", drop_singletons=False, tol=1e-11)
    p = reghdfe(**common, weight_type="pweight", vce="iid")
    a = reghdfe(**common, weight_type="aweight", vce="robust")
    np.testing.assert_allclose(p.params, a.params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(p.stderr, a.stderr, rtol=1e-10, atol=1e-10)
