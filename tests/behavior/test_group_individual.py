import numpy as np
import pytest
from econhdfe import FixedEffect, ivreghdfe, reghdfe


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
