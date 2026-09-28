import numpy as np
import pytest
from pyreghdfe import reghdfe, ivreghdfe


def _weighted_fixture(seed=880, n=1200):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 80, n)
    year = rng.integers(0, 20, n)
    fw = rng.integers(1, 4, n)
    X = rng.normal(size=(n, 2))
    af = rng.normal(size=80)
    at = rng.normal(size=20)
    y = X @ np.array([0.7, -0.3]) + af[firm] + at[year] + rng.normal(size=n)
    return rng, firm, year, fw, X, y, af, at


def test_fweight_ols_iid_robust_and_cluster_equal_expanded_data():
    _, firm, year, fw, X, y, _, _ = _weighted_fixture()
    idx = np.repeat(np.arange(len(y)), fw)
    for vce, cluster_w, cluster_e in [
        ("iid", None, None),
        ("robust", None, None),
        ("cluster", firm, firm[idx]),
    ]:
        a = reghdfe(
            None, y=y, x=X, absorb=[firm, year], weights=fw,
            weight_type="fweight", vce=vce, cluster=cluster_w, tol=1e-11,
        )
        b = reghdfe(
            None, y=y[idx], x=X[idx], absorb=[firm[idx], year[idx]],
            vce=vce, cluster=cluster_e, tol=1e-11,
        )
        np.testing.assert_allclose(a.params, b.params, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(a.stderr, b.stderr, rtol=1e-10, atol=1e-10)
        assert a.nobs == b.nobs == int(fw.sum())
        assert a.df_resid == pytest.approx(b.df_resid)
        for field in ("r2", "r2_within", "r2_adjusted", "r2_adjusted_within",
                      "rss", "tss", "tss_within", "mss", "rmse", "loglike",
                      "loglike_null", "f_statistic", "f_pvalue", "df_model",
                      "df_resid_fit", "vcov_rank"):
            np.testing.assert_allclose(getattr(a, field), getattr(b, field),
                                       rtol=2e-11, atol=2e-11, err_msg=field)


def test_fweight_iv_and_weak_iv_diagnostics_equal_expanded_data():
    rng, firm, year, fw, _, _, af, at = _weighted_fixture(seed=881, n=1500)
    n = len(firm)
    z = rng.normal(size=(n, 2))
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.6*z[:, 0] + 0.25*z[:, 1] + 0.2*w + v
    y = 1.4*x + 0.3*w + af[firm] + at[year] + 0.4*v + rng.normal(size=n)
    idx = np.repeat(np.arange(n), fw)
    a = ivreghdfe(
        None, y=y, exog=w, endog=x, instruments=z, absorb=[firm, year],
        weights=fw, weight_type="fweight", vce="robust", tol=1e-11,
    )
    b = ivreghdfe(
        None, y=y[idx], exog=w[idx], endog=x[idx], instruments=z[idx],
        absorb=[firm[idx], year[idx]], vce="robust", tol=1e-11,
    )
    np.testing.assert_allclose(a.params, b.params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(a.stderr, b.stderr, rtol=1e-10, atol=1e-10)
    for key in ("partial_r2", "f_classic", "f_robust"):
        np.testing.assert_allclose(
            a.first_stage["diagnostics"][0][key],
            b.first_stage["diagnostics"][0][key],
            rtol=1e-10, atol=1e-10,
        )
    np.testing.assert_allclose(a.diagnostics["cragg_donald_f"], b.diagnostics["cragg_donald_f"], rtol=1e-10)
    np.testing.assert_allclose(
        a.diagnostics["kleibergen_paap"]["rk_wald_f"],
        b.diagnostics["kleibergen_paap"]["rk_wald_f"],
        rtol=1e-10,
    )


def test_aweight_normalization_is_scale_invariant_and_tracks_raw_sumweights():
    rng, firm, year, _, X, y, _, _ = _weighted_fixture(seed=883)
    aw = rng.uniform(0.2, 3.0, len(y))
    a = reghdfe(None, y=y, x=X, absorb=[firm, year], weights=aw,
                weight_type="aweight", vce="robust", tol=1e-11)
    b = reghdfe(None, y=y, x=X, absorb=[firm, year], weights=100*aw,
                weight_type="aweight", vce="robust", tol=1e-11)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.stderr, b.stderr, rtol=1e-12, atol=1e-12)
    assert a.nobs == b.nobs == len(y)
    assert b.sum_weights == pytest.approx(100*a.sum_weights)


def test_pweight_forces_robust_when_unadjusted_requested():
    rng, firm, year, _, X, y, _, _ = _weighted_fixture(seed=884)
    pw = rng.uniform(0.2, 3.0, len(y))
    p = reghdfe(None, y=y, x=X, absorb=[firm, year], weights=pw,
                weight_type="pweight", vce="iid", tol=1e-11)
    a = reghdfe(None, y=y, x=X, absorb=[firm, year], weights=pw,
                weight_type="aweight", vce="robust", tol=1e-11)
    np.testing.assert_allclose(p.params, a.params, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p.stderr, a.stderr, rtol=1e-12, atol=1e-12)
    assert p.weight_type == "pweight"


def test_fweight_validation_requires_positive_integers():
    y = np.arange(6.0)
    x = np.arange(6.0)[:, None]
    fe = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="integer-valued"):
        reghdfe(None, y=y, x=x, absorb=fe, weights=np.full(6, 1.5), weight_type="fweight")
    with pytest.raises(ValueError, match="strictly positive"):
        reghdfe(None, y=y, x=x, absorb=fe, weights=np.array([1, 1, 1, 1, 1, 0]), weight_type="fweight")


def test_save_fe_reconstructs_absorbed_component_and_full_fitted_values():
    rng = np.random.default_rng(885)
    n = 5000
    firm = rng.integers(0, 200, n)
    year = rng.integers(0, 25, n)
    X = rng.normal(size=(n, 2))
    y = X @ np.array([0.8, -0.45]) + rng.normal(size=200)[firm] + rng.normal(size=25)[year] + rng.normal(size=n)
    r = reghdfe(None, y=y, x=X, absorb=[firm, year], save_fe=True, tol=1e-11)
    assert r.fixed_effects is not None
    assert r.fixed_effects.converged
    assert r.fixed_effects.reconstruction_error < 1e-8
    target = y - X @ r.params - r.residuals
    np.testing.assert_allclose(r.fixed_effects.fitted, target, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(r.fitted, y - r.residuals, rtol=0, atol=0)


def test_save_fe_handles_heterogeneous_slopes_in_original_basis():
    from pyreghdfe import FixedEffect
    rng = np.random.default_rng(886)
    g = np.repeat(np.arange(80), 30)
    n = len(g)
    trend = rng.normal(size=n)
    x = rng.normal(size=n)
    alpha = rng.normal(size=80)
    gamma = rng.normal(scale=0.3, size=80)
    y = 1.25*x + alpha[g] + gamma[g]*trend + rng.normal(scale=0.2, size=n)
    r = reghdfe(
        None, y=y, x=x, absorb=[FixedEffect(g, slopes=(trend,), intercept=True)],
        save_fe=True, tol=1e-11,
    )
    term = r.fixed_effects.terms[0]
    assert term.intercept.shape == (80,)
    assert term.slopes.shape == (80, 1)
    target = y - x*r.params[0] - r.residuals
    np.testing.assert_allclose(term.contribution, target, rtol=1e-8, atol=1e-8)


def test_save_fe_iv_reconstructs_structural_fixed_effect_part():
    rng = np.random.default_rng(887)
    n = 4000
    firm = rng.integers(0, 150, n)
    year = rng.integers(0, 20, n)
    z = rng.normal(size=(n, 2)); w = rng.normal(size=n); v = rng.normal(size=n)
    x = .7*z[:, 0] + .2*z[:, 1] + .15*w + v
    y = 1.5*x + .25*w + rng.normal(size=150)[firm] + rng.normal(size=20)[year] + .4*v + rng.normal(size=n)
    r = ivreghdfe(None, y=y, exog=w, endog=x, instruments=z, absorb=[firm, year], save_fe=True, tol=1e-11)
    xb = w*r.params[0] + x*r.params[1]
    target = y - xb - r.residuals
    np.testing.assert_allclose(r.fixed_effects.fitted, target, rtol=1e-8, atol=1e-8)


def test_iv_fweight_rejects_kernel_covariance():
    rng, firm, year, _, _, _, af, at = _weighted_fixture(seed=991, n=500)
    z = rng.normal(size=(500, 2))
    w = rng.normal(size=500)
    v = rng.normal(size=500)
    x = 0.6*z[:, 0] + 0.2*z[:, 1] + 0.2*w + v
    y = 1.3*x + 0.25*w + af[firm] + at[year] + 0.3*v + rng.normal(size=500)
    fw = np.full(500, 2)
    with pytest.raises(ValueError, match="fweights with HAC"):
        ivreghdfe(
            None, y=y, exog=w, endog=x, instruments=z,
            absorb=[firm, year], weights=fw, weight_type="fweight",
            vce="hac", time=year, bandwidth=2,
        )
