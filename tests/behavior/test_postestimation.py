import numpy as np
import pytest
from scipy.stats import f as f_dist

from econhdfe import ivhdfe, olshdfe


def _fit():
    rng = np.random.default_rng(123)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    g = np.repeat(np.arange(30), 10)
    fe = rng.normal(size=30)[g]
    y = 1.5 * x1 - 0.5 * x2 + fe + rng.normal(scale=0.4, size=n)
    return olshdfe(y=y, x=np.column_stack([x1, x2]), absorb=[g])


def test_named_linear_combination_matches_matrix_algebra():
    r = _fit()
    got = r.linear_combination({"x1": 1.0, "x2": -1.0})
    w = np.array([1.0, -1.0])
    assert got.estimate == pytest.approx(float(w @ r.params))
    assert got.std_error == pytest.approx(float(np.sqrt(w @ r.vcov @ w)))
    assert got.ci_low < got.estimate < got.ci_high


def test_wald_test_matches_manual_f_statistic():
    r = _fit()
    R = np.eye(2)
    diff = R @ r.params
    S = R @ r.vcov @ R.T
    chi2_stat = float(diff @ np.linalg.solve(S, diff))
    expected_f = chi2_stat / 2
    got = r.wald_test(R)
    assert got.distribution == "F"
    assert got.df_num == 2
    assert got.df_denom == pytest.approx(r.df_resid)
    assert got.statistic == pytest.approx(expected_f)
    assert got.p_value == pytest.approx(float(f_dist.sf(expected_f, 2, r.df_resid)))


def test_named_wald_restrictions_and_nonzero_values():
    r = _fit()
    got = r.wald_test(
        [{"x1": 1.0}, {"x2": 1.0}], values=[1.5, -0.5], distribution="chi2"
    )
    assert got.distribution == "chi2"
    assert got.df_num == 2
    assert got.df_denom is None
    assert 0.0 <= got.p_value <= 1.0


def test_postestimation_rejects_unknown_names_and_bad_shapes():
    r = _fit()
    with pytest.raises(ValueError, match="unknown coefficient"):
        r.linear_combination({"missing": 1.0})
    with pytest.raises(ValueError, match="restriction matrix"):
        r.wald_test(np.ones((2, 3)))


def test_iv_named_restriction_uses_reported_parameter_order():
    rng = np.random.default_rng(124)
    n = 800
    g = np.repeat(np.arange(80), 10)
    w = rng.normal(size=n)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    endog = 0.8 * z + 0.3 * w + v
    y = 0.4 * w + 1.3 * endog + rng.normal(size=80)[g] + 0.4 * v + rng.normal(size=n)
    r = ivhdfe(
        y=y, exog=w, endog=endog, instruments=z, absorb=g, vce="robust"
    )

    assert len(r.names) == 2
    got = r.linear_combination({r.names[0]: 1.0, r.names[1]: -1.0})
    weights = np.array([1.0, -1.0])
    assert got.estimate == pytest.approx(float(weights @ r.params))
    assert got.std_error == pytest.approx(float(np.sqrt(weights @ r.vcov @ weights)))

    joint = r.wald_test([{r.names[0]: 1.0}, {r.names[1]: 1.0}])
    assert joint.df_num == 2
    assert 0.0 <= joint.p_value <= 1.0


def test_singular_wald_uses_consistent_rank_and_rejects_inconsistent_null():
    from econhdfe.postestimation import wald_test
    beta = np.array([.4, -.2])
    V = np.array([[.25, .03], [.03, .16]])
    one = wald_test(beta, V, [1., -1.], df_resid=20)
    repeated = wald_test(beta, V, [[1., -1.], [1e12, -1e12]], df_resid=20)
    assert repeated.df_num == 1
    assert repeated.statistic == pytest.approx(one.statistic)
    with pytest.raises(ValueError, match='no estimated sampling variance'):
        wald_test(beta, V, [[1., -1.], [1., -1.]], values=[0., 1.], df_resid=20)
    with pytest.raises(ValueError, match='positive semidefinite'):
        wald_test(beta, np.diag([1., -1.]), df_resid=20)


@pytest.mark.parametrize('kind', ['linear', 'wald'])
def test_nonfinite_covariance_and_duplicate_names_are_rejected(kind):
    from econhdfe.postestimation import linear_combination, wald_test
    fn = linear_combination if kind == 'linear' else wald_test
    with pytest.raises(ValueError, match='finite'):
        fn([1., 2.], np.diag([1., np.nan]), [1., 0.])
    with pytest.raises(ValueError, match='unique'):
        fn([1., 2.], np.eye(2), {'x': 1.}, names=('x', 'x'))
    with pytest.raises(ValueError, match='finite'):
        fn([1., 2.], np.eye(2), [np.nan, 0.])


def test_zero_variance_contrast_is_not_reported_as_certain_inference():
    from econhdfe.postestimation import linear_combination
    with pytest.raises(ValueError, match='zero estimated variance'):
        linear_combination([0.], [[0.]], [1.])
    with pytest.raises(ValueError, match='degrees of freedom'):
        linear_combination([0.], [[1.]], [1.], df=0)


def test_result_postestimation_uses_structured_public_error_boundary():
    from econhdfe.errors import EconHDFEError
    r = _fit()
    with pytest.raises(EconHDFEError) as exc:
        r.linear_combination([1.0, np.nan])
    assert exc.value.stage == 'postestimation'
    with pytest.raises(EconHDFEError) as exc:
        r.wald_test(np.ones((1, 3)))
    assert exc.value.stage == 'postestimation'
