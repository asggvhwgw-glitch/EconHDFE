import numpy as np
import pytest
from scipy.stats import f as f_dist
from econhdfe import olshdfe


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


def test_zero_variance_contrast_is_not_reported_as_certain_inference():
    from econhdfe.postestimation import linear_combination
    with pytest.raises(ValueError, match='zero estimated variance'):
        linear_combination([0.], [[0.]], [1.])
    with pytest.raises(ValueError, match='degrees of freedom'):
        linear_combination([0.], [[1.]], [1.], df=0)
