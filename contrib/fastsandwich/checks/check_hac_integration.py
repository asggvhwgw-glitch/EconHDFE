"""Independent tests of public HAC adapters, especially weighted scores.

The released statsmodels covariance is the integration oracle; a separate
agent owns explicit double-sum tests of the underlying mathematical kernel.
"""

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac as reference_cov_hac

from fastsandwich import cov_hac


def example_design():
    rng = np.random.default_rng(63291)
    nobs = 81
    exog = sm.add_constant(rng.normal(size=(nobs, 3)))
    residual = rng.standard_normal(nobs)
    for index in range(1, nobs):
        residual[index] += 0.6 * residual[index - 1]
    residual *= np.linspace(0.5, 2.0, nobs)
    endog = exog @ np.array([0.4, -0.2, 0.6, 1.0]) + residual
    weights = np.geomspace(0.2, 4.0, nobs)
    return endog, exog, weights


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("nlags", [None, 0, 1, 12, 81, 90])
@pytest.mark.parametrize("correction", [False, True])
def test_covariance_matches_released_statsmodels(weighted, wrapped, nlags, correction):
    endog, exog, weights = example_design()
    model = sm.WLS(endog, exog, weights=weights) if weighted else sm.OLS(endog, exog)
    fitted = model.fit()
    result = fitted if wrapped else fitted._results
    params_before = fitted.params.copy()
    residual_before = fitted.resid.copy()
    expected = reference_cov_hac(result, nlags=nlags, use_correction=correction)
    actual = cov_hac(result, nlags=nlags, use_correction=correction)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=3e-13)
    np.testing.assert_allclose(np.sqrt(np.diag(actual)), np.sqrt(np.diag(expected)),
                               rtol=5e-11, atol=3e-13)
    np.testing.assert_array_equal(fitted.params, params_before)
    np.testing.assert_array_equal(fitted.resid, residual_before)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_scores_bread_tuple_preserves_weighted_sandwich(weighted, correction):
    endog, exog, weights = example_design()
    model = sm.WLS(endog, exog, weights=weights) if weighted else sm.OLS(endog, exog)
    fitted = model.fit()
    scores = fitted.model.wexog * fitted.wresid[:, None]
    bread = fitted.normalized_cov_params.copy()
    scores_before = scores.copy()
    bread_before = bread.copy()
    expected = reference_cov_hac(fitted, nlags=9, use_correction=correction)
    actual = cov_hac((scores, bread), nlags=9, use_correction=correction)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=3e-13)
    np.testing.assert_array_equal(scores, scores_before)
    np.testing.assert_array_equal(bread, bread_before)


def test_rank_deficient_ols_preserves_pseudoinverse_covariance():
    endog, exog, _ = example_design()
    exog = np.column_stack((exog, 2.0 * exog[:, 1]))
    fitted = sm.OLS(endog, exog).fit()
    expected = reference_cov_hac(fitted, nlags=7)
    actual = cov_hac(fitted, nlags=7)
    np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=3e-13)


def test_glm_requires_explicit_scores_bread_tuple():
    endog, exog, _ = example_design()
    fitted = sm.GLM(np.exp(endog / 10), exog, family=sm.families.Gaussian()).fit()
    with pytest.raises((TypeError, ValueError)):
        cov_hac(fitted, nlags=3)
