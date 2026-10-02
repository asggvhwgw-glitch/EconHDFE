"""Explicit bandwidth/normalization bridge; no estimator state is changed."""
import numpy as np
from fastsandwich import hac_meat, cluster_meat
from econhdfe.compute.kernels import hac_meat as econ_hac
from econhdfe.compute.vcov import score_covariance

def compare():
    rng = np.random.default_rng(20260925)
    scores = rng.normal(size=(500, 3))
    groups = rng.integers(0, 30, len(scores)).astype(np.int32)
    lag = 24
    expected, _ = econ_hac(scores, bandwidth=lag + 1, kernel="bartlett")
    np.testing.assert_allclose(hac_meat(scores, nlags=lag), expected, rtol=1e-11, atol=1e-10)
    expected_cluster = score_covariance(scores, kind="cluster", clusters=[groups], small_sample=False)
    np.testing.assert_allclose(len(scores) * cluster_meat(scores, groups), expected_cluster, rtol=1e-12, atol=1e-10)
    print("EconHDFE score comparison passed: Bartlett bandwidth=L+1; cluster meat multiplied by N.")

if __name__ == "__main__":
    compare()
