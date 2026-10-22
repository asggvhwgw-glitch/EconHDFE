"""Small deterministic CPU example; not a performance or coverage experiment."""
import numpy as np
from boundedqr import bootstrap

rng = np.random.default_rng(20260925)
n = 240
cluster = np.repeat(np.arange(40), 6)
X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
y = X @ np.array([1., .7, -.3]) + .3 * rng.normal(size=40)[cluster] + rng.normal(size=n)
result = bootstrap(X, y, cluster, quantiles=(.25, .5, .75), reps=19,
                   seed=123, backend="cpu", batch_size=4, threads=1)
assert result.bootstrap_coefficients.shape == (19, 3, 3)
assert np.isfinite(result.standard_errors).all()
print("Rows are quantiles .25, .5, .75; columns are intercept, x1, x2.")
print("Coefficients:\n", result.coefficients)
print("Bootstrap standard errors:\n", result.standard_errors)
print("Finite numerical radii:", np.isfinite(result.coefficient_radii).sum(), "/", result.coefficient_radii.size)
print("19 draws demonstrate execution only; use a justified larger count for research.")
