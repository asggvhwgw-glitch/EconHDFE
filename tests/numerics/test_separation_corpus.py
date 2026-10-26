import numpy as np


def test_public_nonexistence_example_flags_obs_1_and_5():
    from econhdfe import ppmlhdfe, PPMLConfig
    y = np.array([0,0,0,0,0,2,3,5,7,10.])
    X = np.array([
        [0,1,0], [0,0,0], [0,0,0], [0,0,0], [1,9,0],
        [21,21,21], [0,0,0], [0,0,0], [0,0,0], [-18,-18,0],
    ], dtype=float)
    result = ppmlhdfe(
        y=y, x=X, vce="model",
        config=PPMLConfig(
            separation=("simplex",),
            tolerance=1e-9,
            target_inner_tol=1e-10,
        ),
    )
    assert (np.flatnonzero(result.separation_mask) + 1).tolist() == [1, 5]
    # The third original regressor survives; upstream reports about 0.066.
    assert abs(result.params[result.names.index("x2")] - 0.066) < 1e-3
