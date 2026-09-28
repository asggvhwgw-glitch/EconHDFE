import numpy as np
import pytest
from pyreghdfe.ppml.separation_simplex import simplex_flag_separated_obs

CASES = [
    (np.array([[1,12],[1,0],[0,-5],[0,6]], float), [0,0,1,1]),
    (np.array([[1,12],[1,0],[0,-5],[0,0]], float), [0,0,0,1]),
    (np.array([[-1,12],[1,0],[0,-5],[0,0]], float), [1,1,1,1]),
    (np.array([[1],[0],[0]], float), [0,1,1]),
    (np.array([[1]], float), [0]),
    (np.array([[0],[0],[0]], float), [1,1,1]),
    (np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1],[0,0,0]], float), [0,0,0,1]),
    (np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1],[1,0,0]], float), [0,1,1,1]),
    (np.array([[1,1,1,1],[2,2,2,2],[-3,-3,-3,-3],[-4,-4,-4,-4]], float), [1,1,1,1]),
    (np.array([[1,2,-3,-4]] * 5, float).T, [1,1,1,1]),
    (np.array([[1,2,3],[2,4,6],[-3,-6,-9],[-4,-8,-12]], float), [1,1,1,1]),
    (np.array([[1,2,3],[2,-2,0],[-3,0,-3],[-4,0,-4]], float), [1,1,1,1]),
    (np.array([[.3,3],[-2,-20]], float), [1,1]),
    (np.array([[.3,-2],[1,-1]], float), [0,0]),
    (np.array([[.3,-2],[1,-1],[0,0]], float), [0,0,1]),
]

@pytest.mark.parametrize('X,keep_expected', CASES)
def test_official_simplex_internal_cases(X, keep_expected):
    sep, _, conv = simplex_flag_separated_obs(X)
    assert conv
    keep = (~sep).astype(int)
    assert keep.tolist() == keep_expected


def test_public_nonexistence_example_flags_obs_1_and_5():
    from pyreghdfe import ppmlhdfe, PPMLConfig
    y = np.array([0,0,0,0,0,2,3,5,7,10.])
    X = np.array([
        [0,1,0], [0,0,0], [0,0,0], [0,0,0], [1,9,0],
        [21,21,21], [0,0,0], [0,0,0], [0,0,0], [-18,-18,0],
    ], dtype=float)
    result = ppmlhdfe(
        y, X, vce="model",
        config=PPMLConfig(
            separation=("simplex",),
            tolerance=1e-9,
            target_inner_tol=1e-10,
        ),
    )
    assert (np.flatnonzero(result.separation_mask) + 1).tolist() == [1, 5]
    # The third original regressor survives; upstream reports about 0.066.
    assert abs(result.params["x2"] - 0.066) < 1e-3
