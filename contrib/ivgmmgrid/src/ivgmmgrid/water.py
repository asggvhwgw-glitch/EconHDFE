"""Do--Jacoby deposited grid's float-storage and tie-selection conventions."""
from dataclasses import dataclass
import numpy as np
from .core import fit_grid, ModelResult


def distributed_lag(lags, order, gamma):
    """Round after every replacement, as Stata's default float variables do."""
    a = np.asarray(lags, dtype=float)
    if a.ndim != 2 or a.shape[1] != 4 or order not in (1, 2, 3, 4):
        raise ValueError("lags must have four columns; order must be 1..4")
    x = a[:, 0].astype(np.float32)
    for j in range(1, order):
        x = (x.astype(float) + float(gamma) ** j * a[:, j]).astype(np.float32)
    return x.astype(float)


@dataclass(frozen=True)
class WaterResult:
    order: int
    gamma: float
    model: ModelResult
    grids: tuple
    order_models: tuple
    order_gammas: np.ndarray


def select_water_grid(y, exog, lags, instruments, absorb, cluster, gammas, *, engine="cached"):
    """Conditional on supplied differenced data and preselected instruments.

    This implements the grid and final refit, not the deposited bootstrap,
    lag-length/LASSO preselection, or survey-data preparation. Explicit gammas
    avoid silently changing the original Stata forvalues endpoints.
    """
    gammas = np.asarray(gammas, dtype=float)
    if gammas.ndim != 1 or not len(gammas) or not np.isfinite(gammas).all():
        raise ValueError("gammas must be a nonempty finite vector")
    lags = np.asarray(lags, dtype=float)
    grids, order_models, chosen = [], [], []
    for order in range(1, 5):
        c = np.column_stack([distributed_lag(lags, order, g) for g in gammas])
        grid = fit_grid(y, exog, c, instruments, absorb, cluster, engine=engine)
        stored_rmse = grid.rmse.astype(np.float32)
        positive = stored_rmse > 0
        if not positive.any():
            raise ValueError("water selector needs a positive RMSE")
        tied = stored_rmse == stored_rmse[positive].min()
        gamma = float(np.mean(gammas.astype(np.float32)[tied].astype(float)))
        model = fit_grid(y, exog, distributed_lag(lags, order, gamma), instruments,
                         absorb, cluster, engine=engine).models[0]
        grids.append(grid)
        order_models.append(model)
        chosen.append(gamma)
    r = np.array([m.rmse for m in order_models], dtype=np.float32)
    tied = r == r.min()
    order = int(np.flatnonzero(tied).max()) + 1
    gamma = float(np.array(chosen, dtype=np.float32)[tied].max())
    final = fit_grid(y, exog, distributed_lag(lags, order, gamma), instruments, absorb, cluster,
                     mask=np.isfinite(lags[:, 3]), engine=engine).models[0]
    return WaterResult(order, gamma, final, tuple(grids), tuple(order_models), np.array(chosen))
