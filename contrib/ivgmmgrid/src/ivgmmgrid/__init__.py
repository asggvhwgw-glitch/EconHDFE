"""Unweighted candidate-grid IV-GMM, implemented entirely in Python."""
from .core import IdentificationError, ModelResult, GridResult, fit_grid
from .water import distributed_lag, select_water_grid, WaterResult

__all__ = ["IdentificationError", "ModelResult", "GridResult", "fit_grid",
           "distributed_lag", "select_water_grid", "WaterResult"]
__version__ = "0.2.0"
