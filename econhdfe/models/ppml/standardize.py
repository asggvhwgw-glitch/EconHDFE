from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ...compute.design_ops import ColumnMoments, column_moments
from ...compute.block_design import BlockDesign
from ...compute.design_plan import StructuralDesignPlan


def _reghdfe_scale(a: np.ndarray, axis=None):
    """Sample-standard-deviation scale used by reghdfe, without demeaning."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0] if a.ndim else 1
    if n <= 1:
        if axis is None:
            return 1e-3
        shape = a.shape[1:] if axis == 0 else ()
        return np.full(shape, 1e-3, dtype=np.float64)
    if axis is None:
        ss = float(np.dot(a.ravel(), a.ravel()))
        s = float(np.sum(a))
        var = max((ss - s * s / n) / (n - 1), 0.0)
        scale = np.sqrt(var)
        return float(max(scale, 1e-3)) if np.isfinite(scale) else 1.0
    ss = np.einsum("ij,ij->j", a, a, optimize=True) if axis == 0 and a.ndim == 2 else np.sum(a * a, axis=axis)
    s = np.sum(a, axis=axis)
    var = np.maximum((ss - s * s / n) / (n - 1), 0.0)
    scale = np.sqrt(var)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return np.maximum(scale, 1e-3)


@dataclass(slots=True)
class Standardization:
    x_scale: np.ndarray
    y_scale: float

    @classmethod
    def fit(cls, y, X, *, structure: StructuralDesignPlan | None = None) -> "Standardization":
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1]:
            moments = column_moments(X, structure=structure)
            xs = moments.sample_scale()
        else:
            xs = np.empty(0, dtype=np.float64)
        ys = _reghdfe_scale(y)
        return cls(np.asarray(xs, dtype=np.float64), float(ys))

    @classmethod
    def fit_block(cls, y, X: BlockDesign) -> "Standardization":
        """Fit exactly from a block-native design without dense materialization."""
        y = np.asarray(y, dtype=np.float64)
        xs = X.column_moments().sample_scale() if X.ncols else np.empty(0, dtype=np.float64)
        ys = _reghdfe_scale(y)
        return cls(np.asarray(xs, dtype=np.float64), float(ys))

    def transform(self, y, X):
        y = np.asarray(y, dtype=np.float64) / self.y_scale
        X = np.asarray(X, dtype=np.float64) / self.x_scale
        return y, X

    def transform_block(self, y, X: BlockDesign):
        """Return standardized outcome plus an exact block-native design."""
        y = np.asarray(y, dtype=np.float64) / self.y_scale
        return y, X.scale_columns(self.x_scale, inverse=True)

    def select_columns(self, keep) -> "Standardization":
        """Return the same fitted scaling restricted to surviving regressors."""
        idx = np.asarray(keep, dtype=np.int64)
        return Standardization(self.x_scale[idx], self.y_scale)

    def unscale_coef(self, beta):
        return np.asarray(beta, dtype=np.float64) / self.x_scale

    def original_scale(self, Xs, frequency_weights=None) -> np.ndarray:
        """``regressor_scale`` of the unstandardized columns of ``Xs``.

        Rescales the column moments instead of materializing ``Xs * x_scale``.
        """
        m = (Xs.column_moments(frequency_weights=frequency_weights) if isinstance(Xs, BlockDesign)
             else column_moments(Xs, frequency_weights=frequency_weights))
        return ColumnMoments(m.nobs, m.sum * self.x_scale, m.sum_squares * (self.x_scale * self.x_scale)).sample_scale()

    def unscale_vcov(self, V):
        s = self.x_scale
        return np.asarray(V, dtype=np.float64) / (s[:, None] * s[None, :])
