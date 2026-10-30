from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ...compute.design_ops import ColumnMoments, column_moments
from ...compute.block_design import BlockDesign
from ...compute.design_plan import StructuralDesignPlan


def outcome_scale(y, weights) -> float:
    """Weighted mean of ``y``, the outcome unit used by every IRLS path.

    Poisson PML slopes do not depend on the units of ``y``, but the IRLS
    safeguards (deviance-change floor, divergence bound, step-halving floor,
    initial-mean floor) are absolute. Iterating on ``y / outcome_scale`` keeps
    the iterations, and hence the estimates, invariant to rescaling ``y``.
    """
    # Form the mean in bounded coordinates: sums of large outcomes or weights
    # can overflow even though their weighted mean is finite.  A fixed lower
    # bound on this scale would reintroduce dependence on the units of y.
    y = np.asarray(y, dtype=np.float64)
    ymax = float(np.max(y, initial=0.0))
    if ymax <= 0:
        return 1.0
    if weights is None:
        mean = float(np.mean(y / ymax))
    else:
        w = np.asarray(weights, dtype=np.float64)
        wmax = float(np.max(w, initial=0.0))
        if wmax <= 0:
            return ymax
        wn = w / wmax
        mean = float(np.dot(y / ymax, wn) / np.sum(wn))
    scale = ymax * mean
    # Only subnormal arithmetic can round a positive mean down to zero.
    # The maximum is still a positive outcome unit in that limiting case.
    return scale if scale > 0 else ymax


@dataclass(slots=True)
class Standardization:
    x_scale: np.ndarray
    y_scale: float

    @classmethod
    def fit(cls, y, X, *, weights=None, structure: StructuralDesignPlan | None = None) -> "Standardization":
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1]:
            moments = column_moments(X, structure=structure)
            xs = moments.sample_scale()
        else:
            xs = np.empty(0, dtype=np.float64)
        ys = outcome_scale(y, weights)
        return cls(np.asarray(xs, dtype=np.float64), float(ys))

    @classmethod
    def fit_block(cls, y, X: BlockDesign, *, weights=None) -> "Standardization":
        """Fit exactly from a block-native design without dense materialization."""
        y = np.asarray(y, dtype=np.float64)
        xs = X.column_moments().sample_scale() if X.ncols else np.empty(0, dtype=np.float64)
        ys = outcome_scale(y, weights)
        return cls(np.asarray(xs, dtype=np.float64), float(ys))

    @classmethod
    def outcome_only(cls, y, weights, k: int) -> "Standardization":
        """Leave regressors in user units; rescale only the outcome."""
        return cls(np.ones(k, dtype=np.float64), outcome_scale(y, weights))

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
