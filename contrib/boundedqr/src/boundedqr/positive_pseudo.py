"""Small sign-region guard for an optional finite-problem candidate strategy."""
import numpy as np
from .certification import _down, _residual_envelope


def pseudo_residual_lower(d, response, beta):
    """Outward lower bound on the original finite pseudo residual M-d'beta.

    d is the stored rounded pseudo row, not an algebraic replacement using W.
    This proves only the sign region under the existing FP64 model. Original
    full-data KKT checks and the independent radius audit remain necessary.
    """
    row = np.asarray(d, dtype=np.float64).reshape(1, -1)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        residual, error = _residual_envelope(row, np.array([response]), beta)
        lower = float(_down(residual[0]-error[0]))
    return lower if np.isfinite(lower) else -np.inf
