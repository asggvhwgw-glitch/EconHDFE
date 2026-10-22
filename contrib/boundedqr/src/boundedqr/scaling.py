"""Loss-preserving power-of-two column coordinates for numerical QR solves.

Let S=diag(2**exponents). We use Z=X/S, theta=S*beta, and V=W/S.
Then X*beta=Z*theta and (W/tau)'beta=(V/tau)'theta, so the full
finite pseudo-observation objective is unchanged. The QR dual is unchanged.
Power-of-two operations are checked for exact binary64 round trips; unsupported
overflow/underflow is rejected rather than silently solving modified data.
This controls column measurement units, not near collinearity, degeneracy,
or uniqueness, and it does not guarantee an informative coefficient radius.
"""
from dataclasses import dataclass
import numpy as np


def _exact_ldexp(values, exponents, name):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{name} must be finite')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = np.ldexp(values, exponents)
        restored = np.ldexp(result, -np.asarray(exponents))
    if not np.isfinite(result).all() or not np.array_equal(restored, values):
        raise ValueError(f'{name} dynamic range cannot be represented by lossless column normalization; rescale the input units')
    return result


@dataclass
class ColumnCoordinates:
    design: np.ndarray
    exponents: np.ndarray

    @classmethod
    def from_design(cls, X):
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError('X must be a nonempty finite matrix')
        largest = np.max(np.abs(x), axis=0)
        _, exponents = np.frexp(largest)
        # All-zero columns keep exponent zero. Identification is a separate
        # question; this coordinate map does not silently remove columns.
        exponents = np.where(largest > 0, exponents, 0).astype(np.int32)
        design = _exact_ldexp(x, -exponents, 'X')
        return cls(design, exponents)

    def coefficients_in(self, beta):
        return _exact_ldexp(beta, self.exponents, 'beta0')

    def coefficients_out(self, theta):
        return _exact_ldexp(theta, -self.exponents, 'computed coefficients')

    def perturbations_in(self, W, tau):
        w = np.asarray(W, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != len(self.exponents):
            raise ValueError('W must have one row per design column')
        mapped = _exact_ldexp(w, -self.exponents[:, None], 'W')
        # The certified finite pseudo row is fl(W/tau), not an arbitrary
        # rearrangement that could round differently near exponent limits.
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            pseudo = w / tau
            scaled_pseudo = mapped / tau
        expected = _exact_ldexp(pseudo, -self.exponents[:, None], 'W/tau')
        if not np.array_equal(scaled_pseudo, expected):
            raise ValueError('Finite pseudo row cannot be normalized without changing its FP64 values')
        return mapped

    def radius_out(self, radius):
        """Outward-round ||S^-1||_2 times a normalized Euclidean radius."""
        radius = float(radius)
        if radius == 0 or np.isinf(radius):
            return radius
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            mapped = float(np.ldexp(radius, -int(np.min(self.exponents))))
        return float(np.nextafter(mapped, np.inf)) if np.isfinite(mapped) else np.inf

    def map_certificate(self, certificate):
        out = dict(certificate)
        normalized = float(out['radius'])
        out['radius'] = self.radius_out(normalized)
        out['normalized_radius'] = normalized
        out['radius_coordinates'] = 'original Euclidean coefficient units'
        out['linear_algebra_coordinates'] = 'power-of-two column-normalized coefficients'
        out['radius_multiplier_power2'] = -int(np.min(self.exponents))
        if np.isfinite(normalized) and not np.isfinite(out['radius']):
            out['status'] = 'uninformative'
            out['reason'] = 'original_coefficient_radius_overflow'
        return out

    def diagnostics(self):
        return dict(method='lossless_power_of_two_columns',
                    column_scale_exponents=self.exponents.tolist(),
                    primal_mapping='theta_j = 2**exponent_j * beta_j',
                    perturbation_mapping='V_jb = 2**(-exponent_j) * W_jb',
                    stationarity_coordinates='column-normalized',
                    loss_and_residual_units='original',
                    output_coefficient_and_radius_units='original',
                    exact_fp64_roundtrip_checked=True)
