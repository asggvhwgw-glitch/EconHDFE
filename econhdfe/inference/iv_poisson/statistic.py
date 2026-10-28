"""Studentized null moments. No coefficient-Jacobian inverse is used."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2

@dataclass(slots=True)
class MomentTestResult:
    statistic: float | None
    pvalue: float | None
    df: int
    status: str
    n_units: int
    null_value: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(statistic=self.statistic, pvalue=self.pvalue, df=self.df,
                    status=self.status, n_units=self.n_units,
                    null_value=None if self.null_value is None else self.null_value.tolist(),
                    diagnostics=self.diagnostics)


def _quadratic(total, covariance_of_sum, *, n_units, diagnostics=None, rcond=1e-11):
    m = np.asarray(total, dtype=float)
    V = np.asarray(covariance_of_sum, dtype=float)
    if m.ndim != 1 or len(m) < 1 or V.shape != (len(m), len(m)):
        raise ValueError("incompatible/nonpositive moment dimension")
    d = dict(diagnostics or {})
    d.update(covariance_at_null=True, target_jacobian_inverted=False,
             degrees_of_freedom="number_of_tested_moments_not_L_minus_K",
             calibration="asymptotic_chi_square_under_stated_null_CLT")
    L = len(m)
    result = MomentTestResult(None, None, L, "pending", int(n_units), diagnostics=d)
    if n_units <= L:
        result.status = "too_few_independent_units"; return result
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(V)):
        result.status = "nonfinite_moment_or_covariance"; return result
    diag = np.diag(V)
    if np.any(diag <= 0):
        result.status = "singular_or_indefinite_covariance"; return result
    scale = np.sqrt(diag)
    A = V / scale[:, None] / scale[None, :]
    A = (A + A.T) * 0.5
    eig = np.linalg.eigvalsh(A)
    d["covariance_rcond"] = float(eig[0] / eig[-1])
    if eig[0] <= rcond * eig[-1]:
        result.status = "singular_or_indefinite_covariance"; return result
    z = m / scale
    value = float(z @ np.linalg.solve(A, z))
    if not np.isfinite(value) or value < 0:
        result.status = "invalid_quadratic_form"; return result
    result.statistic = value
    result.pvalue = float(chi2.sf(value, L))
    result.status = "ok_conditional_on_assumptions"
    return result


def moment_test(unit_scores, *, center=True, diagnostics=None, rcond=1e-11):
    """Test the zero mean of independent unit/cluster scores, L fixed.

    Centering estimates the variance of cluster scores, not observation-level
    errors. Overlapping clusters/cycles cannot be supplied as independent units.
    No pseudoinverse, PSD clipping, or data-dependent rank reduction is applied.
    """
    s = np.asarray(unit_scores, dtype=float)
    if s.ndim == 1: s = s[:, None]
    if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] < 1:
        raise ValueError("scores must be a G by L matrix with G >= 2, L >= 1")
    if not np.all(np.isfinite(s)):
        raise ValueError("nonfinite unit scores")
    # A common sample-dependent rescaling is harmless: the quadratic statistic
    # is algebraically invariant. Never normalize clusters separately.
    scale = np.max(np.abs(s), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = s / scale
    total = scaled.sum(axis=0)
    v = scaled - scaled.mean(axis=0) if center else scaled
    covariance = v.T @ v
    sq = np.einsum("ij,ij->i", v, v)
    d = dict(diagnostics or {})
    d.update(centered_covariance=bool(center),
             max_unit_energy_share=float(sq.max(initial=0.) / max(sq.sum(), np.finfo(float).tiny)),
             few_units_warning=bool(len(s) < 30))
    return _quadratic(total, covariance, n_units=len(s), diagnostics=d, rcond=rcond)
