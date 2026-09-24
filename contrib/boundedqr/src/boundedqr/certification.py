"""A posteriori coefficient radius for finite augmented quantile regression.

This is a model-qualified FP64 numerical enclosure, NOT a formal interval proof.
It assumes correctly rounded IEEE binary64 scalar operations, round-to-nearest,
gradual underflow, and classical FP64 BLAS dot/matrix products satisfying the
usual componentwise gamma_k bound. No FP32/TF32/tensor-core product is used.
Overflow or an unverifiable inverse produces an infinite, uninformative radius.
Tiny absolute underflow allowances supplement the relative-error model. These
assumptions are not verified for the user's BLAS library by this module.

The certified *numerical data* are the binary64 arrays supplied to this function,
augmented by z0 = fl(W_b / tau), t0 = fl(n * max(abs(y))). These match the pilot's
finite pseudo-observation construction. The enclosure does not include uncertainty
in the data, an upstream cluster-gradient calculation, or conversion from FP32.
Pass the original FP64 X/y/W, even if beta and a were proposed on a GPU.

Mathematical argument (ordinary Euclidean coefficient norm): let Z,t be these
augmented data, l=tau-1, u=tau, r=t-Zb, and choose any a in [l,u]. Put e=Z'a and
G=sum_i[rho_tau(r_i)-a_i*r_i]. For any interior-row subset A, choose numerical
weights 0 < m_i <= min(a_i-l,u-a_i) and S=diag(m_A)Z_A. Then

 F(b+d)-F(b) >= (sigma_min(S)-||e||_2)||d||_2-G-sum_A m_i|r_i|.

Hence, if kappa_lower > e_norm_upper, EVERY minimizer is within

 (G_upper + interior_residual_upper)/(kappa_lower-e_norm_upper)

of b. Exact dual stationarity is unnecessary. An infinite radius means this
calculation did not bound the solution set; it does NOT establish nonuniqueness.

To lower-bound sigma_min(S), compute any approximate left inverse R and verify
||I-RS||_F <= eta < 1, including matrix-product and construction roundoff. Then
sigma_min(S) >= (1-eta)/||R||_F. QR constructs R but its accuracy is not assumed;
the residual check, not a raw SVD singular value or rank heuristic, supplies the
bound. Selecting only some interior rows weakens the bound but is legitimate.

The gamma_k model is standard; see Jeannerod and Rump (2013), "Improved Error
Bounds for Inner Products in Floating-Point Arithmetic", and Higham (1993):
https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf
No novelty is claimed for the inverse-residual inequality or gamma_k analysis.

Run only the bounded existing-data smoke check:
  python experiments/error_radius.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.linalg import qr, solve_triangular

_U = np.finfo(np.float64).eps / 2
_TINY = np.finfo(np.float64).tiny
_INF = float("inf")


@dataclass
class RadiusCertificate:
    radius: float = _INF
    status: str = "uninformative"
    reason: str = "not_computed"
    gap_upper: float = _INF
    interior_residual_upper: float = _INF
    kappa_lower: float = 0.0
    stationarity_norm_upper: float = _INF
    denominator_lower: float = 0.0
    inverse_residual_frobenius_upper: float = _INF
    left_inverse_frobenius_upper: float = _INF
    residual_error_max: float = _INF
    clipping_adjustment_max: float = 0.0
    interior_count: int = 0
    selected_count: int = 0
    selected_indices: tuple[int, ...] = ()
    dual_lower_endpoint_lower: float = 0.0
    dual_lower_endpoint_upper: float = 0.0
    unit_roundoff: float = _U
    proof_level: str = "FP64 model-qualified numerical enclosure; not formal interval"

    def as_dict(self) -> dict:
        """Return scalar diagnostics and selected row indices for audit logs."""
        return asdict(self)


def _up(v):
    return np.nextafter(v, np.inf)


def _down(v):
    return np.nextafter(v, -np.inf)


def _nonnegative_up(v):
    """Outward rounding, preserving an exactly computed zero."""
    return np.where(v == 0, 0.0, _up(v))


def _add_up(a, b):
    return _nonnegative_up(np.asarray(a) + np.asarray(b))


def _mul_up(a, b):
    return _nonnegative_up(np.asarray(a) * np.asarray(b))


def _gamma(k: int) -> float:
    ku = float(_up(k * _U))
    if ku >= 1:
        return _INF
    return float(_up(ku / _down(1.0 - ku)))


def _underflow(k: int) -> float:
    # Deliberately use min NORMAL, not min subnormal, as a generous allowance.
    return float(_up((2 * k + 4) * _TINY))


def _positive_reduction_upper(value, k: int):
    """Enclose exact positive dot/sum from its computed FP64 result."""
    g = _gamma(k)
    if g >= 1:
        return np.full_like(np.asarray(value), np.inf, dtype=float)
    return _up(_add_up(value, _underflow(k)) / _down(1.0 - g))


def _sum_upper(v: np.ndarray) -> float:
    if not np.all(np.isfinite(v)):
        return _INF
    return float(_positive_reduction_upper(np.sum(v, dtype=np.float64), v.size))


def _norm_upper(v: np.ndarray) -> float:
    """Scaled Frobenius/Euclidean upper bound without trusting a norm estimate."""
    v = np.abs(np.asarray(v, dtype=np.float64)).ravel()
    if v.size == 0 or np.max(v) == 0:
        return 0.0
    if not np.all(np.isfinite(v)):
        return _INF
    scale = float(np.max(v))
    # Every scaled entry is enlarged after its rounded division. Thus dot(w,w)
    # bounds the norm after a positive-dot gamma allowance and upward sqrt.
    w = _up(v / scale)
    q = float(_positive_reduction_upper(w @ w, v.size))
    return float(_mul_up(scale, _up(np.sqrt(q))))


def _lower_endpoint(tau: float) -> tuple[float, float]:
    """Binary64 outward bracket of the exact real endpoint tau-1."""
    exact = Fraction.from_float(tau) - 1
    nearest = float(exact)
    rounded = Fraction.from_float(nearest)
    low = float(_down(nearest)) if rounded > exact else nearest
    high = float(_up(nearest)) if rounded < exact else nearest
    return low, high


def _matmul_positive_upper(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _positive_reduction_upper(a @ b, a.shape[-1])


def _residual_envelope(x, y, beta):
    dot = x @ beta
    r = y - dot
    absolute_dot = _matmul_positive_upper(np.abs(x), np.abs(beta))
    dot_error = _mul_up(_gamma(x.shape[1]), absolute_dot)
    subtraction_error = _mul_up(_up(_U / (1 - _U)), np.abs(r))
    error = _add_up(_add_up(dot_error, subtraction_error), _underflow(x.shape[1]))
    return r, error


def _select_rows(indices: np.ndarray, margins: np.ndarray, limit: int) -> np.ndarray:
    """Bound the verification workspace; selection is not a rank certificate."""
    if indices.size <= limit:
        return indices
    # Strong margins plus spread-out rows: deterministic and cheap. A failure
    # to capture full rank yields infinity, never a false positive radius.
    strong_count = limit // 2
    strong = (indices[np.argpartition(margins[indices], -strong_count)[-strong_count:]]
              if strong_count else indices[:0])
    spread = indices[np.linspace(0, indices.size - 1, limit - strong_count, dtype=int)]
    selected = np.unique(np.r_[strong, spread])
    if selected.size < limit:
        remaining = np.setdiff1d(indices, selected, assume_unique=True)
        selected = np.r_[selected, remaining[: limit - selected.size]]
    return np.sort(selected)


def _verified_kappa(zrows: np.ndarray, margins: np.ndarray):
    """Return a residual-verified lower bound for the weighted row operator."""
    k, p = zrows.shape
    if k < p:
        return 0.0, _INF, _INF, "fewer_selected_interior_rows_than_coefficients"
    s = margins[:, None] * zrows
    # Exact S is diag(represented margins)*represented Z; account for fl(S).
    s_error = _add_up(_mul_up(_up(_U / (1 - _U)), np.abs(s)), _underflow(1))
    try:
        q, triangular = qr(s, mode="economic", check_finite=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inverse = solve_triangular(triangular, q.T, check_finite=False)
    except (np.linalg.LinAlgError, ValueError, Warning):
        return 0.0, _INF, _INF, "left_inverse_construction_failed"
    if not np.all(np.isfinite(inverse)):
        return 0.0, _INF, _INF, "nonfinite_left_inverse"
    product = inverse @ s
    residual = np.eye(p) - product
    magnitude = _matmul_positive_upper(np.abs(inverse), np.abs(s))
    multiplication_error = _add_up(_mul_up(_gamma(k), magnitude), _underflow(k))
    construction_error = _matmul_positive_upper(np.abs(inverse), s_error)
    subtraction_error = _mul_up(_up(_U / (1 - _U)), np.abs(residual))
    entry_error = _add_up(_add_up(multiplication_error, construction_error),
                          _add_up(subtraction_error, _underflow(1)))
    eta = _norm_upper(_add_up(np.abs(residual), entry_error))
    inverse_norm = _norm_upper(inverse)
    if not np.isfinite(eta) or not np.isfinite(inverse_norm) or inverse_norm == 0:
        return 0.0, eta, inverse_norm, "nonfinite_inverse_verification"
    if eta >= 1:
        return 0.0, eta, inverse_norm, "left_inverse_residual_not_below_one"
    kappa = max(0.0, float(_down(_down(1.0 - eta) / inverse_norm)))
    return kappa, eta, inverse_norm, "verified"


def coefficient_radius(
    X, y, tau: float, W_b, beta, a_aug, *,
    interior_tol: float = 32 * _U,
    max_candidates: int | None = None,
) -> RadiusCertificate:
    """Enclose all augmented-QR optimizers around ``beta``, when informative.

    X is a CPU (n,p) array; y, W_b, beta and a_aug have lengths n,p,p,n+1.
    ``a_aug`` may be approximately feasible or out of the dual box: this
    function clips it inward, then recomputes every quantity from that a.
    A supplied coefficient proposal need not minimize anything. Negative or
    zero denominators, insufficient useful rows, and numerical failures return
    radius=inf with diagnostics. Invalid shapes/nonfinite inputs raise ValueError.

    ``interior_tol`` drops weak-margin rows, which only weakens the bound.
    At most max(32,8*p) rows are verified by default. Increasing max_candidates
    can strengthen a failed/loose bound at extra cost; there is no automatic
    whole-data QR factorization. All n+1 observations always enter e and G.
    """
    x = np.asarray(X, dtype=np.float64)
    y, w, b, a = [np.asarray(v, dtype=np.float64).reshape(-1)
                  for v in (y, W_b, beta, a_aug)]
    if x.ndim != 2 or min(x.shape, default=0) <= 0:
        raise ValueError("X must be a nonempty two-dimensional array")
    n, p = x.shape
    if (y.size, w.size, b.size, a.size) != (n, p, p, n + 1):
        raise ValueError("Expected lengths y=n, W_b=p, beta=p, a_aug=n+1")
    tau = float(tau)
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must be finite and strictly between zero and one")
    if any(not np.all(np.isfinite(v)) for v in (x, y, w, b, a)):
        raise ValueError("All supplied arrays must be finite")
    if not np.isfinite(interior_tol) or interior_tol < 0:
        raise ValueError("interior_tol must be nonnegative and finite")
    if max_candidates is None:
        max_candidates = max(32, 8 * p)
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < p:
        raise ValueError("max_candidates must be an integer at least p")
    result = RadiusCertificate()
    low, high = _lower_endpoint(tau)
    result.dual_lower_endpoint_lower, result.dual_lower_endpoint_upper = low, high
    used_a = np.clip(a, high, tau)
    result.clipping_adjustment_max = float(np.max(np.abs(used_a - a)))
    margins = np.maximum(0.0, np.minimum(_down(used_a - high), _down(tau - used_a)))
    all_interior = np.flatnonzero(margins > interior_tol)
    result.interior_count = int(all_interior.size)
    selected = _select_rows(all_interior, margins, max_candidates)
    result.selected_count = int(selected.size)
    result.selected_indices = tuple(int(i) for i in selected)

    # No copy of the full augmented design: only its additional row is stored.
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        z0 = w / tau
        t0 = float(n * np.max(np.abs(y)))
        if not np.isfinite(t0) or not np.all(np.isfinite(z0)):
            result.reason = "nonfinite_augmented_data"
            return result
        r, r_error = _residual_envelope(x, y, b)
        r0, r0_error = _residual_envelope(z0[None, :], np.array([t0]), b)
        residual = np.r_[r, r0]
        residual_error = np.r_[r_error, r0_error]
        result.residual_error_max = float(np.max(residual_error))
        positive = np.maximum(0.0, _up(residual + residual_error))
        negative = np.maximum(0.0, _up(-residual + residual_error))
        # Never subtract huge primal and dual pseudo-observation constants.
        positive_coefficient = _nonnegative_up(tau - used_a)
        negative_coefficient = _nonnegative_up(used_a - low)
        terms = _add_up(_mul_up(positive, positive_coefficient),
                        _mul_up(negative, negative_coefficient))
        result.gap_upper = _sum_upper(terms)

        e = x.T @ used_a[:-1] + z0 * used_a[-1]
        absolute_e = np.abs(x).T @ np.abs(used_a[:-1]) + np.abs(z0 * used_a[-1])
        magnitude_upper = _positive_reduction_upper(absolute_e, n + 1)
        e_error = _add_up(_mul_up(_gamma(n + 1), magnitude_upper), _underflow(n + 1))
        result.stationarity_norm_upper = _norm_upper(_add_up(np.abs(e), e_error))
        selected_residual = _add_up(np.abs(residual[selected]), residual_error[selected])
        result.interior_residual_upper = _sum_upper(_mul_up(margins[selected], selected_residual))
        if not np.isfinite(result.gap_upper + result.stationarity_norm_upper + result.interior_residual_upper):
            result.reason = "nonfinite_forward_error_envelope"
            return result
        if selected.size < p:
            result.reason = "fewer_selected_interior_rows_than_coefficients"
            return result
        zrows = np.empty((selected.size, p), dtype=np.float64)
        ordinary = selected < n
        zrows[ordinary] = x[selected[ordinary]]
        zrows[~ordinary] = z0
        kappa, eta, inverse_norm, verification = _verified_kappa(zrows, margins[selected])
        result.kappa_lower = kappa
        result.inverse_residual_frobenius_upper = eta
        result.left_inverse_frobenius_upper = inverse_norm
        denominator = float(_down(kappa - result.stationarity_norm_upper))
        result.denominator_lower = max(0.0, denominator)
        if verification != "verified":
            result.reason = verification
            return result
        if denominator <= 0:
            result.reason = "stationarity_envelope_not_below_sharpness_bound"
            return result
        numerator = float(_add_up(result.gap_upper, result.interior_residual_upper))
        result.radius = float(_up(numerator / denominator))
        if not np.isfinite(result.radius):
            result.reason = "radius_overflow"
            return result
        result.status = "finite"
        result.reason = "model_qualified_enclosure"
        return result
