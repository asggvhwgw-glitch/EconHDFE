from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, f as f_dist, t as t_dist


@dataclass(frozen=True, slots=True)
class LinearCombinationResult:
    estimate: float
    std_error: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    df: float
    value: float = 0.0


@dataclass(frozen=True, slots=True)
class WaldTestResult:
    statistic: float
    p_value: float
    df_num: int
    df_denom: float | None
    distribution: str


def _coefficient_vector(spec, names: tuple[str, ...], k: int) -> np.ndarray:
    if isinstance(spec, Mapping):
        if not names or len(names) != k:
            raise ValueError("named restrictions require one coefficient name per parameter")
        index = {name: j for j, name in enumerate(names)}
        unknown = tuple(name for name in spec if name not in index)
        if unknown:
            raise ValueError(f"unknown coefficient name(s): {unknown}")
        out = np.zeros(k, dtype=np.float64)
        for name, value in spec.items():
            out[index[name]] = float(value)
        return out
    out = np.asarray(spec, dtype=np.float64)
    if out.ndim != 1 or len(out) != k:
        raise ValueError(f"restriction vector must have shape ({k},)")
    return out


def linear_combination(params, vcov, weights, *, names=(), value=0.0,
                       df=np.inf, level=0.95) -> LinearCombinationResult:
    beta = np.asarray(params, dtype=np.float64)
    V = np.asarray(vcov, dtype=np.float64)
    if beta.ndim != 1 or V.shape != (len(beta), len(beta)):
        raise ValueError("params/vcov shapes are inconsistent")
    if not 0 < float(level) < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    w = _coefficient_vector(weights, tuple(names), len(beta))
    estimate = float(w @ beta)
    variance = float(w @ V @ w)
    scale = max(1.0, float(np.max(np.abs(V))) if V.size else 1.0)
    if variance < -1e-12 * scale:
        raise ValueError("contrast variance is negative; covariance matrix is not numerically valid")
    std_error = float(np.sqrt(max(variance, 0.0)))
    diff = estimate - float(value)
    if std_error == 0.0:
        if np.isclose(diff, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            raise ValueError("contrast has zero estimated variance but does not satisfy the null")
    else:
        statistic = diff / std_error
        p_value = float(2.0 * t_dist.sf(abs(statistic), df))
    alpha = 1.0 - float(level)
    critical = float(t_dist.ppf(1.0 - alpha / 2.0, df))
    return LinearCombinationResult(
        estimate=estimate, std_error=std_error, statistic=float(statistic),
        p_value=p_value, ci_low=float(estimate - critical * std_error),
        ci_high=float(estimate + critical * std_error), df=float(df), value=float(value),
    )


def _restriction_matrix(restrictions, names: tuple[str, ...], k: int) -> np.ndarray:
    if isinstance(restrictions, Mapping):
        return _coefficient_vector(restrictions, names, k)[None, :]
    if isinstance(restrictions, Sequence) and not isinstance(restrictions, (str, bytes, np.ndarray)):
        if restrictions and all(isinstance(item, Mapping) for item in restrictions):
            return np.vstack([_coefficient_vector(item, names, k) for item in restrictions])
    R = np.asarray(restrictions, dtype=np.float64)
    if R.ndim == 1:
        R = R[None, :]
    if R.ndim != 2 or R.shape[1] != k:
        raise ValueError(f"restriction matrix must have shape (m, {k})")
    return R


def wald_test(params, vcov, restrictions=None, *, values=None, names=(),
              df_resid=np.inf, distribution="F") -> WaldTestResult:
    beta = np.asarray(params, dtype=np.float64)
    V = np.asarray(vcov, dtype=np.float64)
    if beta.ndim != 1 or V.shape != (len(beta), len(beta)):
        raise ValueError("params/vcov shapes are inconsistent")
    k = len(beta)
    R = np.eye(k, dtype=np.float64) if restrictions is None else _restriction_matrix(
        restrictions, tuple(names), k
    )
    if R.shape[0] == 0:
        raise ValueError("at least one restriction is required")
    if values is None:
        q = np.zeros(R.shape[0], dtype=np.float64)
    else:
        q = np.asarray(values, dtype=np.float64)
        if q.ndim == 0:
            q = np.full(R.shape[0], float(q), dtype=np.float64)
        if q.shape != (R.shape[0],):
            raise ValueError(f"values must have shape ({R.shape[0]},)")
    diff = R @ beta - q
    S = R @ V @ R.T
    S = 0.5 * (S + S.T)
    rank = int(np.linalg.matrix_rank(S))
    if rank == 0:
        raise ValueError("restrictions have zero estimated variance and are not statistically testable")
    Sinv = np.linalg.pinv(S, hermitian=True)
    projection_error = np.linalg.norm(diff - S @ (Sinv @ diff))
    tolerance = 1e-9 * max(1.0, np.linalg.norm(diff))
    if projection_error > tolerance:
        raise ValueError("restriction includes a direction with no estimated sampling variance")
    chi2_stat = float(diff @ Sinv @ diff)
    dist = str(distribution).lower()
    if dist in {"f", "f_test"}:
        if not np.isfinite(df_resid) or float(df_resid) <= 0:
            raise ValueError("F-distributed Wald tests require positive finite residual degrees of freedom")
        statistic = chi2_stat / rank
        return WaldTestResult(
            statistic, float(f_dist.sf(statistic, rank, float(df_resid))),
            rank, float(df_resid), "F",
        )
    if dist in {"chi2", "chi-square", "chisquare"}:
        return WaldTestResult(
            chi2_stat, float(chi2.sf(chi2_stat, rank)), rank, None, "chi2"
        )
    raise ValueError("distribution must be 'F' or 'chi2'")
