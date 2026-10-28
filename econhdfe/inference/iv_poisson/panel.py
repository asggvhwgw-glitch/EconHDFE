"""Null moments for one-way panels, with every slope fixed at its null value.

SPJ corrects MOMENTS, not point estimators. Cross-product elimination needs an
extra off-diagonal orthogonality assumption, sufficient under conditional
independence across periods; ordinary additive IV validity alone is insufficient.
"""
from __future__ import annotations
import numpy as np
from .statistic import moment_test


def _matrix(a, n, name):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1: a = a[:, None]
    if a.ndim != 2 or a.shape[0] != n or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite with {n} rows")
    return a


def _sum_cols(a, groups, count):
    if a.ndim == 1:
        return np.bincount(groups, weights=a, minlength=count)
    return np.column_stack([np.bincount(groups, weights=a[:, j], minlength=count)
                            for j in range(a.shape[1])])


def panel_null_scores(y, x, instruments, unit, beta, *, method="spj",
                      offset=None, split=None, assume_long_panel=False,
                      assume_cross_time_independence=False):
    """Return G x L cluster moments and diagnostics without fitting beta.

    No estimated included controls are accepted: put all slopes in x/beta or
    supply a known offset. `split` must be a predetermined binary time split,
    balanced *within each unit*. Serial dependence can invalidate cross-product
    moments, even though the final covariance is clustered by unit.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("y must be a finite nonnegative vector")
    n = len(y); X = _matrix(x, n, "x"); Z = _matrix(instruments, n, "instruments")
    b = np.asarray(beta, dtype=float).reshape(-1)
    if b.shape != (X.shape[1],) or not np.all(np.isfinite(b)):
        raise ValueError("all slope coefficients must be specified in beta")
    u = np.asarray(unit)
    if u.ndim != 1 or len(u) != n: raise ValueError("invalid unit labels")
    labels, g = np.unique(u, return_inverse=True); G = len(labels)
    if G < 2 or Z.shape[1] < 1: raise ValueError("need >=2 units and >=1 moment")
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    if off.shape != (n,) or not np.all(np.isfinite(off)): raise ValueError("invalid offset")
    eta = off + X @ b
    if not np.all(np.isfinite(eta)) or np.ptp(eta) > 600:
        raise ValueError("unsupported predictor dynamic range; no clipping is performed")
    r = np.exp(eta - float(np.max(eta)))  # one COMMON factor cancels from S
    T = np.bincount(g, minlength=G)
    if np.any(T < 2): raise ValueError("every unit needs at least two observations")
    if method not in {"profile", "spj", "cross_product"}:
        raise ValueError("method must be profile/spj/cross_product")
    if method == "cross_product" and assume_cross_time_independence is not True:
        raise ValueError("cross_product requires explicit off-diagonal validity acknowledgment")
    if method in {"profile", "spj"} and assume_long_panel is not True:
        raise ValueError("profile/SPJ requires an explicit long-panel bias-remainder assumption")
    R = _sum_cols(r, g, G); Y = _sum_cols(y, g, G)
    ZY = _sum_cols(Z * y[:, None], g, G); ZR = _sum_cols(Z * r[:, None], g, G)
    full = (ZY - (Y / R)[:, None] * ZR) / T[:, None]
    if method == "profile": out = full
    elif method == "cross_product":
        out = (R[:, None] * ZY - Y[:, None] * ZR) / (T * (T-1))[:, None]
    else:
        if split is None: raise ValueError("SPJ requires a predetermined balanced time split")
        h = np.asarray(split)
        if h.shape != (n,) or not np.all(np.isin(h, [0, 1])):
            raise ValueError("split must contain only 0 and 1")
        sizes = [np.bincount(g[h == j], minlength=G) for j in (0, 1)]
        if np.any(sizes[0] != sizes[1]) or np.any(sizes[0] < 2):
            raise ValueError("SPJ needs equal halves of at least two observations in every unit")
        halves = []
        for j in (0, 1):
            take = h == j; gj = g[take]
            Rj = _sum_cols(r[take], gj, G); Yj = _sum_cols(y[take], gj, G)
            ZYj = _sum_cols(Z[take] * y[take, None], gj, G)
            ZRj = _sum_cols(Z[take] * r[take, None], gj, G)
            halves.append((ZYj - (Yj/Rj)[:, None]*ZRj) / sizes[j][:, None])
        out = 2.0 * full - 0.5 * (halves[0] + halves[1])
    info = dict(method=method, all_slopes_fixed_under_null=True,
                independent_units=G, rows=n, min_T=int(T.min()), max_T=int(T.max()),
                all_zero_units_retained=int(np.sum(Y == 0)),
                point_estimator_used=False, nuisance_FE_count=G,
                growing_FE_scope=("one_way_fixed_T_extra_off_diagonal_moments" if method == "cross_product"
                                  else "one_way_long_panel_conditional_bias_expansion"),
                changes_moment_weighting=bool(method == "cross_product"),
                assumes_cross_time_independence=bool(method == "cross_product"),
                assumptions_are_user_assertions_not_empirical_certificates=True)
    if method == "spj":
        info.update(bias_order="O(T^-2)_if_stable_split_expansion", covariance="joint_unit_SPJ_scores",
                    usual_sufficient_rate="G/T^3 -> 0_when_Var(unit_score)_is_order_1/T")
    elif method == "profile":
        info.update(bias_order="O(T^-1)_generically", usual_sufficient_rate="G/T -> 0_when_Var(unit_score)_is_order_1/T")
    else:
        info.update(moment_bias="exactly_zero_under_extra_off_diagonal_orthogonality",
                    poisson_distribution_required=False, includes_zero_outcome_units=True)
    return out, info


def panel_s_test(y, x, instruments, unit, beta, **kwargs):
    scores, info = panel_null_scores(y, x, instruments, unit, beta, **kwargs)
    result = moment_test(scores, diagnostics=info)
    result.null_value = np.asarray(beta, dtype=float).reshape(-1)
    return result
