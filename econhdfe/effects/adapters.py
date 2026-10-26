from __future__ import annotations

from typing import Sequence
import numpy as np

from .recover import recover_fixed_effects
from .results import FixedEffectRecoveryResult, NormalizationSpec
from .diagnostics import diagnose_fe_recovery
from ..compute.encoding import factorize_1d
from ..hdfe.absorber import iterative_singleton_mask


def _as_2d(X, n: int) -> np.ndarray:
    if X is None:
        return np.empty((n, 0), dtype=np.float64)
    a = np.asarray(X, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] != n:
        raise ValueError("X must have one row per observation")
    return a


def _select_columns(X: np.ndarray, x_names, wanted: Sequence[str]) -> np.ndarray:
    wanted = tuple(str(x) for x in wanted)
    if not wanted:
        return np.empty((X.shape[0], 0), dtype=np.float64)
    if x_names is None:
        if X.shape[1] != len(wanted):
            raise ValueError(
                "X column count does not match the estimator's active coefficients; "
                "pass x_names when the original design contained dropped columns"
            )
        return X
    names = tuple(str(x) for x in x_names)
    if len(names) != X.shape[1] or len(set(names)) != len(names):
        raise ValueError("x_names must contain one unique name per X column")
    pos = {name: j for j, name in enumerate(names)}
    missing = [name for name in wanted if name not in pos]
    if missing:
        raise ValueError(f"X is missing active estimator columns: {missing}")
    return X[:, [pos[name] for name in wanted]]


def _subset_vector(value, mask: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    n = len(mask)
    if value is None:
        return np.full(int(mask.sum()), default, dtype=np.float64)
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0:
        return np.full(int(mask.sum()), float(a), dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("offset/weights must be scalar or one-dimensional")
    if len(a) == n:
        return a[mask]
    if len(a) == int(mask.sum()):
        return a
    raise ValueError("offset/weights length must match full or estimation sample")


def _subset_groups(groups, mask: np.ndarray):
    out = []
    n = len(mask)
    ns = int(mask.sum())
    for group in tuple(groups):
        a = np.asarray(group)
        if a.ndim != 1:
            raise ValueError("each fixed-effect group must be one-dimensional")
        if len(a) == n:
            out.append(a[mask])
        elif len(a) == ns:
            out.append(a)
        else:
            raise ValueError("fixed-effect groups must match full or estimation sample")
    return tuple(out)


def recover_linear_result(
    result,
    X,
    groups,
    *,
    x_names=None,
    fe_names=None,
    levels=None,
    weights=None,
    normalization: NormalizationSpec | str = "canonical",
    solver: str = "auto",
    tol: float = 1e-10,
    max_iter: int = 16_000,
    rank_backend: str = "auto",
    strict_identification: bool = True,
) -> FixedEffectRecoveryResult:
    """Recover indicator FEs from an OLS/linear-IV result without touching the estimator.

    The adapter uses ``result.fitted - X @ result.params`` as the additive FE
    contribution.  ``X`` should contain the estimation-sample rows.  If its
    columns include regressors later dropped for collinearity, pass ``x_names``
    so only the active ``result.names`` are selected.
    """
    fitted = np.asarray(result.fitted, dtype=np.float64)
    beta = np.asarray(result.params, dtype=np.float64)
    active_names = tuple(str(x) for x in getattr(result, "names", ()))
    if len(active_names) != len(beta):
        raise ValueError("linear result names/params are inconsistent")

    raw_groups = tuple(groups)
    if not raw_groups:
        raise ValueError("at least one fixed-effect group is required")
    raw_n = len(np.asarray(raw_groups[0]))
    if any(len(np.asarray(g)) != raw_n for g in raw_groups):
        raise ValueError("fixed-effect groups must have equal length")
    Xraw = _as_2d(X, raw_n)
    n_fit = len(fitted)
    reported_singletons = int(getattr(result, "dropped_singletons", 0) or 0)

    if raw_n == n_fit:
        keep = None
        group_s = raw_groups
        Xs = Xraw
        recovery_w = weights
        diagnosis = diagnose_fe_recovery(
            raw_groups, names=fe_names, reported_singletons=reported_singletons,
            estimator=str(getattr(result, "estimator", "linear")), rank_backend=rank_backend,
        )
    elif raw_n > n_fit and reported_singletons > 0:
        dense = [factorize_1d(np.asarray(g))[0] for g in raw_groups]
        weight_type = str(getattr(result, "weight_type", "none") or "none").lower()
        raw_w = None if weights is None else np.asarray(weights, dtype=np.float64)
        if raw_w is not None and (raw_w.ndim != 1 or len(raw_w) != raw_n):
            raise ValueError("weights must match the full raw sample when raw groups are supplied")
        use_fw = weight_type == "fweight" and raw_w is not None
        keep, dropped = iterative_singleton_mask(
            dense, raw_w if use_fw else None, frequency_weights=use_fw
        )
        if dropped != reported_singletons or int(np.sum(keep)) != n_fit:
            raise ValueError(
                "could not reconstruct the linear estimator sample from the supplied FE groups; "
                "pass X/groups already restricted to the estimation sample"
            )
        Xs = Xraw[keep]
        group_s = tuple(np.asarray(g)[keep] for g in raw_groups)
        recovery_w = None if raw_w is None else raw_w[keep]
        diagnosis = diagnose_fe_recovery(
            raw_groups, names=fe_names, final_mask=keep, singleton_mask=~keep,
            estimator=str(getattr(result, "estimator", "linear")), rank_backend=rank_backend,
        )
    else:
        raise ValueError(
            "X/groups do not match the linear estimator sample; pass either the estimation-sample "
            "rows or the full raw rows for a result whose only row exclusion was recursive singleton pruning"
        )

    Xa = _select_columns(Xs, x_names, active_names)
    target = fitted - Xa @ beta
    return recover_fixed_effects(
        target,
        group_s,
        names=fe_names,
        levels=levels,
        weights=recovery_w,
        normalization=normalization,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        rank_backend=rank_backend,
        strict_identification=strict_identification,
        identification_diagnosis=diagnosis,
    )


def recover_ppml_result(
    result,
    X,
    groups,
    *,
    offset=None,
    x_names=None,
    fe_names=None,
    levels=None,
    weights=None,
    normalization: NormalizationSpec | str = "canonical",
    solver: str = "auto",
    tol: float = 1e-10,
    max_iter: int = 16_000,
    rank_backend: str = "auto",
    strict_identification: bool = True,
) -> FixedEffectRecoveryResult:
    """Recover indicator FEs from a converged PPML result.

    The recovered target is ``eta - offset - X beta`` on the final estimation
    sample.  By default recovery uses the final IRLS mass ``mu``; passing
    ``weights`` changes this to ``weights * mu``.  This is a post-estimation
    adapter and does not rerun PPML.
    """
    eta_full = np.asarray(result.eta, dtype=np.float64)
    sample = np.asarray(result.sample_mask, dtype=bool)
    if eta_full.ndim != 1 or sample.ndim != 1 or len(eta_full) != len(sample):
        raise ValueError("PPML result eta/sample_mask are inconsistent")
    if not np.any(sample):
        raise ValueError("PPML result has an empty estimation sample")

    n = len(sample)
    X0 = _as_2d(X, n)
    coef = np.asarray(result.params, dtype=np.float64)
    names = tuple(str(x) for x in result.names)
    slope_idx = [j for j, name in enumerate(names) if name != "_cons"]
    slope_names = tuple(names[j] for j in slope_idx)
    Xa = _select_columns(X0, x_names, slope_names)[sample]
    xb = Xa @ coef[slope_idx] if slope_idx else np.zeros(int(sample.sum()), dtype=np.float64)
    if "_cons" in names:
        cons = float(coef[names.index("_cons")])
        xb += cons
    off = _subset_vector(offset, sample, default=0.0)
    eta = eta_full[sample]
    target = eta - off - xb

    mu = np.asarray(result.mu, dtype=np.float64)
    if mu.ndim != 1 or len(mu) != n:
        raise ValueError("PPML result mu is inconsistent with sample_mask")
    base_w = _subset_vector(weights, sample, default=1.0)
    recovery_w = np.maximum(base_w * mu[sample], np.finfo(float).tiny)
    raw_groups = tuple(groups)
    lengths = {len(np.asarray(g)) for g in raw_groups}
    if lengths == {n}:
        separation = np.asarray(getattr(result, "separation_mask", np.zeros(n, dtype=bool)), dtype=bool)
        singleton = (~sample) & (~separation)
        diagnosis = diagnose_fe_recovery(
            raw_groups, names=fe_names, final_mask=sample, singleton_mask=singleton,
            separation_mask=separation,
            separation_by_method=dict(getattr(result, "diagnostics", {}).get("separation_by_method", {})),
            estimator="ppml", rank_backend=rank_backend,
        )
    else:
        diagnosis = diagnose_fe_recovery(
            _subset_groups(raw_groups, sample), names=fe_names,
            reported_singletons=int(getattr(result, "n_singletons", 0) or 0),
            reported_separated=int(getattr(result, "n_separated", 0) or 0),
            separation_by_method=dict(getattr(result, "diagnostics", {}).get("separation_by_method", {})),
            estimator="ppml", rank_backend=rank_backend,
        )
    group_s = _subset_groups(raw_groups, sample)

    return recover_fixed_effects(
        target,
        group_s,
        names=fe_names,
        levels=levels,
        weights=recovery_w,
        normalization=normalization,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        rank_backend=rank_backend,
        strict_identification=strict_identification,
        identification_diagnosis=diagnosis,
    )


def recover_ivppml_result(*args, **kwargs) -> FixedEffectRecoveryResult:
    """IV-PPML uses the same eta-based recovery contract as PPML."""
    return recover_ppml_result(*args, **kwargs)
