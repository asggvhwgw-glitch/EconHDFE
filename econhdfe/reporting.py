from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, runtime_checkable
import numpy as np
from scipy.stats import norm, t as student_t, f as fisher_f


def _validate_level(level: float) -> float:
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    return level


def inference_table(names, coef, stderr, *, level: float = 0.95, df: float | None = None,
                    distribution: str = "normal"):
    """Return a publication-ready coefficient table as a pandas DataFrame."""
    import pandas as pd

    level = _validate_level(level)
    b = np.asarray(coef, dtype=np.float64)
    se = np.asarray(stderr, dtype=np.float64)
    if b.ndim != 1 or se.shape != b.shape:
        raise ValueError("coef and stderr must be one-dimensional with matching shape")
    stat = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
    alpha = 1.0 - level
    if distribution not in {"t", "normal"}:
        raise ValueError("distribution must be 't' or 'normal'")
    valid_t = df is not None and np.isfinite(df) and df > 0
    if distribution == "t" and not valid_t:
        # An undefined t reference distribution is not a normal approximation.
        stat[:] = np.nan
        p = np.full_like(b, np.nan)
        crit = float("nan")
    else:
        dist = student_t(df) if distribution == "t" else norm
        p = 2.0 * dist.sf(np.abs(stat))
        crit = float(dist.ppf(1.0 - alpha / 2.0))
    lo, hi = b - crit * se, b + crit * se
    stars = np.full(len(b), "", dtype=object)
    stars[p < 0.10] = "*"
    stars[p < 0.05] = "**"
    stars[p < 0.01] = "***"
    index = list(names) if names else [f"x{i}" for i in range(len(b))]
    return pd.DataFrame(
        {
            "estimate": b,
            "std_error": se,
            "statistic": stat,
            "p_value": p,
            "ci_low": lo,
            "ci_high": hi,
            "stars": stars,
        },
        index=index,
    )


def cluster_counts(clusters) -> tuple[int, ...]:
    if clusters is None:
        return ()
    raw = clusters if isinstance(clusters, (list, tuple)) else [clusters]
    return tuple(int(np.unique(np.asarray(c)).size) for c in raw)


def fe_names_from_metadata(metadata) -> tuple[str, ...]:
    if not metadata:
        return ()
    out = []
    for m in metadata:
        name = getattr(m, "name", None)
        if name is not None:
            out.append(str(name))
    return tuple(out)


def reghdfe_r2_statistics(
    y, y_within, residuals, *, weights=None, effective_n=None, rank=0,
    df_absorbed=0, df_nested=0, has_intercept=True,
) -> dict[str, float]:
    """Compute reghdfe-compatible R-squared statistics.

    The fit residual degrees of freedom are intentionally distinct from the
    inference degrees of freedom reported by clustered VCEs.  reghdfe computes
    adjusted R-squared using ``N - df_a - df_m - df_a_nested`` even when
    ``e(df_r)`` is later capped at ``G-1`` for cluster inference.
    """
    y = np.asarray(y, dtype=np.float64)
    yw = np.asarray(y_within, dtype=np.float64)
    e = np.asarray(residuals, dtype=np.float64)
    if not (len(y) == len(yw) == len(e)):
        raise ValueError("y, y_within, and residuals must have matching lengths")
    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != len(y):
            raise ValueError("weights must have one entry per observation")
    sw = float(np.sum(w))
    if sw <= 0:
        return {
            "r2": float("nan"), "r2_within": float("nan"),
            "r2_adjusted": float("nan"), "r2_adjusted_within": float("nan"),
            "df_resid_fit": float("nan"),
        }
    rss = float(np.sum(w * e ** 2))
    if has_intercept:
        mean = float(np.sum(w * y) / sw)
        tss = float(np.sum(w * (y - mean) ** 2))
    else:
        tss = float(np.sum(w * y ** 2))
    tss_within = float(np.sum(w * yw ** 2))
    eps = np.finfo(float).eps
    r2 = float("nan") if tss <= eps else 1.0 - rss / tss
    r2w = float("nan") if tss_within <= eps else 1.0 - rss / tss_within

    n_eff = float(len(y) if effective_n is None else effective_n)
    rank = int(rank)
    df_fit = n_eff - float(df_absorbed) - float(df_nested) - float(rank)
    if df_fit <= 0 or not np.isfinite(df_fit):
        r2a = r2aw = float("nan")
    else:
        total_df = n_eff - (1.0 if has_intercept else 0.0)
        r2a = (
            float("nan") if tss <= eps or total_df <= 0
            else 1.0 - (rss / df_fit) / (tss / total_df)
        )
        within_df = df_fit + rank
        r2aw = (
            float("nan") if tss_within <= eps or within_df <= 0
            else 1.0 - (rss / df_fit) / (tss_within / within_df)
        )
    return {
        "r2": float(r2), "r2_within": float(r2w),
        "r2_adjusted": float(r2a), "r2_adjusted_within": float(r2aw),
        "df_resid_fit": float(df_fit),
    }


def _weighted_sums_of_squares(y, y_within, residuals, *, weights=None, has_intercept=True):
    """Return weighted RSS/TSS/within-TSS using the estimation metric."""
    y = np.asarray(y, dtype=np.float64)
    yw = np.asarray(y_within, dtype=np.float64)
    e = np.asarray(residuals, dtype=np.float64)
    if not (len(y) == len(yw) == len(e)):
        raise ValueError("y, y_within, and residuals must have matching lengths")
    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != len(y):
            raise ValueError("weights must have one entry per observation")
    sw = float(np.sum(w))
    if sw <= 0:
        return float("nan"), float("nan"), float("nan"), sw
    rss = float(np.sum(w * e ** 2))
    if has_intercept:
        mean = float(np.sum(w * y) / sw)
        tss = float(np.sum(w * (y - mean) ** 2))
    else:
        tss = float(np.sum(w * y ** 2))
    tss_within = float(np.sum(w * yw ** 2))
    return rss, tss, tss_within, sw


def _model_f_test(params, vcov, *, df_model, df_resid):
    """Wald/F test of all active reported slope coefficients.

    This follows reghdfe/ivreg2 reporting semantics: a singular covariance
    block does not get silently pseudo-inverted into a finite model F.
    """
    b = np.asarray(params, dtype=np.float64).reshape(-1)
    V = np.asarray(vcov, dtype=np.float64)
    q = int(df_model)
    if q <= 0 or len(b) != q or V.shape != (q, q):
        return float("nan"), float("nan")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(V)):
        return float("nan"), float("nan")
    if np.linalg.matrix_rank(V) < q:
        return float("nan"), float("nan")
    try:
        wald = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    F = wald / q
    if not np.isfinite(df_resid) or float(df_resid) <= 0:
        return F, float("nan")
    return F, float(fisher_f.sf(F, q, float(df_resid)))


def reghdfe_model_statistics(
    y, y_within, residuals, params, vcov, *, weights=None, effective_n=None, rank=0,
    df_absorbed=0, df_nested=0, df_resid_inference=None, has_intercept=True,
) -> dict[str, float]:
    """Compute the main linear-model scalars using current reghdfe semantics.

    The function deliberately separates inference DoF (possibly ``G-1`` under
    clustering) from fit-statistics DoF.  The latter is
    ``N - df_a - df_m - df_a_nested`` in current reghdfe.
    """
    base = reghdfe_r2_statistics(
        y, y_within, residuals, weights=weights, effective_n=effective_n, rank=rank,
        df_absorbed=df_absorbed, df_nested=df_nested, has_intercept=has_intercept,
    )
    rss, tss, tss_within, _ = _weighted_sums_of_squares(
        y, y_within, residuals, weights=weights, has_intercept=has_intercept,
    )
    n_eff = float(len(np.asarray(y)) if effective_n is None else effective_n)
    df_fit = float(base["df_resid_fit"])
    eps = np.finfo(float).eps
    mss = float(tss - rss) if np.isfinite(tss) and np.isfinite(rss) else float("nan")
    rmse = (float(np.sqrt(rss / df_fit))
            if np.isfinite(rss) and rss >= 0 and np.isfinite(df_fit) and df_fit > 0
            else float("nan"))
    ll = (
        -0.5 * n_eff * (1.0 + np.log(2.0 * np.pi) + np.log(rss / n_eff))
        if n_eff > 0 and np.isfinite(rss) and rss > eps else float("nan")
    )
    ll0 = (
        -0.5 * n_eff * (1.0 + np.log(2.0 * np.pi) + np.log(tss_within / n_eff))
        if n_eff > 0 and np.isfinite(tss_within) and tss_within > eps else float("nan")
    )
    infer_df = float("nan") if df_resid_inference is None else float(df_resid_inference)
    F, Fp = _model_f_test(params, vcov, df_model=int(rank), df_resid=infer_df)
    return {
        **base, "rss": float(rss), "tss": float(tss),
        "tss_within": float(tss_within), "mss": float(mss), "rmse": float(rmse),
        "loglike": float(ll), "loglike_null": float(ll0),
        "f_statistic": float(F), "f_pvalue": float(Fp),
        "df_model": float(rank),
    }


def ivreghdfe_model_statistics(
    y_within, residuals, params, vcov, *, weights=None, effective_n=None, rank=0,
    df_absorbed=0, df_nested=0, df_resid_inference=None,
) -> dict[str, float]:
    """Main unambiguous ivreg2/ivreghdfe-small fit scalars after absorption.

    ``ivreghdfe`` historically reports the partialled-out fit rather than a
    reconstructed overall TSS.  We therefore expose RSS, within TSS, RMSE and
    the model Wald/F test here, while leaving econhdfe's existing extended
    overall R-squared fields unchanged.
    """
    yw = np.asarray(y_within, dtype=np.float64)
    e = np.asarray(residuals, dtype=np.float64)
    if len(yw) != len(e):
        raise ValueError("y_within and residuals must have matching lengths")
    if weights is None:
        w = np.ones(len(yw), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != len(yw):
            raise ValueError("weights must have one entry per observation")
    rss = float(np.sum(w * e ** 2))
    tss_within = float(np.sum(w * yw ** 2))
    n_eff = float(len(yw) if effective_n is None else effective_n)
    # ivreg2-small excludes nested FE from the fit denominator; ivreghdfe
    # retains one absorbed intercept when all FE are nested (df_a == 0).
    iv_df = max(float(df_absorbed), float(df_nested > 0))
    df_fit = n_eff - iv_df - float(rank)
    rmse = (float(np.sqrt(rss / df_fit))
            if rss >= 0 and np.isfinite(df_fit) and df_fit > 0 else float("nan"))
    infer_df = float("nan") if df_resid_inference is None else float(df_resid_inference)
    F, Fp = _model_f_test(params, vcov, df_model=int(rank), df_resid=infer_df)
    return {
        "rss": rss, "tss_within": tss_within, "rmse": rmse,
        "f_statistic": float(F), "f_pvalue": float(Fp),
        "df_model": float(rank), "df_resid_fit": float(df_fit),
    }


@runtime_checkable
class PublicationResult(Protocol):
    """Thin protocol used by table/export tooling across estimator families."""

    def coef_table(self, level: float | None = None): ...
    def model_stats(self) -> dict[str, Any]: ...
    def publication_output(self, *, level: float | None = None,
                           include_diagnostics: bool | None = None,
                           include_profile: bool = False) -> dict[str, Any]: ...


def publication_output(result: PublicationResult, *, level: float | None = None,
                       include_diagnostics: bool | None = None,
                       include_profile: bool = False) -> dict[str, Any]:
    return result.publication_output(
        level=level,
        include_diagnostics=include_diagnostics,
        include_profile=include_profile,
    )


def reproducibility_dict(version: str | None = None, *, solver: str | None = None,
                         tolerance: float | None = None,
                         dof_method: str | None = None,
                         seed: int | None = None) -> dict[str, Any]:
    import platform
    import scipy
    if version is None:
        try:
            from importlib.metadata import version as _dist_version
            version = _dist_version("econhdfe")
        except Exception:
            version = "development"
    out: dict[str, Any] = {
        "econhdfe": str(version),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import numba
        out["numba"] = numba.__version__
    except Exception:  # pragma: no cover - optional runtime metadata only
        pass
    if solver is not None:
        out["solver"] = str(solver)
    if tolerance is not None:
        out["tolerance"] = float(tolerance)
    if dof_method is not None:
        out["dof_method"] = str(dof_method)
    if seed is not None:
        out["seed"] = int(seed)
    return out
