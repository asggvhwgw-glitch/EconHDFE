from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .reporting import inference_table, reproducibility_dict
from .postestimation import linear_combination as _linear_combination, wald_test as _wald_test


@dataclass(slots=True)
class DofInfo:
    """Serializable degrees-of-freedom metadata shared by HDFE and results.

    The type belongs to the result contract rather than the HDFE implementation so
    data/cache infrastructure can persist result metadata without depending on the
    numerical HDFE subsystem.
    """
    df_absorbed: int
    initial: int
    redundant: int
    nested: int
    k_by_component: tuple[int, ...]
    m_by_component: tuple[int, ...]
    exact_by_component: tuple[bool, ...]
    nested_by_component: tuple[bool, ...]
    method: str


@dataclass(slots=True)
class EstimationState:
    absorber: Any
    y_within: np.ndarray
    X_within: np.ndarray
    clusters: tuple[np.ndarray, ...] = ()
    weights: np.ndarray | None = None
    weight_type: str = "none"


@dataclass(slots=True)
class FixedEffectTermEstimate:
    """Recovered coefficients for one absorbed term.

    Level indices correspond to the absorber's dense codes. With multiple FE
    dimensions the decomposition is generally not uniquely identified; the
    coefficients use the minimum-norm solution of the joint weighted least-
    squares problem. The observation-level ``contribution`` is directly usable
    for prediction on the estimation sample.
    """
    name: str
    intercept: np.ndarray | None
    slopes: np.ndarray | None
    contribution: np.ndarray


@dataclass(slots=True)
class FixedEffectEstimates:
    terms: tuple[FixedEffectTermEstimate, ...]
    converged: bool
    iterations: int
    residual_norm: float
    reconstruction_error: float
    normalization: str = "minimum_norm_lsmr"

    @property
    def fitted(self) -> np.ndarray:
        if not self.terms:
            return np.empty(0, dtype=np.float64)
        out = np.zeros_like(self.terms[0].contribution, dtype=np.float64)
        for term in self.terms:
            out += term.contribution
        return out


@dataclass(slots=True)
class RegressionResult:
    params: np.ndarray
    vcov: np.ndarray
    stderr: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    nobs: int
    rank: int
    df_resid: float
    df_absorbed: int
    converged: bool
    iterations: int
    dropped_singletons: int
    names: tuple[str, ...] = ()
    first_stage: dict | None = None
    diagnostics: dict | None = None
    estimator: str = "ols"
    estimator_info: dict | None = None
    dof_info: Any | None = None
    state: EstimationState | None = None
    weight_type: str = "none"
    sum_weights: float | None = None
    nobs_raw: int | None = None
    fixed_effects: FixedEffectEstimates | None = None
    group_info: Any | None = None
    absorb_info: Any | None = None
    collinearity_info: Any | None = None
    vce: str | None = None
    cluster_counts: tuple[int, ...] = ()
    fe_names: tuple[str, ...] = ()
    r2: float | None = None
    r2_within: float | None = None
    r2_adjusted: float | None = None
    r2_adjusted_within: float | None = None
    rss: float | None = None
    tss: float | None = None
    tss_within: float | None = None
    mss: float | None = None
    rmse: float | None = None
    loglike: float | None = None
    loglike_null: float | None = None
    f_statistic: float | None = None
    f_pvalue: float | None = None
    df_model: float | None = None
    df_resid_fit: float | None = None
    vcov_rank: int | None = None
    confidence_level: float = 0.95
    profile: dict | None = None
    reproducibility: dict | None = None
    diagnostics_mode: str = "off"

    @property
    def tvalues(self) -> np.ndarray:
        return self.params / self.stderr

    @property
    def kappa(self) -> float | None:
        if not self.estimator_info:
            return None
        value = self.estimator_info.get("kappa")
        return None if value is None else float(value)


    def coef_table(self, level: float | None = None):
        """Publication-ready coefficient table (estimate/SE/test/p/CI/stars)."""
        return inference_table(
            self.names, self.params, self.stderr,
            level=self.confidence_level if level is None else level,
            df=self.df_resid, distribution="t",
        )

    @property
    def pvalues(self) -> np.ndarray:
        return self.coef_table()["p_value"].to_numpy()

    def conf_int(self, level: float | None = None) -> np.ndarray:
        tab = self.coef_table(level)
        return tab[["ci_low", "ci_high"]].to_numpy()

    def linear_combination(self, weights, *, value: float = 0.0,\n                           level: float | None = None):\n        """Estimate and test one linear combination of reported coefficients."""\n        return _linear_combination(\n            self.params, self.vcov, weights, names=self.names, value=value,\n            df=self.df_resid, level=self.confidence_level if level is None else level,\n        )\n\n    def wald_test(self, restrictions=None, *, values=None, distribution: str = "F"):\n        """Test one or more linear restrictions R @ beta = values."""\n        return _wald_test(\n            self.params, self.vcov, restrictions, values=values, names=self.names,\n            df_resid=self.df_resid, distribution=distribution,\n        )\n\n    def model_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "estimator": self.estimator,
            "nobs": int(self.nobs),
            "nobs_raw": None if self.nobs_raw is None else int(self.nobs_raw),
            "rank": int(self.rank),
            "df_resid": float(self.df_resid),
            "df_absorbed": int(self.df_absorbed),
            "vce": self.vce,
            "cluster_counts": tuple(self.cluster_counts),
            "fixed_effects": tuple(self.fe_names),
            "dropped_singletons": int(self.dropped_singletons),
            "weight_type": self.weight_type,
            "converged": bool(self.converged),
        }
        if self.r2 is not None:
            out["r2"] = float(self.r2)
        if self.r2_within is not None:
            out["r2_within"] = float(self.r2_within)
        if self.r2_adjusted is not None:
            out["r2_adjusted"] = float(self.r2_adjusted)
        if self.r2_adjusted_within is not None:
            out["r2_adjusted_within"] = float(self.r2_adjusted_within)
        for key in (
            "rss", "tss", "tss_within", "mss", "rmse", "loglike",
            "loglike_null", "f_statistic", "f_pvalue", "df_model", "df_resid_fit",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = float(value)
        if self.vcov_rank is not None:
            out["vcov_rank"] = int(self.vcov_rank)
        if self.estimator_info:
            for key in ("kappa", "fuller"):
                if key in self.estimator_info and self.estimator_info[key] is not None:
                    out[key] = self.estimator_info[key]
        return out

    def publication_output(self, *, level: float | None = None,
                           include_diagnostics: bool | None = None,
                           include_profile: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "coefficients": self.coef_table(level),
            "model": self.model_stats(),
        }
        if self.first_stage is not None:
            # Avoid returning the potentially huge observation-level fitted-endog
            # matrix in publication output. Keep only reportable first-stage items.
            out["first_stage"] = {
                key: value for key, value in self.first_stage.items()
                if key in {"coefficients", "diagnostics"}
            }
        if self.diagnostics and self.estimator != "ols":
            # Econometric tests commonly reported with linear-IV estimates are
            # part of the publication surface, while the full diagnostic payload
            # remains opt-in.
            tests = {}
            if "kleibergen_paap" in self.diagnostics:
                kp = self.diagnostics["kleibergen_paap"]
                if isinstance(kp, dict):
                    tests["kleibergen_paap"] = {k: v for k, v in kp.items() if np.isscalar(v)}
            if "cragg_donald_f" in self.diagnostics:
                tests["cragg_donald_f"] = self.diagnostics["cragg_donald_f"]
            if "overidentification" in self.diagnostics:
                tests["overidentification"] = self.diagnostics["overidentification"]
            if tests:
                out["identification_tests"] = tests
        show_diag = self.diagnostics_mode != "off" if include_diagnostics is None else bool(include_diagnostics)
        if show_diag and self.diagnostics is not None:
            out["diagnostics"] = self.diagnostics
        if include_profile and self.profile is not None:
            out["profile"] = self.profile
        out["reproducibility"] = self.reproducibility or reproducibility_dict(
            solver=(self.absorb_info or {}).get("method") if isinstance(self.absorb_info, dict) else None,
            dof_method=getattr(self.dof_info, "method", None),
        )
        return out

    @property
    def omitted_variables(self) -> tuple[dict, ...]:
        """Requested columns omitted for absorption/collinearity.

        OLS stores a single collinearity plan; IV stores role-specific plans.
        The property flattens either representation for convenient auditing.
        """
        info = self.collinearity_info
        if not isinstance(info, dict):
            return ()
        if "omitted" in info:
            return tuple(info.get("omitted") or ())
        out = []
        for key in ("exogenous", "endogenous", "excluded_instruments"):
            block = info.get(key)
            if isinstance(block, dict):
                out.extend(block.get("omitted") or ())
        return tuple(out)

    @property
    def has_omitted_variables(self) -> bool:
        return bool(self.omitted_variables)

    @property
    def user_omitted_variables(self) -> tuple[dict, ...]:
        """Columns deliberately omitted by the specification (e.g. event-study reference periods)."""
        return tuple(o for o in self.omitted_variables if o.get("selected_by_user", False))

    @property
    def automatic_omitted_variables(self) -> tuple[dict, ...]:
        """Columns omitted automatically for absorption or rank deficiency."""
        return tuple(o for o in self.omitted_variables if not o.get("selected_by_user", False))
