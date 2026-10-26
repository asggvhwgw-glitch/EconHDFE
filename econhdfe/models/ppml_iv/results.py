from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ...reporting import inference_table
from ..ppml.results import _PoissonResultMethods


@dataclass(slots=True)
class IVPPMLResult(_PoissonResultMethods):
    params: np.ndarray
    vcov: np.ndarray
    stderr: np.ndarray
    names: tuple[str, ...]
    converged: bool
    iterations: int
    deviance: float
    loglike: float
    nobs: int
    n_separated: int
    sample_mask: np.ndarray
    separation_mask: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    df_absorbed: int
    exog_names: tuple[str, ...]
    endog_names: tuple[str, ...]
    instrument_names: tuple[str, ...]
    moments: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    df_resid: float | None = None
    vce: str | None = None
    cluster_counts: tuple[int, ...] = ()
    fe_names: tuple[str, ...] = ()
    weight_type: str = "none"
    sum_weights: float | None = None
    n_singletons: int = 0
    confidence_level: float = 0.95
    profile: dict | None = None
    reproducibility: dict | None = None
    diagnostics_mode: str = "off"

    def coef_table(self, level=None):
        table = inference_table(self.names, self.params, self.stderr,
                                level=self.confidence_level if level is None else level,
                                distribution="normal")
        normalized = np.array([name == "_cons" for name in self.names]) & (self.stderr == 0)
        table.loc[normalized, ["statistic", "p_value", "ci_low", "ci_high"]] = np.nan
        return table

    def model_stats(self):
        return {
            "estimator": "ivppml", "nobs": int(self.nobs),
            "df_resid": None if self.df_resid is None else float(self.df_resid),
            "df_absorbed": int(self.df_absorbed), "vce": self.vce,
            "cluster_counts": tuple(self.cluster_counts), "fixed_effects": tuple(self.fe_names),
            "n_separated": int(self.n_separated), "n_singletons": int(self.n_singletons),
            "iterations": int(self.iterations), "converged": bool(self.converged),
            "loglike": float(self.loglike), "deviance": float(self.deviance),
            "endogenous": tuple(self.endog_names), "excluded_instruments": tuple(self.instrument_names),
            "weight_type": self.weight_type, "sum_weights": self.sum_weights,
        }
