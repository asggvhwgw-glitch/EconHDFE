from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ...reporting import inference_table, reproducibility_dict


class _PoissonResultMethods:
    """Inference and reporting methods shared by PPML-family result dataclasses."""

    __slots__ = ()

    def coef_table(self, level=None):
        return inference_table(self.names, self.params, self.stderr,
                               level=self.confidence_level if level is None else level,
                               distribution="normal")

    @property
    def pvalues(self):
        return self.coef_table()["p_value"].to_numpy()

    def conf_int(self, level=None):
        t = self.coef_table(level); return t[["ci_low", "ci_high"]].to_numpy()

    def publication_output(self, *, level=None, include_diagnostics=None, include_profile=None):
        """Publication tables; ``diagnostics="full"`` also includes the profile by default."""
        out = {"coefficients": self.coef_table(level), "model": self.model_stats()}
        show_diag = self.diagnostics_mode != "off" if include_diagnostics is None else bool(include_diagnostics)
        if include_profile is None:
            include_profile = self.diagnostics_mode == "full"
        if show_diag: out["diagnostics"] = self.diagnostics
        if include_profile and self.profile is not None: out["profile"] = self.profile
        out["reproducibility"] = self.reproducibility or reproducibility_dict()
        return out


@dataclass(slots=True)
class PPMLResult(_PoissonResultMethods):
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
    df_absorbed: int = 0
    loglike_null: float | None = None
    pseudo_r2: float | None = None
    chi2: float | None = None
    chi2_pvalue: float | None = None
    df_model: int | None = None
    nobs_full: int | None = None
    vcov_rank: int | None = None
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

    def model_stats(self):
        out = {
            "estimator": "ppml", "nobs": int(self.nobs),
            "df_resid": None if self.df_resid is None else float(self.df_resid),
            "df_absorbed": int(self.df_absorbed), "vce": self.vce,
            "cluster_counts": tuple(self.cluster_counts), "fixed_effects": tuple(self.fe_names),
            "n_separated": int(self.n_separated), "n_singletons": int(self.n_singletons),
            "iterations": int(self.iterations), "converged": bool(self.converged),
            "loglike": float(self.loglike), "deviance": float(self.deviance),
            "weight_type": self.weight_type, "sum_weights": self.sum_weights,
        }
        if self.loglike_null is not None: out["loglike_null"] = float(self.loglike_null)
        if self.pseudo_r2 is not None: out["pseudo_r2"] = float(self.pseudo_r2)
        if self.chi2 is not None: out["chi2"] = float(self.chi2)
        if self.chi2_pvalue is not None: out["chi2_pvalue"] = float(self.chi2_pvalue)
        if self.df_model is not None: out["df_model"] = int(self.df_model)
        if self.nobs_full is not None: out["nobs_full"] = int(self.nobs_full)
        if self.vcov_rank is not None: out["vcov_rank"] = int(self.vcov_rank)
        return out
