"""Estimator-level stages shared by OLS and linear IV.

``olshdfe``/``ivhdfe`` and the repeated-spec sessions run the same pipeline
(:mod:`econhdfe.pipeline`) and differ in their estimation kernel. This module
owns the remaining common steps, so direct and session fits cannot drift
apart: config precedence, estimation-sample preparation, group()-mode
completion, FE recovery, fit statistics with ``RegressionResult`` assembly,
prediction metadata and the profile stopwatch.
"""
from __future__ import annotations
from dataclasses import replace
from time import perf_counter
import numpy as np
from ..data import EstimationSampleState
from ..frontend.validate import require_numeric
from ..frontend.roles import VariableRole
from ..compute.weights import normalize_weight_type, prepare_weights
from ..pipeline import column_values, prepare_standard_fe_structure
from ..results import RegressionResult, FixedEffectEstimates, FixedEffectTermEstimate
from ..prediction import linear_prediction_state
from ..effects.prediction import saved_categorical_state
from ..reporting import (
    cluster_counts, reghdfe_model_statistics, reghdfe_r2_statistics, ivreghdfe_model_statistics,
)


# (prediction role, collinearity_info key) of the exog/endog/instrument designs.
IV_DESIGN_ROLES = (
    ("exogenous", "exogenous"),
    ("endogenous", "endogenous"),
    ("excluded_instrument", "excluded_instruments"),
)

_R2_FIELDS = ("r2", "r2_within", "r2_adjusted", "r2_adjusted_within")
_OLS_FIT_FIELDS = _R2_FIELDS + (
    "rss", "tss", "tss_within", "mss", "rmse", "loglike", "loglike_null",
    "f_statistic", "f_pvalue", "df_model", "df_resid_fit",
)
_IV_FIT_FIELDS = ("rss", "tss_within", "rmse", "f_statistic", "f_pvalue", "df_model", "df_resid_fit")


class StageClock:
    """Profile-only stage stopwatch; every method is a no-op when profiling is off."""

    __slots__ = ("start", "mark", "mode", "stages")

    def __init__(self, execution_config):
        enabled = execution_config is not None and execution_config.profile != "off"
        self.start = perf_counter() if enabled else None
        self.mark = self.start
        self.mode = execution_config.profile if enabled else None
        self.stages = {}

    @property
    def enabled(self) -> bool:
        return self.start is not None

    def lap(self, name):
        """Close the running stage as ``name``."""
        if self.start is not None:
            now = perf_counter()
            self.stages[name] = now - self.mark
            self.mark = now

    def credit(self, name, seconds):
        """Book a separately timed interval to ``name`` and exclude it from the next lap."""
        if self.start is not None:
            self.stages[name] = self.stages.get(name, 0.0) + seconds
            self.mark += seconds

    def summary(self):
        return {"total_seconds": perf_counter() - self.start, "mode": self.mode}


def apply_config_overrides(
    hdfe_config, inference_config, execution_config, *, method, tol, max_iter,
    dof_method, canonicalize_fe, vce, memory_budget_mb, absorb_threads,
):
    """Validate config objects and let each replace its scalar arguments.

    A supplied ``HDFEConfig`` replaces all solver scalars, ``ExecutionConfig``
    replaces the memory budget (and threads unless ``"auto"``), and
    ``InferenceConfig`` replaces ``vce`` only when it sets one.
    """
    if hdfe_config is not None:
        hdfe_config.validate()
        method, tol, max_iter = hdfe_config.solver, hdfe_config.tolerance, hdfe_config.max_iter
        dof_method, canonicalize_fe = hdfe_config.dof_method, hdfe_config.canonicalize
    if inference_config is not None:
        inference_config.validate()
        if inference_config.vce is not None:
            vce = inference_config.vce
    if execution_config is not None:
        execution_config.validate()
        memory_budget_mb = execution_config.memory_budget_mb
        if execution_config.threads != "auto":
            absorb_threads = execution_config.threads
    return method, tol, max_iter, dof_method, canonicalize_fe, vce, memory_budget_mb, absorb_threads


def reporting_options(inference_config):
    """Result fields that only record the requested reporting policy."""
    if inference_config is None:
        return {"confidence_level": 0.95, "diagnostics_mode": "off"}
    return {"confidence_level": inference_config.confidence_level,
            "diagnostics_mode": inference_config.diagnostics}


def prepare_standard_sample(data, y, weights, weight_type, absorb, *, canonicalize_fe, drop_singletons):
    """Outcome, raw weights and the FE structure of a standard (non-group) fit.

    Weights are validated on the input rows before singleton pruning; the
    returned ``EstimationSampleState`` records the pruning.
    """
    y0 = require_numeric(column_values(data, y), name=str(y), role=VariableRole.OUTCOME, ndim=1)
    sample_state = EstimationSampleState(len(y0))
    w0 = None if weights is None else require_numeric(
        column_values(data, weights), name=str(weights), role=VariableRole.WEIGHT, ndim=1,
    )
    wkind = normalize_weight_type(weight_type, has_weights=w0 is not None)
    _ = prepare_weights(w0, len(y0), wkind)  # validate before singleton pruning
    fe = prepare_standard_fe_structure(
        data, absorb, canonicalize_fe=canonicalize_fe, drop_singletons=drop_singletons,
        weights=w0, weight_kind=wkind,
    )
    if fe.dropped:
        sample_state.keep(fe.mask, stage="hdfe", reason="singleton_pruning")
    return y0, w0, wkind, fe, sample_state


def finish_group_mode(result, *, allow_nonconverged, inference_config, clock, data_plan):
    if not result.converged and not allow_nonconverged:
        raise RuntimeError("group/individual FE absorption did not converge")
    options = reporting_options(inference_config)
    result.confidence_level = options["confidence_level"]
    result.diagnostics_mode = options["diagnostics_mode"]
    if clock.enabled:
        result.profile = clock.summary()
        if data_plan is not None:
            result.profile["data_ingestion"] = data_plan.as_dict()
    return result


def recover_fixed_effects(absorber, target, names):
    """Solve the absorbed FE coefficients reproducing ``target`` (y - xb - e)."""
    raw_terms, meta = absorber.solve_effects(target)
    return FixedEffectEstimates(
        terms=tuple(
            FixedEffectTermEstimate(names[i], intercept, slope_coef, contribution)
            for i, (intercept, slope_coef, contribution) in enumerate(raw_terms)
        ),
        converged=bool(meta["converged"]),
        iterations=int(meta["iterations"]),
        residual_norm=float(meta["residual_norm"]),
        reconstruction_error=float(meta["reconstruction_error"]),
    )


def linear_result(beta, V, resid, *, y, y_within, rank, df_resid, dof, winfo, clusters, vce,
                  has_intercept, iv=False, **fields):
    """Assemble the ``RegressionResult`` of one OLS/IV fit on its estimation sample.

    OLS fit scalars follow reghdfe. IV keeps reghdfe's R-squared family but
    takes RSS/RMSE/F and the fit DoF from ivreghdfe's partialled-out
    convention. ``fields`` carries the estimator-specific metadata.
    """
    common = dict(weights=winfo.estimation, effective_n=winfo.effective_n, rank=rank,
                  df_absorbed=dof.df_absorbed, df_nested=dof.nested)
    if iv:
        r2 = reghdfe_r2_statistics(y, y_within, resid, **common, has_intercept=has_intercept)
        fit = ivreghdfe_model_statistics(y_within, resid, beta, V, **common, df_resid_inference=df_resid)
        stats = {**{k: r2[k] for k in _R2_FIELDS}, **{k: fit[k] for k in _IV_FIT_FIELDS}}
    else:
        fit = reghdfe_model_statistics(
            y, y_within, resid, beta, V, **common, df_resid_inference=df_resid,
            has_intercept=has_intercept,
        )
        stats = {k: fit[k] for k in _OLS_FIT_FIELDS}
    return RegressionResult(
        params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
        residuals=resid, fitted=y - resid, nobs=int(round(winfo.effective_n)), rank=rank,
        df_resid=df_resid, df_absorbed=dof.df_absorbed, dof_info=dof,
        weight_type=winfo.kind, sum_weights=winfo.sum_weights, nobs_raw=len(y_within),
        vce=vce, cluster_counts=cluster_counts(clusters),
        vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        **stats, **fields,
    )


def prediction_state(estimator, names, roles, designs, fe, **sample):
    """Prediction metadata for the retained coefficients on FE structure ``fe``."""
    return linear_prediction_state(
        estimator=estimator,
        coefficient_names=names,
        coefficient_roles=roles,
        designs=designs,
        fe_plan=fe.plan,
        fe_names=fe.names,
        requested_fe_groups=() if fe.inference is None else fe.inference.groups,
        effective_fe_groups=fe.groups,
        fe_intercepts=fe.intercepts,
        fe_slopes=fe.slopes,
        **sample,
    )


def with_saved_effects(state, fe, *, data, recovered, beta, identifier_levels):
    """Attach save_fe() categorical FE state so new rows can include FE predictions."""
    fe_state = saved_categorical_state(
        state.fixed_effects, data=data, mask=fe.mask,
        inference=fe.inference, plan=fe.plan, groups=fe.groups,
        recovered=recovered, beta=beta, identifier_levels=identifier_levels,
    )
    return replace(state, fixed_effects=fe_state)
