from __future__ import annotations
from dataclasses import fields
from time import perf_counter
from ...errors import error_boundary, SpecificationError, ShapeError, DataTypeError
from ...config import InferenceConfig, ExecutionConfig, resolve_override
from ...compute.context import ExecutionContext
from ...planner.scheduler import managed_execution
from ...frontend.validate import require_numeric
from ...frontend.roles import VariableRole
import numpy as np
import pandas as pd
from .config import PPMLConfig
from .estimator import fit_arrays
from ...hdfe.plan import FEPlan
from ...compute.wls import as_2d
from ..poisson import prepare_poisson_base, prepare_ppml_weights
from ...design import is_heterogeneous_spec_candidate
from .heterogeneous import fit_structured_dataframe


def _prepare_inputs(y, X, *, offset, exposure, weights, names, weight_type=None):
    y, off, w = prepare_poisson_base(
        y, offset=offset, exposure=exposure, weights=weights
    )
    X = as_2d(X, len(y))
    X = require_numeric(X, name="X", role=VariableRole.REGRESSOR, ndim=2)
    names = tuple(names or [f"x{i}" for i in range(X.shape[1])])
    if len(names) != X.shape[1]:
        raise ShapeError("names length mismatch", details={"expected": X.shape[1], "actual": len(names)})
    w, weight_kind = prepare_ppml_weights(weights, len(y), weight_type)
    return y, X, off, w, names, weight_kind


#: PPML-family default VCE. It deliberately differs from OLS/IV (``None``,
#: reghdfe's unadjusted default) because Stata's ppmlhdfe defaults to robust.
PPML_DEFAULT_VCE = "robust"


def resolve_ppml_vce(vce, inference_config):
    """Combine the scalar ``vce`` argument with ``InferenceConfig.vce``.

    Each source counts as unset at its own default (``"robust"`` for the
    argument, the ``InferenceConfig`` field default for the config); setting
    both to different values raises ``specification.config_conflict``.
    """
    config_default = next(f.default for f in fields(InferenceConfig) if f.name == "vce")
    config_value = config_default if inference_config is None else inference_config.vce
    return resolve_override("vce", vce, PPML_DEFAULT_VCE, config_value, config_default)


def _resolve_call_configs(vce, config, config_type, inference_config, execution_config):
    """Validate functional-entry configs and resolve the VCE request."""
    if inference_config is not None:
        inference_config.validate()
    vce = resolve_ppml_vce(vce, inference_config)
    execution_config = ExecutionConfig() if execution_config is None else execution_config
    execution_config.validate()
    context = ExecutionContext(execution_config)
    t0 = perf_counter() if execution_config.profile != "off" else None
    config = config_type() if config is None else config
    config.validate()
    return vce, config, execution_config, context, t0


def _finish_result(result, inference_config, execution_config, t0):
    """Attach reporting settings and, when profiling, wall-clock timing."""
    result.confidence_level = 0.95 if inference_config is None else inference_config.confidence_level
    result.diagnostics_mode = "off" if inference_config is None else inference_config.diagnostics
    if t0 is not None:
        result.profile = {
            "total_seconds": perf_counter() - t0, "mode": execution_config.profile,
            "separation_seconds": dict(result.diagnostics.get("separation_seconds", {})),
        }
    return result


class _ReusableFEModel:
    """DataFrame, compiled FE plan and cache state shared by PPML-family models."""

    def _setup(self, data, absorb, config, config_type, inference_config, execution_config):
        self.data = data
        self.absorb = tuple(absorb)
        self.config = config_type() if config is None else config
        self.config.validate()
        self.inference_config = InferenceConfig() if inference_config is None else inference_config
        self.inference_config.validate()
        self.execution_config = ExecutionConfig() if execution_config is None else execution_config
        self.execution_config.validate()
        self.context = ExecutionContext(self.execution_config)
        self.plan = FEPlan.from_dataframe(data, self.absorb)
        self._source_fingerprint = (
            FEPlan.source_fingerprint(data, self.absorb)
            if self.execution_config.cache_validation == "signature" else None
        )
        self.context.put(self.plan.fingerprint, self.plan)

    def _prepare_resource_admission(self):
        # self.plan/data remain owned by the model even if cache entries go.
        self.context.prepare_admission(self.execution_config, protected=(self.plan, self.data))

    def _ensure_plan_current(self):
        if not self.absorb:
            return
        rebuild = self.plan.nobs != len(self.data)
        if not rebuild and self.execution_config.cache_validation == "signature":
            current = FEPlan.source_fingerprint(self.data, self.absorb)
            rebuild = current != self._source_fingerprint
        if rebuild:
            self.plan = FEPlan.from_dataframe(self.data, self.absorb)
            self._source_fingerprint = (
                FEPlan.source_fingerprint(self.data, self.absorb)
                if self.execution_config.cache_validation == "signature" else None
            )
            self.context.clear()
            self.context.put(self.plan.fingerprint, self.plan)


def _require_frame(data, estimator, **array_only):
    """DataFrame-mode guard: reject non-frames and array-only label options."""
    if not isinstance(data, pd.DataFrame):
        raise DataTypeError(
            f"{estimator} data must be a pandas DataFrame or None",
            code="input.data_type", details={"actual_type": type(data).__name__},
            suggestion="Pass data=None with array inputs, or a DataFrame with column-name specifications.",
        )
    given = sorted(name for name, value in array_only.items() if value is not None)
    if given:
        raise SpecificationError(
            f"{', '.join(given)} only label array inputs; DataFrame columns supply the names",
            code="specification.array_only_option", stage="specification",
            details={"options": given},
        )


def _spec_tuple(value):
    """Role specification as a term tuple; a single term may be passed unwrapped."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


@error_boundary("ppml")
def ppmlhdfe(
    data=None,
    *,
    y,
    x=None,
    absorb=None,
    offset=None,
    exposure=None,
    weights=None,
    weight_type=None,
    vce="robust",
    cluster=None,
    names=None,
    config: PPMLConfig | None = None,
    inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    warm_start=None,
):
    """Estimate PPML with optional high-dimensional categorical fixed effects.

    With a DataFrame ``data``, ``y``/``x``/``absorb``/``offset``/``exposure``/
    ``weights``/``cluster`` are column specifications (``x`` may contain
    ``factor()``/interaction terms) and the call is exactly
    ``PPMLHDFE(data, absorb=...).fit(y, x, ...)``.  With ``data=None`` they
    are arrays: ``absorb`` and ``cluster`` are sequences of 1-D arrays and
    ``names`` labels the columns of ``x``.

    ``weight_type="fw"`` counts frequency copies in N and the robust meat;
    ``"pw"`` counts sampled rows. Omission preserves legacy generic weights.
    ``config.singleton_mode`` selects ``"stata_compat"`` (default) or the
    frequency-invariant ``"unified"`` sample policy.

    ``vce`` defaults to ``"robust"`` (Stata ppmlhdfe's default), unlike
    OLS/IV whose ``None`` default is reghdfe's unadjusted VCE.  A non-default
    ``vce`` and a different ``inference_config.vce`` raise
    ``specification.config_conflict``; either one alone takes effect.
    """
    if data is None:
        return _ppmlhdfe_arrays(
            y, x, absorb=absorb, offset=offset, exposure=exposure, weights=weights, weight_type=weight_type,
            vce=vce, cluster=cluster, names=names, config=config,
            inference_config=inference_config, execution_config=execution_config,
            warm_start=warm_start,
        )
    _require_frame(data, "ppmlhdfe", names=names)
    model = PPMLHDFE(
        data, absorb=_spec_tuple(absorb), config=config,
        inference_config=inference_config, execution_config=execution_config,
    )
    return model.fit(
        y, _spec_tuple(x), offset=offset, exposure=exposure, weights=weights, weight_type=weight_type,
        vce=vce, cluster=cluster, warm_start=warm_start,
    )


@managed_execution
def _ppmlhdfe_arrays(
    y, x, *, absorb, offset, exposure, weights, weight_type, vce, cluster, names,
    config, inference_config, execution_config, warm_start,
):
    vce, config, execution_config, context, t0 = _resolve_call_configs(
        vce, config, PPMLConfig, inference_config, execution_config,
    )
    y, X, off, w, names, weight_kind = _prepare_inputs(
        y, x, offset=offset, exposure=exposure, weights=weights, names=names, weight_type=weight_type,
    )
    plan = FEPlan.from_arrays(absorb or ())
    if plan.groups and plan.nobs != len(y):
        raise ShapeError("absorb arrays must have nobs rows", details={"expected": len(y), "actual": plan.nobs})
    result = fit_arrays(
        y, X, plan, offset=off, true_w=w, weight_kind=weight_kind, vce=vce,
        clusters=cluster, names=names, config=config, warm_start=warm_start,
        execution_config=execution_config, execution_context=context,
    )
    return _finish_result(result, inference_config, execution_config, t0)


class PPMLHDFE(_ReusableFEModel):
    """Reusable DataFrame model; categorical FE encoding is compiled once."""

    @error_boundary("ppml")
    def __init__(self, data: pd.DataFrame, *, absorb=(), config: PPMLConfig | None = None,
                 inference_config: InferenceConfig | None = None,
                 execution_config: ExecutionConfig | None = None):
        self._setup(data, absorb, config, PPMLConfig, inference_config, execution_config)

    @error_boundary("ppml")
    @managed_execution
    def fit(self, y: str, x=(), *, offset=None, exposure=None, weights=None, weight_type=None, vce="robust", cluster=None, warm_start=None):
        """Fit one PPML specification on the model's DataFrame.

        ``vce`` defaults to ``"robust"`` (ppmlhdfe's default); it conflicts
        with a different non-default ``inference_config.vce``.
        """
        self._ensure_plan_current()
        vce = resolve_ppml_vce(vce, self.inference_config)
        with self.context.maintain(self.execution_config, protected=(self.plan, self.data)):
            if is_heterogeneous_spec_candidate(x):
                result, _ = fit_structured_dataframe(
                    self.data, y=y, x=x, plan=self.plan,
                    offset=offset, exposure=exposure, weights=weights, weight_type=weight_type,
                    vce=vce,
                    clusters=cluster, config=self.config,
                    execution_config=self.execution_config,
                    warm_start=warm_start,
                )
                return _finish_result(result, self.inference_config, self.execution_config, None)

            cols = tuple(x)
            yv, Xv, off, w, names, weight_kind = _prepare_inputs(
                self.data[y].to_numpy(),
                self.data[list(cols)].to_numpy() if cols else None,
                offset=None if offset is None else self.data[offset].to_numpy(),
                exposure=None if exposure is None else self.data[exposure].to_numpy(),
                weights=None if weights is None else self.data[weights].to_numpy(),
                names=cols, weight_type=weight_type,
            )
            cluster_arrays = None
            if cluster is not None:
                ccols = (cluster,) if isinstance(cluster, str) else tuple(cluster)
                cluster_arrays = [self.data[c].to_numpy() for c in ccols]
            t0 = perf_counter() if self.execution_config.profile != "off" else None
            result = fit_arrays(
                yv, Xv, self.plan, offset=off, true_w=w, weight_kind=weight_kind, vce=vce,
                clusters=cluster_arrays, names=names, config=self.config, warm_start=warm_start,
                execution_config=self.execution_config, execution_context=self.context,
            )
            return _finish_result(result, self.inference_config, self.execution_config, t0)
