from __future__ import annotations
from time import perf_counter
from ...errors import error_boundary, SpecificationError, ShapeError
from ...config import InferenceConfig, ExecutionConfig
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
from ..poisson import prepare_poisson_base
from ...design import is_heterogeneous_spec_candidate
from .heterogeneous import fit_structured_dataframe


def _prepare_inputs(y, X, *, offset, exposure, weights, names):
    y, off, w = prepare_poisson_base(
        y, offset=offset, exposure=exposure, weights=weights
    )
    X = as_2d(X, len(y))
    X = require_numeric(X, name="X", role=VariableRole.REGRESSOR, ndim=2)
    names = tuple(names or [f"x{i}" for i in range(X.shape[1])])
    if len(names) != X.shape[1]:
        raise ShapeError("names length mismatch", details={"expected": X.shape[1], "actual": len(names)})
    return y, X, off, w, names


def _resolve_call_configs(vce, config, config_type, inference_config, execution_config):
    """Validate functional-entry configs; an explicit inference VCE wins."""
    if inference_config is not None:
        inference_config.validate()
        if inference_config.vce is not None:
            vce = inference_config.vce
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


@error_boundary("ppml")
@managed_execution
def ppmlhdfe(
    y,
    X=None,
    *,
    absorb=None,
    offset=None,
    exposure=None,
    weights=None,
    vce="robust",
    clusters=None,
    names=None,
    config: PPMLConfig | None = None,
    inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    warm_start=None,
):
    """Estimate PPML with optional high-dimensional categorical fixed effects."""
    vce, config, execution_config, context, t0 = _resolve_call_configs(
        vce, config, PPMLConfig, inference_config, execution_config,
    )
    y, X, off, w, names = _prepare_inputs(
        y, X, offset=offset, exposure=exposure, weights=weights, names=names,
    )
    plan = FEPlan.from_arrays(absorb or ())
    if plan.groups and plan.nobs != len(y):
        raise ShapeError("absorb arrays must have nobs rows", details={"expected": len(y), "actual": plan.nobs})
    result = fit_arrays(
        y, X, plan, offset=off, true_w=w, vce=vce,
        clusters=clusters, names=names, config=config, warm_start=warm_start,
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
    def fit(self, y: str, X=(), *, offset=None, exposure=None, weights=None, vce="robust", clusters=None, warm_start=None):
        self._ensure_plan_current()
        with self.context.maintain(self.execution_config, protected=(self.plan, self.data)):
            if is_heterogeneous_spec_candidate(X):
                result, _ = fit_structured_dataframe(
                    self.data, y=y, x=X, plan=self.plan,
                    offset=offset, exposure=exposure, weights=weights,
                    vce=self.inference_config.vce if self.inference_config.vce is not None else vce,
                    clusters=clusters, config=self.config,
                    execution_config=self.execution_config,
                    warm_start=warm_start,
                )
                return _finish_result(result, self.inference_config, self.execution_config, None)

            cols = tuple(X)
            yv, Xv, off, w, names = _prepare_inputs(
                self.data[y].to_numpy(),
                self.data[list(cols)].to_numpy() if cols else None,
                offset=None if offset is None else self.data[offset].to_numpy(),
                exposure=None if exposure is None else self.data[exposure].to_numpy(),
                weights=None if weights is None else self.data[weights].to_numpy(),
                names=cols,
            )
            cluster_arrays = None
            if clusters is not None:
                ccols = (clusters,) if isinstance(clusters, str) else tuple(clusters)
                cluster_arrays = [self.data[c].to_numpy() for c in ccols]
            if self.inference_config.vce is not None:
                vce = self.inference_config.vce
            t0 = perf_counter() if self.execution_config.profile != "off" else None
            result = fit_arrays(
                yv, Xv, self.plan, offset=off, true_w=w, vce=vce,
                clusters=cluster_arrays, names=names, config=self.config, warm_start=warm_start,
                execution_config=self.execution_config, execution_context=self.context,
            )
            return _finish_result(result, self.inference_config, self.execution_config, t0)
