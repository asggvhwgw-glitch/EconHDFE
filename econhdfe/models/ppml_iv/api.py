from __future__ import annotations
from time import perf_counter
from ...errors import error_boundary, SpecificationError, UnderidentifiedError, ShapeError
from ...config import InferenceConfig, ExecutionConfig
from ...compute.context import ExecutionContext
from ...planner.scheduler import managed_execution
from ...frontend.validate import require_numeric
from ...frontend.roles import VariableRole
import numpy as np
import pandas as pd
from ...compute.wls import as_2d
from ...compute.weights import prepare_weights
from ...hdfe.plan import FEPlan
from ..poisson import prepare_poisson_base
from .config import IVPPMLConfig
from .estimator import fit_arrays
from .heterogeneous import fit_structured_dataframe
from ...design import build_design, is_heterogeneous_spec_candidate


def _prepare(y, exog, endog, instruments, *, offset, exposure, weights, weight_type, exog_names, endog_names, instrument_names):
    y, off, _ = prepare_poisson_base(
        y, offset=offset, exposure=exposure, weights=weights
    )
    n = len(y)
    winfo = prepare_weights(weights, n, weight_type, normalize_ap=False)
    if winfo.kind == "aweight":
        raise SpecificationError("IV-PPML supports upstream-compatible fweight/pweight, not aweight", code="specification.weight_type")
    w = np.ones(n, dtype=np.float64) if winfo.estimation is None else winfo.estimation
    C, E, I = as_2d(exog, n), as_2d(endog, n), as_2d(instruments, n)
    if E.shape[1] == 0:
        raise UnderidentifiedError("at least one endogenous regressor is required", details={"n_endog": 0})
    if I.shape[1] < E.shape[1]:
        raise UnderidentifiedError("number of excluded instruments must be >= endogenous regressors", details={"n_endog": E.shape[1], "n_excluded_instruments": I.shape[1]}, suggestion="Add excluded instruments or remove endogenous regressors.")
    for label, A in (("exog", C), ("endog", E), ("instruments", I)):
        if np.any(~np.isfinite(A)):
            raise SpecificationError(f"{label} contains non-finite values", code="input.non_finite", stage="frontend", details={"role": label})
    names_c = tuple(exog_names or [f"exog{i}" for i in range(C.shape[1])])
    names_e = tuple(endog_names or [f"endog{i}" for i in range(E.shape[1])])
    names_i = tuple(instrument_names or [f"z{i}" for i in range(I.shape[1])])
    if len(names_c) != C.shape[1] or len(names_e) != E.shape[1] or len(names_i) != I.shape[1]:
        raise ShapeError("IV-PPML name lengths must match their matrices")
    return y, C, E, I, off, w, winfo.kind, names_c, names_e, names_i


@error_boundary("ivppml")
@managed_execution
def ivppmlhdfe(
    y, *, exog=None, endog=None, instruments=None, absorb=None,
    offset=None, exposure=None, weights=None, weight_type=None, vce="robust", clusters=None,
    exog_names=None, endog_names=None, instrument_names=None,
    config: IVPPMLConfig | None = None, inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None, warm_start=None,
):
    """Estimate additive-moment IV-PPML with optional high-dimensional FEs."""
    confidence_level = 0.95
    if inference_config is not None:
        inference_config.validate()
        confidence_level = inference_config.confidence_level
        if inference_config.vce is not None:
            vce = inference_config.vce
    execution_config = ExecutionConfig() if execution_config is None else execution_config
    execution_config.validate()
    context = ExecutionContext(execution_config)
    _t0 = perf_counter() if execution_config.profile != "off" else None
    config = IVPPMLConfig() if config is None else config
    config.validate()
    vals = _prepare(
        y, exog, endog, instruments, offset=offset, exposure=exposure, weights=weights,
        weight_type=weight_type, exog_names=exog_names, endog_names=endog_names, instrument_names=instrument_names,
    )
    yv, C, E, I, off, w, wkind, cn, en, zn = vals
    plan = FEPlan.from_arrays(absorb or ())
    if plan.groups and plan.nobs != len(yv):
        raise ShapeError("absorb arrays must have nobs rows", details={"expected": len(yv), "actual": plan.nobs})
    result = fit_arrays(
        yv, C, E, I, plan, offset=off, true_w=w, weight_kind=wkind, vce=vce, clusters=clusters,
        exog_names=cn, endog_names=en, instrument_names=zn, config=config,
        warm_start=warm_start, execution_config=execution_config, execution_context=context,
    )
    result.confidence_level = confidence_level
    result.diagnostics_mode = inference_config.diagnostics if inference_config is not None else "off"
    if _t0 is not None:
        result.profile = {
            "total_seconds": perf_counter() - _t0, "mode": execution_config.profile,
            "separation_seconds": dict(result.diagnostics.get("separation_seconds", {})),
        }
    return result


class IVPPMLHDFE:
    """Reusable DataFrame IV-PPML model with a compiled FE plan."""

    @error_boundary("ivppml")
    def __init__(self, data: pd.DataFrame, *, absorb=(), config: IVPPMLConfig | None = None,
                 inference_config: InferenceConfig | None = None,
                 execution_config: ExecutionConfig | None = None):
        self.data = data
        self.absorb = tuple(absorb)
        self.config = IVPPMLConfig() if config is None else config
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

    @error_boundary("ivppml")
    @managed_execution
    def fit(self, y: str, *, exog=(), endog=(), instruments=(), offset=None, exposure=None,
            weights=None, weight_type=None, vce="robust", clusters=None, warm_start=None):
        self._ensure_plan_current()
        with self.context.maintain(self.execution_config, protected=(self.plan, self.data)):
            def _spec_tuple(value):
                if value is None:
                    return ()
                if isinstance(value, str):
                    return (value,)
                if isinstance(value, (tuple, list)):
                    return tuple(value)
                return (value,)

            ccols, ecols, zcols = _spec_tuple(exog), _spec_tuple(endog), _spec_tuple(instruments)
            if self.inference_config.vce is not None:
                vce = self.inference_config.vce
            t0 = perf_counter() if self.execution_config.profile != "off" else None

            result = None
            if self.plan.groups and any(is_heterogeneous_spec_candidate(v) for v in (ccols, ecols, zcols)):
                result = fit_structured_dataframe(
                    self.data, y=y, exog=ccols, endog=ecols, instruments=zcols, plan=self.plan,
                    offset=offset, exposure=exposure, weights=weights, weight_type=weight_type,
                    vce=vce, clusters=clusters, config=self.config,
                    execution_config=self.execution_config, warm_start=warm_start,
                )

            if result is None:
                n = len(self.data)
                cdesign = build_design(self.data, ccols, n, prefix="exog", structural=True)
                edesign = build_design(self.data, ecols, n, prefix="endog", structural=True, protected_terms=cdesign.structural_terms)
                zdesign = build_design(self.data, zcols, n, prefix="z", structural=True, protected_terms=cdesign.structural_terms)
                vals = _prepare(
                    self.data[y].to_numpy(), cdesign.values, edesign.values, zdesign.values,
                    offset=None if offset is None else self.data[offset].to_numpy(),
                    exposure=None if exposure is None else self.data[exposure].to_numpy(),
                    weights=None if weights is None else self.data[weights].to_numpy(),
                    weight_type=weight_type, exog_names=cdesign.names, endog_names=edesign.names,
                    instrument_names=zdesign.names,
                )
                yv, C, E, I, off, w, wkind, cn, en, zn = vals
                cls = None
                if clusters is not None:
                    cvars = (clusters,) if isinstance(clusters, str) else tuple(clusters)
                    cls = [self.data[c].to_numpy() if isinstance(c, str) else np.asarray(c) for c in cvars]
                result = fit_arrays(
                    yv, C, E, I, self.plan, offset=off, true_w=w, weight_kind=wkind, vce=vce, clusters=cls,
                    exog_names=cn, endog_names=en, instrument_names=zn, config=self.config,
                    warm_start=warm_start, execution_config=self.execution_config, execution_context=self.context,
                )
            result.confidence_level = self.inference_config.confidence_level
            result.diagnostics_mode = self.inference_config.diagnostics
            if t0 is not None:
                result.profile = {
                    "total_seconds": perf_counter() - t0, "mode": self.execution_config.profile,
                    "separation_seconds": dict(result.diagnostics.get("separation_seconds", {})),
                }
            return result
