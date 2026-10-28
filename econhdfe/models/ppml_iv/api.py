from __future__ import annotations
from time import perf_counter
from ...errors import error_boundary, SpecificationError, UnderidentifiedError, ShapeError
from ...config import InferenceConfig, ExecutionConfig
from ...planner.scheduler import managed_execution
from ...frontend.validate import require_numeric
from ...frontend.roles import VariableRole
import numpy as np
import pandas as pd
from ...compute.wls import as_2d
from ...hdfe.plan import FEPlan
from ..poisson import prepare_poisson_base, prepare_ppml_weights
from ..ppml.api import (
    _ReusableFEModel, _finish_result, _require_frame, _resolve_call_configs, _spec_tuple,
    resolve_ppml_vce,
)
from .config import IVPPMLConfig
from .estimator import fit_arrays
from .heterogeneous import fit_structured_dataframe
from ...design import build_design, is_heterogeneous_spec_candidate


def _prepare(y, exog, endog, instruments, *, offset, exposure, weights, weight_type, exog_names, endog_names, instrument_names):
    y, off, _ = prepare_poisson_base(
        y, offset=offset, exposure=exposure, weights=weights
    )
    n = len(y)
    w, weight_kind = prepare_ppml_weights(weights, n, weight_type, family="ivppml")
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
    return y, C, E, I, off, w, weight_kind, names_c, names_e, names_i


@error_boundary("ivppml")
def ivppmlhdfe(
    data=None, *, y, exog=None, endog=None, instruments=None, absorb=None,
    offset=None, exposure=None, weights=None, weight_type=None, vce="robust", cluster=None,
    exog_names=None, endog_names=None, instrument_names=None,
    config: IVPPMLConfig | None = None, inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None, warm_start=None,
):
    """Estimate additive-moment IV-PPML with optional high-dimensional FEs.

    With a DataFrame ``data``, ``y``/``exog``/``endog``/``instruments``/
    ``absorb``/``offset``/``exposure``/``weights``/``cluster`` are column
    specifications (roles may contain ``factor()``/interaction terms) and the
    call is exactly ``IVPPMLHDFE(data, absorb=...).fit(y, ...)``.  With
    ``data=None`` they are arrays; ``absorb``/``cluster`` are sequences of
    1-D arrays and ``*_names`` label the role matrices.

    ``vce`` defaults to ``"robust"`` (Stata ivppmlhdfe's default), unlike
    OLS/IV whose ``None`` default is reghdfe's unadjusted VCE.  A non-default
    ``vce`` and a different ``inference_config.vce`` raise
    ``specification.config_conflict``; either one alone takes effect.
    """
    if data is None:
        return _ivppmlhdfe_arrays(
            y, exog=exog, endog=endog, instruments=instruments, absorb=absorb,
            offset=offset, exposure=exposure, weights=weights, weight_type=weight_type,
            vce=vce, cluster=cluster, exog_names=exog_names, endog_names=endog_names,
            instrument_names=instrument_names, config=config,
            inference_config=inference_config, execution_config=execution_config,
            warm_start=warm_start,
        )
    _require_frame(
        data, "ivppmlhdfe", exog_names=exog_names, endog_names=endog_names,
        instrument_names=instrument_names,
    )
    model = IVPPMLHDFE(
        data, absorb=_spec_tuple(absorb), config=config,
        inference_config=inference_config, execution_config=execution_config,
    )
    return model.fit(
        y, exog=exog, endog=endog, instruments=instruments, offset=offset,
        exposure=exposure, weights=weights, weight_type=weight_type, vce=vce,
        cluster=cluster, warm_start=warm_start,
    )


@managed_execution
def _ivppmlhdfe_arrays(
    y, *, exog, endog, instruments, absorb, offset, exposure, weights, weight_type, vce,
    cluster, exog_names, endog_names, instrument_names, config, inference_config,
    execution_config, warm_start,
):
    vce, config, execution_config, context, t0 = _resolve_call_configs(
        vce, config, IVPPMLConfig, inference_config, execution_config,
    )
    vals = _prepare(
        y, exog, endog, instruments, offset=offset, exposure=exposure, weights=weights,
        weight_type=weight_type, exog_names=exog_names, endog_names=endog_names, instrument_names=instrument_names,
    )
    yv, C, E, I, off, w, wkind, cn, en, zn = vals
    plan = FEPlan.from_arrays(absorb or ())
    if plan.groups and plan.nobs != len(yv):
        raise ShapeError("absorb arrays must have nobs rows", details={"expected": len(yv), "actual": plan.nobs})
    result = fit_arrays(
        yv, C, E, I, plan, offset=off, true_w=w, weight_kind=wkind, vce=vce, clusters=cluster,
        exog_names=cn, endog_names=en, instrument_names=zn, config=config,
        warm_start=warm_start, execution_config=execution_config, execution_context=context,
    )
    return _finish_result(result, inference_config, execution_config, t0)


class IVPPMLHDFE(_ReusableFEModel):
    """Reusable DataFrame IV-PPML model with a compiled FE plan."""

    @error_boundary("ivppml")
    def __init__(self, data: pd.DataFrame, *, absorb=(), config: IVPPMLConfig | None = None,
                 inference_config: InferenceConfig | None = None,
                 execution_config: ExecutionConfig | None = None):
        self._setup(data, absorb, config, IVPPMLConfig, inference_config, execution_config)

    @error_boundary("ivppml")
    @managed_execution
    def fit(self, y: str, *, exog=(), endog=(), instruments=(), offset=None, exposure=None,
            weights=None, weight_type=None, vce="robust", cluster=None, warm_start=None):
        """Fit one IV-PPML specification on the model's DataFrame.

        ``vce`` defaults to ``"robust"`` (ivppmlhdfe's default); it conflicts
        with a different non-default ``inference_config.vce``.
        """
        self._ensure_plan_current()
        vce = resolve_ppml_vce(vce, self.inference_config)
        with self.context.maintain(self.execution_config, protected=(self.plan, self.data)):
            ccols, ecols, zcols = _spec_tuple(exog), _spec_tuple(endog), _spec_tuple(instruments)
            t0 = perf_counter() if self.execution_config.profile != "off" else None

            result = None
            if self.plan.groups and any(is_heterogeneous_spec_candidate(v) for v in (ccols, ecols, zcols)):
                result = fit_structured_dataframe(
                    self.data, y=y, exog=ccols, endog=ecols, instruments=zcols, plan=self.plan,
                    offset=offset, exposure=exposure, weights=weights, weight_type=weight_type,
                    vce=vce, clusters=cluster, config=self.config,
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
                if cluster is not None:
                    cvars = (cluster,) if isinstance(cluster, str) else tuple(cluster)
                    cls = [self.data[c].to_numpy() if isinstance(c, str) else np.asarray(c) for c in cvars]
                result = fit_arrays(
                    yv, C, E, I, self.plan, offset=off, true_w=w, weight_kind=wkind, vce=vce, clusters=cls,
                    exog_names=cn, endog_names=en, instrument_names=zn, config=self.config,
                    warm_start=warm_start, execution_config=self.execution_config, execution_context=self.context,
                )
            return _finish_result(result, self.inference_config, self.execution_config, t0)
