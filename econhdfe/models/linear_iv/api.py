from __future__ import annotations
from dataclasses import replace
from time import perf_counter
from ...errors import error_boundary
from ...config import HDFEConfig, InferenceConfig, ExecutionConfig
from ...planner.scheduler import managed_execution
from ...data import materialize_model_data
from ...frontend.validate import require_numeric
from ...frontend.roles import VariableRole
import numpy as np
from ...pipeline import (
    column_values, compile_design, compile_execution_design, prepare_group_mode, make_group_absorber,
    residualize_blocks, resolve_iv_columns, group_dof, group_info, group_has_intercept,
    check_collinearity_mode, resolve_method, resolve_acceleration, standard_dof,
    make_standard_absorber, check_absorption, cluster_arrays, masked_array, absorb_metadata,
    block_absorb_metadata, resolve_pool_size,
)
from ..linear import (
    IV_DESIGN_ROLES, StageClock, apply_config_overrides, reporting_options, prepare_standard_sample,
    finish_group_mode, recover_fixed_effects, linear_result, prediction_state, with_saved_effects,
)
from ...compute.weights import prepare_weights, resolve_vce_for_weights
from ...compute.vcov import normalize_vce, BLOCK_VCE_KINDS
from .estimators import fit_iv_kclass, fit_iv_gmm2s, fit_iv_kclass_block
from .absorption import refine_iv_control_absorption
from .stages import bind_callback, completed, validate_callback
from ...compute.block_design import BlockDesign, hstack_block_designs, align_dense_to_blocks, same_row_partition
from ...hdfe.block_projection import BlockWeightedFEProjector
from ...hdfe.plan import FEPlan
from ...design import is_heterogeneous_spec_candidate
from ...results import EstimationState
from ...prediction import design_state


def _split_within(within, widths, start):
    """Column blocks of a stacked within matrix, in ``widths`` order from ``start``."""
    out = []
    for width in widths:
        out.append(within[:, start:start + width])
        start += width
    return out


def _fe_target(y1, C1, E1, beta, resid):
    """y - xb - e on the estimation sample, the input of FE recovery."""
    kc = C1.shape[1]
    xb = np.zeros_like(y1, dtype=np.float64)
    if kc:
        xb += C1 @ beta[:kc]
    if E1.shape[1]:
        xb += E1 @ beta[kc:]
    return y1 - xb - resid


def _fit_dense_iv(est, yw, Cw, Ew, Zw, *, kappa, fuller, center, **common):
    """Dispatch a dense within-transformed IV system to GMM2S or the k-class family."""
    if est == "gmm2s":
        return fit_iv_gmm2s(yw, Cw, Ew, Zw, center=center, **common)
    return fit_iv_kclass(yw, Cw, Ew, Zw, estimator=est, kappa=kappa, fuller=fuller, **common)


def _ivreghdfe_group_mode(
    data=None, *, y, exog=None, endog, instruments, absorb, group, individual=None,
    aggregation="mean", weights=None, weight_type=None, cluster=None, vce=None,
    drop_singletons=True, tol=1e-8, max_iter=16_000, method="map",
    transform="symmetric", acceleration="none", preconditioner="diagonal",
    backend="numpy", pool_size="auto", memory_budget_mb=512, dof_method="pairwise",
    estimator="2sls", kappa=None, fuller=0.0, gmm_center=False, keep_state=False,
    time=None, panel=None, bandwidth=None, kernel="bartlett", save_fe=False,
    projection_backend="auto", absorb_threads="auto",
    incidence_backend="auto", collinearity="warn", collinear_tol=1e-10,
    omit_exog=None, omit_endog=None, omit_instruments=None,
    advanced_statistics=True, on_stage=None,
):
    method, _ = resolve_method(method, group_mode=True, individual=individual is not None, backend=backend)
    acceleration = resolve_acceleration(method, acceleration, group_mode=True)
    ylong = require_numeric(column_values(data, y), name=str(y), role=VariableRole.OUTCOME, ndim=1)
    cdesign = compile_design(data, exog, len(ylong), "exog", structural=False, omit=omit_exog)
    edesign = compile_design(data, endog, len(ylong), "endog", structural=False, omit=omit_endog)
    zdesign = compile_design(data, instruments, len(ylong), "z", structural=False, omit=omit_instruments)
    Clong, Elong, Zlong = cdesign.values, edesign.values, zdesign.values
    prep = prepare_group_mode(
        data, group=group, individual=individual, absorb=absorb,
        arrays=[("dependent variable", ylong), ("exogenous regressors", Clong),
                ("endogenous regressors", Elong), ("instruments", Zlong)],
        weights=weights, cluster=cluster, time=time, panel=panel,
        drop_singletons=drop_singletons, weight_type=weight_type, aggregation=aggregation,
    )
    y1, C1, E1, Z1 = prep["arrays"]
    absorber = make_group_absorber(
        prep, method=method, transform=transform, backend=backend, acceleration=acceleration,
        preconditioner=preconditioner, tol=tol, max_iter=max_iter, aggregation=aggregation,
        incidence_backend=incidence_backend, memory_budget_mb=memory_budget_mb,
        projection_backend=projection_backend, absorb_threads=absorb_threads,
    )
    block = np.column_stack([y1, C1, E1, Z1])
    within, info = residualize_blocks(absorber, block, pool_size, memory_budget_mb)
    yw = within[:, 0]
    Cw, Ew, Zw = _split_within(within, (C1.shape[1], E1.shape[1], Z1.shape[1]), 1)
    Cw, Ew, Zw, C1_active, E1_active, Z1_active, c_names, e_names, z_names, collin_info = resolve_iv_columns(
        Cw, Ew, Zw, C1, E1, Z1, cdesign.names, edesign.names, zdesign.names,
        weights=prep["weights"], tolerance=collinear_tol, mode=collinearity,
        c_design=cdesign, e_design=edesign, z_design=zdesign,
    )
    dof = group_dof(prep, absorber, dof_method)
    winfo = prep["winfo"]
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(prep["clusters"]))
    est = estimator.lower()
    beta, V, resid, rank, df_resid, first, diagnostics, meta = _fit_dense_iv(
        est, yw, Cw, Ew, Zw, kappa=kappa, fuller=fuller, center=gmm_center,
        weights=prep["weights"], weight_info=winfo, advanced_statistics=advanced_statistics,
        on_stage=bind_callback(on_stage, c_names + e_names, y, df_absorbed=dof.df_absorbed),
        vce=vce_kind, clusters=prep["clusters"], df_absorbed=dof.df_absorbed,
        nested_adj=int(dof.nested > 0), time=prep["time"], panel=prep["panel"],
        bandwidth=bandwidth, kernel=kernel,
    )
    fixed_effects = None
    if save_fe:
        fixed_effects = recover_fixed_effects(
            absorber, _fe_target(y1, C1_active, E1_active, beta, resid), prep["names"],
        )
    return linear_result(
        beta, V, resid, y=y1, y_within=yw, rank=rank, df_resid=df_resid, dof=dof, winfo=winfo,
        clusters=prep["clusters"], vce=vce_kind, has_intercept=group_has_intercept(prep), iv=True,
        converged=info.converged, iterations=info.iterations, dropped_singletons=prep["dropped"],
        names=tuple(c_names + e_names), first_stage=first, diagnostics=diagnostics,
        estimator=meta.get("estimator", est), estimator_info=meta, fixed_effects=fixed_effects,
        group_info=group_info(prep, absorber, aggregation), collinearity_info=collin_info,
        fe_names=tuple(prep["names"]),
        state=EstimationState(absorber, yw, np.column_stack([Cw, Ew]),
                              tuple(prep["clusters"] or ()), prep["weights"], winfo.kind)
        if keep_state else None,
    )

@error_boundary("linear_iv")
@managed_execution
def ivhdfe(
    data=None, *, y, exog=None, endog, instruments, absorb,
    weights=None, weight_type=None, cluster=None, vce=None, drop_singletons=True,
    tol=1e-8, max_iter=16_000, method="auto", transform="symmetric",
    acceleration="auto", preconditioner="diagonal", backend="numpy",
    pool_size="auto", memory_budget_mb="auto", projection_backend="auto", absorb_threads="auto",
    dof_method="pairwise", estimator="2sls",
    kappa=None, fuller=0.0, gmm_center=False, keep_state=False, time=None,
    panel=None, bandwidth=None, kernel="bartlett", save_fe=False,
    group=None, individual=None, aggregation="mean", incidence_backend="auto",
    canonicalize_fe=True, allow_nonconverged=False, collinearity="warn", collinear_tol=1e-10,
    structural_collinearity=True, omit_exog=None, omit_endog=None, omit_instruments=None,
    hdfe_config: HDFEConfig | None = None, inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    on_stage=None,
):
    """IV/LIML/GMM2S with high-dimensional fixed effects.

    ``estimator`` may be ``"2sls"`` (default), ``"liml"``, ``"kclass"`` or
    ``"gmm2s"``. Exogenous controls are automatically included in the
    instrument matrix, matching ivreg2/ivreghdfe convention.

    ``InferenceConfig.iv_advanced_statistics=False`` skips Shea, AP/SW, CD
    and Stock--Yogo while retaining ordinary first-stage tests, KP and over-ID.
    ``on_stage(event)`` receives synchronous independent snapshots in original
    units at estimation, first_stage, diagnostics and complete. Diagnostics
    may still fail after estimation; only complete denotes a finished fit.
    """
    clock = StageClock(execution_config)
    validate_callback(on_stage)
    (method, tol, max_iter, dof_method, canonicalize_fe, vce, memory_budget_mb,
     absorb_threads) = apply_config_overrides(
        hdfe_config, inference_config, execution_config, method=method, tol=tol,
        max_iter=max_iter, dof_method=dof_method, canonicalize_fe=canonicalize_fe, vce=vce,
        memory_budget_mb=memory_budget_mb, absorb_threads=absorb_threads,
    )
    advanced_statistics = True if inference_config is None else inference_config.iv_advanced_statistics
    if individual is not None and group is None:
        raise ValueError("individual= requires group=")
    identifier_levels = getattr(data, "identifier_levels", {})
    data, data_plan = materialize_model_data(
        data, memory_budget_mb=memory_budget_mb,
        y=y, exog=exog, endog=endog, instruments=instruments, absorb=absorb,
        weights=weights, cluster=cluster, time=time, panel=panel,
        group=group, individual=individual,
    )
    if data_plan is not None:
        identifier_levels = data_plan.identifier_levels

    collinearity = check_collinearity_mode(collinearity)
    if group is not None:
        result = _ivreghdfe_group_mode(
            data, y=y, exog=exog, endog=endog, instruments=instruments, absorb=absorb,
            group=group, individual=individual, aggregation=aggregation, weights=weights,
            weight_type=weight_type, cluster=cluster, vce=vce, drop_singletons=drop_singletons,
            tol=tol, max_iter=max_iter, method=method, transform=transform,
            acceleration=acceleration, preconditioner=preconditioner, backend=backend,
            pool_size=pool_size, memory_budget_mb=memory_budget_mb, dof_method=dof_method,
            estimator=estimator, kappa=kappa, fuller=fuller, gmm_center=gmm_center,
            keep_state=keep_state, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            save_fe=save_fe, incidence_backend=incidence_backend,
            projection_backend=projection_backend, absorb_threads=absorb_threads,
            collinearity=collinearity, collinear_tol=collinear_tol,
            omit_exog=omit_exog, omit_endog=omit_endog, omit_instruments=omit_instruments,
            advanced_statistics=advanced_statistics, on_stage=on_stage,
        )
        finish_group_mode(
            result, allow_nonconverged=allow_nonconverged, inference_config=inference_config,
            clock=clock, data_plan=data_plan,
        )
        completed(on_stage, result, outcome=y)
        return result

    y0, w0, wkind, fe, sample_state = prepare_standard_sample(
        data, y, weights, weight_type, absorb,
        canonicalize_fe=canonicalize_fe, drop_singletons=drop_singletons,
    )
    groups, slopes, intercepts, mask = fe.groups, fe.slopes, fe.intercepts, fe.mask
    method_resolved, solver_selection = resolve_method(
        method, groups, slopes, intercepts, backend=backend
    )
    acceleration_resolved = resolve_acceleration(method_resolved, acceleration)
    active_mask = None if fe.all_kept else mask
    y1 = fe.active(y0)
    absorbed_intercept_groups, absorbed_intercept_names = fe.absorbed_intercepts()
    hetero_candidate = any(is_heterogeneous_spec_candidate(s) for s in (exog, endog, instruments))
    block_solver_ok = (
        backend == "numpy" and all(s is None for s in slopes) and all(intercepts)
        and projection_backend == "auto" and transform == "symmetric"
        and preconditioner == "diagonal" and not keep_state and not save_fe
        and estimator.lower() != "gmm2s" and not gmm_center
        and normalize_vce(vce, "") in BLOCK_VCE_KINDS
        and (method_resolved == "twoway" or (method_resolved == "map" and acceleration_resolved == "cg"))
        and hetero_candidate
    )
    # Compile role-specific designs after FE canonicalization so factor and
    # interaction columns absorbed by the effective FE system need not be
    # materialized or residualized at all. Cross-role IV priority is still
    # resolved numerically below.
    compiler = compile_execution_design if block_solver_ok else compile_design
    common = dict(
        row_mask=active_mask, absorbed_groups=absorbed_intercept_groups,
        absorbed_names=absorbed_intercept_names, structural=bool(structural_collinearity),
    )
    if block_solver_ok:
        common.update(groups=groups, expected_passes=2, memory_budget_mb=memory_budget_mb)
    cdesign = compiler(data, exog, len(y0), "exog", omit=omit_exog, **common)
    edesign = compiler(
        data, endog, len(y0), "endog", protected_terms=cdesign.structural_terms,
        omit=omit_endog, **common,
    )
    zdesign = compiler(
        data, instruments, len(y0), "z", protected_terms=cdesign.structural_terms,
        omit=omit_instruments, **common,
    )
    C1, E1, Z1 = cdesign.values, edesign.values, zdesign.values
    winfo = prepare_weights(None if w0 is None else fe.active(w0), len(y1), wkind)
    w1 = winfo.estimation
    block_native = any(isinstance(A, BlockDesign) for A in (C1, E1, Z1))
    absorber = None
    block_projector = None
    block = None
    refinement_metadata = None
    refinement_seconds = 0.0
    refine_controls = None
    clock.lap("preparation_seconds")
    if block_native:
        native_roles = tuple(A for A in (C1, E1, Z1) if isinstance(A, BlockDesign))
        if not same_row_partition(*native_roles):
            C1 = C1.materialize() if isinstance(C1, BlockDesign) else C1
            E1 = E1.materialize() if isinstance(E1, BlockDesign) else E1
            Z1 = Z1.materialize() if isinstance(Z1, BlockDesign) else Z1
            block_native = False
        else:
            template = native_roles[0]
            C1 = C1 if isinstance(C1, BlockDesign) else align_dense_to_blocks(C1, template)
            E1 = E1 if isinstance(E1, BlockDesign) else align_dense_to_blocks(E1, template)
            Z1 = Z1 if isinstance(Z1, BlockDesign) else align_dense_to_blocks(Z1, template)
    if block_native:
        combined = hstack_block_designs(C1, E1, Z1)
        block_projector = BlockWeightedFEProjector.from_design(
            combined, FEPlan.from_arrays(groups),
            engine="optimized" if method_resolved == "twoway" else "replica",
            method="map", max_iter=max_iter, absorb_threads=absorb_threads,
            projection_memory_budget_mb=memory_budget_mb,
        )
        wproj = np.ones(len(y1), dtype=np.float64) if w1 is None else w1
        projected = block_projector.residualize_response_design(y1, combined, wproj, tol=tol)
        info = projected
        if not info.converged and not allow_nonconverged:
            raise RuntimeError("heterogeneous-spec IV FE projection did not converge")
        yw = projected.response
        p = 0; kc0 = C1.ncols; ke0 = E1.ncols; kz0 = Z1.ncols
        Cw = projected.design.select_columns(np.arange(p, p + kc0)); p += kc0
        Ew = projected.design.select_columns(np.arange(p, p + ke0)); p += ke0
        Zw = projected.design.select_columns(np.arange(p, p + kz0))
    else:
        absorber = make_standard_absorber(
            groups, slopes, intercepts, weights=w1, tol=tol, max_iter=max_iter,
            backend=backend, method=method_resolved, transform=transform, acceleration=acceleration_resolved,
            preconditioner=preconditioner, projection_backend=projection_backend,
            absorb_threads=absorb_threads, memory_budget_mb=memory_budget_mb,
        )
        block = np.column_stack([y1, C1, E1, Z1])
        within, info = residualize_blocks(
            absorber, block, pool_size, memory_budget_mb,
            adaptive_auto=(str(method).lower() == "auto" and str(acceleration).lower() == "auto"),
        )
        check_absorption(info, allow_nonconverged=allow_nonconverged)
        yw = within[:, 0]
        Cw, Ew, Zw = _split_within(within, (C1.shape[1], E1.shape[1], Z1.shape[1]), 1)
        if estimator.lower() in {"2sls", "iv"} and not fuller and kappa is None:
            def refine_controls(original, controls):
                nonlocal info, refinement_metadata, refinement_seconds
                started = perf_counter() if clock.enabled else None
                controls, refinement_info, refinement_metadata = refine_iv_control_absorption(
                    absorber, original, controls, weights=w1,
                )
                if started is not None:
                    refinement_seconds = perf_counter() - started
                if refinement_info is not None:
                    check_absorption(refinement_info, allow_nonconverged=allow_nonconverged)
                    info = replace(info, converged=info.converged and refinement_info.converged)
                return controls
    clusters = cluster_arrays(data, cluster, mask)
    clock.lap("absorption_seconds")
    time1 = masked_array(data, time, mask)
    panel1 = masked_array(data, panel, mask)
    Cw, Ew, Zw, C1_active, E1_active, Z1_active, c_names, e_names, z_names, collin_info = resolve_iv_columns(
        Cw, Ew, Zw, C1, E1, Z1, cdesign.names, edesign.names, zdesign.names,
        weights=w1, tolerance=collinear_tol, mode=collinearity,
        c_design=cdesign, e_design=edesign, z_design=zdesign,
        _refine_controls=refine_controls,
    )
    dof = standard_dof(fe, clusters, dof_method)
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(clusters))
    if winfo.kind == "fweight" and vce_kind in {"hac", "newey_west", "neweywest", "dkraay", "driscoll_kraay", "dk"}:
        raise ValueError(
            "fweights with HAC/kernel/Driscoll-Kraay covariance are not supported by "
            "ivreg2/ivreghdfe semantics; use robust/cluster VCE or another weight type"
        )
    est = estimator.lower()
    clock.credit("control_refinement_seconds", refinement_seconds)
    clock.lap("columns_and_dof_seconds")
    stage_callback = bind_callback(on_stage, c_names + e_names, y, df_absorbed=dof.df_absorbed)
    if isinstance(Cw, BlockDesign):
        beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_kclass_block(
            yw, Cw, Ew, Zw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            advanced_statistics=advanced_statistics, on_stage=stage_callback,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            estimator=est, kappa=kappa, fuller=fuller,
        )
    else:
        beta, V, resid, rank, df_resid, first, diagnostics, meta = _fit_dense_iv(
            est, yw, Cw, Ew, Zw, kappa=kappa, fuller=fuller, center=gmm_center,
            weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            advanced_statistics=advanced_statistics, on_stage=stage_callback,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            time=time1, panel=panel1, bandwidth=bandwidth, kernel=kernel,
        )
    clock.lap("core_seconds")
    fixed_effects = None
    if save_fe:
        fixed_effects = recover_fixed_effects(
            absorber, _fe_target(y1, C1_active, E1_active, beta, resid), fe.names,
        )
    names = tuple(c_names + e_names)
    if isinstance(Cw, BlockDesign):
        absorb_info = block_absorb_metadata(
            fe.plan, block_projector, info, method=method_resolved, transform=transform,
            acceleration=acceleration_resolved, pool_size=Cw.ncols + Ew.ncols + Zw.ncols + 1,
            solver_selection=solver_selection,
            heterogeneous_spec={"storage": {
                "exog": cdesign.storage_plan.as_dict(),
                "endog": edesign.storage_plan.as_dict(),
                "instruments": zdesign.storage_plan.as_dict(),
            }},
        )
    else:
        absorb_info = absorb_metadata(
            fe.plan, absorber, info,
            resolve_pool_size(absorber, block.shape[1], pool_size, memory_budget_mb),
            solver_selection=solver_selection,
        )
        if refinement_metadata is not None and refinement_metadata["weak_directions"]:
            absorb_info["control_refinement"] = refinement_metadata
    prediction = prediction_state(
        meta.get("estimator", est), names, ("exogenous", "endogenous"),
        tuple(
            design_state(design, role=role, active_indices=collin_info[key]["active_indices"],
                         identifier_levels=identifier_levels)
            for design, (role, key) in zip((cdesign, edesign, zdesign), IV_DESIGN_ROLES)
        ),
        fe, sample_state=sample_state, recovered_effects=fixed_effects,
    )
    if save_fe:
        prediction = with_saved_effects(
            prediction, fe, data=data, recovered=fixed_effects, beta=beta,
            identifier_levels=identifier_levels,
        )
    result = linear_result(
        beta, V, resid, y=y1, y_within=yw, rank=rank, df_resid=df_resid, dof=dof, winfo=winfo,
        clusters=clusters, vce=vce_kind, has_intercept=bool(any(fe.inference.intercepts)), iv=True,
        converged=info.converged, iterations=info.iterations, dropped_singletons=fe.dropped,
        names=names, first_stage=first, diagnostics=diagnostics,
        estimator=meta.get("estimator", est), estimator_info=meta, fixed_effects=fixed_effects,
        absorb_info=absorb_info, collinearity_info=collin_info, fe_names=tuple(fe.names),
        state=EstimationState(
            absorber, yw, np.column_stack([Cw, Ew]), tuple(clusters or ()), w1, winfo.kind
        ) if keep_state else None,
        prediction_state=prediction, **reporting_options(inference_config),
    )
    if clock.enabled:
        result.profile = clock.summary()
        if data_plan is not None:
            result.profile["data_ingestion"] = data_plan.as_dict()
        result.profile["sample"] = sample_state.summary()
        result.profile["iv_stages"] = clock.stages
    completed(on_stage, result, outcome=y)
    return result


# Backward-compatible estimator name.
ivreghdfe = ivhdfe
