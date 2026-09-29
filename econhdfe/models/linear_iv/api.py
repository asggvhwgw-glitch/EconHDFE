from __future__ import annotations
from time import perf_counter
from ...errors import error_boundary
from ...config import HDFEConfig, InferenceConfig, ExecutionConfig
from ...data import materialize_model_data, EstimationSampleState
from ...frontend.validate import require_numeric
from ...frontend.roles import VariableRole
import numpy as np
from ...pipeline import (
    _col, _design, _execution_design, _prepare_group_mode, _make_group_absorber, _residualize_blocks,
    _resolve_iv_columns, _group_dof, _group_info, _collinearity_metadata,
    _check_collinearity_mode, _resolve_method, _resolve_acceleration,
    _prepare_standard_fe_structure, _make_standard_absorber, _check_absorption,
    _cluster_arrays, _masked_array, _absorb_metadata, _resolve_pool_size,
)
from ...hdfe.dof import absorbed_dof
from ...compute.weights import normalize_weight_type, prepare_weights, resolve_vce_for_weights
from .estimators import fit_iv_kclass, fit_iv_gmm2s, fit_iv_kclass_block
from ...compute.block_design import BlockDesign, hstack_block_designs, align_dense_to_blocks, same_row_partition
from ...hdfe.block_projection import BlockWeightedFEProjector
from ...hdfe.plan import FEPlan
from ...design import is_heterogeneous_spec_candidate
from ...results import RegressionResult, EstimationState, FixedEffectEstimates, FixedEffectTermEstimate
from ...prediction import design_state, linear_prediction_state
from ...reporting import (
    cluster_counts as _cluster_counts, reghdfe_r2_statistics, ivreghdfe_model_statistics,
)

def _ivreghdfe_group_mode(
    data=None, *, y, exog=None, endog, instruments, absorb, group, individual=None,
    aggregation="mean", weights=None, weight_type=None, cluster=None, vce=None,
    drop_singletons=True, tol=1e-8, max_iter=16_000, method="map",
    transform="symmetric", acceleration="none", preconditioner="diagonal",
    backend="numpy", pool_size="auto", memory_budget_mb=512, dof_method="pairwise",
    estimator="2sls", kappa=None, fuller=0.0, gmm_center=False, keep_state=False,
    time=None, panel=None, bandwidth=None, kernel="bartlett", save_fe=False,
    incidence_backend="auto", collinearity="warn", collinear_tol=1e-10,
    omit_exog=None, omit_endog=None, omit_instruments=None,
):
    ylong = require_numeric(_col(data, y), name=str(y), role=VariableRole.OUTCOME, ndim=1)
    cdesign = _design(data, exog, len(ylong), "exog", structural=False, omit=omit_exog)
    edesign = _design(data, endog, len(ylong), "endog", structural=False, omit=omit_endog)
    zdesign = _design(data, instruments, len(ylong), "z", structural=False, omit=omit_instruments)
    Clong, Elong, Zlong = cdesign.values, edesign.values, zdesign.values
    prep = _prepare_group_mode(
        data, group=group, individual=individual, absorb=absorb,
        arrays=[("dependent variable", ylong), ("exogenous regressors", Clong),
                ("endogenous regressors", Elong), ("instruments", Zlong)],
        weights=weights, cluster=cluster, time=time, panel=panel,
        drop_singletons=drop_singletons, weight_type=weight_type, aggregation=aggregation,
    )
    y1, C1, E1, Z1 = prep["arrays"]
    absorber = _make_group_absorber(
        prep, method=method, transform=transform, backend=backend, acceleration=acceleration,
        preconditioner=preconditioner, tol=tol, max_iter=max_iter, aggregation=aggregation,
        incidence_backend=incidence_backend, memory_budget_mb=memory_budget_mb,
    )
    block = np.column_stack([y1, C1, E1, Z1])
    within, info = _residualize_blocks(absorber, block, pool_size, memory_budget_mb)
    p = 1
    yw = within[:, 0]
    Cw = within[:, p:p+C1.shape[1]]; p += C1.shape[1]
    Ew = within[:, p:p+E1.shape[1]]; p += E1.shape[1]
    Zw = within[:, p:p+Z1.shape[1]]
    Cw, Ew, Zw, C1_active, E1_active, Z1_active, c_names, e_names, z_names, collin_info = _resolve_iv_columns(
        Cw, Ew, Zw, C1, E1, Z1, cdesign.names, edesign.names, zdesign.names,
        weights=prep["weights"], tolerance=collinear_tol, mode=collinearity,
        c_origins=cdesign.origins, e_origins=edesign.origins, z_origins=zdesign.origins,
        c_design=cdesign, e_design=edesign, z_design=zdesign,
    )
    dof = _group_dof(prep, absorber, dof_method)
    winfo = prep["winfo"]
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(prep["clusters"]))
    est = estimator.lower()
    if est == "gmm2s":
        beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_gmm2s(
            yw, Cw, Ew, Zw, weights=prep["weights"], weight_info=winfo,
            vce=vce_kind, clusters=prep["clusters"], df_absorbed=dof.df_absorbed,
            nested_adj=int(dof.nested > 0), center=gmm_center, time=prep["time"],
            panel=prep["panel"], bandwidth=bandwidth, kernel=kernel,
        )
    else:
        beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_kclass(
            yw, Cw, Ew, Zw, weights=prep["weights"], weight_info=winfo,
            vce=vce_kind, clusters=prep["clusters"], df_absorbed=dof.df_absorbed,
            nested_adj=int(dof.nested > 0), estimator=est, kappa=kappa, fuller=fuller,
            time=prep["time"], panel=prep["panel"], bandwidth=bandwidth, kernel=kernel,
        )
    fixed_effects = None
    if save_fe:
        kc = C1_active.shape[1]
        xb = np.zeros_like(y1, dtype=np.float64)
        if kc:
            xb += C1_active @ beta[:kc]
        if E1_active.shape[1]:
            xb += E1_active @ beta[kc:]
        target = y1 - xb - resid
        raw_terms, fe_meta = absorber.solve_effects(target)
        fixed_effects = FixedEffectEstimates(
            terms=tuple(
                FixedEffectTermEstimate(prep["names"][i], intercept, slope_coef, contribution)
                for i, (intercept, slope_coef, contribution) in enumerate(raw_terms)
            ),
            converged=bool(fe_meta["converged"]), iterations=int(fe_meta["iterations"]),
            residual_norm=float(fe_meta["residual_norm"]),
            reconstruction_error=float(fe_meta["reconstruction_error"]),
        )
    has_intercept = bool(any(prep["intercepts"])) or bool(
        prep.get("individual_spec") is not None and prep["individual_spec"].intercept
    )
    r2_stats = reghdfe_r2_statistics(
        y1, yw, resid, weights=prep["weights"], effective_n=winfo.effective_n,
        rank=rank, df_absorbed=dof.df_absorbed, df_nested=dof.nested,
        has_intercept=has_intercept,
    )
    fit_stats = ivreghdfe_model_statistics(
        yw, resid, beta, V, weights=prep["weights"], effective_n=winfo.effective_n,
        rank=rank, df_absorbed=dof.df_absorbed, df_nested=dof.nested,
        df_resid_inference=df_resid,
    )
    return RegressionResult(
        params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
        residuals=resid, fitted=y1-resid, nobs=int(round(winfo.effective_n)), rank=rank,
        df_resid=df_resid, df_absorbed=dof.df_absorbed, converged=info.converged,
        iterations=info.iterations, dropped_singletons=prep["dropped"],
        names=tuple(c_names + e_names), first_stage=first,
        diagnostics=diagnostics, estimator=meta.get("estimator", est), estimator_info=meta,
        dof_info=dof, weight_type=winfo.kind, sum_weights=winfo.sum_weights, nobs_raw=len(yw),
        fixed_effects=fixed_effects, group_info=_group_info(prep, absorber, aggregation),
        collinearity_info=collin_info,
        vce=vce_kind, cluster_counts=_cluster_counts(prep["clusters"]), fe_names=tuple(prep["names"]),
        r2=r2_stats["r2"], r2_within=r2_stats["r2_within"],
        r2_adjusted=r2_stats["r2_adjusted"], r2_adjusted_within=r2_stats["r2_adjusted_within"],
        rss=fit_stats["rss"], tss_within=fit_stats["tss_within"], rmse=fit_stats["rmse"],
        f_statistic=fit_stats["f_statistic"], f_pvalue=fit_stats["f_pvalue"],
        df_model=fit_stats["df_model"], df_resid_fit=fit_stats["df_resid_fit"], vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        state=EstimationState(absorber, yw, np.column_stack([Cw, Ew]),
                              tuple(prep["clusters"] or ()), prep["weights"], winfo.kind)
        if keep_state else None,
    )

@error_boundary("linear_iv")
def ivhdfe(
    data=None, *, y, exog=None, endog, instruments, absorb,
    weights=None, weight_type=None, cluster=None, vce=None, drop_singletons=True,
    tol=1e-8, max_iter=16_000, method="auto", transform="symmetric",
    acceleration="auto", preconditioner="diagonal", backend="numpy",
    pool_size="auto", memory_budget_mb=512, projection_backend="auto", absorb_threads="auto",
    dof_method="pairwise", estimator="2sls",
    kappa=None, fuller=0.0, gmm_center=False, keep_state=False, time=None,
    panel=None, bandwidth=None, kernel="bartlett", save_fe=False,
    group=None, individual=None, aggregation="mean", incidence_backend="auto",
    canonicalize_fe=True, allow_nonconverged=False, collinearity="warn", collinear_tol=1e-10,
    structural_collinearity=True, omit_exog=None, omit_endog=None, omit_instruments=None,
    hdfe_config: HDFEConfig | None = None, inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None,
):
    """IV/LIML/GMM2S with high-dimensional fixed effects.

    ``estimator`` may be ``"2sls"`` (default), ``"liml"``, ``"kclass"`` or
    ``"gmm2s"``. Exogenous controls are automatically included in the
    instrument matrix, matching ivreg2/ivreghdfe convention.
    """
    _t0 = perf_counter() if execution_config is not None and execution_config.profile != "off" else None
    confidence_level = 0.95
    if hdfe_config is not None:
        hdfe_config.validate()
        method, tol, max_iter = hdfe_config.solver, hdfe_config.tolerance, hdfe_config.max_iter
        dof_method, canonicalize_fe = hdfe_config.dof_method, hdfe_config.canonicalize
    if inference_config is not None:
        inference_config.validate()
        confidence_level = inference_config.confidence_level
        if inference_config.vce is not None:
            vce = inference_config.vce
    if execution_config is not None:
        execution_config.validate()
        memory_budget_mb = execution_config.memory_budget_mb
        if execution_config.threads != "auto":
            absorb_threads = execution_config.threads
    if individual is not None and group is None:
        raise ValueError("individual= requires group=")
    data, data_plan = materialize_model_data(
        data, memory_budget_mb=memory_budget_mb,
        y=y, exog=exog, endog=endog, instruments=instruments, absorb=absorb,
        weights=weights, cluster=cluster, time=time, panel=panel,
        group=group, individual=individual,
    )
    collinearity = _check_collinearity_mode(collinearity)
    if group is not None:
        method_resolved, solver_selection = _resolve_method(
            method, group_mode=True, individual=individual is not None, backend=backend
        )
        acceleration_resolved = _resolve_acceleration(method_resolved, acceleration, group_mode=True)
        result = _ivreghdfe_group_mode(
            data, y=y, exog=exog, endog=endog, instruments=instruments, absorb=absorb,
            group=group, individual=individual, aggregation=aggregation, weights=weights,
            weight_type=weight_type, cluster=cluster, vce=vce, drop_singletons=drop_singletons,
            tol=tol, max_iter=max_iter, method=method_resolved, transform=transform,
            acceleration=acceleration_resolved, preconditioner=preconditioner, backend=backend,
            pool_size=pool_size, memory_budget_mb=memory_budget_mb, dof_method=dof_method,
            estimator=estimator, kappa=kappa, fuller=fuller, gmm_center=gmm_center,
            keep_state=keep_state, time=time, panel=panel, bandwidth=bandwidth, kernel=kernel,
            save_fe=save_fe, incidence_backend=incidence_backend,
            collinearity=collinearity, collinear_tol=collinear_tol,
            omit_exog=omit_exog, omit_endog=omit_endog, omit_instruments=omit_instruments,
        )
        if not result.converged and not allow_nonconverged:
            raise RuntimeError("group/individual FE absorption did not converge")
        result.confidence_level = confidence_level
        result.diagnostics_mode = inference_config.diagnostics if inference_config is not None else "off"
        if _t0 is not None:
            result.profile = {"total_seconds": perf_counter() - _t0, "mode": execution_config.profile}
            if data_plan is not None:
                result.profile["data_ingestion"] = data_plan.as_dict()
        return result

    y0 = require_numeric(_col(data, y), name=str(y), role=VariableRole.OUTCOME, ndim=1)
    sample_state = EstimationSampleState(len(y0))
    w0 = None if weights is None else require_numeric(_col(data, weights), name=str(weights), role=VariableRole.WEIGHT, ndim=1)
    wkind = normalize_weight_type(weight_type, has_weights=w0 is not None)
    _ = prepare_weights(w0, len(y0), wkind)  # validate before singleton pruning
    groups, slopes, intercepts, fe_names, fe_plan, mask, dropped, inference_fe = _prepare_standard_fe_structure(
        data, absorb, canonicalize_fe=canonicalize_fe, drop_singletons=drop_singletons,
        weights=w0, weight_kind=wkind,
    )
    if dropped:
        sample_state.keep(mask, stage="hdfe", reason="singleton_pruning")
    method_resolved, solver_selection = _resolve_method(
        method, groups, slopes, intercepts, backend=backend
    )
    acceleration_resolved = _resolve_acceleration(method_resolved, acceleration)
    all_kept = dropped == 0 and bool(np.all(mask))
    active_mask = None if all_kept else mask
    y1 = y0 if all_kept else y0[mask]
    absorbed_intercept_groups = [g for g, has_i in zip(groups, intercepts, strict=False) if has_i]
    absorbed_intercept_names = [n for n, has_i in zip(fe_names, intercepts, strict=False) if has_i]
    vce_requested = "" if vce is None else str(vce).lower().replace("-", "_")
    hetero_candidate = any(is_heterogeneous_spec_candidate(s) for s in (exog, endog, instruments))
    block_solver_ok = (
        backend == "numpy" and all(s is None for s in slopes) and all(intercepts)
        and projection_backend == "auto" and transform == "symmetric"
        and preconditioner == "diagonal" and not keep_state and not save_fe
        and estimator.lower() != "gmm2s" and not gmm_center
        and vce_requested in {"", "iid", "unadjusted", "homoskedastic", "robust", "hc1", "heteroskedastic", "cluster"}
        and (method_resolved == "twoway" or (method_resolved == "map" and acceleration_resolved == "cg"))
        and hetero_candidate
    )
    # Compile role-specific designs after FE canonicalization so factor and
    # interaction columns absorbed by the effective FE system need not be
    # materialized or residualized at all. Cross-role IV priority is still
    # resolved numerically below.
    compiler = _execution_design if block_solver_ok else _design
    common_exec = dict(groups=groups, expected_passes=2, memory_budget_mb=memory_budget_mb) if block_solver_ok else {}
    cdesign = compiler(
        data, exog, len(y0), "exog", row_mask=active_mask,
        absorbed_groups=absorbed_intercept_groups, absorbed_names=absorbed_intercept_names,
        structural=bool(structural_collinearity), omit=omit_exog, **common_exec,
    )
    edesign = compiler(
        data, endog, len(y0), "endog", row_mask=active_mask,
        absorbed_groups=absorbed_intercept_groups, absorbed_names=absorbed_intercept_names,
        structural=bool(structural_collinearity), protected_terms=cdesign.structural_terms,
        omit=omit_endog, **common_exec,
    )
    zdesign = compiler(
        data, instruments, len(y0), "z", row_mask=active_mask,
        absorbed_groups=absorbed_intercept_groups, absorbed_names=absorbed_intercept_names,
        structural=bool(structural_collinearity), protected_terms=cdesign.structural_terms,
        omit=omit_instruments, **common_exec,
    )
    C1, E1, Z1 = cdesign.values, edesign.values, zdesign.values
    w_active = w0 if (w0 is not None and all_kept) else (None if w0 is None else w0[mask])
    winfo = prepare_weights(w_active, len(y1), wkind)
    w1 = winfo.estimation
    block_native = any(isinstance(A, BlockDesign) for A in (C1, E1, Z1))
    absorber = None
    block_projector = None
    block = None
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
        fe_exec_plan = FEPlan.from_arrays(groups)
        block_projector = BlockWeightedFEProjector.from_design(
            combined, fe_exec_plan,
            engine="optimized" if method_resolved == "twoway" else "replica",
            method="map", absorb_threads=absorb_threads,
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
        absorber = _make_standard_absorber(
            groups, slopes, intercepts, weights=w1, tol=tol, max_iter=max_iter,
            backend=backend, method=method_resolved, transform=transform, acceleration=acceleration_resolved,
            preconditioner=preconditioner, projection_backend=projection_backend,
            absorb_threads=absorb_threads, memory_budget_mb=memory_budget_mb,
        )
        block = np.column_stack([y1, C1, E1, Z1])
        within, info = _residualize_blocks(
            absorber, block, pool_size, memory_budget_mb,
            adaptive_auto=(str(method).lower() == "auto" and str(acceleration).lower() == "auto"),
        )
        _check_absorption(info, allow_nonconverged=allow_nonconverged)
        p = 1
        yw = within[:, 0]
        Cw = within[:, p:p+C1.shape[1]]; p += C1.shape[1]
        Ew = within[:, p:p+E1.shape[1]]; p += E1.shape[1]
        Zw = within[:, p:p+Z1.shape[1]]
    clusters = _cluster_arrays(data, cluster, mask)
    time1 = _masked_array(data, time, mask)
    panel1 = _masked_array(data, panel, mask)
    Cw, Ew, Zw, C1_active, E1_active, Z1_active, c_names, e_names, z_names, collin_info = _resolve_iv_columns(
        Cw, Ew, Zw, C1, E1, Z1, cdesign.names, edesign.names, zdesign.names,
        weights=w1, tolerance=collinear_tol, mode=collinearity,
        c_origins=cdesign.origins, e_origins=edesign.origins, z_origins=zdesign.origins,
        c_design=cdesign, e_design=edesign, z_design=zdesign,
    )
    dof = absorbed_dof(
        inference_fe.groups, slopes=inference_fe.slopes, intercepts=inference_fe.intercepts,
        clusters=clusters, method=dof_method, adjust_nested=bool(clusters),
        adjust_continuous=True, groups_are_dense=True,
    )
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(clusters))
    if winfo.kind == "fweight" and vce_kind in {"hac", "newey_west", "neweywest", "dkraay", "driscoll_kraay", "dk"}:
        raise ValueError(
            "fweights with HAC/kernel/Driscoll-Kraay covariance are not supported by "
            "ivreg2/ivreghdfe semantics; use robust/cluster VCE or another weight type"
        )
    est = estimator.lower()
    if isinstance(Cw, BlockDesign):
        beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_kclass_block(
            yw, Cw, Ew, Zw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            estimator=est, kappa=kappa, fuller=fuller,
        )
    elif est == "gmm2s":
        beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_gmm2s(
            yw, Cw, Ew, Zw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            center=gmm_center, time=time1, panel=panel1, bandwidth=bandwidth, kernel=kernel,
        )
    else:
        beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_kclass(
            yw, Cw, Ew, Zw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            estimator=est, kappa=kappa, fuller=fuller, time=time1, panel=panel1,
            bandwidth=bandwidth, kernel=kernel,
        )
    fitted = y1 - resid
    fixed_effects = None
    if save_fe:
        kc = C1_active.shape[1]
        xb = np.zeros_like(y1, dtype=np.float64)
        if kc:
            xb += C1_active @ beta[:kc]
        if E1_active.shape[1]:
            xb += E1_active @ beta[kc:]
        target = y1 - xb - resid
        raw_terms, fe_meta = absorber.solve_effects(target)
        fixed_effects = FixedEffectEstimates(
            terms=tuple(
                FixedEffectTermEstimate(fe_names[i], intercept, slope_coef, contribution)
                for i, (intercept, slope_coef, contribution) in enumerate(raw_terms)
            ),
            converged=bool(fe_meta["converged"]),
            iterations=int(fe_meta["iterations"]),
            residual_norm=float(fe_meta["residual_norm"]),
            reconstruction_error=float(fe_meta["reconstruction_error"]),
        )
    names = tuple(c_names + e_names)
    r2_stats = reghdfe_r2_statistics(
        y1, yw, resid, weights=w1, effective_n=winfo.effective_n,
        rank=rank, df_absorbed=dof.df_absorbed, df_nested=dof.nested,
        has_intercept=bool(any(inference_fe.intercepts)),
    )
    fit_stats = ivreghdfe_model_statistics(
        yw, resid, beta, V, weights=w1, effective_n=winfo.effective_n,
        rank=rank, df_absorbed=dof.df_absorbed, df_nested=dof.nested,
        df_resid_inference=df_resid,
    )
    if isinstance(Cw, BlockDesign):
        resources = block_projector.resource_info if block_projector is not None else {"blocks": ()}
        actual_threads = [
            int(item["projector"].get("actual_threads") or 1)
            for item in resources.get("blocks", ()) if item.get("projector") is not None
        ]
        storage = {
            "exog": cdesign.storage_plan.as_dict(),
            "endog": edesign.storage_plan.as_dict(),
            "instruments": zdesign.storage_plan.as_dict(),
        }
        absorb_info = {
            "canonicalization": fe_plan.as_dict() if fe_plan is not None else None,
            "method": method_resolved,
            "transform": transform,
            "acceleration": acceleration_resolved,
            "projection_backend": "heterogeneous_components",
            "absorb_threads": max(actual_threads, default=1),
            "iterations": int(info.iterations),
            "converged": bool(info.converged),
            "convergence_metric": float(info.max_update),
            "convergence_criterion": str(info.criterion),
            "pool_size": int(Cw.ncols + Ew.ncols + Zw.ncols + 1),
            "solver_selection": dict(solver_selection),
            "heterogeneous_spec": {"storage": storage},
        }
    else:
        absorb_info = _absorb_metadata(
            fe_plan, absorber, info,
            _resolve_pool_size(absorber, block.shape[1], pool_size, memory_budget_mb),
            solver_selection=solver_selection,
        )
    prediction_state = linear_prediction_state(
        estimator=meta.get("estimator", est),
        coefficient_names=names,
        coefficient_roles=("exogenous", "endogenous"),
        designs=(
            design_state(
                cdesign, role="exogenous",
                active_indices=collin_info["exogenous"]["active_indices"],
            ),
            design_state(
                edesign, role="endogenous",
                active_indices=collin_info["endogenous"]["active_indices"],
            ),
            design_state(
                zdesign, role="excluded_instrument",
                active_indices=collin_info["excluded_instruments"]["active_indices"],
            ),
        ),
        sample_state=sample_state,
        fe_plan=fe_plan,
        fe_names=fe_names,
        requested_fe_groups=inference_fe.groups,
        effective_fe_groups=groups,
        fe_intercepts=intercepts,
        fe_slopes=slopes,
        recovered_effects=fixed_effects,
    )
    result = RegressionResult(
        params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
        residuals=resid, fitted=fitted, nobs=int(round(winfo.effective_n)), rank=rank,
        df_resid=df_resid, df_absorbed=dof.df_absorbed,
        converged=info.converged, iterations=info.iterations,
        dropped_singletons=dropped, names=names, first_stage=first,
        diagnostics=diagnostics, estimator=meta.get("estimator", est),
        estimator_info=meta, dof_info=dof, weight_type=winfo.kind,
        sum_weights=winfo.sum_weights, nobs_raw=len(yw), fixed_effects=fixed_effects,
        absorb_info=absorb_info,
        collinearity_info=collin_info,
        vce=vce_kind, cluster_counts=_cluster_counts(clusters), fe_names=tuple(fe_names),
        r2=r2_stats["r2"], r2_within=r2_stats["r2_within"],
        r2_adjusted=r2_stats["r2_adjusted"], r2_adjusted_within=r2_stats["r2_adjusted_within"],
        rss=fit_stats["rss"], tss_within=fit_stats["tss_within"], rmse=fit_stats["rmse"],
        f_statistic=fit_stats["f_statistic"], f_pvalue=fit_stats["f_pvalue"],
        df_model=fit_stats["df_model"], df_resid_fit=fit_stats["df_resid_fit"], vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        state=EstimationState(
            absorber, yw, np.column_stack([Cw, Ew]), tuple(clusters or ()), w1, winfo.kind
        ) if keep_state else None,
        confidence_level=confidence_level,
        diagnostics_mode=inference_config.diagnostics if inference_config is not None else "off",
        prediction_state=prediction_state,
    )
    if _t0 is not None:
        result.profile = {"total_seconds": perf_counter() - _t0, "mode": execution_config.profile}
        if data_plan is not None:
            result.profile["data_ingestion"] = data_plan.as_dict()
        result.profile["sample"] = sample_state.summary()
    return result


# Backward-compatible estimator name.
ivreghdfe = ivhdfe
