from __future__ import annotations
from dataclasses import replace
from time import perf_counter
from ..errors import error_boundary
from ..config import HDFEConfig, InferenceConfig, ExecutionConfig
from ..data import materialize_model_data, EstimationSampleState
from ..frontend.validate import require_numeric
from ..frontend.roles import VariableRole
import numpy as np
import scipy.linalg as la
from ..pipeline import (
    _col, _design, _execution_design, _prepare_group_mode, _make_group_absorber, _residualize_blocks,
    _resolve_ols_columns, _group_dof, _group_info, _collinearity_metadata,
    _check_collinearity_mode, _resolve_method, _resolve_acceleration,
    _prepare_standard_fe_structure, _make_standard_absorber, _check_absorption,
    _cluster_arrays, _masked_array, _absorb_metadata, _resolve_pool_size,
)
from ..hdfe.dof import absorbed_dof
from ..compute.weights import (
    normalize_weight_type, prepare_weights, resolve_vce_for_weights, robust_score_scale,
)
from ..compute.wls import weighted_arrays
from ..compute.stable_linalg import equilibrated_lstsq, equilibrated_gram_inverse
from ..compute.vcov import ols_vcov, sandwich_vcov_block_xe
from ..compute.block_design import BlockDesign
from ..compute.partitioned_lstsq import partitioned_weighted_lstsq
from ..hdfe.block_projection import BlockWeightedFEProjector
from ..hdfe.plan import FEPlan
from ..results import RegressionResult, EstimationState, FixedEffectEstimates, FixedEffectTermEstimate
from ..prediction import design_state, linear_prediction_state
from ..effects.prediction import saved_categorical_state
from ..reporting import cluster_counts as _cluster_counts, reghdfe_model_statistics
from ..design import is_heterogeneous_spec_candidate


def _fit_ols(y, X, *, weights=None, weight_info=None, vce="iid", clusters=None,
             df_absorbed=0, nested_adj=0, time=None, panel=None, bandwidth=None,
             kernel="bartlett"):
    """OLS estimator-specific inference built on generic weighted arrays."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    yw, Xw, _, sw = weighted_arrays(y, X, None, weights)
    beta, bread, rank = equilibrated_lstsq(Xw, yw)
    resid = y - X @ beta
    ew = resid if sw is None else resid * sw
    k_total = int(rank) + int(df_absorbed)
    n_eff = float(len(y) if weight_info is None else weight_info.effective_n)
    score_scale = None if weight_info is None else robust_score_scale(weight_info, vce, len(y))
    V = ols_vcov(
        Xw, ew, bread, kind=vce, clusters=clusters, k_total=k_total,
        nested_adj=int(bool(nested_adj)), time=time, panel=panel,
        bandwidth=bandwidth, kernel=kernel, effective_n=n_eff,
        score_scale=score_scale,
    )
    df_resid = max(n_eff - k_total, 0.0)
    if str(vce).lower().replace("-", "_") == "cluster" and clusters:
        df_resid = min(df_resid, float(min(len(np.unique(c)) for c in clusters) - 1))
    return beta, V, resid, int(rank), df_resid


def _fit_ols_block(y, X: BlockDesign, *, weights=None, weight_info=None, vce="iid",
                   clusters=None, df_absorbed=0, nested_adj=0):
    """High-accuracy OLS/WLS on a heterogeneous-spec BlockDesign."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    beta, resid, lsinfo = partitioned_weighted_lstsq(X, y, w)
    bread = equilibrated_gram_inverse(X.gram(weights=w))
    rank = int(lsinfo.rank)
    k_total = rank + int(df_absorbed)
    n_eff = float(len(y) if weight_info is None else weight_info.effective_n)
    score_scale = None if weight_info is None else robust_score_scale(weight_info, vce, len(y))
    kind0 = "iid" if vce is None else str(vce).lower().replace("-", "_")
    if kind0 in {"iid", "unadjusted", "homoskedastic"}:
        df = max(n_eff - k_total, 1)
        V = bread * (float(np.dot(w * resid, resid)) / df)
    else:
        V = sandwich_vcov_block_xe(
            X, w * resid, bread, kind=kind0, clusters=clusters,
            k_total=k_total, nested_adj=int(bool(nested_adj)), effective_n=n_eff,
            score_scale=score_scale,
        )
    df_resid = max(n_eff - k_total, 0.0)
    if kind0 == "cluster" and clusters:
        df_resid = min(df_resid, float(min(len(np.unique(c)) for c in clusters) - 1))
    return beta, V, resid, rank, df_resid, lsinfo

def _reghdfe_group_mode(
    data=None, *, y, x, absorb, group, individual=None, aggregation="mean",
    weights=None, weight_type=None, cluster=None, vce=None, drop_singletons=True,
    tol=1e-8, max_iter=16_000, method="map", transform="symmetric",
    acceleration="none", preconditioner="diagonal", backend="numpy",
    pool_size="auto", memory_budget_mb=512, dof_method="pairwise", keep_state=False,
    save_fe=False, time=None, panel=None, bandwidth=None, kernel="bartlett",
    incidence_backend="auto", collinearity="warn", collinear_tol=1e-10, omit=None,
):
    ylong = require_numeric(_col(data, y), name=str(y), role=VariableRole.OUTCOME, ndim=1)
    xdesign = _design(data, x, len(ylong), "x", structural=False, omit=omit)
    Xlong = xdesign.values
    prep = _prepare_group_mode(
        data, group=group, individual=individual, absorb=absorb,
        arrays=[("dependent variable", ylong), ("regressors", Xlong)],
        weights=weights, cluster=cluster, time=time, panel=panel,
        drop_singletons=drop_singletons, weight_type=weight_type, aggregation=aggregation,
    )
    y1, X1 = prep["arrays"]
    absorber = _make_group_absorber(
        prep, method=method, transform=transform, backend=backend, acceleration=acceleration,
        preconditioner=preconditioner, tol=tol, max_iter=max_iter, aggregation=aggregation,
        incidence_backend=incidence_backend, memory_budget_mb=memory_budget_mb,
    )
    block = np.column_stack([y1, X1])
    within, info = _residualize_blocks(absorber, block, pool_size, memory_budget_mb)
    yw, Xw = within[:, 0], within[:, 1:]
    Xw, X1_active, collin_plan = _resolve_ols_columns(
        Xw, X1, xdesign.names, weights=prep["weights"],
        tolerance=collinear_tol, mode=collinearity,
    )
    dof = _group_dof(prep, absorber, dof_method)
    winfo = prep["winfo"]
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(prep["clusters"]))
    beta, V, resid, rank, df_resid = _fit_ols(
        yw, Xw, weights=prep["weights"], weight_info=winfo, vce=vce_kind,
        clusters=prep["clusters"], df_absorbed=dof.df_absorbed,
        nested_adj=int(dof.nested > 0), time=prep["time"], panel=prep["panel"],
        bandwidth=bandwidth, kernel=kernel,
    )
    fixed_effects = None
    if save_fe:
        target = y1 - X1_active @ beta - resid
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
    fit_stats = reghdfe_model_statistics(
        y1, yw, resid, beta, V, weights=prep["weights"], effective_n=winfo.effective_n,
        rank=rank, df_absorbed=dof.df_absorbed, df_nested=dof.nested,
        df_resid_inference=df_resid, has_intercept=has_intercept,
    )
    return RegressionResult(
        params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
        residuals=resid, fitted=y1-resid, nobs=int(round(winfo.effective_n)), rank=rank,
        df_resid=df_resid, df_absorbed=dof.df_absorbed, converged=info.converged,
        iterations=info.iterations, dropped_singletons=prep["dropped"],
        names=tuple(collin_plan.active_names), estimator="ols", dof_info=dof,
        weight_type=winfo.kind, sum_weights=winfo.sum_weights, nobs_raw=len(yw),
        fixed_effects=fixed_effects, group_info=_group_info(prep, absorber, aggregation),
        collinearity_info=_collinearity_metadata(
            collin_plan, xdesign.origins, structural_plan=xdesign.structural_plan,
            requested_origins=xdesign.requested_origins, requested_names=xdesign.requested_names,
            role="regressor", user_omissions=xdesign.user_omissions,
        ),
        vce=vce_kind, cluster_counts=_cluster_counts(prep["clusters"]), fe_names=tuple(prep["names"]),
        r2=fit_stats["r2"], r2_within=fit_stats["r2_within"],
        r2_adjusted=fit_stats["r2_adjusted"], r2_adjusted_within=fit_stats["r2_adjusted_within"],
        rss=fit_stats["rss"], tss=fit_stats["tss"], tss_within=fit_stats["tss_within"],
        mss=fit_stats["mss"], rmse=fit_stats["rmse"], loglike=fit_stats["loglike"],
        loglike_null=fit_stats["loglike_null"], f_statistic=fit_stats["f_statistic"],
        f_pvalue=fit_stats["f_pvalue"], df_model=fit_stats["df_model"],
        df_resid_fit=fit_stats["df_resid_fit"], vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        state=EstimationState(absorber, yw, Xw, tuple(prep["clusters"] or ()), prep["weights"], winfo.kind)
        if keep_state else None,
    )

@error_boundary("ols")
def olshdfe(
    data=None, *, y, x, absorb, weights=None, weight_type=None, cluster=None,
    vce=None, drop_singletons=True, tol=1e-8, max_iter=16_000,
    method="auto", transform="symmetric", acceleration="auto",
    preconditioner="diagonal", backend="numpy", pool_size="auto", memory_budget_mb=512,
    projection_backend="auto", absorb_threads="auto",
    dof_method="pairwise", keep_state=False, save_fe=False, time=None, panel=None, bandwidth=None, kernel="bartlett",
    group=None, individual=None, aggregation="mean", incidence_backend="auto",
    canonicalize_fe=True, allow_nonconverged=False, collinearity="warn", collinear_tol=1e-10,
    structural_collinearity=True, omit=None,
    hdfe_config: HDFEConfig | None = None, inference_config: InferenceConfig | None = None,
    execution_config: ExecutionConfig | None = None,
):
    """High-dimensional FE OLS with optional heterogeneous slopes.

    Examples
    --------
    ``absorb=["firm", "year"]`` absorbs standard intercept FEs.

    ``absorb=[FixedEffect("firm", slopes=("tenure",), intercept=True), "year"]``
    absorbs firm intercepts and firm-specific tenure slopes plus year FE.
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
    identifier_levels = getattr(data, "identifier_levels", {})
    data, data_plan = materialize_model_data(
        data, memory_budget_mb=memory_budget_mb,
        y=y, x=x, absorb=absorb, weights=weights, cluster=cluster,
        time=time, panel=panel, group=group, individual=individual,
    )
    if data_plan is not None:
        identifier_levels = data_plan.identifier_levels

    collinearity = _check_collinearity_mode(collinearity)
    if group is not None:
        method_resolved, solver_selection = _resolve_method(
            method, group_mode=True, individual=individual is not None, backend=backend
        )
        acceleration_resolved = _resolve_acceleration(method_resolved, acceleration, group_mode=True)
        result = _reghdfe_group_mode(
            data, y=y, x=x, absorb=absorb, group=group, individual=individual,
            aggregation=aggregation, weights=weights, weight_type=weight_type, cluster=cluster,
            vce=vce, drop_singletons=drop_singletons, tol=tol, max_iter=max_iter, method=method_resolved,
            transform=transform, acceleration=acceleration_resolved, preconditioner=preconditioner,
            backend=backend, pool_size=pool_size, memory_budget_mb=memory_budget_mb,
            dof_method=dof_method, keep_state=keep_state, save_fe=save_fe, time=time, panel=panel,
            bandwidth=bandwidth, kernel=kernel, incidence_backend=incidence_backend,
            collinearity=collinearity, collinear_tol=collinear_tol, omit=omit,
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
    # Compile the regressor design only on the final estimation sample and
    # only after FE canonicalization. This lets structural factor/interaction
    # dependencies be removed before N x K dummy columns are materialized.
    all_kept = dropped == 0 and bool(np.all(mask))
    active_mask = None if all_kept else mask
    y1 = y0 if all_kept else y0[mask]
    absorbed_intercept_groups = [g for g, has_i in zip(groups, intercepts, strict=False) if has_i]
    absorbed_intercept_names = [n for n, has_i in zip(fe_names, intercepts, strict=False) if has_i]
    vce_requested = "" if vce is None else str(vce).lower().replace("-", "_")
    block_vce_ok = vce_requested in {
        "", "iid", "unadjusted", "homoskedastic", "robust", "hc1",
        "heteroskedastic", "cluster",
    }
    block_solver_ok = (
        backend == "numpy" and all(s is None for s in slopes) and all(intercepts)
        and projection_backend == "auto" and transform == "symmetric"
        and preconditioner == "diagonal" and not keep_state and not save_fe
        and block_vce_ok
        and (method_resolved == "twoway" or (method_resolved == "map" and acceleration_resolved == "cg"))
        and is_heterogeneous_spec_candidate(x)
    )
    if block_solver_ok:
        xdesign = _execution_design(
            data, x, len(y0), "x", groups=groups, expected_passes=2,
            memory_budget_mb=memory_budget_mb, row_mask=active_mask,
            absorbed_groups=absorbed_intercept_groups, absorbed_names=absorbed_intercept_names,
            structural=bool(structural_collinearity), omit=omit,
        )
    else:
        xdesign = _design(
            data, x, len(y0), "x", row_mask=active_mask,
            absorbed_groups=absorbed_intercept_groups, absorbed_names=absorbed_intercept_names,
            structural=bool(structural_collinearity), omit=omit,
        )
    X1 = xdesign.values
    w_active = w0 if (w0 is not None and all_kept) else (None if w0 is None else w0[mask])
    winfo = prepare_weights(w_active, len(y1), wkind)
    w1 = winfo.estimation
    block_native = isinstance(X1, BlockDesign)
    absorber = None
    block_projector = None
    block = None
    if block_native:
        fe_exec_plan = FEPlan.from_arrays(groups)
        block_projector = BlockWeightedFEProjector.from_design(
            X1, fe_exec_plan,
            engine="optimized" if method_resolved == "twoway" else "replica",
            method="map", absorb_threads=absorb_threads,
            projection_memory_budget_mb=memory_budget_mb,
        )
        wproj = np.ones(len(y1), dtype=np.float64) if w1 is None else w1
        projected = block_projector.residualize_response_design(
            y1, X1, wproj, tol=tol,
        )
        yw, Xw = projected.response, projected.design
        info = projected
        if not info.converged and not allow_nonconverged:
            raise RuntimeError("heterogeneous-spec FE projection did not converge")
    else:
        absorber = _make_standard_absorber(
            groups, slopes, intercepts, weights=w1, tol=tol, max_iter=max_iter,
            backend=backend, method=method_resolved, transform=transform, acceleration=acceleration_resolved,
            preconditioner=preconditioner, projection_backend=projection_backend,
            absorb_threads=absorb_threads, memory_budget_mb=memory_budget_mb,
        )
        block = np.column_stack([y1, X1])
        within, info = _residualize_blocks(
            absorber, block, pool_size, memory_budget_mb,
            adaptive_auto=(str(method).lower() == "auto" and str(acceleration).lower() == "auto"),
        )
        _check_absorption(info, allow_nonconverged=allow_nonconverged)
        yw, Xw = within[:, 0], within[:, 1:]
    # Cluster/time/panel metadata does not participate in FE absorption. Delay
    # encoding until the large within transform is complete to lower peak RSS.
    clusters = _cluster_arrays(data, cluster, mask)
    time1 = _masked_array(data, time, mask)
    panel1 = _masked_array(data, panel, mask)
    Xw, X1_active, collin_plan = _resolve_ols_columns(
        Xw, X1, xdesign.names, weights=w1, tolerance=collinear_tol, mode=collinearity,
        structural_plan=xdesign.structural_plan,
    )
    dof = absorbed_dof(
        inference_fe.groups, slopes=inference_fe.slopes, intercepts=inference_fe.intercepts,
        clusters=clusters, method=dof_method, adjust_nested=bool(clusters),
        adjust_continuous=True, groups_are_dense=True,
    )
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(clusters))
    if isinstance(Xw, BlockDesign):
        beta, V, resid, rank, df_resid, lsinfo = _fit_ols_block(
            yw, Xw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
        )
    else:
        beta, V, resid, rank, df_resid = _fit_ols(
            yw, Xw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            time=time1, panel=panel1, bandwidth=bandwidth, kernel=kernel,
        )
    fitted = y1 - resid
    fixed_effects = None
    if save_fe:
        target = y1 - X1_active @ beta - resid
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
    names = tuple(collin_plan.active_names)
    fit_stats = reghdfe_model_statistics(
        y1, yw, resid, beta, V, weights=w1, effective_n=winfo.effective_n,
        rank=rank, df_absorbed=dof.df_absorbed, df_nested=dof.nested,
        df_resid_inference=df_resid, has_intercept=bool(any(inference_fe.intercepts)),
    )
    if isinstance(Xw, BlockDesign):
        resources = block_projector.resource_info if block_projector is not None else {"blocks": ()}
        actual_threads = [
            int(item["projector"].get("actual_threads") or 1)
            for item in resources.get("blocks", ()) if item.get("projector") is not None
        ]
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
            "pool_size": int(Xw.ncols),
            "solver_selection": dict(solver_selection),
            "heterogeneous_spec": {
                "storage": xdesign.storage_plan.as_dict(),
                "structure": xdesign.execution_structure.as_dict(),
                "wls_method": lsinfo.method,
            },
        }
    else:
        absorb_info = _absorb_metadata(
            fe_plan, absorber, info,
            _resolve_pool_size(absorber, block.shape[1], pool_size, memory_budget_mb),
            solver_selection=solver_selection,
        )
    prediction_state = linear_prediction_state(
        estimator="ols",
        coefficient_names=names,
        coefficient_roles=("regressor",),
        designs=(
            design_state(
                xdesign, role="regressor", active_indices=collin_plan.active_indices,
                identifier_levels=identifier_levels,
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
    if save_fe:
        fe_state = saved_categorical_state(
            prediction_state.fixed_effects, data=data, mask=mask,
            inference=inference_fe, plan=fe_plan, groups=groups,
            recovered=fixed_effects, beta=beta, identifier_levels=identifier_levels,
        )
        prediction_state = replace(prediction_state, fixed_effects=fe_state)
    result = RegressionResult(
        params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
        residuals=resid, fitted=fitted, nobs=int(round(winfo.effective_n)), rank=rank,
        df_resid=df_resid, df_absorbed=dof.df_absorbed,
        converged=info.converged, iterations=info.iterations,
        dropped_singletons=dropped, names=names, estimator="ols",
        dof_info=dof, weight_type=winfo.kind, sum_weights=winfo.sum_weights,
        nobs_raw=len(yw), fixed_effects=fixed_effects,
        absorb_info=absorb_info,
        collinearity_info=_collinearity_metadata(
            collin_plan, xdesign.origins, structural_plan=xdesign.structural_plan,
            requested_origins=xdesign.requested_origins, requested_names=xdesign.requested_names,
            role="regressor", user_omissions=xdesign.user_omissions,
        ),
        vce=vce_kind, cluster_counts=_cluster_counts(clusters), fe_names=tuple(fe_names),
        r2=fit_stats["r2"], r2_within=fit_stats["r2_within"],
        r2_adjusted=fit_stats["r2_adjusted"], r2_adjusted_within=fit_stats["r2_adjusted_within"],
        rss=fit_stats["rss"], tss=fit_stats["tss"], tss_within=fit_stats["tss_within"],
        mss=fit_stats["mss"], rmse=fit_stats["rmse"], loglike=fit_stats["loglike"],
        loglike_null=fit_stats["loglike_null"], f_statistic=fit_stats["f_statistic"],
        f_pvalue=fit_stats["f_pvalue"], df_model=fit_stats["df_model"],
        df_resid_fit=fit_stats["df_resid_fit"], vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        state=EstimationState(absorber, yw, Xw, tuple(clusters or ()), w1, winfo.kind)
        if keep_state else None,
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
reghdfe = olshdfe
