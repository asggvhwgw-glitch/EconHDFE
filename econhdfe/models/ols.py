from __future__ import annotations
from time import perf_counter
from ..errors import error_boundary
from ..config import HDFEConfig, InferenceConfig, ExecutionConfig
from ..planner.scheduler import managed_execution
from ..data import materialize_model_data
from ..frontend.validate import require_numeric
from ..frontend.roles import VariableRole
import numpy as np
from ..pipeline import (
    column_values, compile_design, compile_execution_design, prepare_group_mode, make_group_absorber,
    residualize_blocks, resolve_ols_columns, group_dof, group_info, group_has_intercept,
    design_collinearity_metadata, check_collinearity_mode, resolve_method, resolve_acceleration,
    standard_dof, make_standard_absorber, check_absorption, cluster_arrays, masked_array,
    absorb_metadata, block_absorb_metadata, conditioning_info, resolve_pool_size,
    weak_basis_eligible,
)
from .linear import (
    StageClock, config_precedence, sample_policy, sample_policy_scope, reporting_options, prepare_standard_sample,
    finish_group_mode, nonconvergence_error, recover_fixed_effects, linear_result, prediction_state,
    with_saved_effects,
)
from ..compute.weights import prepare_weights, resolve_vce_for_weights, robust_score_scale
from ..compute.wls import weighted_arrays
from ..compute.stable_linalg import (
    equilibrated_lstsq, equilibrated_gram_inverse,
    _absorption_basis, _apply_absorption_basis,
)
from ..compute.vcov import (
    ols_vcov, sandwich_vcov_block_xe, normalize_vce, is_iid_vce, cluster_df_resid, BLOCK_VCE_KINDS,
    finalize_cluster_vcov, regressor_scale,
)
from ..compute.block_design import BlockDesign
from ..compute.execution import _check_allocation_headroom
from ..compute.partitioned_lstsq import partitioned_weighted_lstsq
from ..hdfe.block_projection import BlockWeightedFEProjector
from ..hdfe.plan import FEPlan
from ..results import EstimationState
from ..prediction import design_state
from ..design import is_heterogeneous_spec_candidate
from ..collinearity import _weighted_gram


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
    df_resid = cluster_df_resid(max(n_eff - k_total, 0.0), vce, clusters)
    return beta, V, resid, int(rank), df_resid


def fit_ols_in_basis(y, X, *, basis=None, **kwargs):
    """OLS inference in stable coordinates; restores the owned design afterwards.

    Shared by ``olshdfe`` and ``OLSHDFESession`` so both report identical
    estimates for a conditioned (near-collinear) design.
    """
    if basis is None:
        return _fit_ols(y, X, **kwargs)
    _apply_absorption_basis(X, basis)
    try:
        beta, V, resid, rank, df_resid = _fit_ols(y, X, **kwargs)
        scale, forward, _, _ = basis
        beta = (forward @ beta) / scale
        V = ((forward @ V @ forward.T) / scale[:, None]) / scale[None, :]
        return beta, (V + V.T) / 2, resid, rank, df_resid
    finally:
        _apply_absorption_basis(X, basis, restore=True)


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
    kind0 = normalize_vce(vce, "iid")
    if is_iid_vce(kind0):
        df = max(n_eff - k_total, 1)
        V = bread * (float(np.dot(w * resid, resid)) / df)
    else:
        V = sandwich_vcov_block_xe(
            X, w * resid, bread, kind=kind0, clusters=clusters,
            k_total=k_total, nested_adj=int(bool(nested_adj)), effective_n=n_eff,
            score_scale=score_scale,
        )
    df_resid = cluster_df_resid(max(n_eff - k_total, 0.0), kind0, clusters)
    return beta, V, resid, rank, df_resid, lsinfo

def _reghdfe_group_mode(
    data=None, *, y, x, absorb, group, individual=None, aggregation="mean",
    weights=None, weight_type=None, cluster=None, vce=None, drop_singletons=True,
    tol=1e-8, max_iter=16_000, method="map", transform="symmetric",
    acceleration="none", preconditioner="diagonal", backend="numpy",
    pool_size="auto", memory_budget_mb=512, dof_method="pairwise", keep_state=False,
    save_fe=False, time=None, panel=None, bandwidth=None, kernel="bartlett",
    projection_backend="auto", absorb_threads="auto",
    incidence_backend="auto", collinearity="warn", collinear_tol=1e-10, omit=None,
):
    method, _ = resolve_method(method, group_mode=True, individual=individual is not None, backend=backend)
    acceleration = resolve_acceleration(method, acceleration, group_mode=True)
    ylong = require_numeric(column_values(data, y), name=str(y), role=VariableRole.OUTCOME, ndim=1)
    xdesign = compile_design(data, x, len(ylong), "x", structural=False, omit=omit)
    Xlong = xdesign.values
    prep = prepare_group_mode(
        data, group=group, individual=individual, absorb=absorb,
        arrays=[("dependent variable", ylong), ("regressors", Xlong)],
        weights=weights, cluster=cluster, time=time, panel=panel,
        drop_singletons=drop_singletons, weight_type=weight_type, aggregation=aggregation,
    )
    y1, X1 = prep["arrays"]
    absorber = make_group_absorber(
        prep, method=method, transform=transform, backend=backend, acceleration=acceleration,
        preconditioner=preconditioner, tol=tol, max_iter=max_iter, aggregation=aggregation,
        incidence_backend=incidence_backend, memory_budget_mb=memory_budget_mb,
        projection_backend=projection_backend, absorb_threads=absorb_threads,
    )
    block = np.column_stack([y1, X1])
    within, info = residualize_blocks(absorber, block, pool_size, memory_budget_mb)
    yw, Xw = within[:, 0], within[:, 1:]
    Xw, X1_active, collin_plan = resolve_ols_columns(
        Xw, X1, xdesign.names, weights=prep["weights"],
        tolerance=collinear_tol, mode=collinearity,
    )
    dof = group_dof(prep, absorber, dof_method)
    winfo = prep["winfo"]
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(prep["clusters"]))
    beta, V, resid, rank, df_resid = _fit_ols(
        yw, Xw, weights=prep["weights"], weight_info=winfo, vce=vce_kind,
        clusters=prep["clusters"], df_absorbed=dof.df_absorbed,
        nested_adj=int(dof.nested > 0), time=prep["time"], panel=prep["panel"],
        bandwidth=bandwidth, kernel=kernel,
    )
    # group()/individual() is compatibility-only, so the scale is reghdfe's unweighted one.
    V = finalize_cluster_vcov(V, vce_kind, prep["clusters"], lambda: regressor_scale(X1_active))
    fixed_effects = None
    if save_fe:
        fixed_effects = recover_fixed_effects(absorber, y1 - X1_active @ beta - resid, prep["names"])
    return linear_result(
        beta, V, resid, y=y1, y_within=yw, rank=rank, df_resid=df_resid, dof=dof, winfo=winfo,
        clusters=prep["clusters"], vce=vce_kind, has_intercept=group_has_intercept(prep),
        converged=info.converged, iterations=info.iterations, dropped_singletons=prep["dropped"],
        names=tuple(collin_plan.active_names), estimator="ols", fixed_effects=fixed_effects,
        group_info=group_info(prep, absorber, aggregation),
        collinearity_info=design_collinearity_metadata(collin_plan, xdesign, "regressor"),
        fe_names=tuple(prep["names"]),
        state=EstimationState(absorber, yw, Xw, tuple(prep["clusters"] or ()), prep["weights"], winfo.kind)
        if keep_state else None,
    )

@error_boundary("ols")
@config_precedence
@sample_policy_scope
@managed_execution
def olshdfe(
    data=None, *, y, x, absorb, weights=None, weight_type=None, cluster=None,
    vce=None, drop_singletons=True, tol=1e-8, max_iter=16_000,
    method="auto", transform="symmetric", acceleration="auto",
    preconditioner="diagonal", backend="numpy", pool_size="auto", memory_budget_mb="auto",
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

    Config objects and scalar arguments are merged field by field:
    ``HDFEConfig`` solver/tolerance/max_iter/dof_method/canonicalize pair with
    ``method``/``tol``/``max_iter``/``dof_method``/``canonicalize_fe``,
    ``InferenceConfig.vce`` with ``vce`` and ``ExecutionConfig``
    memory_budget_mb/threads with ``memory_budget_mb``/``absorb_threads``. A
    value equal to its default counts as unset and the set side wins; setting
    both sides to different values raises ``SpecificationError``
    (``code="specification.config_conflict"``). Remaining config fields have
    no scalar counterpart and always apply.

    FE absorption that does not converge raises ``HDFEConvergenceError`` (a
    ``ConvergenceError``) unless ``allow_nonconverged=True``.
    """
    clock = StageClock(execution_config)
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

    collinearity = check_collinearity_mode(collinearity)
    if group is not None:
        result = _reghdfe_group_mode(
            data, y=y, x=x, absorb=absorb, group=group, individual=individual,
            aggregation=aggregation, weights=weights, weight_type=weight_type, cluster=cluster,
            vce=vce, drop_singletons=drop_singletons, tol=tol, max_iter=max_iter, method=method,
            transform=transform, acceleration=acceleration, preconditioner=preconditioner,
            backend=backend, pool_size=pool_size, memory_budget_mb=memory_budget_mb,
            dof_method=dof_method, keep_state=keep_state, save_fe=save_fe, time=time, panel=panel,
            bandwidth=bandwidth, kernel=kernel, incidence_backend=incidence_backend,
            projection_backend=projection_backend, absorb_threads=absorb_threads,
            collinearity=collinearity, collinear_tol=collinear_tol, omit=omit,
        )
        return finish_group_mode(
            result, allow_nonconverged=allow_nonconverged, inference_config=inference_config,
            clock=clock, data_plan=data_plan,
        )

    y0, w0, wkind, fe, sample_state = prepare_standard_sample(
        data, y, weights, weight_type, absorb,
        canonicalize_fe=canonicalize_fe, drop_singletons=drop_singletons,
        policy=sample_policy(hdfe_config),
    )
    groups, slopes, intercepts, mask = fe.groups, fe.slopes, fe.intercepts, fe.mask
    method_resolved, solver_selection = resolve_method(
        method, groups, slopes, intercepts, backend=backend
    )
    acceleration_resolved = resolve_acceleration(method_resolved, acceleration)
    clock.lap("data_and_fe_seconds")
    # Compile the regressor design only on the final estimation sample and
    # only after FE canonicalization. This lets structural factor/interaction
    # dependencies be removed before N x K dummy columns are materialized.
    active_mask = None if fe.all_kept else mask
    y1 = fe.active(y0)
    absorbed_intercept_groups, absorbed_intercept_names = fe.absorbed_intercepts()
    block_vce_ok = normalize_vce(vce, "") in BLOCK_VCE_KINDS
    block_solver_ok = (
        backend == "numpy" and all(s is None for s in slopes) and all(intercepts)
        and projection_backend == "auto" and transform == "symmetric"
        and preconditioner == "diagonal" and not keep_state and not save_fe
        and block_vce_ok
        and (method_resolved == "twoway" or (method_resolved == "map" and acceleration_resolved == "cg"))
        and is_heterogeneous_spec_candidate(x)
    )
    design_options = dict(
        row_mask=active_mask, absorbed_groups=absorbed_intercept_groups,
        absorbed_names=absorbed_intercept_names, structural=bool(structural_collinearity), omit=omit,
    )
    if block_solver_ok:
        xdesign = compile_execution_design(
            data, x, len(y0), "x", groups=groups, expected_passes=2,
            memory_budget_mb=memory_budget_mb, **design_options,
        )
    else:
        xdesign = compile_design(data, x, len(y0), "x", **design_options)
    X1 = xdesign.values
    winfo = prepare_weights(None if w0 is None else fe.active(w0), len(y1), wkind)
    w1 = winfo.estimation
    block_native = isinstance(X1, BlockDesign)
    absorber = None
    block_projector = None
    block = None
    conditioning = None
    raw_gram = None
    clock.lap("design_seconds")
    if block_native:
        block_projector = BlockWeightedFEProjector.from_design(
            X1, FEPlan.from_arrays(groups),
            engine="optimized" if method_resolved == "twoway" else "replica",
            method="map", max_iter=max_iter, absorb_threads=absorb_threads,
            projection_memory_budget_mb=memory_budget_mb,
        )
        wproj = np.ones(len(y1), dtype=np.float64) if w1 is None else w1
        clock.lap("absorber_setup_seconds")
        projected = block_projector.residualize_response_design(
            y1, X1, wproj, tol=tol,
        )
        yw, Xw = projected.response, projected.design
        info = projected
        if not info.converged and not allow_nonconverged:
            raise nonconvergence_error(
                "heterogeneous-spec FE projection did not converge",
                iterations=info.iterations, metric=info.max_update, info=info,
            )
    else:
        absorber = make_standard_absorber(
            groups, slopes, intercepts, weights=w1, tol=tol, max_iter=max_iter,
            backend=backend, method=method_resolved, transform=transform, acceleration=acceleration_resolved,
            preconditioner=preconditioner, projection_backend=projection_backend,
            absorb_threads=absorb_threads, memory_budget_mb=memory_budget_mb,
        )
        _check_allocation_headroom(len(y1) * (X1.shape[1] + 1) * 8)
        block = np.column_stack([y1, X1])
        basis = None
        conditioning_eligible = weak_basis_eligible(
            groups, slopes, intercepts, method=method_resolved, backend=backend,
        )
        if conditioning_eligible:
            raw_gram = _weighted_gram(X1, w1)
            basis = _absorption_basis(X1, w1, rank_tolerance=collinear_tol, gram=raw_gram)
            if basis is not None:
                _apply_absorption_basis(block[:, 1:], basis)
                absorber.tol = min(tol, 1e-11)
                conditioning = conditioning_info(basis, "before_absorption", absorber.tol)
        clock.lap("absorber_setup_seconds")
        within, info = residualize_blocks(
            absorber, block, pool_size, memory_budget_mb,
            adaptive_auto=(str(method).lower() == "auto" and str(acceleration).lower() == "auto"),
            profile=clock.enabled,
        )
        check_absorption(info, allow_nonconverged=allow_nonconverged)
        if basis is not None:
            _apply_absorption_basis(within[:, 1:], basis, restore=True)
        yw, Xw = within[:, 0], within[:, 1:]
    clock.lap("absorption_seconds")
    # Cluster/time/panel metadata does not participate in FE absorption. Delay
    # encoding until the large within transform is complete to lower peak RSS.
    clusters = cluster_arrays(data, cluster, mask)
    time1 = masked_array(data, time, mask)
    panel1 = masked_array(data, panel, mask)
    within_gram = _weighted_gram(Xw, w1) if raw_gram is not None else None
    Xw, X1_active, collin_plan = resolve_ols_columns(
        Xw, X1, xdesign.names, weights=w1, tolerance=collinear_tol, mode=collinearity,
        structural_plan=xdesign.structural_plan,
        _grams=None if raw_gram is None else (within_gram, raw_gram),
    )
    if not block_native and conditioning_eligible and conditioning is None:
        # Dependencies can become nearly exact only after FE removal. Inspect
        # the retained design, then reproject ORIGINAL columns in that basis;
        # transforming inaccurate within columns alone cannot recover a weak
        # direction lost during the first projection.
        active = np.ix_(collin_plan.active_indices, collin_plan.active_indices)
        basis = _absorption_basis(Xw, w1, rank_tolerance=collinear_tol, gram=within_gram[active])
        if basis is not None:
            refinement_start = perf_counter() if clock.enabled else None
            first_profile = getattr(absorber, "execution_profile", None)
            # Reuse the owned block, including when omissions reduced width.
            block = block[:, :X1_active.shape[1]+1]
            block[:, 0] = y1
            block[:, 1:] = X1_active
            _apply_absorption_basis(block[:, 1:], basis)
            absorber.tol = min(tol, 1e-11)
            within, corrected_info = residualize_blocks(
                absorber, block, pool_size, memory_budget_mb, profile=clock.enabled,
            )
            check_absorption(corrected_info, allow_nonconverged=allow_nonconverged)
            _apply_absorption_basis(within[:, 1:], basis, restore=True)
            yw, Xw = within[:, 0], within[:, 1:]
            corrected_info.iterations += info.iterations
            info = corrected_info
            if clock.enabled:
                clock.credit("absorption_seconds", perf_counter() - refinement_start)
                if first_profile is not None:
                    absorber.execution_profile["pools"] = (
                        first_profile["pools"] + absorber.execution_profile["pools"]
                    )
            conditioning = conditioning_info(basis, "after_absorption", absorber.tol)
    clock.lap("clusters_and_collinearity_seconds")
    dof = standard_dof(fe, clusters, dof_method)
    clock.lap("dof_seconds")
    vce_kind = resolve_vce_for_weights(vce, winfo, has_clusters=bool(clusters))
    if isinstance(Xw, BlockDesign):
        beta, V, resid, rank, df_resid, lsinfo = _fit_ols_block(
            yw, Xw, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
        )
    else:
        # Keep the sandwich computation in the well-scaled basis too. Forming
        # bread and cluster scores in the original near-collinear basis loses
        # digits even when the within projection itself is accurate.
        fit_basis = basis if conditioning is not None else None
        if fit_basis is not None and len(fit_basis[0]) != Xw.shape[1]:
            fit_basis = _absorption_basis(Xw, w1, rank_tolerance=collinear_tol)
        beta, V, resid, rank, df_resid = fit_ols_in_basis(
            yw, Xw, basis=fit_basis, weights=w1, weight_info=winfo, vce=vce_kind, clusters=clusters,
            df_absorbed=dof.df_absorbed, nested_adj=int(dof.nested > 0),
            time=time1, panel=panel1, bandwidth=bandwidth, kernel=kernel,
        )
    fw_scale = w1 if sample_policy(hdfe_config).psd_frequency_scale(winfo.kind) else None
    V = finalize_cluster_vcov(
        V, vce_kind, clusters, lambda: regressor_scale(X1_active, frequency_weights=fw_scale),
    )
    clock.lap("ols_and_vce_seconds")
    fixed_effects = None
    if save_fe:
        fixed_effects = recover_fixed_effects(absorber, y1 - X1_active @ beta - resid, fe.names)
    names = tuple(collin_plan.active_names)
    if isinstance(Xw, BlockDesign):
        absorb_info = block_absorb_metadata(
            fe.plan, block_projector, info, method=method_resolved, transform=transform,
            acceleration=acceleration_resolved, pool_size=Xw.ncols, solver_selection=solver_selection,
            heterogeneous_spec={
                "storage": xdesign.storage_plan.as_dict(),
                "structure": xdesign.execution_structure.as_dict(),
                "wls_method": lsinfo.method,
            },
        )
    else:
        absorb_info = absorb_metadata(
            fe.plan, absorber, info,
            resolve_pool_size(absorber, block.shape[1], pool_size, memory_budget_mb),
            solver_selection=solver_selection,
        )
    if conditioning is not None:
        absorb_info["conditioning"] = conditioning
    prediction = prediction_state(
        "ols", names, ("regressor",),
        (design_state(xdesign, role="regressor", active_indices=collin_plan.active_indices,
                      identifier_levels=identifier_levels),),
        fe, sample_state=sample_state, recovered_effects=fixed_effects,
    )
    if save_fe:
        prediction = with_saved_effects(
            prediction, fe, data=data, recovered=fixed_effects, beta=beta,
            identifier_levels=identifier_levels,
        )
    result = linear_result(
        beta, V, resid, y=y1, y_within=yw, rank=rank, df_resid=df_resid, dof=dof, winfo=winfo,
        clusters=clusters, vce=vce_kind, has_intercept=bool(any(fe.inference.intercepts)),
        converged=info.converged, iterations=info.iterations, dropped_singletons=fe.dropped,
        names=names, estimator="ols", fixed_effects=fixed_effects, absorb_info=absorb_info,
        collinearity_info=design_collinearity_metadata(collin_plan, xdesign, "regressor"),
        fe_names=tuple(fe.names),
        state=EstimationState(absorber, yw, Xw, tuple(clusters or ()), w1, winfo.kind)
        if keep_state else None,
        prediction_state=prediction, **reporting_options(inference_config),
    )
    if clock.enabled:
        result.profile = clock.summary()
        clock.lap("results_seconds")
        result.profile["stages"] = clock.stages
        if absorber is not None:
            result.profile["absorption"] = getattr(absorber, "execution_profile", None)
        if data_plan is not None:
            result.profile["data_ingestion"] = data_plan.as_dict()
        result.profile["sample"] = sample_state.summary()
    return result


# Backward-compatible estimator name.
reghdfe = olshdfe
