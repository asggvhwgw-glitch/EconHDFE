from __future__ import annotations
from ...planner.scheduler import managed_execution
from ...compute.vcov import regressor_scale
import math
import numpy as np
from ...errors import CollinearityError, UnderidentifiedError, ConvergenceError
from ...config import ExecutionConfig
from ...compute.context import ExecutionContext
from scipy.special import gammaln
from ...compute.linalg import independent_columns
from ...hdfe.plan import FEPlan
from ...iv.moments import additive_moments, overid_test
from ...iv.solve import weighted_2sls
from ..ppml.separation import detect_separation, initial_sample_selection, sample_policy_diagnostics
from ..ppml.execution import resolve_execution, get_projector
from ..poisson import (
    initial_active_rows, separation_survivors, poisson_clusters, poisson_inference, poisson_df_resid,
)
from .config import IVPPMLConfig
from .irls import fit_iv_irls, final_working_system, in_user_units
from .results import IVPPMLResult
from .standardize import IVPPMLStandardization
from .vce import ivppml_vcov, ivppml_ee_vcov, normalize_ivppml_vce
from ...compute.weights import frequency_score_scale
from ...reporting import cluster_counts as _cluster_counts, fe_names_from_metadata


def _project_for_rank(plan, A, w, config, projector=None):
    if not plan.groups:
        return np.asarray(A, dtype=np.float64)
    if projector is None:
        out, _ = plan.residualize(A, w, tol=config.target_inner_tol, engine=config.engine)
        return out
    out, _ = projector.residualize(A, w, tol=config.target_inner_tol, return_info=True)
    return out


def _prune_iv_columns(C, E, I, names_c, names_e, names_i, plan, weights, config, projector=None):
    """Remove post-FE collinearity while preserving exogenous instrument roles."""
    C = np.asarray(C, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    X = np.column_stack([C, E])
    block = np.column_stack([X, I])
    within = _project_for_rank(plan, block, weights, config, projector=projector)
    kx = X.shape[1]
    Xw, Iw = within[:, :kx], within[:, kx:]

    keep_x, _ = independent_columns(Xw, tol=max(config.target_inner_tol, 1e-12))
    if not len(keep_x):
        raise CollinearityError("all IV-PPML regressors are collinear after FE absorption")
    kc = C.shape[1]
    keep_c = keep_x[keep_x < kc]
    keep_e = keep_x[keep_x >= kc] - kc
    C, E = C[:, keep_c], E[:, keep_e]
    Cw, Ew = Xw[:, keep_c], Xw[:, kc + keep_e]
    names_c = tuple(np.asarray(names_c, dtype=object)[keep_c])
    names_e = tuple(np.asarray(names_e, dtype=object)[keep_e])
    if E.shape[1] == 0:
        raise UnderidentifiedError("all endogenous regressors are collinear after FE absorption")

    # Exogenous regressors are automatically instruments and must stay in the
    # instrument space. Remove from excluded Z only the columns explained by C
    # or by other excluded instruments.
    if I.shape[1]:
        if Cw.shape[1]:
            coef = np.linalg.lstsq(Cw, Iw, rcond=None)[0]
            Ires = Iw - Cw @ coef
        else:
            Ires = Iw
        keep_i, _ = independent_columns(Ires, tol=max(config.target_inner_tol, 1e-12))
        I = I[:, keep_i]
        names_i = tuple(np.asarray(names_i, dtype=object)[keep_i])
    if I.shape[1] < E.shape[1]:
        raise UnderidentifiedError("equation not identified after collinearity removal")
    return C, E, I, names_c, names_e, names_i


@managed_execution
def fit_arrays(
    y, exog, endog, instruments, plan: FEPlan, *, offset, true_w, weight_kind="none", vce,
    clusters, exog_names, endog_names, instrument_names, config: IVPPMLConfig,
    misspecification_robust=False, warm_start=None, execution_config: ExecutionConfig | None = None,
    execution_context: ExecutionContext | None = None,
):
    y = np.asarray(y, dtype=np.float64)
    C = np.asarray(exog, dtype=np.float64)
    E = np.asarray(endog, dtype=np.float64)
    I = np.asarray(instruments, dtype=np.float64)
    w = np.asarray(true_w, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    n = len(y)
    execution, exec_context = resolve_execution(execution_config, execution_context)

    # No-FE IV-PPML is represented by a single constant FE, matching upstream
    # noabsorb semantics and keeping the IRLS-IV slope system intercept-free.
    # ivppmlhdfe first stage: fixed_effects(..., "iweight", depvar) when
    # separation includes "fe" -- singletons by row count (user weights never
    # enter) pruned jointly with FE separation; all counted as singletons.
    if not plan.groups:
        plan = FEPlan.from_arrays([np.zeros(n, dtype=np.int8)])
    initial = initial_sample_selection(y, plan, config, weights=w, weight_kind=weight_kind)
    active0 = initial_active_rows(initial, require_rows=True)
    if np.all(active0):
        y0, C0, E0, I0, w0, off0, p0 = y, C, E, I, w, off, plan
    else:
        y0, C0, E0, I0, w0, off0 = y[active0], C[active0], E[active0], I[active0], w[active0], off[active0]
        p0 = plan.subset(active0)

    pre_projector = get_projector(
        p0, engine=config.engine, method="map", execution=execution,
        context=exec_context, cache=True,
    )
    C0, E0, I0, exog_names, endog_names, instrument_names = _prune_iv_columns(
        C0, E0, I0, exog_names, endog_names, instrument_names,
        p0, w0, config, projector=pre_projector,
    )
    X0 = np.column_stack([C0, E0])
    sep = detect_separation(
        y0, X0, p0, w0, config, execution, frequency_weights=(weight_kind == "fweight"),
    )
    # Upstream re-prunes singletons after simplex/ReLU trims (fweight mass for
    # fweights) and counts them in num_separated (num_sep_advanced).
    keep_sep = separation_survivors(y0, sep, initial.n_fe)

    separation = initial.separated.copy()
    if not np.all(keep_sep):
        idx0 = np.flatnonzero(active0)
        sample = np.zeros(n, dtype=bool)
        sample[idx0[keep_sep]] = True
        separation[idx0[sep.separated if config.sample_policy.disjoint_reasons else ~keep_sep]] = True
        y1, C1, E1, I1, w1, off1 = y0[keep_sep], C0[keep_sep], E0[keep_sep], I0[keep_sep], w0[keep_sep], off0[keep_sep]
        p1 = p0.subset(keep_sep)
        pre_projector = get_projector(
            p1, engine=config.engine, method="map", execution=execution,
            context=exec_context, cache=False,
        )
        C1, E1, I1, exog_names, endog_names, instrument_names = _prune_iv_columns(
            C1, E1, I1, exog_names, endog_names, instrument_names,
            p1, w1, config, projector=pre_projector,
        )
    else:
        sample = active0.copy()
        y1, C1, E1, I1, w1, off1, p1 = y0, C0, E0, I0, w0, off0, p0

    X1 = np.column_stack([C1, E1])
    scaling = IVPPMLStandardization.fit(X1, I1, w1) if config.standardize else IVPPMLStandardization.identity(X1.shape[1], I1.shape[1])
    Xs, Is = scaling.transform(X1, I1)
    kc = C1.shape[1]
    Cs, Es = Xs[:, :kc], Xs[:, kc:]

    solver_plan, canonicalization = p1.for_engine(config.engine)
    projector = pre_projector if solver_plan is p1 else get_projector(
        solver_plan, engine=config.engine, method="map", execution=execution,
        context=exec_context, cache=bool(np.all(keep_sep)),
    )
    initial_eta = None
    if warm_start is not None:
        warm_eta = np.asarray(getattr(warm_start, "eta", np.empty(0)), dtype=np.float64)
        if len(warm_eta) == n:
            candidate = warm_eta[sample]
            if np.all(np.isfinite(candidate)):
                initial_eta = candidate

    state = fit_iv_irls(
        y1, Cs, Es, Is, solver_plan, w1, off1, config,
        projector=projector, initial_eta=initial_eta, detect_mu="mu" in config.separation_methods,
    )
    if not state.converged:
        raise ConvergenceError(
            f"IV-PPML IRLS failed to converge after {state.iterations} iterations "
            f"(eps={state.epsilon:.3e})",
            code="convergence.ivppml",
            details={"iterations": int(state.iterations), "epsilon": float(state.epsilon)},
        )
    # Upstream mu-separation does not trim/refit the sample.  It keeps the
    # rows in the IRLS system with near-zero mu/weights and reports the mask.
    # Preserve that distinction: ``sample`` is unchanged while ``separation``
    # records the observations detected by the mu criterion.
    num_sep_mu = int(np.sum(state.separated))
    if num_sep_mu:
        sample_idx = np.flatnonzero(sample)
        separation[sample_idx[state.separated]] = True

    # Final projection supplies inference at the accepted state, in solver units.
    irls_w, z, weighted_residual = final_working_system(state, y1, w1, off1)
    block = np.empty((len(y1), 1 + Xs.shape[1] + Is.shape[1]), dtype=np.float64)
    block[:, 0] = z
    block[:, 1:1 + Xs.shape[1]] = Xs
    if Is.shape[1]:
        block[:, 1 + Xs.shape[1]:] = Is
    block, _ = projector.residualize(block, irls_w, tol=config.target_inner_tol, return_info=True, copy=False)
    z_dm = block[:, 0]
    X_dm = block[:, 1:1 + Xs.shape[1]]
    if kc and Is.shape[1]:
        Z_dm = np.column_stack([X_dm[:, :kc], block[:, 1 + Xs.shape[1]:]])
    elif kc:
        Z_dm = X_dm[:, :kc]
    else:
        Z_dm = block[:, 1 + Xs.shape[1]:]
    solved = weighted_2sls(z_dm, X_dm, Z_dm, weights=irls_w)
    # An extra 2SLS step must not overwrite beta while eta/mu stay at the
    # preceding iterate. The jointly accepted IRLS state is the estimate.
    beta = scaling.unscale_coef(state.beta)

    cls = poisson_clusters(clusters, sample, vce)
    dof = p1.dof_info(clusters=cls, method=config.dof_method)
    effective_n = float(np.sum(w1)) if weight_kind == "fweight" else float(len(y1))
    vce_kw = dict(kind=vce, clusters=cls, effective_n=effective_n, base_weights=w1, weight_kind=weight_kind)
    overidentified = Z_dm.shape[1] > X_dm.shape[1]
    if misspecification_robust and overidentified:
        Vstd = ivppml_ee_vcov(
            solved.projected_X, X_dm, Z_dm, irls_w, weighted_residual, projector,
            tol=config.target_inner_tol, **vce_kw,
        )
    else:
        Vstd = ivppml_vcov(
            solved.projected_X, X_dm, weighted_residual / irls_w,
            irls_w, bread=solved.bread, **vce_kw,
        )
    # Hansen J on the additive moments Z_dm' w0 (y - mu), as for linear IV.
    overid = overid_test(
        weighted_residual, Z_dm, X=irls_w[:, None] * X_dm, kind=normalize_ivppml_vce(vce),
        clusters=cls, effective_n=effective_n,
        score_scale=frequency_score_scale(weight_kind, w1, normalize_ivppml_vce(vce)),
    )
    Xraw = np.column_stack([C1, E1])
    V, se = poisson_inference(
        scaling.unscale_vcov(Vstd), kind=normalize_ivppml_vce(vce), clusters=cls,
        scale=lambda: regressor_scale(
            Xraw, frequency_weights=w1 if config.sample_policy.psd_frequency_scale(weight_kind) else None),
    )

    mu, eta, deviance = in_user_units(state)
    # Upstream posts a normalized constant with zero reported variance.
    b_cons = float(np.average(eta - off1, weights=irls_w) - np.average(Xraw, axis=0, weights=irls_w) @ beta)
    names = tuple(exog_names) + tuple(endog_names) + ("_cons",)
    coef = np.r_[beta, b_cons]
    Vfull = np.zeros((len(coef), len(coef)), dtype=np.float64)
    Vfull[:-1, :-1] = V
    sefull = np.r_[se, 0.0]

    ll = float(np.sum(w1 * (y1 * eta - mu - gammaln(y1 + 1.0))))
    residual_struct = y1 - mu
    Qraw = np.column_stack([C1, I1])
    moments = additive_moments(residual_struct, Qraw, weights=w1, normalize=True)
    full_mu = np.full(n, np.nan)
    full_eta = np.full(n, np.nan)
    full_mu[sample] = mu
    full_eta[sample] = eta

    sep_by_method = dict(sep.by_method)
    if num_sep_mu:
        sep_by_method["mu"] = num_sep_mu

    n_singletons = int(np.sum(initial.singletons)) + (int(np.sum(sep.singletons)) if config.sample_policy.disjoint_reasons else 0)
    return IVPPMLResult(
        coef, Vfull, sefull, names, True, state.iterations, deviance, ll,
        int(round(effective_n)), int(np.sum(separation)), sample, separation,
        full_mu, full_eta, int(dof.df_absorbed), tuple(exog_names), tuple(endog_names),
        tuple(instrument_names), moments,
        diagnostics={
            "engine": config.engine,
            "sample_policy": sample_policy_diagnostics(config, sample, w, weight_kind),
            "separation_by_method": sep_by_method,
            "separation_seconds": sep.timings,
            "separation_iterations": sep.iterations,
            "separation_solvers": sep.solvers,
            "projection_resources": None if projector is None else projector.resource_info,
            "execution": {
                "threads": execution.threads,
                "memory_budget_mb": execution.memory_budget_mb,
                "cache": execution.cache,
            },
            "num_sep_mu": num_sep_mu,
            "n_rows": int(np.sum(sample)),
            "weight_type": weight_kind,
            "sum_weights": float(np.sum(w1)),
            "n_singletons": n_singletons,
            "absorb_iterations": state.absorb_iterations,
            "epsilon": state.epsilon,
            "standardized": config.standardize,
            "canonicalization": canonicalization,
            "first_stage": scaling.unscale_first_stage(solved.first_stage, kc),
            "first_stage_units": "original",
            "vce_requested": vce,
            "overidentification": overid,
            "misspecification_robust": bool(misspecification_robust and overidentified),
            "rank": solved.rank,
            "warm_started": initial_eta is not None,
        },
        df_resid=poisson_df_resid(cls, effective_n, len(coef), int(dof.df_absorbed)),
        vce=normalize_ivppml_vce(vce),
        cluster_counts=_cluster_counts(cls),
        fe_names=fe_names_from_metadata(p1.metadata),
        weight_type=weight_kind,
        sum_weights=float(np.sum(w1)),
        n_singletons=n_singletons,
    )
