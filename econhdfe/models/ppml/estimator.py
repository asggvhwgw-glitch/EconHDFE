from __future__ import annotations
from ...planner.scheduler import managed_execution
from ...compute.clusters import normalize_vce_clusters
import math
import numpy as np
from ...errors import InputError, ConvergenceError
from ...config import ExecutionConfig
from ...compute.context import ExecutionContext
from scipy.special import gammaln
from scipy.stats import chi2 as chi2_dist
from .config import PPMLConfig
from ...hdfe.plan import FEPlan
from .irls import fit_irls
from ...compute.linalg import independent_columns, weighted_lstsq
from .results import PPMLResult
from .separation import detect_separation, initial_sample_mask
from .standardize import Standardization
from ...compute.vcov import normalize_vce
from .vce import ppml_vcov, MODEL_VCE_KINDS
from .execution import resolve_execution, get_projector
from ...reporting import cluster_counts as _cluster_counts, fe_names_from_metadata


def _filter_columns(X, names, plan: FEPlan, weights, tol, *, engine="replica", projector=None):
    """Remove raw and post-absorption collinearity; return original-column indices."""
    X = np.asarray(X, dtype=np.float64)
    names = np.asarray(names, dtype=object)
    keep, _ = independent_columns(X, tol=1e-12)
    X, names = X[:, keep], names[keep]
    original = keep
    if plan.groups and X.shape[1]:
        if projector is None:
            residual, _ = plan.residualize(X, weights, tol=tol, engine=engine)
        else:
            residual, _ = projector.residualize(X, weights, tol=tol, return_info=True)
        keep2, _ = independent_columns(residual, tol=max(tol, 1e-12))
        X, names = X[:, keep2], names[keep2]
        original = original[keep2]
    return X, tuple(names), original


def _ppml_wald(coef, vcov, names):
    idx = [i for i, name in enumerate(names) if str(name) != "_cons"]
    q = len(idx)
    if q == 0:
        return float("nan"), float("nan"), 0
    b = np.asarray(coef, dtype=np.float64)[idx]
    V = np.asarray(vcov, dtype=np.float64)[np.ix_(idx, idx)]
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(V)) or np.linalg.matrix_rank(V) < q:
        return float("nan"), float("nan"), q
    try:
        stat = float(b @ np.linalg.solve(V, b))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), q
    return stat, float(chi2_dist.sf(stat, q)), q


@managed_execution
def fit_arrays(
    y,
    X,
    plan: FEPlan,
    *,
    offset,
    true_w,
    vce,
    clusters,
    names,
    config: PPMLConfig,
    warm_start=None,
    execution_config: ExecutionConfig | None = None,
    execution_context: ExecutionContext | None = None,
) -> PPMLResult:
    """Estimator core on validated dense arrays and a precompiled FE plan."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    true_w = np.asarray(true_w, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    n = len(y)
    execution, exec_context = resolve_execution(execution_config, execution_context)

    if not plan.groups:
        X = np.column_stack([X, np.ones(n)])
        names = tuple(names) + ("_cons",)

    # ppmlhdfe first stage: singletons (row counts) and, with "fe", FE
    # separation pruned jointly; e(num_singletons) counts both.
    singleton, n_fe_separated = initial_sample_mask(y, plan, config)
    has_singletons = bool(np.any(singleton))
    if has_singletons:
        active0 = ~singleton
        if not np.any(active0):
            raise InputError(
            "no observations remain after recursive singleton removal",
            code="input.empty_after_singletons", stage="sample",
            details={"nobs_raw": int(n), "n_singletons": int(np.sum(singleton)), "reason": "all_rows_pruned_as_singletons"},
            suggestion="Simplify the FE specification or retain levels with repeated support; no FE coefficient can be recovered from an empty sample.",
        )
        y0, X0, w0, off0 = y[active0], X[active0], true_w[active0], off[active0]
        p0 = plan.subset(active0) if plan.groups else plan
    else:
        active0 = np.ones(n, dtype=bool)
        y0, X0, w0, off0, p0 = y, X, true_w, off, plan

    standardizer = (
        Standardization.fit(y0, X0)
        if config.standardize
        else Standardization(np.ones(X0.shape[1]), 1.0)
    )
    y0s, X0s = standardizer.transform(y0, X0)

    # The initial rank check and, when the sample/FE plan is unchanged, IRLS
    # share the same compiled projector topology.
    pre_projector = get_projector(
        p0, engine=config.engine, method=config.absorb_method, execution=execution,
        context=exec_context, cache=True,
    ) if p0.groups else None
    X0s, names, keep = _filter_columns(
        X0s, names, p0, w0, config.target_inner_tol,
        engine=config.engine, projector=pre_projector,
    )
    standardizer = standardizer.select_columns(keep)

    sep0 = detect_separation(y0s, X0s, p0, w0, config, execution)
    if "fe" in sep0.by_method:
        sep0.by_method["fe"] = n_fe_separated
    survivor0 = ~(sep0.separated | sep0.singletons)
    any_sep = not bool(np.all(survivor0))
    if not np.any(survivor0) or not np.any(y0s[survivor0] > 0):
        raise InputError(
            "PPML has no finite estimable sample after separation",
            code="identification.no_finite_estimate", stage="separation",
            details={
                "nobs_after_singletons": int(len(y0s)),
                "n_separated": int(np.sum(sep0.separated)),
                "separation_by_method": dict(sep0.by_method),
                "reason": "ppml_separation_no_finite_mle",
            },
            suggestion="Inspect all-zero FE levels and separated support; changing normalization cannot create a finite PPML estimate.",
        )

    separation = np.zeros(n, dtype=bool)
    if not any_sep:
        sample = active0.copy() if has_singletons else np.ones(n, dtype=bool)
        ystd, Xstd = y0s, X0s
        ys_orig, ws, offs = y0, w0, off0
        ps_requested = p0
    else:
        idx0 = np.flatnonzero(active0)
        sample = np.zeros(n, dtype=bool)
        sample[idx0[survivor0]] = True
        separation[idx0[sep0.separated]] = True
        ystd, Xstd = y0s[survivor0], X0s[survivor0]
        ys_orig, ws, offs = y0[survivor0], w0[survivor0], off0[survivor0]
        ps_requested = p0.subset(survivor0) if p0.groups else p0
        # Rebuild for the surviving topology under the original resource policy.
        pre_projector = get_projector(
            ps_requested, engine=config.engine, method=config.absorb_method, execution=execution,
            context=exec_context, cache=False,
        )
        # Only a changed estimation sample can change post-FE rank.
        Xstd, names, keep = _filter_columns(
            Xstd, names, ps_requested, ws, config.target_inner_tol,
            engine=config.engine, projector=pre_projector,
        )
        standardizer = standardizer.select_columns(keep)

    ps_solver, canonicalization = ps_requested.for_engine(config.engine)
    if ps_solver is ps_requested:
        projector = pre_projector
    else:
        projector = get_projector(
            ps_solver, engine=config.engine, method=config.absorb_method, execution=execution,
            context=exec_context, cache=not any_sep,
        ) if ps_solver.groups else None

    initial_eta = None
    if warm_start is not None:
        warm_eta = np.asarray(getattr(warm_start, "eta", np.empty(0)), dtype=np.float64)
        if len(warm_eta) == n:
            candidate = warm_eta[sample] - math.log(standardizer.y_scale)
            if np.all(np.isfinite(candidate)):
                initial_eta = candidate

    state = fit_irls(
        ystd, Xstd, ps_solver, ws, offs, config,
        detect_mu="mu" in config.separation, projector=projector,
        initial_eta=initial_eta,
    )
    if not state.converged:
        raise ConvergenceError(
            f"PPML IRLS failed to converge after {state.iterations} iterations "
            f"(eps={state.epsilon:.3e})",
            code="convergence.ppml",
            details={"iterations": int(state.iterations), "epsilon": float(state.epsilon)},
        )

    mu_sep = state.separated
    if np.any(mu_sep):
        current = np.flatnonzero(sample)
        separation[current[mu_sep]] = True
        keep_mu = ~mu_sep
        if ps_requested.groups:
            # ppmlhdfe trims mu-separated rows and reloads the HDFE, pruning
            # the singletons this creates (counted as singletons).
            keep_mu[np.flatnonzero(keep_mu)[ps_requested.subset(keep_mu).singleton_mask()]] = False
        sample[current[~keep_mu]] = False
        ystd, Xstd = ystd[keep_mu], Xstd[keep_mu]
        ys_orig, ws, offs = ys_orig[keep_mu], ws[keep_mu], offs[keep_mu]
        ps_requested = ps_requested.subset(keep_mu) if ps_requested.groups else ps_requested
        rank_projector = get_projector(
            ps_requested, engine=config.engine, method=config.absorb_method,
            execution=execution, context=exec_context, cache=False,
        ) if ps_requested.groups else None
        Xstd, names, keep = _filter_columns(
            Xstd, names, ps_requested, ws, config.target_inner_tol,
            engine=config.engine, projector=rank_projector,
        )
        standardizer = standardizer.select_columns(keep)
        ps_solver, canonicalization = ps_requested.for_engine(config.engine)
        projector = get_projector(
            ps_solver, engine=config.engine, method=config.absorb_method, execution=execution,
            context=exec_context, cache=False,
        ) if ps_solver.groups else None
        state = fit_irls(
            ystd, Xstd, ps_solver, ws, offs, config, detect_mu=False,
            projector=projector, initial_eta=state.eta[keep_mu],
        )
        if not state.converged:
            raise ConvergenceError(
                f"PPML IRLS failed after mu-separation trim "
                f"({state.iterations} iterations, eps={state.epsilon:.3e})",
                code="convergence.ppml_after_mu",
                details={"iterations": int(state.iterations), "epsilon": float(state.epsilon)},
            )
        sep0.by_method["mu"] = int(np.sum(mu_sep))

    # Final high-accuracy weighted solve. Reuse the IRLS block as workspace and
    # the same compiled FE projector whenever the sample did not change.
    wfinal = ws * state.mu
    z = state.eta - offs - 1.0
    pos = ystd > 0
    z[pos] += ystd[pos] / state.mu[pos]
    block = state.transformed
    if block is None or block.shape != (len(ystd), Xstd.shape[1] + 1):
        block = np.empty((len(ystd), Xstd.shape[1] + 1), dtype=np.float64)
    block[:, 0] = z
    if Xstd.shape[1]:
        block[:, 1:] = Xstd
    if projector is not None:
        block, _ = projector.residualize(
            block, wfinal, tol=config.target_inner_tol, return_info=True, copy=False
        )
    zt, Xt = block[:, 0], block[:, 1:]
    beta_std, _ = weighted_lstsq(Xt, zt, wfinal, fast=False)
    beta = standardizer.unscale_coef(beta_std)
    if names and names[-1] == "_cons":
        beta[-1] += math.log(standardizer.y_scale)

    cls_s = None
    if clusters is not None:
        cls_list = clusters if isinstance(clusters, (list, tuple)) else [clusters]
        cls_s = [np.asarray(c)[sample] for c in cls_list]
    if normalize_vce(vce) == "cluster":
        cls_s = normalize_vce_clusters(cls_s, nobs=len(ystd))
    dof = ps_requested.dof_info(clusters=cls_s, method=config.dof_method)
    df_a = int(dof.df_absorbed)
    Vstd = ppml_vcov(
        Xt, ystd, state.mu, ws,
        kind=vce, clusters=cls_s, df_absorbed=df_a,
        nested_adj=int(dof.nested > 0),
    )
    V = standardizer.unscale_vcov(Vstd)
    if normalize_vce(vce, "model") in MODEL_VCE_KINDS:
        # H_std = H_original / y_scale. Robust sandwich scaling cancels,
        # but the model-based inverse Hessian requires this additional factor.
        V /= standardizer.y_scale
    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    # The IRLS state is no longer needed in standardized units, so rescale in
    # place and avoid two additional N-vector copies on the common full-sample
    # path.
    state.mu *= standardizer.y_scale
    state.eta += math.log(standardizer.y_scale)
    mu_orig, eta_orig = state.mu, state.eta
    deviance = float(state.deviance * standardizer.y_scale)
    ll = float(np.sum(ws * (ys_orig * eta_orig - mu_orig - gammaln(ys_orig + 1.0))))
    sw = float(np.sum(ws))
    mu0 = float(np.sum(ws * ys_orig) / sw) if sw > 0 else float("nan")
    ll0 = (
        float(np.sum(ws * (ys_orig * math.log(mu0) - mu0 - gammaln(ys_orig + 1.0))))
        if np.isfinite(mu0) and mu0 > 0 else float("nan")
    )
    pseudo_r2 = (1.0 - ll / ll0) if np.isfinite(ll0) and ll0 != 0 else float("nan")
    wald_chi2, wald_p, df_m = _ppml_wald(beta, V, names)
    if bool(np.all(sample)):
        full_mu, full_eta = mu_orig, eta_orig
    else:
        full_mu = np.full(n, np.nan)
        full_eta = np.full(n, np.nan)
        full_mu[sample] = mu_orig
        full_eta[sample] = eta_orig

    return PPMLResult(
        params=beta, vcov=V, stderr=se, names=tuple(names), converged=True,
        iterations=state.iterations, deviance=deviance, loglike=ll,
        nobs=int(np.sum(sample)), n_separated=int(np.sum(separation)),
        sample_mask=sample, separation_mask=separation, mu=full_mu, eta=full_eta,
        df_absorbed=df_a, loglike_null=ll0, pseudo_r2=pseudo_r2,
        chi2=wald_chi2, chi2_pvalue=wald_p, df_model=df_m, nobs_full=int(n),
        vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        diagnostics={
            "engine": config.engine,
            "separation_by_method": sep0.by_method,
            "separation_seconds": sep0.timings,
            "separation_iterations": sep0.iterations,
            "separation_solvers": sep0.solvers,
            "projection_resources": None if projector is None else projector.resource_info,
            "execution": {
                "threads": execution.threads,
                "memory_budget_mb": execution.memory_budget_mb,
                "cache": execution.cache,
            },
            "n_singletons": int(n - np.sum(sample) - np.sum(separation)),
            "absorb_iterations": state.absorb_iterations,
            "epsilon": state.epsilon,
            "standardized": config.standardize,
            "canonicalization": canonicalization,
            "dof_nested": int(dof.nested),
            "warm_started": initial_eta is not None,
        },
        df_resid=(min([max(len(np.unique(c)) - 1, 0) for c in cls_s]) if cls_s else max(float(np.sum(sample)) - len(beta) - df_a, 0.0)),
        vce="model" if vce is None else str(vce),
        cluster_counts=_cluster_counts(cls_s),
        fe_names=fe_names_from_metadata(ps_requested.metadata),
        weight_type="none" if np.allclose(ws, 1.0) else "weight",
        sum_weights=float(np.sum(ws)),
        n_singletons=int(n - np.sum(sample) - np.sum(separation)),
    )
