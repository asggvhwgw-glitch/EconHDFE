from __future__ import annotations
from ...planner.scheduler import managed_execution

"""Internal heterogeneous-specification execution path for IV-PPML.

The path is deliberately narrow: common factor/interaction-rich DataFrame
specifications with exact row partitions and ordinary intercept FEs.  Anything
outside that certified contract falls back to the established dense estimator.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.special import gammaln

from ...collinearity import resolve_collinearity
from ...compute.block_design import BlockDesign, align_dense_to_blocks, hstack_block_designs, same_row_partition
from ...compute.vcov import block_cluster_subset_meats_xe, fix_psd
from ...compute.weights import prepare_weights
from ...compute.clusters import normalize_vce_clusters
from ...compute.stable_linalg import equilibrated_gram_inverse
from .vce import normalize_ivppml_vce
from ...errors import (
    CollinearityError, ConvergenceError, DivergenceError, InputError,
    NumericalError, ShapeError, UnderidentifiedError,
)
from ...hdfe.block_projection import BlockWeightedFEProjector
from ...hdfe.plan import FEPlan
from ...iv.solve import weighted_2sls_block
from ...reporting import cluster_counts as _cluster_counts, fe_names_from_metadata
from ..poisson import exp_clip_into, poisson_deviance, prepare_poisson_base, working_state
from ..ppml.execution import resolve_execution
from ..ppml.heterogeneous import detect_separation_block
from ..ppml.separation import initial_sample_mask
from .config import IVPPMLConfig
from .results import IVPPMLResult
from .standardize import IVPPMLStandardization


@dataclass(slots=True)
class BlockIVIRLSState:
    beta: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    converged: bool
    iterations: int
    deviance: float
    epsilon: float
    absorb_iterations: int
    separated: np.ndarray
    residual: np.ndarray
    irls_w: np.ndarray


def _slice_roles(design: BlockDesign, kc: int, ke: int):
    C = design.select_columns(np.arange(kc, dtype=np.int64))
    E = design.select_columns(np.arange(kc, kc + ke, dtype=np.int64))
    I = design.select_columns(np.arange(kc + ke, design.ncols, dtype=np.int64))
    return C, E, I


def _prune_iv_columns_block(C, E, I, names_c, names_e, names_i, plan, weights, config, execution):
    """Post-FE rank pruning while preserving included-exogenous instrument roles."""
    combined = hstack_block_designs(C, E, I)
    projector = BlockWeightedFEProjector.from_design(
        combined, plan, engine=config.engine, method=("within" if config.engine == "within" else "map"),
        absorb_threads=execution.threads,
        projection_memory_budget_mb=execution.memory_budget_mb,
    )
    within = projector.residualize_design(combined, weights, tol=config.target_inner_tol)
    Cw, Ew, Iw = _slice_roles(within, C.ncols, E.ncols)
    Xw = hstack_block_designs(Cw, Ew)
    Xraw = hstack_block_designs(C, E)
    xnames = tuple(names_c) + tuple(names_e)
    xplan = resolve_collinearity(
        Xw, names=xnames, original=Xraw, tolerance=max(config.target_inner_tol, 1e-12)
    )
    keep_x = np.asarray(xplan.active_indices, dtype=np.int64)
    if not len(keep_x):
        raise CollinearityError("all IV-PPML regressors are collinear after FE absorption")
    kc0 = C.ncols
    keep_c = keep_x[keep_x < kc0]
    keep_e = keep_x[keep_x >= kc0] - kc0
    C = C.select_columns(keep_c)
    E = E.select_columns(keep_e)
    Cw = Cw.select_columns(keep_c)
    Iw0 = Iw
    names_c = tuple(np.asarray(names_c, dtype=object)[keep_c])
    names_e = tuple(np.asarray(names_e, dtype=object)[keep_e])
    if E.ncols == 0:
        raise UnderidentifiedError("all endogenous regressors are collinear after FE absorption")

    if I.ncols:
        iplan = resolve_collinearity(
            Iw0, names=tuple(names_i), original=I, tolerance=max(config.target_inner_tol, 1e-12),
            protected_basis=Cw, protected_names=names_c,
        )
        keep_i = np.asarray(iplan.active_indices, dtype=np.int64)
        I = I.select_columns(keep_i)
        names_i = tuple(np.asarray(names_i, dtype=object)[keep_i])
    if I.ncols < E.ncols:
        raise UnderidentifiedError("equation not identified after collinearity removal")
    return C, E, I, names_c, names_e, names_i


def _fit_iv_irls_block(
    y, C: BlockDesign, E: BlockDesign, I: BlockDesign, plan: FEPlan,
    true_w, offset, config: IVPPMLConfig, *, projector, initial_eta=None, detect_mu=False,
):
    y = np.asarray(y, dtype=np.float64)
    w0 = np.asarray(true_w, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    X = hstack_block_designs(C, E)
    combined = hstack_block_designs(C, E, I)
    n = len(y)
    if X.nobs != n or offset.shape != (n,) or w0.shape != (n,):
        raise ShapeError("structured IV-PPML arrays have incompatible observation counts")

    mu = np.empty(n, dtype=np.float64)
    if initial_eta is not None:
        eta = np.asarray(initial_eta, dtype=np.float64).copy()
        if eta.shape != y.shape or not np.all(np.isfinite(eta)):
            raise ShapeError("initial_eta must be finite with one value per observation")
        exp_clip_into(eta, mu)
    else:
        mean_y = float(np.average(y, weights=w0))
        mu[:] = 0.5 * (y + mean_y)
        np.maximum(mu, 0.05 * y, out=mu)
        np.maximum(mu, 1e-3, out=mu)
        # mu guesses the TOTAL mean. Offset enters the working response;
        # adding it again here would violate mu == exp(eta).
        eta = np.log(mu)

    target_tol = max(1e-12, min(config.target_inner_tol, 0.1 * config.tolerance))
    hdfe_tol = max(config.start_inner_tol, config.tolerance)
    alt_tol = config.start_inner_tol
    irls_w = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    eta_candidate = np.empty(n, dtype=np.float64)
    mu_candidate = np.empty(n, dtype=np.float64)
    beta = np.zeros(X.ncols, dtype=np.float64)
    anchor_eta = eta.copy()
    in_step_halving = False
    previous_deviance = None
    deviance = math.inf
    epsilon = math.inf
    ok = 0
    step_halving = 0
    absorb_iterations = 0
    separated = np.zeros(n, dtype=bool)
    final_resid = None

    for it in range(1, config.max_iter + 1):
        if not in_step_halving:
            anchor_eta[:] = eta
        if not np.all(np.isfinite(mu)) or not np.isfinite(np.sum(mu, dtype=np.float64)):
            raise DivergenceError(
                f"IV-PPML IRLS produced non-finite mu at iteration {it}",
                code="numerical.non_finite_mu", details={"iteration": int(it)},
            )
        if it > 10 and (np.max(np.abs(X.matvec(beta)), initial=0.0) > 1e6 or np.max(np.abs(eta), initial=0.0) > 30):
            raise DivergenceError(f"IV-PPML IRLS diverged at iteration {it}", details={"iteration": int(it)})

        working_state(y, mu, eta, offset, w0, irls_w, z)
        np.maximum(irls_w, 1e-20, out=irls_w)
        projected = projector.residualize_response_design(z, combined, irls_w, tol=hdfe_tol)
        absorb_iterations += int(projected.iterations)
        if not projected.converged:
            raise ConvergenceError("structured IV-PPML FE projection failed to converge", code="convergence.ivppml_hdfe")
        Cdm, Edm, Idm = _slice_roles(projected.design, C.ncols, E.ncols)
        Xdm = hstack_block_designs(Cdm, Edm)
        Zdm = hstack_block_designs(Cdm, Idm)
        solved = weighted_2sls_block(projected.response, Xdm, Zdm, weights=irls_w)
        beta = solved.beta
        np.subtract(z, solved.residuals, out=eta_candidate)
        eta_candidate += offset

        if detect_mu and it > 1:
            threshold = math.log(config.mu_tol)
            if np.any(y > 0):
                threshold += min(float(np.min(eta_candidate[y > 0])) + 5.0, 0.0)
            separated |= (y == 0) & (eta_candidate <= threshold)

        eta_candidate[:] = np.clip(eta_candidate, -745.0, 709.0)
        exp_clip_into(eta_candidate, mu_candidate)
        if np.any(separated):
            mu_candidate[separated] = np.finfo(float).eps * 100.0
        dev_candidate = poisson_deviance(y, mu_candidate, w0, eta_candidate)
        if not np.isfinite(dev_candidate):
            raise NumericalError(
                f"IV-PPML IRLS produced non-finite deviance at iteration {it}",
                code="numerical.non_finite_deviance", details={"iteration": int(it)},
            )

        if previous_deviance is not None:
            delta = previous_deviance - dev_candidate
            if dev_candidate < 0.1 * delta:
                delta = dev_candidate
            denom = max(min(dev_candidate, previous_deviance), 0.1)
            epsilon = abs(delta) / denom
            if epsilon < config.tolerance and (len(plan.groups) <= 1 or hdfe_tol <= 1.1 * target_tol):
                ok += 1
            elif delta < 0 and step_halving < config.max_step_halving:
                eta_candidate[:] = (
                    config.step_halving_memory * anchor_eta
                    + (1.0 - config.step_halving_memory) * eta_candidate
                )
                if step_halving > 0:
                    np.maximum(eta_candidate, -10.0, out=eta_candidate)
                exp_clip_into(eta_candidate, mu_candidate)
                eta, eta_candidate = eta_candidate, eta
                mu, mu_candidate = mu_candidate, mu
                step_halving += 1
                in_step_halving = True
                ok = 0
                continue
            else:
                step_halving = 0
                in_step_halving = False
                ok = 0

        eta, eta_candidate = eta_candidate, eta
        mu, mu_candidate = mu_candidate, mu
        previous_deviance = dev_candidate
        deviance = dev_candidate
        final_resid = solved.residuals
        if ok >= config.min_ok:
            return BlockIVIRLSState(
                beta, mu, eta, True, it, deviance, float(epsilon), absorb_iterations,
                separated.copy(), final_resid, irls_w.copy(),
            )
        if np.isfinite(epsilon) and epsilon < hdfe_tol:
            hdfe_tol = max(min(0.1 * hdfe_tol, alt_tol), target_tol)
            scaled = max(0.1 * epsilon, np.finfo(float).eps)
            alt_tol = 10.0 ** (-math.ceil(math.log10(1.0 / scaled)))

    if final_resid is None:
        raise ConvergenceError("IV-PPML IRLS did not complete an iteration", code="convergence.ivppml_no_iteration")
    return BlockIVIRLSState(
        beta, mu, eta, False, config.max_iter, deviance, float(epsilon), absorb_iterations,
        separated.copy(), final_resid, irls_w.copy(),
    )


def _ivppml_vcov_block(
    X: BlockDesign, Z: BlockDesign, first_stage, residual, irls_w, *, kind,
    clusters, effective_n, base_weights, weight_kind, bread=None,
):
    gamma = np.asarray(first_stage, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    irls_w = np.asarray(irls_w, dtype=np.float64)
    A = gamma.T @ Z.cross_gram(X, weights=irls_w)
    bread = equilibrated_gram_inverse((A + A.T) / 2.0) if bread is None else np.asarray(bread, dtype=np.float64)
    kind0 = normalize_ivppml_vce(kind)
    score = irls_w * residual

    if kind0 in {"robust", "unadjusted", "iid", "model", "homoskedastic"}:
        robust_score = score
        if str(weight_kind).lower() == "fweight":
            bw = np.asarray(base_weights, dtype=np.float64)
            robust_score = (irls_w / np.sqrt(bw)) * residual
        Sz = Z.gram(weights=robust_score * robust_score)
        if effective_n > 1:
            Sz *= float(effective_n) / (float(effective_n) - 1.0)
        meat = gamma.T @ Sz @ gamma
        V = bread @ meat @ bread
        return (V + V.T) / 2.0
    if kind0 != "cluster":
        raise ValueError("structured IV-PPML supports robust or cluster VCE")
    clusters = normalize_vce_clusters(clusters, nobs=len(residual))
    V = np.zeros_like(bread)
    for sign, Sz, g, _ in block_cluster_subset_meats_xe(Z, score, clusters):
        meat = gamma.T @ Sz @ gamma
        V += sign * (g / (g - 1.0)) * (bread @ meat @ bread)
    V = (V + V.T) / 2.0
    return fix_psd(V) if len(clusters) > 1 else V


@managed_execution
def fit_block_arrays(
    y, C: BlockDesign, E: BlockDesign, I: BlockDesign, plan: FEPlan, *, offset,
    true_w, weight_kind="none", vce="robust", clusters=None, exog_names=(),
    endog_names=(), instrument_names=(), config: IVPPMLConfig, warm_start=None,
    execution_config=None,
):
    execution, _ = resolve_execution(execution_config, None)
    if not plan.groups:
        raise ValueError("structured IV-PPML requires absorbed FE")
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(true_w, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    n = len(y)
    if any(A.nobs != n for A in (C, E, I)) or w.shape != (n,) or off.shape != (n,):
        raise ShapeError("structured IV-PPML inputs have incompatible observation counts")

    singleton, n_fe_separated = initial_sample_mask(y, plan, config)
    active0 = ~singleton
    if not np.any(active0):
        raise InputError(
            "no observations remain after recursive singleton removal",
            code="input.empty_after_singletons", stage="sample",
            details={"nobs_raw": int(n), "n_singletons": int(np.sum(singleton)), "reason": "all_rows_pruned_as_singletons"},
            suggestion="Simplify the FE specification or retain levels with repeated support; no FE coefficient can be recovered from an empty sample.",
        )
    if np.all(active0):
        y0, C0, E0, I0, w0, off0, p0 = y, C, E, I, w, off, plan
    else:
        y0, w0, off0 = y[active0], w[active0], off[active0]
        C0, E0, I0 = C.subset_rows(active0), E.subset_rows(active0), I.subset_rows(active0)
        p0 = plan.subset(active0)

    C0, E0, I0, exog_names, endog_names, instrument_names = _prune_iv_columns_block(
        C0, E0, I0, exog_names, endog_names, instrument_names, p0, w0, config, execution
    )
    X0 = hstack_block_designs(C0, E0)
    sep = detect_separation_block(
        y0, X0, p0, w0, config, execution, frequency_weights=(weight_kind == "fweight"),
    )
    if "fe" in sep.by_method:
        sep.by_method["fe"] = n_fe_separated
    keep_sep = ~(sep.separated | sep.singletons)
    if not np.any(keep_sep) or not np.any(y0[keep_sep] > 0):
        raise InputError(
            "PPML has no finite estimable sample after separation",
            code="identification.no_finite_estimate", stage="separation",
            details={
                "nobs_after_singletons": int(len(y0)),
                "n_separated": int(np.sum(sep.separated)),
                "separation_by_method": dict(sep.by_method),
                "reason": "ppml_separation_no_finite_mle",
            },
            suggestion="Inspect all-zero FE levels and separated support; changing normalization cannot create a finite PPML estimate.",
        )

    separation = np.zeros(n, dtype=bool)
    if not np.all(keep_sep):
        idx0 = np.flatnonzero(active0)
        sample = np.zeros(n, dtype=bool)
        sample[idx0[keep_sep]] = True
        separation[idx0[~keep_sep]] = True
        y1, w1, off1 = y0[keep_sep], w0[keep_sep], off0[keep_sep]
        C1, E1, I1 = C0.subset_rows(keep_sep), E0.subset_rows(keep_sep), I0.subset_rows(keep_sep)
        p1 = p0.subset(keep_sep)
        C1, E1, I1, exog_names, endog_names, instrument_names = _prune_iv_columns_block(
            C1, E1, I1, exog_names, endog_names, instrument_names, p1, w1, config, execution
        )
    else:
        sample = active0.copy()
        y1, C1, E1, I1, w1, off1, p1 = y0, C0, E0, I0, w0, off0, p0

    X1 = hstack_block_designs(C1, E1)
    scaling = IVPPMLStandardization.fit(X1, I1, w1) if config.standardize else IVPPMLStandardization.identity(X1.ncols, I1.ncols)
    Xs, Is = scaling.transform(X1, I1)
    kc = C1.ncols
    Cs = Xs.select_columns(np.arange(kc, dtype=np.int64))
    Es = Xs.select_columns(np.arange(kc, Xs.ncols, dtype=np.int64))
    combined = hstack_block_designs(Cs, Es, Is)
    projector = BlockWeightedFEProjector.from_design(
        combined, p1, engine=config.engine, method=("within" if config.engine == "within" else "map"),
        absorb_threads=execution.threads,
        projection_memory_budget_mb=execution.memory_budget_mb,
    )
    initial_eta = None
    if warm_start is not None:
        warm_eta = np.asarray(getattr(warm_start, "eta", np.empty(0)), dtype=np.float64)
        if len(warm_eta) == n:
            candidate = warm_eta[sample]
            if np.all(np.isfinite(candidate)):
                initial_eta = candidate

    state = _fit_iv_irls_block(
        y1, Cs, Es, Is, p1, w1, off1, config, projector=projector,
        initial_eta=initial_eta, detect_mu="mu" in config.separation,
    )
    if not state.converged:
        raise ConvergenceError(
            f"IV-PPML IRLS failed to converge after {state.iterations} iterations (eps={state.epsilon:.3e})",
            code="convergence.ivppml", details={"iterations": int(state.iterations), "epsilon": float(state.epsilon)},
        )
    num_sep_mu = int(np.sum(state.separated))
    if num_sep_mu:
        sample_idx = np.flatnonzero(sample)
        separation[sample_idx[state.separated]] = True

    irls_w = np.maximum(w1 * state.mu, 1e-20)
    z = state.eta - off1 - 1.0
    pos = y1 > 0
    z[pos] += y1[pos] / state.mu[pos]
    projected = projector.residualize_response_design(z, combined, irls_w, tol=config.target_inner_tol)
    Cdm, Edm, Idm = _slice_roles(projected.design, Cs.ncols, Es.ncols)
    Xdm = hstack_block_designs(Cdm, Edm)
    Zdm = hstack_block_designs(Cdm, Idm)
    solved = weighted_2sls_block(projected.response, Xdm, Zdm, weights=irls_w)
    beta = scaling.unscale_coef(solved.beta)

    cls = None
    if clusters is not None:
        raw = clusters if isinstance(clusters, (list, tuple)) else [clusters]
        cls = [np.asarray(c)[sample] for c in raw]
    if str(vce).lower() == "cluster":
        cls = normalize_vce_clusters(cls, nobs=len(y1))
    dof = p1.dof_info(clusters=cls, method=config.dof_method)
    effective_n = float(np.sum(w1)) if weight_kind == "fweight" else float(len(y1))
    Vstd = _ivppml_vcov_block(
        Xdm, Zdm, solved.first_stage, solved.residuals, irls_w, kind=vce, clusters=cls,
        effective_n=effective_n, base_weights=w1, weight_kind=weight_kind, bread=solved.bread,
    )
    V = scaling.unscale_vcov(Vstd)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    Xraw = hstack_block_designs(C1, E1)
    mean_x = Xraw.t_matvec(irls_w) / float(np.sum(irls_w))
    b_cons = float(np.average(state.eta - off1, weights=irls_w) - mean_x @ beta)
    names = tuple(exog_names) + tuple(endog_names) + ("_cons",)
    coef = np.r_[beta, b_cons]
    Vfull = np.zeros((len(coef), len(coef)), dtype=np.float64)
    Vfull[:-1, :-1] = V
    sefull = np.r_[se, 0.0]

    ll = float(np.sum(w1 * (y1 * state.eta - state.mu - gammaln(y1 + 1.0))))
    residual_struct = y1 - state.mu
    Qraw = hstack_block_designs(C1, I1)
    moments = Qraw.t_matvec(w1 * residual_struct) / len(y1)
    full_mu = np.full(n, np.nan); full_eta = np.full(n, np.nan)
    full_mu[sample] = state.mu; full_eta[sample] = state.eta
    sep_by_method = dict(sep.by_method)
    if num_sep_mu:
        sep_by_method["mu"] = num_sep_mu

    return IVPPMLResult(
        coef, Vfull, sefull, names, True, state.iterations, state.deviance, ll,
        int(round(effective_n)), int(np.sum(separation)), sample, separation,
        full_mu, full_eta, int(dof.df_absorbed), tuple(exog_names), tuple(endog_names),
        tuple(instrument_names), moments,
        diagnostics={
            "engine": config.engine,
            "separation_by_method": sep_by_method,
            "separation_seconds": sep.timings,
            "separation_iterations": sep.iterations,
            "separation_solvers": sep.solvers,
            "projection_resources": projector.resource_info,
            "num_sep_mu": num_sep_mu,
            "n_rows": int(np.sum(sample)),
            "weight_type": weight_kind,
            "sum_weights": float(np.sum(w1)),
            "n_singletons": int(np.sum(singleton)),
            "absorb_iterations": state.absorb_iterations,
            "epsilon": state.epsilon,
            "standardized": config.standardize,
            "first_stage": scaling.unscale_first_stage(solved.first_stage, kc),
            "first_stage_units": "original",
            "vce_requested": vce,
            "rank": solved.rank,
            "warm_started": initial_eta is not None,
            "heterogeneous_spec_path": True,
        },
        df_resid=(min([max(len(np.unique(c)) - 1, 0) for c in cls]) if cls else max(effective_n - len(coef) - int(dof.df_absorbed), 0.0)),
        vce=normalize_ivppml_vce(vce), cluster_counts=_cluster_counts(cls),
        fe_names=fe_names_from_metadata(p1.metadata), weight_type=weight_kind,
        sum_weights=float(np.sum(w1)), n_singletons=int(np.sum(singleton)),
    )



def fit_structured_dataframe(
    data, *, y, exog, endog, instruments, plan: FEPlan, offset=None, exposure=None,
    weights=None, weight_type=None, vce="robust", clusters=None,
    config: IVPPMLConfig | None = None, execution_config=None, warm_start=None,
):
    """Compile common heterogeneous IV-PPML specs before global dense materialization.

    Returns ``None`` when the roles do not admit one common certified row
    partition, allowing the caller to use the established dense frontend.
    """
    from ...design import _compile_execution_design

    cfg = IVPPMLConfig() if config is None else config
    cfg.validate()
    yraw = np.asarray(data[y] if isinstance(y, str) else y)

    def col_or_value(v):
        if v is None:
            return None
        return np.asarray(data[v]) if isinstance(v, str) else np.asarray(v)

    yv, off, base_w = prepare_poisson_base(
        yraw, offset=col_or_value(offset), exposure=col_or_value(exposure),
        weights=col_or_value(weights),
    )
    winfo = prepare_weights(col_or_value(weights), len(yv), weight_type, normalize_ap=False)
    if winfo.kind == "aweight":
        return None
    w = np.ones(len(yv), dtype=np.float64) if winfo.estimation is None else winfo.estimation
    memory_budget = getattr(execution_config, "memory_budget_mb", 512.0)
    c = _compile_execution_design(
        data, exog, len(yv), groups=plan.groups, expected_passes=2,
        memory_budget_mb=memory_budget, structural=True,
    )
    e = _compile_execution_design(
        data, endog, len(yv), groups=plan.groups, expected_passes=2,
        memory_budget_mb=memory_budget, structural=True, protected_terms=c.structural_terms,
    )
    i = _compile_execution_design(
        data, instruments, len(yv), groups=plan.groups, expected_passes=2,
        memory_budget_mb=memory_budget, structural=True, protected_terms=c.structural_terms,
    )
    if e.ncols == 0 or i.ncols < e.ncols:
        return None
    vals = (c.values, e.values, i.values)
    blocks = [A for A in vals if isinstance(A, BlockDesign)]
    if not blocks or not plan.groups:
        return None
    template = blocks[0]
    if not same_row_partition(*blocks):
        return None
    C = c.values if isinstance(c.values, BlockDesign) else align_dense_to_blocks(c.values, template)
    E = e.values if isinstance(e.values, BlockDesign) else align_dense_to_blocks(e.values, template)
    I = i.values if isinstance(i.values, BlockDesign) else align_dense_to_blocks(i.values, template)
    Xcheck = hstack_block_designs(C, E)
    if (not Xcheck.columns_disjoint) and any(m in {"simplex", "relu"} for m in cfg.separation):
        return None

    cls = None
    if clusters is not None:
        raw = clusters if isinstance(clusters, (list, tuple)) else [clusters]
        cls = [np.asarray(data[v]) if isinstance(v, str) else np.asarray(v) for v in raw]
    result = fit_block_arrays(
        yv, C, E, I, plan, offset=off, true_w=w, weight_kind=winfo.kind,
        vce=vce, clusters=cls, exog_names=c.names, endog_names=e.names,
        instrument_names=i.names, config=cfg, warm_start=warm_start,
        execution_config=execution_config,
    )
    return result
