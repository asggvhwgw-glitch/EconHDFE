from __future__ import annotations
from ...planner.scheduler import managed_execution
from ...compute.clusters import normalize_vce_clusters

"""Heterogeneous-specification PPML execution operators.

This module is intentionally internal. The DataFrame estimator may dispatch
here for certified interaction-rich specifications, while array APIs and
unsupported structures retain the established dense path. The outer PPML state
(mu, eta, deviance, convergence and mu-separation) remains global; only exact
physical-design, FE-projection and WLS opportunities are decomposed.
"""

from dataclasses import dataclass
from time import perf_counter
import math
import numpy as np

from ...compute.block_design import BlockDesign, DenseDesignBlock
from ...compute.linalg import weighted_lstsq, independent_columns
from ...compute.partitioned_lstsq import partitioned_weighted_lstsq
from ...compute.stable_linalg import equilibrated_gram_inverse
from ...compute.vcov import sandwich_vcov_block_xe, normalize_vce
from ...compute.weights import frequency_score_scale
from ...errors import ShapeError, ConvergenceError
from ...hdfe.block_projection import BlockWeightedFEProjector
from ...hdfe.plan import FEPlan
from ..poisson import (
    exp_clip_into, working_state, poisson_deviance, initial_active_rows, separation_survivors,
    poisson_clusters, poisson_inference, poisson_df_resid,
)
from .config import PPMLConfig
from .irls import IRLSState, _predict_eps
from .separation import SeparationResult, detect_separation, initial_sample_selection, sample_policy_diagnostics
from .standardize import Standardization
from .results import PPMLResult
from .execution import resolve_execution
from .vce import MODEL_VCE_KINDS, ppml_covariance_dof, ppml_covariance_diagnostics
from ...reporting import cluster_counts as _cluster_counts, fe_names_from_metadata
from scipy.special import gammaln


def _component_layout(design: BlockDesign, plan: FEPlan):
    design._ensure_layout()
    if plan.groups and plan.nobs != design.nobs:
        raise ValueError("FE plan and BlockDesign have different observation counts")
    seen = np.zeros(design.nobs, dtype=bool)
    for block in design.blocks:
        rows = np.asarray(block.rows, dtype=np.int64)
        if np.any(seen[rows]):
            raise ValueError("BlockDesign rows must form disjoint components")
        seen[rows] = True
        local_plan = plan.take(rows) if plan.groups else plan
        yield block, rows, local_plan
    if design.nobs and not np.all(seen):
        raise ValueError("heterogeneous-spec PPML requires every observation in one component")


def detect_separation_block(
    y,
    X: BlockDesign,
    plan: FEPlan,
    weights,
    config: PPMLConfig,
    execution_config=None,
    *,
    frequency_weights: bool = False,
) -> SeparationResult:
    """Exact component-wise wrapper around the established separation stack.

    The routine is valid only when ``X`` was compiled from a certified joint
    design/FE component structure.  Each local call therefore solves the same
    separation problem on one disconnected component; results are mapped back
    to the global observation order.  This function deliberately reuses the
    established FE/simplex/ReLU implementations rather than introducing new
    separation logic.
    """
    if (not X.columns_disjoint) and any(m in {"simplex", "relu"} for m in config.separation_methods):
        raise ValueError(
            "component separation is not exact when coefficients are shared across row components; "
            "use the dense separation path or separation methods limited to FE/mu"
        )
    y0 = np.asarray(y, dtype=np.float64)
    w0 = np.ones(len(y0), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if y0.ndim != 1 or len(y0) != X.nobs:
        raise ValueError("y must have one value per BlockDesign observation")
    if w0.ndim != 1 or len(w0) != X.nobs:
        raise ValueError("weights must have one value per BlockDesign observation")

    separated = np.zeros(X.nobs, dtype=bool)
    singletons = np.zeros(X.nobs, dtype=bool)
    counts = {m: 0 for m in config.separation_methods}
    timings = {m: 0.0 for m in config.separation_methods}
    iterations = {m: 0 for m in config.separation_methods}
    solver_sets = {m: set() for m in config.separation_methods}

    # Timing is accumulated by method across components.  Local detect_separation
    # already records stage-specific timings; the outer wall timer is retained
    # only as a defensive fallback for future implementations.
    wall0 = perf_counter()
    for block, rows, local_plan in _component_layout(X, plan):
        local = detect_separation(
            y0[rows], block.values, local_plan, w0[rows], config, execution_config,
            frequency_weights=frequency_weights,
        )
        separated[rows] = local.separated
        singletons[rows] = local.singletons
        for method in config.separation_methods:
            counts[method] += int(local.by_method.get(method, 0))
            timings[method] += float(local.timings.get(method, 0.0))
            iterations[method] = max(iterations[method], int(local.iterations.get(method, 0)))
            solver_sets[method].add(str(local.solvers.get(method, "unknown")))
    _ = perf_counter() - wall0

    solvers = {}
    for method in config.separation_methods:
        values = sorted(solver_sets[method])
        solvers[method] = values[0] if len(values) == 1 else "component[" + ",".join(values) + "]"
    return SeparationResult(separated, counts, timings, iterations, solvers, singletons)



def filter_columns_block(
    X: BlockDesign, names, plan: FEPlan, weights, tol, *, engine="replica", projector=None,
):
    """Block-native counterpart of the PPML raw/post-FE rank filter.

    Empty-column components are deliberately retained so nonlinear outcome/FE
    state is never discarded merely because no explicit coefficient survives.
    """
    names = np.asarray(names, dtype=object)
    if len(names) != X.ncols:
        raise ValueError("names must have one entry per BlockDesign column")
    keep, _ = independent_columns(X, tol=1e-12)
    out = X.select_columns(keep)
    names = names[keep]
    original = np.asarray(keep, dtype=np.int64)
    if plan.groups and out.ncols:
        if projector is None:
            projector = BlockWeightedFEProjector.from_design(out, plan, engine=engine, method="map")
        residual = projector.residualize_design(out, weights, tol=tol)
        keep2, _ = independent_columns(residual, tol=max(tol, 1e-12))
        out = out.select_columns(keep2)
        names = names[keep2]
        original = original[keep2]
    return out, tuple(names), original

def _block_weighted_solve(
    transformed_response: np.ndarray,
    transformed_design: BlockDesign,
    weights: np.ndarray,
    *,
    fast: bool,
    resid_out: np.ndarray,
) -> np.ndarray:
    """Solve block/partitioned WLS without materializing global N x K.

    Disjoint coefficient blocks keep the cheapest local solve.  Shared global
    columns require one coupled coefficient solution: intermediate IRLS steps
    may use the same normal-equation fast path as pooled PPML, while accurate
    iterations and the final WLS use the QR-compressed partitioned solver.
    """
    if not transformed_design.columns_disjoint:
        if fast:
            G = transformed_design.gram(weights=weights)
            rhs = transformed_design.t_matvec(weights * transformed_response)
            beta = np.linalg.pinv(G, hermitian=True) @ rhs
            resid_out[:] = transformed_response
            for block in transformed_design.blocks:
                if block.values.shape[1]:
                    resid_out[block.rows] -= block.values @ beta[block.columns]
            return np.asarray(beta, dtype=np.float64)
        beta, _, _ = partitioned_weighted_lstsq(
            transformed_design, transformed_response, weights, resid_out=resid_out
        )
        return beta

    beta = np.zeros(transformed_design.ncols, dtype=np.float64)
    resid_out.fill(0.0)
    seen = np.zeros(transformed_design.nobs, dtype=bool)
    for block in transformed_design.blocks:
        rows = block.rows
        seen[rows] = True
        if block.values.shape[1] == 0:
            # No explicit regressor remains in this component.  The FWL
            # residual is just the projected working response.
            resid_out[rows] = transformed_response[rows]
            continue
        local_resid = np.empty(len(rows), dtype=np.float64)
        b, _ = weighted_lstsq(
            block.values,
            transformed_response[rows],
            weights[rows],
            fast=fast,
            resid_out=local_resid,
        )
        beta[block.columns] = b
        resid_out[rows] = local_resid
    if transformed_design.nobs and not np.all(seen):
        raise ValueError("transformed BlockDesign does not cover all observations")
    return beta


def fit_irls_block(
    y,
    X: BlockDesign,
    plan: FEPlan,
    true_w,
    offset,
    config: PPMLConfig,
    *,
    structure,
    detect_mu: bool = False,
    projector: BlockWeightedFEProjector | None = None,
    initial_eta=None,
) -> IRLSState:
    """Exact heterogeneous-specification IRLS with global convergence state.

    This mirrors :func:`fit_irls` but decomposes only the weighted FWL/WLS
    operation.  Deviance, epsilon extrapolation, mu-separation and stopping
    rules remain pooled over all observations.  The first implementation
    intentionally does not use the dense fast-partial shortcut; establishing
    parity is more important than workspace reuse at this stage.
    """
    y = np.asarray(y, dtype=np.float64)
    w0 = np.asarray(true_w, dtype=np.float64)
    offset = np.zeros(len(y), dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    if y.ndim != 1 or len(y) != X.nobs:
        raise ShapeError("block PPML y must have one value per observation")
    if w0.shape != y.shape or offset.shape != y.shape:
        raise ShapeError("block PPML weights/offset must match y")
    n, k = X.shape

    if projector is None:
        projector = BlockWeightedFEProjector.from_design(
            X, plan, engine=config.engine, method="map"
        )

    mu = np.empty(n, dtype=np.float64)
    if initial_eta is not None:
        eta = np.asarray(initial_eta, dtype=np.float64).copy()
        if eta.shape != y.shape or not np.all(np.isfinite(eta)):
            raise ShapeError("initial_eta must be finite with one value per observation")
        exp_clip_into(eta, mu)
    else:
        ybar = float(np.average(y, weights=w0))
        mu[:] = 0.5 * (y + ybar)
        np.maximum(mu, 0.05 * y, out=mu)
        np.maximum(mu, 1e-3, out=mu)
        eta = np.log(mu)
    prev_dev = poisson_deviance(y, mu, w0, eta)

    highest_tol = max(1e-12, min(config.target_inner_tol, 0.1 * config.tolerance))
    hdfe_tol = max(config.start_inner_tol, config.tolerance)
    alt_tol = math.inf
    eps = None
    eps_history: list[float] = []
    transformed_design = None
    transformed_response = None
    ok = 0
    absorb_iters = 0
    beta = np.zeros(k, dtype=np.float64)
    separation_mask = np.zeros(n, dtype=bool)
    zero = y == 0
    positive = ~zero
    log_sep_tol = math.log(config.mu_tol)

    irls_w = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    z_prev = np.empty(n, dtype=np.float64)
    z_prev.fill(0.0)
    eta_next = np.empty(n, dtype=np.float64)
    mu_next = np.empty(n, dtype=np.float64)
    resid = np.empty(n, dtype=np.float64)

    for it in range(1, config.max_iter + 1):
        predicted_eps = _predict_eps(eps_history, eps)
        working_state(y, mu, eta, offset, w0, irls_w, z)

        fast_partial = config.fast_partial and it > 1 and transformed_design is not None
        if fast_partial:
            # Exact block analogue of the pooled fast-partial update: the
            # previous residualized design is a valid FWL starting point under
            # the new weights, while only the working-response delta changes.
            response_source = transformed_response + (z - z_prev)
            design_source = transformed_design
        else:
            response_source = z
            design_source = X
        projected = projector.residualize_response_design(
            response_source, design_source, irls_w, tol=hdfe_tol
        )
        transformed_response = projected.response
        transformed_design = projected.design
        absorb_iters += int(projected.iterations)

        fast = bool(
            config.fast_solver and k
            and hdfe_tol > config.tolerance * 11
            and predicted_eps > config.tolerance
        )
        beta = _block_weighted_solve(
            transformed_response, transformed_design, irls_w, fast=fast, resid_out=resid
        )
        np.subtract(z, resid, out=eta_next)
        eta_next += offset
        if detect_mu and np.any(zero):
            positive_eta = eta_next[positive]
            adjusted = log_sep_tol + min(float(np.min(positive_eta)) + 5.0, 0.0)
            separation_mask |= zero & (eta_next <= adjusted)
        exp_clip_into(eta_next, mu_next)
        if np.any(separation_mask):
            mu_next[separation_mask] = np.finfo(np.float64).eps * 100.0
        dev = poisson_deviance(y, mu_next, w0, eta_next)
        delta = prev_dev - dev
        if dev < 0.1 * delta:
            delta = dev
        denom = max(min(dev, prev_dev), 0.1)
        eps = abs(delta) / denom

        accurate_inner = (len(plan.groups) <= 1) or (hdfe_tol <= 1.1 * highest_tol)
        if eps < config.tolerance and (not fast) and accurate_inner:
            ok += 1
        else:
            ok = 0

        z_prev[:] = z
        eta, eta_next = eta_next, eta
        mu, mu_next = mu_next, mu
        prev_dev = dev
        if ok >= config.min_ok or (ok >= 1 and dev == 0.0):
            return IRLSState(
                beta, mu, eta, z, transformed_design, True, it, dev, float(eps),
                absorb_iters, separation_mask.copy(),
            )

        if eps < hdfe_tol:
            hdfe_tol = max(min(0.1 * hdfe_tol, alt_tol), highest_tol)
            scaled = max(0.1 * eps, np.finfo(float).eps)
            alt_tol = 10.0 ** (-math.ceil(math.log10(1.0 / scaled)))

    return IRLSState(
        beta, mu, eta, z_prev, transformed_design, False, config.max_iter,
        prev_dev, float(eps or np.inf), absorb_iters, separation_mask.copy(),
    )


def final_wls_block(
    y, X: BlockDesign, plan: FEPlan, true_w, offset, state: IRLSState, config: PPMLConfig,
    *, structure, projector: BlockWeightedFEProjector | None = None,
):
    """Final high-accuracy PPML WLS in block-native representation."""
    y = np.asarray(y, dtype=np.float64)
    tw = np.asarray(true_w, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    if projector is None:
        projector = BlockWeightedFEProjector.from_design(
            X, plan, engine=config.engine, method="map"
        )
    wfinal = tw * state.mu
    z = state.eta - off - 1.0
    pos = y > 0
    z = np.asarray(z, dtype=np.float64).copy()
    z[pos] += y[pos] / state.mu[pos]
    projected = projector.residualize_response_design(z, X, wfinal, tol=config.target_inner_tol)
    resid = np.empty(X.nobs, dtype=np.float64)
    beta = _block_weighted_solve(
        projected.response, projected.design, wfinal, fast=False, resid_out=resid
    )
    return beta, projected.response, projected.design, wfinal


def ppml_vcov_block(
    X_tilde: BlockDesign, y, mu, true_w, *, kind="robust", clusters=None,
    df_absorbed=0, nested_adj=0, weight_kind="none", small_sample="asymptotic",
):
    """PPML model/robust/cluster VCE on an exact BlockDesign."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    tw = np.asarray(true_w, dtype=np.float64)
    bread = equilibrated_gram_inverse(X_tilde.gram(weights=tw * mu))
    kind0 = normalize_vce(kind, "model")
    if kind0 in MODEL_VCE_KINDS:
        return bread
    score_scalar = tw * (y - mu)
    score_scale = frequency_score_scale(weight_kind, tw, kind0)
    effective_n = float(np.sum(tw)) if weight_kind == "fweight" else float(len(y))
    if kind0 == "cluster":
        clusters = normalize_vce_clusters(clusters, nobs=len(y))
    k_total, nested = ppml_covariance_dof(small_sample, X_tilde.ncols + int(df_absorbed), nested_adj)
    return sandwich_vcov_block_xe(
        X_tilde, score_scalar, bread, kind=kind0, clusters=clusters,
        k_total=k_total, nested_adj=nested,
        effective_n=effective_n, score_scale=score_scale,
    )



def _resource_block_projector(X: BlockDesign, plan: FEPlan, config: PPMLConfig, execution):
    if not plan.groups:
        return None
    return BlockWeightedFEProjector.from_design(
        X, plan, engine=config.engine, method="map",
        absorb_threads=execution.threads,
        projection_memory_budget_mb=execution.memory_budget_mb,
    )


@managed_execution
def fit_block_arrays(
    y,
    X: BlockDesign,
    plan: FEPlan,
    *,
    offset,
    true_w,
    weight_kind="none",
    vce,
    clusters,
    names,
    config: PPMLConfig,
    small_sample="asymptotic",
    warm_start=None,
    execution_config=None,
) -> PPMLResult:
    """Internal heterogeneous-spec PPML execution path.

    This is deliberately not exported by ``econhdfe`` and is not selected by
    the public dispatcher.  It exists to verify that a certified BlockDesign
    can traverse the complete PPML statistical contract without dense global-X
    materialization after compilation.
    """
    if not isinstance(X, BlockDesign):
        raise TypeError("heterogeneous-spec PPML requires BlockDesign")
    if not plan.groups:
        raise ValueError("heterogeneous-spec PPML currently requires absorbed FE; a global intercept destroys block separability")
    y = np.asarray(y, dtype=np.float64)
    true_w = np.asarray(true_w, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    if y.shape != (X.nobs,) or true_w.shape != y.shape or off.shape != y.shape:
        raise ShapeError("block PPML inputs must share one observation dimension")
    if len(names) != X.ncols:
        raise ShapeError("names must match BlockDesign columns")
    n = len(y)
    execution, _ = resolve_execution(execution_config, None)

    initial = initial_sample_selection(y, plan, config, weights=true_w, weight_kind=weight_kind)
    active0 = initial_active_rows(initial, require_rows=False)
    has_singletons = bool(np.any(initial.dropped))
    if has_singletons:
        y0, B0, w0, off0 = y[active0], X.subset_rows(active0), true_w[active0], off[active0]
        p0 = plan.subset(active0)
    else:
        y0, B0, w0, off0, p0 = y, X, true_w, off, plan

    standardizer = Standardization.fit_block(y0, B0, weights=w0) if config.standardize else Standardization.outcome_only(y0, w0, B0.ncols)
    y0s, B0s = standardizer.transform_block(y0, B0)
    pre_projector = _resource_block_projector(B0s, p0, config, execution)
    B0s, names, keep = filter_columns_block(
        B0s, names, p0, w0, config.target_inner_tol,
        engine=config.engine, projector=pre_projector,
    )
    standardizer = standardizer.select_columns(keep)

    sep0 = detect_separation_block(y0s, B0s, p0, w0, config, execution, frequency_weights=(weight_kind == "fweight"))
    survivor0 = separation_survivors(y0s, sep0, initial.n_fe)
    any_sep = not bool(np.all(survivor0))

    separation = initial.separated.copy()
    if not any_sep:
        sample = active0.copy() if has_singletons else np.ones(n, dtype=bool)
        ystd, Bstd = y0s, B0s
        ys_orig, ws, offs = y0, w0, off0
        ps_requested = p0
    else:
        idx0 = np.flatnonzero(active0)
        sample = np.zeros(n, dtype=bool)
        sample[idx0[survivor0]] = True
        separation[idx0[sep0.separated]] = True
        ystd, Bstd = y0s[survivor0], B0s.subset_rows(survivor0)
        ys_orig, ws, offs = y0[survivor0], w0[survivor0], off0[survivor0]
        ps_requested = p0.subset(survivor0)
        Bstd, names, keep = filter_columns_block(
            Bstd, names, ps_requested, ws, config.target_inner_tol, engine=config.engine
        )
        standardizer = standardizer.select_columns(keep)

    ps_solver, canonicalization = ps_requested.for_engine(config.engine)
    if (not any_sep) and (ps_solver is p0) and pre_projector is not None:
        projector = pre_projector
    else:
        projector = _resource_block_projector(Bstd, ps_solver, config, execution)

    initial_eta = None
    if warm_start is not None:
        warm_eta = np.asarray(getattr(warm_start, "eta", np.empty(0)), dtype=np.float64)
        if len(warm_eta) == n:
            candidate = warm_eta[sample] - math.log(standardizer.y_scale)
            if np.all(np.isfinite(candidate)):
                initial_eta = candidate

    state = fit_irls_block(
        ystd, Bstd, ps_solver, ws, offs, config,
        structure=None, detect_mu="mu" in config.separation_methods,
        projector=projector, initial_eta=initial_eta,
    )
    if not state.converged:
        raise ConvergenceError(
            f"PPML IRLS failed to converge after {state.iterations} iterations (eps={state.epsilon:.3e})",
            code="convergence.ppml",
            details={"iterations": int(state.iterations), "epsilon": float(state.epsilon)},
        )

    mu_sep = state.separated
    if np.any(mu_sep):
        current = np.flatnonzero(sample)
        separation[current[mu_sep]] = True
        keep_mu = ~mu_sep
        # Upstream trim + HDFE reload also prunes the singletons this creates.
        if config.drop_singletons:
            fw = ws[keep_mu] if weight_kind == "fweight" else None
            keep_mu[np.flatnonzero(keep_mu)[ps_requested.subset(keep_mu).singleton_mask(frequency_weights=fw)]] = False
        sample[current[~keep_mu]] = False
        ystd, Bstd = ystd[keep_mu], Bstd.subset_rows(keep_mu)
        ys_orig, ws, offs = ys_orig[keep_mu], ws[keep_mu], offs[keep_mu]
        ps_requested = ps_requested.subset(keep_mu)
        Bstd, names, keep = filter_columns_block(
            Bstd, names, ps_requested, ws, config.target_inner_tol, engine=config.engine
        )
        standardizer = standardizer.select_columns(keep)
        ps_solver, canonicalization = ps_requested.for_engine(config.engine)
        projector = _resource_block_projector(Bstd, ps_solver, config, execution)
        state = fit_irls_block(
            ystd, Bstd, ps_solver, ws, offs, config,
            structure=None, detect_mu=False, projector=projector,
            initial_eta=state.eta[keep_mu],
        )
        if not state.converged:
            raise ConvergenceError(
                f"PPML IRLS failed after mu-separation trim ({state.iterations} iterations, eps={state.epsilon:.3e})",
                code="convergence.ppml_after_mu",
                details={"iterations": int(state.iterations), "epsilon": float(state.epsilon)},
            )
        sep0.by_method["mu"] = int(np.sum(mu_sep))

    beta_std, _, Xt, _ = final_wls_block(
        ystd, Bstd, ps_solver, ws, offs, state, config,
        structure=None, projector=projector,
    )
    beta = standardizer.unscale_coef(beta_std)

    cls_s = poisson_clusters(clusters, sample, vce)
    dof = ps_requested.dof_info(clusters=cls_s, method=config.dof_method)
    df_a = int(dof.df_absorbed)
    Vstd = ppml_vcov_block(
        Xt, ystd, state.mu, ws, kind=vce, clusters=cls_s,
        df_absorbed=df_a, nested_adj=int(dof.nested > 0), weight_kind=weight_kind,
        small_sample=small_sample,
    )
    V = standardizer.unscale_vcov(Vstd)
    if normalize_vce(vce, "model") in MODEL_VCE_KINDS:
        V /= standardizer.y_scale
    # Block PPML always absorbs FE, so no constant is reported.
    V, se = poisson_inference(
        V, kind=vce, clusters=cls_s,
        scale=lambda: standardizer.original_scale(
            Bstd, ws if config.sample_policy.psd_frequency_scale(weight_kind) else None),
    )

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
    from .estimator import _ppml_wald
    wald_chi2, wald_p, df_m = _ppml_wald(beta, V, names)
    if bool(np.all(sample)):
        full_mu, full_eta = mu_orig, eta_orig
    else:
        full_mu = np.full(n, np.nan)
        full_eta = np.full(n, np.nan)
        full_mu[sample] = mu_orig
        full_eta[sample] = eta_orig

    effective_n = float(np.sum(ws)) if weight_kind == "fweight" else float(np.sum(sample))
    return PPMLResult(
        params=beta, vcov=V, stderr=se, names=tuple(names), converged=True,
        iterations=state.iterations, deviance=deviance, loglike=ll,
        nobs=int(round(effective_n)), n_separated=int(np.sum(separation)),
        sample_mask=sample, separation_mask=separation, mu=full_mu, eta=full_eta,
        df_absorbed=df_a, loglike_null=ll0, pseudo_r2=pseudo_r2,
        chi2=wald_chi2, chi2_pvalue=wald_p, df_model=df_m, nobs_full=int(round(np.sum(true_w))) if weight_kind == "fweight" else int(n),
        vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
        diagnostics={
            "engine": config.engine,
            "sample_policy": sample_policy_diagnostics(config, sample, true_w, weight_kind),
            "finite_sample_correction": ppml_covariance_diagnostics(
                kind=vce, small_sample=small_sample, effective_n=effective_n,
                k_total=len(beta) + df_a, nested_adj=dof.nested, clusters=cls_s),
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
            "heterogeneous_spec_path": True,
        },
        df_resid=poisson_df_resid(cls_s, effective_n, len(beta), df_a),
        vce="model" if vce is None else str(vce),
        cluster_counts=_cluster_counts(cls_s),
        fe_names=fe_names_from_metadata(ps_requested.metadata),
        weight_type=weight_kind if weight_kind in {"fweight", "pweight"} else ("none" if np.allclose(ws, 1.0) else "weight"),
        sum_weights=float(np.sum(ws)),
        n_singletons=int(n - np.sum(sample) - np.sum(separation)),
    )


def fit_structured_dataframe(
    data,
    *,
    y,
    x,
    plan: FEPlan,
    offset=None,
    exposure=None,
    weights=None,
    weight_type=None,
    vce="robust",
    clusters=None,
    config: PPMLConfig | None = None,
    small_sample="asymptotic",
    execution_config=None,
    structural_collinearity: bool = True,
    omit=None,
    warm_start=None,
):
    """Internal structured-design PPML frontend with pre-materialization planning.

    Factor/interaction metadata are compiled and routed to dense or
    heterogeneous-specification storage *before* a global N x K regressor
    matrix is created. The established dense ``fit_arrays`` path remains the
    correctness fallback whenever the structured execution contract is not
    exact or not worthwhile.
    """
    from ...design import _compile_execution_design
    from ..poisson import prepare_poisson_base, prepare_ppml_weights

    cfg = PPMLConfig() if config is None else config
    cfg.validate()
    if isinstance(y, str):
        y_raw = np.asarray(data[y])
    else:
        y_raw = np.asarray(y)

    def col_or_value(value):
        if value is None:
            return None
        if isinstance(value, str):
            return np.asarray(data[value])
        return np.asarray(value)

    yv, off, w = prepare_poisson_base(
        y_raw,
        offset=col_or_value(offset),
        exposure=col_or_value(exposure),
        weights=col_or_value(weights),
    )
    w, weight_kind = prepare_ppml_weights(col_or_value(weights), len(yv), weight_type)
    if plan.groups and plan.nobs != len(yv):
        raise ShapeError("FE plan and outcome have different observation counts")

    memory_budget = getattr(execution_config, "memory_budget_mb", 512.0)
    design = _compile_execution_design(
        data, x, len(yv), groups=plan.groups,
        # PPML necessarily revisits X across rank/separation/IRLS/final WLS.
        expected_passes=2, memory_budget_mb=memory_budget,
        structural=bool(structural_collinearity), omit=omit,
    )

    cls = clusters
    if cls is not None:
        raw = cls if isinstance(cls, (list, tuple)) else [cls]
        cls = [np.asarray(data[c]) if isinstance(c, str) else np.asarray(c) for c in raw]

    use_block = isinstance(design.values, BlockDesign)
    shared_separation = (
        use_block and (not design.values.columns_disjoint)
        and any(m in {"simplex", "relu"} for m in cfg.separation_methods)
    )
    if use_block and not shared_separation:
        result = fit_block_arrays(
            yv, design.values, plan, offset=off, true_w=w, weight_kind=weight_kind, vce=vce,
            clusters=cls, names=design.names, config=cfg,
            small_sample=small_sample,
            execution_config=execution_config, warm_start=warm_start,
        )
    else:
        from .estimator import fit_arrays
        values = design.values.materialize() if use_block else design.values
        result = fit_arrays(
            yv, values, plan, offset=off, true_w=w, weight_kind=weight_kind, vce=vce,
            clusters=cls, names=design.names, config=cfg,
            small_sample=small_sample,
            execution_config=execution_config, warm_start=warm_start,
        )
        if shared_separation:
            result.diagnostics["heterogeneous_spec_fallback"] = "shared_coefficients_with_global_separation"
    return result, design

