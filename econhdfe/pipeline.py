from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import warnings
import numpy as np
from .hdfe.absorber import HDFEAbsorber, HDFEConvergenceError, iterative_singleton_mask
from .hdfe.two_way import TwoWayFEAbsorber
from .hdfe.group_individual import (
    GroupIndividualAbsorber, GroupIndividualInfo, group_compression,
    compress_constant, validate_unique_memberships, group_individual_singleton_mask,
    restrict_group_problem, group_individual_dof,
)
from .hdfe.dof import absorbed_dof
from .compute.encoding import factorize_1d, factorize_interaction, combine_dense_codes
from .hdfe.structure import FEPartitionMeta, canonicalize_fixed_effects, identity_fixed_effect_plan
from .hdfe.specs import FixedEffect, Interaction
from .hdfe.factorvars import compile_absorb_factor_variable
from .factorvars import FactorVariableExpression
from .design import build_design, Factor, RegressorInteraction, _compile_execution_design
from .compute.block_design import BlockDesign
from .collinearity import resolve_collinearity, OmittedVariableWarning, format_omission_warning
from .compute.execution import plan_workspace, should_probe_plain_map, ReusableRHSWorkspace
from .compute.weights import normalize_weight_type, prepare_weights, resolve_vce_for_weights
from .frontend.validate import require_numeric, require_identifier
from .frontend.roles import VariableRole


def _col(data, spec):
    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"column name {spec!r} requires data")
        return np.asarray(data[spec])
    return np.asarray(spec)


def _matrix(data, specs, n=None):
    if specs is None:
        return np.empty((n, 0), dtype=np.float64)
    if isinstance(specs, np.ndarray):
        a = require_numeric(specs, name="matrix", role=VariableRole.REGRESSOR, ndim=(1, 2))
        return a[:, None] if a.ndim == 1 else a
    if isinstance(specs, str):
        return require_numeric(data[[specs]].to_numpy(), name=str(specs), role=VariableRole.REGRESSOR, ndim=2)
    specs = list(specs)
    if not specs:
        return np.empty((n, 0), dtype=np.float64)
    if data is not None and all(isinstance(s, str) for s in specs):
        return require_numeric(data[specs].to_numpy(), name="regressors", role=VariableRole.REGRESSOR, ndim=2)
    cols = [require_numeric(_col(data, s), name=str(s), role=VariableRole.REGRESSOR, ndim=1) for s in specs]
    return np.column_stack(cols)


def _as_absorb_list(absorb):
    if isinstance(absorb, (str, FixedEffect, Interaction, FactorVariableExpression)):
        return [absorb]
    if isinstance(absorb, np.ndarray) and absorb.ndim == 1:
        return [absorb]
    return list(absorb)


def _normalize_fe_spec(raw_spec):
    if isinstance(raw_spec, FixedEffect):
        return raw_spec
    if isinstance(raw_spec, Interaction):
        return FixedEffect(raw_spec)
    if isinstance(raw_spec, (tuple, list)) and len(raw_spec) >= 2 and all(
        isinstance(v, str) for v in raw_spec
    ):
        return FixedEffect(Interaction(*raw_spec))
    if isinstance(raw_spec, dict):
        group = raw_spec.get("group")
        if isinstance(group, (tuple, list)) and len(group) >= 2 and all(
            isinstance(v, str) for v in group
        ):
            group = Interaction(*group)
        return FixedEffect(
            group, raw_spec.get("slopes", ()),
            raw_spec.get("intercept", True), raw_spec.get("name")
        )
    return FixedEffect(raw_spec)


def _encode_absorb(data, absorb, *, component_cache=None):
    groups, slopes, intercepts, names, metadata = [], [], [], [], []
    component_cache = {} if component_cache is None else component_cache

    def component_codes(part):
        key = ("column", part) if isinstance(part, str) else ("array", id(part))
        hit = component_cache.get(key)
        if hit is not None:
            return hit
        raw = require_identifier(_col(data, part), name=str(part), role=VariableRole.FE)
        codes, nlev = factorize_1d(raw)
        hit = (codes, int(nlev))
        component_cache[key] = hit
        return hit

    expanded_absorb = []
    for raw_spec in _as_absorb_list(absorb):
        if isinstance(raw_spec, FactorVariableExpression):
            expanded_absorb.extend(compile_absorb_factor_variable(raw_spec))
        else:
            expanded_absorb.append(raw_spec)

    for i, raw_spec in enumerate(expanded_absorb):
        spec = _normalize_fe_spec(raw_spec)
        components = ()
        if isinstance(spec.group, Interaction):
            dense = [component_codes(part) for part in spec.group.parts]
            codes, _ = combine_dense_codes([v[0] for v in dense], [v[1] for v in dense])
            default_name = spec.group.name or "#".join(
                part if isinstance(part, str) else f"part{j+1}"
                for j, part in enumerate(spec.group.parts)
            )
            if all(isinstance(part, str) for part in spec.group.parts):
                components = tuple(dict.fromkeys(map(str, spec.group.parts)))
        else:
            codes, _ = component_codes(spec.group)
            default_name = spec.group if isinstance(spec.group, str) else f"fe{i+1}"
            if isinstance(spec.group, str):
                components = (str(spec.group),)
        S = _matrix(data, spec.slopes, len(codes)) if spec.slopes else None
        if S is not None and S.shape[1] == 0:
            S = None
        name = str(spec.name or default_name)
        groups.append(codes)
        slopes.append(S)
        intercepts.append(spec.intercept)
        names.append(name)
        metadata.append(FEPartitionMeta(
            name=name, components=components,
            pure_intercept=bool(spec.intercept and S is None),
        ))
    return groups, slopes, intercepts, names, metadata


def _dense_active(groups, slopes, mask):
    out_g, out_s = [], []
    all_kept = bool(np.all(mask))
    for g, s in zip(groups, slopes, strict=False):
        if all_kept:
            out_g.append(np.asarray(g, dtype=np.int32))
            out_s.append(None if s is None else np.asarray(s, dtype=np.float64))
        else:
            _, inv = np.unique(g[mask], return_inverse=True)
            out_g.append(inv.astype(np.int32, copy=False))
            out_s.append(None if s is None else np.asarray(s[mask], dtype=np.float64))
    return out_g, out_s


@dataclass(slots=True)
class StandardFEInference:
    """Requested FE structure on the final estimation sample.

    Solver canonicalization may remove partitions whose column spaces are
    already spanned by finer FEs. Inference retains the requested topology so
    absorbed DoF and cluster nesting do not depend on an execution optimization.
    """

    groups: list[np.ndarray]
    slopes: list[np.ndarray | None]
    intercepts: list[bool]
    names: list[str]
    metadata: tuple[FEPartitionMeta, ...] = ()


def _prepare_standard_fe_structure(
    data, absorb, *, canonicalize_fe, drop_singletons, weights, weight_kind,
    canonical_sample_size=32_768, component_cache=None,
):
    """Encode, canonicalize and singleton-prune a standard HDFE structure.

    Canonicalization is run once before singleton pruning and, when the sample
    changes, once again on the final estimation sample using the *original* FE
    set.  This is the joint fixed point needed for interaction-heavy empirical
    specifications: pruning can reveal new exact functional dependencies,
    while dropping a coarse partition that is already spanned by a finer one
    cannot create additional singletons.
    """
    all_groups, all_slopes, all_intercepts, all_names, metadata = _encode_absorb(data, absorb, component_cache=component_cache)
    levels_all = [int(g.max()) + 1 if len(g) else 0 for g in all_groups]

    if canonicalize_fe:
        pre_groups, pre_slopes, pre_intercepts, pre_names, pre_plan = canonicalize_fixed_effects(
            all_groups, all_slopes, all_intercepts, all_names,
            metadata=metadata, sample_size=canonical_sample_size, passes=1,
        )
    else:
        pre_groups, pre_slopes = all_groups, all_slopes
        pre_intercepts, pre_names = all_intercepts, all_names
        pre_plan = identity_fixed_effect_plan(all_names, levels_all)

    n = len(all_groups[0]) if all_groups else 0
    mask = np.ones(n, dtype=bool)
    dropped = 0
    if drop_singletons:
        mask, dropped = iterative_singleton_mask(
            pre_groups, weights if weight_kind == "fweight" else None,
            frequency_weights=(weight_kind == "fweight"),
        )

    requested_groups, requested_slopes = _dense_active(all_groups, all_slopes, mask)
    inference = StandardFEInference(
        requested_groups, requested_slopes, list(all_intercepts), list(all_names), tuple(metadata)
    )

    if canonicalize_fe and dropped:
        groups, slopes, intercepts, names, plan = canonicalize_fixed_effects(
            requested_groups, requested_slopes, all_intercepts, all_names,
            metadata=metadata, sample_size=canonical_sample_size, passes=2,
        )
    else:
        groups, slopes = _dense_active(pre_groups, pre_slopes, mask)
        intercepts, names, plan = pre_intercepts, pre_names, pre_plan

    return groups, slopes, intercepts, names, plan, mask, dropped, inference


def _cluster_arrays(data, cluster, mask):
    if cluster is None:
        return None
    if isinstance(cluster, str):
        specs = [cluster]
    elif isinstance(cluster, np.ndarray):
        if cluster.ndim == 1:
            specs = [cluster]
        elif cluster.ndim == 2:
            specs = [cluster[:, j] for j in range(cluster.shape[1])]
        else:
            raise ValueError("cluster array must be 1d or 2d")
    else:
        specs = list(cluster)
    if len(specs) > 10:
        raise ValueError("at most 10 cluster dimensions are supported, matching reghdfe")
    out = []
    all_kept = bool(np.all(mask))
    for c in specs:
        raw = require_identifier(_col(data, c), name=str(c), role=VariableRole.CLUSTER)
        codes, _ = factorize_1d(raw if all_kept else raw[mask])
        out.append(codes)
    return out


def _masked_array(data, spec, mask):
    if spec is None:
        return None
    return np.asarray(_col(data, spec))[mask]


def _resolve_pool_size(absorber, ncols, pool_size, memory_budget_mb):
    if pool_size not in (None, "auto"):
        return max(1, min(int(pool_size), ncols))
    # Count *additional* per-RHS scratch arrays, not the input matrix that is
    # already resident. Plain MAP needs only the previous-iterate buffer; CG
    # carries several Krylov vectors. This lets indexed/fused multi-RHS kernels
    # use wider pools at 10M+ rows without understating peak scratch memory.
    work_arrays = 4 if absorber.acceleration == "cg" else 1
    if absorber.method in {"lsmr", "lsqr"}:
        work_arrays = 2
    budget = max(float(memory_budget_mb), 64.0) * (1024**2)
    bytes_per_col = max(absorber.nobs * 8 * work_arrays, 1)
    return max(1, min(ncols, int(budget // bytes_per_col)))


def _residualize_blocks(
    absorber, matrix, pool_size, memory_budget_mb=512, *, adaptive_auto=False
):
    """Residualize columns with one reusable contiguous workspace.

    Execution policy is deliberately separate from the absorber mathematics.
    For very large wide MAP problems requested through ``method="auto"``,
    the first *real* pool is a bounded plain-MAP probe. If it converges quickly
    the remaining pools stay on plain MAP; otherwise that same pool is reset
    from its source and the executor falls back to CG. No extra full-data probe
    pass is introduced.
    """
    ncols = int(matrix.shape[1])
    scratch = 4 if absorber.acceleration == "cg" else 2
    plan = plan_workspace(
        nobs=absorber.nobs, ncols=ncols, pool_size=pool_size,
        memory_budget_mb=memory_budget_mb, scratch_arrays_per_rhs=scratch,
        requested_threads=getattr(absorber, "absorb_threads", "auto"),
    )
    absorber.execution_plan = plan
    thread_kwargs = {"absorb_threads": plan.threads} if hasattr(absorber, "absorb_threads") else {}

    adaptive = bool(adaptive_auto and should_probe_plain_map(
        nobs=absorber.nobs, ncols=ncols, method=absorber.method,
        acceleration=absorber.acceleration, plan=plan,
    ))

    # Small/full-width problems keep the zero-copy in-place fast path.
    if plan.pool_size >= ncols and not adaptive:
        out, info = absorber.residualize(matrix, copy=False, return_info=True, **thread_kwargs)
        if out is not matrix:
            matrix[...] = out
        return matrix, info

    workspace = ReusableRHSWorkspace(absorber.nobs, plan.pool_size, dtype=matrix.dtype)
    infos = []
    original_accel = absorber.acceleration
    original_max_iter = absorber.max_iter
    chosen_accel = original_accel

    try:
        for pool_idx, lo in enumerate(range(0, ncols, plan.pool_size)):
            hi = min(ncols, lo + plan.pool_size)
            work = workspace.load(matrix, lo, hi)

            if adaptive and pool_idx == 0:
                # The first required RHS pool doubles as the solver probe.
                absorber.acceleration = "none"
                absorber.max_iter = min(original_max_iter, 64)
                res, info = absorber.residualize(
                    work, copy=False, return_info=True, **thread_kwargs
                )
                if info.converged:
                    chosen_accel = "none"
                else:
                    # Restore the original pool and fall back to the robust CG
                    # path without accepting a partially converged probe.
                    work = workspace.load(matrix, lo, hi)
                    absorber.acceleration = original_accel
                    absorber.max_iter = original_max_iter
                    res, info = absorber.residualize(
                        work, copy=False, return_info=True, **thread_kwargs
                    )
                    chosen_accel = original_accel
                absorber.acceleration = chosen_accel
                absorber.max_iter = original_max_iter
            else:
                absorber.acceleration = chosen_accel
                absorber.max_iter = original_max_iter
                res, info = absorber.residualize(
                    work, copy=False, return_info=True, **thread_kwargs
                )

            workspace.write(matrix, lo, hi, res)
            infos.append(info)
    finally:
        absorber.acceleration = chosen_accel
        absorber.max_iter = original_max_iter

    worst = max(infos, key=lambda z: (not z.converged, z.iterations, z.max_update))
    return matrix, worst


def _resolve_method(method, groups=None, slopes=None, intercepts=None, *, backend="numpy", group_mode=False, individual=False):
    """Resolve the execution method after FE canonicalization.

    ``auto`` is deliberately conservative: use the specialized two-way solver
    only for exactly two pure categorical intercept FEs on the NumPy backend;
    otherwise use MAP+CG.  Group+individual problems use LSMR, where MAP is
    not applicable.
    """
    requested = str(method).lower()
    if requested != "auto":
        return requested, {"requested": requested, "resolved": requested, "reason": "explicit"}
    if group_mode:
        resolved = "lsmr" if individual else "map"
        reason = "group_individual_requires_krylov" if individual else "group_only_standard_map"
        return resolved, {"requested": "auto", "resolved": resolved, "reason": reason}
    groups = [] if groups is None else groups
    slopes = [None] * len(groups) if slopes is None else slopes
    intercepts = [True] * len(groups) if intercepts is None else intercepts
    pure_two_way = (
        backend == "numpy" and len(groups) == 2 and all(intercepts)
        and all(s is None for s in slopes)
    )
    if pure_two_way:
        return "twoway", {
            "requested": "auto", "resolved": "twoway",
            "reason": "canonical_system_is_two_pure_intercept_fes",
        }
    return "map", {
        "requested": "auto", "resolved": "map",
        "reason": "general_hdfe_uses_symmetric_map_cg",
    }


def _resolve_acceleration(method, acceleration, *, group_mode=False):
    if acceleration == "auto":
        return "none" if group_mode or method != "map" else "cg"
    return acceleration


def _check_absorption(info, *, allow_nonconverged=False):
    if not info.converged and not allow_nonconverged:
        raise HDFEConvergenceError(info)


def _absorb_metadata(plan, absorber, info, pool_size, solver_selection=None):
    out = {
        "canonicalization": plan.as_dict() if plan is not None else None,
        "method": absorber.method,
        "transform": absorber.transform,
        "acceleration": absorber.acceleration,
        "projection_backend": absorber.projection_backend,
        "absorb_threads": absorber.absorb_threads,
        "iterations": int(info.iterations),
        "converged": bool(info.converged),
        "convergence_metric": float(info.max_update),
        "convergence_criterion": getattr(info, "criterion", "unknown"),
        "pool_size": (absorber.execution_plan.pool_size if hasattr(absorber, "execution_plan") else pool_size),
    }
    if solver_selection is not None:
        out["solver_selection"] = dict(solver_selection)
    if hasattr(absorber, "execution_plan"):
        out["execution_plan"] = absorber.execution_plan.as_dict()
    if hasattr(absorber, "solve_info"):
        si = absorber.solve_info
        out["two_way"] = {
            "connected_components": int(si.connected_components),
            "cross_nnz": int(si.cross_nnz),
            "solved_side_levels": int(si.solved_side_levels),
            "eliminated_side_levels": int(si.eliminated_side_levels),
        }
    return out



def _design(data, specs, n, prefix, *, row_mask=None, absorbed_groups=(), absorbed_names=(), structural=True, protected_terms=(), omit=None):
    return build_design(
        data, specs, n, prefix=prefix, row_mask=row_mask,
        absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
        structural=structural, protected_terms=protected_terms, omit=omit,
    )


def _execution_design(
    data, specs, n, prefix, *, groups=(), expected_passes=1,
    memory_budget_mb=None, row_mask=None, absorbed_groups=(), absorbed_names=(),
    structural=True, protected_terms=(), omit=None,
):
    """Compile a design for execution, using heterogeneous-spec storage when useful.

    This is an internal performance contract. Statistical parsing and omission
    semantics are identical to ``_design``; only the physical representation
    may become a row-partitioned ``BlockDesign`` when an exact certificate and
    the storage planner both justify it.
    """
    return _compile_execution_design(
        data, specs, n, groups=groups, expected_passes=expected_passes,
        memory_budget_mb=memory_budget_mb, prefix=prefix, row_mask=row_mask,
        absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
        structural=structural, protected_terms=protected_terms, omit=omit,
    )


def _select_design_columns(X, idx):
    idx = np.asarray(idx, dtype=np.int64)
    if isinstance(X, BlockDesign):
        return X.select_columns(idx)
    return X[:, idx]


def _check_collinearity_mode(mode):
    mode = str(mode).lower()
    if mode not in {"warn", "drop", "raise"}:
        raise ValueError("collinearity must be 'warn', 'drop', or 'raise'")
    return mode


def _enforce_collinearity_mode(plan, mode, structural_plan=None):
    structural = () if structural_plan is None else structural_plan.omissions
    items = tuple(structural) + tuple(plan.omitted)
    if not items:
        return
    if mode == "raise":
        parts = [f"{o.name} ({o.reason})" for o in items]
        raise np.linalg.LinAlgError(f"collinear/absorbed columns detected: {', '.join(parts)}")
    if mode == "warn":
        warnings.warn(format_omission_warning(items), OmittedVariableWarning, stacklevel=4)


def _collinearity_metadata(
    plan, origins=None, *, structural_plan=None, requested_origins=None, requested_names=None, role="regressor", user_omissions=()
):
    out = plan.as_dict()
    origins = tuple(() if origins is None else origins)
    req_origins = tuple(requested_origins) if requested_origins else origins
    req_names = tuple(requested_names) if requested_names else tuple(plan.requested_names)

    origin_by_name = {o.name: o for o in req_origins}
    columns = []
    for name in req_names:
        o = origin_by_name.get(name)
        columns.append({
            "name": name,
            "term": None if o is None else o.term,
            "kind": None if o is None else o.kind,
            "components": () if o is None else o.components,
            "level": None if o is None else o.level,
        })

    omitted = []
    for item in tuple(user_omissions or ()):
        d = item.as_dict() if hasattr(item, "as_dict") else dict(item)
        d.update({"role": role, "relative_norm": 0.0, "structural": False, "selected_by_user": True})
        omitted.append(d)
    if structural_plan is not None:
        for item in structural_plan.omissions:
            d = item.as_dict()
            structural_reason = d["reason"]
            public_reason = (
                "absorbed_or_zero" if structural_reason == "absorbed_by_fe"
                else "linear_combination"
            )
            d.update({
                "reason": public_reason,
                "structural_reason": structural_reason,
                "role": role,
                "relative_norm": 0.0,
                "structural": True,
                "selected_by_user": False,
            })
            o = origin_by_name.get(d["name"])
            if o is not None:
                d.update({"term": o.term, "kind": o.kind, "components": o.components, "level": o.level})
            omitted.append(d)

    for item in out["omitted"]:
        d = dict(item)
        i = int(d["index"])
        o = origins[i] if i < len(origins) else origin_by_name.get(d["name"])
        if o is not None:
            d.update({"term": o.term, "kind": o.kind, "components": o.components, "level": o.level})
        d["structural"] = False
        d["selected_by_user"] = False
        omitted.append(d)

    out["requested"] = req_names
    out["columns"] = tuple(columns)
    out["omitted"] = tuple(omitted)
    if structural_plan is not None:
        out["structural"] = structural_plan.as_dict()
    return out


def _resolve_ols_columns(Xw, X0, names, *, weights, tolerance, mode, structural_plan=None):
    plan = resolve_collinearity(
        Xw, names=names, role="regressor", original=X0, weights=weights,
        tolerance=tolerance,
    )
    _enforce_collinearity_mode(plan, mode, structural_plan)
    idx = tuple(plan.active_indices)
    if idx == tuple(range(Xw.shape[1])):
        return Xw, X0, plan
    return _select_design_columns(Xw, idx), _select_design_columns(X0, idx), plan


def _resolve_iv_columns(Cw, Ew, Zw, C0, E0, Z0, c_names, e_names, z_names, *, weights, tolerance, mode, c_origins=None, e_origins=None, z_origins=None, c_design=None, e_design=None, z_design=None):
    cplan = resolve_collinearity(
        Cw, names=c_names, role="exogenous", original=C0, weights=weights, tolerance=tolerance,
    )
    _enforce_collinearity_mode(cplan, mode, None if c_design is None else c_design.structural_plan)
    ci = tuple(cplan.active_indices)
    Cwa, C0a = (Cw, C0) if ci == tuple(range(Cw.shape[1])) else (_select_design_columns(Cw, ci), _select_design_columns(C0, ci))
    cn = tuple(cplan.active_names)

    eplan = resolve_collinearity(
        Ew, names=e_names, role="endogenous", original=E0, weights=weights, tolerance=tolerance,
        protected_basis=Cwa, protected_names=cn,
    )
    _enforce_collinearity_mode(eplan, mode, None if e_design is None else e_design.structural_plan)
    ei = tuple(eplan.active_indices)
    Ewa, E0a = (Ew, E0) if ei == tuple(range(Ew.shape[1])) else (_select_design_columns(Ew, ei), _select_design_columns(E0, ei))
    en = tuple(eplan.active_names)

    # Excluded instruments are tested conditional on included exogenous
    # regressors.  This is the important ivreg2-style priority rule: an
    # excluded instrument redundant with included exog is omitted, not vice
    # versa.  It also catches instruments fully absorbed by HDFE.
    zplan = resolve_collinearity(
        Zw, names=z_names, role="excluded_instrument", original=Z0, weights=weights, tolerance=tolerance,
        protected_basis=Cwa, protected_names=cn,
    )
    _enforce_collinearity_mode(zplan, mode, None if z_design is None else z_design.structural_plan)
    zi = tuple(zplan.active_indices)
    Zwa, Z0a = (Zw, Z0) if zi == tuple(range(Zw.shape[1])) else (_select_design_columns(Zw, zi), _select_design_columns(Z0, zi))
    zn = tuple(zplan.active_names)
    if Ewa.shape[1] == 0:
        raise ValueError("all endogenous regressors were absorbed or collinear")
    if Zwa.shape[1] < Ewa.shape[1]:
        raise ValueError(
            f"underidentified after collinearity resolution: {Zwa.shape[1]} active excluded "
            f"instruments for {Ewa.shape[1]} active endogenous regressors"
        )
    def meta(plan, origins, design, role):
        if design is None:
            return _collinearity_metadata(plan, origins, role=role)
        return _collinearity_metadata(
            plan, origins, structural_plan=design.structural_plan,
            requested_origins=design.requested_origins, requested_names=design.requested_names,
            role=role, user_omissions=design.user_omissions,
        )

    info = {
        "exogenous": meta(cplan, c_origins, c_design, "exogenous"),
        "endogenous": meta(eplan, e_origins, e_design, "endogenous"),
        "excluded_instruments": meta(zplan, z_origins, z_design, "excluded_instrument"),
    }
    return Cwa, Ewa, Zwa, C0a, E0a, Z0a, cn, en, zn, info



def _same_fe_group(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is b:
        return True
    try:
        aa, bb = np.asarray(a), np.asarray(b)
        return aa.shape == bb.shape and bool(np.array_equal(aa, bb, equal_nan=True))
    except Exception:
        return False


def _split_individual_absorb(absorb, individual):
    raw_specs = []
    for value in _as_absorb_list(absorb):
        if isinstance(value, FactorVariableExpression):
            raw_specs.extend(compile_absorb_factor_variable(value))
        else:
            raw_specs.append(value)
    specs = [_normalize_fe_spec(v) for v in raw_specs]
    if individual is None:
        return specs, None
    hits = [i for i, spec in enumerate(specs) if _same_fe_group(spec.group, individual)]
    if len(hits) != 1:
        raise ValueError(
            "individual() must appear exactly once in absorb(), matching reghdfe semantics"
        )
    j = hits[0]
    indiv = specs[j]
    return specs[:j] + specs[j+1:], indiv


def _encode_absorb_grouped(data, specs, comp):
    groups, slopes, intercepts, names = [], [], [], []
    for i, spec in enumerate(specs):
        raw_group = _col(data, spec.group)
        group_value = compress_constant(raw_group, comp, name=f"absorbed FE {spec.group!r}")
        codes, _ = factorize_1d(group_value)
        S = None
        if spec.slopes:
            raw_s = _matrix(data, spec.slopes, comp.n_memberships)
            S = compress_constant(raw_s, comp, name=f"slopes for FE {spec.group!r}").astype(np.float64, copy=False)
            if S.ndim == 1:
                S = S[:, None]
        groups.append(codes)
        slopes.append(S)
        intercepts.append(spec.intercept)
        names.append(spec.name or (spec.group if isinstance(spec.group, str) else f"fe{i+1}"))
    return groups, slopes, intercepts, names


def _group_cluster_arrays(data, cluster, comp, keep):
    if cluster is None:
        return None
    if isinstance(cluster, str):
        specs = [cluster]
    elif isinstance(cluster, np.ndarray):
        if cluster.ndim == 1:
            specs = [cluster]
        elif cluster.ndim == 2:
            specs = [cluster[:, j] for j in range(cluster.shape[1])]
        else:
            raise ValueError("cluster array must be 1d or 2d")
    else:
        specs = list(cluster)
    if len(specs) > 10:
        raise ValueError("at most 10 cluster dimensions are supported, matching reghdfe")
    out = []
    for c in specs:
        vals = compress_constant(_col(data, c), comp, name=f"cluster {c!r}")[keep]
        codes, _ = factorize_1d(vals)
        out.append(codes)
    return out


def _group_optional_array(data, spec, comp, keep, label):
    if spec is None:
        return None
    return np.asarray(compress_constant(_col(data, spec), comp, name=label))[keep]


def _prepare_group_mode(data, *, group, individual, absorb, arrays, weights, cluster, time, panel,
                        drop_singletons, weight_type, aggregation):
    raw_group = _col(data, group)
    group_codes, _ = factorize_1d(raw_group)
    comp = group_compression(group_codes)
    standard_specs, indiv_spec = _split_individual_absorb(absorb, individual)
    std_groups0, std_slopes0, std_intercepts, std_names = _encode_absorb_grouped(data, standard_specs, comp)

    compressed_arrays = [compress_constant(a, comp, name=name) for name, a in arrays]
    w0 = None
    if weights is not None:
        w0 = compress_constant(_col(data, weights), comp, name="weights").astype(np.float64, copy=False)
    wkind = normalize_weight_type(weight_type, has_weights=w0 is not None)
    if indiv_spec is not None and wkind == "fweight":
        # Current reghdfe's Factor_Indiv.mata explicitly notes that fweights are
        # meaningless because group/individual membership rows cannot be duplicated.
        raise ValueError("fweights are not supported with group()+individual() fixed effects")
    _ = prepare_weights(w0, comp.n_groups, wkind)

    indiv_codes0 = None
    indiv_slopes0 = None
    if indiv_spec is not None:
        raw_ind = _col(data, indiv_spec.group)
        indiv_codes0, _ = factorize_1d(raw_ind)
        validate_unique_memberships(comp.row_to_group, indiv_codes0)
        if indiv_spec.slopes:
            indiv_slopes0 = _matrix(data, indiv_spec.slopes, comp.n_memberships)
            if indiv_slopes0.ndim == 1:
                indiv_slopes0 = indiv_slopes0[:, None]
        if str(aggregation).lower() not in {"mean", "sum"}:
            raise ValueError("aggregation must be 'mean' or 'sum'")

    if drop_singletons:
        if indiv_spec is None:
            keep, dropped = iterative_singleton_mask(std_groups0)
        else:
            keep, dropped = group_individual_singleton_mask(
                std_groups0, comp.row_to_group, indiv_codes0
            )
    else:
        keep = np.ones(comp.n_groups, dtype=bool)
        dropped = 0
    if not np.any(keep):
        raise ValueError("insufficient observations after singleton pruning")

    grouped_arrays = [np.asarray(a)[keep] for a in compressed_arrays]
    wgrp = None if w0 is None else w0[keep]
    winfo = prepare_weights(wgrp, int(keep.sum()), wkind)
    clusters = _group_cluster_arrays(data, cluster, comp, keep)
    time1 = _group_optional_array(data, time, comp, keep, "time")
    panel1 = _group_optional_array(data, panel, comp, keep, "panel")

    if indiv_spec is None:
        std_groups, std_slopes = _dense_active(std_groups0, std_slopes0, keep)
        return {
            "arrays": grouped_arrays, "winfo": winfo, "weights": winfo.estimation,
            "groups": std_groups, "slopes": std_slopes, "intercepts": std_intercepts,
            "names": std_names, "clusters": clusters, "time": time1, "panel": panel1,
            "dropped": dropped, "individual_spec": None, "comp": comp, "keep": keep,
        }

    std_groups, std_slopes, mg, ind, islopes, member_keep = restrict_group_problem(
        std_groups0, std_slopes0, comp.row_to_group, indiv_codes0, indiv_slopes0, keep
    )
    ind_name = indiv_spec.name or (indiv_spec.group if isinstance(indiv_spec.group, str) else "individual")
    return {
        "arrays": grouped_arrays, "winfo": winfo, "weights": winfo.estimation,
        "groups": std_groups, "slopes": std_slopes, "intercepts": std_intercepts,
        "names": std_names + [ind_name], "clusters": clusters, "time": time1, "panel": panel1,
        "dropped": dropped, "individual_spec": indiv_spec, "membership_group": mg,
        "individual_codes": ind, "individual_slopes": islopes, "member_keep": member_keep,
        "comp": comp, "keep": keep,
    }


def _make_standard_absorber(
    groups, slopes, intercepts, *, weights, tol, max_iter, backend, method,
    transform, acceleration, preconditioner, projection_backend, absorb_threads,
    memory_budget_mb,
):
    if method == "twoway":
        if backend != "numpy":
            raise ValueError("method='twoway' currently requires backend='numpy'")
        if len(groups) != 2 or any(s is not None for s in slopes) or not all(intercepts):
            raise ValueError(
                "method='twoway' requires exactly two intercept-only fixed effects "
                "after FE canonicalization"
            )
        return TwoWayFEAbsorber(groups, weights=weights, tol=tol, max_iter=max_iter)
    return HDFEAbsorber(
        groups, slopes=slopes, intercepts=intercepts, weights=weights,
        tol=tol, max_iter=max_iter, backend=backend, method=method,
        transform=transform, acceleration=acceleration,
        preconditioner=preconditioner, projection_backend=projection_backend,
        absorb_threads=absorb_threads, projection_memory_budget_mb=memory_budget_mb,
    )


def _make_group_absorber(prep, *, method, transform, backend, acceleration, preconditioner, tol, max_iter, aggregation, incidence_backend, memory_budget_mb):
    indiv = prep["individual_spec"]
    if indiv is None:
        return HDFEAbsorber(
            prep["groups"], slopes=prep["slopes"], intercepts=prep["intercepts"],
            weights=prep["weights"], tol=tol, max_iter=max_iter, backend=backend,
            method=method, transform=transform, acceleration=acceleration,
            preconditioner=preconditioner,
        )
    if backend != "numpy":
        raise ValueError("group()+individual() currently uses SciPy LSMR/LSQR on CPU; backend must be 'numpy'")
    if acceleration != "none":
        raise ValueError("CG/MAP acceleration is unavailable with group()+individual(); use LSMR/LSQR")
    solver = "lsmr" if method == "map" else method
    if solver not in {"lsmr", "lsqr"}:
        raise ValueError("group()+individual() requires method='lsmr' or method='lsqr'")
    return GroupIndividualAbsorber(
        prep["groups"], standard_slopes=prep["slopes"], standard_intercepts=prep["intercepts"],
        membership_group=prep["membership_group"], individual_codes=prep["individual_codes"],
        individual_slopes=prep["individual_slopes"], individual_intercept=indiv.intercept,
        aggregation=aggregation, weights=prep["weights"], tol=tol, max_iter=max_iter,
        method=solver, preconditioner=preconditioner, incidence_backend=incidence_backend,
        memory_budget_mb=memory_budget_mb,
    )


def _group_dof(prep, absorber, dof_method):
    if prep["individual_spec"] is None:
        return absorbed_dof(
            prep["groups"], slopes=prep["slopes"], intercepts=prep["intercepts"],
            clusters=prep["clusters"], method=dof_method, adjust_nested=bool(prep["clusters"]),
            adjust_continuous=True, groups_are_dense=True,
        )
    indiv = prep["individual_spec"]
    return group_individual_dof(
        prep["groups"], standard_slopes=prep["slopes"], standard_intercepts=prep["intercepts"],
        clusters=prep["clusters"], individual_levels=absorber.n_individuals,
        individual_intercept=indiv.intercept,
        individual_nslopes=absorber.n_individual_slopes, method=dof_method,
    )


def _group_info(prep, absorber, aggregation):
    if prep["individual_spec"] is None:
        return GroupIndividualInfo(
            n_groups=int(prep["keep"].sum()), n_memberships=int(prep["member_keep"].sum()) if "member_keep" in prep else int(prep["comp"].n_memberships),
            n_individuals=0, aggregation="none", group_sizes=prep["comp"].group_sizes[prep["keep"]],
            first_rows=prep["comp"].first_rows[prep["keep"]], individual_intercept=False, individual_slopes=0,
            solver=getattr(absorber, "method", "map"), incidence_backend="none", dof_note="group() only: estimation is performed on one observation per group",
        )
    kept_sizes = np.bincount(prep["membership_group"], minlength=absorber.nobs).astype(np.int64, copy=False)
    return GroupIndividualInfo(
        n_groups=absorber.nobs, n_memberships=len(prep["membership_group"]),
        n_individuals=absorber.n_individuals, aggregation=str(aggregation).lower(),
        group_sizes=kept_sizes, first_rows=prep["comp"].first_rows[prep["keep"]],
        individual_intercept=prep["individual_spec"].intercept,
        individual_slopes=absorber.n_individual_slopes, solver=absorber.method,
        incidence_backend=absorber.incidence_backend,
    )


