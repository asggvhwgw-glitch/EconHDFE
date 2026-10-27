"""Linear HDFE pipeline stages shared by OLS, linear IV and repeated-spec sessions.

Public (un-prefixed) names are the internal stage interface; underscore names
are local helpers. Stages, in execution order:

* data access: ``column_values``, ``cluster_arrays``, ``masked_array``;
* sample and FE preparation: ``prepare_standard_fe_structure`` (returns a
  ``StandardFEStructure``), ``standard_dof``, ``prepare_group_mode``,
  ``group_dof``, ``group_info``, ``group_has_intercept``;
* design compilation: ``compile_design``, ``compile_execution_design``;
* absorption: ``resolve_method``, ``resolve_acceleration``,
  ``make_standard_absorber``, ``make_group_absorber``, ``residualize_blocks``,
  ``residualize_in_basis``, ``check_absorption``;
* collinearity: ``check_collinearity_mode``, ``resolve_ols_columns``,
  ``resolve_iv_columns``, ``collinearity_metadata``,
  ``design_collinearity_metadata``;
* absorption metadata: ``absorb_metadata``, ``block_absorb_metadata``,
  ``conditioning_info``, ``resolve_pool_size``.
"""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
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
from .hdfe.specs import Interaction
from .hdfe.absorb_spec import absorb_terms, expand_factor_variables, fixed_effect_spec
from .design import build_design, Factor, RegressorInteraction, _compile_execution_design
from .compute.block_design import BlockDesign
from .collinearity import resolve_collinearity, OmittedVariableWarning, format_omission_warning
from .compute.execution import _plan_workspace, should_probe_plain_map, ReusableRHSWorkspace
from .hdfe.projection import estimate_index_bytes, resolve_threads
from .hdfe.workspace import _absorption_workspace_cost, _absorption_build_cost
from .planner import scheduler
from .compute.weights import normalize_weight_type, prepare_weights, resolve_vce_for_weights
from .compute.stable_linalg import _apply_absorption_basis
from .frontend.validate import require_numeric, require_identifier
from .frontend.roles import VariableRole


def column_values(data, spec):
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
    cols = [require_numeric(column_values(data, s), name=str(s), role=VariableRole.REGRESSOR, ndim=1) for s in specs]
    return np.column_stack(cols)


def _encode_absorb(data, absorb, *, component_cache=None):
    groups, slopes, intercepts, names, metadata = [], [], [], [], []
    component_cache = {} if component_cache is None else component_cache

    def component_codes(part):
        key = ("column", part) if isinstance(part, str) else ("array", id(part))
        hit = component_cache.get(key)
        if hit is not None:
            return hit
        raw = require_identifier(column_values(data, part), name=str(part), role=VariableRole.FE)
        codes, nlev = factorize_1d(raw)
        hit = (codes, int(nlev))
        component_cache[key] = hit
        return hit

    for i, raw_spec in enumerate(expand_factor_variables(absorb_terms(absorb))):
        spec = fixed_effect_spec(raw_spec)
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


@dataclass(slots=True)
class StandardFEStructure:
    """Solver FE system on the final estimation sample.

    ``groups``/``slopes``/``intercepts``/``names``/``plan`` describe the
    canonicalized partitions the absorber solves; ``inference`` keeps the
    requested topology (``None`` when nothing is absorbed). ``mask`` selects
    the estimation sample from the input rows after singleton pruning.
    """

    groups: list[np.ndarray]
    slopes: list[np.ndarray | None]
    intercepts: list[bool]
    names: list[str]
    plan: object
    mask: np.ndarray
    dropped: int
    inference: StandardFEInference | None

    @property
    def all_kept(self) -> bool:
        return self.dropped == 0 and bool(np.all(self.mask))

    def active(self, values):
        """Restrict an input-row array to the estimation sample."""
        return values if self.all_kept else values[self.mask]

    def absorbed_intercepts(self):
        """Intercept FEs (and their names) that can structurally absorb design columns."""
        return (
            [g for g, has_i in zip(self.groups, self.intercepts, strict=False) if has_i],
            [n for n, has_i in zip(self.names, self.intercepts, strict=False) if has_i],
        )


def prepare_standard_fe_structure(
    data, absorb, *, canonicalize_fe, drop_singletons, weights, weight_kind,
    canonical_sample_size=32_768, component_cache=None, singleton_mode="stata_compat",
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
        eligible = None
        if singleton_mode == "unified":
            # A slope-only singleton with all slopes zero cannot fit its row.
            eligible = [np.ones(n, dtype=bool) if intercept else
                        (np.zeros(n, dtype=bool) if slope is None else
                         np.any(np.asarray(slope).reshape(n, -1) != 0, axis=1))
                        for slope, intercept in zip(pre_slopes, pre_intercepts)]
        mask, dropped = iterative_singleton_mask(
            pre_groups, weights if weight_kind == "fweight" else None,
            frequency_weights=(weight_kind == "fweight"), singleton_eligible=eligible,
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

    return StandardFEStructure(groups, slopes, intercepts, names, plan, mask, dropped, inference)


def standard_dof(fe, clusters, method):
    """Absorbed DoF of the requested FE topology on the estimation sample."""
    return absorbed_dof(
        fe.inference.groups, slopes=fe.inference.slopes, intercepts=fe.inference.intercepts,
        clusters=clusters, method=method, adjust_nested=bool(clusters),
        adjust_continuous=True, groups_are_dense=True,
    )


def _cluster_specs(cluster):
    """One column spec per cluster dimension."""
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
    return specs


def cluster_arrays(data, cluster, mask):
    if cluster is None:
        return None
    specs = _cluster_specs(cluster)
    out = []
    all_kept = bool(np.all(mask))
    for c in specs:
        raw = require_identifier(column_values(data, c), name=str(c), role=VariableRole.CLUSTER)
        codes, _ = factorize_1d(raw if all_kept else raw[mask])
        out.append(codes)
    return out


def masked_array(data, spec, mask):
    if spec is None:
        return None
    return np.asarray(column_values(data, spec))[mask]


def resolve_pool_size(absorber, ncols, pool_size, memory_budget_mb):
    # Report the actual execution plan, rather than running a second,
    # inconsistent memory heuristic after absorption.
    if hasattr(absorber, "execution_plan"):
        return absorber.execution_plan.pool_size
    resident, per_col = _absorption_workspace_cost(absorber)
    return _plan_workspace(
        nobs=absorber.nobs, ncols=ncols, pool_size=pool_size,
        memory_budget_mb=memory_budget_mb, bytes_per_column=per_col,
        resident_solver_bytes=resident,
        requested_threads=getattr(absorber, "absorb_threads", "auto"),
    ).pool_size



def residualize_blocks(
    absorber, matrix, pool_size, memory_budget_mb=512, *, adaptive_auto=False,
    profile=False,
):
    """Residualize columns with one reusable contiguous workspace.

    Execution policy is deliberately separate from the absorber mathematics.
    For very large wide MAP problems requested through ``method="auto"``,
    the first real pool uses the absorber's contraction probe. Fast contraction
    keeps plain MAP; hard systems continue CG on the partially transformed pool.
    Later pools retain the first pool's choice without repeating the probe.
    """
    ncols = int(matrix.shape[1])
    resident, per_col = _absorption_workspace_cost(absorber)
    plan = _plan_workspace(
        nobs=absorber.nobs, ncols=ncols, pool_size=pool_size,
        memory_budget_mb=memory_budget_mb, bytes_per_column=per_col,
        resident_solver_bytes=resident,
        requested_threads=getattr(absorber, "absorb_threads", "auto"),
    )
    absorber.execution_plan = plan
    absorber.execution_memory = {
        "scope": "resident_solver_state_and_pooled_workspace",
        "input_matrix_bytes": int(matrix.nbytes),
        "resident_solver_bytes": resident,
        "workspace_bytes_per_column": per_col,
        "estimated_solver_bytes": resident + per_col * plan.pool_size,
        "budget_bytes": int(float(memory_budget_mb) * 1024**2),
    }
    absorber._profile_enabled = bool(profile)
    absorber.execution_profile = {"pools": []} if profile else None
    thread_kwargs = {"absorb_threads": plan.threads} if hasattr(absorber, "absorb_threads") else {}

    adaptive = bool(adaptive_auto and should_probe_plain_map(
        nobs=absorber.nobs, ncols=ncols, method=absorber.method,
        acceleration=absorber.acceleration, plan=plan,
    ))

    # Small/full-width problems keep the zero-copy in-place fast path.
    if plan.pool_size >= ncols and not adaptive:
        started = perf_counter() if profile else None
        out, info = absorber.residualize(matrix, copy=False, return_info=True, **thread_kwargs)
        if out is not matrix:
            matrix[...] = out
        if profile:
            absorber.execution_profile["pools"].append({
                "columns": ncols, "seconds": perf_counter() - started,
                "iterations": int(info.iterations), "criterion": getattr(info, "criterion", "unknown"),
                "converged": bool(info.converged),
                "stages": dict(getattr(absorber, "_iteration_profile", {})),
            })
        return matrix, info

    workspace = ReusableRHSWorkspace(absorber.nobs, plan.pool_size, dtype=matrix.dtype)
    infos = []
    original_accel = absorber.acceleration
    original_plain_limit = getattr(absorber, "auto_plain_limit", None)
    chosen_accel = original_accel

    try:
        for pool_idx, lo in enumerate(range(0, ncols, plan.pool_size)):
            hi = min(ncols, lo + plan.pool_size)
            work = workspace.load(matrix, lo, hi)
            started = perf_counter() if profile else None

            if adaptive and pool_idx == 0:
                # Keep the previous allowance for easy, fast-contracting
                # systems (some need >8 sweeps), but reject hard ones after
                # the shared two-sweep probe without throwing away its work.
                absorber.acceleration = "auto"
                absorber.auto_plain_limit = max(original_plain_limit, 64)
                res, info = absorber.residualize(
                    work, copy=False, return_info=True, **thread_kwargs
                )
                chosen_accel = "cg" if "cg" in info.criterion else "none"
                absorber.acceleration = chosen_accel
                absorber.auto_plain_limit = original_plain_limit
            else:
                absorber.acceleration = chosen_accel
                res, info = absorber.residualize(
                    work, copy=False, return_info=True, **thread_kwargs
                )

            if profile:
                absorber.execution_profile["pools"].append({
                    "columns": hi - lo, "seconds": perf_counter() - started,
                    "iterations": int(info.iterations), "criterion": getattr(info, "criterion", "unknown"),
                    "converged": bool(info.converged),
                    "stages": dict(getattr(absorber, "_iteration_profile", {})),
                })
            workspace.write(matrix, lo, hi, res)
            infos.append(info)
    finally:
        absorber.acceleration = chosen_accel
        if original_plain_limit is not None:
            absorber.auto_plain_limit = original_plain_limit

    worst = max(infos, key=lambda z: (not z.converged, z.iterations, z.max_update))
    return matrix, worst


def resolve_method(method, groups=None, slopes=None, intercepts=None, *, backend="numpy", group_mode=False, individual=False):
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


def resolve_acceleration(method, acceleration, *, group_mode=False):
    if acceleration == "auto":
        return "none" if group_mode or method != "map" else "cg"
    return acceleration


def check_absorption(info, *, allow_nonconverged=False):
    if not info.converged and not allow_nonconverged:
        raise HDFEConvergenceError(info)


def residualize_in_basis(absorber, y, X, basis, pool_size, memory_budget_mb,
                         *, adaptive_auto=False, allow_nonconverged=False):
    """Reproject original columns without changing a shared solver's tolerance."""
    block = np.column_stack((y, X))
    _apply_absorption_basis(block[:, 1:], basis)
    requested_tol = absorber.tol
    try:
        absorber.tol = min(requested_tol, 1e-11)
        within, info = residualize_blocks(
            absorber, block, pool_size, memory_budget_mb, adaptive_auto=adaptive_auto,
        )
        check_absorption(info, allow_nonconverged=allow_nonconverged)
    finally:
        absorber.tol = requested_tol
    _apply_absorption_basis(within[:, 1:], basis, restore=True)
    return within[:, 0], within[:, 1:], info


def conditioning_info(basis, stage, tolerance):
    """``absorb_info["conditioning"]`` for a weak-direction absorption basis."""
    return {"basis": "symmetric_weak_direction_scaling", "stage": stage,
            "condition": basis[3], "tolerance": tolerance}


def absorb_metadata(plan, absorber, info, pool_size, solver_selection=None):
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
    if absorber.method == "within":
        out["within"] = absorber.resource_info
    if solver_selection is not None:
        out["solver_selection"] = dict(solver_selection)
    if hasattr(absorber, "execution_plan"):
        out["execution_plan"] = absorber.execution_plan.as_dict()
    if hasattr(absorber, "execution_memory"):
        out["execution_memory"] = dict(absorber.execution_memory)
    if hasattr(absorber, "solve_info"):
        si = absorber.solve_info
        out["two_way"] = {
            "connected_components": int(si.connected_components),
            "cross_nnz": int(si.cross_nnz),
            "solved_side_levels": int(si.solved_side_levels),
            "eliminated_side_levels": int(si.eliminated_side_levels),
        }
    return out


def block_absorb_metadata(plan, projector, info, *, method, transform, acceleration,
                          pool_size, solver_selection, heterogeneous_spec):
    """``absorb_info`` for a heterogeneous-spec (``BlockDesign``) projection."""
    resources = projector.resource_info if projector is not None else {"blocks": ()}
    actual_threads = [
        int(item["projector"].get("actual_threads") or 1)
        for item in resources.get("blocks", ()) if item.get("projector") is not None
    ]
    return {
        "canonicalization": plan.as_dict() if plan is not None else None,
        "method": method,
        "transform": transform,
        "acceleration": acceleration,
        "projection_backend": "heterogeneous_components",
        "absorb_threads": max(actual_threads, default=1),
        "iterations": int(info.iterations),
        "converged": bool(info.converged),
        "convergence_metric": float(info.max_update),
        "convergence_criterion": str(info.criterion),
        "pool_size": int(pool_size),
        "solver_selection": dict(solver_selection),
        "heterogeneous_spec": heterogeneous_spec,
    }



def compile_design(data, specs, n, prefix, *, row_mask=None, absorbed_groups=(), absorbed_names=(), structural=True, protected_terms=(), omit=None):
    return build_design(
        data, specs, n, prefix=prefix, row_mask=row_mask,
        absorbed_groups=absorbed_groups, absorbed_names=absorbed_names,
        structural=structural, protected_terms=protected_terms, omit=omit,
    )


def compile_execution_design(
    data, specs, n, prefix, *, groups=(), expected_passes=1,
    memory_budget_mb=None, row_mask=None, absorbed_groups=(), absorbed_names=(),
    structural=True, protected_terms=(), omit=None,
):
    """Compile a design for execution, using heterogeneous-spec storage when useful.

    This is an internal performance contract. Statistical parsing and omission
    semantics are identical to ``compile_design``; only the physical representation
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


def check_collinearity_mode(mode):
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


def collinearity_metadata(
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


def design_collinearity_metadata(plan, design, role):
    """Collinearity metadata of one compiled design role, including structural/user omissions."""
    return collinearity_metadata(
        plan, design.origins, structural_plan=design.structural_plan,
        requested_origins=design.requested_origins, requested_names=design.requested_names,
        role=role, user_omissions=design.user_omissions,
    )


def resolve_ols_columns(Xw, X0, names, *, weights, tolerance, mode, structural_plan=None, _grams=None):
    plan = resolve_collinearity(
        Xw, names=names, role="regressor", original=X0, weights=weights,
        tolerance=tolerance,
        _gram=None if _grams is None else _grams[0],
        _original_gram=None if _grams is None else _grams[1],
    )
    _enforce_collinearity_mode(plan, mode, structural_plan)
    idx = tuple(plan.active_indices)
    if idx == tuple(range(Xw.shape[1])):
        return Xw, X0, plan
    return _select_design_columns(Xw, idx), _select_design_columns(X0, idx), plan


def resolve_iv_columns(Cw, Ew, Zw, C0, E0, Z0, c_names, e_names, z_names, *, weights, tolerance, mode, c_design=None, e_design=None, z_design=None, _refine_controls=None):
    cplan = resolve_collinearity(
        Cw, names=c_names, role="exogenous", original=C0, weights=weights, tolerance=tolerance,
    )
    _enforce_collinearity_mode(cplan, mode, None if c_design is None else c_design.structural_plan)
    ci = tuple(cplan.active_indices)
    Cwa, C0a = (Cw, C0) if ci == tuple(range(Cw.shape[1])) else (_select_design_columns(Cw, ci), _select_design_columns(C0, ci))
    cn = tuple(cplan.active_names)
    # Resolve redundant controls first, then refine the retained basis before
    # it protects the endogenous and excluded-instrument column selections.
    if _refine_controls is not None:
        Cwa = _refine_controls(C0a, Cwa)

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
    def meta(plan, design, role):
        if design is None:
            return collinearity_metadata(plan, role=role)
        return design_collinearity_metadata(plan, design, role)

    info = {
        "exogenous": meta(cplan, c_design, "exogenous"),
        "endogenous": meta(eplan, e_design, "endogenous"),
        "excluded_instruments": meta(zplan, z_design, "excluded_instrument"),
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
    specs = [fixed_effect_spec(v) for v in expand_factor_variables(absorb_terms(absorb))]
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
        raw_group = column_values(data, spec.group)
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
    specs = _cluster_specs(cluster)
    out = []
    for c in specs:
        vals = compress_constant(column_values(data, c), comp, name=f"cluster {c!r}")[keep]
        codes, _ = factorize_1d(vals)
        out.append(codes)
    return out


def _group_optional_array(data, spec, comp, keep, label):
    if spec is None:
        return None
    return np.asarray(compress_constant(column_values(data, spec), comp, name=label))[keep]


def prepare_group_mode(data, *, group, individual, absorb, arrays, weights, cluster, time, panel,
                        drop_singletons, weight_type, aggregation):
    raw_group = column_values(data, group)
    group_codes, _ = factorize_1d(raw_group)
    comp = group_compression(group_codes)
    standard_specs, indiv_spec = _split_individual_absorb(absorb, individual)
    std_groups0, std_slopes0, std_intercepts, std_names = _encode_absorb_grouped(data, standard_specs, comp)

    compressed_arrays = [compress_constant(a, comp, name=name) for name, a in arrays]
    w0 = None
    if weights is not None:
        w0 = compress_constant(column_values(data, weights), comp, name="weights").astype(np.float64, copy=False)
    wkind = normalize_weight_type(weight_type, has_weights=w0 is not None)
    if indiv_spec is not None and wkind == "fweight":
        # Current reghdfe's Factor_Indiv.mata explicitly notes that fweights are
        # meaningless because group/individual membership rows cannot be duplicated.
        raise ValueError("fweights are not supported with group()+individual() fixed effects")
    _ = prepare_weights(w0, comp.n_groups, wkind)

    indiv_codes0 = None
    indiv_slopes0 = None
    if indiv_spec is not None:
        raw_ind = column_values(data, indiv_spec.group)
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
            # reghdfe applies the fweight-mass rule to group()-level FEs too.
            keep, dropped = iterative_singleton_mask(
                std_groups0, w0 if wkind == "fweight" else None,
                frequency_weights=(wkind == "fweight"),
            )
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


def _slope_build_bytes(groups, slopes):
    """Slope copies, centered vectors, means and Gram/inverse work of slope FEs."""
    total = 0
    for codes, slope in zip(groups, slopes, strict=True):
        if slope is not None:
            width = 1 if slope.ndim == 1 else slope.shape[1]
            levels = int(codes.max()) + 1 if len(codes) else 0
            total += 8 * (len(codes) * (width + 5) + levels * (3 * width**2 + width))
    return total


def make_standard_absorber(
    groups, slopes, intercepts, *, weights, tol, max_iter, backend, method,
    transform, acceleration, preconditioner, projection_backend, absorb_threads,
    memory_budget_mb,
):
    if method == "within":
        if backend != "numpy" or any(s is not None for s in slopes) or not all(intercepts):
            raise ValueError("method='within' requires CPU pure-intercept categorical FEs")
    if method == "twoway":
        if backend != "numpy":
            raise ValueError("method='twoway' currently requires backend='numpy'")
        if len(groups) != 2 or any(s is not None for s in slopes) or not all(intercepts):
            raise ValueError(
                "method='twoway' requires exactly two intercept-only fixed effects "
                "after FE canonicalization"
            )
    build_bytes = _absorption_build_cost(groups, "twoway" if method == "twoway" else "generic")
    build_bytes += _slope_build_bytes(groups, slopes)
    if backend == "numpy" and groups:
        # Warm the saturation cache before construction reserves workspace.
        # Small samples/explicit requests do not run a calibration probe.
        resolve_threads(absorb_threads, nobs=len(groups[0]))
    with scheduler.reserve_memory(build_bytes, label="linear_fe_build"):
        if method == "within":
            from .hdfe.within_backend import WithinFEAbsorber
            return WithinFEAbsorber(
                groups, weights=weights, tol=tol, max_iter=max_iter,
                absorb_threads=absorb_threads, projection_memory_budget_mb=memory_budget_mb,
            )
        if method == "twoway":
            return TwoWayFEAbsorber(
                groups, weights=weights, tol=tol, max_iter=max_iter
            )._configure_execution(absorb_threads=absorb_threads)
        return HDFEAbsorber(
            groups, slopes=slopes, intercepts=intercepts, weights=weights,
            tol=tol, max_iter=max_iter, backend=backend, method=method,
            transform=transform, acceleration=acceleration,
            preconditioner=preconditioner, projection_backend=projection_backend,
            absorb_threads=absorb_threads, projection_memory_budget_mb=memory_budget_mb,
        )


def make_group_absorber(prep, *, method, transform, backend, acceleration, preconditioner, tol, max_iter, aggregation, incidence_backend, memory_budget_mb, projection_backend="auto", absorb_threads="auto"):
    indiv = prep["individual_spec"]
    if indiv is None:
        return make_standard_absorber(
            prep["groups"], prep["slopes"], prep["intercepts"],
            weights=prep["weights"], tol=tol, max_iter=max_iter, backend=backend,
            method=method, transform=transform, acceleration=acceleration,
            preconditioner=preconditioner, projection_backend=projection_backend,
            absorb_threads=absorb_threads, memory_budget_mb=memory_budget_mb,
        )
    if backend != "numpy":
        raise ValueError("group()+individual() currently uses SciPy LSMR/LSQR on CPU; backend must be 'numpy'")
    if acceleration != "none":
        raise ValueError("CG/MAP acceleration is unavailable with group()+individual(); use LSMR/LSQR")
    solver = "lsmr" if method == "map" else method
    if solver not in {"lsmr", "lsqr"}:
        raise ValueError("group()+individual() requires method='lsmr' or method='lsqr'")
    membership = prep["membership_group"]
    n = int(membership.max()) + 1 if len(membership) else len(prep["groups"][0])
    ni = int(prep["individual_codes"].max()) + 1 if len(membership) else 0
    individual_slopes = prep["individual_slopes"]
    p = 0 if individual_slopes is None else (1 if individual_slopes.ndim == 1 else individual_slopes.shape[1])
    q = int(indiv.intercept) + p
    widths = [int(has_int) + (0 if slope is None else (1 if slope.ndim == 1 else slope.shape[1]))
              for has_int, slope in zip(prep["intercepts"], prep["slopes"], strict=True)]
    nnz = n * sum(widths) + len(membership) * q
    coefficient_count = ni * q + sum((int(g.max()) + 1) * width for g, width in zip(prep["groups"], widths, strict=True) if len(g))
    base_bytes = 32 * n + 32 * len(membership) + 24 * coefficient_count
    csr_bytes = base_bytes + 64 * nnz + 16 * n
    selected = str(incidence_backend).lower().replace("-", "_")
    if selected == "auto":
        allocation = scheduler.current_allocation()
        available = int(memory_budget_mb * 1024**2) if allocation is None else allocation.unreserved_memory_bytes()
        # Preserve the constructor's sparse heuristic, but allow its matrix-free
        # alternative when concurrent workspace leaves less construction room.
        selected = "csr" if (32 * nnz + 16 * n <= .45 * memory_budget_mb * 1024**2
                             and csr_bytes <= available) else "matrix_free"
    build_bytes = csr_bytes if selected == "csr" else base_bytes + _absorption_build_cost(prep["groups"], "generic")
    if selected != "csr":
        build_bytes += _slope_build_bytes(prep["groups"], prep["slopes"])
    with scheduler.reserve_memory(build_bytes, label="group_individual_build"):
        return GroupIndividualAbsorber(
            prep["groups"], standard_slopes=prep["slopes"], standard_intercepts=prep["intercepts"],
            membership_group=prep["membership_group"], individual_codes=prep["individual_codes"],
            individual_slopes=prep["individual_slopes"], individual_intercept=indiv.intercept,
            aggregation=aggregation, weights=prep["weights"], tol=tol, max_iter=max_iter,
            method=solver, preconditioner=preconditioner, incidence_backend=selected,
            memory_budget_mb=memory_budget_mb,
        )


def group_dof(prep, absorber, dof_method):
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


def group_has_intercept(prep):
    """Whether the group()/individual() FE system contains an intercept."""
    return bool(any(prep["intercepts"])) or bool(
        prep.get("individual_spec") is not None and prep["individual_spec"].intercept
    )


def group_info(prep, absorber, aggregation):
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


