from __future__ import annotations
import numpy as np
from .projection import estimate_index_bytes


def _absorption_workspace_cost(absorber):
    """Conservative byte inventory for pooled CPU absorption, excluding input."""
    if getattr(absorber, "method", None) == "within":
        return absorber.workspace_cost()
    arrays = {}

    def add(a):
        if isinstance(a, np.ndarray):
            arrays[id(a)] = a.nbytes

    for name in ("groups", "slopes", "_denoms"):
        for a in getattr(absorber, name, ()) or ():
            add(a)
    for name in ("weights", "_projection_scratch", "left", "right",
                 "denom_left", "denom_right", "_diag", "_precond_diag", "keep_right"):
        add(getattr(absorber, name, None))
    for proj in getattr(absorber, "_projectors", ()):
        for name in ("denom", "slope_means", "slope_gram_inv", "slope_inv_ss"):
            add(getattr(proj, name))
    fixed = int(getattr(absorber, "index_bytes", 0)) + sum(arrays.values())
    n = int(absorber.nobs)
    if absorber.method == "twoway":
        # Includes the lazy group indexes and bounded coefficient-space PCG
        # arrays/advanced-indexing temporaries, not an N x FE dummy matrix.
        if absorber.absorb_threads > 1 and absorber._left_index is None:
            fixed += estimate_index_bytes(n, [absorber.n_left, absorber.n_right])
        per_col = 8 * (n + 4 * absorber.n_left + 20 * absorber.n_right)
    else:
        vectors = 4 if absorber.acceleration in ("cg", "auto") else 2
        per_col = 8 * n * vectors
        if absorber.acceleration in ("cg", "auto") and absorber.absorb_threads > 1:
            # At most 16 partial reductions plus their per-column result.
            per_col += 17 * 8
        indexes = getattr(absorber, "_group_indexes", ())
        levels = getattr(absorber, "levels", ())
        # Fused fallback accumulators are reused across FE, so reserve only
        # the largest group matrix. This also bounds one-column bincount work.
        unindexed = [L for i, L in enumerate(levels) if not indexes or indexes[i] is None]
        if unindexed:
            per_col += 8 * max(unindexed) * 3
        if any(s is not None for s in getattr(absorber, "slopes", ())):
            per_col += 8 * n * 4
        core = getattr(absorber, "_core_plan", None)
        if core is not None:
            fixed += core.core_index.nbytes + sum(g.nbytes for g in core.core_groups)
            core_levels = [int(g.max()) + 1 if g.size else 0 for g in core.core_groups]
            fixed += estimate_index_bytes(core.core_nobs, core_levels) + 8 * sum(core_levels)
            per_col += 8 * core.core_nobs
    return int(fixed), max(1, int(per_col))


def _absorption_build_cost(groups, kind):
    """Bound known constructor arrays without first constructing FE indexes.

    Two-way construction holds sorted unique keys/inverses, edge arrays, CSR
    storage and component/diagonal work concurrently (edges <= observations).
    Generic construction can retain group indexes plus numerical-core copies.
    These byte inventories exclude caller-owned groups/weights and allocator
    overhead; they are cooperative admission estimates, not hard RSS limits.
    """
    n = len(groups[0]) if groups else 0
    levels = [int(g.max()) + 1 if len(g) else 0 for g in groups]
    if kind == "twoway":
        return int(96 * n + 64 * sum(levels))
    indexes = estimate_index_bytes(n, levels)
    return int(indexes + 24 * n * len(groups) + 32 * sum(levels) + 32 * n)


def _absorption_pending_cost(absorber):
    """Known lazy topology allocations not yet represented in process RSS."""
    if absorber.method == "twoway":
        if absorber.absorb_threads > 1 and absorber._left_index is None:
            return estimate_index_bytes(absorber.nobs, [absorber.n_left, absorber.n_right])
        return 0
    core = getattr(absorber, "_core_plan", None)
    if core is not None and getattr(absorber, "_core_absorber", None) is None:
        return _absorption_build_cost(core.core_groups, "generic")
    return 0
