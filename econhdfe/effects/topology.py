from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit

from ..hdfe.rank import categorical_rank, _union_edge_pairs as _union_pairs


@njit(cache=True, nogil=True)
def _compress(parent):
    out = np.empty(parent.size, dtype=np.int64)
    for i in range(parent.size):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        out[i] = x
    return out


@dataclass(frozen=True, slots=True)
class FEComponentIdentification:
    """Exact identification status for one disconnected FE-incidence block."""

    component: int
    n_obs: int
    n_levels: int
    rank: int
    nullity: int
    structural_shift_nullity: int
    extra_nullity: int

    @property
    def estimable(self) -> bool:
        return self.extra_nullity == 0


@dataclass(frozen=True, slots=True)
class FEIdentification:
    """Identification metadata for intercept-only categorical fixed effects."""

    rank: int
    n_levels: int
    nullity: int
    n_components: int
    structural_shift_nullity: int
    extra_nullity: int
    component_by_term: tuple[np.ndarray, ...]
    exact_rank: bool = True
    components: tuple[FEComponentIdentification, ...] = ()

    @property
    def structural_normalization_complete(self) -> bool:
        return self.extra_nullity == 0

    @property
    def identified_components(self) -> tuple[int, ...]:
        return tuple(c.component for c in self.components if c.estimable)

    @property
    def unidentified_components(self) -> tuple[int, ...]:
        return tuple(c.component for c in self.components if not c.estimable)


def identification_structure(groups, *, rank_backend="auto") -> FEIdentification:
    """Return exact rank/nullity plus incidence-connected components.

    For K categorical FE dimensions, every observation connects one level from
    each dimension.  The component labels are therefore obtained without
    materializing a dummy matrix.  Each connected component contributes at
    least K-1 additive shift directions; ``extra_nullity`` reports dependencies
    beyond those ordinary shifts.
    """
    groups = tuple(np.asarray(g, dtype=np.int32) for g in groups)
    if not groups:
        return FEIdentification(0, 0, 0, 0, 0, 0, ())
    n = len(groups[0])
    if any(len(g) != n for g in groups):
        raise ValueError("all fixed-effect groups must have equal length")

    levels = tuple(int(g.max()) + 1 if g.size else 0 for g in groups)
    offsets = np.cumsum((0,) + levels[:-1], dtype=np.int64)
    total = int(sum(levels))
    if total == 0:
        return FEIdentification(0, 0, 0, 0, 0, 0, tuple(np.empty(0, dtype=np.int32) for _ in groups))

    parent = np.arange(total, dtype=np.int64)
    size = np.ones(total, dtype=np.int64)
    base = groups[0].astype(np.int64, copy=False) + offsets[0]
    for j in range(1, len(groups)):
        other = groups[j].astype(np.int64, copy=False) + offsets[j]
        _union_pairs(parent, size, base, other)
    roots = _compress(parent)
    _, comp_all = np.unique(roots, return_inverse=True)
    comp_all = comp_all.astype(np.int32, copy=False)
    comp_by_term = tuple(
        comp_all[int(offsets[j]): int(offsets[j]) + levels[j]].copy()
        for j in range(len(groups))
    )
    ncomp = int(comp_all.max()) + 1 if comp_all.size else 0

    rank = int(categorical_rank(list(groups), assume_dense=True, backend=rank_backend))
    nullity = int(total - rank)
    shifts_per_component = max(len(groups) - 1, 0)
    structural = int(ncomp * shifts_per_component)
    extra = int(max(nullity - structural, 0))

    # Component-level exact ranks are only needed on the exceptional path where
    # the global design has additional dependencies.  In the common case,
    # additivity of block-diagonal ranks implies every component has only the
    # ordinary K-1 shift directions, so we avoid C separate rank calculations.
    obs_comp = comp_by_term[0][groups[0]] if n else np.empty(0, dtype=np.int32)
    obs_counts = np.bincount(obs_comp, minlength=ncomp) if ncomp else np.empty(0, dtype=np.int64)
    level_counts = np.zeros(ncomp, dtype=np.int64)
    for term_comp in comp_by_term:
        level_counts += np.bincount(term_comp, minlength=ncomp)

    components = []
    if extra == 0:
        for comp in range(ncomp):
            nl = int(level_counts[comp])
            q = int(shifts_per_component)
            components.append(FEComponentIdentification(
                component=comp, n_obs=int(obs_counts[comp]), n_levels=nl,
                rank=nl - q, nullity=q,
                structural_shift_nullity=q, extra_nullity=0,
            ))
    else:
        for comp in range(ncomp):
            mask = obs_comp == comp
            local = []
            for g in groups:
                _, inv = np.unique(g[mask], return_inverse=True)
                local.append(inv.astype(np.int32, copy=False))
            nl = int(sum((int(g.max()) + 1 if g.size else 0) for g in local))
            cr = int(categorical_rank(local, assume_dense=True, backend=rank_backend))
            cq = int(nl - cr)
            cstruct = int(shifts_per_component)
            components.append(FEComponentIdentification(
                component=comp, n_obs=int(mask.sum()), n_levels=nl,
                rank=cr, nullity=cq, structural_shift_nullity=cstruct,
                extra_nullity=int(max(cq - cstruct, 0)),
            ))

    return FEIdentification(
        rank=rank,
        n_levels=total,
        nullity=nullity,
        n_components=ncomp,
        structural_shift_nullity=structural,
        extra_nullity=extra,
        component_by_term=comp_by_term,
        components=tuple(components),
    )
