from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit
from .rank import categorical_prefix_ranks


@dataclass(slots=True)
class DofInfo:
    """Serializable absorbed degrees-of-freedom metadata.

    Produced by the HDFE DoF engine and carried unchanged on result objects
    (``RegressionResult.dof_info``); ``econhdfe.results`` re-exports it.
    """
    df_absorbed: int
    initial: int
    redundant: int
    nested: int
    k_by_component: tuple[int, ...]
    m_by_component: tuple[int, ...]
    exact_by_component: tuple[bool, ...]
    nested_by_component: tuple[bool, ...]
    method: str


@njit(cache=True)
def _bipartite_components_dense(a, b, n1, n2):
    m = n1 + n2
    parent = np.arange(m, dtype=np.int64)
    size = np.ones(m, dtype=np.int64)
    for i in range(a.size):
        x = int(a[i]); y = int(b[i]) + n1
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        while parent[y] != y:
            parent[y] = parent[parent[y]]; y = parent[y]
        if x != y:
            if size[x] < size[y]: x, y = y, x
            parent[y] = x; size[x] += size[y]
    c = 0
    for i in range(m):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        if x == i:
            c += 1
    return c


@njit(cache=True)
def _slope_redundancy(codes, x, nlev, has_intercept, tol):
    # Mirrors reghdfe's continuous-DoF logic: with an intercept, a slope is
    # redundant in groups where x is constant; without one, only groups where
    # x is identically zero are redundant.
    if has_intercept:
        lo = np.full(nlev, np.inf)
        hi = np.full(nlev, -np.inf)
        seen = np.zeros(nlev, dtype=np.uint8)
        for i in range(codes.size):
            g = int(codes[i]); v = float(x[i])
            if v < lo[g]: lo[g] = v
            if v > hi[g]: hi[g] = v
            seen[g] = 1
        out = 0
        for g in range(nlev):
            if seen[g] and hi[g] - lo[g] <= tol:
                out += 1
        return out
    maxabs = np.zeros(nlev, dtype=np.float64)
    seen = np.zeros(nlev, dtype=np.uint8)
    for i in range(codes.size):
        g = int(codes[i]); v = abs(float(x[i]))
        if v > maxabs[g]: maxabs[g] = v
        seen[g] = 1
    out = 0
    for g in range(nlev):
        if seen[g] and maxabs[g] <= tol:
            out += 1
    return out


def _dense(g):
    g = np.asarray(g)
    if g.dtype.kind in "iu" and g.size and g.min() >= 0 and np.unique(g).size == int(g.max()) + 1:
        return g.astype(np.int64, copy=False), int(g.max()) + 1
    _, inv = np.unique(g, return_inverse=True)
    return inv.astype(np.int64, copy=False), int(inv.max()) + 1 if inv.size else 0


def _nested_within(fe: np.ndarray, cluster: np.ndarray) -> bool:
    fe = np.asarray(fe, dtype=np.int64)
    cluster = np.asarray(cluster, dtype=np.int64)
    nlev = int(fe.max()) + 1 if fe.size else 0
    # A FE is nested if every FE level maps to exactly one cluster. Sorting is
    # avoided; min/max cluster id per FE gives an O(N) check with O(G) memory.
    lo = np.full(nlev, np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(nlev, -1, dtype=np.int64)
    np.minimum.at(lo, fe, cluster)
    np.maximum.at(hi, fe, cluster)
    return bool(np.all(lo == hi))


def pairwise_components(a: np.ndarray, b: np.ndarray, *, assume_dense: bool = False) -> int:
    if assume_dense:
        aa = np.asarray(a, dtype=np.int64); bb = np.asarray(b, dtype=np.int64)
        n1 = int(aa.max()) + 1 if aa.size else 0
        n2 = int(bb.max()) + 1 if bb.size else 0
    else:
        aa, n1 = _dense(a); bb, n2 = _dense(b)
    return int(_bipartite_components_dense(aa, bb, n1, n2))


def absorbed_dof(
    groups: list[np.ndarray],
    *,
    slopes: list[np.ndarray | None] | None = None,
    intercepts: list[bool] | None = None,
    clusters: list[np.ndarray] | None = None,
    method: str = "pairwise",
    adjust_nested: bool = True,
    adjust_continuous: bool = True,
    continuous_tol: float = 1e-6,
    groups_are_dense: bool = False,
) -> DofInfo:
    """Absorbed degrees of freedom with reghdfe-compatible adjustments.

    ``pairwise`` reproduces reghdfe's conservative mobility-group treatment
    for the third and later intercept FEs. ``exact`` instead computes the
    rational rank of the complete active categorical dummy design, while
    retaining reghdfe's factor-level cluster-nesting convention. Continuous
    slope adjustments remain separate from the categorical rank engine.
    """
    if method not in {"exact", "pairwise", "firstpair", "none"}:
        raise ValueError("dof method must be 'exact', 'pairwise', 'firstpair', or 'none'")
    G = len(groups)
    if G == 0:
        return DofInfo(0, 0, 0, 0, (), (), (), (), method)
    slopes = [None] * G if slopes is None else list(slopes)
    intercepts = [True] * G if intercepts is None else list(intercepts)
    if len(slopes) != G or len(intercepts) != G:
        raise ValueError("groups/slopes/intercepts length mismatch")

    levels = [(int(np.asarray(g).max()) + 1 if len(g) else 0) if groups_are_dense else int(np.unique(g).size) for g in groups]
    K: list[int] = []
    M: list[int] = []
    exact: list[bool] = []
    nested_flags: list[bool] = []
    intercept_component_index: list[int | None] = []
    component_to_group: list[int] = []

    for gi in range(G):
        idx = None
        if intercepts[gi]:
            idx = len(K)
            K.append(levels[gi]); M.append(0); exact.append(False); nested_flags.append(False)
            component_to_group.append(gi)
        intercept_component_index.append(idx)
        s = slopes[gi]
        ns = 0 if s is None else (1 if np.asarray(s).ndim == 1 else np.asarray(s).shape[1])
        for _ in range(ns):
            K.append(levels[gi]); M.append(0); exact.append(False); nested_flags.append(False)
            component_to_group.append(gi)

    nested_total = 0
    active_intercepts: list[tuple[int, int]] = []  # (component idx, group idx)
    for gi, ci in enumerate(intercept_component_index):
        if ci is None:
            continue
        is_nested = False
        if adjust_nested and clusters:
            is_nested = any(_nested_within(groups[gi], c) for c in clusters)
        if is_nested:
            M[ci] = K[ci]
            exact[ci] = True
            nested_flags[ci] = True
            nested_total += K[ci]
        else:
            active_intercepts.append((ci, gi))

    # Structural collinearity across intercept FE sets: every intercept after
    # the first has at least one redundant coefficient even if mobility-group
    # calculations are disabled.
    for pos, (ci, _) in enumerate(active_intercepts):
        if pos > 0 or nested_total > 0:
            M[ci] = max(M[ci], 1)
        if pos == 0:
            exact[ci] = True

    if method == "exact" and active_intercepts:
        active_groups = [groups[gi] for _, gi in active_intercepts]
        ranks = categorical_prefix_ranks(active_groups, assume_dense=groups_are_dense)
        prev = 0
        for pos, ((ci, _), rank_now) in enumerate(zip(active_intercepts, ranks, strict=False)):
            added = int(rank_now) - int(prev)
            redundancy = int(K[ci]) - added
            if pos == 0 and nested_total > 0:
                # reghdfe treats nested cluster FEs as a free normalization
                # anchor, so the first remaining intercept also loses one DoF.
                redundancy += 1
            M[ci] = min(int(K[ci]), redundancy)
            exact[ci] = True
            prev = int(rank_now)
    elif method in {"pairwise", "firstpair"} and len(active_intercepts) >= 2:
        pair_count = 0
        for i in range(len(active_intercepts)):
            ci, gi = active_intercepts[i]
            for j in range(i + 1, len(active_intercepts)):
                cj, gj = active_intercepts[j]
                m = pairwise_components(groups[gi], groups[gj], assume_dense=groups_are_dense)
                M[cj] = max(M[cj], m)
                pair_count += 1
                if pair_count == 1:
                    exact[cj] = True
                if method == "firstpair":
                    break
            if method == "firstpair" and pair_count:
                break

    if adjust_continuous:
        comp = 0
        for gi in range(G):
            if intercepts[gi]:
                comp += 1
            s = slopes[gi]
            if s is None:
                continue
            S = np.asarray(s, dtype=np.float64)
            if S.ndim == 1:
                S = S[:, None]
            for j in range(S.shape[1]):
                red = int(_slope_redundancy(
                    np.asarray(groups[gi], dtype=np.int64), S[:, j], levels[gi],
                    bool(intercepts[gi]), float(continuous_tol)
                ))
                M[comp] = max(M[comp], red)
                # The zero/constant check itself is exact for this source of
                # redundancy, but there may still be cross-FE collinearity.
                comp += 1

    initial = int(sum(K)); redundant = int(sum(M))
    return DofInfo(
        df_absorbed=initial - redundant,
        initial=initial,
        redundant=redundant,
        nested=nested_total,
        k_by_component=tuple(map(int, K)),
        m_by_component=tuple(map(int, M)),
        exact_by_component=tuple(map(bool, exact)),
        nested_by_component=tuple(map(bool, nested_flags)),
        method=method,
    )
