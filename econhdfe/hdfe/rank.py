from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Literal
import numpy as np
from numba import njit
from .codes import validate_dense_codes


RankBackend = Literal["auto", "native", "sympy", "flint"]

# The fast GF(2) certificate is exact when it reaches the characteristic-zero
# upper bound: a nonzero minor modulo 2 is also nonzero over Q.  The limits
# keep pathological rank-deficient cores from turning the certificate pass
# into an unbounded memory/time sink.
_GF2_MAX_COLS = 50_000
_GF2_MIN_XOR_BUDGET = 1_000_000
_GF2_MAX_XOR_BUDGET = 50_000_000

# python-flint currently exposes dense fmpz_mat objects.  Keep its automatic
# use behind a conservative cell cap; users can request backend="flint" and a
# larger cap explicitly when memory is known to be ample.
_DEFAULT_FLINT_MAX_CELLS = 4_000_000
_EXACT_PRIME = 2_147_483_647

# Per-component native-elimination safety limits, shared by the modular and
# rational passes. These are cooperative arithmetic/storage guards, NOT a hard
# process RSS or wall-clock limit. Optional external backends are not metered.
_NATIVE_MAX_WORK = 50_000_000
_NATIVE_MAX_BYTES = 256 * 1024 * 1024
_NATIVE_MAX_INTEGER_BITS = 16_384


class _ExactRankResourceError(RuntimeError):
    """No exact answer was computed; never substitute a finite-field bound."""

    def __init__(self, stage: str, resource: str, used: int, limit: int):
        self.stage, self.resource = stage, resource
        self.used, self.limit = used, limit
        super().__init__(
            f"exact rank resource limit: stage={stage}, {resource}={used}, "
            f"limit={limit}; no exact rank returned"
        )


def _row_storage(entries: int, bits: int = 1) -> int:
    # Conservative accounting for a dict, keys, Python integers and spare
    # hash-table capacity. Input arrays, interpreter/JIT and allocator overhead
    # are outside this estimate; callers must not report it as measured RSS.
    return 256 + int(entries) * (160 + 4 * ((int(bits) + 29) // 30))


class _NativeRankBudget:
    def __init__(self):
        self.work = 0
        self.peak_bytes = 0
        self.max_integer_bits = 0

    def check(self, stage: str, *, work: int = 0, storage: int = 0,
              bits: int = 0) -> None:
        self.work += int(work)
        self.peak_bytes = max(self.peak_bytes, int(storage))
        self.max_integer_bits = max(self.max_integer_bits, int(bits))
        for resource, used, limit in (
            ("work_units", self.work, _NATIVE_MAX_WORK),
            ("estimated_bytes", self.peak_bytes, _NATIVE_MAX_BYTES),
            ("integer_bits", self.max_integer_bits, _NATIVE_MAX_INTEGER_BITS),
        ):
            if used > limit:
                raise _ExactRankResourceError(stage, resource, used, limit)

    def prepare(self, edges: np.ndarray, stage: str) -> int:
        # Check BEFORE materializing Python row dictionaries. Retain the full
        # initial-row allowance even after rows are consumed (a safe overcount).
        storage = int(len(edges)) * _row_storage(int(edges.shape[1]))
        self.check(stage, work=int(edges.size), storage=storage)
        return storage


@dataclass(frozen=True, slots=True)
class CategoricalRankInfo:
    """Diagnostics for the exact rank of an intercept-only categorical FE design.

    The design matrix is the horizontal concatenation of one-hot FE blocks.
    Duplicate observation rows are discarded because they cannot change rank.
    Degree-one vertices are then peeled exactly before the residual core is
    decomposed and rank-reduced over the rationals.
    """

    rank: int
    n_levels: int
    n_unique_edges: int
    peeled_rank: int
    core_edges: int
    core_vertices: int
    core_components: int


@njit(cache=True, nogil=True)
def _peel_degree_one(edges: np.ndarray, nvertices: int):
    """Exact leaf elimination on a multipartite incidence matrix.

    If a column occurs in exactly one active row, that row/column pair supplies
    one pivot. Removing both preserves rank up to +1.  Linked incidence lists
    plus monotone cursors keep the pass O(number of incidences).
    """
    nedge, width = edges.shape
    head = np.full(nvertices, -1, dtype=np.int64)
    nxt = np.empty(nedge * width, dtype=np.int64)
    edge_of = np.empty(nedge * width, dtype=np.int64)
    degree = np.zeros(nvertices, dtype=np.int64)

    k = 0
    for e in range(nedge):
        for j in range(width):
            v = int(edges[e, j])
            nxt[k] = head[v]
            edge_of[k] = e
            head[v] = k
            degree[v] += 1
            k += 1

    active = np.ones(nedge, dtype=np.uint8)
    cursor = head.copy()
    queue = np.empty(nvertices, dtype=np.int64)
    qhead = 0
    qtail = 0
    for v in range(nvertices):
        if degree[v] == 1:
            queue[qtail] = v
            qtail += 1

    peeled = 0
    while qhead < qtail:
        v = int(queue[qhead])
        qhead += 1
        if degree[v] != 1:
            continue
        inc = int(cursor[v])
        while inc != -1 and active[int(edge_of[inc])] == 0:
            inc = int(nxt[inc])
        cursor[v] = inc
        if inc == -1:
            continue
        e = int(edge_of[inc])
        if active[e] == 0:
            continue
        active[e] = 0
        peeled += 1
        for j in range(width):
            u = int(edges[e, j])
            if degree[u] > 0:
                degree[u] -= 1
                if degree[u] == 1:
                    queue[qtail] = u
                    qtail += 1

    return active, degree, peeled


@njit(cache=True, nogil=True)
def _union_edge_pairs(parent: np.ndarray, size: np.ndarray, left: np.ndarray, right: np.ndarray):
    for k in range(left.size):
        x = int(left[k])
        y = int(right[k])
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x == y:
            continue
        if size[x] < size[y]:
            x, y = y, x
        parent[y] = x
        size[x] += size[y]


def _properly_connected(component_edges: np.ndarray) -> bool:
    """Whether a uniform hypergraph is connected by one-coordinate changes.

    Consecutive hyperedges are adjacent when they share G-1 vertices.  For a
    G-partite G-uniform component, proper connectivity implies that each FE
    partition is one pivot-equivalence class, hence the incidence rank is
    exactly V-G+1 in characteristic zero.
    """
    nedge, width = component_edges.shape
    if nedge <= 1:
        return True
    parent = np.arange(nedge, dtype=np.int64)
    size = np.ones(nedge, dtype=np.int64)
    for omit in range(width):
        keep = [j for j in range(width) if j != omit]
        sig = np.ascontiguousarray(component_edges[:, keep])
        view = sig.view(np.dtype([(f"f{j}", sig.dtype) for j in range(sig.shape[1])])).reshape(-1)
        _, inv = np.unique(view, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        labels = inv[order]
        same = labels[1:] == labels[:-1]
        if np.any(same):
            _union_edge_pairs(parent, size, order[:-1][same], order[1:][same])
    roots = parent.copy()
    for i in range(nedge):
        x = i
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        roots[i] = x
    return bool(np.all(roots == roots[0]))


@njit(cache=True, nogil=True)
def _edge_component_roots(edges: np.ndarray, nvertices: int):
    """Union vertices appearing in each edge and return one root per edge."""
    nedge, width = edges.shape
    parent = np.arange(nvertices, dtype=np.int64)
    size = np.ones(nvertices, dtype=np.int64)

    for e in range(nedge):
        base = int(edges[e, 0])
        # Find root of base.
        x = base
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        rb = x
        for j in range(1, width):
            y = int(edges[e, j])
            while parent[y] != y:
                parent[y] = parent[parent[y]]
                y = parent[y]
            ry = y
            if rb != ry:
                if size[rb] < size[ry]:
                    rb, ry = ry, rb
                parent[ry] = rb
                size[rb] += size[ry]

    roots = np.empty(nedge, dtype=np.int64)
    for e in range(nedge):
        x = int(edges[e, 0])
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        roots[e] = x
    return roots


def _levels(groups: list[np.ndarray] | tuple[np.ndarray, ...], assume_dense: bool):
    dense: list[np.ndarray] = []
    levels: list[int] = []
    n = None
    for raw in groups:
        g = np.asarray(raw)
        if g.ndim != 1:
            raise ValueError("categorical FE groups must be one-dimensional")
        if n is None:
            n = len(g)
        elif len(g) != n:
            raise ValueError("categorical FE groups must have equal length")
        if assume_dense:
            gg = validate_dense_codes(g, dtype=np.int64)
            if gg.size and int(gg.min()) < 0:
                raise ValueError("dense FE codes must be nonnegative")
            L = int(gg.max()) + 1 if gg.size else 0
        else:
            _, inv = np.unique(g, return_inverse=True)
            gg = inv.astype(np.int64, copy=False)
            L = int(gg.max()) + 1 if gg.size else 0
        dense.append(gg)
        levels.append(L)
    return dense, levels


def _unique_edge_indices(groups: list[np.ndarray], levels: list[int]) -> np.ndarray:
    """Indices of unique FE tuples, using one packed uint64 key when possible."""
    if not groups or not len(groups[0]):
        return np.empty(0, dtype=np.int64)
    product = 1
    fits = True
    limit = (1 << 64) - 1
    for L in levels:
        product *= max(int(L), 1)
        if product > limit:
            fits = False
            break
    if fits:
        key = np.asarray(groups[0], dtype=np.uint64).copy()
        for j in range(1, len(groups)):
            key *= np.uint64(levels[j])
            key += np.asarray(groups[j], dtype=np.uint64)
        _, idx = np.unique(key, return_index=True)
        idx.sort()
        return idx.astype(np.int64, copy=False)

    # Extremely high-cardinality products can overflow a mixed-radix uint64.
    # A structured view still avoids Python tuples and remains exact.
    mat = np.empty((len(groups[0]), len(groups)), dtype=np.int64)
    for j, g in enumerate(groups):
        mat[:, j] = g
    view = np.ascontiguousarray(mat).view(
        np.dtype([(f"f{j}", mat.dtype) for j in range(mat.shape[1])])
    ).reshape(-1)
    _, idx = np.unique(view, return_index=True)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def _global_edges(groups: list[np.ndarray], levels: list[int], idx: np.ndarray):
    offsets = np.zeros(len(levels), dtype=np.int64)
    if len(levels) > 1:
        offsets[1:] = np.cumsum(np.asarray(levels[:-1], dtype=np.int64))
    edges = np.empty((len(idx), len(groups)), dtype=np.int64)
    for j, g in enumerate(groups):
        edges[:, j] = np.asarray(g, dtype=np.int64)[idx] + offsets[j]
    return edges, offsets, int(sum(levels))


def _component_edge_lists(edges: np.ndarray, nvertices: int):
    if not len(edges):
        return []
    roots = _edge_component_roots(edges, nvertices)
    order = np.argsort(roots, kind="stable")
    roots_sorted = roots[order]
    cuts = np.flatnonzero(np.r_[True, roots_sorted[1:] != roots_sorted[:-1], True])
    return [order[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


def available_rank_backends() -> tuple[str, ...]:
    """Return exact-rank backends importable in the current environment.

    ``native`` is always available. ``sympy`` uses sparse ``DomainMatrix``
    arithmetic over exact domains. ``flint`` uses python-flint's dense
    ``fmpz_mat.rank`` and is therefore guarded by a cell-count limit.
    """
    out = ["native"]
    try:
        import sympy  # noqa: F401
    except Exception:
        pass
    else:
        out.append("sympy")
    try:
        import flint  # noqa: F401
    except Exception:
        pass
    else:
        out.append("flint")
    return tuple(out)


def _reduced_component_rows(component_edges: np.ndarray):
    """Drop the G-1 universal dummy dependencies and localize columns.

    For a connected G-partite component, one arbitrary column from each FE
    block 2..G is an exact linear combination of block 1 and the remaining
    columns in its own block. Removing those G-1 columns therefore preserves
    rank while reducing the core matrix to at most ``V-G+1`` columns.
    """
    width = int(component_edges.shape[1])
    dropped = {int(component_edges[0, j]) for j in range(1, width)}
    vertices = np.unique(component_edges)
    kept = [int(v) for v in vertices if int(v) not in dropped]
    local = {v: j for j, v in enumerate(kept)}
    rows: list[tuple[int, ...]] = []
    for edge in component_edges:
        rows.append(tuple(local[int(v)] for v in edge if int(v) not in dropped))
    return rows, len(kept)


def _gf2_bitset_rank(
    rows: list[tuple[int, ...]], target_rank: int, *, max_xors: int
) -> int:
    """Fast finite-field lower bound using C-level Python big-int XOR.

    Returning ``target_rank`` is an exact characteristic-zero certificate.
    A smaller value is only a lower bound and must never be treated as the
    final rank.
    """
    pivots: dict[int, int] = {}
    rank = 0
    xors = 0
    for cols in rows:
        bits = 0
        for c in cols:
            bits |= 1 << int(c)
        while bits:
            pivot_col = bits.bit_length() - 1
            pivot = pivots.get(pivot_col)
            if pivot is None:
                pivots[pivot_col] = bits
                rank += 1
                if rank >= target_rank:
                    return rank
                break
            bits ^= pivot
            xors += 1
            if xors >= max_xors:
                return rank
    return rank


def _sympy_sparse_rank(
    rows: list[tuple[int, ...]], ncols: int, target_rank: int
) -> int:
    """Certified sparse exact rank through SymPy DomainMatrix.

    A large-prime pass is tried first. Reaching ``target_rank`` certifies the
    rational rank immediately. Otherwise the matrix is rerun over ZZ so the
    result remains deterministic and exact rather than Monte Carlo.
    """
    try:
        from sympy.polys.domains import GF, ZZ
        from sympy.polys.matrices import DomainMatrix
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError("backend='sympy' requires SymPy") from exc

    field = GF(_EXACT_PRIME)
    one = field.one
    mod_rows = {i: {int(c): one for c in row} for i, row in enumerate(rows)}
    rmod = int(DomainMatrix(mod_rows, (len(rows), ncols), field).rank())
    if rmod >= target_rank:
        return target_rank

    one_z = ZZ.one
    int_rows = {i: {int(c): one_z for c in row} for i, row in enumerate(rows)}
    return int(DomainMatrix(int_rows, (len(rows), ncols), ZZ).rank())


def _flint_dense_rank(
    rows: list[tuple[int, ...]], ncols: int, *, max_cells: int
) -> int:
    """Certified exact rank through python-flint's dense integer matrix API."""
    cells = int(len(rows)) * int(ncols)
    if cells > int(max_cells):
        raise ValueError(
            f"FLINT dense core would require {cells:,} cells; "
            f"limit is {int(max_cells):,}"
        )
    try:
        from flint import fmpz_mat
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ImportError("backend='flint' requires python-flint") from exc

    flat = [0] * cells
    for i, row in enumerate(rows):
        base = i * ncols
        for c in row:
            flat[base + int(c)] = 1
    return int(fmpz_mat(len(rows), ncols, flat).rank())


def _column_order(component_edges: np.ndarray) -> dict[int, int]:
    flat = component_edges.ravel()
    vertices, counts = np.unique(flat, return_counts=True)
    # Static minimum-degree ordering is a cheap Markowitz surrogate for the
    # sparse exact fallback. Stable vertex-id tie breaking is deterministic.
    order = np.lexsort((vertices, counts))
    return {int(vertices[k]): i for i, k in enumerate(order)}


def _rows_as_unit_dicts(component_edges: np.ndarray, col_order: dict[int, int]):
    rows = []
    for edge in component_edges:
        row = {col_order[int(v)]: 1 for v in edge}
        rows.append(row)
    rows.sort(key=lambda r: min(r) if r else -1)
    return rows


def _modular_sparse_rank(
    component_edges: np.ndarray, prime: int, *, target_rank: int | None = None,
    _budget: _NativeRankBudget | None = None,
) -> int:
    """Rank modulo ``prime`` using sparse row echelon.

    ``target_rank`` is a certified characteristic-zero upper bound.  Reaching
    it modulo ``prime`` proves equality over Q, so later rows cannot change
    the answer and are skipped.  This matters when unique edges greatly
    outnumber FE levels, the common HDFE case.
    """
    budget = _NativeRankBudget() if _budget is None else _budget
    source_storage = budget.prepare(component_edges, "native_modular")
    pivot_storage = 0
    value_bits = int(prime).bit_length()
    col_order = _column_order(component_edges)
    rows = _rows_as_unit_dicts(component_edges, col_order)
    pivots: dict[int, dict[int, int]] = {}

    for raw in rows:
        row = raw
        while row:
            budget.check("native_modular", work=len(row), bits=value_bits,
                         storage=source_storage + pivot_storage + 3 * _row_storage(len(row), value_bits))
            c = min(row)
            a = row[c] % prime
            if a == 0:
                row.pop(c, None)
                continue
            pivot = pivots.get(c)
            if pivot is None:
                if a != 1:
                    inv = pow(a, prime - 2, prime)
                    row = {k: (v * inv) % prime for k, v in row.items() if (v * inv) % prime}
                pivot_storage += _row_storage(len(row), value_bits)
                pivots[c] = row
                if target_rank is not None and len(pivots) >= target_rank:
                    return len(pivots)
                break
            # Stored pivots are normalized to coefficient one at c.
            budget.check("native_modular", work=len(row) + len(pivot),
                         storage=source_storage + pivot_storage
                         + 3 * _row_storage(len(row) + len(pivot), value_bits))
            out = dict(row)
            for k, pv in pivot.items():
                nv = (out.get(k, 0) - a * pv) % prime
                if nv:
                    out[k] = nv
                else:
                    out.pop(k, None)
            row = out
    return len(pivots)


def _normalize_integer_row(row: dict[int, int]) -> dict[int, int]:
    if not row:
        return row
    d = 0
    for v in row.values():
        d = gcd(d, abs(int(v)))
        if d == 1:
            break
    if d > 1:
        row = {k: int(v // d) for k, v in row.items() if v}
    c = min(row)
    if row[c] < 0:
        row = {k: -v for k, v in row.items()}
    return row


def _rational_sparse_rank(
    component_edges: np.ndarray, *, _budget: _NativeRankBudget | None = None
) -> int:
    """Deterministic exact rank over Q using primitive integer row operations.

    This path is used only when the fast modular rank does not attain the
    multipartite upper bound, i.e. precisely when there may be a genuine 3+
    FE dependency. Cross-multiplication avoids floating-point tolerances;
    primitive-row normalization controls integer growth.
    """
    budget = _NativeRankBudget() if _budget is None else _budget
    source_storage = budget.prepare(component_edges, "native_rational")
    pivot_storage = 0
    col_order = _column_order(component_edges)
    rows = _rows_as_unit_dicts(component_edges, col_order)
    pivots: dict[int, dict[int, int]] = {}

    for raw in rows:
        row = raw
        while row:
            budget.check("native_rational", work=len(row),
                         storage=source_storage + pivot_storage)
            row_bits = max(abs(v).bit_length() for v in row.values())
            budget.check("native_rational", bits=row_bits,
                         storage=source_storage + pivot_storage + 3 * _row_storage(len(row), row_bits))
            c = min(row)
            a = int(row[c])
            pivot = pivots.get(c)
            if pivot is None:
                pivot_storage += _row_storage(len(row), row_bits)
                pivots[c] = _normalize_integer_row(row)
                break
            b = int(pivot[c])
            g = gcd(abs(a), abs(b))
            mr = b // g
            mp = a // g
            pivot_bits = max(abs(v).bit_length() for v in pivot.values())
            # Cross-products are bounded BEFORE creating big integers or union
            # sets. Cancellation may make this conservative; failure is explicit.
            product_bits = max(abs(mr).bit_length() + row_bits,
                               abs(mp).bit_length() + pivot_bits) + 1
            width = len(row) + len(pivot)
            budget.check("native_rational", work=3 * width, bits=product_bits,
                         storage=source_storage + pivot_storage
                         + 3 * _row_storage(width, product_bits))
            keys = set(row)
            keys.update(pivot)
            out: dict[int, int] = {}
            for k in keys:
                v = mr * int(row.get(k, 0)) - mp * int(pivot.get(k, 0))
                if v:
                    out[k] = v
            row = _normalize_integer_row(out)
    return len(pivots)


def _component_rank(
    component_edges: np.ndarray,
    n_partitions: int,
    *,
    backend: RankBackend = "auto",
    flint_max_cells: int = _DEFAULT_FLINT_MAX_CELLS,
) -> int:
    """Certified exact rank of one connected residual incidence component."""
    if backend not in {"auto", "native", "sympy", "flint"}:
        raise ValueError("backend must be one of: auto, native, sympy, flint")

    nr = int(component_edges.shape[0])
    nv = int(np.unique(component_edges).size)
    upper = min(nr, nv - (n_partitions - 1))
    if upper <= 0:
        return 0
    if nr >= nv - (n_partitions - 1) and _properly_connected(component_edges):
        return int(nv - (n_partitions - 1))

    # Remove the known G-1 block-sum dependencies before any algebra backend.
    rows, ncols = _reduced_component_rows(component_edges)
    if ncols != nv - (n_partitions - 1):
        raise RuntimeError("internal categorical-rank column reduction mismatch")

    # The GF(2) pass is extremely cheap on incidence matrices. If it reaches
    # the Q-rank upper bound, the corresponding odd determinant minor is an
    # unconditional exact certificate and no heavier backend is needed.
    if ncols <= _GF2_MAX_COLS:
        xor_budget = min(
            _GF2_MAX_XOR_BUDGET,
            max(_GF2_MIN_XOR_BUDGET, 512 * len(rows)),
        )
        if _gf2_bitset_rank(rows, upper, max_xors=xor_budget) >= upper:
            return int(upper)

    cells = int(len(rows)) * int(ncols)
    if backend in {"auto", "flint"}:
        try:
            if backend == "flint" or cells <= int(flint_max_cells):
                return _flint_dense_rank(
                    rows, ncols, max_cells=int(flint_max_cells)
                )
        except ImportError:
            if backend == "flint":
                raise
        except ValueError:
            if backend == "flint":
                raise

    if backend in {"auto", "sympy"}:
        try:
            return _sympy_sparse_rank(rows, ncols, upper)
        except ImportError:
            if backend == "sympy":
                raise

    # Dependency-free fallback retained for minimal installations. A nonzero
    # mod-p minor certifies the same lower bound over Q; only a shortfall from
    # the deterministic multipartite upper bound triggers rational elimination.
    budget = _NativeRankBudget()
    rmod = _modular_sparse_rank(
        component_edges, _EXACT_PRIME, target_rank=upper, _budget=budget
    )
    if rmod == upper:
        return int(rmod)
    return int(_rational_sparse_rank(component_edges, _budget=budget))


def categorical_rank(
    groups: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    assume_dense: bool = False,
    return_info: bool = False,
    backend: RankBackend = "auto",
    flint_max_cells: int = _DEFAULT_FLINT_MAX_CELLS,
):
    """Exact rank of concatenated intercept-only categorical FE dummies.

    For one FE the rank is its number of observed levels. For two FEs this is
    equivalent to the standard connected-component formula. For three or more
    FEs the routine treats observations as multipartite hyperedges, removes
    duplicate rows, performs exact degree-one elimination, decomposes the
    residual core, and computes any remaining collective dependencies over Q.

    ``backend="auto"`` first uses structural and GF(2) certificates, then
    prefers optional compiled/external exact backends (python-flint when the
    dense core is safely bounded, then SymPy's sparse DomainMatrix) before
    falling back to the dependency-free native exact implementation.

    The result contains no floating-point rank tolerance. Native elimination
    has cooperative per-component work, estimated-storage and integer-growth
    limits shared by its modular/rational passes; exhaustion raises RuntimeError
    instead of returning an uncertified bound. These are not process RSS/time
    limits and do not meter optional SymPy/FLINT calls or topology construction.
    """
    if backend not in {"auto", "native", "sympy", "flint"}:
        raise ValueError("backend must be one of: auto, native, sympy, flint")
    if int(flint_max_cells) < 0:
        raise ValueError("flint_max_cells must be nonnegative")

    dense, levels = _levels(groups, assume_dense)
    G = len(dense)
    if G == 0:
        info = CategoricalRankInfo(0, 0, 0, 0, 0, 0, 0)
        return info if return_info else 0
    if not dense[0].size:
        info = CategoricalRankInfo(0, int(sum(levels)), 0, 0, 0, 0, 0)
        return info if return_info else 0
    if G == 1:
        rank = int(levels[0])
        info = CategoricalRankInfo(rank, rank, rank, rank, 0, 0, 0)
        return info if return_info else rank
    if G == 2:
        idx_all = np.arange(len(dense[0]), dtype=np.int64)
        edges_all, _, nvertices = _global_edges(dense, levels, idx_all)
        roots = _edge_component_roots(edges_all, nvertices)
        ncomp = int(np.unique(roots).size)
        rank = int(nvertices - ncomp)
        if not return_info:
            return rank
        idx = _unique_edge_indices(dense, levels)
        info = CategoricalRankInfo(
            rank, nvertices, int(len(idx)), 0, int(len(idx)), nvertices, ncomp
        )
        return info

    idx = _unique_edge_indices(dense, levels)
    edges, _, nvertices = _global_edges(dense, levels, idx)
    active, degree, peeled = _peel_degree_one(edges, nvertices)
    core = edges[active.astype(bool)]
    core_vertices = int(np.count_nonzero(degree))

    if not len(core):
        rank = int(peeled)
        info = CategoricalRankInfo(
            rank, nvertices, len(idx), int(peeled), 0, 0, 0
        )
        return info if return_info else rank

    components = _component_edge_lists(core, nvertices)
    core_rank = 0
    for edge_idx in components:
        core_rank += _component_rank(
            core[edge_idx],
            G,
            backend=backend,
            flint_max_cells=flint_max_cells,
        )
    rank = int(peeled + core_rank)
    info = CategoricalRankInfo(
        rank=rank,
        n_levels=nvertices,
        n_unique_edges=int(len(idx)),
        peeled_rank=int(peeled),
        core_edges=int(len(core)),
        core_vertices=core_vertices,
        core_components=int(len(components)),
    )
    return info if return_info else rank


def categorical_prefix_ranks(
    groups: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    assume_dense: bool = False,
    backend: RankBackend = "auto",
    flint_max_cells: int = _DEFAULT_FLINT_MAX_CELLS,
) -> tuple[int, ...]:
    """Exact ranks of ``[D1]``, ``[D1,D2]``, ..., ``[D1,...,DG]``."""
    out = []
    for j in range(1, len(groups) + 1):
        out.append(
            int(
                categorical_rank(
                    groups[:j],
                    assume_dense=assume_dense,
                    backend=backend,
                    flint_max_cells=flint_max_cells,
                )
            )
        )
    return tuple(out)
