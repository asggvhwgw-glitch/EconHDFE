from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .design_ops import ColumnMoments
from .design_plan import StructuralDesignPlan, analyze_design_structure


@dataclass(frozen=True, slots=True)
class DenseDesignBlock:
    index: int
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.rows.nbytes + self.columns.nbytes + self.values.nbytes)


@dataclass(slots=True)
class BlockDesign:
    """Exact global-block / local-dense representation of a numerical design.

    This is intentionally a compute-layer object, not an estimator.  Each
    observation belongs to exactly one stored row block and every omitted entry
    is certified to be a structural zero in the original dense design.  Global
    coefficient columns may appear in more than one row block; exact
    disconnected designs are the important special case where column sets are
    disjoint.
    """

    nobs: int
    ncols: int
    blocks: tuple[DenseDesignBlock, ...]
    _layout_validated: bool = field(default=False, init=False, repr=False)

    def _ensure_layout(self) -> None:
        """Validate row/column topology once for externally constructed designs.

        Internal transformations return trusted layouts and skip the O(N) check.
        Direct construction remains supported for tests and advanced internal use,
        but malformed layouts fail before any operator can silently overwrite or
        double-count observations.
        """
        if self._layout_validated:
            return
        n = int(self.nobs); k = int(self.ncols)
        if n < 0 or k < 0:
            raise ValueError("BlockDesign dimensions must be nonnegative")
        seen_rows = np.zeros(n, dtype=bool)
        seen_indices = set()
        total_rows = 0
        normalized_blocks = []
        for block in self.blocks:
            idx = int(block.index)
            if idx in seen_indices:
                raise ValueError("BlockDesign block indices must be unique")
            seen_indices.add(idx)
            rows = np.asarray(block.rows)
            cols = np.asarray(block.columns)
            vals = np.asarray(block.values)
            if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
                raise ValueError("BlockDesign rows must be a 1D integer array")
            if cols.ndim != 1 or not np.issubdtype(cols.dtype, np.integer):
                raise ValueError("BlockDesign columns must be a 1D integer array")
            if vals.ndim != 2 or vals.shape != (len(rows), len(cols)):
                raise ValueError("BlockDesign block values must match row/column dimensions")
            if not np.issubdtype(vals.dtype, np.number):
                raise TypeError("BlockDesign values must be numeric")
            if len(rows):
                r64 = rows.astype(np.int64, copy=False)
                if np.any(r64 < 0) or np.any(r64 >= n):
                    raise ValueError("BlockDesign row index is out of range")
                if len(np.unique(r64)) != len(r64):
                    raise ValueError("BlockDesign rows must be unique within each block")
                if np.any(seen_rows[r64]):
                    raise ValueError("BlockDesign row components must be disjoint")
                seen_rows[r64] = True
                total_rows += len(r64)
            if len(cols):
                c64 = cols.astype(np.int64, copy=False)
                if np.any(c64 < 0) or np.any(c64 >= k):
                    raise ValueError("BlockDesign column index is out of range")
                if len(np.unique(c64)) != len(c64):
                    raise ValueError("BlockDesign columns must be unique within each block")
            # External/direct construction gets owned, read-only metadata so the
            # one-time layout certificate cannot be invalidated by later mutation
            # of caller-owned row/column arrays. Values remain mutable numeric
            # payload by design.
            r_owned = np.array(rows, dtype=np.int64, copy=True)
            c_owned = np.array(cols, dtype=np.int64, copy=True)
            r_owned.flags.writeable = False
            c_owned.flags.writeable = False
            normalized_blocks.append(DenseDesignBlock(idx, r_owned, c_owned, vals))
        if total_rows != n or (n and not np.all(seen_rows)):
            raise ValueError("BlockDesign blocks must cover every observation exactly once")
        self.blocks = tuple(normalized_blocks)
        self._layout_validated = True

    @classmethod
    def _from_trusted(cls, nobs: int, ncols: int, blocks) -> "BlockDesign":
        obj = cls(int(nobs), int(ncols), tuple(blocks))
        obj._layout_validated = True
        return obj

    @classmethod
    def from_dense(
        cls, X, structure: StructuralDesignPlan, *, copy: bool = True, verify: bool = True
    ) -> "BlockDesign":
        A = np.asarray(X, dtype=np.float64)
        if A.ndim == 1:
            A = A[:, None]
        if A.ndim != 2:
            raise ValueError("X must be a 2D numerical design")
        if A.shape != (int(structure.nobs), int(structure.ncols)):
            raise ValueError("structural design plan does not match X shape")
        if not structure.certified_block_separable:
            raise ValueError("BlockDesign requires an exact block-separability certificate")
        rb = np.asarray(structure.row_blocks)
        if np.any(rb < 0):
            raise ValueError("BlockDesign does not accept nuisance-only row components")
        if verify:
            # A plan may be cached or supplied independently of X.  Shape equality
            # is insufficient: a same-shape design could have acquired a nonzero
            # outside the certified component.  Refuse to discard such data.
            all_cols = np.arange(A.shape[1], dtype=np.int64)
            for block in structure.blocks:
                rows = np.flatnonzero(rb == int(block.index))
                allowed = np.fromiter(block.columns, dtype=np.int64, count=len(block.columns))
                forbidden = np.setdiff1d(all_cols, allowed, assume_unique=True)
                if forbidden.size and np.any(A[np.ix_(rows, forbidden)] != 0.0):
                    raise ValueError("structural design plan is stale for the supplied X")
        blocks = []
        for block in structure.blocks:
            rows64 = np.flatnonzero(rb == int(block.index)).astype(np.int64, copy=False)
            cols64 = np.fromiter(block.columns, dtype=np.int64, count=len(block.columns))
            values = A[np.ix_(rows64, cols64)]
            if copy:
                values = np.array(values, dtype=np.float64, order="C", copy=True)
            blocks.append(DenseDesignBlock(int(block.index), rows64, cols64, values))
        return cls._from_trusted(int(A.shape[0]), int(A.shape[1]), tuple(blocks))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nobs, self.ncols)

    @property
    def payload_bytes(self) -> int:
        self._ensure_layout()
        return int(sum(b.values.nbytes for b in self.blocks))

    @property
    def metadata_bytes(self) -> int:
        self._ensure_layout()
        return int(sum(b.rows.nbytes + b.columns.nbytes for b in self.blocks))

    @property
    def nbytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes

    @property
    def dense_equivalent_bytes(self) -> int:
        return int(self.nobs * self.ncols * np.dtype(np.float64).itemsize)

    @property
    def column_block_counts(self) -> np.ndarray:
        self._ensure_layout()
        counts = np.zeros(self.ncols, dtype=np.int64)
        for b in self.blocks:
            counts[b.columns] += 1
        return counts

    @property
    def columns_disjoint(self) -> bool:
        return bool(np.all(self.column_block_counts <= 1))

    def materialize(self, *, out=None) -> np.ndarray:
        self._ensure_layout()
        if out is None:
            out = np.zeros(self.shape, dtype=np.float64)
        else:
            out = np.asarray(out, dtype=np.float64)
            if out.shape != self.shape:
                raise ValueError("out has the wrong shape")
            out.fill(0.0)
        for b in self.blocks:
            out[np.ix_(b.rows, b.columns)] = b.values
        return out

    def column_moments(self, *, frequency_weights=None) -> ColumnMoments:
        """Column moments; integer ``frequency_weights`` give those of the expanded rows."""
        self._ensure_layout()
        sums = np.zeros(self.ncols, dtype=np.float64)
        ss = np.zeros(self.ncols, dtype=np.float64)
        if frequency_weights is None:
            for b in self.blocks:
                sums[b.columns] += np.sum(b.values, axis=0, dtype=np.float64)
                ss[b.columns] += np.einsum("ij,ij->j", b.values, b.values, optimize=True)
            return ColumnMoments(self.nobs, sums, ss)
        w = np.asarray(frequency_weights, dtype=np.float64)
        for b in self.blocks:
            wb = w[b.rows]
            sums[b.columns] += wb @ b.values
            ss[b.columns] += np.einsum("i,ij,ij->j", wb, b.values, b.values, optimize=True)
        return ColumnMoments(int(round(float(np.sum(w)))), sums, ss)

    def gram(self, *, weights=None) -> np.ndarray:
        self._ensure_layout()
        w = None if weights is None else np.asarray(weights, dtype=np.float64)
        if w is not None and (w.ndim != 1 or len(w) != self.nobs):
            raise ValueError("weights must have one value per observation")
        G = np.zeros((self.ncols, self.ncols), dtype=np.float64)
        for b in self.blocks:
            if w is None:
                Gb = b.values.T @ b.values
            else:
                Gb = b.values.T @ (b.values * w[b.rows, None])
            G[np.ix_(b.columns, b.columns)] += Gb
        return G

    def matvec(self, beta, *, out=None) -> np.ndarray:
        self._ensure_layout()
        beta = np.asarray(beta, dtype=np.float64)
        if beta.ndim != 1 or len(beta) != self.ncols:
            raise ValueError("beta must have one value per design column")
        if out is None:
            out = np.zeros(self.nobs, dtype=np.float64)
        else:
            out = np.asarray(out, dtype=np.float64)
            if out.shape != (self.nobs,):
                raise ValueError("out has the wrong shape")
            out.fill(0.0)
        for b in self.blocks:
            out[b.rows] = b.values @ beta[b.columns]
        return out

    def t_matvec(self, vector, *, out=None) -> np.ndarray:
        self._ensure_layout()
        v = np.asarray(vector, dtype=np.float64)
        if v.ndim != 1 or len(v) != self.nobs:
            raise ValueError("vector must have one value per observation")
        if out is None:
            out = np.zeros(self.ncols, dtype=np.float64)
        else:
            out = np.asarray(out, dtype=np.float64)
            if out.shape != (self.ncols,):
                raise ValueError("out has the wrong shape")
            out.fill(0.0)
        for b in self.blocks:
            out[b.columns] += b.values.T @ v[b.rows]
        return out

    def matmat(self, beta, *, out=None) -> np.ndarray:
        """Multiply by one or more coefficient vectors without densifying X."""
        self._ensure_layout()
        B = np.asarray(beta, dtype=np.float64)
        if B.ndim == 1:
            return self.matvec(B, out=out)
        if B.ndim != 2 or B.shape[0] != self.ncols:
            raise ValueError("beta must have shape (ncols, nrhs)")
        if out is None:
            out = np.zeros((self.nobs, B.shape[1]), dtype=np.float64)
        else:
            out = np.asarray(out, dtype=np.float64)
            if out.shape != (self.nobs, B.shape[1]):
                raise ValueError("out has the wrong shape")
            out.fill(0.0)
        for b in self.blocks:
            out[b.rows] = b.values @ B[b.columns]
        return out

    def t_matmat(self, values, *, out=None) -> np.ndarray:
        """Return X' @ values for one or more observation-space RHS columns."""
        self._ensure_layout()
        A = np.asarray(values, dtype=np.float64)
        if A.ndim == 1:
            return self.t_matvec(A, out=out)
        if A.ndim != 2 or A.shape[0] != self.nobs:
            raise ValueError("values must have shape (nobs, nrhs)")
        if out is None:
            out = np.zeros((self.ncols, A.shape[1]), dtype=np.float64)
        else:
            out = np.asarray(out, dtype=np.float64)
            if out.shape != (self.ncols, A.shape[1]):
                raise ValueError("out has the wrong shape")
            out.fill(0.0)
        for b in self.blocks:
            out[b.columns] += b.values.T @ A[b.rows]
        return out

    def cross_gram(self, other, *, weights=None) -> np.ndarray:
        """Return X' W Y for a dense or row-compatible BlockDesign ``other``.

        This is the basic sufficient-statistic primitive needed by IV/GMM and
        cross-role collinearity checks.  Shared global columns are accumulated
        across row components exactly once per observation.
        """
        self._ensure_layout()
        w = None if weights is None else np.asarray(weights, dtype=np.float64)
        if w is not None and (w.ndim != 1 or len(w) != self.nobs):
            raise ValueError("weights must have one value per observation")
        if isinstance(other, BlockDesign):
            other._ensure_layout()
            if other.nobs != self.nobs:
                raise ValueError("designs must have the same observation count")
            if len(other.blocks) != len(self.blocks):
                raise ValueError("block designs must have the same row partition")
            out = np.zeros((self.ncols, other.ncols), dtype=np.float64)
            for a, b in zip(self.blocks, other.blocks, strict=True):
                if int(a.index) != int(b.index) or not np.array_equal(a.rows, b.rows):
                    raise ValueError("block designs must have the same row partition")
                if a.values.shape[1] == 0 or b.values.shape[1] == 0:
                    continue
                if w is None:
                    G = a.values.T @ b.values
                else:
                    G = a.values.T @ (b.values * w[a.rows, None])
                out[np.ix_(a.columns, b.columns)] += G
            return out

        B = np.asarray(other, dtype=np.float64)
        if B.ndim == 1:
            B = B[:, None]
        if B.ndim != 2 or B.shape[0] != self.nobs:
            raise ValueError("other must have the same observation count")
        out = np.zeros((self.ncols, B.shape[1]), dtype=np.float64)
        for a in self.blocks:
            if a.values.shape[1] == 0:
                continue
            rhs = B[a.rows]
            if w is not None:
                rhs = rhs * w[a.rows, None]
            out[a.columns] += a.values.T @ rhs
        return out

    def select_columns(self, keep) -> "BlockDesign":
        """Restrict/reindex columns without dense materialization."""
        self._ensure_layout()
        keep = np.asarray(keep, dtype=np.int64)
        if keep.ndim != 1 or np.any(keep < 0) or np.any(keep >= self.ncols):
            raise ValueError("keep contains an invalid column index")
        if len(np.unique(keep)) != len(keep):
            raise ValueError("keep must not contain duplicate column indices")
        old_to_new = np.full(self.ncols, -1, dtype=np.int64)
        old_to_new[keep] = np.arange(len(keep), dtype=np.int64)
        blocks = []
        for b in self.blocks:
            positions = np.flatnonzero(old_to_new[b.columns] >= 0)
            if positions.size == 0:
                # Keep the observation component even when its final explicit
                # regressor is omitted.  The component can still contain FE,
                # outcomes and separation information needed by nonlinear
                # estimators.  Dropping the block here would silently drop rows.
                blocks.append(DenseDesignBlock(
                    b.index, b.rows, np.empty(0, dtype=np.int64),
                    np.empty((len(b.rows), 0), dtype=np.float64),
                ))
                continue
            old_cols = b.columns[positions]
            new_cols = old_to_new[old_cols]
            order = np.argsort(new_cols)
            blocks.append(DenseDesignBlock(
                b.index, b.rows, new_cols[order], np.array(b.values[:, positions[order]], copy=True)
            ))
        return BlockDesign._from_trusted(self.nobs, int(len(keep)), tuple(blocks))

    def subset_rows(self, keep) -> "BlockDesign":
        """Restrict observations and reindex row positions without densifying X."""
        self._ensure_layout()
        mask = np.asarray(keep)
        if mask.dtype == bool:
            if mask.ndim != 1 or len(mask) != self.nobs:
                raise ValueError("boolean keep must have one value per observation")
            old_rows = np.flatnonzero(mask)
        else:
            old_rows = np.asarray(mask, dtype=np.int64)
            if old_rows.ndim != 1 or np.any(old_rows < 0) or np.any(old_rows >= self.nobs):
                raise ValueError("row indices are out of bounds")
            if len(np.unique(old_rows)) != len(old_rows):
                raise ValueError("row indices must not contain duplicates")
        reindex = np.full(self.nobs, -1, dtype=np.int64)
        reindex[old_rows] = np.arange(len(old_rows), dtype=np.int64)
        blocks = []
        for b in self.blocks:
            use = reindex[b.rows] >= 0
            if not np.any(use):
                continue
            rows = reindex[b.rows[use]]
            order = np.argsort(rows)
            blocks.append(DenseDesignBlock(
                b.index, rows[order], b.columns, np.array(b.values[use][order], copy=True)
            ))
        return BlockDesign._from_trusted(int(len(old_rows)), self.ncols, tuple(blocks))

    def scale_columns(self, scale, *, inverse: bool = True) -> "BlockDesign":
        self._ensure_layout()
        s = np.asarray(scale, dtype=np.float64)
        if s.ndim != 1 or len(s) != self.ncols:
            raise ValueError("scale must have one value per design column")
        if not np.all(np.isfinite(s)):
            raise ValueError("scale must contain only finite values")
        if inverse and np.any(s == 0.0):
            raise ValueError("inverse column scaling requires nonzero scale values")
        blocks = []
        for b in self.blocks:
            vals = np.array(b.values, copy=True)
            if inverse:
                vals /= s[b.columns]
            else:
                vals *= s[b.columns]
            blocks.append(DenseDesignBlock(b.index, b.rows, b.columns, vals))
        return BlockDesign._from_trusted(self.nobs, self.ncols, tuple(blocks))


def compile_block_design(X, *, structure: StructuralDesignPlan | None = None, groups=()) -> tuple[BlockDesign, StructuralDesignPlan]:
    """Analyze (if necessary) and compile an exact block-native design.

    When this function creates the structural plan itself, the plan and design
    necessarily refer to the same immutable snapshot for this call and a second
    full structural-zero verification pass is unnecessary.  Externally supplied
    plans are verified before any values are discarded.
    """
    own_plan = structure is None
    if structure is None:
        structure = analyze_design_structure(X, groups=groups)
    return BlockDesign.from_dense(X, structure, verify=not own_plan), structure


def same_row_partition(*designs: BlockDesign) -> bool:
    """Return whether BlockDesign objects share the identical row-component layout."""
    designs = tuple(designs)
    if len(designs) < 2:
        return True
    for d in designs:
        if not isinstance(d, BlockDesign):
            raise TypeError("same_row_partition requires BlockDesign inputs")
        d._ensure_layout()
    ref = designs[0]
    for other in designs[1:]:
        if other.nobs != ref.nobs or len(other.blocks) != len(ref.blocks):
            return False
        for a, b in zip(ref.blocks, other.blocks, strict=True):
            if int(a.index) != int(b.index) or not np.array_equal(a.rows, b.rows):
                return False
    return True


def hstack_block_designs(*designs: BlockDesign) -> BlockDesign:
    """Column-bind row-compatible BlockDesign objects without global dense X.

    The resulting column order is the concatenation order.  Empty-role designs
    are supported and row components with no active columns are preserved.
    """
    designs = tuple(designs)
    if not designs:
        return BlockDesign._from_trusted(0, 0, ())
    for d in designs:
        if not isinstance(d, BlockDesign):
            raise TypeError("all inputs must be BlockDesign objects")
        d._ensure_layout()
    n = designs[0].nobs
    if any(d.nobs != n for d in designs):
        raise ValueError("all designs must have the same observation count")
    if any(len(d.blocks) != len(designs[0].blocks) for d in designs[1:]):
        raise ValueError("all designs must have the same row partition")
    offsets = np.cumsum([0] + [d.ncols for d in designs[:-1]], dtype=np.int64)
    blocks = []
    for block_pos in range(len(designs[0].blocks)):
        ref = designs[0].blocks[block_pos]
        values = []
        columns = []
        for d, off in zip(designs, offsets, strict=True):
            b = d.blocks[block_pos]
            if int(b.index) != int(ref.index) or not np.array_equal(b.rows, ref.rows):
                raise ValueError("all designs must have the same row partition")
            if b.values.shape[1]:
                values.append(b.values)
                columns.append(b.columns + int(off))
        vals = (
            np.column_stack(values)
            if values else np.empty((len(ref.rows), 0), dtype=np.float64)
        )
        cols = (
            np.concatenate(columns).astype(np.int64, copy=False)
            if columns else np.empty(0, dtype=np.int64)
        )
        blocks.append(DenseDesignBlock(int(ref.index), ref.rows, cols, vals))
    return BlockDesign._from_trusted(n, int(sum(d.ncols for d in designs)), tuple(blocks))


def align_dense_to_blocks(X, template: BlockDesign) -> BlockDesign:
    """Represent a dense shared-control matrix on an existing row partition."""
    template._ensure_layout()
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2 or A.shape[0] != template.nobs:
        raise ValueError("dense design and template must have the same observation count")
    cols = np.arange(A.shape[1], dtype=np.int64)
    blocks = [
        DenseDesignBlock(int(b.index), b.rows, cols, np.array(A[b.rows], copy=True, order="C"))
        for b in template.blocks
    ]
    return BlockDesign._from_trusted(template.nobs, int(A.shape[1]), tuple(blocks))
