from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from .absorber import AbsorbInfo
from .projection import (
    build_group_index, estimate_index_bytes, group_sums_indexed_weighted, resolve_threads,
    numba_thread_limit,
)


@njit(cache=True, nogil=True)
def _component_anchors(indptr, indices, n_left, n_right):
    """Return one right-side anchor per bipartite connected component."""
    total = n_left + n_right
    parent = np.arange(total, dtype=np.int64)
    size = np.ones(total, dtype=np.int64)

    for left in range(n_left):
        for pos in range(indptr[left], indptr[left + 1]):
            right = n_left + int(indices[pos])
            x = left
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            y = right
            while parent[y] != y:
                parent[y] = parent[parent[y]]; y = parent[y]
            if x != y:
                if size[x] < size[y]: x, y = y, x
                parent[y] = x; size[x] += size[y]

    roots = np.empty(n_right, dtype=np.int64)
    for j in range(n_right):
        x = n_left + j
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        roots[j] = x

    seen = np.full(total, -1, dtype=np.int64)
    anchors = np.zeros(n_right, dtype=np.uint8)
    for j in range(n_right):
        r = roots[j]
        if seen[r] == -1:
            seen[r] = j
            anchors[j] = 1
    return anchors


@njit(cache=True, nogil=True)
def _schur_diag(indptr, indices, data, denom_left, denom_right):
    out = denom_right.copy()
    for i in range(len(denom_left)):
        d = denom_left[i]
        if d <= 0:
            continue
        for pos in range(indptr[i], indptr[i + 1]):
            j = int(indices[pos]); v = data[pos]
            out[j] -= (v * v) / d
    return out


@njit(cache=True, nogil=True, parallel=True)
def _subtract_effects_inplace(out, left_codes, right_codes, alpha, gamma):
    n, k = out.shape
    for i in prange(n):
        a = int(left_codes[i]); b = int(right_codes[i])
        for j in range(k):
            out[i, j] -= alpha[a, j] + gamma[b, j]


@dataclass(slots=True)
class TwoWaySolveInfo:
    connected_components: int
    cross_nnz: int
    solved_side_levels: int
    eliminated_side_levels: int


class TwoWayFEAbsorber:
    """Exact two-way intercept-FE projection via a Schur complement + PCG.

    The larger FE dimension is analytically eliminated because its normal
    block is diagonal.  PCG then operates only on the smaller FE dimension.
    This is particularly useful for empirical structures such as
    ``firm + city#year`` with hundreds of thousands of firms and only a few
    thousand interaction cells.
    """

    def __init__(self, groups, *, weights=None, tol=1e-8, max_iter=16_000):
        if len(groups) != 2:
            raise ValueError("TwoWayFEAbsorber requires exactly two FE groups")
        g0 = np.asarray(groups[0], dtype=np.int32)
        g1 = np.asarray(groups[1], dtype=np.int32)
        if len(g0) != len(g1):
            raise ValueError("FE groups must have equal length")
        self.nobs = len(g0)
        lev0 = int(g0.max()) + 1 if len(g0) else 0
        lev1 = int(g1.max()) + 1 if len(g1) else 0
        # Solve on the smaller side to minimize Krylov dimension.
        self._swapped = lev0 < lev1
        if self._swapped:
            self.left, self.right = g1, g0
            self._original_levels = (lev0, lev1)
        else:
            self.left, self.right = g0, g1
            self._original_levels = (lev0, lev1)
        self.n_left = int(self.left.max()) + 1 if self.nobs else 0
        self.n_right = int(self.right.max()) + 1 if self.nobs else 0
        self.weights = np.ones(self.nobs, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights.ndim != 1 or len(self.weights) != self.nobs or not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise ValueError("invalid weights")
        self.tol = float(tol); self.max_iter = int(max_iter)
        self.method = "twoway"
        self.transform = "schur"
        self.acceleration = "pcg"
        self.preconditioner = "schur_diagonal"
        self.projection_backend = "two_way_schur"
        self.absorb_threads = 1
        self._left_index = None
        self._right_index = None

        # Compile the immutable bipartite topology with one sorted unique pass.
        # Keys are left-major, so the unique ordering is already valid CSR
        # ordering; this avoids the former COO->CSR sort plus a second unique.
        keys = self.left.astype(np.int64) * self.n_right + self.right.astype(np.int64)
        unique, inverse = np.unique(keys, return_inverse=True)
        self._edge_inverse = inverse.astype(np.int32, copy=False)
        self._edge_left = (unique // self.n_right).astype(np.int32, copy=False)
        self._edge_right = (unique % self.n_right).astype(np.int32, copy=False)
        edge_counts_left = np.bincount(self._edge_left, minlength=self.n_left)
        indptr = np.empty(self.n_left + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(edge_counts_left, out=indptr[1:])
        edge_w = np.bincount(self._edge_inverse, weights=self.weights, minlength=len(unique)).astype(np.float64, copy=False)
        C = csr_matrix(
            (edge_w, self._edge_right, indptr),
            shape=(self.n_left, self.n_right), dtype=np.float64,
        )
        self.cross = C
        # Node denominators can now be obtained from the unique-edge weights
        # rather than rescanning all observations.
        self.denom_left = np.bincount(self._edge_left, weights=edge_w, minlength=self.n_left).astype(np.float64, copy=False)
        self.denom_right = np.bincount(self._edge_right, weights=edge_w, minlength=self.n_right).astype(np.float64, copy=False)
        if np.any(self.denom_left <= 0) or np.any(self.denom_right <= 0):
            raise ValueError("two-way FE codes must be dense over observed levels")

        anchor = _component_anchors(C.indptr, C.indices, self.n_left, self.n_right).astype(bool)
        self.keep_right = ~anchor
        self.connected_components = int(anchor.sum())
        self._diag = _schur_diag(C.indptr, C.indices, C.data, self.denom_left, self.denom_right)
        kept_diag = self._diag[self.keep_right]
        eps = np.finfo(np.float64).eps
        self._precond_diag = np.where(kept_diag > eps, kept_diag, 1.0)
        self.solve_info = TwoWaySolveInfo(
            self.connected_components, int(C.nnz), int(self.keep_right.sum()), self.n_left
        )
        self.index_bytes = int(
            C.data.nbytes + C.indices.nbytes + C.indptr.nbytes
            + self._edge_inverse.nbytes + self._edge_left.nbytes + self._edge_right.nbytes
        )

    def _configure_execution(self, *, absorb_threads=1):
        """Private execution-policy hook used by reusable projectors.

        Threading is intentionally not part of the public absorber constructor
        contract.  Estimator/runtime layers configure it after construction so
        release-compatible numerical APIs remain stable.
        """
        self.absorb_threads = resolve_threads(absorb_threads, nobs=self.nobs)
        return self

    # Re-projecting an already-within design can report false nonconvergence
    # (the PCG denominator is a near-zero Schur RHS). Keep the legacy wild
    # bootstrap route until this solver has a reliable final certificate.
    supports_projection_shortcut = False
    supports_fe_orthogonality_certificate = False
    supports_wild_cluster_test = True

    def _pending_index_bytes(self):
        if self.absorb_threads > 1 and self._left_index is None:
            return estimate_index_bytes(self.nobs, [self.n_left, self.n_right])
        return 0

    def workspace_cost(self):
        # Includes the lazy group indexes and bounded coefficient-space PCG
        # arrays/advanced-indexing temporaries, not an N x FE dummy matrix.
        arrays = {}
        for a in (self.weights, self.left, self.right, self.denom_left, self.denom_right,
                  self._diag, self._precond_diag, self.keep_right):
            arrays[id(a)] = a.nbytes
        fixed = int(self.index_bytes) + sum(arrays.values()) + self._pending_index_bytes()
        per_col = 8 * (int(self.nobs) + 4 * self.n_left + 20 * self.n_right)
        return int(fixed), max(1, int(per_col))

    def pending_cost(self):
        return self._pending_index_bytes()

    def absorb_info_extras(self):
        return {}

    def update_weights(self, weights, *, tol=None):
        """Update WLS weights while preserving the bipartite graph topology."""
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != self.nobs or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("invalid weights")
        self.weights = w
        if tol is not None:
            self.tol = float(tol)
        edge_w = np.bincount(
            self._edge_inverse, weights=w, minlength=self.cross.nnz
        ).astype(np.float64, copy=False)
        self.cross.data[:] = edge_w
        self.denom_left = np.bincount(
            self._edge_left, weights=edge_w, minlength=self.n_left
        ).astype(np.float64, copy=False)
        self.denom_right = np.bincount(
            self._edge_right, weights=edge_w, minlength=self.n_right
        ).astype(np.float64, copy=False)
        if np.any(self.denom_left <= 0) or np.any(self.denom_right <= 0):
            raise ValueError("all FE levels must retain positive weight")
        correction = np.bincount(
            self.cross.indices,
            weights=(self.cross.data * self.cross.data) / self.denom_left[self._edge_left],
            minlength=self.n_right,
        )
        self._diag = self.denom_right - correction
        kept = self._diag[self.keep_right]
        eps = np.finfo(np.float64).eps
        self._precond_diag = np.where(kept > eps, kept, 1.0)
        return self

    def _schur_mv(self, x):
        full = np.zeros(self.n_right, dtype=np.float64)
        full[self.keep_right] = x
        z = self.denom_right * full - self.cross.T @ ((self.cross @ full) / self.denom_left)
        return np.asarray(z)[self.keep_right]

    def _schur_mm(self, X):
        """Apply the Schur operator to multiple RHS in one sparse traversal."""
        X = np.asarray(X, dtype=np.float64)
        full = np.zeros((self.n_right, X.shape[1]), dtype=np.float64)
        full[self.keep_right, :] = X
        z = self.denom_right[:, None] * full - self.cross.T @ ((self.cross @ full) / self.denom_left[:, None])
        return np.asarray(z)[self.keep_right, :]

    def _pcg_multi(self, rhs):
        """Independent PCG recurrences fused across RHS columns.

        Algebraically this is the same diagonal-preconditioned CG solve used
        per column previously. Sparse Schur matvecs are executed as matmat
        operations so event-study/IV blocks reuse memory traffic.
        """
        B = np.asarray(rhs, dtype=np.float64)
        if B.ndim == 1:
            B = B[:, None]
        n, k = B.shape
        X = np.zeros((n, k), dtype=np.float64)
        R = B.copy()
        norm_b = np.linalg.norm(B, axis=0)
        eps = np.finfo(np.float64).eps
        scale = np.maximum(norm_b, eps)
        active = norm_b > eps
        if not np.any(active):
            return X, True, 0, 0.0
        Z = R / self._precond_diag[:, None]
        P = Z.copy()
        rz = np.sum(R * Z, axis=0)
        rel = np.linalg.norm(R, axis=0) / scale
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):
            AP = self._schur_mm(P)
            pap = np.sum(P * AP, axis=0)
            safe = active & (np.abs(pap) > eps * np.maximum(np.abs(rz), eps))
            if not np.all(safe[active]):
                break
            alpha = np.zeros(k, dtype=np.float64)
            alpha[safe] = rz[safe] / pap[safe]
            X[:, safe] += P[:, safe] * alpha[safe]
            R[:, safe] -= AP[:, safe] * alpha[safe]
            rel = np.linalg.norm(R, axis=0) / scale
            done = active & (rel <= self.tol)
            active[done] = False
            P[:, done] = 0.0
            if not np.any(active):
                converged = True
                break
            Z[:, active] = R[:, active] / self._precond_diag[:, None]
            rz_new = np.sum(R * Z, axis=0)
            beta = np.zeros(k, dtype=np.float64)
            beta[active] = rz_new[active] / rz[active]
            P[:, active] = Z[:, active] + P[:, active] * beta[active]
            rz = rz_new
        metric = float(np.max(rel)) if rel.size else 0.0
        return X, converged, it, metric

    def _operators(self):
        n = int(self.keep_right.sum())
        A = LinearOperator((n, n), matvec=self._schur_mv, dtype=np.float64)
        M = LinearOperator((n, n), matvec=lambda x: x / self._precond_diag, dtype=np.float64)
        return A, M

    def _ensure_group_indexes(self):
        if self._left_index is None:
            self._left_index = build_group_index(self.left, self.n_left)
            self._right_index = build_group_index(self.right, self.n_right)
            self.index_bytes += int(self._left_index.nbytes + self._right_index.nbytes)

    def _group_sums(self, data, side, *, threads=None):
        if side == "left":
            codes, levels, index = self.left, self.n_left, self._left_index
        elif side == "right":
            codes, levels, index = self.right, self.n_right, self._right_index
        else:
            raise ValueError("side must be left/right")
        nthreads = self.absorb_threads if threads is None else threads
        if nthreads > 1:
            self._ensure_group_indexes()
            index = self._left_index if side == "left" else self._right_index
            return group_sums_indexed_weighted(
                data, index, self.weights, threads=nthreads
            )
        out = np.empty((levels, data.shape[1]), dtype=np.float64)
        weighted = self.weights
        for j in range(data.shape[1]):
            out[:, j] = np.bincount(codes, weights=weighted * data[:, j], minlength=levels)
        return out

    def _solve(self, data, *, threads=None):
        Y = np.asarray(data, dtype=np.float64)
        ba = self._group_sums(Y, "left", threads=threads)
        bb = self._group_sums(Y, "right", threads=threads)
        rhs = bb - self.cross.T @ (ba / self.denom_left[:, None])
        gamma = np.zeros((self.n_right, Y.shape[1]), dtype=np.float64)
        sol, converged, worst_it, worst_metric = self._pcg_multi(np.asarray(rhs)[self.keep_right, :])
        gamma[self.keep_right, :] = sol
        alpha = (ba - self.cross @ gamma) / self.denom_left[:, None]
        return alpha, gamma, AbsorbInfo(converged, worst_it, worst_metric, "schur_pcg_residual")

    def residualize(self, data, *, copy=True, return_info=False, absorb_threads=None):
        a = np.asarray(data, dtype=np.float64)
        one_dim = a.ndim == 1
        if one_dim: a = a[:, None]
        if a.shape[0] != self.nobs:
            raise ValueError("data row count does not match fixed effects")
        out = a.copy() if copy else a
        nthreads = self.absorb_threads if absorb_threads is None else resolve_threads(absorb_threads, nobs=self.nobs)
        with numba_thread_limit(nthreads):
            alpha, gamma, info = self._solve(a, threads=nthreads)
            _subtract_effects_inplace(out, self.left, self.right, alpha, gamma)
        result = out[:, 0] if one_dim else out
        return (result, info) if return_info else result

    def solve_coefficients(self, target):
        """Return level coefficients in requested FE order, without N-row copies."""
        y = np.asarray(target, dtype=np.float64)
        if y.ndim != 1 or len(y) != self.nobs or not np.all(np.isfinite(y)):
            raise ValueError("target must be a finite vector matching fixed-effect rows")
        alpha, gamma, info = self._solve(y[:, None])
        pair = (alpha[:, 0], gamma[:, 0])
        return (pair[::-1] if self._swapped else pair), info

    def solve_effects(self, target):
        y = np.asarray(target, dtype=np.float64)
        alpha, gamma, info = self._solve(y[:, None])
        alpha = alpha[:, 0]; gamma = gamma[:, 0]
        left_contrib = alpha[self.left]
        right_contrib = gamma[self.right]
        if self._swapped:
            terms = [(gamma, None, right_contrib), (alpha, None, left_contrib)]
        else:
            terms = [(alpha, None, left_contrib), (gamma, None, right_contrib)]
        recon = left_contrib + right_contrib
        denom = max(float(np.linalg.norm(y)), np.finfo(float).eps)
        return terms, {
            "converged": info.converged,
            "iterations": info.iterations,
            "residual_norm": float(np.linalg.norm(y - recon)),
            "reconstruction_error": float(np.linalg.norm(y - recon) / denom),
        }
