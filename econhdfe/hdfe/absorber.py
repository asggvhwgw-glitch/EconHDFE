from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from threading import RLock
import numpy as np
from numba import get_num_threads, njit, prange
from scipy.sparse.linalg import LinearOperator, lsmr
from ..compute.backend import get_array_module
from ..errors import ConvergenceError
from .numerical_core import build_numerical_core
from .projection import (
    build_group_index, estimate_index_bytes, numba_thread_limit,
    _project_indexed_inplace_current, _max_relative_update_parallel, resolve_threads,
)


@njit(cache=True, nogil=True)
def _project_intercept_matrix_unweighted(a, codes, denom, sums):
    """Fused intercept-only FE projection for many RHS columns.

    Scans FE codes once for aggregation and once for write-back instead of
    calling bincount separately for every RHS. This is the critical fast path
    for 10M+ rows with many controls.
    """
    n, k = a.shape
    L = denom.shape[0]
    sums[:, :] = 0.0
    for i in range(n):
        g = int(codes[i])
        for j in range(k):
            sums[g, j] += a[i, j]
    for i in range(n):
        g = int(codes[i])
        d = denom[g]
        if d != 0.0:
            for j in range(k):
                a[i, j] -= sums[g, j] / d


@njit(cache=True, nogil=True)
def _project_intercept_matrix_weighted(a, codes, weights, denom, sums):
    """Fused bounded-RHS weighted intercept projection."""
    n, k = a.shape
    L = denom.shape[0]
    sums[:, :] = 0.0
    for i in range(n):
        g = int(codes[i])
        wi = weights[i]
        for j in range(k):
            sums[g, j] += wi * a[i, j]
    for i in range(n):
        g = int(codes[i])
        d = denom[g]
        if d != 0.0:
            inv_d = 1.0 / d
            for j in range(k):
                a[i, j] -= sums[g, j] * inv_d


@dataclass(slots=True)
class AbsorbInfo:
    converged: bool
    iterations: int
    max_update: float
    criterion: str = "relative_update"


@njit(cache=True, nogil=True)
def _cg_update_residuals(a, r, u, v, alpha):
    """Fuse two CG updates without N x RHS multiplication temporaries."""
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] -= u[i, j] * alpha[j]
            r[i, j] -= v[i, j] * alpha[j]


@njit(cache=True, nogil=True)
def _cg_update_direction(u, r, beta):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            u[i, j] = r[i, j] + u[i, j] * beta[j]


# Fixed reduction chunks make the reduction order independent of thread masks.
# Large CPU float64 blocks use the parallel path with fixed-size reduction chunks.
_CG_REDUCTION_CHUNKS = 16
_CG_PARALLEL_MIN_ROWS = 100_000


@njit(cache=True, nogil=True, parallel=True)
def _cg_update_direction_parallel(u, r, beta):
    for i in prange(u.shape[0]):
        for j in range(u.shape[1]):
            u[i, j] = r[i, j] + u[i, j] * beta[j]

@njit(cache=True, nogil=True, parallel=True)
def _cg_update_residuals_and_norm(a, r, u, v, alpha, weights):
    n, k = a.shape
    parts = np.zeros((_CG_REDUCTION_CHUNKS, k), dtype=np.float64)
    span = (n + _CG_REDUCTION_CHUNKS - 1) // _CG_REDUCTION_CHUNKS
    for chunk in prange(_CG_REDUCTION_CHUNKS):
        for i in range(chunk * span, min(n, (chunk + 1) * span)):
            for j in range(k):
                a[i, j] -= u[i, j] * alpha[j]
                r[i, j] -= v[i, j] * alpha[j]
                if weights is None:
                    parts[chunk, j] += r[i, j] * r[i, j]
                else:
                    parts[chunk, j] += weights[i] * r[i, j] * r[i, j]
    sums = np.zeros(k, dtype=np.float64)
    for chunk in range(_CG_REDUCTION_CHUNKS):
        for j in range(k):
            sums[j] += parts[chunk, j]
    return sums

@njit(cache=True, nogil=True, parallel=True)
def _cg_subtract_and_inner_product(a, z, weights):
    n, k = a.shape
    parts = np.zeros((_CG_REDUCTION_CHUNKS, k), dtype=np.float64)
    span = (n + _CG_REDUCTION_CHUNKS - 1) // _CG_REDUCTION_CHUNKS
    for chunk in prange(_CG_REDUCTION_CHUNKS):
        for i in range(chunk * span, min(n, (chunk + 1) * span)):
            for j in range(k):
                z[i, j] = a[i, j] - z[i, j]
                if weights is None:
                    parts[chunk, j] += a[i, j] * z[i, j]
                else:
                    parts[chunk, j] += weights[i] * a[i, j] * z[i, j]
    sums = np.zeros(k, dtype=np.float64)
    for chunk in range(_CG_REDUCTION_CHUNKS):
        for j in range(k):
            sums[j] += parts[chunk, j]
    return sums

@dataclass(slots=True)
class _TermProjector:
    denom: np.ndarray | None
    slope_means: np.ndarray | None
    slope_gram_inv: np.ndarray | None
    slope_inv_ss: np.ndarray | None


class HDFEConvergenceError(ConvergenceError):
    """Raised by the public estimator when FE absorption fails to converge."""

    def __init__(self, info, *, context="HDFE absorption"):
        self.info = info
        super().__init__(
            f"{context} did not converge after {info.iterations} iterations "
            f"(last convergence metric={info.max_update:.3e}).",
            code="convergence.hdfe", stage="hdfe",
            details={"iterations": int(info.iterations), "metric": float(info.max_update)},
            suggestion="Try symmetric MAP + CG, LSMR, or increase max_iter.",
        )


class HDFEAbsorber:
    """Reusable high-dimensional fixed-effect residualizer.

    Categorical identifiers are stored as dense int32 codes. Heterogeneous
    slopes are projected group-by-group without materializing dummy matrices.
    For intercept+slope terms the slope is weighted-centered inside each group,
    which orthogonalizes it from the group intercept and substantially improves
    conditioning while preserving the exact absorbed column space.
    """

    def __init__(
        self,
        groups,
        *,
        slopes=None,
        intercepts=None,
        weights=None,
        tol=1e-8,
        max_iter=16_000,
        backend="numpy",
        method="map",
        transform="symmetric",
        acceleration="cg",
        preconditioner="diagonal",
        dtype=np.float64,
        gram_chunk_levels=100_000,
        projection_backend="auto",
        absorb_threads="auto",
        projection_memory_budget_mb=512,
        projection_min_nobs=100_000,
        core_reduction="auto",
        core_min_peel_fraction=0.05,
        auto_plain_limit=8,
        auto_polish_limit=32,
        fused_rhs_memory_mb=128,
    ):
        self.backend = backend
        self.method = method
        self.preconditioner = preconditioner
        self.acceleration = acceleration
        if method not in ("map", "lsmr"):
            raise ValueError("method must be 'map' or 'lsmr'")
        if method == "lsmr" and backend != "numpy":
            raise ValueError("LSMR currently uses SciPy/CPU; use method='map' for CuPy GPU")
        if preconditioner not in ("diagonal", "none"):
            raise ValueError("preconditioner must be 'diagonal' or 'none'")
        if acceleration not in ("auto", "none", "cg"):
            raise ValueError("acceleration must be 'auto', 'none', or 'cg'")
        if acceleration in ("auto", "cg") and transform != "symmetric":
            raise ValueError("auto/CG acceleration requires transform='symmetric'")
        if core_reduction not in ("auto", "on", "off"):
            raise ValueError("core_reduction must be 'auto', 'on', or 'off'")
        self.xp = get_array_module(backend)
        self.groups = [np.asarray(g, dtype=np.int32) for g in groups]
        if not self.groups:
            raise ValueError("at least one fixed-effect group is required")
        n = len(self.groups[0])
        if any(len(g) != n for g in self.groups):
            raise ValueError("all fixed-effect arrays must have equal length")
        self.nobs = n
        self.levels = [int(g.max()) + 1 if len(g) else 0 for g in self.groups]
        G = len(self.groups)
        if slopes is None:
            slopes = [None] * G
        if intercepts is None:
            intercepts = [True] * G
        if len(slopes) != G or len(intercepts) != G:
            raise ValueError("groups/slopes/intercepts length mismatch")
        self.slopes: list[np.ndarray | None] = []
        for s in slopes:
            if s is None:
                self.slopes.append(None)
                continue
            a = np.asarray(s, dtype=np.float64)
            if a.ndim == 1:
                a = a[:, None]
            if a.ndim != 2 or a.shape[0] != n:
                raise ValueError("each slope block must be nobs x nslopes")
            self.slopes.append(a)
        self.intercepts = [bool(v) for v in intercepts]
        for gi, (has_int, s) in enumerate(zip(self.intercepts, self.slopes, strict=False)):
            if not has_int and (s is None or s.shape[1] == 0):
                raise ValueError(f"absorbed term {gi} has neither intercept nor slopes")

        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights is not None:
            if self.weights.ndim != 1 or len(self.weights) != n:
                raise ValueError("weights length mismatch")
            if not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0):
                raise ValueError("weights must be finite and nonnegative")
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.transform = transform
        self.dtype = dtype
        self.gram_chunk_levels = int(gram_chunk_levels)
        if projection_backend not in ("auto", "fused", "indexed"):
            raise ValueError("projection_backend must be 'auto', 'fused', or 'indexed'")
        self.absorb_threads = resolve_threads(absorb_threads, nobs=n, calibration_min_nobs=projection_min_nobs) if backend == "numpy" else 1
        self.projection_backend_requested = projection_backend
        self.projection_memory_budget_mb = float(projection_memory_budget_mb)
        self.projection_min_nobs = int(projection_min_nobs)
        self.core_reduction_requested = core_reduction
        self.core_min_peel_fraction = float(core_min_peel_fraction)
        self.auto_plain_limit = max(1, int(auto_plain_limit))
        self.auto_polish_limit = max(0, int(auto_polish_limit))
        # Auto-planner internals are deliberately not public knobs: two real
        # symmetric sweeps on the full RHS block both warm-start the solve and
        # estimate contraction before continuing MAP or switching to CG.
        # Empirical thresholding is validated by the complex-FE corpus.
        self._auto_probe_iterations = 2
        self._auto_contraction_threshold = 0.02
        self.fused_rhs_memory_mb = float(fused_rhs_memory_mb)
        self._projection_scratch = np.empty(0, dtype=np.float64)
        # Scratch belongs to this reusable absorber. Protect same-instance
        # calls even on OpenMP/TBB, where the global workqueue guard is absent.
        self._execution_lock = RLock()
        self._profile_enabled = False
        self._iteration_profile = {}

        # Group count/weight sums are invariant across sweeps and reused by
        # intercept projections and slope centering.
        self._denoms = [
            np.bincount(g, weights=self.weights, minlength=L).astype(np.float64, copy=False)
            if self.weights is not None else
            np.bincount(g, minlength=L).astype(np.float64, copy=False)
            for g, L in zip(self.groups, self.levels, strict=False)
        ]
        self._projectors = [self._build_projector(i) for i in range(G)]

        # Exact numerical degree-one peeling for pure intercept 3+ FE.  The
        # topology is weight-invariant as long as all observation weights stay
        # strictly positive.  We first check existing/count denominators for an
        # initial singleton; without one recursive peeling cannot start.
        self._core_plan = None
        self._core_absorber = None
        core_eligible = (
            backend == "numpy" and G >= 3 and all(self.intercepts)
            and all(s is None for s in self.slopes) and core_reduction != "off"
            and (self.weights is None or np.all(self.weights > 0))
        )
        if core_eligible:
            has_singleton = False
            for g, L, d in zip(self.groups, self.levels, self._denoms, strict=False):
                counts = d if self.weights is None else np.bincount(g, minlength=L)
                if np.any(counts == 1):
                    has_singleton = True
                    break
            if has_singleton:
                plan = build_numerical_core(self.groups)
                use_core = plan.changed and (
                    core_reduction == "on" or plan.peeled_fraction >= self.core_min_peel_fraction
                )
                if use_core:
                    self._core_plan = plan

        # CPU indexed projection is a compact CSR-like row index. It turns the
        # reduction into independent group jobs, so Numba can parallelize
        # without atomics or thread-private G x K accumulators. The O(N) index
        # is cached because MAP revisits each FE many times.
        self._group_indexes = [None] * G
        eligible = ([] if self._core_plan is not None else
                    [i for i in range(G) if self.intercepts[i] and self.slopes[i] is None])
        index_bytes = estimate_index_bytes(n, [self.levels[i] for i in eligible])
        budget = max(self.projection_memory_budget_mb, 0.0) * (1024**2)
        use_indexed = (
            backend == "numpy" and method == "map" and bool(eligible)
            and (projection_backend == "indexed" or (
                projection_backend == "auto" and self.absorb_threads > 1
                and G > 1 and n >= self.projection_min_nobs and index_bytes <= budget
            ))
        )
        if projection_backend == "indexed" and backend != "numpy":
            raise ValueError("projection_backend='indexed' currently requires backend='numpy'")
        if projection_backend == "indexed" and method != "map":
            raise ValueError("projection_backend='indexed' is only used with method='map'")
        if use_indexed:
            for i in eligible:
                self._group_indexes[i] = build_group_index(self.groups[i], self.levels[i])
        self.index_bytes = int(sum(idx.nbytes for idx in self._group_indexes if idx is not None))
        n_indexed = sum(idx is not None for idx in self._group_indexes)
        if n_indexed == 0:
            self.projection_backend = "fused"
        elif n_indexed == G:
            self.projection_backend = "indexed"
        else:
            self.projection_backend = "hybrid"

        self._gpu_groups = None
        self._gpu_weights = None
        self._gpu_projectors = None
        self._gpu_slopes = None
        if backend == "cupy":
            cp = self.xp
            self._gpu_groups = [cp.asarray(g) for g in self.groups]
            self._gpu_weights = None if self.weights is None else cp.asarray(self.weights)
            self._gpu_slopes = [None if s is None else cp.asarray(s) for s in self.slopes]
            self._gpu_projectors = []
            for p in self._projectors:
                self._gpu_projectors.append(_TermProjector(
                    None if p.denom is None else cp.asarray(p.denom),
                    None if p.slope_means is None else cp.asarray(p.slope_means),
                    None if p.slope_gram_inv is None else cp.asarray(p.slope_gram_inv),
                    None if p.slope_inv_ss is None else cp.asarray(p.slope_inv_ss),
                ))

    def _basis_weight(self):
        return 1.0 if self.weights is None else self.weights

    def _build_projector(self, gi: int) -> _TermProjector:
        has_int = self.intercepts[gi]
        S = self.slopes[gi]
        denom = self._denoms[gi] if has_int else None
        if S is None or S.shape[1] == 0:
            return _TermProjector(denom, None, None, None)

        codes = self.groups[gi]
        L = self.levels[gi]
        w = self.weights
        p = S.shape[1]
        means = None
        if has_int:
            means = np.empty((L, p), dtype=np.float64)
            d = self._denoms[gi]
            for j in range(p):
                vals = S[:, j] if w is None else w * S[:, j]
                sums = np.bincount(codes, weights=vals, minlength=L)
                means[:, j] = np.divide(sums, d, out=np.zeros_like(sums), where=d != 0)

        # Fast scalar slope projector: only one G-vector is needed.
        if p == 1:
            z = S[:, 0] if means is None else S[:, 0] - means[codes, 0]
            vals = z * z if w is None else w * z * z
            ss = np.bincount(codes, weights=vals, minlength=L)
            inv_ss = np.divide(1.0, ss, out=np.zeros_like(ss), where=ss > 0)
            return _TermProjector(denom, means, None, inv_ss)

        # Multi-slope terms use batched group Gram inverses.  We aggregate each
        # p x p entry using bincount, so complexity is O(N p^2 + G p^3) with
        # no N x G dummy representation.
        gram = np.empty((L, p, p), dtype=np.float64)
        for a in range(p):
            za = S[:, a] if means is None else S[:, a] - means[codes, a]
            for b in range(a, p):
                zb = S[:, b] if means is None else S[:, b] - means[codes, b]
                vals = za * zb if w is None else w * za * zb
                v = np.bincount(codes, weights=vals, minlength=L)
                gram[:, a, b] = v
                gram[:, b, a] = v
        inv = np.empty_like(gram)
        chunk = max(1, self.gram_chunk_levels)
        for lo in range(0, L, chunk):
            hi = min(L, lo + chunk)
            inv[lo:hi] = np.linalg.pinv(gram[lo:hi], hermitian=True)
        return _TermProjector(denom, means, inv, None)

    def _project_inplace_numpy(self, a: np.ndarray, gi: int) -> None:
        codes = self.groups[gi]
        L = self.levels[gi]
        w = self.weights
        proj = self._projectors[gi]
        S = self.slopes[gi]
        has_int = self.intercepts[gi]

        if has_int and S is None:
            index = self._group_indexes[gi]
            bytes_per_col = max(1, L) * 8
            budget = max(1, int(self.fused_rhs_memory_mb * (1024**2)))
            fused_block = max(1, min(a.shape[1], budget // bytes_per_col))
            # Keep the compiled backend and inherit the caller's thread mask.
            # Scalar accumulators avoid tile/scratch overhead for narrow pools.
            if index is not None:
                _project_indexed_inplace_current(a, index, w, proj.denom)
                return
            if a.shape[1] > 0:
                for lo in range(0, a.shape[1], fused_block):
                    hi = min(a.shape[1], lo + fused_block)
                    size = L * (hi - lo)
                    if self._projection_scratch.size < size:
                        self._projection_scratch = np.empty(size, dtype=np.float64)
                    sums = self._projection_scratch[:size].reshape(L, hi - lo)
                    if w is None:
                        _project_intercept_matrix_unweighted(a[:, lo:hi], codes, proj.denom, sums)
                    else:
                        _project_intercept_matrix_weighted(a[:, lo:hi], codes, w, proj.denom, sums)
                return

        for col in range(a.shape[1]):
            y = a[:, col]
            if has_int:
                vals = y if w is None else w * y
                sums = np.bincount(codes, weights=vals, minlength=L)
                means_y = np.divide(
                    sums, proj.denom, out=np.zeros_like(sums), where=proj.denom != 0
                )
                y -= means_y[codes]
            if S is None or S.shape[1] == 0:
                continue
            p = S.shape[1]
            if p == 1:
                z = S[:, 0] if proj.slope_means is None else S[:, 0] - proj.slope_means[codes, 0]
                vals = z * y if w is None else w * z * y
                rhs = np.bincount(codes, weights=vals, minlength=L)
                coef = rhs * proj.slope_inv_ss
                y -= z * coef[codes]
                continue
            rhs = np.empty((L, p), dtype=np.float64)
            for j in range(p):
                z = S[:, j] if proj.slope_means is None else S[:, j] - proj.slope_means[codes, j]
                vals = z * y if w is None else w * z * y
                rhs[:, j] = np.bincount(codes, weights=vals, minlength=L)
            coef = np.einsum("gij,gj->gi", proj.slope_gram_inv, rhs, optimize=True)
            for j in range(p):
                z = S[:, j] if proj.slope_means is None else S[:, j] - proj.slope_means[codes, j]
                y -= z * coef[codes, j]

    def _project_inplace_gpu(self, a, gi: int) -> None:
        cp = self.xp
        codes = self._gpu_groups[gi]
        L = self.levels[gi]
        w = self._gpu_weights
        proj = self._gpu_projectors[gi]
        S = self._gpu_slopes[gi]
        has_int = self.intercepts[gi]

        for col in range(a.shape[1]):
            y = a[:, col]
            if has_int:
                vals = y if w is None else w * y
                sums = cp.bincount(codes, weights=vals, minlength=L)
                means_y = cp.divide(
                    sums, proj.denom, out=cp.zeros_like(sums), where=proj.denom != 0
                )
                y -= means_y[codes]
            if S is None or S.shape[1] == 0:
                continue
            p = S.shape[1]
            if p == 1:
                z = S[:, 0] if proj.slope_means is None else S[:, 0] - proj.slope_means[codes, 0]
                vals = z * y if w is None else w * z * y
                rhs = cp.bincount(codes, weights=vals, minlength=L)
                coef = rhs * proj.slope_inv_ss
                y -= z * coef[codes]
                continue
            rhs = cp.empty((L, p), dtype=cp.float64)
            for j in range(p):
                z = S[:, j] if proj.slope_means is None else S[:, j] - proj.slope_means[codes, j]
                vals = z * y if w is None else w * z * y
                rhs[:, j] = cp.bincount(codes, weights=vals, minlength=L)
            coef = cp.einsum("gij,gj->gi", proj.slope_gram_inv, rhs)
            for j in range(p):
                z = S[:, j] if proj.slope_means is None else S[:, j] - proj.slope_means[codes, j]
                y -= z * coef[codes, j]

    def _compile_group_indexes_if_needed(self):
        if any(idx is not None for idx in self._group_indexes):
            return
        if self.backend != "numpy" or self.method != "map":
            return
        eligible = [i for i in range(len(self.groups)) if self.intercepts[i] and self.slopes[i] is None]
        index_bytes = estimate_index_bytes(self.nobs, [self.levels[i] for i in eligible])
        budget = max(self.projection_memory_budget_mb, 0.0) * (1024**2)
        use = (self.projection_backend_requested == "indexed" or (
            self.projection_backend_requested == "auto" and self.absorb_threads > 1
            and len(self.groups) > 1 and self.nobs >= self.projection_min_nobs and index_bytes <= budget
        ))
        if use:
            for i in eligible:
                self._group_indexes[i] = build_group_index(self.groups[i], self.levels[i])
            self.index_bytes = int(sum(idx.nbytes for idx in self._group_indexes if idx is not None))
            self.projection_backend = "indexed" if len(eligible) == len(self.groups) else "hybrid"

    def _ensure_core_absorber(self):
        if self._core_plan is None or self._core_plan.core_nobs == 0:
            return None
        if self._core_absorber is None:
            idx = self._core_plan.core_index
            core = HDFEAbsorber(
                self._core_plan.core_groups,
                weights=None if self.weights is None else self.weights[idx],
                tol=self.tol, max_iter=self.max_iter, backend=self.backend,
                method=self.method, transform=self.transform, acceleration=self.acceleration,
                preconditioner=self.preconditioner, dtype=self.dtype,
                gram_chunk_levels=self.gram_chunk_levels, projection_backend=self.projection_backend_requested,
                absorb_threads=self.absorb_threads, projection_memory_budget_mb=self.projection_memory_budget_mb,
                projection_min_nobs=self.projection_min_nobs, core_reduction="off",
                auto_plain_limit=self.auto_plain_limit, auto_polish_limit=self.auto_polish_limit,
                fused_rhs_memory_mb=self.fused_rhs_memory_mb,
            )
            self._core_absorber = core
        return self._core_absorber

    def update_weights(self, weights, *, tol=None):
        """Update numerical weights without rebuilding categorical topology.

        Group codes and cached row indexes are structural and remain valid.
        Only weighted group denominators and slope projectors are refreshed.
        This is the shared fast path for iterative WLS estimators such as PPML.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != self.nobs or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("invalid weights")
        self.weights = w
        if tol is not None:
            self.tol = float(tol)
        self._denoms = [
            np.bincount(g, weights=w, minlength=L).astype(np.float64, copy=False)
            for g, L in zip(self.groups, self.levels, strict=False)
        ]
        self._projectors = [self._build_projector(i) for i in range(len(self.groups))]
        if self.backend == "cupy":
            cp = self.xp
            self._gpu_weights = cp.asarray(w)
            self._gpu_projectors = [
                _TermProjector(
                    None if q.denom is None else cp.asarray(q.denom),
                    None if q.slope_means is None else cp.asarray(q.slope_means),
                    None if q.slope_gram_inv is None else cp.asarray(q.slope_gram_inv),
                    None if q.slope_inv_ss is None else cp.asarray(q.slope_inv_ss),
                )
                for q in self._projectors
            ]
        if self._core_plan is not None:
            if np.any(w <= 0):
                # Zero-weight rows change the effective numerical topology;
                # disable the structural peeling fast path for correctness.
                self._core_plan = None
                self._core_absorber = None
                self._compile_group_indexes_if_needed()
            elif self._core_absorber is not None:
                self._core_absorber.update_weights(w[self._core_plan.core_index], tol=tol)
        return self

    def residualize(self, data, *, copy=True, return_info=False, absorb_threads=None):
        self._iteration_profile = {}
        xp = self.xp
        a = xp.asarray(data, dtype=self.dtype)
        one_dim = a.ndim == 1
        if one_dim:
            a = a[:, None]
        if a.shape[0] != self.nobs:
            raise ValueError("data row count does not match fixed effects")
        if copy:
            a = a.copy()

        if self._core_plan is not None:
            idx = self._core_plan.core_index
            if idx.size:
                # Advanced indexing materializes only the reduced core.  Once
                # that input is detached, the caller-owned/workspace buffer can
                # be reused for the final result, avoiding an additional full
                # N x RHS allocation on the large-data execution path.
                core_input = np.asarray(a)[idx]
                core_absorber = self._ensure_core_absorber()
                core_absorber._profile_enabled = self._profile_enabled
                core_out, core_info = core_absorber.residualize(
                    core_input, copy=False, return_info=True, absorb_threads=absorb_threads
                )
                self._iteration_profile = dict(core_absorber._iteration_profile)
                # Core CG solves can be checked cheaply because peeling already
                # reduced the system.  Keep strict weighted/core accuracy here
                # without charging every no-core multiway solve for a full FE
                # orthogonality scan.
                if (
                    ("cg" in core_info.criterion or core_info.criterion == "hestenes")
                    and core_absorber.supports_fe_orthogonality_certificate
                ):
                    err = core_absorber.fe_orthogonality_error(core_out)
                    if err > self.tol and core_info.iterations < self.max_iter:
                        nthreads = core_absorber.absorb_threads if absorb_threads is None else resolve_threads(absorb_threads)
                        with core_absorber._execution_lock, numba_thread_limit(nthreads):
                            polish = core_absorber._polish_orthogonality(
                                core_out, min(
                                    self.auto_polish_limit,
                                    self.max_iter - core_info.iterations,
                                )
                            )
                        core_info = AbsorbInfo(
                            polish.converged,
                            core_info.iterations + polish.iterations,
                            polish.max_update,
                            "auto_cg_polish",
                        )
                a[...] = 0
                a[idx] = core_out
                info = AbsorbInfo(
                    core_info.converged, core_info.iterations, core_info.max_update,
                    "core_" + core_info.criterion,
                )
            else:
                a[...] = 0
                info = AbsorbInfo(True, 0, 0.0, "core_peeling")
            result = a[:, 0] if one_dim else a
            return (result, info) if return_info else result

        if self.backend == "numpy" and self.method == "map":
            nthreads = self.absorb_threads if absorb_threads is None else resolve_threads(absorb_threads, nobs=self.nobs, calibration_min_nobs=self.projection_min_nobs)
            with self._execution_lock, numba_thread_limit(nthreads):
                a, info = self._residualize_map(a)
        elif self.method == "lsmr":
            a, info = self._residualize_lsmr(np.asarray(a))
        else:
            a, info = self._residualize_map(a)

        out = a[:, 0] if one_dim else a
        if self.backend == "cupy":
            out = xp.asnumpy(out)
        return (out, info) if return_info else out

    def _residualize_map(self, a):
        self._iteration_profile = {}
        if len(self.groups) == 1:
            if self.backend == "numpy":
                self._project_inplace_numpy(a, 0)
            else:
                self._project_inplace_gpu(a, 0)
            return a, AbsorbInfo(True, 1, 0.0)
        if self.acceleration == "cg":
            info = self._cg(a)
        else:
            xp = self.xp
            # MAP uses an absolute/relative update metric. Work in per-column
            # units for both auto probes and subsequent plain-MAP pools.
            # Two reductions avoid an N x RHS absolute-value temporary.
            scale = xp.maximum(xp.abs(xp.max(a, axis=0)), xp.abs(xp.min(a, axis=0)))
            scale = xp.where(scale > 0, scale, 1.0)
            a /= scale
            try:
                if self.acceleration == "auto":
                    info = self._auto_map(a)
                else:
                    started = perf_counter() if self._profile_enabled else None
                    info = self._iterate(a)
                    if self._profile_enabled:
                        self._iteration_profile.update(
                            plain_seconds=perf_counter() - started,
                            plain_iterations=info.iterations,
                        )
            finally:
                a *= scale
        return a, info

    @property
    def supports_fe_orthogonality_certificate(self) -> bool:
        """Whether simple FE mean orthogonality exactly certifies absorption."""
        return (
            self.backend == "numpy" and self.method == "map"
            and all(self.intercepts) and all(s is None for s in self.slopes)
        )

    def fe_orthogonality_error(self, data) -> float:
        """Maximum normalized within-FE mean left in a residualized block.

        For pure categorical intercept FEs, exact FWL residuals are orthogonal
        to every FE dummy. This provides a direct post-hoc convergence
        certificate without retaining an N x K previous-iterate matrix.
        """
        if not self.supports_fe_orthogonality_certificate:
            raise ValueError("FE orthogonality certificate requires pure intercept MAP FEs")
        a = np.asarray(data, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        scale = np.maximum(1.0, np.max(np.abs(a), axis=0))
        worst = 0.0
        for codes, L, denom in zip(self.groups, self.levels, self._denoms, strict=False):
            for j in range(a.shape[1]):
                vals = a[:, j] if self.weights is None else self.weights * a[:, j]
                sums = np.bincount(codes, weights=vals, minlength=L)
                means = np.divide(sums, denom, out=np.zeros_like(sums), where=denom != 0)
                err = float(np.max(np.abs(means)) / scale[j]) if means.size else 0.0
                if err > worst:
                    worst = err
        return worst

    def residualize_adaptive_map(
        self, data, *, return_info=False, absorb_threads=None,
        initial_iterations=8, batch_iterations=4, plain_limit=64,
    ):
        """Low-memory checkpointed MAP for ultra-large pure-intercept FE.

        The real full RHS block is transformed in place. Convergence is tested
        by FE orthogonality checkpoints rather than an N x K previous iterate.
        This keeps the execution to one sustained parallel phase and avoids the
        repeated-pool runtime instability seen near cgroup memory limits. If a
        problem remains poorly conditioned after ``plain_limit`` sweeps, LSMR
        is used as a low-memory correctness fallback on the partially
        residualized block.
        """
        if not self.supports_fe_orthogonality_certificate:
            raise ValueError("adaptive MAP requires pure categorical intercept FEs")
        a = np.asarray(data, dtype=self.dtype)
        one_dim = a.ndim == 1
        if one_dim:
            a = a[:, None]
        nthreads = self.absorb_threads if absorb_threads is None else resolve_threads(absorb_threads, nobs=self.nobs, calibration_min_nobs=self.projection_min_nobs)
        g = len(self.groups)
        done = 0
        target = min(self.max_iter, max(1, int(initial_iterations)))
        err = float('inf')
        with self._execution_lock, numba_thread_limit(nthreads):
            while True:
                for _ in range(target - done):
                    self._sweep(a, range(g))
                    if self.transform == "symmetric":
                        self._sweep(a, range(g - 2, -1, -1))
                    elif self.transform != "kaczmarz":
                        raise ValueError("transform must be 'kaczmarz' or 'symmetric'")
                done = target
                err = self.fe_orthogonality_error(a)
                if err <= self.tol:
                    info = AbsorbInfo(True, done, float(err), "fe_orthogonality")
                    break
                if done >= min(self.max_iter, int(plain_limit)):
                    # LSMR is intentionally the fallback here: unlike CG on a
                    # wide 10M-row block it does not require several additional
                    # N x K Krylov arrays. Projection onto the final FE-complement
                    # is unchanged when starting from partially residualized data.
                    out2, linfo = self._residualize_lsmr(a)
                    a[...] = out2
                    info = AbsorbInfo(
                        bool(linfo.converged), done + int(linfo.iterations),
                        float(linfo.max_update), "lsmr_fallback"
                    )
                    break
                target = min(self.max_iter, done + max(1, int(batch_iterations)))
        out = a[:, 0] if one_dim else a
        return (out, info) if return_info else out

    def residualize_fixed_map(
        self, data, iterations: int, *, return_info=False, absorb_threads=None,
        certify=True, batch_iterations=4,
    ):
        """Run fixed MAP sweeps then certify FE orthogonality.

        This execution primitive is used by the large-data planner after a
        real first pool has established a suitable MAP iteration count. It
        changes no projection mathematics; it only avoids a second parallel
        convergence scan on every subsequent RHS pool.
        """
        if self.backend != "numpy" or self.method != "map":
            raise ValueError("fixed MAP execution currently requires NumPy MAP")
        a = np.asarray(data, dtype=self.dtype)
        one_dim = a.ndim == 1
        if one_dim:
            a = a[:, None]
        nthreads = self.absorb_threads if absorb_threads is None else resolve_threads(absorb_threads, nobs=self.nobs, calibration_min_nobs=self.projection_min_nobs)
        g = len(self.groups)
        done = 0
        target = max(1, int(iterations))
        with self._execution_lock, numba_thread_limit(nthreads):
            while True:
                need = target - done
                for _ in range(max(0, need)):
                    self._sweep(a, range(g))
                    if self.transform == "symmetric":
                        self._sweep(a, range(g - 2, -1, -1))
                    elif self.transform != "kaczmarz":
                        raise ValueError("transform must be 'kaczmarz' or 'symmetric'")
                done = target
                if not certify:
                    err = float('nan'); converged = True; break
                err = self.fe_orthogonality_error(a)
                if err <= self.tol:
                    converged = True; break
                if done >= self.max_iter:
                    converged = False; break
                target = min(self.max_iter, done + max(1, int(batch_iterations)))
        info = AbsorbInfo(converged, done, float(err), "fe_orthogonality")
        out = a[:, 0] if one_dim else a
        return (out, info) if return_info else out

    def _term_centered_slope(self, gi: int, j: int) -> np.ndarray:
        S = self.slopes[gi]
        proj = self._projectors[gi]
        if proj.slope_means is None:
            return S[:, j]
        return S[:, j] - proj.slope_means[self.groups[gi], j]

    def _lsmr_operator(self):
        n = self.nobs
        widths = [self.levels[g] * (int(self.intercepts[g]) + (0 if self.slopes[g] is None else self.slopes[g].shape[1])) for g in range(len(self.groups))]
        offsets = np.cumsum([0] + widths)
        p_total = int(offsets[-1])
        sw = np.ones(n, dtype=np.float64) if self.weights is None else np.sqrt(self.weights)

        inv_scales: list[np.ndarray] = []
        for gi in range(len(self.groups)):
            L = self.levels[gi]
            cols = []
            if self.intercepts[gi]:
                cols.append(self._denoms[gi])
            S = self.slopes[gi]
            if S is not None:
                proj = self._projectors[gi]
                for j in range(S.shape[1]):
                    z = self._term_centered_slope(gi, j)
                    vals = z*z if self.weights is None else self.weights*z*z
                    cols.append(np.bincount(self.groups[gi], weights=vals, minlength=L))
            diag = np.column_stack(cols)
            if self.preconditioner == "diagonal":
                inv = np.divide(1.0, np.sqrt(diag), out=np.zeros_like(diag), where=diag > 0)
            else:
                inv = np.ones_like(diag)
            inv_scales.append(inv)

        def matvec(theta):
            out = np.zeros(n, dtype=np.float64)
            for gi, codes in enumerate(self.groups):
                L = self.levels[gi]
                q = inv_scales[gi].shape[1]
                lo, hi = offsets[gi], offsets[gi+1]
                coef = theta[lo:hi].reshape(L, q) * inv_scales[gi]
                c = 0
                if self.intercepts[gi]:
                    out += coef[codes, c]; c += 1
                S = self.slopes[gi]
                if S is not None:
                    for j in range(S.shape[1]):
                        z = self._term_centered_slope(gi, j)
                        out += z * coef[codes, c+j]
            return sw * out

        def rmatvec(v):
            zobs = sw * np.asarray(v, dtype=np.float64)
            parts = []
            for gi, codes in enumerate(self.groups):
                L = self.levels[gi]
                cols = []
                if self.intercepts[gi]:
                    cols.append(np.bincount(codes, weights=zobs, minlength=L))
                S = self.slopes[gi]
                if S is not None:
                    for j in range(S.shape[1]):
                        sj = self._term_centered_slope(gi, j)
                        cols.append(np.bincount(codes, weights=zobs*sj, minlength=L))
                block = np.column_stack(cols) * inv_scales[gi]
                parts.append(block.ravel())
            return np.concatenate(parts)

        return LinearOperator((n, p_total), matvec=matvec, rmatvec=rmatvec, dtype=np.float64), inv_scales, offsets, sw

    def _residualize_lsmr(self, a):
        A, inv_scales, offsets, sw = self._lsmr_operator()
        out = np.empty_like(a, dtype=np.float64)
        worst_it = 0
        converged = True
        max_resid = 0.0
        for j in range(a.shape[1]):
            sol = lsmr(A, sw * a[:, j], atol=self.tol, btol=self.tol, maxiter=self.max_iter)
            theta, istop, itn, normr = sol[0], int(sol[1]), int(sol[2]), float(sol[3])
            fitted_w = A.matvec(theta)
            fitted = np.divide(fitted_w, sw, out=np.zeros_like(fitted_w), where=sw != 0)
            out[:, j] = a[:, j] - fitted
            worst_it = max(worst_it, itn)
            converged = converged and (istop in {0, 1, 2, 4, 5})
            max_resid = max(max_resid, normr)
        return out, AbsorbInfo(converged, worst_it, max_resid)

    def solve_effects(self, target):
        """Recover one minimum-norm decomposition of the absorbed contribution.

        The sum of returned term contributions is identified on the estimation
        sample even when individual FE coefficient vectors are not. Coefficients
        are reported in the original (uncentered) slope basis.
        """
        target = np.asarray(target, dtype=np.float64)
        if target.ndim != 1 or len(target) != self.nobs or not np.all(np.isfinite(target)):
            raise ValueError("target must be a one-dimensional nobs vector")
        A, inv_scales, offsets, sw = self._lsmr_operator()
        sol = lsmr(A, sw * target, atol=self.tol, btol=self.tol, maxiter=self.max_iter)
        theta, istop, itn, normr = sol[0], int(sol[1]), int(sol[2]), float(sol[3])
        terms = []
        reconstructed = np.zeros(self.nobs, dtype=np.float64)
        for gi, codes in enumerate(self.groups):
            L = self.levels[gi]
            q = inv_scales[gi].shape[1]
            lo, hi = offsets[gi], offsets[gi+1]
            coef = theta[lo:hi].reshape(L, q) * inv_scales[gi]
            col = 0
            intercept = None
            if self.intercepts[gi]:
                intercept = coef[:, col].copy()
                col += 1
            slopes = None
            S = self.slopes[gi]
            if S is not None and S.shape[1]:
                slopes = coef[:, col:col+S.shape[1]].copy()
                proj = self._projectors[gi]
                if intercept is not None and proj.slope_means is not None:
                    # alpha + (s-m)'gamma == (alpha-m'gamma) + s'gamma
                    intercept -= np.einsum("gj,gj->g", proj.slope_means, slopes)
            contribution = np.zeros(self.nobs, dtype=np.float64)
            if intercept is not None:
                contribution += intercept[codes]
            if slopes is not None:
                for j in range(slopes.shape[1]):
                    contribution += S[:, j] * slopes[codes, j]
            reconstructed += contribution
            terms.append((intercept, slopes, contribution))
        denom = max(float(np.linalg.norm(target)), np.finfo(float).eps)
        recon_error = float(np.linalg.norm(target - reconstructed) / denom)
        return terms, {
            "converged": istop in {0, 1, 2, 4, 5},
            "iterations": itn,
            "residual_norm": normr,
            "reconstruction_error": recon_error,
        }

    def _sweep(self, a, order):
        fn = self._project_inplace_numpy if self.backend == "numpy" else self._project_inplace_gpu
        for gi in order:
            fn(a, gi)

    def _weighted_colsum(self, a, b):
        xp = self.xp
        if self.backend == "numpy":
            # Fuse multiplication and row reduction: avoid N x RHS temporaries
            # in each CG step. Do not choose a contraction path that recreates
            # them. Floating-point summation order can differ across layouts;
            # keep the existing dtypes, convergence checks and GPU path.
            if self.weights is None:
                return np.einsum("ij,ij->j", a, b, optimize=False)
            return np.einsum("i,ij,ij->j", self.weights, a, b, optimize=False)
        if self._gpu_weights is None:
            return xp.sum(a*b, axis=0)
        return xp.sum(self._gpu_weights[:, None]*a*b, axis=0)

    def _sym_projection(self, a, out=None):
        z = a.copy() if out is None else out
        if out is not None:
            z[...] = a
        g = len(self.groups)
        self._sweep(z, range(g))
        self._sweep(z, range(g-2, -1, -1))
        self.xp.subtract(a, z, out=z)
        return z

    def _sym_projection_with_inner_product(self, a, out):
        out[...] = a
        g = len(self.groups)
        self._sweep(out, range(g))
        self._sweep(out, range(g-2, -1, -1))
        return out, _cg_subtract_and_inner_product(a, out, self.weights)

    def _cg(self, a, *, max_iter=None):
        """Symmetric-Kaczmarz CG with reghdfe-style Hestenes stopping rule."""
        xp = self.xp
        eps = xp.finfo(a.dtype).eps
        limit = self.max_iter if max_iter is None else min(self.max_iter, max_iter)
        # Use fused reductions for large CPU float64 blocks with multiple columns.
        parallel_cg = (
            self.backend == "numpy" and a.dtype == np.float64
            and a.shape[0] >= _CG_PARALLEL_MIN_ROWS
            and a.shape[1] >= 2
            and get_num_threads() > 1
        )
        timed = self._profile_enabled
        started = perf_counter() if timed else None
        projection_seconds = 0.0

        # Hestenes/Stiefel stopping rule used by reghdfe: compare the most
        # recent objective improvement with the remaining improvement
        # potential, rather than requiring the raw Krylov residual itself to
        # become tiny.  This avoids false non-convergence once the absorbed
        # variables are already numerically accurate.
        initial_energy = self._weighted_colsum(a, a)
        improvement_potential = initial_energy.copy()
        # Preserve units: all stopping denominators scale with each RHS energy.
        # Reject unrepresentable norms instead of reporting a zero-step success.
        if not bool(xp.all(xp.isfinite(initial_energy))):
            return AbsorbInfo(False, 0, float("inf"), "cg_breakdown")
        for j in range(a.shape[1]):
            if initial_energy[j] == 0:
                values = a[:, j]
                nonzero = values != 0
                if self.weights is not None:
                    nonzero &= (self._gpu_weights if self.backend == "cupy" else self.weights) > 0
                if bool(xp.any(nonzero)):
                    return AbsorbInfo(False, 0, float("inf"), "cg_breakdown")
        energy_floor = float(eps) * initial_energy
        # Squared machine precision is a per-column roundoff threshold, not
        # an absolute floor in the user's units. It prevents CG from amplifying
        # projection noise after an exact/near-exact FE removal.
        roundoff_floor = float(eps) * energy_floor
        floor_host = xp.asnumpy(energy_floor) if self.backend == "cupy" else np.asarray(energy_floor)
        projection_start = perf_counter() if timed else None
        r = self._sym_projection(a)
        if timed:
            projection_seconds += perf_counter() - projection_start
        u = r.copy()
        v = xp.empty_like(a)
        rr = self._weighted_colsum(r, r)
        max_update = float("inf")

        converged = False
        criterion = "hestenes"
        for it in range(1, limit + 1):
            projection_start = perf_counter() if timed else None
            if parallel_cg:
                _, uv = self._sym_projection_with_inner_product(u, out=v)
            else:
                self._sym_projection(u, out=v)
            if timed:
                projection_seconds += perf_counter() - projection_start
            if not parallel_cg:
                uv = self._weighted_colsum(u, v)
            active = rr > roundoff_floor
            if not bool(xp.all(xp.isfinite(uv) & xp.isfinite(rr) & (rr >= 0) & ((uv > 0) | ~active))):
                criterion = "cg_breakdown"
                max_update = float("inf")
                break
            alpha = xp.divide(rr, uv, out=xp.zeros_like(rr), where=active & (uv > 0))
            recent = alpha * rr
            improvement_potential = improvement_potential - recent
            if parallel_cg:
                rr_new = _cg_update_residuals_and_norm(a, r, u, v, alpha, self.weights)
            elif self.backend == "numpy":
                _cg_update_residuals(a, r, u, v, alpha)
            else:
                a -= u * alpha[None, :]
                r -= v * alpha[None, :]

            recent_host = xp.asnumpy(recent) if self.backend == "cupy" else np.asarray(recent)
            pot_host = xp.asnumpy(improvement_potential) if self.backend == "cupy" else np.asarray(improvement_potential)
            num = np.maximum(recent_host, 0.0)
            den = np.maximum(pot_host, floor_host)
            ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            max_update = float(np.sqrt(np.max(ratio)))
            if max_update <= self.tol:
                converged = True
                break

            if not parallel_cg:
                rr_new = self._weighted_colsum(r, r)
            beta = xp.divide(rr_new, rr, out=xp.zeros_like(rr), where=active & (rr > 0))
            if parallel_cg:
                _cg_update_direction_parallel(u, r, beta)
            elif self.backend == "numpy":
                _cg_update_direction(u, r, beta)
            else:
                u = r + u * beta[None, :]
            rr = rr_new
        if timed:
            elapsed = perf_counter() - started
            self._iteration_profile.update(
                cg_seconds=elapsed, cg_iterations=it,
                cg_projection_seconds=projection_seconds,
                cg_algebra_seconds=elapsed - projection_seconds,
                cg_parallel_fused=parallel_cg,
                cg_projection_includes_inner_product=parallel_cg,
            )
        return AbsorbInfo(converged, it, max_update, criterion)

    def _iterate_limit(self, a, limit):
        xp = self.xp
        eps = xp.finfo(a.dtype).eps
        g = len(self.groups)
        last = xp.empty_like(a)
        max_update = xp.inf
        for it in range(1, min(int(limit), self.max_iter) + 1):
            last[...] = a
            self._sweep(a, range(g))
            self._sweep(a, range(g - 2, -1, -1))
            if self.backend == "numpy":
                max_update_f = float(_max_relative_update_parallel(a, last, float(eps)))
            else:
                scale = xp.maximum(1.0, xp.abs(last))
                max_update_f = float(xp.max(xp.abs(a-last)/(scale+eps)))
            max_update = max_update_f
            if max_update_f <= self.tol:
                return AbsorbInfo(True, it, float(max_update_f), "auto_plain")
        return AbsorbInfo(False, min(int(limit), self.max_iter), float(max_update), "auto_plain")

    def _auto_warmup(self, a):
        """Run two real symmetric-MAP sweeps and estimate the contraction rate.

        The sweeps mutate the actual RHS block.  If the topology is hard, CG
        starts from this partially residualized block; since the warmup only
        subtracts vectors in the FE span, the final FWL projection is unchanged.
        """
        xp = self.xp
        eps = float(xp.finfo(a.dtype).eps)
        g = len(self.groups)
        last = xp.empty_like(a)
        updates = []
        nprobe = min(self._auto_probe_iterations, self.max_iter)
        for it in range(1, nprobe + 1):
            last[...] = a
            self._sweep(a, range(g))
            self._sweep(a, range(g - 2, -1, -1))
            if self.backend == "numpy":
                upd = float(_max_relative_update_parallel(a, last, eps))
            else:
                scale = xp.maximum(1.0, xp.abs(last))
                upd = float(xp.max(xp.abs(a - last) / (scale + eps)))
            updates.append(float(upd))
            if upd <= self.tol:
                return AbsorbInfo(True, it, float(upd), "auto_plain"), 0.0
        if len(updates) < 2:
            return AbsorbInfo(False, nprobe, updates[-1] if updates else float("inf"), "auto_plain"), float("inf")
        ratio = updates[-1] / max(updates[-2], eps)
        return AbsorbInfo(False, nprobe, updates[-1], "auto_plain"), float(ratio)

    def _polish_orthogonality(self, a, limit):
        """Plain symmetric sweeps used only when CG's cheap stop is too loose."""
        if not self.supports_fe_orthogonality_certificate or limit <= 0:
            return AbsorbInfo(True, 0, float("nan"), "orthogonality_unchecked")
        err = self.fe_orthogonality_error(a)
        if err <= self.tol:
            return AbsorbInfo(True, 0, err, "fe_orthogonality")
        g = len(self.groups)
        for it in range(1, min(int(limit), self.max_iter) + 1):
            self._sweep(a, range(g))
            self._sweep(a, range(g - 2, -1, -1))
            err = self.fe_orthogonality_error(a)
            if err <= self.tol:
                return AbsorbInfo(True, it, err, "fe_orthogonality")
        return AbsorbInfo(False, min(int(limit), self.max_iter), err, "fe_orthogonality")

    def _auto_map(self, a):
        """Two-sweep warmup; continue plain MAP only for fast contractions."""
        timed = self._profile_enabled
        started = perf_counter() if timed else None
        warm, ratio = self._auto_warmup(a)
        if timed:
            self._iteration_profile.update(
                probe_seconds=perf_counter() - started,
                probe_iterations=warm.iterations, contraction_ratio=ratio,
            )
        if warm.converged or warm.iterations >= self.max_iter:
            return warm

        total = warm.iterations
        if ratio <= self._auto_contraction_threshold:
            remain = max(0, min(self.auto_plain_limit, self.max_iter) - total)
            if remain:
                started = perf_counter() if timed else None
                first = self._iterate_limit(a, remain)
                if timed:
                    self._iteration_profile.update(
                        plain_seconds=perf_counter() - started,
                        plain_iterations=first.iterations,
                    )
                total += first.iterations
                if first.converged or total >= self.max_iter:
                    return AbsorbInfo(
                        first.converged, total, first.max_update, "auto_plain"
                    )
            # Safety valve for a rare optimistic contraction classification.
            cg = self._cg(a, max_iter=self.max_iter - total)
            total += cg.iterations
        else:
            # Hard spectrum.  The two warmup sweeps are useful preconditioning,
            # so CG starts from the partially residualized block rather than
            # restarting from the raw RHS.
            cg = self._cg(a, max_iter=self.max_iter - total)
            total += cg.iterations

        # Use the same CG stopping rule as direct calls. FE orthogonality is
        # an explicit diagnostic; default no-core solves need no extra full
        # FE x RHS scan after CG.
        return AbsorbInfo(cg.converged, total, cg.max_update, "auto_cg")

    def _iterate(self, a):
        xp = self.xp
        eps = xp.finfo(a.dtype).eps
        g = len(self.groups)
        last = xp.empty_like(a)
        max_update = xp.inf
        # Chunked convergence check prevents three additional full-size
        # N x RHS temporaries (scale, diff, ratio) at 10M+ observations.
        check_rows = 250_000
        for it in range(1, self.max_iter + 1):
            last[...] = a
            self._sweep(a, range(g))
            if self.transform == "symmetric":
                self._sweep(a, range(g - 2, -1, -1))
            elif self.transform != "kaczmarz":
                raise ValueError("transform must be 'kaczmarz' or 'symmetric'")
            if self.backend == "numpy":
                max_update_f = float(_max_relative_update_parallel(a, last, float(eps)))
            else:
                max_update_f = 0.0
                for lo in range(0, a.shape[0], check_rows):
                    hi = min(a.shape[0], lo + check_rows)
                    prev = last[lo:hi]
                    cur = a[lo:hi]
                    scale = xp.maximum(1.0, xp.abs(prev))
                    upd = xp.max(xp.abs(cur - prev) / (scale + eps))
                    u = float(upd)
                    if u > max_update_f:
                        max_update_f = u
            max_update = max_update_f
            if max_update_f <= self.tol:
                return AbsorbInfo(True, it, max_update_f)
        return AbsorbInfo(False, self.max_iter, float(max_update))


def iterative_singleton_mask(
    groups: list[np.ndarray], weights=None, *, frequency_weights: bool = False
) -> tuple[np.ndarray, int]:
    """Iteratively remove observations singleton in any absorbed FE group.

    With Stata frequency weights, group degree is the sum of frequency mass,
    matching the dataset that would be obtained by physically expanding the
    rows. Analytic/probability/generic weights must leave ``frequency_weights``
    false because a weighted row still counts as one sampled observation.
    """
    if not groups:
        return np.ones(0, dtype=bool), 0
    n = len(groups[0])
    keep = np.ones(n, dtype=bool)
    fw = None
    if frequency_weights:
        if weights is None:
            raise ValueError("frequency_weights=True requires weights")
        fw = np.asarray(weights, dtype=np.float64)
        if len(fw) != n:
            raise ValueError("frequency weights length mismatch")
    dropped_total = 0
    while True:
        drop = np.zeros(n, dtype=bool)
        idx = np.flatnonzero(keep)
        for g in groups:
            active = g[idx]
            nlev = int(g.max()) + 1 if len(g) else 0
            if fw is None:
                counts = np.bincount(active, minlength=nlev)
            else:
                counts = np.bincount(active, weights=fw[idx], minlength=nlev)
            drop[idx] |= counts[active] <= 1.0 + 1e-12
        new = drop & keep
        c = int(new.sum())
        if c == 0:
            break
        keep[new] = False
        dropped_total += c
    return keep, dropped_total
