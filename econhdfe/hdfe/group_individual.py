from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from .absorber import AbsorbInfo, HDFEAbsorber, singleton_rows
from .dof import DofInfo, absorbed_dof


@dataclass(slots=True)
class GroupCompression:
    """Mapping from long membership rows to group-level estimation rows."""

    row_to_group: np.ndarray
    first_rows: np.ndarray
    group_sizes: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(self.first_rows.size)

    @property
    def n_memberships(self) -> int:
        return int(self.row_to_group.size)


@dataclass(slots=True)
class GroupIndividualInfo:
    n_groups: int
    n_memberships: int
    n_individuals: int
    aggregation: str
    group_sizes: np.ndarray
    first_rows: np.ndarray
    individual_intercept: bool
    individual_slopes: int
    solver: str
    incidence_backend: str
    dof_note: str = "individual FE redundancy is conservatively unadjusted, matching current reghdfe"


def group_compression(group_codes: np.ndarray) -> GroupCompression:
    codes = np.asarray(group_codes, dtype=np.int32)
    if codes.ndim != 1:
        raise ValueError("group must be one-dimensional")
    if codes.size == 0:
        return GroupCompression(codes.copy(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    ng = int(codes.max()) + 1
    first = np.full(ng, codes.size, dtype=np.int64)
    np.minimum.at(first, codes, np.arange(codes.size, dtype=np.int64))
    sizes = np.bincount(codes, minlength=ng).astype(np.int64, copy=False)
    if np.any(first == codes.size):
        raise ValueError("group codes must be dense")
    return GroupCompression(codes, first, sizes)


def _equal_with_nan(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


def compress_constant(values, comp: GroupCompression, *, name: str = "variable"):
    """Return one value per group, requiring exact within-group constancy."""
    a = np.asarray(values)
    if a.shape[0] != comp.n_memberships:
        raise ValueError(f"{name} length mismatch")
    ref = a[comp.first_rows]
    expanded = ref[comp.row_to_group]
    eq = _equal_with_nan(a, expanded)
    if a.ndim > 1:
        eq = np.all(eq, axis=tuple(range(1, a.ndim)))
    if not bool(np.all(eq)):
        bad = int(np.flatnonzero(~eq)[0])
        raise ValueError(
            f"{name} must be constant within group(); first mismatch is long-format row {bad}"
        )
    return ref


def validate_unique_memberships(group_codes: np.ndarray, individual_codes: np.ndarray) -> None:
    g = np.asarray(group_codes, dtype=np.uint64)
    i = np.asarray(individual_codes, dtype=np.uint64)
    if len(g) != len(i):
        raise ValueError("group and individual lengths differ")
    # Dense codes are int32, so a collision-free 64-bit packed pair is enough.
    packed = (g << np.uint64(32)) | i
    if np.unique(packed).size != packed.size:
        raise ValueError("group() and individual() must uniquely identify long-format membership rows")


def group_individual_singleton_mask(
    standard_groups: list[np.ndarray],
    membership_group: np.ndarray,
    individual_codes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Recursively prune singleton group observations.

    A group is removed if it is singleton in any ordinary FE or contains an
    individual whose remaining graph degree is one. Removing a group can make
    other individual vertices degree one, so pruning is repeated to the 1-core.
    This is equivalent to reghdfe's bipartite core-number logic for individual
    fixed effects, while avoiding an explicit graph object.
    """
    mg = np.asarray(membership_group, dtype=np.int32)
    ind = np.asarray(individual_codes, dtype=np.int32)
    ng = int(mg.max()) + 1 if mg.size else (len(standard_groups[0]) if standard_groups else 0)
    keep = np.ones(ng, dtype=bool)
    nind = int(ind.max()) + 1 if ind.size else 0
    std = [np.asarray(g, dtype=np.int32) for g in standard_groups]
    std_levels = [int(g.max()) + 1 if g.size else 0 for g in std]
    dropped = 0
    while True:
        drop = np.zeros(ng, dtype=bool)
        active_groups = np.flatnonzero(keep)
        if active_groups.size == 0:
            break

        drop[active_groups] = singleton_rows(std, std_levels, active_groups)

        active_membership = keep[mg]
        degree = np.bincount(ind[active_membership], minlength=nind)
        singleton_member = active_membership & (degree[ind] <= 1)
        if np.any(singleton_member):
            bad_groups = np.zeros(ng, dtype=np.uint8)
            np.maximum.at(bad_groups, mg[singleton_member], 1)
            drop |= bad_groups.astype(bool)

        new = drop & keep
        c = int(new.sum())
        if not c:
            break
        keep[new] = False
        dropped += c
    return keep, dropped


def _redense_group_vector(a: np.ndarray, keep: np.ndarray) -> np.ndarray:
    vals = np.asarray(a)[keep]
    _, inv = np.unique(vals, return_inverse=True)
    return inv.astype(np.int32, copy=False)


def restrict_group_problem(
    standard_groups: list[np.ndarray],
    standard_slopes: list[np.ndarray | None],
    membership_group: np.ndarray,
    individual_codes: np.ndarray,
    individual_slopes: np.ndarray | None,
    keep_groups: np.ndarray,
):
    """Apply group pruning and densely recode all remaining FE identifiers."""
    keep_groups = np.asarray(keep_groups, dtype=bool)
    old_to_new = np.full(keep_groups.size, -1, dtype=np.int32)
    old_to_new[keep_groups] = np.arange(int(keep_groups.sum()), dtype=np.int32)
    member_keep = keep_groups[np.asarray(membership_group, dtype=np.int32)]
    mg = old_to_new[np.asarray(membership_group, dtype=np.int32)[member_keep]]
    old_ind = np.asarray(individual_codes, dtype=np.int32)[member_keep]
    _, ind = np.unique(old_ind, return_inverse=True)
    ind = ind.astype(np.int32, copy=False)
    islopes = None if individual_slopes is None else np.asarray(individual_slopes, dtype=np.float64)[member_keep]
    groups = [_redense_group_vector(g, keep_groups) for g in standard_groups]
    slopes = [None if s is None else np.asarray(s, dtype=np.float64)[keep_groups] for s in standard_slopes]
    return groups, slopes, mg, ind, islopes, member_keep


class GroupIndividualAbsorber:
    """Matrix-free absorber for group-level outcomes with individual FEs.

    Estimation rows are unique groups, while ``membership_group`` and
    ``individual_codes`` describe the long-format incidence relation. Ordinary
    FE terms live at the group level. The joint design is solved with LSMR or
    LSQR through ``LinearOperator``; no group-by-individual dummy matrix is
    materialized.
    """

    def __init__(
        self,
        standard_groups,
        *,
        standard_slopes=None,
        standard_intercepts=None,
        membership_group,
        individual_codes,
        individual_slopes=None,
        individual_intercept=True,
        aggregation="mean",
        weights=None,
        tol=1e-8,
        max_iter=10_000,
        method="lsmr",
        preconditioner="diagonal",
        incidence_backend="auto",
        memory_budget_mb=512,
    ):
        if method not in {"lsmr", "lsqr"}:
            raise ValueError("group/individual fixed effects require method='lsmr' or 'lsqr'")
        if preconditioner not in {"diagonal", "none"}:
            raise ValueError("preconditioner must be 'diagonal' or 'none'")
        aggregation = str(aggregation).lower()
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be 'mean' or 'sum'")
        self.method = method
        self.preconditioner = preconditioner
        self.acceleration = "none"
        self.backend = "numpy"
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.aggregation = aggregation
        incidence_backend = str(incidence_backend).lower().replace("-", "_")
        if incidence_backend not in {"auto", "matrix_free", "csr"}:
            raise ValueError("incidence_backend must be auto, matrix_free, or csr")
        self.memory_budget_mb = float(memory_budget_mb)

        self.standard_groups = [np.asarray(g, dtype=np.int32) for g in standard_groups]
        self.standard_slopes = [None if s is None else np.asarray(s, dtype=np.float64) for s in (standard_slopes or [None] * len(self.standard_groups))]
        self.standard_intercepts = list(standard_intercepts or [True] * len(self.standard_groups))
        if len(self.standard_slopes) != len(self.standard_groups) or len(self.standard_intercepts) != len(self.standard_groups):
            raise ValueError("standard FE specification length mismatch")

        self.membership_group = np.asarray(membership_group, dtype=np.int32)
        self.individual_codes = np.asarray(individual_codes, dtype=np.int32)
        if self.membership_group.size != self.individual_codes.size:
            raise ValueError("membership group and individual code lengths differ")
        if self.membership_group.size:
            self.nobs = int(self.membership_group.max()) + 1
        elif self.standard_groups:
            self.nobs = len(self.standard_groups[0])
        else:
            raise ValueError("empty group/individual FE problem")
        if any(len(g) != self.nobs for g in self.standard_groups):
            raise ValueError("ordinary fixed effects must have one value per group")

        self.n_individuals = int(self.individual_codes.max()) + 1 if self.individual_codes.size else 0
        self.individual_intercept = bool(individual_intercept)
        self.individual_slopes = None if individual_slopes is None else np.asarray(individual_slopes, dtype=np.float64)
        if self.individual_slopes is not None:
            if self.individual_slopes.ndim == 1:
                self.individual_slopes = self.individual_slopes[:, None]
            if self.individual_slopes.shape[0] != self.membership_group.size:
                raise ValueError("individual slope block must have one row per membership")
        self.n_individual_slopes = 0 if self.individual_slopes is None else int(self.individual_slopes.shape[1])
        if not self.individual_intercept and self.n_individual_slopes == 0:
            raise ValueError("individual FE term has neither intercept nor slopes")

        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights is not None:
            if len(self.weights) != self.nobs:
                raise ValueError("group-level weights length mismatch")
            if np.any(self.weights < 0):
                raise ValueError("weights must be nonnegative")
        self._sw = np.ones(self.nobs, dtype=np.float64) if self.weights is None else np.sqrt(self.weights)

        group_sizes = np.bincount(self.membership_group, minlength=self.nobs).astype(np.float64, copy=False)
        if np.any(group_sizes == 0):
            raise ValueError("every group estimation row must have at least one individual membership")
        self.group_sizes = group_sizes
        self.member_scale = np.ones_like(self.membership_group, dtype=np.float64)
        if self.aggregation == "mean":
            self.member_scale = 1.0 / group_sizes[self.membership_group]

        self._std_absorber = None
        self._std_op = None
        self._std_inv_scales = []
        self._std_offsets = np.array([0], dtype=np.int64)
        self._std_width = 0
        self._ind_q = int(self.individual_intercept) + self.n_individual_slopes
        self._ind_width = self.n_individuals * self._ind_q
        self._csr_inv_scales = None
        self._csr_offsets = None
        self._csr_weighted = None

        std_nnz = 0
        for has_int, S in zip(self.standard_intercepts, self.standard_slopes, strict=False):
            std_nnz += self.nobs * (int(has_int) + (0 if S is None else S.shape[1]))
        ind_nnz = self.membership_group.size * self._ind_q
        estimated_peak = 32.0 * (std_nnz + ind_nnz) + 16.0 * self.nobs
        budget = max(self.memory_budget_mb, 0.0) * (1024.0**2)
        if incidence_backend == "auto":
            # CSR is substantially faster for the common sparse incidence case,
            # but COO->CSR construction temporarily holds multiple index/data
            # arrays. Fall back to the fully matrix-free operator under a tight
            # memory budget or a very large membership graph.
            incidence_backend = "csr" if estimated_peak <= 0.45 * budget else "matrix_free"
        self.incidence_backend = incidence_backend

        if self.incidence_backend == "csr":
            self._operator = self._build_csr_operator()
        else:
            if self.standard_groups:
                self._std_absorber = HDFEAbsorber(
                    self.standard_groups,
                    slopes=self.standard_slopes,
                    intercepts=self.standard_intercepts,
                    weights=self.weights,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    method="lsmr",
                    preconditioner=self.preconditioner,
                    projection_memory_budget_mb=self.memory_budget_mb,
                )
                self._std_op, self._std_inv_scales, self._std_offsets, _ = self._std_absorber._lsmr_operator()
                self._std_width = int(self._std_op.shape[1])
            self._ind_inv_scales = self._build_individual_preconditioner()
            self._operator = self._build_operator()

    def _raw_standard_term_csr(self, codes, slopes, has_intercept):
        L = int(codes.max()) + 1 if len(codes) else 0
        p = 0 if slopes is None else int(slopes.shape[1])
        q = int(has_intercept) + p
        rows = np.repeat(np.arange(self.nobs, dtype=np.int32), q)
        comps = np.arange(q, dtype=np.int32)
        cols = (codes[:, None] * q + comps[None, :]).ravel()
        vals = np.empty((self.nobs, q), dtype=np.float64)
        c = 0
        if has_intercept:
            vals[:, c] = 1.0
            c += 1
        if p:
            vals[:, c:c+p] = slopes
        return coo_matrix((vals.ravel(), (rows, cols)), shape=(self.nobs, L*q)).tocsr()

    def _raw_individual_term_csr(self):
        q = self._ind_q
        M = self.membership_group.size
        rows = np.repeat(self.membership_group, q)
        comps = np.arange(q, dtype=np.int32)
        cols = (self.individual_codes[:, None] * q + comps[None, :]).ravel()
        vals = np.empty((M, q), dtype=np.float64)
        c = 0
        if self.individual_intercept:
            vals[:, c] = self.member_scale
            c += 1
        if self.n_individual_slopes:
            vals[:, c:c+self.n_individual_slopes] = self.member_scale[:, None] * self.individual_slopes
        return coo_matrix(
            (vals.ravel(), (rows, cols)), shape=(self.nobs, self.n_individuals*q)
        ).tocsr()

    def _build_csr_operator(self):
        terms = []
        offsets = [0]
        for codes, slopes, has_int in zip(
            self.standard_groups, self.standard_slopes, self.standard_intercepts, strict=False
        ):
            term = self._raw_standard_term_csr(codes, slopes, has_int)
            terms.append(term)
            offsets.append(offsets[-1] + term.shape[1])
        ind_term = self._raw_individual_term_csr()
        terms.append(ind_term)
        offsets.append(offsets[-1] + ind_term.shape[1])
        raw = terms[0] if len(terms) == 1 else hstack(terms, format="csr")
        weighted = raw if self.weights is None else raw.multiply(self._sw[:, None]).tocsr()
        diag = np.asarray(weighted.power(2).sum(axis=0)).ravel()
        if self.preconditioner == "none":
            inv = np.ones_like(diag)
        else:
            inv = np.divide(1.0, np.sqrt(diag), out=np.zeros_like(diag), where=diag > 0)
        self._csr_weighted = weighted
        self._csr_inv_scales = inv
        self._csr_offsets = np.asarray(offsets, dtype=np.int64)
        self._std_width = int(offsets[-2])
        total = int(weighted.shape[1])

        def matvec(theta):
            return weighted @ (np.asarray(theta, dtype=np.float64) * inv)

        def rmatvec(v):
            return np.asarray(weighted.T @ np.asarray(v, dtype=np.float64)).ravel() * inv

        return LinearOperator((self.nobs, total), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    def _build_individual_preconditioner(self):
        q = int(self.individual_intercept) + self.n_individual_slopes
        diag = np.empty((self.n_individuals, q), dtype=np.float64)
        obs_w = np.ones(self.nobs, dtype=np.float64) if self.weights is None else self.weights
        base = obs_w[self.membership_group] * self.member_scale * self.member_scale
        c = 0
        if self.individual_intercept:
            diag[:, c] = np.bincount(self.individual_codes, weights=base, minlength=self.n_individuals)
            c += 1
        if self.individual_slopes is not None:
            for j in range(self.n_individual_slopes):
                x = self.individual_slopes[:, j]
                diag[:, c+j] = np.bincount(
                    self.individual_codes, weights=base*x*x, minlength=self.n_individuals
                )
        if self.preconditioner == "none":
            return np.ones_like(diag)
        return np.divide(1.0, np.sqrt(diag), out=np.zeros_like(diag), where=diag > 0)

    def _individual_matvec(self, theta):
        coef = np.asarray(theta, dtype=np.float64).reshape(self.n_individuals, self._ind_q) * self._ind_inv_scales
        c = 0
        member = np.zeros(self.membership_group.size, dtype=np.float64)
        if self.individual_intercept:
            member += coef[self.individual_codes, c]
            c += 1
        if self.individual_slopes is not None:
            for j in range(self.n_individual_slopes):
                member += self.individual_slopes[:, j] * coef[self.individual_codes, c+j]
        member *= self.member_scale
        return np.bincount(self.membership_group, weights=member, minlength=self.nobs)

    def _individual_rmatvec(self, v_weighted):
        # ``v_weighted`` is already sqrt(W) * v because the outer operator is
        # expressed on weighted rows. Expand group-level scores to memberships.
        expanded = np.asarray(v_weighted, dtype=np.float64)[self.membership_group] * self.member_scale
        parts = np.empty((self.n_individuals, self._ind_q), dtype=np.float64)
        c = 0
        if self.individual_intercept:
            parts[:, c] = np.bincount(self.individual_codes, weights=expanded, minlength=self.n_individuals)
            c += 1
        if self.individual_slopes is not None:
            for j in range(self.n_individual_slopes):
                parts[:, c+j] = np.bincount(
                    self.individual_codes,
                    weights=expanded*self.individual_slopes[:, j],
                    minlength=self.n_individuals,
                )
        return (parts * self._ind_inv_scales).ravel()

    def _build_operator(self):
        total = self._std_width + self._ind_width
        sw = self._sw

        def matvec(theta):
            theta = np.asarray(theta, dtype=np.float64)
            out_w = np.zeros(self.nobs, dtype=np.float64)
            if self._std_width:
                out_w += self._std_op.matvec(theta[:self._std_width])
            if self._ind_width:
                out_w += sw * self._individual_matvec(theta[self._std_width:])
            return out_w

        def rmatvec(v):
            v = np.asarray(v, dtype=np.float64)
            parts = []
            if self._std_width:
                parts.append(self._std_op.rmatvec(v))
            if self._ind_width:
                parts.append(self._individual_rmatvec(sw * v))
            return np.concatenate(parts) if len(parts) > 1 else parts[0]

        return LinearOperator((self.nobs, total), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    def _solve(self, rhs):
        b = self._sw * np.asarray(rhs, dtype=np.float64)
        if self.method == "lsmr":
            sol = lsmr(self._operator, b, atol=self.tol, btol=self.tol, maxiter=self.max_iter)
            theta, istop, itn, normr = sol[0], int(sol[1]), int(sol[2]), float(sol[3])
        else:
            sol = lsqr(self._operator, b, atol=self.tol, btol=self.tol, iter_lim=self.max_iter)
            theta, istop, itn, normr = sol[0], int(sol[1]), int(sol[2]), float(sol[3])
        return theta, istop in {0, 1, 2, 4, 5}, itn, normr

    def residualize(self, data, *, copy=True, return_info=False):
        a = np.asarray(data, dtype=np.float64)
        one_dim = a.ndim == 1
        if one_dim:
            a = a[:, None]
        if a.shape[0] != self.nobs:
            raise ValueError("data must have one row per unique group")
        out = np.empty_like(a, dtype=np.float64)
        converged = True
        worst_it = 0
        worst_norm = 0.0
        for j in range(a.shape[1]):
            theta, ok, itn, normr = self._solve(a[:, j])
            fitted_w = self._operator.matvec(theta)
            fitted = np.divide(fitted_w, self._sw, out=np.zeros_like(fitted_w), where=self._sw != 0)
            out[:, j] = a[:, j] - fitted
            converged &= ok
            worst_it = max(worst_it, itn)
            worst_norm = max(worst_norm, normr)
        result = out[:, 0] if one_dim else out
        info = AbsorbInfo(converged, worst_it, worst_norm)
        return (result, info) if return_info else result

    def solve_effects(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.ndim != 1 or target.size != self.nobs:
            raise ValueError("target must be one-dimensional with one value per group")
        theta, ok, itn, normr = self._solve(target)
        terms = []
        reconstructed = np.zeros(self.nobs, dtype=np.float64)

        if self.incidence_backend == "csr":
            rawcoef = theta * self._csr_inv_scales
            term_index = 0
            for codes, S, has_int in zip(
                self.standard_groups, self.standard_slopes, self.standard_intercepts, strict=False
            ):
                lo, hi = self._csr_offsets[term_index], self._csr_offsets[term_index+1]
                L = int(codes.max()) + 1 if len(codes) else 0
                p = 0 if S is None else S.shape[1]
                q = int(has_int) + p
                coef = rawcoef[lo:hi].reshape(L, q)
                c = 0
                intercept = None
                if has_int:
                    intercept = coef[:, c].copy(); c += 1
                slopes = None if p == 0 else coef[:, c:c+p].copy()
                contrib = np.zeros(self.nobs, dtype=np.float64)
                if intercept is not None:
                    contrib += intercept[codes]
                if slopes is not None:
                    for j in range(p):
                        contrib += S[:, j] * slopes[codes, j]
                reconstructed += contrib
                terms.append((intercept, slopes, contrib))
                term_index += 1
            lo, hi = self._csr_offsets[-2], self._csr_offsets[-1]
            tind = rawcoef[lo:hi].reshape(self.n_individuals, self._ind_q)
            c = 0
            intercept = None
            if self.individual_intercept:
                intercept = tind[:, c].copy(); c += 1
            slopes = None if self.n_individual_slopes == 0 else tind[:, c:c+self.n_individual_slopes].copy()
            member = np.zeros(self.membership_group.size, dtype=np.float64)
            c = 0
            if intercept is not None:
                member += intercept[self.individual_codes]; c += 1
            if slopes is not None:
                for j in range(self.n_individual_slopes):
                    member += self.individual_slopes[:, j] * slopes[self.individual_codes, j]
            member *= self.member_scale
            contrib = np.bincount(self.membership_group, weights=member, minlength=self.nobs)
            reconstructed += contrib
            terms.append((intercept, slopes, contrib))
        else:
            if self._std_width:
                for gi, codes in enumerate(self.standard_groups):
                    L = self._std_absorber.levels[gi]
                    q = self._std_inv_scales[gi].shape[1]
                    lo, hi = self._std_offsets[gi], self._std_offsets[gi+1]
                    coef = theta[lo:hi].reshape(L, q) * self._std_inv_scales[gi]
                    col = 0
                    intercept = None
                    if self.standard_intercepts[gi]:
                        intercept = coef[:, col].copy(); col += 1
                    slopes = None
                    S = self.standard_slopes[gi]
                    if S is not None and S.shape[1]:
                        slopes = coef[:, col:col+S.shape[1]].copy()
                        proj = self._std_absorber._projectors[gi]
                        if intercept is not None and proj.slope_means is not None:
                            intercept -= np.einsum("gj,gj->g", proj.slope_means, slopes)
                    contrib = np.zeros(self.nobs, dtype=np.float64)
                    if intercept is not None:
                        contrib += intercept[codes]
                    if slopes is not None:
                        for j in range(slopes.shape[1]):
                            contrib += S[:, j] * slopes[codes, j]
                    reconstructed += contrib
                    terms.append((intercept, slopes, contrib))

            tind = theta[self._std_width:].reshape(self.n_individuals, self._ind_q) * self._ind_inv_scales
            c = 0
            intercept = None
            if self.individual_intercept:
                intercept = tind[:, c].copy(); c += 1
            slopes = None
            if self.n_individual_slopes:
                slopes = tind[:, c:c+self.n_individual_slopes].copy()
            contrib = self._individual_matvec(theta[self._std_width:])
            reconstructed += contrib
            terms.append((intercept, slopes, contrib))

        denom = max(float(np.linalg.norm(target)), np.finfo(float).eps)
        return terms, {
            "converged": ok,
            "iterations": itn,
            "residual_norm": normr,
            "reconstruction_error": float(np.linalg.norm(target-reconstructed)/denom),
        }


def group_individual_dof(
    standard_groups,
    *,
    standard_slopes=None,
    standard_intercepts=None,
    clusters=None,
    individual_levels: int,
    individual_intercept: bool,
    individual_nslopes: int,
    method="pairwise",
):
    """Current-reghdfe-compatible conservative DoF for individual FEs.

    Current reghdfe explicitly leaves ``dof_update_individual_fe`` as a TODO,
    so no redundancy correction is applied to individual FE coefficients. We
    mirror that behavior and reuse the ordinary reghdfe-style DoF rules for all
    group-level fixed effects.
    """
    if standard_groups:
        base = absorbed_dof(
            list(standard_groups), slopes=standard_slopes,
            intercepts=standard_intercepts, clusters=clusters,
            method=method, adjust_nested=bool(clusters), adjust_continuous=True,
            groups_are_dense=True,
        )
    else:
        base = DofInfo(0, 0, 0, 0, (), (), (), (), method)
    q = int(bool(individual_intercept)) + int(individual_nslopes)
    k_ind = tuple([int(individual_levels)] * q)
    m_ind = tuple([0] * q)
    return DofInfo(
        df_absorbed=int(base.df_absorbed + individual_levels*q),
        initial=int(base.initial + individual_levels*q),
        redundant=int(base.redundant),
        nested=int(base.nested),
        k_by_component=tuple(base.k_by_component) + k_ind,
        m_by_component=tuple(base.m_by_component) + m_ind,
        exact_by_component=tuple(base.exact_by_component) + tuple([False] * q),
        nested_by_component=tuple(base.nested_by_component) + tuple([False] * q),
        method=f"{base.method}+individual_conservative",
    )
