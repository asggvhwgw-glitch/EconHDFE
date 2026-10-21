from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from .compute.block_design import BlockDesign


class OmittedVariableWarning(UserWarning):
    """A requested regressor/instrument was omitted for exact or near collinearity."""


def format_omission_warning(items, *, max_items: int = 8) -> str:
    items = tuple(items)
    shown = items[:max_items]
    detail = ", ".join(f"{getattr(o, 'name', '?')} ({getattr(o, 'reason', '?')})" for o in shown)
    extra = len(items) - len(shown)
    if extra > 0:
        detail += f", ... (+{extra} more)"
    return f"{len(items)} requested column(s) omitted for collinearity/absorption: {detail}"


@dataclass(frozen=True, slots=True)
class OmittedColumn:
    index: int
    name: str
    role: str
    reason: str
    relative_norm: float
    dependent_on: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "index": int(self.index), "name": self.name, "role": self.role,
            "reason": self.reason, "relative_norm": float(self.relative_norm),
            "dependent_on": self.dependent_on,
        }


@dataclass(frozen=True, slots=True)
class CollinearityPlan:
    requested_names: tuple[str, ...]
    active_names: tuple[str, ...]
    active_indices: tuple[int, ...]
    omitted: tuple[OmittedColumn, ...]
    tolerance: float
    rank: int

    def as_dict(self) -> dict:
        return {
            "requested": self.requested_names,
            "active": self.active_names,
            "active_indices": self.active_indices,
            "omitted": tuple(o.as_dict() for o in self.omitted),
            "tolerance": float(self.tolerance),
            "rank": int(self.rank),
        }


def _weighted_gram(X, weights=None) -> np.ndarray:
    if X.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if isinstance(X, BlockDesign):
        return X.gram(weights=weights)
    if weights is None:
        return X.T @ X
    w = np.asarray(weights, dtype=np.float64)
    return X.T @ (X * w[:, None])


def _cross_gram(A, B, weights=None) -> np.ndarray:
    if isinstance(A, BlockDesign):
        return A.cross_gram(B, weights=weights)
    if isinstance(B, BlockDesign):
        return B.cross_gram(A, weights=weights).T
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if weights is None:
        return A.T @ B
    w = np.asarray(weights, dtype=np.float64)
    return A.T @ (B * w[:, None])


def _joint_gram(P, X, weights=None) -> tuple[np.ndarray, int]:
    p = 0 if P is None else int(P.shape[1])
    k = int(X.shape[1])
    if p == 0:
        return _weighted_gram(X, weights), 0
    Gpp = _weighted_gram(P, weights)
    Gpx = _cross_gram(P, X, weights)
    Gxx = _weighted_gram(X, weights)
    G = np.empty((p + k, p + k), dtype=np.float64)
    G[:p, :p] = Gpp
    G[:p, p:] = Gpx
    G[p:, :p] = Gpx.T
    G[p:, p:] = Gxx
    return G, p


def _gram_coordinates(G: np.ndarray) -> np.ndarray:
    """Return B with B.T @ B ~= G using a symmetric eigensystem.

    Working in K-dimensional Gram coordinates avoids repeatedly projecting
    million-row columns and is substantially more stable than subtracting
    Schur complements formed with a pseudo inverse near exact collinearity.
    """
    if G.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    scale_col = np.sqrt(np.maximum(np.diag(G), 0.0))
    scale_col[scale_col == 0] = 1.0
    G = (G / scale_col[:, None]) / scale_col[None, :]
    Gs = (G + G.T) / 2
    vals, vecs = np.linalg.eigh(Gs)
    scale = max(float(np.max(np.abs(vals))), 1.0)
    floor = np.finfo(float).eps * max(G.shape) * scale
    keep = vals > floor
    if not np.any(keep):
        return np.zeros((0, G.shape[0]), dtype=np.float64)
    return np.sqrt(vals[keep])[:, None] * vecs[:, keep].T


def _orthogonal_residual(v: np.ndarray, Q: list[np.ndarray]) -> np.ndarray:
    r = np.asarray(v, dtype=np.float64).copy()
    if not Q:
        return r
    # Two-pass modified Gram-Schmidt is cheap in K-space and reduces false
    # non-collinearity from loss of orthogonality in nearly dependent designs.
    for _ in range(2):
        for q in Q:
            r -= q * float(q @ r)
    return r


def resolve_collinearity(
    X_within,
    *,
    names,
    role: str = "regressor",
    original: np.ndarray | None = None,
    weights=None,
    tolerance: float = 1e-10,
    protected_basis: np.ndarray | None = None,
    protected_names=(),
    _gram=None,
) -> CollinearityPlan:
    """Order-preserving rank resolution using small weighted Gram matrices.

    Columns are first classified as absorbed/zero when their within norm is
    negligible relative to their pre-absorption norm. Remaining columns are
    scanned in user order in a numerically stable Gram-coordinate basis.
    Optional protected columns (e.g. included exogenous regressors in IV) are
    never displaced by candidate columns.
    """
    if isinstance(X_within, BlockDesign):
        X = X_within
    else:
        X = np.asarray(X_within, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
    k = X.shape[1]
    names = tuple(str(n) for n in names)
    if len(names) != k:
        raise ValueError("column-name count must match design columns")
    if k == 0:
        return CollinearityPlan(names, (), (), (), float(tolerance), 0)

    if original is None:
        orig = X
    elif isinstance(original, BlockDesign):
        orig = original
    else:
        orig = np.asarray(original, dtype=np.float64)
    if orig.shape != X.shape:
        raise ValueError("original and within design shapes must match")

    if protected_basis is None:
        P = None
    elif isinstance(protected_basis, BlockDesign):
        P = protected_basis
    else:
        P = np.asarray(protected_basis, dtype=np.float64)
        if P.ndim == 1:
            P = P[:, None]
    if P is not None and int(P.shape[0]) != int(X.shape[0]):
        raise ValueError("protected basis row count must match design")
    # Recursive original-design checks can reuse their already formed Gram.
    G, p = _joint_gram(P, X, weights) if _gram is None else (_gram, 0)
    B = _gram_coordinates(G)
    if B.shape[0] < G.shape[0] and not isinstance(X, BlockDesign) and not isinstance(P, BlockDesign):
        # Normal equations square the condition number. A small Gram eigenvalue
        # is a reason to inspect the original design, not proof of collinearity.
        A = X if P is None else np.column_stack((P, X))
        scale = np.max(np.abs(A), axis=0) if len(A) else np.ones(A.shape[1])
        scale[scale == 0] = 1.0
        B = np.empty((0, A.shape[1]))
        chunk = max(1, min(32768, 2_000_000 // max(1, A.shape[1])))
        for lo in range(0, len(A), chunk):
            hi = min(len(A), lo + chunk)
            block = A[lo:hi] / scale
            if weights is not None:
                block = block * np.sqrt(np.asarray(weights)[lo:hi, None])
            B = la.qr(np.vstack((B, block)), mode="r", check_finite=False)[0][:A.shape[1]]
        norms = np.linalg.norm(B, axis=0)
        B /= np.where(norms > 0, norms, 1.0)
    original_gram = G[p:, p:] if original is None else _weighted_gram(orig, weights)
    orig_diag = np.diag(original_gram)
    within_diag = np.diag(G)[p:]

    tol = float(tolerance)
    eps = np.finfo(float).eps
    Q: list[np.ndarray] = []
    # Protected basis is itself expected to have been rank-resolved already,
    # but orthogonalizing defensively makes this function safe as a standalone
    # building block.
    for j in range(p):
        r = _orthogonal_residual(B[:, j], Q)
        nr = float(np.linalg.norm(r))
        if nr > np.sqrt(eps) * max(float(np.linalg.norm(B[:, j])), 1.0):
            Q.append(r / nr)

    # A linear relation already exact before FE projection remains exact
    # afterward. Iterative absorption error must not manufacture an extra
    # regressor direction (e.g. x1+x2 projected in a separate RHS column).
    original_omissions = {}
    if original is not None:
        original_plan = resolve_collinearity(
            orig, names=names, role=role, weights=weights, tolerance=tol, _gram=original_gram,
        )
        original_omissions = {
            o.index: o for o in original_plan.omitted
            if o.reason != "absorbed_or_zero"
        }
    active = []
    omitted = []
    protected_names = tuple(map(str, protected_names))
    for j in range(k):
        base = max(float(orig_diag[j]), np.finfo(float).tiny)
        rel = float(np.sqrt(max(within_diag[j], 0.0) / base))
        if within_diag[j] <= (tol * tol) * base:
            omitted.append(OmittedColumn(j, names[j], role, "absorbed_or_zero", rel, ()))
            continue

        if j in original_omissions:
            omitted.append(original_omissions[j])
            continue
        v = B[:, p+j]
        nv = float(np.linalg.norm(v))
        r = _orthogonal_residual(v, Q)
        nr = float(np.linalg.norm(r))
        cond_rel = nr / max(nv, np.sqrt(eps))
        if cond_rel <= tol:
            selected_names = protected_names + tuple(names[i] for i in active)
            selected_cols = list(range(p)) + [p+i for i in active]
            deps = selected_names
            reason = "linear_combination"
            if selected_cols:
                basis = B[:, selected_cols]
                coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
                cscale = max(float(np.max(np.abs(coef))), 1.0) if coef.size else 1.0
                hit = np.flatnonzero(np.abs(coef) > max(tol, 1e-12) * cscale)
                if hit.size:
                    deps = tuple(selected_names[int(h)] for h in hit)
                if hit.size == 1:
                    reason = "duplicate_or_scaled"
            omitted.append(OmittedColumn(j, names[j], role, reason, cond_rel, deps))
            continue
        active.append(j)
        Q.append(r / nr)

    active_t = tuple(active)
    return CollinearityPlan(
        requested_names=names,
        active_names=tuple(names[i] for i in active_t),
        active_indices=active_t,
        omitted=tuple(omitted),
        tolerance=tol,
        rank=len(active_t),
    )
