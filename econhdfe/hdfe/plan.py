from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .absorber import HDFEAbsorber, iterative_singleton_mask
from .absorb_spec import absorb_columns, term_columns
from .two_way import TwoWayFEAbsorber
from ..compute.encoding import factorize_1d, factorize_interaction
from .dof import absorbed_dof
from .structure import canonicalize_fixed_effects, FEPartitionMeta


def _fingerprint(groups, metadata=()) -> str:
    h = hashlib.blake2b(digest_size=16)
    for g in groups:
        a = np.ascontiguousarray(g)
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.view(np.uint8))
    for m in metadata or ():
        h.update(str(getattr(m, "name", "")).encode())
        h.update(repr(tuple(getattr(m, "components", ()))).encode())
    return h.hexdigest()


@dataclass(slots=True)
class FEPlan:
    """Compiled categorical FE topology shared by iterative HDFE estimators.

    ``metadata`` is structural only: it allows canonicalization to certify
    obvious interaction containment from the specification before scanning
    observation-level mappings.
    """

    groups: tuple[np.ndarray, ...]
    levels: tuple[int, ...]
    metadata: tuple[FEPartitionMeta, ...] = ()
    fingerprint: str = ""

    @classmethod
    def from_arrays(cls, groups) -> "FEPlan":
        dense, levels = [], []
        for g in groups or ():
            c, L = factorize_1d(g)
            dense.append(c)
            levels.append(int(L))
        meta = tuple(FEPartitionMeta(f"fe{i}", (), True) for i in range(len(dense)))
        groups_t = tuple(dense)
        return cls(groups_t, tuple(levels), meta, _fingerprint(groups_t, meta))

    @classmethod
    def from_dataframe(cls, data, absorb) -> "FEPlan":
        dense, levels, meta = [], [], []
        for term in absorb or ():
            cols = term_columns(term)
            arrays = [data[c].to_numpy() for c in cols]
            c, L = factorize_1d(arrays[0]) if len(arrays) == 1 else factorize_interaction(arrays)
            dense.append(c); levels.append(int(L))
            meta.append(FEPartitionMeta("#".join(map(str, cols)), tuple(map(str, cols)), True))
        groups_t, meta_t = tuple(dense), tuple(meta)
        return cls(groups_t, tuple(levels), meta_t, _fingerprint(groups_t, meta_t))


    @staticmethod
    def source_fingerprint(data, absorb) -> str:
        """Hash DataFrame FE source columns for reusable-model cache invalidation."""
        import pandas as pd
        cols = list(absorb_columns(absorb or ()))
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray([len(data)], dtype=np.int64).tobytes())
        if cols:
            vals = pd.util.hash_pandas_object(data[cols], index=True).to_numpy(dtype=np.uint64, copy=False)
            h.update(np.ascontiguousarray(vals).view(np.uint8))
        return h.hexdigest()

    @property
    def nobs(self) -> int:
        return len(self.groups[0]) if self.groups else 0

    def take(self, rows) -> "FEPlan":
        """Restrict observations by integer positions and refactorize FE codes."""
        idx = np.asarray(rows, dtype=np.int64)
        if idx.ndim != 1 or np.any(idx < 0) or (idx.size and np.any(idx >= self.nobs)):
            raise ValueError("FEPlan row indices are out of bounds")
        out = FEPlan.from_arrays([g[idx] for g in self.groups])
        return FEPlan(out.groups, out.levels, self.metadata, _fingerprint(out.groups, self.metadata))

    def subset(self, mask) -> "FEPlan":
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != self.nobs:
            raise ValueError("FEPlan subset mask must have one value per observation")
        return self.take(np.flatnonzero(mask))

    def singleton_mask(self, *, frequency_weights=None, positive=None) -> np.ndarray:
        """Iteratively flag observations in singleton FE levels.

        Degrees are row counts, or frequency-weight mass when
        ``frequency_weights`` is given (reghdfe's fweight rule).  A boolean
        ``positive`` outcome mask also flags levels without a positive
        outcome in the same fixed point (ppmlhdfe's ``separation(fe)`` stage).
        """
        keep, _ = iterative_singleton_mask(
            self.groups, frequency_weights, frequency_weights=frequency_weights is not None,
            positive=positive,
        )
        return ~keep

    def canonicalized(self) -> tuple["FEPlan", dict]:
        """Drop exactly redundant pure-intercept FE partitions."""
        if len(self.groups) < 2:
            return self, {"requested": len(self.groups), "effective": len(self.groups), "changed": False}
        names = [m.name for m in self.metadata] if self.metadata else [f"fe{i}" for i in range(len(self.groups))]
        groups, _, _, _, plan = canonicalize_fixed_effects(
            self.groups, [None] * len(self.groups), [True] * len(self.groups), names,
            metadata=self.metadata or None,
        )
        meta = plan.as_dict()
        meta["changed"] = bool(plan.changed)
        if not plan.changed:
            return self, meta
        out0 = FEPlan.from_arrays(groups)
        kept_meta = tuple((self.metadata or tuple(FEPartitionMeta(n, (), True) for n in names))[i] for i in plan.effective_indices)
        out = FEPlan(out0.groups, out0.levels, kept_meta, _fingerprint(out0.groups, kept_meta))
        return out, meta


    def for_engine(self, engine="replica"):
        """Return the numerical FE plan and structural metadata for an engine."""
        if engine == "optimized" and self.groups:
            return self.canonicalized()
        return self, {"requested": len(self.groups), "effective": len(self.groups), "changed": False}

    def projector(self, *, engine="replica", method="map"):
        from .weighted_projection import WeightedFEProjector
        return WeightedFEProjector(self.groups, engine=engine, method=method)

    def absorber(self, weights, *, tol, engine="replica", method="map"):
        if not self.groups:
            return None
        if engine == "within":
            return self.projector(engine=engine, method=method).prepare(weights, tol=tol)
        if method == "within":
            from .within_backend import WithinFEAbsorber
            return WithinFEAbsorber(self.groups, weights=weights, tol=tol)
        if engine == "optimized" and len(self.groups) == 2 and method != "lsmr":
            return TwoWayFEAbsorber(self.groups, weights=weights, tol=tol)
        return HDFEAbsorber(
            self.groups,
            weights=weights,
            tol=tol,
            method=method,
            transform="symmetric",
            acceleration="auto" if method == "map" else "none",
            projection_backend="auto",
            absorb_threads="auto",
        )

    def residualize(self, A, weights, *, tol, engine="replica"):
        if engine == "within" and self.groups:
            # The shared wrapper enforces convergence even when callers discard info.
            return self.projector(engine=engine).residualize(A, weights, tol=tol, return_info=True)
        absorber = self.absorber(weights, tol=tol, engine=engine)
        if absorber is None:
            return np.asarray(A, dtype=np.float64).copy(), None
        out, info = absorber.residualize(A, return_info=True)
        return out, info

    def dof_info(self, clusters=None, *, method="pairwise"):
        if not self.groups:
            return absorbed_dof([], method=method, groups_are_dense=True)
        return absorbed_dof(
            self.groups, clusters=clusters, method=method, groups_are_dense=True
        )

    def df_absorbed(self, clusters=None, *, method="pairwise") -> int:
        return int(self.dof_info(clusters=clusters, method=method).df_absorbed)


def fe_separated(y, plan: FEPlan) -> np.ndarray:
    """Exact FE-only separation: levels containing no positive outcome."""
    y = np.asarray(y, dtype=np.float64)
    sep = np.zeros(len(y), dtype=bool)
    positive = y > 0
    for g, L in zip(plan.groups, plan.levels, strict=False):
        has_positive = np.bincount(g, weights=positive.astype(np.int8), minlength=L) > 0
        sep |= ~has_positive[g]
    return sep & ~positive
