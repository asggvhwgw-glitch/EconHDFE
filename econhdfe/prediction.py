from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SampleExclusionState:
    stage: str
    reason: str
    dropped: int
    remaining: int


@dataclass(frozen=True, slots=True)
class EstimationSampleSnapshot:
    """Compact immutable description of the realized estimation sample.

    A packed bit mask is stored only when rows were excluded. This preserves
    exact raw-to-estimation row alignment without retaining the original data or
    an N-byte boolean array on the fitted result.
    """

    nobs_raw: int
    nobs_final: int
    packed_mask: bytes | None = None
    history: tuple[SampleExclusionState, ...] = ()

    @classmethod
    def from_state(cls, state) -> "EstimationSampleSnapshot":
        mask = np.asarray(state.mask, dtype=bool)
        packed = None if bool(np.all(mask)) else np.packbits(mask, bitorder="little").tobytes()
        history = tuple(
            SampleExclusionState(
                str(item.stage), str(item.reason), int(item.dropped), int(item.remaining)
            )
            for item in state.history
        )
        return cls(int(state.nobs_raw), int(state.nobs), packed, history)

    def mask(self) -> np.ndarray:
        if self.packed_mask is None:
            out = np.ones(self.nobs_raw, dtype=bool)
        else:
            raw = np.frombuffer(self.packed_mask, dtype=np.uint8)
            out = np.unpackbits(raw, bitorder="little", count=self.nobs_raw).astype(bool, copy=False)
        out.flags.writeable = False
        return out


@dataclass(frozen=True, slots=True)
class PredictionInput:
    """One source column needed to rebuild a design term.

    source is None for opaque array-like inputs. Those designs remain valid
    for in-sample bookkeeping but cannot be reconstructed from a new DataFrame
    without the caller supplying a transformed matrix explicitly.
    """

    name: str
    source: str | None


@dataclass(frozen=True, slots=True)
class CategoricalEncodingState:
    name: str
    source: str | None
    levels: tuple[Any, ...]
    selected: tuple[bool, ...]

    @property
    def base_levels(self) -> tuple[Any, ...]:
        return tuple(level for level, keep in zip(self.levels, self.selected, strict=False) if not keep)


@dataclass(frozen=True, slots=True)
class DesignTermState:
    name: str
    kind: str
    column_names: tuple[str, ...]
    active_mask: tuple[bool, ...]
    categorical: tuple[CategoricalEncodingState, ...] = ()
    continuous: tuple[PredictionInput, ...] = ()
    cells: tuple[tuple[Any, ...], ...] = ()

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, keep in zip(self.column_names, self.active_mask, strict=False) if keep
        )

    @property
    def reconstructable(self) -> bool:
        sources = tuple(item.source for item in self.categorical) + tuple(
            item.source for item in self.continuous
        )
        return all(source is not None for source in sources)


@dataclass(frozen=True, slots=True)
class DesignPredictionState:
    role: str
    requested_names: tuple[str, ...]
    materialized_names: tuple[str, ...]
    active_names: tuple[str, ...]
    active_indices: tuple[int, ...]
    terms: tuple[DesignTermState, ...] = ()

    @property
    def reconstructable(self) -> bool:
        if not self.active_names:
            return True
        active = set(self.active_names)
        relevant = tuple(
            term for term in self.terms
            if any(name in active for name in term.active_names)
        )
        return bool(relevant) and all(term.reconstructable for term in relevant)


@dataclass(frozen=True, slots=True)
class DroppedFixedEffectState:
    name: str
    spanned_by: str
    reason: str
    proof_type: str


@dataclass(frozen=True, slots=True)
class FixedEffectPredictionState:
    requested_names: tuple[str, ...]
    effective_names: tuple[str, ...]
    requested_levels: tuple[int, ...]
    effective_levels: tuple[int, ...]
    dropped: tuple[DroppedFixedEffectState, ...]
    intercepts: tuple[bool, ...]
    slope_counts: tuple[int, ...]
    coefficients_saved: bool
    normalization: str | None
    level_maps_available: bool = False

    @property
    def has_fixed_effects(self) -> bool:
        return bool(self.requested_names)

    @property
    def has_varying_slopes(self) -> bool:
        return any(count > 0 for count in self.slope_counts)


@dataclass(frozen=True, slots=True)
class PredictionState:
    """Read-only fitted-design contract used by future post-estimation APIs."""

    estimator: str
    coefficient_names: tuple[str, ...]
    coefficient_roles: tuple[str, ...]
    designs: tuple[DesignPredictionState, ...]
    sample: EstimationSampleSnapshot
    fixed_effects: FixedEffectPredictionState
    link: str = "identity"
    response: str = "identity"

    def design(self, role: str) -> DesignPredictionState:
        key = str(role)
        for item in self.designs:
            if item.role == key:
                return item
        raise KeyError(key)

    @property
    def can_rebuild_linear_predictor(self) -> bool:
        return all(self.design(role).reconstructable for role in self.coefficient_roles)

    @property
    def can_include_fixed_effects(self) -> bool:
        return (
            not self.fixed_effects.has_fixed_effects
            or (
                self.fixed_effects.coefficients_saved
                and self.fixed_effects.level_maps_available
                and not self.fixed_effects.has_varying_slopes
            )
        )


def _source_name(token: str) -> str | None:
    prefix = "column:"
    return token[len(prefix):] if str(token).startswith(prefix) else None


def design_state(design, *, role: str, active_indices) -> DesignPredictionState:
    active = tuple(int(i) for i in active_indices)
    names = tuple(str(x) for x in design.names)
    if any(i < 0 or i >= len(names) for i in active):
        raise ValueError("prediction active-column indices are out of range")
    active_names = tuple(names[i] for i in active)
    return DesignPredictionState(
        role=str(role),
        requested_names=tuple(str(x) for x in design.requested_names),
        materialized_names=names,
        active_names=active_names,
        active_indices=active,
        terms=tuple(getattr(design, "prediction_terms", ())),
    )


def fixed_effect_state(
    plan,
    *,
    effective_names,
    requested_groups,
    effective_groups,
    intercepts,
    slopes,
    recovered_effects=None,
) -> FixedEffectPredictionState:
    requested_names = tuple(str(x) for x in getattr(plan, "requested_names", effective_names))
    effective_names = tuple(str(x) for x in effective_names)
    requested_levels = tuple(
        int(np.asarray(group).max()) + 1 if len(group) else 0
        for group in requested_groups
    )
    effective_levels = tuple(
        int(np.asarray(group).max()) + 1 if len(group) else 0
        for group in effective_groups
    )
    dropped = tuple(
        DroppedFixedEffectState(
            str(item.name), str(item.spanned_by), str(item.reason), str(item.proof_type)
        )
        for item in getattr(plan, "dropped", ())
    )
    intercepts = tuple(bool(x) for x in intercepts)
    slope_counts = tuple(
        0 if value is None else int(np.asarray(value).shape[1] if np.asarray(value).ndim > 1 else 1)
        for value in slopes
    )
    normalization = None
    if recovered_effects is not None:
        normalization = str(getattr(recovered_effects, "normalization", "minimum_norm_lsmr"))
    return FixedEffectPredictionState(
        requested_names=requested_names,
        effective_names=effective_names,
        requested_levels=requested_levels,
        effective_levels=effective_levels,
        dropped=dropped,
        intercepts=intercepts,
        slope_counts=slope_counts,
        coefficients_saved=recovered_effects is not None,
        normalization=normalization,
        level_maps_available=False,
    )


def linear_prediction_state(
    *,
    estimator: str,
    coefficient_names,
    coefficient_roles,
    designs,
    sample_state,
    fe_plan,
    fe_names,
    requested_fe_groups,
    effective_fe_groups,
    fe_intercepts,
    fe_slopes,
    recovered_effects=None,
) -> PredictionState:
    return PredictionState(
        estimator=str(estimator),
        coefficient_names=tuple(str(x) for x in coefficient_names),
        coefficient_roles=tuple(str(x) for x in coefficient_roles),
        designs=tuple(designs),
        sample=EstimationSampleSnapshot.from_state(sample_state),
        fixed_effects=fixed_effect_state(
            fe_plan,
            effective_names=fe_names,
            requested_groups=requested_fe_groups,
            effective_groups=effective_fe_groups,
            intercepts=fe_intercepts,
            slopes=fe_slopes,
            recovered_effects=recovered_effects,
        ),
    )
