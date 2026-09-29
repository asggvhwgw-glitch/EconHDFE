from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


def _readonly(values, dtype=None) -> np.ndarray:
    """Detach level-sized state; numeric/string buffers cannot be made writable."""
    a = np.asarray(values, dtype=dtype)
    if a.dtype.kind == "O" and all(isinstance(x, str) for x in a.flat):
        a = a.astype(str)
    if not a.dtype.hasobject:
        return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
    a = a.copy()
    a.flags.writeable = False
    return a


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
    def from_mask(cls, mask, *, history=()) -> "EstimationSampleSnapshot":
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("estimation sample mask must be one-dimensional")
        packed = None if bool(np.all(mask)) else np.packbits(mask, bitorder="little").tobytes()
        return cls(int(mask.size), int(np.count_nonzero(mask)), packed, tuple(history))

    @classmethod
    def from_state(cls, state) -> "EstimationSampleSnapshot":
        history = tuple(
            SampleExclusionState(
                str(item.stage), str(item.reason), int(item.dropped), int(item.remaining)
            )
            for item in state.history
        )
        return cls.from_mask(state.mask, history=history)

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


@dataclass(frozen=True, slots=True, eq=False)
class CategoricalEncodingState:
    name: str
    source: str | None
    levels: np.ndarray
    selected: np.ndarray

    def __post_init__(self):
        levels = _readonly(self.levels)
        selected = _readonly(self.selected, bool)
        if levels.ndim != 1 or selected.shape != levels.shape:
            raise ValueError("categorical prediction levels/selection must be aligned vectors")
        levels.flags.writeable = False
        selected.flags.writeable = False
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "selected", selected)

    @property
    def base_levels(self) -> tuple[Any, ...]:
        return tuple(
            value.item() if isinstance(value, np.generic) else value
            for value in self.levels[~self.selected]
        )


@dataclass(frozen=True, slots=True, eq=False)
class DesignTermState:
    name: str
    kind: str
    column_names: tuple[str, ...]
    active_mask: tuple[bool, ...]
    categorical: tuple[CategoricalEncodingState, ...] = ()
    continuous: tuple[PredictionInput, ...] = ()
    cell_codes: np.ndarray | None = None

    def __post_init__(self):
        if len(self.column_names) != len(self.active_mask):
            raise ValueError("prediction column names and active mask must align")
        if self.cell_codes is None:
            if self.categorical:
                raise ValueError("categorical prediction terms require observed cells")
            return
        cells = _readonly(self.cell_codes, np.int32)
        if cells.shape != (len(self.column_names), len(self.categorical)):
            raise ValueError("prediction cell codes must align with categorical components")
        for j, encoding in enumerate(self.categorical):
            if np.any(cells[:, j] < 0) or np.any(cells[:, j] >= len(encoding.levels)):
                raise ValueError("prediction cell code is outside its fitted levels")
        object.__setattr__(self, "cell_codes", cells)

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

    @property
    def cells(self) -> tuple[tuple[Any, ...], ...]:
        if self.cell_codes is None or not self.categorical:
            return ()
        return tuple(
            tuple(
                self.categorical[j].levels[int(code)].item()
                if isinstance(self.categorical[j].levels[int(code)], np.generic)
                else self.categorical[j].levels[int(code)]
                for j, code in enumerate(row)
            )
            for row in self.cell_codes
        )


@dataclass(frozen=True, slots=True, eq=False)
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
        covered = {name for term in relevant for name in term.active_names}
        return active.issubset(covered) and all(term.reconstructable for term in relevant)


@dataclass(frozen=True, slots=True)
class DroppedFixedEffectState:
    name: str
    spanned_by: str
    reason: str
    proof_type: str


@dataclass(frozen=True, slots=True, eq=False)
class FixedEffectLevelState:
    """Original labels in fitted dense-code order, one entry per FE level."""
    name: str
    sources: tuple[str, ...]
    labels: tuple[np.ndarray, ...]

    def __post_init__(self):
        labels = tuple(_readonly(a) for a in self.labels)
        if not labels or len(labels) != len(self.sources):
            raise ValueError("fixed-effect labels must align with their source columns")
        if any(a.ndim != 1 or a.shape != labels[0].shape for a in labels):
            raise ValueError("fixed-effect label columns must be aligned vectors")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True, slots=True, eq=False)
class FixedEffectNestingState:
    fine: int
    coarse: int
    coarse_by_fine: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "coarse_by_fine", _readonly(self.coarse_by_fine, np.int32))


@dataclass(frozen=True, slots=True, eq=False)
class CategoricalFixedEffectState:
    """Saved FE coefficients plus certificates needed for new observation rows.

    Identification refers to the effective design. Requested partitions and
    nesting maps additionally prevent canonicalization from accepting new
    combinations that violate a fitted exact dependency.
    """
    terms: tuple[FixedEffectLevelState, ...]
    effective_indices: tuple[int, ...]
    coefficients: tuple[np.ndarray, ...]
    components: tuple[np.ndarray, ...]
    identified_components: tuple[int, ...]
    nesting: tuple[FixedEffectNestingState, ...]
    beta: np.ndarray

    def __post_init__(self):
        if not (len(self.effective_indices) == len(self.coefficients) == len(self.components)):
            raise ValueError("saved fixed-effect coefficients and components must align")
        coefficients = tuple(_readonly(a, float) for a in self.coefficients)
        components = tuple(_readonly(a, np.int32) for a in self.components)
        for i, coef, comp in zip(self.effective_indices, coefficients, components):
            if coef.shape != self.terms[i].labels[0].shape or comp.shape != coef.shape:
                raise ValueError("saved fixed effects must align with fitted levels")
            if not np.isfinite(coef).all():
                raise ValueError("saved fixed-effect coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "beta", _readonly(self.beta, float))


@dataclass(frozen=True, slots=True, eq=False)
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
    categorical: CategoricalFixedEffectState | None = None
    unavailable_reason: str | None = None

    @property
    def has_fixed_effects(self) -> bool:
        return bool(self.requested_names)

    @property
    def has_varying_slopes(self) -> bool:
        return any(count > 0 for count in self.slope_counts)


@dataclass(frozen=True, slots=True, eq=False)
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
                and self.fixed_effects.categorical is not None
                and not self.fixed_effects.has_varying_slopes
            )
        )


def design_state(design, *, role: str, active_indices, identifier_levels=None) -> DesignPredictionState:
    active = tuple(int(i) for i in active_indices)
    names = tuple(str(x) for x in design.names)
    if any(i < 0 or i >= len(names) for i in active):
        raise ValueError("prediction active-column indices are out of range")
    active_names = tuple(names[i] for i in active)
    terms = tuple(getattr(design, "prediction_terms", ()))
    if identifier_levels:
        terms = tuple(replace(term, categorical=tuple(
            replace(enc, levels=np.asarray(identifier_levels[enc.source])[enc.levels.astype(np.intp)])
            if enc.source in identifier_levels else enc for enc in term.categorical
        )) for term in terms)
    return DesignPredictionState(
        role=str(role),
        requested_names=tuple(str(x) for x in design.requested_names),
        materialized_names=names,
        active_names=active_names,
        active_indices=active,
        terms=terms,
    )


def named_design_state(names, *, role: str, active_indices) -> DesignPredictionState:
    names = tuple(str(x) for x in names)
    terms = tuple(
        DesignTermState(
            name=name,
            kind="continuous",
            column_names=(name,),
            active_mask=(True,),
            continuous=(PredictionInput(name, name),),
        )
        for name in names
    )
    active = tuple(int(i) for i in active_indices)
    if any(i < 0 or i >= len(names) for i in active):
        raise ValueError("prediction active-column indices are out of range")
    return DesignPredictionState(
        role=str(role),
        requested_names=names,
        materialized_names=names,
        active_names=tuple(names[i] for i in active),
        active_indices=active,
        terms=terms,
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
    fe_plan,
    fe_names,
    requested_fe_groups,
    effective_fe_groups,
    fe_intercepts,
    fe_slopes,
    recovered_effects=None,
    sample_state=None,
    sample_mask=None,
    sample_history=(),
) -> PredictionState:
    if (sample_state is None) == (sample_mask is None):
        raise ValueError("provide exactly one of sample_state or sample_mask")
    sample = (
        EstimationSampleSnapshot.from_state(sample_state)
        if sample_state is not None
        else EstimationSampleSnapshot.from_mask(sample_mask, history=sample_history)
    )
    return PredictionState(
        estimator=str(estimator),
        coefficient_names=tuple(str(x) for x in coefficient_names),
        coefficient_roles=tuple(str(x) for x in coefficient_roles),
        designs=tuple(designs),
        sample=sample,
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
