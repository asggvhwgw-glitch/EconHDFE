"""Fit-time categorical prediction certificates; no additional coefficient solve."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..hdfe.rank import _ExactRankResourceError
from ..prediction import (
    CategoricalFixedEffectState, FixedEffectLevelState, FixedEffectNestingState,
)
from .topology import identification_structure


def saved_categorical_state(state, *, data, mask, inference, plan, groups,
                            recovered, beta, identifier_levels):
    """Build level-sized state only after an explicit save_fe=True request.

    Rank certification uses the resource-guarded native backend. Resource failure
    disables new-row FE prediction, not the already computed regression result.
    Opaque inputs and varying slopes remain explicitly unsupported.
    """
    def unavailable(reason):
        return replace(state, level_maps_available=False, unavailable_reason=reason)

    if recovered is None or not recovered.converged:
        return unavailable("fixed_effect_recovery_unavailable_or_nonconverged")
    metadata = inference.metadata
    if not metadata or any(not m.pure_intercept or not m.components for m in metadata):
        return unavailable("requires_named_categorical_intercepts")
    try:
        topology = identification_structure(groups, rank_backend="native")
    except _ExactRankResourceError:
        return unavailable("identification_resource_budget")

    # Decode only representative rows instead of allocating N original labels.
    rows = None if bool(np.all(mask)) else np.flatnonzero(mask)
    terms, representatives = [], []
    for group, meta in zip(inference.groups, metadata):
        _, first = np.unique(group, return_index=True)
        representatives.append(first)
        raw_rows = first if rows is None else rows[first]
        labels = []
        for source in meta.components:
            values = np.asarray(data[source])[raw_rows]
            if source in identifier_levels:
                values = np.asarray(identifier_levels[source])[values.astype(np.intp)]
            labels.append(values)
        terms.append(FixedEffectLevelState(meta.name, meta.components, tuple(labels)))
    nesting = tuple(
        FixedEffectNestingState(
            d.spanned_by_index, d.index,
            inference.groups[d.index][representatives[d.spanned_by_index]],
        ) for d in plan.dropped
    )
    payload = CategoricalFixedEffectState(
        tuple(terms), tuple(plan.effective_indices),
        tuple(term.intercept for term in recovered.terms),
        topology.component_by_term, topology.identified_components, nesting, beta,
    )
    return replace(state, level_maps_available=True, categorical=payload,
                   unavailable_reason=None)
