"""Shared parsing of user ``absorb=`` specifications and their source columns.

Estimator front ends accept different subsets of the absorb syntax (sessions
and reusable PPML models only take DataFrame column terms, while direct
OLS/IV calls also take arrays, :class:`FixedEffect` and the historical dict
form). Each caller keeps its own acceptance range and error messages; the
term-level parsing and source-column collection live here.
"""
from __future__ import annotations

import numpy as np

from ..factorvars import FactorVariableExpression
from .factorvars import compile_absorb_factor_variable
from .specs import FixedEffect, Interaction


def is_column_interaction(term) -> bool:
    """``("a", "b")`` shorthand for the categorical interaction FE ``a#b``."""
    return isinstance(term, (tuple, list)) and len(term) >= 2 and all(
        isinstance(v, str) for v in term
    )


def absorb_terms(absorb) -> list:
    """Split ``absorb`` into FE terms; a single term may be passed unwrapped."""
    if isinstance(absorb, (str, FixedEffect, Interaction, FactorVariableExpression)):
        return [absorb]
    if isinstance(absorb, np.ndarray) and absorb.ndim == 1:
        return [absorb]
    return list(absorb)


def expand_factor_variables(terms) -> list:
    """Replace ``fv()`` absorb expressions by their compiled FE terms, in place order."""
    out = []
    for term in terms:
        if isinstance(term, FactorVariableExpression):
            out.extend(compile_absorb_factor_variable(term))
        else:
            out.append(term)
    return out


def fixed_effect_spec(term) -> FixedEffect:
    """Normalize one (non-``fv()``) absorb term to a :class:`FixedEffect`."""
    if isinstance(term, FixedEffect):
        return term
    if isinstance(term, Interaction):
        return FixedEffect(term)
    if is_column_interaction(term):
        return FixedEffect(Interaction(*term))
    if isinstance(term, dict):
        group = term.get("group")
        if is_column_interaction(group):
            group = Interaction(*group)
        return FixedEffect(
            group, term.get("slopes", ()),
            term.get("intercept", True), term.get("name")
        )
    return FixedEffect(term)


def term_columns(term) -> tuple:
    """Columns of one DataFrame FE term: a column name or a column tuple."""
    return (term,) if isinstance(term, str) else tuple(term)


def absorb_columns(terms) -> tuple[str, ...]:
    """Unique column names, in first-use order, referenced by DataFrame FE terms."""
    cols: list[str] = []
    for term in terms:
        for c in term_columns(term):
            if isinstance(c, str) and c not in cols:
                cols.append(c)
    return tuple(cols)
