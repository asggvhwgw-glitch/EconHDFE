"""Regressor-term specification vocabulary.

These immutable specs carry no compilation logic, so input checks and source
column projection can name them without importing :mod:`econhdfe.design`,
which expands them into design matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Factor:
    """Categorical regressor expanded to observed-level indicators."""

    variable: Any
    base: Any | None = None
    drop_base: bool = True
    name: str | None = None


@dataclass(frozen=True, slots=True)
class RegressorInteraction:
    """Multiplicative interaction among continuous/factor regressor terms."""

    parts: tuple[Any, ...]
    name: str | None = None

    def __init__(self, *parts: Any, name: str | None = None):
        if len(parts) < 2:
            raise ValueError("a regressor interaction requires at least two parts")
        object.__setattr__(self, "parts", tuple(parts))
        object.__setattr__(self, "name", name)


def factor(variable: Any, *, base: Any | None = None, drop_base: bool = True,
           name: str | None = None) -> Factor:
    return Factor(variable, base=base, drop_base=drop_base, name=name)


def reg_interaction(*parts: Any, name: str | None = None) -> RegressorInteraction:
    return RegressorInteraction(*parts, name=name)
