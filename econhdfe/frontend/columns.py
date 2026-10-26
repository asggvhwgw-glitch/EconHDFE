from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..design_specs import Factor, RegressorInteraction
from ..factorvars import FactorVariableExpression
from ..hdfe.specs import FixedEffect, Interaction


@dataclass(frozen=True, slots=True)
class ColumnUse:
    name: str
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DataRequirements:
    uses: tuple[ColumnUse, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(u.name for u in self.uses)

    @property
    def identifier_only_columns(self) -> tuple[str, ...]:
        safe = {"fixed_effect", "cluster", "group", "individual"}
        return tuple(u.name for u in self.uses if set(u.roles).issubset(safe))

    def roles_for(self, name: str) -> tuple[str, ...]:
        for use in self.uses:
            if use.name == name:
                return use.roles
        return ()


_TOP_ROLE = {
    "y": "outcome", "outcome": "outcome",
    "x": "regressor", "regressors": "regressor",
    "exog": "exogenous", "endog": "endogenous", "instruments": "instrument",
    "absorb": "fixed_effect", "weights": "weight", "weight": "weight",
    "cluster": "cluster", "clusters": "cluster", "offset": "offset", "exposure": "exposure",
    "time": "time", "panel": "panel", "group": "group", "individual": "individual",
}


def _role_name(key: str) -> str:
    return _TOP_ROLE.get(str(key), str(key))


def compile_data_requirements(**role_specs: Any) -> DataRequirements:
    order: list[str] = []
    roles: dict[str, list[str]] = {}

    def add(name: str, role: str):
        if name not in roles:
            order.append(name)
            roles[name] = []
        if role not in roles[name]:
            roles[name].append(role)

    def visit(value, role: str):
        if value is None:
            return
        if isinstance(value, str):
            add(value, role); return
        if isinstance(value, FactorVariableExpression):
            for term in value.terms:
                for atom in term.atoms:
                    add(atom.variable, "factor" if atom.kind == "factor" else role)
            return
        if isinstance(value, Factor):
            visit(value.variable, "factor"); return
        if isinstance(value, RegressorInteraction):
            for part in value.parts:
                visit(part, role)
            return
        if isinstance(value, Interaction):
            for part in value.parts:
                visit(part, "fixed_effect" if role == "fixed_effect" else role)
            return
        if isinstance(value, FixedEffect):
            visit(value.group, "fixed_effect")
            for slope in value.slopes:
                visit(slope, "slope")
            return
        if isinstance(value, Mapping):
            # Historical absorb=dict(group=..., slopes=...) syntax.
            if "group" in value:
                visit(value.get("group"), "fixed_effect")
                visit(value.get("slopes"), "slope")
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item, role)
            return
        # ndarray / pandas Series / user arrays are already resident and need no
        # raw-source column projection.

    for key, spec in role_specs.items():
        visit(spec, _role_name(key))
    return DataRequirements(tuple(ColumnUse(name, tuple(roles[name])) for name in order))




def merge_data_requirements(*requirements: DataRequirements) -> DataRequirements:
    """Stable union of column roles across multiple economic specifications."""
    order: list[str] = []
    roles: dict[str, list[str]] = {}
    for req in requirements:
        for use in req.uses:
            if use.name not in roles:
                order.append(use.name); roles[use.name] = []
            for role in use.roles:
                if role not in roles[use.name]:
                    roles[use.name].append(role)
    return DataRequirements(tuple(ColumnUse(name, tuple(roles[name])) for name in order))

def column_names_in_spec(spec: Any) -> tuple[str, ...]:
    """Return raw DataFrame columns referenced by one specification object."""
    return compile_data_requirements(spec=spec).columns


def required_columns(**role_specs: Any) -> tuple[str, ...]:
    """Union raw source columns across outcome/regressor/FE/IV/inference roles."""
    return compile_data_requirements(**role_specs).columns
