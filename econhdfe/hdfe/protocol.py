"""Structural contract implemented by every FE absorber in this package.

Public low-level APIs such as ``wild_cluster_bootstrap_ols`` accept caller
supplied residualizers. Callers therefore query optional capabilities with
``capability``; an object that does not declare one keeps the general path.
"""
from __future__ import annotations
from typing import Protocol


class Absorber(Protocol):
    nobs: int
    tol: float
    max_iter: int
    weights: object
    method: str
    acceleration: str
    # Exact-type fast paths: subclasses may change the projection itself.
    supports_projection_shortcut: bool
    supports_fe_orthogonality_certificate: bool
    supports_wild_cluster_test: bool

    def residualize(self, data, *, copy=True, return_info=False): ...

    def solve_effects(self, target): ...

    def workspace_cost(self) -> tuple[int, int]:
        """(resident bytes, bytes per pooled RHS column), excluding input."""

    def pending_cost(self) -> int:
        """Known lazy topology allocations not yet represented in process RSS."""

    def absorb_info_extras(self) -> dict:
        """Backend-specific entries appended to ``absorb_info``."""


class ReweightableAbsorber(Absorber, Protocol):
    def update_weights(self, weights, *, tol=None): ...


def capability(obj, name: str) -> bool:
    return bool(getattr(obj, name, False))
