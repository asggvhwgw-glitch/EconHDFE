from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from ...errors import SpecificationError


@dataclass(slots=True)
class PPMLConfig:
    """Numerical controls kept separate from the estimator state."""

    tolerance: float = 1e-8
    target_inner_tol: float = 1e-9
    start_inner_tol: float = 1e-4
    max_iter: int = 10_000
    min_ok: int = 1
    standardize: bool = True
    fast_partial: bool = True
    fast_solver: bool = True
    separation: tuple[str, ...] = ("fe", "simplex", "relu")
    mu_tol: float = 1e-6
    simplex_tol: float = 1e-12
    simplex_max_iter: int = 1_000
    relu_tol: float = 1e-4
    relu_zero_tol: float = 1e-8
    relu_max_iter: int = 100
    engine: str = "replica"
    absorb_method: str = "map"  # map, lsmr, or optional native within
    dof_method: str = "pairwise"

    def validate(self) -> None:
        if not 0 < self.tolerance <= 1:
            raise SpecificationError("tolerance must lie in (0, 1]")
        if not 0 < self.target_inner_tol <= 1:
            raise SpecificationError("target_inner_tol must lie in (0, 1]")
        if not 0 < self.start_inner_tol <= 1:
            raise SpecificationError("start_inner_tol must lie in (0, 1]")
        for name in ("max_iter", "min_ok", "simplex_max_iter", "relu_max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise SpecificationError(f"{name} must be a positive integer", code="specification.iteration_count")
        if self.absorb_method not in {"map", "lsmr", "within"}:
            raise SpecificationError("absorb_method must be map/lsmr/within", code="specification.absorb_method")
        if self.engine not in {"replica", "optimized"}:
            raise SpecificationError("engine must be 'replica' or 'optimized'")
        if self.dof_method not in {"exact", "pairwise", "firstpair", "none"}:
            raise SpecificationError("dof_method must be exact/pairwise/firstpair/none")
        if not 0 < self.mu_tol < 0.1:
            raise SpecificationError("mu_tol must lie in (0, 0.1)")
        allowed = {"fe", "simplex", "relu", "mu"}
        bad = set(self.separation) - allowed
        if bad:
            raise SpecificationError(f"unknown separation method(s): {sorted(bad)}")
