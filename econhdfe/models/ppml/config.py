from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from ...errors import SpecificationError
from ...config import SamplePolicy, validate_singleton_mode


@dataclass(slots=True)
class PPMLConfig:
    """Numerical controls kept separate from the estimator state.

    ``engine='within'`` opts into the optional native weighted FE backend for
    rank checks, IRLS and final inference; separation retains replica solvers.
    """

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
    dof_method: str = "pairwise"
    singleton_mode: str = "stata_compat"
    drop_singletons: bool = True

    @property
    def sample_policy(self) -> SamplePolicy:
        return SamplePolicy(self.singleton_mode)

    @property
    def separation_methods(self) -> tuple[str, ...]:
        """Separation methods to run, in the order given."""
        return tuple(self.separation)

    def validate(self) -> None:
        validate_singleton_mode(self.singleton_mode)
        if not isinstance(self.separation, (tuple, list)):
            raise SpecificationError("separation must be a tuple or list of method names")
        if not isinstance(self.drop_singletons, bool):
            raise SpecificationError("drop_singletons must be boolean")
        if self.sample_policy.unified and "mu" in self.separation:
            raise SpecificationError(
                "unified mode requires a separation certificate; mu thresholding is not supported",
                code="specification.unified_separation",
            )
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
        if self.engine not in {"replica", "optimized", "within"}:
            raise SpecificationError("engine must be 'replica', 'optimized' or 'within'")
        if self.dof_method not in {"exact", "pairwise", "firstpair", "none"}:
            raise SpecificationError("dof_method must be exact/pairwise/firstpair/none")
        if not 0 < self.mu_tol < 0.1:
            raise SpecificationError("mu_tol must lie in (0, 0.1)")
        allowed = {"fe", "simplex", "relu", "mu"}
        bad = set(self.separation) - allowed
        if bad:
            raise SpecificationError(f"unknown separation method(s): {sorted(bad)}")
