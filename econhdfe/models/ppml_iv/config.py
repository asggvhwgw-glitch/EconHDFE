from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from ...errors import SpecificationError
from ..ppml.config import PPMLConfig


@dataclass(slots=True)
class IVPPMLConfig(PPMLConfig):
    """Numerical controls for IV-PPML-HDFE.

    Standardization is opt-in to match the upstream ``ivppmlhdfe`` command.
    Compatibility defaults to upstream FE/simplex/ReLU checks. Unified mode
    defaults to FE zero-level checks only: ordinary PPML regressor separation
    is not an existence test for the IV moments. Construct a new config when
    switching policy; explicitly unsafe method combinations are rejected.
    """

    separation: tuple[str, ...] | None = None
    standardize: bool = False
    max_step_halving: int = 2
    step_halving_memory: float = 0.9
    # Opt-in outer algorithm; the legacy estimator remains the default.
    outer_solver: str = "legacy"
    outer_linear_solver: str = "auto"
    outer_chunk_rows: int = 65536
    outer_gram_rcond: float = 1e-8
    outer_max_backtracks: int = 20
    outer_armijo: float = 1e-4
    outer_forcing: float = 0.25
    outer_record_history: bool = True
    outer_profile_max_iter: int = 80
    outer_inference: str = "standard"
    outer_overid_test: bool = False

    def __post_init__(self):
        if self.separation is None:
            self.separation = ("fe",) if (self.singleton_mode == "unified" or self.outer_solver in {"moment", "profile"}) else ("fe", "simplex", "relu")

    def validate(self) -> None:
        PPMLConfig.validate(self)
        if self.singleton_mode == "unified" and any(m != "fe" for m in self.separation):
            raise SpecificationError(
                "unified IV-PPML permits FE separation only; PPML regressor separation is not an IV existence test",
                code="specification.unified_iv_separation",
            )
        if isinstance(self.max_step_halving, bool) or not isinstance(self.max_step_halving, Integral) or self.max_step_halving < 0:
            raise SpecificationError("max_step_halving must be a nonnegative integer")
        if not 0.0 <= self.step_halving_memory < 1.0:
            raise SpecificationError("step_halving_memory must lie in [0, 1)")

        if self.outer_solver not in {"legacy", "moment", "profile"}:
            raise SpecificationError("outer_solver must be legacy/moment/profile")
        if self.outer_linear_solver not in {"auto", "stable"}:
            raise SpecificationError("outer_linear_solver must be auto/stable")
        if self.outer_solver in {"moment", "profile"} and any(m != "fe" for m in self.separation):
            raise SpecificationError("moment outer solver permits FE separation only; PPML regressor separation is not an IV existence test")
        for name in ("outer_chunk_rows", "outer_max_backtracks", "outer_profile_max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise SpecificationError(f"{name} must be a positive integer")
        for name in ("outer_gram_rcond", "outer_armijo", "outer_forcing"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise SpecificationError(f"{name} must lie in (0, 1)")
        if not isinstance(self.outer_record_history, bool):
            raise SpecificationError("outer_record_history must be boolean")
        if self.outer_inference not in {"standard", "estimating_equation"}:
            raise SpecificationError("outer_inference must be standard/estimating_equation")
        if not isinstance(self.outer_overid_test, bool):
            raise SpecificationError("outer_overid_test must be boolean")
        if self.outer_solver == "legacy" and (self.outer_inference != "standard" or self.outer_overid_test):
            raise SpecificationError("new overidentified inference requires a certified moment/profile fit")
