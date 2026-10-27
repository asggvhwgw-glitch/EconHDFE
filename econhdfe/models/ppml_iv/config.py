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

    def __post_init__(self):
        if self.separation is None:
            self.separation = ("fe",) if self.singleton_mode == "unified" else ("fe", "simplex", "relu")

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
