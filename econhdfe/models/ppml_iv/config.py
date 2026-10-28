from __future__ import annotations
from dataclasses import dataclass, replace
from numbers import Integral
from ...errors import SpecificationError
from ..ppml.config import PPMLConfig


@dataclass(slots=True)
class IVPPMLConfig(PPMLConfig):
    """Numerical controls for IV-PPML-HDFE.

    Standardization is opt-in to match the upstream ``ivppmlhdfe`` command.
    ``separation=None`` (default) follows the sample policy wherever the
    config is used: compatibility runs upstream FE/simplex/ReLU checks,
    unified mode FE zero-level checks only, because ordinary PPML regressor
    separation is not an existence test for the IV moments. The default
    therefore follows ``dataclasses.replace`` of ``singleton_mode``;
    explicitly unsafe method combinations are rejected.
    """

    separation: tuple[str, ...] | None = None
    standardize: bool = False
    max_step_halving: int = 2
    step_halving_memory: float = 0.9

    @property
    def separation_methods(self) -> tuple[str, ...]:
        if self.separation is None:
            return self.sample_policy.iv_default_separation
        return tuple(self.separation)

    def validate(self) -> None:
        PPMLConfig.validate(self if self.separation is not None else replace(self, separation=self.separation_methods))
        if self.sample_policy.unified and any(m != "fe" for m in self.separation_methods):
            raise SpecificationError(
                "unified IV-PPML permits FE separation only; PPML regressor separation is not an IV existence test",
                code="specification.unified_iv_separation",
            )
        if isinstance(self.max_step_halving, bool) or not isinstance(self.max_step_halving, Integral) or self.max_step_halving < 0:
            raise SpecificationError("max_step_halving must be a nonnegative integer")
        if not 0.0 <= self.step_halving_memory < 1.0:
            raise SpecificationError("step_halving_memory must lie in [0, 1)")
