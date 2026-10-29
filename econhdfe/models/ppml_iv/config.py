from __future__ import annotations
from dataclasses import dataclass, replace
from numbers import Integral
import warnings
from ...errors import SpecificationError
from ..ppml.config import PPMLConfig


@dataclass(slots=True)
class IVPPMLConfig(PPMLConfig):
    """Numerical controls for IV-PPML-HDFE.

    Standardization is opt-in to match the upstream ``ivppmlhdfe`` command.
    ``separation=None`` (default) uses FE zero-level checks in both sample
    policies. Ordinary PPML regressor separation is not an existence test for
    IV moments and can remove observations from an identified IV model.
    Explicitly requesting the legacy methods warns when the config is created
    in compatibility mode; unified mode rejects them in ``validate``.
    """

    separation: tuple[str, ...] | None = None
    standardize: bool = False
    max_step_halving: int = 2
    step_halving_memory: float = 0.9

    def __post_init__(self):
        # Warn where the user asks for it, once; validate() has no side effects.
        if not self.sample_policy.unified and any(m != "fe" for m in self.separation_methods):
            warnings.warn(
                "Legacy IV-PPML separation was explicitly requested. PPML regressor "
                "and mu-threshold checks do not certify existence of IV moments and "
                "can change an identified IV sample; use separation=('fe',) for the safe default.",
                RuntimeWarning, stacklevel=3,
            )

    @property
    def separation_methods(self) -> tuple[str, ...]:
        return ("fe",) if self.separation is None else tuple(self.separation)

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
