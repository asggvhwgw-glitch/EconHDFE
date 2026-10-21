from __future__ import annotations
from dataclasses import dataclass
from .errors import SpecificationError


@dataclass(slots=True, frozen=True)
class HDFEConfig:
    """Stable strategy-level controls for HDFE execution.

    Implementation details such as Krylov restart thresholds intentionally stay
    private so solver internals can evolve without breaking the public API.
    """
    solver: str = "auto"
    tolerance: float = 1e-8
    max_iter: int = 16_000
    dof_method: str = "pairwise"
    canonicalize: bool = True

    def validate(self) -> None:
        if self.solver not in {"auto", "map", "lsmr", "twoway"}:
            raise SpecificationError("HDFE solver must be auto/map/lsmr/twoway")
        if not 0 < self.tolerance <= 1:
            raise SpecificationError("HDFE tolerance must lie in (0, 1]")
        if self.max_iter < 1:
            raise SpecificationError("HDFE max_iter must be positive")
        if self.dof_method not in {"exact", "pairwise", "firstpair", "none"}:
            raise SpecificationError("dof_method must be exact/pairwise/firstpair/none")


@dataclass(slots=True, frozen=True)
class InferenceConfig:
    """Inference/reporting controls shared across estimator families."""
    vce: str | None = None
    confidence_level: float = 0.95
    diagnostics: str = "off"  # off | publication | full
    iv_advanced_statistics: bool = True

    def validate(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise SpecificationError("confidence_level must lie in (0, 1)")
        if self.diagnostics not in {"off", "publication", "full"}:
            raise SpecificationError("diagnostics must be off/publication/full")
        if not isinstance(self.iv_advanced_statistics, bool):
            raise SpecificationError("iv_advanced_statistics must be a boolean")


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """Resource policy. Defaults are intentionally automatic and conservative."""
    threads: int | str = "auto"
    memory_budget_mb: int = 512
    profile: str = "off"  # off | summary | full
    cache: str = "auto"   # auto | on | off
    cache_validation: str = "signature"  # signature | none

    def validate(self) -> None:
        if self.threads != "auto" and (not isinstance(self.threads, int) or self.threads < 1):
            raise SpecificationError("threads must be 'auto' or a positive integer")
        if self.memory_budget_mb < 32:
            raise SpecificationError("memory_budget_mb must be at least 32")
        if self.profile not in {"off", "summary", "full"}:
            raise SpecificationError("profile must be off/summary/full")
        if self.cache not in {"auto", "on", "off"}:
            raise SpecificationError("cache must be auto/on/off")
        if self.cache_validation not in {"signature", "none"}:
            raise SpecificationError("cache_validation must be signature/none")
