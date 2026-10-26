from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from .errors import SpecificationError


@dataclass(slots=True, frozen=True)
class HDFEConfig:
    """Stable strategy-level controls for HDFE execution.

    Implementation details such as Krylov restart thresholds intentionally stay
    private so solver internals can evolve without breaking the public API.

    For ``olshdfe``/``ivhdfe``, ``solver``/``tolerance``/``max_iter``/
    ``dof_method``/``canonicalize`` pair with ``method``/``tol``/``max_iter``/
    ``dof_method``/``canonicalize_fe`` under :func:`resolve_override`.
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
    """Inference/reporting controls shared across estimator families.

    ``diagnostics`` sets what ``publication_output()`` reports by default:
    ``"off"`` gives estimates, model statistics and reproducibility metadata
    only; ``"publication"`` also includes the stored diagnostics; ``"full"``
    additionally includes the profile trace (``include_profile`` defaults to
    True; a profile exists only when ``ExecutionConfig.profile`` is not
    ``"off"``). Explicit ``include_diagnostics``/``include_profile`` arguments
    take precedence.

    For ``olshdfe``/``ivhdfe`` and the linear sessions, ``vce`` pairs with the
    ``vce`` argument under :func:`resolve_override`.
    """
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
    """Resource policy. Defaults are intentionally automatic and conservative.

    For ``olshdfe``/``ivhdfe``, ``memory_budget_mb``/``threads`` pair with
    ``memory_budget_mb``/``absorb_threads`` under :func:`resolve_override`.
    """
    threads: int | str = "auto"
    memory_budget_mb: int | float | str = "auto"
    profile: str = "off"  # off | summary | full
    cache: str = "auto"   # auto | on | off
    cache_validation: str = "signature"  # signature | none

    def validate(self) -> None:
        if self.threads != "auto" and (not isinstance(self.threads, int) or self.threads < 1):
            raise SpecificationError("threads must be 'auto' or a positive integer")
        if self.memory_budget_mb != "auto" and (
            not isinstance(self.memory_budget_mb, (int, float))
            or not isfinite(self.memory_budget_mb) or self.memory_budget_mb < 32
        ):
            raise SpecificationError("memory_budget_mb must be auto or at least 32")
        if self.profile not in {"off", "summary", "full"}:
            raise SpecificationError("profile must be off/summary/full")
        if self.cache not in {"auto", "on", "off"}:
            raise SpecificationError("cache must be auto/on/off")
        if self.cache_validation not in {"signature", "none"}:
            raise SpecificationError("cache_validation must be signature/none")


def resolve_override(name: str, value, default, config_value, config_default):
    """Choose between an explicit scalar argument and its config-object field.

    A value equal to its own default counts as unset. When both sources are
    set and disagree the request is ambiguous, so it is rejected rather than
    silently preferring either one.
    """
    value_set = value != default
    config_set = config_value != config_default
    if value_set and config_set and value != config_value:
        raise SpecificationError(
            f"{name}={value!r} conflicts with the config value {config_value!r}",
            code="specification.config_conflict", stage="specification",
            details={"argument": name},
            suggestion="Set the option either as a keyword argument or in the config object, not both.",
        )
    return value if value_set or not config_set else config_value
