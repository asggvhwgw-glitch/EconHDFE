from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from .errors import SpecificationError


@dataclass(slots=True, frozen=True)
class SamplePolicy:
    """Semantics of a ``singleton_mode`` value, queried instead of comparing names.

    ``stata_compat`` follows the upstream singleton/FE-zero counting
    conventions; ``unified`` counts frequency multiplicity at every stage
    and keeps singleton and FE-zero removals as separate reasons. IV-PPML's
    default FE-only separation is a correctness exception in both modes;
    this label is not a promise to reproduce all upstream sample rules.
    """
    mode: str = "stata_compat"

    @property
    def unified(self) -> bool:
        return self.mode == "unified"

    @property
    def profile(self) -> str:
        """Versioned profile label reported in sample-policy diagnostics."""
        return "frequency-invariant-v1" if self.unified else "stata-source-20261027"

    @property
    def requires_positive_user_weights(self) -> bool:
        return self.unified

    @property
    def allows_group_individual(self) -> bool:
        return not self.unified

    @property
    def slope_only_eligibility(self) -> bool:
        """A slope-only singleton whose slopes are all zero cannot fit its row."""
        return self.unified

    @property
    def disjoint_reasons(self) -> bool:
        """Report singleton and FE-zero (separation) removals separately."""
        return self.unified

    def psd_frequency_scale(self, weight_kind) -> bool:
        """Multiway PSD-repair scale from frequency-expanded rows (otherwise reghdfe's unweighted rows)."""
        return self.unified and weight_kind == "fweight"

    def initial_frequency_count(self, weight_kind) -> bool:
        """Whether initial Poisson singleton clearing counts fweight mass."""
        return self.unified and weight_kind == "fweight"

    def fe_zero_peeling(self, methods, drop_singletons) -> bool:
        """Initial FE-zero clearing; compatibility couples it to singleton dropping."""
        return "fe" in methods and (self.unified or drop_singletons)

    @property
    def iv_default_separation(self) -> tuple[str, ...]:
        """IV-PPML separation when unset: PPML regressor separation is not an IV existence test."""
        return ("fe",)


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
    singleton_mode: str = "stata_compat"

    @property
    def sample_policy(self) -> SamplePolicy:
        return SamplePolicy(self.singleton_mode)

    def validate(self) -> None:
        validate_singleton_mode(self.singleton_mode)
        if self.solver not in {"auto", "map", "lsmr", "twoway", "within"}:
            raise SpecificationError("HDFE solver must be auto/map/lsmr/twoway/within")
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

    ``ivppml_misspecification_robust`` makes overidentified IV-PPML use the
    estimating-equation sandwich, which stays valid when the overidentifying
    moments do not hold; the default keeps the ``ivppmlhdfe`` sandwich. Both
    agree asymptotically under correct specification, and at a solved root
    in exactly identified models. Regular identification and suitable
    dependence/asymptotic assumptions remain necessary.

    ``ppml_small_sample`` selects ordinary PPML's finite-sample covariance
    convention, independently of sample selection. ``"asymptotic"`` uses
    ppmlhdfe's N/(N-1) robust and G/(G-1) cluster factors; ``"residual"``
    retains the residual-degrees-of-freedom factors used before 0.9.4.
    """
    vce: str | None = None
    confidence_level: float = 0.95
    diagnostics: str = "off"  # off | publication | full
    iv_advanced_statistics: bool = True
    ivppml_misspecification_robust: bool = False
    ppml_small_sample: str = "asymptotic"

    def validate(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise SpecificationError("confidence_level must lie in (0, 1)")
        if self.diagnostics not in {"off", "publication", "full"}:
            raise SpecificationError("diagnostics must be off/publication/full")
        if not isinstance(self.iv_advanced_statistics, bool):
            raise SpecificationError("iv_advanced_statistics must be a boolean")
        if not isinstance(self.ivppml_misspecification_robust, bool):
            raise SpecificationError("ivppml_misspecification_robust must be a boolean")
        if not isinstance(self.ppml_small_sample, str) or self.ppml_small_sample not in {"asymptotic", "residual"}:
            raise SpecificationError("ppml_small_sample must be asymptotic/residual")


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
    memory_policy: str = "conservative"  # conservative | relaxed

    def validate(self) -> None:
        if self.memory_policy not in {"conservative", "relaxed"}:
            raise SpecificationError("memory_policy must be conservative/relaxed")
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


def validate_singleton_mode(mode):
    if not isinstance(mode, str) or mode not in {"stata_compat", "unified"}:
        raise SpecificationError(
            "singleton_mode must be 'stata_compat' or 'unified'",
            code="specification.singleton_mode", stage="specification",
        )
