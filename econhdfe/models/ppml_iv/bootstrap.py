from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from ...errors import BootstrapError, SpecificationError, error_boundary
from ...resampling import ClusterSampler, run_replicates
from .bias import SPJPanel, SPJResult, _balanced_random_half, ivppml_spj
from .config import IVPPMLConfig


@dataclass(slots=True)
class SPJBootstrapResult:
    point: SPJResult
    draws: np.ndarray
    stderr: np.ndarray
    stderr_ci_implied: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    requested: int
    completed: int
    failed: int
    failure_counts: dict[str, int]
    confidence: float
    seed: int

    @property
    def se_ci_implied(self) -> np.ndarray:
        """Alias matching the terminology used by the companion implementations."""
        return self.stderr_ci_implied


@error_boundary("ivppml_bootstrap")
def ivppml_spj_bootstrap(
    data: pd.DataFrame, *, panel: SPJPanel, y: str, exog=(), endog=(), instruments=(),
    weights: str | None = None, weight_type=None, offset: str | None = None,
    exposure: str | None = None, config: IVPPMLConfig | None = None,
    reps: int = 999, seed: int = 0, confidence: float = 0.95, n_jobs: int = 1,
) -> SPJBootstrapResult:
    """Cluster bootstrap for the Class A/B/C IV-PPML SPJ estimators.

    Resampling units match the upstream templates: individuals for Class A and
    directed pairs for Classes B/C.  Failed SPJ replications are skipped and
    reported, mirroring the upstream ``capture`` logic.  Both the usual
    bootstrap standard deviation and the percentile-CI-implied standard error
    are returned.
    """
    if reps < 2:
        raise SpecificationError("reps must be at least 2", code="resampling.invalid_reps", stage="resampling")
    if not 0.0 < confidence < 1.0:
        raise SpecificationError("confidence must lie in (0, 1)", code="resampling.invalid_confidence", stage="resampling")
    if n_jobs == 0:
        raise SpecificationError("n_jobs cannot be zero", code="resampling.invalid_workers", stage="resampling")
    panel.validate(data)
    config = IVPPMLConfig() if config is None else config
    config.validate()

    point = ivppml_spj(
        data, panel=panel, y=y, exog=exog, endog=endog, instruments=instruments,
        weights=weights, weight_type=weight_type, offset=offset, exposure=exposure,
        config=config, seed=seed,
    )
    tvals = np.asarray(data[panel.time])
    time_mid = int(np.floor((np.nanmin(tvals) + np.nanmax(tvals)) / 2.0))
    seeds = np.random.SeedSequence(seed).spawn(reps)
    unit = panel.unit if panel.kind == "A" else panel.pair
    sampler = ClusterSampler.compile(data, unit)

    def one(ss):
        rng = np.random.default_rng(ss)
        boot = sampler.draw(data, rng)
        if panel.kind == "A":
            kwargs = {}
        else:
            eh = _balanced_random_half(boot[panel.exporter].to_numpy(), rng)
            ih = _balanced_random_half(boot[panel.importer].to_numpy(), rng)
            kwargs = {"_export_half": eh, "_import_half": ih}
        r = ivppml_spj(
            boot, panel=panel, y=y, exog=exog, endog=endog, instruments=instruments,
            weights=weights, weight_type=weight_type, offset=offset, exposure=exposure,
            config=config, seed=int(rng.integers(0, 2**32 - 1)), _time_mid=time_mid, **kwargs,
        )
        if r.names != point.names or not np.all(np.isfinite(r.params)):
            raise BootstrapError(
                "bootstrap replicate changed the estimable coefficient set or returned non-finite coefficients",
                code="resampling.invalid_draw", stage="resampling",
            )
        return r.params

    batch = run_replicates(one, seeds, n_jobs=n_jobs)
    good = [np.asarray(x, dtype=np.float64) for x in batch.values]
    if len(good) < 2:
        raise BootstrapError(
            "fewer than two successful IV-PPML bootstrap replications",
            code="resampling.too_few_successes", stage="resampling",
            details={"requested": int(reps), "completed": len(good), "failures": batch.failure_counts},
        )
    draws = np.vstack(good)
    alpha = 1.0 - confidence
    ci_low = np.quantile(draws, alpha / 2.0, axis=0)
    ci_high = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    zcrit = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    ci_implied = (ci_high - ci_low) / (2.0 * zcrit)

    return SPJBootstrapResult(
        point=point,
        draws=draws,
        stderr=np.std(draws, axis=0, ddof=1),
        stderr_ci_implied=ci_implied,
        ci_low=ci_low,
        ci_high=ci_high,
        requested=int(reps),
        completed=len(good),
        failed=int(reps - len(good)),
        failure_counts=batch.failure_counts,
        confidence=float(confidence),
        seed=int(seed),
    )
