from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..errors import InvalidWeightError, SpecificationError, ShapeError
from .vcov import IID_VCE_KINDS, ROBUST_VCE_KINDS, normalize_vce


_WEIGHT_ALIASES = {
    None: "none",
    "": "none",
    "none": "none",
    "unweighted": "none",
    "generic": "generic",
    "wls": "generic",
    "fweight": "fweight",
    "fw": "fweight",
    "frequency": "fweight",
    "aweight": "aweight",
    "aw": "aweight",
    "analytic": "aweight",
    "pweight": "pweight",
    "pw": "pweight",
    "probability": "pweight",
}


@dataclass(slots=True, frozen=True)
class WeightInfo:
    """Prepared regression weights with Stata-compatible scaling metadata.

    ``estimation`` is the vector used by the absorber and weighted least
    squares.  Analytic/probability weights are normalized to sum to the number
    of physical rows, as in reghdfe/ivreg2.  This scalar normalization leaves
    point estimates unchanged but matters for reproducing VCE bookkeeping.
    """

    kind: str
    raw: np.ndarray | None
    estimation: np.ndarray | None
    row_n: int
    effective_n: float
    sum_weights: float

    @property
    def is_frequency(self) -> bool:
        return self.kind == "fweight"

    @property
    def force_robust(self) -> bool:
        return self.kind == "pweight"


def normalize_weight_type(weight_type, *, has_weights: bool) -> str:
    if weight_type is None:
        return "generic" if has_weights else "none"
    key = str(weight_type).lower().replace("_", "").replace("-", "")
    # Preserve ordinary spellings before removing punctuation.
    aliases = {str(k).lower().replace("_", "").replace("-", ""): v
               for k, v in _WEIGHT_ALIASES.items() if k is not None}
    if key not in aliases:
        raise SpecificationError("weight_type must be one of generic, fweight, aweight, or pweight", code="specification.weight_type", stage="frontend")
    kind = aliases[key]
    if kind == "none" and has_weights:
        raise SpecificationError("weight_type='none' cannot be used with weights", code="specification.weight_type", stage="frontend")
    if kind != "none" and not has_weights:
        raise SpecificationError(f"weight_type={kind!r} requires weights", code="specification.weight_type", stage="frontend")
    return kind


def prepare_weights(weights, n: int, weight_type=None, *, normalize_ap: bool = True) -> WeightInfo:
    has = weights is not None
    kind = normalize_weight_type(weight_type, has_weights=has)
    if not has:
        return WeightInfo("none", None, None, int(n), float(n), float(n))

    raw = np.asarray(weights, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != n:
        raise ShapeError("weights must be a one-dimensional array with nobs entries", details={"expected": int(n), "shape": raw.shape})
    if not np.all(np.isfinite(raw)):
        raise InvalidWeightError("weights must be finite", code="input.invalid_weight.non_finite")
    if kind == "generic":
        if np.any(raw < 0) or not np.any(raw > 0):
            raise InvalidWeightError("generic weights must be nonnegative with at least one positive value", code="input.invalid_weight.range")
    elif np.any(raw <= 0):
        # Typed Stata weights are kept strict here. Zero-weight sample-selection
        # semantics will be added only together with an explicit e(sample)-like
        # API so FE/singleton behavior cannot change silently.
        raise InvalidWeightError("weights must be strictly positive", code="input.invalid_weight.range")

    if kind == "fweight":
        rounded = np.rint(raw)
        if not np.allclose(raw, rounded, rtol=0.0, atol=1e-12):
            raise InvalidWeightError("fweights must be integer-valued", code="input.invalid_weight.frequency")
        estimation = rounded.astype(np.float64, copy=False)
        effective_n = float(np.sum(estimation))
    elif kind in {"aweight", "pweight"}:
        total = float(np.sum(raw))
        estimation = raw * (float(n) / total) if normalize_ap else raw
        effective_n = float(n)
    else:  # legacy/generic WLS semantics
        estimation = raw
        effective_n = float(n)

    return WeightInfo(
        kind=kind,
        raw=raw,
        estimation=np.asarray(estimation, dtype=np.float64),
        row_n=int(n),
        effective_n=effective_n,
        sum_weights=float(np.sum(raw)),
    )


def resolve_vce_for_weights(vce, info: WeightInfo, *, has_clusters: bool = False):
    """Apply Stata's pweight => robust rule without overriding cluster/HAC VCE."""
    if vce is None:
        return "cluster" if has_clusters else ("robust" if info.force_robust else "iid")
    kind = normalize_vce(vce)
    if info.force_robust and kind in IID_VCE_KINDS:
        return "robust"
    return kind


def robust_score_scale(info: WeightInfo, vce_kind: str, n: int) -> np.ndarray | None:
    """Observation score correction for frequency weights.

    A transformed WLS row has score ``w*x*e``.  For heteroskedastic-robust VCE,
    a frequency-weight row represents ``w`` independent duplicate observations,
    whose combined meat contribution is ``w*(x*e)(x*e)'``.  Multiplying the
    transformed score by ``1/sqrt(w)`` reproduces that contribution.  Cluster
    scores are *not* adjusted because duplicates inside a cluster aggregate to
    ``w*x*e`` before the outer product.
    """
    kind = normalize_vce(vce_kind)
    if info.kind != "fweight" or kind not in ROBUST_VCE_KINDS:
        return None
    return 1.0 / np.sqrt(np.asarray(info.estimation, dtype=np.float64))
