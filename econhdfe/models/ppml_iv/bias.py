from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from ...errors import error_boundary, InputError, BootstrapError, SpecificationError
from .api import IVPPMLHDFE
from .config import IVPPMLConfig


PanelClass = Literal["A", "B", "C"]


@dataclass(frozen=True, slots=True)
class SPJPanel:
    """Panel metadata for the three bias structures in Kwon et al. (2026)."""

    kind: PanelClass
    time: str
    unit: str | None = None
    exporter: str | None = None
    importer: str | None = None
    pair: str | None = None

    @classmethod
    def class_a(cls, *, unit: str = "id", time: str = "year") -> "SPJPanel":
        return cls("A", time=time, unit=unit)

    @classmethod
    def class_b(cls, *, exporter: str = "exp", importer: str = "imp", time: str = "year", pair: str = "pair") -> "SPJPanel":
        return cls("B", time=time, exporter=exporter, importer=importer, pair=pair)

    @classmethod
    def class_c(cls, *, exporter: str = "exp", importer: str = "imp", time: str = "year", pair: str = "pair") -> "SPJPanel":
        return cls("C", time=time, exporter=exporter, importer=importer, pair=pair)

    def validate(self, data: pd.DataFrame) -> None:
        needed = [self.time]
        if self.kind == "A":
            needed.append(self.unit)
        else:
            needed.extend([self.exporter, self.importer, self.pair])
        missing = [c for c in needed if c is None or c not in data.columns]
        if missing:
            raise InputError(f"missing SPJ panel columns: {missing}", code="resampling.missing_panel_columns", stage="resampling", details={"missing": missing})

    @property
    def absorb(self):
        if self.kind == "A":
            return (self.unit, self.time)
        if self.kind == "B":
            return ((self.exporter, self.time), (self.importer, self.time))
        return ((self.exporter, self.importer), (self.exporter, self.time), (self.importer, self.time))

    @property
    def cluster(self):
        return None if self.kind == "A" else (self.pair,)


@dataclass(slots=True)
class SPJResult:
    params: np.ndarray
    names: tuple[str, ...]
    full_coef: np.ndarray
    bias: np.ndarray
    components: dict[str, np.ndarray]
    panel_class: str
    seed: int
    fit_count: int


def _combine_a(full, time_half, unit_half):
    return 3.0 * full - time_half - unit_half


def _combine_b(full, country_mean):
    return 2.0 * full - country_mean


def _combine_c(full, country_mean, time_mean, cross_mean):
    return 4.0 * full - 2.0 * country_mean - 2.0 * time_mean + cross_mean


def _balanced_random_half(values, rng):
    vals = np.asarray(pd.unique(values))
    if len(vals) < 2:
        raise SpecificationError("SPJ split requires at least two panel units", code="resampling.spj_split", stage="resampling")
    # Match the upstream Bernoulli split while rejecting the degenerate all-in-
    # one-half event that would make a sub-estimation undefined.
    for _ in range(32):
        h = rng.random(len(vals)) < 0.5
        if np.any(h) and np.any(~h):
            return dict(zip(vals.tolist(), h.astype(np.int8).tolist(), strict=False))
    # Deterministic fallback is only for an astronomically unlikely RNG event.
    order = rng.permutation(len(vals))
    h = np.zeros(len(vals), dtype=np.int8); h[order[:len(vals)//2]] = 1
    return dict(zip(vals.tolist(), h.tolist(), strict=False))


def _slope_vector(result):
    names = tuple(n for n in result.names if n != "_cons")
    idx = [result.names.index(n) for n in names]
    return np.asarray(result.params[idx], dtype=np.float64), names


def _fit_frame(data, panel, *, y, exog, endog, instruments, config, weights, weight_type, offset, exposure):
    model = IVPPMLHDFE(data, absorb=panel.absorb, config=config)
    return model.fit(
        y, exog=exog, endog=endog, instruments=instruments,
        weights=weights, weight_type=weight_type,
        offset=offset, exposure=exposure,
        vce="robust" if panel.kind == "A" else "cluster",
        cluster=panel.cluster,
    )


def _mean_subfits(data, masks, panel, fit_kwargs, expected_names):
    vals = []
    for label, mask in masks:
        sub = data.loc[np.asarray(mask, dtype=bool)]
        if sub.empty:
            raise BootstrapError(f"SPJ subpanel {label!r} is empty", code="resampling.empty_subpanel")
        try:
            r = _fit_frame(sub, panel, **fit_kwargs)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            raise BootstrapError(f"SPJ subpanel {label!r} failed", code="resampling.subpanel_failure", details={"label": label, "cause_type": type(exc).__name__}) from exc
        b, names = _slope_vector(r)
        if names != expected_names:
            raise BootstrapError(f"SPJ subpanel {label!r} changed the estimable coefficient set", code="resampling.coefficient_set_changed")
        vals.append(b)
    return np.mean(vals, axis=0), vals


@error_boundary("ivppml_spj")
def ivppml_spj(
    data: pd.DataFrame, *, panel: SPJPanel, y: str, exog=(), endog=(), instruments=(),
    weights: str | None = None, weight_type=None, offset: str | None = None,
    exposure: str | None = None, config: IVPPMLConfig | None = None, seed: int = 0,
    _export_half=None, _import_half=None, _time_mid=None,
) -> SPJResult:
    """Split-panel jackknife bias correction for IV-PPML-HDFE.

    The formulas and split structures match the repository templates shipped
    with `ivppmlhdfe`: Class A (individual + time), Class B (four country
    subpanels), and Class C (country, time, and eight cross subpanels).
    """
    panel.validate(data)
    config = IVPPMLConfig() if config is None else config
    config.validate()
    rng = np.random.default_rng(seed)
    fit_kwargs = dict(
        y=y, exog=tuple(exog), endog=tuple(endog), instruments=tuple(instruments),
        config=config, weights=weights, weight_type=weight_type, offset=offset, exposure=exposure,
    )
    full_r = _fit_frame(data, panel, **fit_kwargs)
    full, names = _slope_vector(full_r)
    components: dict[str, np.ndarray] = {"full": full.copy()}
    fit_count = 1

    years = np.asarray(data[panel.time])
    time_mid = int(np.floor((np.nanmin(years) + np.nanmax(years)) / 2.0)) if _time_mid is None else _time_mid

    if panel.kind == "A":
        time_mean, _ = _mean_subfits(
            data, [("time_low", years <= time_mid), ("time_high", years > time_mid)],
            panel, fit_kwargs, names,
        )
        fit_count += 2
        half = _balanced_random_half(data[panel.unit].to_numpy(), rng)
        uh = data[panel.unit].map(half).to_numpy()
        unit_mean, _ = _mean_subfits(
            data, [("unit_0", uh == 0), ("unit_1", uh == 1)], panel, fit_kwargs, names,
        )
        fit_count += 2
        coef = _combine_a(full, time_mean, unit_mean)
        components.update(time_half=time_mean, unit_half=unit_mean)

    else:
        if _export_half is None or _import_half is None:
            common = _balanced_random_half(
                np.concatenate([data[panel.exporter].to_numpy(), data[panel.importer].to_numpy()]), rng
            )
            export_half = common; import_half = common
        else:
            export_half, import_half = _export_half, _import_half
        eh = data[panel.exporter].map(export_half).to_numpy()
        ih = data[panel.importer].map(import_half).to_numpy()
        if np.any(pd.isna(eh)) or np.any(pd.isna(ih)):
            raise SpecificationError("SPJ country split does not cover all exporter/importer IDs", code="resampling.invalid_country_split", stage="resampling")
        country_masks = [
            (f"country_{ee}{ii}", (eh == ee) & (ih == ii))
            for ee in (0,1) for ii in (0,1)
        ]
        country_mean, _ = _mean_subfits(data, country_masks, panel, fit_kwargs, names)
        fit_count += 4
        components["country_mean"] = country_mean
        if panel.kind == "B":
            coef = _combine_b(full, country_mean)
        else:
            th = years <= time_mid
            time_mean, _ = _mean_subfits(
                data, [("time_0", ~th), ("time_1", th)], panel, fit_kwargs, names,
            )
            fit_count += 2
            cross_masks = [
                (f"cross_{ee}{ii}{tt}", (eh == ee) & (ih == ii) & (th == bool(tt)))
                for ee in (0,1) for ii in (0,1) for tt in (0,1)
            ]
            cross_mean, _ = _mean_subfits(data, cross_masks, panel, fit_kwargs, names)
            fit_count += 8
            coef = _combine_c(full, country_mean, time_mean, cross_mean)
            components.update(time_mean=time_mean, cross_mean=cross_mean)

    return SPJResult(
        params=np.asarray(coef), names=names, full_coef=full,
        bias=full-np.asarray(coef), components=components,
        panel_class=panel.kind, seed=int(seed), fit_count=fit_count,
    )
