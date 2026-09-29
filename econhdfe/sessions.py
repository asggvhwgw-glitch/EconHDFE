from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ExecutionConfig, HDFEConfig, InferenceConfig
from .data.dataset import EncodedEconometricDataset, prepare_repeated_dataset
from .data.persistent import PersistentSessionStore
from .data.source import DataSource, as_data_source
from .frontend.columns import compile_data_requirements, merge_data_requirements
from .errors import SpecificationError, error_boundary
from .frontend.roles import VariableRole
from .frontend.validate import require_numeric
from .compute.weights import normalize_weight_type, prepare_weights, resolve_vce_for_weights
from .hdfe.dof import absorbed_dof
from .hdfe.factorvars import compile_absorb_factor_variable
from .hdfe.specs import Interaction
from .factorvars import FactorVariableExpression
from .models.ols import _fit_ols
from .models.linear_iv.estimators import fit_iv_gmm2s, fit_iv_kclass
from .pipeline import (
    _absorb_metadata,
    _check_absorption,
    _check_collinearity_mode,
    _cluster_arrays,
    _collinearity_metadata,
    _make_standard_absorber,
    _prepare_standard_fe_structure,
    _resolve_acceleration,
    _resolve_iv_columns,
    _resolve_method,
    _resolve_ols_columns,
    _resolve_pool_size,
    _residualize_blocks,
)
from .reporting import (
    cluster_counts as _cluster_counts, reghdfe_model_statistics, reghdfe_r2_statistics,
    ivreghdfe_model_statistics,
)
from .results import RegressionResult
from .prediction import SampleExclusionState, named_design_state, linear_prediction_state


def _as_tuple(value) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (str, FactorVariableExpression)):
        return (value,)
    return tuple(value)


def _normalize_absorb(absorb) -> tuple:
    """Canonical, hashable representation for intercept-only DataFrame FEs."""
    out = []
    for term in _as_tuple(absorb):
        if isinstance(term, FactorVariableExpression):
            compiled = compile_absorb_factor_variable(term)
            for spec in compiled:
                if spec.slopes or not spec.intercept:
                    raise SpecificationError(
                        "repeated-spec sessions currently optimize intercept-only DataFrame fixed effects; "
                        "use olshdfe()/ivhdfe() directly for fv() heterogeneous-slope absorb specifications",
                        code="specification.session_absorb_slope",
                        stage="session",
                        details={"expression": term.expression},
                    )
                if isinstance(spec.group, str):
                    out.append(spec.group)
                elif isinstance(spec.group, Interaction) and all(isinstance(x, str) for x in spec.group.parts):
                    out.append(tuple(spec.group.parts))
                else:
                    raise SpecificationError(
                        "session fv() absorb expressions must resolve to DataFrame categorical columns",
                        code="specification.session_absorb",
                        stage="session",
                        details={"expression": term.expression},
                    )
        elif isinstance(term, str):
            out.append(term)
        elif isinstance(term, (tuple, list)) and len(term) >= 2 and all(isinstance(x, str) for x in term):
            out.append(tuple(term))
        else:
            raise SpecificationError(
                "repeated-spec sessions currently optimize intercept-only DataFrame fixed effects; "
                "use olshdfe()/ivhdfe() directly for slope FEs or array-valued FE specifications"
            )
    return tuple(out)


def _hash_frame_columns(data: pd.DataFrame, columns: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray([len(data)], dtype=np.int64).tobytes())
    cols = tuple(dict.fromkeys(columns))
    if cols:
        vals = pd.util.hash_pandas_object(data[list(cols)], index=True).to_numpy(dtype=np.uint64, copy=False)
        h.update(np.ascontiguousarray(vals).view(np.uint8))
    return h.hexdigest()


def _absorb_columns(absorb: tuple) -> tuple[str, ...]:
    cols: list[str] = []
    for term in absorb:
        parts = (term,) if isinstance(term, str) else tuple(term)
        for c in parts:
            if c not in cols:
                cols.append(c)
    return tuple(cols)


@dataclass(frozen=True, slots=True)
class OLSSpec:
    y: str
    x: tuple[str, ...]
    absorb: tuple = ()
    label: str | None = None

    @classmethod
    def from_any(cls, spec) -> "OLSSpec":
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Mapping):
            if "y" not in spec:
                raise SpecificationError(
                    "OLS session spec requires key 'y'",
                    code="specification.session_spec",
                    details={"estimator": "ols", "missing": ["y"]},
                )
            return cls(
                str(spec["y"]), tuple(_as_tuple(spec.get("x"))),
                _normalize_absorb(spec.get("absorb", ())), spec.get("label"),
            )
        raise SpecificationError(
            "OLS specs must be OLSSpec objects or mappings",
            code="specification.session_spec",
            details={"estimator": "ols", "received_type": type(spec).__name__},
        )


@dataclass(frozen=True, slots=True)
class IVSpec:
    y: str
    exog: tuple[str, ...]
    endog: tuple[str, ...]
    instruments: tuple[str, ...]
    absorb: tuple = ()
    label: str | None = None

    @classmethod
    def from_any(cls, spec) -> "IVSpec":
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Mapping):
            missing = [name for name in ("y", "endog", "instruments") if name not in spec]
            if missing:
                raise SpecificationError(
                    "IV session spec requires y, endog, and instruments",
                    code="specification.session_spec",
                    details={"estimator": "iv", "missing": missing},
                )
            return cls(
                str(spec["y"]), tuple(_as_tuple(spec.get("exog"))),
                tuple(_as_tuple(spec.get("endog"))), tuple(_as_tuple(spec.get("instruments"))),
                _normalize_absorb(spec.get("absorb", ())), spec.get("label"),
            )
        raise SpecificationError(
            "IV specs must be IVSpec objects or mappings",
            code="specification.session_spec",
            details={"estimator": "iv", "received_type": type(spec).__name__},
        )


@dataclass(slots=True)
class _FECacheEntry:
    absorb: tuple
    mask: np.ndarray
    dropped: int
    groups: list[np.ndarray]
    slopes: list[np.ndarray | None]
    intercepts: list[bool]
    fe_names: list[str]
    fe_plan: object
    inference_fe: object
    absorber: object
    weights: np.ndarray | None
    winfo: object
    clusters: list[np.ndarray] | None
    dof: object
    solver_selection: dict
    pool_size: int | str
    memory_budget_mb: int
    source_signature: str | None
    within: dict[str, np.ndarray] = field(default_factory=dict)
    raw_active: dict[str, np.ndarray] = field(default_factory=dict)
    variable_signatures: dict[str, str] = field(default_factory=dict)
    last_info: object | None = None
    residualize_calls: int = 0
    residualized_columns: int = 0


class _LinearHDFESessionBase:
    """Shared repeated-spec execution cache for linear HDFE estimators.

    The cache is exact: within-transformed variables are never reused across
    different FE/sample signatures. Adding or removing an FE therefore creates
    (or reuses) a distinct compiled FE entry rather than applying an invalid
    sequential-projection shortcut.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        weights=None,
        weight_type=None,
        cluster=None,
        vce=None,
        drop_singletons: bool = True,
        hdfe_config: HDFEConfig | None = None,
        inference_config: InferenceConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        collinearity: str = "warn",
        collinear_tol: float = 1e-10,
    ):
        self._data_source = None
        self._source_role_specs: list[dict] = []
        self._loaded_columns: tuple[str, ...] = ()
        self._identifier_columns: tuple[str, ...] = ()
        self._data_materializations = 0
        self._persistent_store: PersistentSessionStore | None = None
        if isinstance(data, EncodedEconometricDataset):
            self.dataset = data
            self.data = data
            self._loaded_columns = data.columns
            self._identifier_columns = tuple(data.identifier_levels)
        elif isinstance(data, pd.DataFrame):
            self.dataset = None
            self.data = data
            self._loaded_columns = tuple(map(str, data.columns))
        else:
            try:
                self._data_source = as_data_source(data)
            except (TypeError, SpecificationError) as exc:
                raise SpecificationError(
                    "repeated-spec sessions require a pandas DataFrame, EncodedEconometricDataset, or supported DataSource",
                    code="data.session_source", stage="session",
                ) from exc
            self.dataset = None
            self.data = None
        self.weights_spec = weights
        self.weight_type = weight_type
        self.cluster_spec = cluster
        self.vce = vce
        self.drop_singletons = bool(drop_singletons)
        self.hdfe = hdfe_config or HDFEConfig()
        self.inference = inference_config or InferenceConfig(vce=vce)
        self.execution = execution_config or ExecutionConfig()
        self.hdfe.validate(); self.inference.validate(); self.execution.validate()
        self.collinearity = _check_collinearity_mode(collinearity)
        self.collinear_tol = float(collinear_tol)
        self._entries: dict[tuple, _FECacheEntry] = {}
        self._component_cache: dict = {}
        self._prime_component_cache()
        self._raw_full: dict[str, np.ndarray] = {}
        self._raw_signatures: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        self._validation_round: dict[str, str] = {}

    def enable_persistent_cache(self, root, *, source_validation: str = "strict") -> "_LinearHDFESessionBase":
        """Enable disposable cross-process reuse for exploratory workflows.

        The on-disk cache is never part of estimator semantics. Deleting it can
        only remove reuse. The method leaves constructor/public estimator
        signatures unchanged while this feature is validated experimentally.
        """
        self._persistent_store = PersistentSessionStore(root, source_validation=source_validation)
        return self

    def disable_persistent_cache(self) -> None:
        self._persistent_store = None

    def clear_persistent_cache(self) -> None:
        if self._persistent_store is not None:
            self._persistent_store.clear()

    def persistent_cache_info(self) -> dict:
        return {} if self._persistent_store is None else self._persistent_store.info()

    def _common_source_roles(self) -> dict:
        out = {}
        if isinstance(self.weights_spec, str):
            out["weights"] = self.weights_spec
        cluster = self.cluster_spec
        if isinstance(cluster, str):
            out["cluster"] = cluster
        elif isinstance(cluster, (tuple, list)) and all(isinstance(c, str) for c in cluster):
            out["cluster"] = tuple(cluster)
        return out

    def _ensure_source_specs(self, specs: Sequence[dict]) -> None:
        if self._data_source is None:
            return
        proposed = self._source_role_specs + [dict(s) for s in specs]
        reqs = []
        common = self._common_source_roles()
        for spec in proposed:
            merged = dict(common); merged.update(spec)
            reqs.append(compile_data_requirements(**merged))
        req = merge_data_requirements(*reqs)
        loaded = set(self._loaded_columns)
        identifier = tuple(req.identifier_only_columns)
        if self.dataset is not None and set(req.columns).issubset(loaded) and identifier == self._identifier_columns:
            self._source_role_specs = proposed
            return
        if self._persistent_store is None:
            dataset = prepare_repeated_dataset(
                self._data_source, proposed, common_roles=common,
                memory_budget_mb=self.execution.memory_budget_mb,
            )
        else:
            dataset = self._persistent_store.prepare_repeated_dataset(
                self._data_source, proposed, common_roles=common,
                memory_budget_mb=self.execution.memory_budget_mb,
            )
        self.dataset = dataset
        self.data = dataset
        self._source_role_specs = proposed
        self._loaded_columns = dataset.columns
        self._identifier_columns = tuple(dataset.identifier_levels)
        self._data_materializations += 1
        # A source expansion creates a new immutable snapshot. Everything below
        # the data layer must be recomputed against that snapshot.
        self._entries.clear(); self._component_cache.clear(); self._raw_full.clear(); self._raw_signatures.clear()
        self._validation_round.clear(); self._prime_component_cache()

    def _prime_component_cache(self) -> None:
        if self.dataset is None:
            return
        for name, levels in self.dataset.identifier_levels.items():
            codes = np.asarray(self.dataset.column(name), dtype=np.int32)
            self._component_cache[("column", name)] = (codes, int(len(levels)))

    def _start_validation_round(self) -> None:
        self._validation_round = {}

    def _column_signature(self, name: str) -> str:
        hit = self._validation_round.get(name)
        if hit is not None:
            return hit
        if name not in self.data.columns:
            raise SpecificationError(f"column {name!r} is not present")
        if self.dataset is not None:
            out = self.dataset.column_signature(name)
        else:
            h = hashlib.blake2b(digest_size=16)
            h.update(np.asarray([len(self.data)], dtype=np.int64).tobytes())
            h.update(name.encode())
            vals = pd.util.hash_pandas_object(self.data[name], index=True).to_numpy(dtype=np.uint64, copy=False)
            h.update(np.ascontiguousarray(vals).view(np.uint8))
            out = h.hexdigest()
        self._validation_round[name] = out
        return out

    def _source_signature(self, columns: Sequence[str]) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray([len(self.data)], dtype=np.int64).tobytes())
        for name in tuple(dict.fromkeys(columns)):
            h.update(name.encode()); h.update(self._column_signature(name).encode())
        return h.hexdigest()

    def clear_cache(self) -> None:
        self._entries.clear()
        self._component_cache.clear()
        self._prime_component_cache()
        self._raw_full.clear()
        self._raw_signatures.clear()
        self._validation_round.clear()
        self._hits = self._misses = 0

    def cache_info(self) -> dict:
        out = {
            "fe_entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "within_columns": sum(len(e.within) for e in self._entries.values()),
            "residualize_calls": sum(e.residualize_calls for e in self._entries.values()),
            "residualized_columns": sum(e.residualized_columns for e in self._entries.values()),
            "fe_specs": tuple(self._entries),
        }
        if self.dataset is not None:
            out["encoded_dataset"] = self.dataset.as_dict()
        if self._data_source is not None:
            out["data_source"] = {
                "source_type": type(self._data_source).__name__,
                "materializations": self._data_materializations,
                "loaded_column_count": len(self._loaded_columns),
            }
        if self._persistent_store is not None:
            out["persistent_cache"] = self._persistent_store.info()
        return out

    @staticmethod
    def _hash_numeric_array(values) -> str:
        arr = np.asarray(values)
        h = hashlib.blake2b(digest_size=16)
        h.update(str(arr.dtype).encode())
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(arr).view(np.uint8))
        return h.hexdigest()

    def _persistent_external_descriptor(self, value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (tuple, list)):
            return tuple(self._persistent_external_descriptor(v) for v in value)
        try:
            return {"array_signature": self._hash_numeric_array(value)}
        except Exception:
            return {"unsupported": type(value).__name__}

    def _persistent_entry_signature(self, entry: _FECacheEntry) -> str | None:
        if self._persistent_store is None:
            return None
        source_sig = entry.source_signature
        if source_sig is None:
            source_sig = self._source_signature(self._entry_source_columns(entry.absorb))
        mask_sig = self._hash_numeric_array(np.asarray(entry.mask, dtype=np.uint8))
        payload = {
            "absorb": entry.absorb,
            "source_signature": source_sig,
            "mask_signature": mask_sig,
            "weights": self._persistent_external_descriptor(self.weights_spec),
            "weight_type": self.weight_type,
            "drop_singletons": self.drop_singletons,
            "hdfe": asdict(self.hdfe),
            "memory_budget_mb": self.execution.memory_budget_mb,
        }
        return self._persistent_store.within_signature(entry_payload=payload)

    def _persistent_result_key(self, estimator: str, spec, *, extra: dict | None = None) -> str | None:
        if self._persistent_store is None or self.execution.profile != "off":
            return None
        if hasattr(spec, "y"):
            names = [spec.y]
            for attr in ("x", "exog", "endog", "instruments"):
                names.extend(getattr(spec, attr, ()) or ())
        else:
            return None
        # A completed-result key is independent of the physical FE solver
        # object. Exact source/column signatures plus the full econometric and
        # numerical configuration determine the result, so a cache hit can be
        # returned before rebuilding FE topology.
        source_names = list(names)
        source_names.extend(self._entry_source_columns(getattr(spec, "absorb", ())))
        variables = {name: self._column_signature(name) for name in dict.fromkeys(source_names)}
        payload = {
            "estimator": estimator,
            "spec": asdict(spec),
            "variables": variables,
            "weights": self._persistent_external_descriptor(self.weights_spec),
            "weight_type": self.weight_type,
            "cluster": self._persistent_external_descriptor(self.cluster_spec),
            "vce": self.vce,
            "drop_singletons": self.drop_singletons,
            "hdfe": asdict(self.hdfe),
            "inference": asdict(self.inference),
            "memory_budget_mb": self.execution.memory_budget_mb,
            "collinearity": self.collinearity,
            "collinear_tol": self.collinear_tol,
            "extra": dict(extra or {}),
        }
        return self._persistent_store.result_key(payload)

    def _full_numeric(self, name: str) -> np.ndarray:
        if name not in self.data.columns:
            raise SpecificationError(f"column {name!r} is not present")
        validate = self.execution.cache_validation == "signature"
        sig = self._column_signature(name) if validate else None
        if name in self._raw_full and (not validate or self._raw_signatures.get(name) == sig):
            return self._raw_full[name]
        if self.dataset is not None:
            a = self.dataset.numeric_column(name)
        else:
            a = require_numeric(self.data[name].to_numpy(), name=name, role=VariableRole.REGRESSOR, ndim=1)
        self._raw_full[name] = a
        if validate:
            self._raw_signatures[name] = sig
        # Any mutation to a cached numeric column invalidates its within copies.
        for entry in self._entries.values():
            entry.within.pop(name, None)
            entry.raw_active.pop(name, None)
            entry.variable_signatures.pop(name, None)
        return a

    def _weights_full(self):
        if self.weights_spec is None:
            return None
        if isinstance(self.weights_spec, str):
            if self.dataset is not None:
                return self.dataset.numeric_column(self.weights_spec)
            return require_numeric(
                self.data[self.weights_spec].to_numpy(), name=self.weights_spec,
                role=VariableRole.WEIGHT, ndim=1,
            )
        return require_numeric(self.weights_spec, name="weights", role=VariableRole.WEIGHT, ndim=1)

    def _entry_key(self, absorb: tuple) -> tuple:
        return absorb

    def _entry_source_columns(self, absorb: tuple) -> tuple[str, ...]:
        cols = list(_absorb_columns(absorb))
        if isinstance(self.weights_spec, str) and self.weights_spec not in cols:
            cols.append(self.weights_spec)
        cluster = self.cluster_spec
        if isinstance(cluster, str):
            cluster = (cluster,)
        elif isinstance(cluster, (list, tuple)):
            cluster = tuple(c for c in cluster if isinstance(c, str))
        else:
            cluster = ()
        for c in cluster:
            if c not in cols:
                cols.append(c)
        return tuple(cols)

    def _get_entry(self, absorb) -> _FECacheEntry:
        absorb = _normalize_absorb(absorb)
        key = self._entry_key(absorb)
        validate = self.execution.cache_validation == "signature"
        source_cols = self._entry_source_columns(absorb)
        source_sig = self._source_signature(source_cols) if validate else None
        hit = self._entries.get(key)
        if hit is not None and (not validate or hit.source_signature == source_sig):
            self._hits += 1
            return hit
        if hit is not None:
            # FE/weight/cluster source mutation can stale the shared dense-code
            # cache. Invalidate every compiled FE entry rather than attempting
            # partial dependency tracking. Mutations are rare; correctness wins.
            self._entries.clear()
            self._component_cache.clear()
            self._prime_component_cache()
        self._misses += 1

        w0 = self._weights_full()
        wkind = normalize_weight_type(self.weight_type, has_weights=w0 is not None)
        _ = prepare_weights(w0, len(self.data), wkind)
        if absorb:
            groups, slopes, intercepts, fe_names, fe_plan, mask, dropped, inference_fe = _prepare_standard_fe_structure(
                self.data, absorb,
                canonicalize_fe=self.hdfe.canonicalize,
                drop_singletons=self.drop_singletons,
                weights=w0,
                weight_kind=wkind,
                component_cache=self._component_cache,
            )
            method, solver_selection = _resolve_method(
                self.hdfe.solver, groups, slopes, intercepts, backend="numpy"
            )
            acceleration = _resolve_acceleration(method, "auto")
            all_kept = dropped == 0 and bool(np.all(mask))
            w_active = w0 if (w0 is not None and all_kept) else (None if w0 is None else w0[mask])
            winfo = prepare_weights(w_active, int(np.sum(mask)), wkind)
            absorber = _make_standard_absorber(
                groups, slopes, intercepts,
                weights=winfo.estimation,
                tol=self.hdfe.tolerance,
                max_iter=self.hdfe.max_iter,
                backend="numpy",
                method=method,
                transform="symmetric",
                acceleration=acceleration,
                preconditioner="diagonal",
                projection_backend="auto",
                absorb_threads=self.execution.threads,
                memory_budget_mb=self.execution.memory_budget_mb,
            )
            clusters = _cluster_arrays(self.data, self.cluster_spec, mask)
            dof = absorbed_dof(
                inference_fe.groups, slopes=inference_fe.slopes, intercepts=inference_fe.intercepts,
                clusters=clusters, method=self.hdfe.dof_method, adjust_nested=bool(clusters),
                adjust_continuous=True, groups_are_dense=True,
            )
        else:
            groups, slopes, intercepts, fe_names = [], [], [], []
            fe_plan, inference_fe = None, None
            mask = np.ones(len(self.data), dtype=bool); dropped = 0
            method = "none"; solver_selection = {"requested": self.hdfe.solver, "resolved": "none", "reason": "no_fixed_effects"}
            winfo = prepare_weights(w0, len(self.data), wkind)
            absorber = None
            clusters = _cluster_arrays(self.data, self.cluster_spec, mask)
            dof = absorbed_dof([], clusters=clusters, method=self.hdfe.dof_method, groups_are_dense=True)
        entry = _FECacheEntry(
            absorb=absorb, mask=np.asarray(mask, dtype=bool), dropped=int(dropped),
            groups=groups, slopes=slopes, intercepts=intercepts, fe_names=fe_names,
            fe_plan=fe_plan, inference_fe=inference_fe, absorber=absorber,
            weights=winfo.estimation, winfo=winfo, clusters=clusters, dof=dof,
            solver_selection=solver_selection, pool_size="auto",
            memory_budget_mb=self.execution.memory_budget_mb,
            source_signature=source_sig,
        )
        self._entries[key] = entry
        return entry

    def _ensure_within(self, entry: _FECacheEntry, columns: Iterable[str]) -> None:
        cols = tuple(dict.fromkeys(map(str, columns)))
        missing = []
        entry_sig = self._persistent_entry_signature(entry)
        for name in cols:
            full = self._full_numeric(name)
            validate = self.execution.cache_validation == "signature"
            sig = self._raw_signatures.get(name) if validate else None
            if self._persistent_store is not None:
                sig = self._column_signature(name)
            if name in entry.within and (not validate or entry.variable_signatures.get(name) == sig):
                continue
            raw = full if bool(np.all(entry.mask)) else full[entry.mask]
            entry.raw_active[name] = raw
            if self._persistent_store is not None and entry_sig is not None and sig is not None:
                cached = self._persistent_store.load_within(entry_sig, name, sig, len(raw))
                if cached is not None:
                    entry.within[name] = cached
                    entry.variable_signatures[name] = sig
                    continue
            missing.append(name)
        if not missing:
            return
        block = np.column_stack([entry.raw_active[n] for n in missing])
        if entry.absorber is None:
            within, info = block.copy(), None
        else:
            within, info = _residualize_blocks(
                entry.absorber, block, entry.pool_size, entry.memory_budget_mb,
                adaptive_auto=(self.hdfe.solver == "auto"),
            )
            _check_absorption(info, allow_nonconverged=False)
        entry.last_info = info
        entry.residualize_calls += 1
        entry.residualized_columns += len(missing)
        for j, name in enumerate(missing):
            entry.within[name] = within[:, j].copy()
            sig = self._column_signature(name) if self._persistent_store is not None else self._raw_signatures.get(name)
            if self.execution.cache_validation == "signature" or self._persistent_store is not None:
                entry.variable_signatures[name] = sig
            if self._persistent_store is not None and entry_sig is not None and sig is not None:
                self._persistent_store.save_within(entry_sig, name, sig, entry.within[name])

    @error_boundary("session")
    def prewarm(self, *, absorb, columns: Sequence[str]) -> None:
        self._ensure_source_specs([{"x": tuple(columns), "absorb": absorb}])
        self._start_validation_round()
        entry = self._get_entry(absorb)
        self._ensure_within(entry, columns)


class OLSHDFESession(_LinearHDFESessionBase):
    """Repeated OLS-HDFE specifications with exact FE/sample-aware caching."""

    @error_boundary("session")
    def fit(self, *, y: str, x=(), absorb=()) -> RegressionResult:
        spec = OLSSpec(str(y), tuple(map(str, _as_tuple(x))), _normalize_absorb(absorb))
        self._ensure_source_specs([{"y": spec.y, "x": spec.x, "absorb": spec.absorb}])
        self._start_validation_round()
        return self._fit_spec(spec)

    @error_boundary("session")
    def fit_many_y(self, *, y: Sequence[str], x=(), absorb=()) -> list[RegressionResult]:
        specs = [OLSSpec(str(v), tuple(map(str, _as_tuple(x))), _normalize_absorb(absorb)) for v in y]
        return self.fit_many(specs)

    @error_boundary("session")
    def fit_many(self, specs: Sequence[OLSSpec | Mapping]) -> list[RegressionResult]:
        parsed = [OLSSpec.from_any(s) for s in specs]
        self._ensure_source_specs([{"y": s.y, "x": s.x, "absorb": s.absorb} for s in parsed])
        self._start_validation_round()
        cached: dict[int, RegressionResult] = {}
        if self._persistent_store is not None and self.execution.profile == "off":
            for i, spec in enumerate(parsed):
                key = self._persistent_result_key("ols", spec)
                if key is not None:
                    hit = self._persistent_store.load_result(key)
                    if hit is not None:
                        cached[i] = hit
        grouped: dict[tuple, list[OLSSpec]] = defaultdict(list)
        for i, s in enumerate(parsed):
            if i not in cached:
                grouped[s.absorb].append(s)
        # One multi-RHS within transform per FE specification for all newly
        # requested outcomes/regressors. This is the main batch speed path.
        for absorb in sorted(grouped, key=len):
            group = grouped[absorb]
            entry = self._get_entry(absorb)
            cols: list[str] = []
            for s in group:
                cols.append(s.y); cols.extend(s.x)
            self._ensure_within(entry, cols)
        out = []
        for i, s in enumerate(parsed):
            out.append(cached[i] if i in cached else self._fit_spec(s, prewarmed=True))
        return out

    def _fit_spec(self, spec: OLSSpec, *, prewarmed=False) -> RegressionResult:
        t0 = perf_counter()
        result_key = self._persistent_result_key("ols", spec)
        if result_key is not None:
            cached = self._persistent_store.load_result(result_key)
            if cached is not None:
                return cached
        entry = self._get_entry(spec.absorb)
        if not prewarmed:
            self._ensure_within(entry, (spec.y, *spec.x))
        yw = entry.within[spec.y]
        y1 = entry.raw_active[spec.y]
        if spec.x:
            Xw0 = np.column_stack([entry.within[n] for n in spec.x])
            X0 = np.column_stack([entry.raw_active[n] for n in spec.x])
        else:
            Xw0 = np.empty((len(yw), 0), dtype=np.float64)
            X0 = Xw0.copy()
        Xw, _, collin_plan = _resolve_ols_columns(
            Xw0, X0, spec.x, weights=entry.weights,
            tolerance=self.collinear_tol, mode=self.collinearity,
        )
        vce_kind = resolve_vce_for_weights(
            self.inference.vce if self.inference.vce is not None else self.vce,
            entry.winfo, has_clusters=bool(entry.clusters),
        )
        beta, V, resid, rank, df_resid = _fit_ols(
            yw, Xw, weights=entry.weights, weight_info=entry.winfo, vce=vce_kind,
            clusters=entry.clusters, df_absorbed=entry.dof.df_absorbed,
            nested_adj=int(entry.dof.nested > 0),
        )
        fit_stats = reghdfe_model_statistics(
            y1, yw, resid, beta, V, weights=entry.weights, effective_n=entry.winfo.effective_n,
            rank=rank, df_absorbed=entry.dof.df_absorbed, df_nested=entry.dof.nested,
            df_resid_inference=df_resid, has_intercept=bool(any(entry.intercepts)),
        )
        info = entry.last_info
        absorb_info = None
        if info is not None:
            absorb_info = _absorb_metadata(
                entry.fe_plan, entry.absorber, info,
                _resolve_pool_size(entry.absorber, max(len(spec.x) + 1, 1), "auto", entry.memory_budget_mb),
                solver_selection=entry.solver_selection,
            )
        sample_history = (
            SampleExclusionState("hdfe", "singleton_pruning", entry.dropped, int(np.sum(entry.mask))),
        ) if entry.dropped else ()
        prediction_state = linear_prediction_state(
            estimator="ols",
            coefficient_names=collin_plan.active_names,
            coefficient_roles=("regressor",),
            designs=(
                named_design_state(
                    spec.x, role="regressor", active_indices=collin_plan.active_indices,
                ),
            ),
            fe_plan=entry.fe_plan,
            fe_names=entry.fe_names,
            requested_fe_groups=() if entry.inference_fe is None else entry.inference_fe.groups,
            effective_fe_groups=entry.groups,
            fe_intercepts=entry.intercepts,
            fe_slopes=entry.slopes,
            sample_mask=entry.mask,
            sample_history=sample_history,
        )
        result = RegressionResult(
            params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
            residuals=resid, fitted=y1-resid, nobs=int(round(entry.winfo.effective_n)), rank=rank,
            df_resid=df_resid, df_absorbed=entry.dof.df_absorbed,
            converged=True if info is None else bool(info.converged),
            iterations=0 if info is None else int(info.iterations),
            dropped_singletons=entry.dropped, names=tuple(collin_plan.active_names), estimator="ols",
            dof_info=entry.dof, weight_type=entry.winfo.kind, sum_weights=entry.winfo.sum_weights,
            nobs_raw=len(yw), absorb_info=absorb_info,
            collinearity_info=_collinearity_metadata(collin_plan, role="regressor"),
            vce=vce_kind, cluster_counts=_cluster_counts(entry.clusters), fe_names=tuple(entry.fe_names),
            r2=fit_stats["r2"], r2_within=fit_stats["r2_within"],
            r2_adjusted=fit_stats["r2_adjusted"], r2_adjusted_within=fit_stats["r2_adjusted_within"],
            rss=fit_stats["rss"], tss=fit_stats["tss"], tss_within=fit_stats["tss_within"],
            mss=fit_stats["mss"], rmse=fit_stats["rmse"], loglike=fit_stats["loglike"],
            loglike_null=fit_stats["loglike_null"], f_statistic=fit_stats["f_statistic"],
            f_pvalue=fit_stats["f_pvalue"], df_model=fit_stats["df_model"],
            df_resid_fit=fit_stats["df_resid_fit"], vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
            confidence_level=self.inference.confidence_level,
            diagnostics_mode=self.inference.diagnostics,
            prediction_state=prediction_state,
        )
        if self.execution.profile != "off":
            result.profile = {
                "total_seconds": perf_counter() - t0,
                "mode": self.execution.profile,
                "repeated_spec": True,
                "fe_cache_entries": len(self._entries),
                "within_cache_columns": len(entry.within),
            }
        if result_key is not None:
            self._persistent_store.save_result(result_key, result)
        return result


class IVHDFESession(_LinearHDFESessionBase):
    """Repeated linear-IV HDFE specifications with FE/within-column caching."""

    def __init__(self, *args, estimator="2sls", kappa=None, fuller=0.0, gmm_center=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = str(estimator).lower()
        self.kappa = kappa
        self.fuller = float(fuller)
        self.gmm_center = bool(gmm_center)

    @error_boundary("session")
    def fit(self, *, y: str, exog=(), endog=(), instruments=(), absorb=()) -> RegressionResult:
        spec = IVSpec(
            str(y), tuple(map(str, _as_tuple(exog))), tuple(map(str, _as_tuple(endog))),
            tuple(map(str, _as_tuple(instruments))), _normalize_absorb(absorb),
        )
        self._ensure_source_specs([{
            "y": spec.y, "exog": spec.exog, "endog": spec.endog,
            "instruments": spec.instruments, "absorb": spec.absorb,
        }])
        self._start_validation_round()
        return self._fit_spec(spec)

    @error_boundary("session")
    def fit_many(self, specs: Sequence[IVSpec | Mapping]) -> list[RegressionResult]:
        parsed = [IVSpec.from_any(s) for s in specs]
        self._ensure_source_specs([{
            "y": s.y, "exog": s.exog, "endog": s.endog,
            "instruments": s.instruments, "absorb": s.absorb,
        } for s in parsed])
        self._start_validation_round()
        cached: dict[int, RegressionResult] = {}
        if self._persistent_store is not None and self.execution.profile == "off":
            for i, spec in enumerate(parsed):
                key = self._persistent_result_key(
                    "linear_iv", spec,
                    extra={
                        "estimator": self.estimator,
                        "kappa": self.kappa,
                        "fuller": self.fuller,
                        "gmm_center": self.gmm_center,
                    },
                )
                if key is not None:
                    hit = self._persistent_store.load_result(key)
                    if hit is not None:
                        cached[i] = hit
        grouped: dict[tuple, list[IVSpec]] = defaultdict(list)
        for i, s in enumerate(parsed):
            if i not in cached:
                grouped[s.absorb].append(s)
        for absorb in sorted(grouped, key=len):
            group = grouped[absorb]
            entry = self._get_entry(absorb)
            cols: list[str] = []
            for s in group:
                cols.append(s.y); cols.extend(s.exog); cols.extend(s.endog); cols.extend(s.instruments)
            self._ensure_within(entry, cols)
        out = []
        for i, s in enumerate(parsed):
            out.append(cached[i] if i in cached else self._fit_spec(s, prewarmed=True))
        return out

    def _fit_spec(self, spec: IVSpec, *, prewarmed=False) -> RegressionResult:
        if not spec.endog or not spec.instruments:
            raise SpecificationError("IVHDFESession requires at least one endogenous regressor and excluded instrument")
        t0 = perf_counter()
        result_key = self._persistent_result_key(
            "linear_iv", spec,
            extra={
                "estimator": self.estimator,
                "kappa": self.kappa,
                "fuller": self.fuller,
                "gmm_center": self.gmm_center,
            },
        )
        if result_key is not None:
            cached = self._persistent_store.load_result(result_key)
            if cached is not None:
                return cached
        entry = self._get_entry(spec.absorb)
        cols = (spec.y, *spec.exog, *spec.endog, *spec.instruments)
        if not prewarmed:
            self._ensure_within(entry, cols)
        yw = entry.within[spec.y]
        y1 = entry.raw_active[spec.y]

        def stack(names):
            if not names:
                return np.empty((len(yw), 0), dtype=np.float64), np.empty((len(yw), 0), dtype=np.float64)
            return (
                np.column_stack([entry.within[n] for n in names]),
                np.column_stack([entry.raw_active[n] for n in names]),
            )

        Cw0, C0 = stack(spec.exog); Ew0, E0 = stack(spec.endog); Zw0, Z0 = stack(spec.instruments)
        Cw, Ew, Zw, _, _, _, cn, en, _zn, collin_info = _resolve_iv_columns(
            Cw0, Ew0, Zw0, C0, E0, Z0, spec.exog, spec.endog, spec.instruments,
            weights=entry.weights, tolerance=self.collinear_tol, mode=self.collinearity,
        )
        vce_kind = resolve_vce_for_weights(
            self.inference.vce if self.inference.vce is not None else self.vce,
            entry.winfo, has_clusters=bool(entry.clusters),
        )
        if self.estimator == "gmm2s":
            beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_gmm2s(
                yw, Cw, Ew, Zw, weights=entry.weights, weight_info=entry.winfo,
                vce=vce_kind, clusters=entry.clusters, df_absorbed=entry.dof.df_absorbed,
                nested_adj=int(entry.dof.nested > 0), center=self.gmm_center,
            )
        else:
            beta, V, resid, rank, df_resid, first, diagnostics, meta = fit_iv_kclass(
                yw, Cw, Ew, Zw, weights=entry.weights, weight_info=entry.winfo,
                vce=vce_kind, clusters=entry.clusters, df_absorbed=entry.dof.df_absorbed,
                nested_adj=int(entry.dof.nested > 0), estimator=self.estimator,
                kappa=self.kappa, fuller=self.fuller,
            )
        r2_stats = reghdfe_r2_statistics(
            y1, yw, resid, weights=entry.weights, effective_n=entry.winfo.effective_n,
            rank=rank, df_absorbed=entry.dof.df_absorbed, df_nested=entry.dof.nested,
            has_intercept=bool(any(entry.intercepts)),
        )
        fit_stats = ivreghdfe_model_statistics(
            yw, resid, beta, V, weights=entry.weights, effective_n=entry.winfo.effective_n,
            rank=rank, df_absorbed=entry.dof.df_absorbed, df_nested=entry.dof.nested,
            df_resid_inference=df_resid,
        )
        info = entry.last_info
        absorb_info = None
        if info is not None:
            absorb_info = _absorb_metadata(
                entry.fe_plan, entry.absorber, info,
                _resolve_pool_size(entry.absorber, max(len(cols), 1), "auto", entry.memory_budget_mb),
                solver_selection=entry.solver_selection,
            )
        sample_history = (
            SampleExclusionState("hdfe", "singleton_pruning", entry.dropped, int(np.sum(entry.mask))),
        ) if entry.dropped else ()
        prediction_state = linear_prediction_state(
            estimator=meta.get("estimator", self.estimator),
            coefficient_names=tuple(cn + en),
            coefficient_roles=("exogenous", "endogenous"),
            designs=(
                named_design_state(
                    spec.exog, role="exogenous",
                    active_indices=collin_info["exogenous"]["active_indices"],
                ),
                named_design_state(
                    spec.endog, role="endogenous",
                    active_indices=collin_info["endogenous"]["active_indices"],
                ),
                named_design_state(
                    spec.instruments, role="excluded_instrument",
                    active_indices=collin_info["excluded_instruments"]["active_indices"],
                ),
            ),
            fe_plan=entry.fe_plan,
            fe_names=entry.fe_names,
            requested_fe_groups=() if entry.inference_fe is None else entry.inference_fe.groups,
            effective_fe_groups=entry.groups,
            fe_intercepts=entry.intercepts,
            fe_slopes=entry.slopes,
            sample_mask=entry.mask,
            sample_history=sample_history,
        )
        result = RegressionResult(
            params=beta, vcov=V, stderr=np.sqrt(np.clip(np.diag(V), 0, None)),
            residuals=resid, fitted=y1-resid, nobs=int(round(entry.winfo.effective_n)), rank=rank,
            df_resid=df_resid, df_absorbed=entry.dof.df_absorbed,
            converged=True if info is None else bool(info.converged),
            iterations=0 if info is None else int(info.iterations), dropped_singletons=entry.dropped,
            names=tuple(cn + en), first_stage=first, diagnostics=diagnostics,
            estimator=meta.get("estimator", self.estimator), estimator_info=meta,
            dof_info=entry.dof, weight_type=entry.winfo.kind, sum_weights=entry.winfo.sum_weights,
            nobs_raw=len(yw), absorb_info=absorb_info, collinearity_info=collin_info,
            vce=vce_kind, cluster_counts=_cluster_counts(entry.clusters), fe_names=tuple(entry.fe_names),
            r2=r2_stats["r2"], r2_within=r2_stats["r2_within"],
            r2_adjusted=r2_stats["r2_adjusted"], r2_adjusted_within=r2_stats["r2_adjusted_within"],
            rss=fit_stats["rss"], tss_within=fit_stats["tss_within"], rmse=fit_stats["rmse"],
            f_statistic=fit_stats["f_statistic"], f_pvalue=fit_stats["f_pvalue"],
            df_model=fit_stats["df_model"], df_resid_fit=fit_stats["df_resid_fit"], vcov_rank=(int(np.linalg.matrix_rank(V)) if V.size else 0),
            confidence_level=self.inference.confidence_level,
            diagnostics_mode=self.inference.diagnostics,
            prediction_state=prediction_state,
        )
        if self.execution.profile != "off":
            result.profile = {
                "total_seconds": perf_counter() - t0,
                "mode": self.execution.profile,
                "repeated_spec": True,
                "fe_cache_entries": len(self._entries),
                "within_cache_columns": len(entry.within),
            }
        if result_key is not None:
            self._persistent_store.save_result(result_key, result)
        return result
