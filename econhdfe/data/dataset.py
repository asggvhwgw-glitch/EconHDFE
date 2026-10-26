from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from collections.abc import Mapping, Sequence
import hashlib
import numpy as np
import pandas as pd

from .source import DataSource, DataFrameSource, as_data_source
from .planner import DataIngestionPlan, estimate_frame_bytes_per_row
from .encoding import StableCategoricalEncoder
from ..errors import DataTypeError, MissingDataError, SpecificationError
from ..frontend.columns import compile_data_requirements, merge_data_requirements


def _array_signature(name: str, values: np.ndarray) -> str:
    """Stable one-time content signature for an immutable dataset column."""
    arr = np.asarray(values)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(name).encode())
    h.update(str(arr.dtype).encode())
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.dtype.kind in "biufcMm" and arr.flags.c_contiguous:
        h.update(arr.view(np.uint8))
    else:
        # pandas' hash handles object/string/categorical payloads without
        # serializing repr() values into diagnostics or logs.
        hv = pd.util.hash_array(arr, categorize=True)
        h.update(np.ascontiguousarray(hv, dtype=np.uint64).view(np.uint8))
    return h.hexdigest()


def _validity(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in "fc":
        return np.isfinite(arr)
    if arr.dtype.kind in "biuMm":
        return ~pd.isna(arr)
    return ~pd.isna(arr)


@dataclass(slots=True)
class MaterializedData:
    frame: pd.DataFrame
    plan: DataIngestionPlan
    materialization_seconds: float = 0.0
    materialized_bytes: int = 0
    identifier_levels: dict[str, tuple[object, ...]] = field(default_factory=dict)

    @property
    def nobs(self) -> int:
        return len(self.frame)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(map(str, self.frame.columns))

    def as_dict(self, *, include_sensitive: bool = False) -> dict:
        out = self.plan.as_dict(include_sensitive=include_sensitive)
        out.update({
            "nobs_materialized": self.nobs,
            "materialized_bytes": int(self.materialized_bytes),
            "materialization_seconds": float(self.materialization_seconds),
            "encoded_identifier_count": len(self.identifier_levels),
        })
        return out


class EncodedEconometricDataset:
    """Immutable econometric column store for repeated specifications.

    The object snapshots only the columns retained for estimation, stores
    identifier-only categoricals as compact integer codes, computes validity
    and content signatures once, and lazily caches float64 numeric views.
    Estimators may still consume ``frame`` for compatibility; repeated-session
    code can use direct arrays/signatures without rescanning pandas columns.
    """

    __slots__ = (
        "_columns", "_names", "_nobs", "_identifier_levels", "_signatures",
        "_valid_packed", "_numeric", "_frame", "_sample_masks", "_fingerprint",
        "_source_metadata", "_encoded_bytes", "_invalid_counts",
    )

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        *,
        identifier_levels: dict[str, tuple[object, ...]] | None = None,
        source_metadata: dict | None = None,
    ):
        if not columns:
            raise SpecificationError(
                "EncodedEconometricDataset requires at least one column",
                code="data.empty_dataset", stage="data",
            )
        names = tuple(map(str, columns))
        normalized: dict[str, np.ndarray] = {}
        nobs = None
        signatures: dict[str, str] = {}
        valid_packed: dict[str, np.ndarray] = {}
        invalid_counts: dict[str, int] = {}
        encoded_bytes = 0
        for name in names:
            arr = np.asarray(columns[name])
            if arr.ndim != 1:
                raise SpecificationError(
                    "encoded dataset columns must be one-dimensional",
                    code="data.column_shape", stage="data",
                    details={"column": name, "shape": tuple(arr.shape)},
                )
            if nobs is None:
                nobs = int(arr.size)
            elif int(arr.size) != nobs:
                raise SpecificationError(
                    "encoded dataset columns must have equal length",
                    code="data.column_length", stage="data",
                    details={"column": name, "expected": nobs, "actual": int(arr.size)},
                )
            # Snapshot mutable caller-owned arrays once. Object/string columns
            # must also be copied because their pointer array is mutable.
            owned = np.array(arr, copy=True, order="C")
            normalized[name] = owned
            signatures[name] = _array_signature(name, owned)
            valid = np.asarray(_validity(owned), dtype=bool)
            invalid_counts[name] = int(valid.size - np.count_nonzero(valid))
            valid_packed[name] = np.packbits(valid.astype(np.uint8, copy=False), bitorder="little")
            if owned.dtype.kind in {"O", "U", "S"}:
                payload = int(pd.Series(owned, copy=False).memory_usage(index=False, deep=True))
            else:
                payload = int(owned.nbytes)
            encoded_bytes += int(payload + valid_packed[name].nbytes)
            owned.flags.writeable = False

        self._columns = normalized
        self._names = names
        self._nobs = int(nobs or 0)
        self._identifier_levels = dict(identifier_levels or {})
        self._signatures = signatures
        self._valid_packed = valid_packed
        self._numeric: dict[str, np.ndarray] = {}
        self._frame = None
        self._sample_masks: dict[tuple[str, ...], np.ndarray] = {}
        source = dict(source_metadata or {})
        # Keep potentially sensitive source labels out of ordinary metadata.
        self._source_metadata = source
        self._encoded_bytes = int(encoded_bytes)
        self._invalid_counts = invalid_counts
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray([self._nobs], dtype=np.int64).tobytes())
        for name in self._names:
            h.update(name.encode()); h.update(signatures[name].encode())
        self._fingerprint = h.hexdigest()

    @classmethod
    def _from_persistent_snapshot(
        cls,
        columns: dict[str, np.ndarray],
        *,
        signatures: dict[str, str],
        valid_packed: dict[str, np.ndarray],
        invalid_counts: dict[str, int],
        identifier_levels: dict[str, tuple[object, ...]] | None = None,
        source_metadata: dict | None = None,
    ) -> "EncodedEconometricDataset":
        """Restore a trusted immutable snapshot without rescanning N rows.

        This is intentionally private to the data layer.  Persistent-cache
        loaders validate their manifest and array shapes/dtypes before calling
        it.  The fast path avoids recomputing column hashes and validity masks
        every time a research process is restarted.
        """
        if not columns:
            raise SpecificationError(
                "persistent encoded dataset requires at least one column",
                code="data.empty_dataset", stage="data",
            )
        names = tuple(map(str, columns))
        nobs = None
        normalized: dict[str, np.ndarray] = {}
        encoded_bytes = 0
        for name in names:
            arr = np.asarray(columns[name])
            if arr.ndim != 1:
                raise SpecificationError(
                    "persistent encoded dataset columns must be one-dimensional",
                    code="data.column_shape", stage="data",
                    details={"column": name, "shape": tuple(arr.shape)},
                )
            if nobs is None:
                nobs = int(arr.size)
            elif int(arr.size) != nobs:
                raise SpecificationError(
                    "persistent encoded dataset columns must have equal length",
                    code="data.column_length", stage="data",
                    details={"column": name, "expected": nobs, "actual": int(arr.size)},
                )
            if name not in signatures or name not in valid_packed or name not in invalid_counts:
                raise SpecificationError(
                    "persistent encoded dataset metadata is incomplete",
                    code="data.cache_manifest", stage="data",
                    details={"column": name},
                )
            packed = np.asarray(valid_packed[name], dtype=np.uint8)
            need = (int(arr.size) + 7) // 8
            if packed.ndim != 1 or packed.size != need:
                raise SpecificationError(
                    "persistent validity bitmap has wrong size",
                    code="data.cache_manifest", stage="data",
                    details={"column": name, "expected_bytes": need, "actual_bytes": int(packed.size)},
                )
            arr.flags.writeable = False
            packed.flags.writeable = False
            normalized[name] = arr
            encoded_bytes += int(arr.nbytes + packed.nbytes)

        obj = cls.__new__(cls)
        obj._columns = normalized
        obj._names = names
        obj._nobs = int(nobs or 0)
        obj._identifier_levels = dict(identifier_levels or {})
        obj._signatures = {name: str(signatures[name]) for name in names}
        obj._valid_packed = {name: np.asarray(valid_packed[name], dtype=np.uint8) for name in names}
        obj._numeric = {}
        obj._frame = None
        obj._sample_masks = {}
        obj._source_metadata = dict(source_metadata or {})
        obj._encoded_bytes = int(encoded_bytes)
        obj._invalid_counts = {name: int(invalid_counts[name]) for name in names}
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray([obj._nobs], dtype=np.int64).tobytes())
        for name in obj._names:
            h.update(name.encode()); h.update(obj._signatures[name].encode())
        obj._fingerprint = h.hexdigest()
        return obj

    def _persistent_column_snapshot(self, name: str) -> dict:
        """Return immutable metadata used by the local persistent cache."""
        key = str(name)
        if key not in self._columns:
            raise SpecificationError(
                f"column {name!r} is not present",
                code="data.missing_column", stage="data",
            )
        return {
            "values": self._columns[key],
            "signature": self._signatures[key],
            "valid_packed": self._valid_packed[key],
            "invalid_count": int(self._invalid_counts[key]),
            "identifier_levels": self._identifier_levels.get(key),
        }

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        identifier_columns: Sequence[str] = (),
        source_metadata: dict | None = None,
    ) -> "EncodedEconometricDataset":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("from_frame requires a pandas DataFrame")
        names = tuple(map(str, frame.columns if columns is None else columns))
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise SpecificationError(
                "requested encoded-dataset columns are not present",
                code="data.missing_column", stage="data", details={"missing": missing},
            )
        identifier = set(map(str, identifier_columns))
        arrays: dict[str, np.ndarray] = {}
        levels: dict[str, tuple[object, ...]] = {}
        for name in names:
            raw = frame[name].to_numpy(copy=False)
            if name in identifier:
                enc = StableCategoricalEncoder()
                batch = enc.transform(raw)
                arrays[name] = batch.codes
                levels[name] = enc.levels
            else:
                arrays[name] = raw
        return cls(arrays, identifier_levels=levels, source_metadata=source_metadata)

    @classmethod
    def from_source(
        cls,
        source,
        required_columns: Sequence[str],
        *,
        identifier_columns: Sequence[str] = (),
        memory_budget_mb: float = 512,
        batch_fraction: float = 0.20,
    ) -> tuple["EncodedEconometricDataset", MaterializedData]:
        materialized = materialize_required_data(
            as_data_source(source), required_columns,
            memory_budget_mb=memory_budget_mb, batch_fraction=batch_fraction,
            identifier_columns=identifier_columns,
        )
        dataset = cls.from_frame(
            materialized.frame,
            identifier_columns=(),  # already encoded by materialize_required_data
            source_metadata=materialized.as_dict(include_sensitive=False),
        )
        dataset._identifier_levels.update(materialized.identifier_levels)
        return dataset, materialized

    def __len__(self) -> int:
        return self._nobs

    @property
    def columns(self) -> tuple[str, ...]:
        return self._names

    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def encoded_bytes(self) -> int:
        extra = 0
        for name, arr in self._numeric.items():
            base = self._columns[name]
            if not np.shares_memory(arr, base):
                extra += int(arr.nbytes)
        return int(self._encoded_bytes + extra)

    @property
    def identifier_levels(self) -> dict[str, tuple[object, ...]]:
        return dict(self._identifier_levels)

    @property
    def source_metadata(self) -> dict:
        return dict(self._source_metadata)

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            # copy=False allows pandas to reuse numeric backing arrays where
            # possible; this frame is an estimator compatibility view.
            self._frame = pd.DataFrame(self._columns, copy=False)
        return self._frame

    def __getitem__(self, name):
        if isinstance(name, str):
            return self._columns[name]
        names = list(map(str, name))
        return self.frame.loc[:, names]

    def column(self, name: str) -> np.ndarray:
        try:
            return self._columns[str(name)]
        except KeyError as exc:
            raise SpecificationError(
                f"column {name!r} is not present",
                code="data.missing_column", stage="data",
            ) from exc

    def column_signature(self, name: str) -> str:
        try:
            return self._signatures[str(name)]
        except KeyError as exc:
            raise SpecificationError(
                f"column {name!r} is not present",
                code="data.missing_column", stage="data",
            ) from exc

    def valid_mask(self, name: str) -> np.ndarray:
        key = str(name)
        try:
            packed = self._valid_packed[key]
        except KeyError as exc:
            raise SpecificationError(
                f"column {name!r} is not present",
                code="data.missing_column", stage="data",
            ) from exc
        out = np.unpackbits(packed, count=self._nobs, bitorder="little").astype(bool, copy=False)
        out.flags.writeable = False
        return out

    def sample_mask(self, columns: Sequence[str]) -> np.ndarray:
        key = tuple(sorted(dict.fromkeys(map(str, columns))))
        hit = self._sample_masks.get(key)
        if hit is not None:
            return hit
        mask = np.ones(self._nobs, dtype=bool)
        for name in key:
            mask &= self.valid_mask(name)
        mask.flags.writeable = False
        self._sample_masks[key] = mask
        return mask

    def numeric_column(self, name: str, *, finite: bool = True) -> np.ndarray:
        key = str(name)
        hit = self._numeric.get(key)
        if hit is not None:
            if finite and self._invalid_counts.get(key, 0):
                bad = int(self._invalid_counts[key])
                raise MissingDataError(
                    f"numeric column {key!r} contains {bad} missing or non-finite value(s)",
                    code="input.non_finite", stage="data",
                    details={"variable": key, "count": bad},
                )
            return hit
        raw = self.column(key)
        if raw.dtype.kind not in "biufc":
            raise DataTypeError(
                f"numeric column {key!r} must be numeric; got dtype {raw.dtype}",
                code="input.non_numeric", stage="data",
                details={"variable": key, "dtype": str(raw.dtype)},
            )
        if finite and self._invalid_counts.get(key, 0):
            bad = int(self._invalid_counts[key])
            raise MissingDataError(
                f"numeric column {key!r} contains {bad} missing or non-finite value(s)",
                code="input.non_finite", stage="data",
                details={"variable": key, "count": bad},
            )
        out = np.asarray(raw, dtype=np.float64)
        if not out.flags.c_contiguous or out.flags.writeable:
            out = np.array(out, dtype=np.float64, copy=True, order="C")
        out.flags.writeable = False
        self._numeric[key] = out
        return out

    def as_dict(self) -> dict:
        return {
            "nobs": self._nobs,
            "column_count": len(self._names),
            "encoded_identifier_count": len(self._identifier_levels),
            "numeric_cache_columns": len(self._numeric),
            "sample_mask_cache_entries": len(self._sample_masks),
            "encoded_bytes": self.encoded_bytes,
            "fingerprint": self._fingerprint,
            "source_type": self._source_metadata.get("source_type"),
        }


def materialize_required_data(
    source: DataSource,
    required_columns: Sequence[str],
    *,
    memory_budget_mb: float = 512,
    batch_fraction: float = 0.20,
    identifier_columns: Sequence[str] = (),
) -> MaterializedData:
    src = as_data_source(source)
    cols = tuple(dict.fromkeys(map(str, required_columns)))
    # A materializing caller ultimately needs the full projected frame.  Do not
    # perform a separate preview pass merely to choose a chunk size; that is
    # valuable for scan/out-of-core execution but can dominate fast binary
    # formats such as Stata.
    from ..compute.execution import _check_allocation_headroom
    from ..planner import plan_memory

    budget = plan_memory(
        requested_budget_bytes=max(int(float(memory_budget_mb) * 1024**2), 1),
    ).effective_budget_bytes
    # Only inspect already resident columns; unknown-size file sources retain
    # their single-pass fast path. This is admission, not a hard RSS guarantee.
    if isinstance(src, DataFrameSource):
        projected_bytes = sum(
            int(src.frame[name].memory_usage(index=False, deep=True)) for name in cols
        )
        _check_allocation_headroom(projected_bytes)
    t0 = perf_counter()
    frame = src.materialize(cols)
    levels: dict[str, tuple[object, ...]] = {}
    idcols = tuple(dict.fromkeys(map(str, identifier_columns)))
    if idcols:
        missing_ids = [c for c in idcols if c not in frame.columns]
        if missing_ids:
            raise KeyError(f"identifier columns not present after projection: {missing_ids}")
        # Never mutate an in-memory source frame through a projected view.
        # Encoding retains the original column while creating codes and
        # factorization scratch; account for this known-size temporary work.
        _check_allocation_headroom(len(frame) * 24)
        frame = frame.copy(deep=False)
        for name in idcols:
            enc = StableCategoricalEncoder()
            encoded = enc.transform(frame[name].to_numpy())
            frame[name] = encoded.codes
            levels[name] = enc.levels
    elapsed = perf_counter() - t0
    nbytes = int(frame.memory_usage(index=False, deep=True).sum())
    bpr = estimate_frame_bytes_per_row(frame)
    target = max(int(budget * min(max(float(batch_fraction), 0.01), 0.80)), 1)
    plan = DataIngestionPlan(
        required_columns=cols, memory_budget_bytes=budget, target_batch_bytes=target,
        estimated_bytes_per_row=bpr, batch_rows=max(len(frame), 1),
        source_label=src.label, source_type=type(src).__name__, strategy="materialize",
    )
    return MaterializedData(
        frame=frame, plan=plan, materialization_seconds=elapsed, materialized_bytes=nbytes,
        identifier_levels=levels,
    )


def materialize_model_data(
    data,
    *,
    memory_budget_mb: float = 512,
    batch_fraction: float = 0.20,
    **role_specs,
):
    """Materialize only raw columns referenced by a model specification.

    Existing pandas DataFrames, mapping-backed column stores, and encoded datasets are returned unchanged.
    File/DataSource inputs are column-projected before parsing/materialization.
    """
    if isinstance(data, EncodedEconometricDataset):
        return data.frame, None
    if isinstance(data, (pd.DataFrame, Mapping)) or data is None:
        # Direct estimators historically accept dict/Mapping column stores via
        # the shared _col() accessor.  The Data Layer must preserve that public
        # compatibility rather than forcing mapping inputs through file/source
        # materialization.
        return data, None
    requirements = compile_data_requirements(**role_specs)
    src = as_data_source(data)
    materialized = materialize_required_data(
        src, requirements.columns, memory_budget_mb=memory_budget_mb, batch_fraction=batch_fraction,
        identifier_columns=requirements.identifier_only_columns,
    )
    return materialized.frame, materialized


def prepare_encoded_dataset(
    data,
    *,
    required_columns: Sequence[str] | None = None,
    identifier_columns: Sequence[str] = (),
    memory_budget_mb: float = 512,
    batch_fraction: float = 0.20,
) -> EncodedEconometricDataset:
    """Create one immutable reusable dataset snapshot for repeated estimation."""
    if isinstance(data, EncodedEconometricDataset):
        return data
    if isinstance(data, pd.DataFrame):
        return EncodedEconometricDataset.from_frame(
            data, columns=required_columns, identifier_columns=identifier_columns,
            source_metadata={"source_type": "DataFrameSource"},
        )
    if required_columns is None:
        raise SpecificationError(
            "file/DataSource inputs require required_columns when preparing an encoded dataset",
            code="data.required_columns", stage="data",
            suggestion="Compile the repeated specifications first and pass the union of required raw columns.",
        )
    dataset, _ = EncodedEconometricDataset.from_source(
        data, required_columns, identifier_columns=identifier_columns,
        memory_budget_mb=memory_budget_mb, batch_fraction=batch_fraction,
    )
    return dataset


def prepare_repeated_dataset(
    data,
    specifications: Sequence[dict],
    *,
    common_roles: dict | None = None,
    memory_budget_mb: float = 512,
    batch_fraction: float = 0.20,
) -> EncodedEconometricDataset:
    """Compile a union column store for a repeated-estimation workflow.

    ``specifications`` contains estimator-agnostic role mappings such as
    ``{"y": "sales", "x": ["price"], "absorb": ["firm", "year"]}``.
    Roles shared by every specification (weights, clusters, etc.) may be placed
    in ``common_roles``.  The union is resolved before file materialization, so
    wide sources parse only columns used anywhere in the workflow.
    """
    common = dict(common_roles or {})
    reqs = []
    for spec in specifications:
        merged = dict(common)
        merged.update(dict(spec))
        reqs.append(compile_data_requirements(**merged))
    if not reqs:
        if isinstance(data, EncodedEconometricDataset):
            return data
        raise SpecificationError(
            "prepare_repeated_dataset requires at least one specification",
            code="data.empty_specifications", stage="data",
        )
    req = merge_data_requirements(*reqs)
    return prepare_encoded_dataset(
        data, required_columns=req.columns, identifier_columns=req.identifier_only_columns,
        memory_budget_mb=memory_budget_mb, batch_fraction=batch_fraction,
    )
