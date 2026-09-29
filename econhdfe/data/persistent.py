from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import SpecificationError
from ..results import DofInfo, RegressionResult, FixedEffectEstimates, FixedEffectTermEstimate
from ..prediction import (
    CategoricalEncodingState, DesignPredictionState, DesignTermState,
    DroppedFixedEffectState, EstimationSampleSnapshot, FixedEffectPredictionState,
    PredictionInput, PredictionState, SampleExclusionState,
    FixedEffectLevelState, FixedEffectNestingState, CategoricalFixedEffectState,
)
from ..frontend.columns import compile_data_requirements, merge_data_requirements
from .dataset import EncodedEconometricDataset, materialize_required_data
from .source import DataFrameSource, DataSource, as_data_source


PERSISTENT_CACHE_FORMAT = 1
PERSISTENT_NUMERICAL_ABI = "linear-session-1"
PERSISTENT_RESULT_ABI = "linear-session-result-3"

_PREDICTION_STATE_TYPES = {
    cls.__name__: cls
    for cls in (
        SampleExclusionState, EstimationSampleSnapshot, PredictionInput,
        CategoricalEncodingState, DesignTermState, DesignPredictionState,
        DroppedFixedEffectState, FixedEffectPredictionState, PredictionState,
        FixedEffectLevelState, FixedEffectNestingState, CategoricalFixedEffectState,
        FixedEffectEstimates, FixedEffectTermEstimate,
    )
}


def _json_scalar(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_json(value: Any, *, digest_size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(_stable_json(value).encode("utf-8"))
    return h.hexdigest()



def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(tmp, np.asarray(values), allow_pickle=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _source_options(source: DataSource) -> dict:
    out = {"source_type": type(source).__name__}
    if hasattr(source, "kwargs"):
        # Parser options affect the interpreted data. repr() is used only for a
        # local cache key and is never emitted through ordinary profiles.
        out["kwargs"] = {str(k): repr(v) for k, v in sorted(source.kwargs.items())}
    if hasattr(source, "backend"):
        out["backend"] = str(source.backend)
    return out


def strict_source_fingerprint(source) -> str | None:
    """Content fingerprint for file-backed sources.

    Cross-process reuse is deliberately conservative.  We hash the complete
    source bytes plus parser options instead of trusting path/mtime alone.  A
    DataFrame has no stable cross-process identity and therefore returns None.
    """
    src = as_data_source(source)
    if isinstance(src, DataFrameSource) or not hasattr(src, "path"):
        return None
    path = Path(src.path)
    h = hashlib.blake2b(digest_size=20)
    h.update(_stable_json(_source_options(src)).encode("utf-8"))
    with path.open("rb") as fh:
        while True:
            block = fh.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def metadata_source_fingerprint(source) -> str | None:
    """Fast local-file identity for exploratory caches.

    Normal file writes change size/mtime/ctime and usually inode identity.
    This avoids rescanning a large source merely to validate a disposable
    cache.  It is intentionally opt-in because it is not a cryptographic
    content certificate: an adversarial/manual timestamp restoration could
    defeat it.
    """
    src = as_data_source(source)
    if isinstance(src, DataFrameSource) or not hasattr(src, "path"):
        return None
    path = Path(src.path)
    st = path.stat()
    payload = {
        "source": _source_options(src),
        "path": str(path.resolve()),
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }
    return _hash_json(payload, digest_size=20)


def _safe_levels(levels: tuple[object, ...] | None):
    if levels is None:
        return None
    out = []
    for value in levels:
        try:
            out.append(_json_scalar(value))
        except TypeError:
            return None
    return out


def _persistable_values(values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values)
    if arr.dtype.kind in "biufcMmUS":
        return arr
    if arr.dtype.kind != "O":
        return None
    # A common explicit factor case is an object array of strings.  Persist it
    # safely as a Unicode array.  Missing/mixed arbitrary Python objects are
    # intentionally not serialized; they simply fall back to source parsing.
    vals = arr.tolist()
    if all(isinstance(v, str) for v in vals):
        return np.asarray(vals, dtype=np.str_)
    return None


class PersistentSessionStore:
    """Disposable on-disk cache for exploratory linear-HDFE workflows.

    It is not part of the statistical state of an estimator.  Deleting the
    directory can only remove reuse and make a subsequent run slower.
    """

    def __init__(self, root, *, source_validation: str = "strict"):
        if source_validation not in {"strict", "metadata"}:
            raise SpecificationError(
                "persistent cache source_validation must be strict/metadata",
                code="data.cache_validation", stage="data",
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_validation = source_validation
        self.stats = {
            "source_hashes": 0,
            "column_hits": 0,
            "column_misses": 0,
            "within_hits": 0,
            "within_misses": 0,
            "result_hits": 0,
            "result_misses": 0,
        }
        self._source_hash_memo: dict[int, str | None] = {}

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def source_fingerprint(self, source) -> str | None:
        key = id(source)
        if key in self._source_hash_memo:
            return self._source_hash_memo[key]
        fp = (
            strict_source_fingerprint(source)
            if self.source_validation == "strict"
            else metadata_source_fingerprint(source)
        )
        self.stats["source_hashes"] += int(fp is not None)
        self._source_hash_memo[key] = fp
        return fp

    def _source_dir(self, source_fp: str) -> Path:
        return self.root / "sources" / source_fp

    @staticmethod
    def _column_token(name: str, representation: str) -> str:
        return _hash_json({"name": str(name), "representation": representation})

    def _column_paths(self, source_fp: str, name: str, representation: str):
        base = self._source_dir(source_fp) / "columns" / self._column_token(name, representation)
        return base.with_suffix(".values.npy"), base.with_suffix(".valid.npy"), base.with_suffix(".json")

    def _load_column(self, source_fp: str, name: str, representation: str):
        values_path, valid_path, meta_path = self._column_paths(source_fp, name, representation)
        if not (values_path.exists() and valid_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("cache_format") != PERSISTENT_CACHE_FORMAT:
                return None
            if meta.get("name") != name or meta.get("representation") != representation:
                return None
            values = np.load(values_path, mmap_mode="r", allow_pickle=False)
            valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
            if values.ndim != 1 or int(values.size) != int(meta["nobs"]):
                return None
            if valid.ndim != 1 or int(valid.size) != (int(values.size) + 7) // 8:
                return None
            return {
                "values": values,
                "signature": str(meta["signature"]),
                "valid_packed": valid,
                "invalid_count": int(meta["invalid_count"]),
                "identifier_levels": tuple(meta["identifier_levels"]) if meta.get("identifier_levels") is not None else None,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _save_column(self, source_fp: str, name: str, representation: str, snapshot: dict) -> bool:
        persist = _persistable_values(snapshot["values"])
        levels = _safe_levels(snapshot.get("identifier_levels"))
        if persist is None or (representation == "identifier" and levels is None):
            return False
        values_path, valid_path, meta_path = self._column_paths(source_fp, name, representation)
        _atomic_npy(values_path, persist)
        _atomic_npy(valid_path, np.asarray(snapshot["valid_packed"], dtype=np.uint8))
        # Signature belongs to the persisted representation, not necessarily the
        # original pandas object dtype.  Restore once and compute only on misses.
        restored = EncodedEconometricDataset({name: persist}, identifier_levels={name: tuple(levels)} if representation == "identifier" else {})
        rs = restored._persistent_column_snapshot(name)
        meta = {
            "cache_format": PERSISTENT_CACHE_FORMAT,
            "name": name,
            "representation": representation,
            "nobs": int(persist.size),
            "dtype": str(persist.dtype),
            "signature": rs["signature"],
            "invalid_count": int(rs["invalid_count"]),
            "identifier_levels": levels,
        }
        _atomic_json(meta_path, meta)
        # Persist the validity bitmap generated for the restored representation.
        _atomic_npy(valid_path, np.asarray(rs["valid_packed"], dtype=np.uint8))
        return True

    def prepare_repeated_dataset(
        self,
        source,
        specifications: Sequence[dict],
        *,
        common_roles: Mapping | None = None,
        memory_budget_mb: float = 512,
        batch_fraction: float = 0.20,
    ) -> EncodedEconometricDataset:
        src = as_data_source(source)
        source_fp = self.source_fingerprint(src)
        if source_fp is None:
            from .dataset import prepare_repeated_dataset
            return prepare_repeated_dataset(
                src, specifications, common_roles=dict(common_roles or {}),
                memory_budget_mb=memory_budget_mb, batch_fraction=batch_fraction,
            )
        common = dict(common_roles or {})
        reqs = []
        for spec in specifications:
            merged = dict(common); merged.update(dict(spec))
            reqs.append(compile_data_requirements(**merged))
        if not reqs:
            raise SpecificationError(
                "persistent repeated dataset requires at least one specification",
                code="data.empty_specifications", stage="data",
            )
        req = merge_data_requirements(*reqs)
        identifiers = set(req.identifier_only_columns)
        snapshots: dict[str, dict] = {}
        missing: list[str] = []
        missing_identifier: list[str] = []
        for name in req.columns:
            rep = "identifier" if name in identifiers else "raw"
            snap = self._load_column(source_fp, name, rep)
            if snap is None:
                self.stats["column_misses"] += 1
                missing.append(name)
                if rep == "identifier":
                    missing_identifier.append(name)
            else:
                self.stats["column_hits"] += 1
                snapshots[name] = snap

        if missing:
            materialized = materialize_required_data(
                src, missing, memory_budget_mb=memory_budget_mb,
                batch_fraction=batch_fraction, identifier_columns=missing_identifier,
            )
            fresh = EncodedEconometricDataset.from_frame(materialized.frame)
            fresh._identifier_levels.update(materialized.identifier_levels)
            for name in missing:
                snap = fresh._persistent_column_snapshot(name)
                if name in materialized.identifier_levels:
                    snap = dict(snap); snap["identifier_levels"] = materialized.identifier_levels[name]
                rep = "identifier" if name in identifiers else "raw"
                saved = self._save_column(source_fp, name, rep, snap)
                if saved:
                    loaded = self._load_column(source_fp, name, rep)
                    snapshots[name] = loaded if loaded is not None else snap
                else:
                    snapshots[name] = snap

        columns = {name: snapshots[name]["values"] for name in req.columns}
        signatures = {name: snapshots[name]["signature"] for name in req.columns}
        valid = {name: snapshots[name]["valid_packed"] for name in req.columns}
        invalid = {name: int(snapshots[name]["invalid_count"]) for name in req.columns}
        levels = {
            name: snapshots[name]["identifier_levels"]
            for name in req.columns if snapshots[name].get("identifier_levels") is not None
        }
        return EncodedEconometricDataset._from_persistent_snapshot(
            columns, signatures=signatures, valid_packed=valid, invalid_counts=invalid,
            identifier_levels=levels,
            source_metadata={
                "source_type": type(src).__name__, "persistent_cache": True,
                "source_fingerprint": source_fp,
            },
        )

    def within_signature(self, *, entry_payload: dict) -> str:
        payload = dict(entry_payload)
        payload["cache_format"] = PERSISTENT_CACHE_FORMAT
        payload["numerical_abi"] = PERSISTENT_NUMERICAL_ABI
        return _hash_json(payload)

    def load_within(self, entry_signature: str, name: str, column_signature: str, nobs: int):
        token = _hash_json({"name": name, "column_signature": column_signature})
        base = self.root / "within" / entry_signature / token
        values_path = base.with_suffix(".npy")
        meta_path = base.with_suffix(".json")
        if not values_path.exists() or not meta_path.exists():
            self.stats["within_misses"] += 1
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("cache_format") != PERSISTENT_CACHE_FORMAT
                or meta.get("numerical_abi") != PERSISTENT_NUMERICAL_ABI
                or meta.get("column_signature") != column_signature
                or int(meta.get("nobs", -1)) != int(nobs)
            ):
                self.stats["within_misses"] += 1
                return None
            out = np.load(values_path, mmap_mode="r", allow_pickle=False)
            if out.ndim != 1 or int(out.size) != int(nobs) or out.dtype != np.float64:
                self.stats["within_misses"] += 1
                return None
            self.stats["within_hits"] += 1
            return out
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self.stats["within_misses"] += 1
            return None

    def save_within(self, entry_signature: str, name: str, column_signature: str, values: np.ndarray) -> None:
        token = _hash_json({"name": name, "column_signature": column_signature})
        base = self.root / "within" / entry_signature / token
        _atomic_npy(base.with_suffix(".npy"), np.asarray(values, dtype=np.float64))
        _atomic_json(base.with_suffix(".json"), {
            "cache_format": PERSISTENT_CACHE_FORMAT,
            "numerical_abi": PERSISTENT_NUMERICAL_ABI,
            "column_signature": column_signature,
            "nobs": int(np.asarray(values).size),
        })

    def result_key(self, payload: dict) -> str:
        p = dict(payload)
        p["cache_format"] = PERSISTENT_CACHE_FORMAT
        p["numerical_abi"] = PERSISTENT_NUMERICAL_ABI
        p["result_abi"] = PERSISTENT_RESULT_ABI
        return _hash_json(p, digest_size=20)

    def _encode_result_value(self, value, arrays: dict[str, np.ndarray], key: str):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            arr_key = f"a{len(arrays)}"
            arrays[arr_key] = np.asarray(value)
            return {"__array__": arr_key}
        if isinstance(value, bytes):
            arr_key = f"a{len(arrays)}"
            arrays[arr_key] = np.frombuffer(value, dtype=np.uint8)
            return {"__bytes_array__": arr_key}
        if isinstance(value, tuple):
            return {"__tuple__": [self._encode_result_value(v, arrays, f"{key}[]") for v in value]}
        if isinstance(value, list):
            return {"__list__": [self._encode_result_value(v, arrays, f"{key}[]") for v in value]}
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("persistent result cache supports only string-keyed dictionaries")
            return {"__dict__": {k: self._encode_result_value(v, arrays, f"{key}.{k}") for k, v in value.items()}}
        if isinstance(value, DofInfo):
            return {"__dof_info__": self._encode_result_value(asdict(value), arrays, f"{key}.dof")}
        cls = type(value)
        if cls.__name__ in _PREDICTION_STATE_TYPES and _PREDICTION_STATE_TYPES[cls.__name__] is cls:
            payload = {
                field.name: self._encode_result_value(
                    getattr(value, field.name), arrays, f"{key}.{field.name}"
                )
                for field in fields(cls)
            }
            return {"__prediction_state__": cls.__name__, "fields": payload}
        raise TypeError(f"unsupported persistent result field {key}: {type(value).__name__}")

    def _decode_result_value(self, value, array_dir: Path):
        if not isinstance(value, dict):
            return value
        if "__array__" in value:
            return np.load(array_dir / f"{value['__array__']}.npy", mmap_mode="r", allow_pickle=False)
        if "__bytes_array__" in value:
            raw = np.load(
                array_dir / f"{value['__bytes_array__']}.npy",
                mmap_mode="r", allow_pickle=False,
            )
            return np.asarray(raw, dtype=np.uint8).tobytes()
        if "__tuple__" in value:
            return tuple(self._decode_result_value(v, array_dir) for v in value["__tuple__"])
        if "__list__" in value:
            return [self._decode_result_value(v, array_dir) for v in value["__list__"]]
        if "__dict__" in value:
            return {k: self._decode_result_value(v, array_dir) for k, v in value["__dict__"].items()}
        if "__dof_info__" in value:
            payload = self._decode_result_value(value["__dof_info__"], array_dir)
            return DofInfo(**payload)
        if "__prediction_state__" in value:
            name = str(value["__prediction_state__"])
            cls = _PREDICTION_STATE_TYPES.get(name)
            if cls is None:
                raise ValueError(f"unknown prediction-state type {name!r}")
            payload = {
                key: self._decode_result_value(item, array_dir)
                for key, item in value["fields"].items()
            }
            return cls(**payload)
        raise ValueError("unknown persistent result encoding")

    def save_result(self, key: str, result: RegressionResult) -> bool:
        base = self.root / "results" / key
        tmp = base.parent / f".{base.name}.tmp-{os.getpid()}"
        try:
            if tmp.exists(): shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=False)
            arrays: dict[str, np.ndarray] = {}
            payload = {}
            for f in fields(RegressionResult):
                payload[f.name] = self._encode_result_value(getattr(result, f.name), arrays, f.name)
            for name, arr in arrays.items():
                np.save(tmp / f"{name}.npy", np.asarray(arr), allow_pickle=False)
            _atomic_json(tmp / "result.json", {
                "cache_format": PERSISTENT_CACHE_FORMAT,
                "numerical_abi": PERSISTENT_NUMERICAL_ABI,
                "result_abi": PERSISTENT_RESULT_ABI,
                "fields": payload,
            })
            base.parent.mkdir(parents=True, exist_ok=True)
            if base.exists(): shutil.rmtree(base)
            os.replace(tmp, base)
            return True
        except (TypeError, ValueError, OSError):
            if tmp.exists(): shutil.rmtree(tmp, ignore_errors=True)
            return False

    def load_result(self, key: str) -> RegressionResult | None:
        base = self.root / "results" / key
        meta_path = base / "result.json"
        if not meta_path.exists():
            self.stats["result_misses"] += 1
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("cache_format") != PERSISTENT_CACHE_FORMAT
                or meta.get("numerical_abi") != PERSISTENT_NUMERICAL_ABI
                or meta.get("result_abi") != PERSISTENT_RESULT_ABI
            ):
                self.stats["result_misses"] += 1
                return None
            payload = {k: self._decode_result_value(v, base) for k, v in meta["fields"].items()}
            self.stats["result_hits"] += 1
            return RegressionResult(**payload)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.stats["result_misses"] += 1
            return None

    def info(self) -> dict:
        out = dict(self.stats)
        out["source_validation"] = self.source_validation
        return out
