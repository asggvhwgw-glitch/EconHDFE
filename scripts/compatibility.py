"""Public-contract snapshot and compatibility gate for econhdfe releases.

The contract is intentionally narrower than implementation identity.  It tracks
public exports/signatures, dataclass result schemas, configuration defaults and
stable structured-error codes.  Any change relative to the previous released
snapshot must be explicitly approved before a release can pass the gate.
"""
from __future__ import annotations

import argparse
import dataclasses
from enum import Enum
import hashlib
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
BASELINES = ROOT / "compatibility" / "baselines"
APPROVALS = ROOT / "compatibility" / "APPROVED_CHANGES.json"
PYPROJECT = ROOT / "pyproject.toml"


def project_version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    if not m:
        raise SystemExit("project version not found")
    return m.group(1)


def _jsonable_default(value: Any) -> str:
    if value is dataclasses.MISSING:
        return "<required>"
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    return repr(value)


def _dataclass_schema(cls: type) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for f in dataclasses.fields(cls):
        default = f.default
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            default_repr = f"<factory:{getattr(f.default_factory, '__name__', type(f.default_factory).__name__)}>"  # type: ignore[misc]
        else:
            default_repr = _jsonable_default(default)
        out.append({"name": f.name, "default": default_repr})
    return out


def _public_signature(obj: Any) -> str:
    """Keep inherited Enum lookup independent of CPython's metaclass helpers.

    Nonempty enums with the standard metaclass use the existing snapshot's
    ``(*values)`` representation. Empty enums (the functional factory), custom
    metaclasses and explicit signatures still use normal introspection.
    Historical snapshot files are never rewritten.
    """
    if isinstance(obj, type(Enum)) and obj.__members__:
        explicit = any("__signature__" in vars(base) for base in obj.__mro__ if base is not Enum)
        if not explicit:
            if type(obj).__call__ is type(Enum).__call__:
                return "(*values)"
            signature = inspect.signature(type(obj).__call__)
            return str(signature.replace(parameters=list(signature.parameters.values())[1:]))
    return str(inspect.signature(obj))


def capture_contract() -> dict[str, Any]:
    pkg = importlib.import_module("econhdfe")
    errors_mod = importlib.import_module("econhdfe.errors")
    exports = sorted(set(pkg.__all__))

    signatures: dict[str, str] = {}
    dataclass_schemas: dict[str, list[dict[str, str]]] = {}
    objects = {name: getattr(pkg, name) for name in exports}
    public_namespaces = ("econhdfe", "econhdfe.effects")
    for namespace in public_namespaces[1:]:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            objects[f"{namespace}.{name}"] = getattr(module, name)
    for name, obj in sorted(objects.items()):
        if callable(obj):
            try:
                signatures[name] = _public_signature(obj)
            except (TypeError, ValueError):
                pass
        if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
            dataclass_schemas[name] = _dataclass_schema(obj)

    base = errors_mod.EconHDFEError
    error_catalog: dict[str, dict[str, str]] = {}
    for name, obj in vars(errors_mod).items():
        if not inspect.isclass(obj) or not issubclass(obj, base) or name.startswith("_"):
            continue
        error_catalog[name] = {
            "code": str(getattr(obj, "default_code", "")),
            "stage": str(getattr(obj, "default_stage", "")),
        }

    config_names = ("HDFEConfig", "InferenceConfig", "ExecutionConfig")
    config_schemas = {
        name: _dataclass_schema(getattr(pkg, name))
        for name in config_names
    }

    result_names = (
        "RegressionResult", "PPMLResult", "IVPPMLResult", "SPJResult",
        "SPJBootstrapResult", "PreflightReport",
    )
    result_schemas = {
        name: _dataclass_schema(getattr(pkg, name))
        for name in result_names
        if hasattr(pkg, name) and dataclasses.is_dataclass(getattr(pkg, name))
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": project_version(),
        "public_symbols": sorted(objects),
        "public_namespaces": list(public_namespaces),
        "signatures": signatures,
        "result_schemas": result_schemas,
        "config_schemas": config_schemas,
        "error_catalog": error_catalog,
        "exported_dataclass_schemas": dataclass_schemas,
    }


def write_snapshot(path: Path, *, version: str | None = None) -> Path:
    data = capture_contract()
    if version is not None:
        data["package_version"] = version
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text()
        if existing != payload:
            raise SystemExit(
                f"refusing to overwrite a different public-contract snapshot: {path}; "
                "published baselines are immutable"
            )
        return path
    path.write_text(payload)
    return path


def require_baseline(version: str) -> Path:
    path = BASELINES / f"econhdfe-{version}.json"
    data = load_json(path)
    if data.get("package_version") != version:
        raise SystemExit(f"compatibility baseline metadata/version mismatch for {version}")
    return path


def load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing compatibility file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def _diff_values(path: str, old: Any, new: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            p = f"{path}.{key}" if path else str(key)
            if key not in old:
                _append_change(p, "added", None, new[key], out)
            elif key not in new:
                _append_change(p, "removed", old[key], None, out)
            else:
                _diff_values(p, old[key], new[key], out)
        return
    if isinstance(old, list) and isinstance(new, list):
        if old != new:
            _append_change(path, "changed", old, new, out)
        return
    if old != new:
        _append_change(path, "changed", old, new, out)


def _append_change(path: str, kind: str, old: Any, new: Any, out: list[dict[str, Any]]) -> None:
    payload = json.dumps([path, kind, old, new], sort_keys=True, separators=(",", ":"), default=str)
    change_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    out.append({"id": change_id, "path": path, "kind": kind, "old": old, "new": new})


def diff_contracts(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old = dict(old)
    new = dict(new)
    old.pop("package_version", None)
    new.pop("package_version", None)
    changes: list[dict[str, Any]] = []
    _diff_values("", old, new, changes)
    return changes


def approvals_template(*, baseline_version: str, release_version: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_version": baseline_version,
        "release_version": release_version,
        "approved": [],
    }


def reset_approvals(*, baseline_version: str, release_version: str) -> None:
    APPROVALS.parent.mkdir(parents=True, exist_ok=True)
    APPROVALS.write_text(
        json.dumps(
            approvals_template(baseline_version=baseline_version, release_version=release_version),
            indent=2,
            sort_keys=True,
        ) + "\n"
    )


def current_diff(approval_data: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    approval_data = approval_data or load_json(APPROVALS)
    baseline_version = str(approval_data.get("baseline_version", ""))
    baseline_path = require_baseline(baseline_version)
    baseline = load_json(baseline_path)
    current = capture_contract()
    return current, diff_contracts(baseline, current)


def check_compatibility(*, expected_version: str | None = None) -> list[dict[str, Any]]:
    approval_data = load_json(APPROVALS)
    if approval_data.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"compatibility approval schema_version must be {SCHEMA_VERSION}")
    version = expected_version or project_version()
    if approval_data.get("release_version") != version:
        raise SystemExit(
            f"compatibility release_version mismatch: {approval_data.get('release_version')!r} != {version!r}"
        )
    baseline_version = str(approval_data.get("baseline_version", ""))
    if not baseline_version:
        raise SystemExit("compatibility baseline_version is missing")
    baseline_path = require_baseline(baseline_version)
    baseline = load_json(baseline_path)
    if baseline.get("package_version") != baseline_version:
        raise SystemExit("compatibility baseline metadata/version mismatch")
    changes = diff_contracts(baseline, capture_contract())
    approvals = approval_data.get("approved", [])
    if not isinstance(approvals, list):
        raise SystemExit("compatibility approved must be a list")
    approved_by_id = {
        item.get("id"): item
        for item in approvals
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    change_ids = {c["id"] for c in changes}
    missing = [c for c in changes if c["id"] not in approved_by_id]
    stale = sorted(set(approved_by_id) - change_ids)
    bad_reason = [cid for cid, item in approved_by_id.items() if not str(item.get("reason", "")).strip()]
    problems: list[str] = []
    if missing:
        problems.append("unapproved public-contract changes: " + ", ".join(f"{c['id']}:{c['path']}" for c in missing))
    if stale:
        problems.append("stale compatibility approvals: " + ", ".join(stale))
    if bad_reason:
        problems.append("approvals missing a reason: " + ", ".join(sorted(bad_reason)))
    if problems:
        raise SystemExit("compatibility gate failed:\n- " + "\n- ".join(problems))
    return changes


def approve_change(change_id: str, reason: str) -> None:
    if not reason.strip():
        raise SystemExit("approval reason must be non-empty")
    data = load_json(APPROVALS)
    _, changes = current_diff(data)
    match = next((c for c in changes if c["id"] == change_id), None)
    if match is None:
        raise SystemExit(f"unknown current compatibility change id: {change_id}")
    approved = [x for x in data.get("approved", []) if isinstance(x, dict) and x.get("id") != change_id]
    approved.append({"id": change_id, "path": match["path"], "kind": match["kind"], "reason": reason.strip()})
    data["approved"] = sorted(approved, key=lambda x: x["id"])
    APPROVALS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _print_changes(changes: list[dict[str, Any]]) -> None:
    if not changes:
        print("public contract: no changes")
        return
    for c in changes:
        print(f"{c['id']}  {c['kind']:<7}  {c['path']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("snapshot")
    sp.add_argument("--version", default=None)
    sp.add_argument("--output", type=Path, default=None)
    dp = sub.add_parser("diff")
    dp.add_argument("--json", action="store_true")
    cp = sub.add_parser("check")
    cp.add_argument("--version", default=None)
    apv = sub.add_parser("approve")
    apv.add_argument("change_id")
    apv.add_argument("--reason", required=True)
    args = ap.parse_args()

    if args.command == "snapshot":
        version = args.version or project_version()
        path = args.output or (BASELINES / f"econhdfe-{version}.json")
        print(write_snapshot(path, version=version))
    elif args.command == "diff":
        _, changes = current_diff()
        if args.json:
            print(json.dumps(changes, indent=2, sort_keys=True))
        else:
            _print_changes(changes)
    elif args.command == "check":
        changes = check_compatibility(expected_version=args.version)
        print(f"public contract: {len(changes)} approved change(s)")
    else:
        approve_change(args.change_id, args.reason)
        print(f"approved compatibility change: {args.change_id}")


if __name__ == "__main__":
    main()
