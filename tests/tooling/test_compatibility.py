from __future__ import annotations
import json


def test_public_contract_gate_requires_exact_approval(monkeypatch, tmp_path):
    import scripts.compatibility as compat

    baselines = tmp_path / "baselines"
    baselines.mkdir()
    approvals = tmp_path / "APPROVED_CHANGES.json"
    base = {
        "schema_version": 1,
        "package_version": "0.4.2",
        "public_symbols": ["a"],
        "signatures": {},
        "result_schemas": {},
        "config_schemas": {},
        "error_catalog": {},
        "exported_dataclass_schemas": {},
    }
    current = dict(base)
    current["package_version"] = "0.4.3"
    current["public_symbols"] = ["a", "b"]
    (baselines / "econhdfe-0.4.2.json").write_text(json.dumps(base))
    approvals.write_text(json.dumps({
        "schema_version": 1,
        "baseline_version": "0.4.2",
        "release_version": "0.4.3",
        "approved": [],
    }))
    monkeypatch.setattr(compat, "BASELINES", baselines)
    monkeypatch.setattr(compat, "APPROVALS", approvals)
    monkeypatch.setattr(compat, "capture_contract", lambda: current)

    import pytest
    with pytest.raises(SystemExit, match="unapproved public-contract changes"):
        compat.check_compatibility(expected_version="0.4.3")

    change = compat.diff_contracts(base, current)[0]
    data = json.loads(approvals.read_text())
    data["approved"] = [{"id": change["id"], "path": change["path"], "kind": change["kind"], "reason": "intentional additive public symbol"}]
    approvals.write_text(json.dumps(data))
    assert compat.check_compatibility(expected_version="0.4.3") == [change]
