from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from econhdfe import __version__


def test_skill_router_is_progressively_disclosed():
    text = Path("skills/econhdfe/SKILL.md").read_text()
    assert len(text.splitlines()) < 100
    for name in (
        "installation.md", "configuration.md", "empirical-research.md",
        "advanced-validation.md", "developer-guide.md",
    ):
        assert f"references/{name}" in text


def test_skill_zip_writer_preserves_canonical_tree(tmp_path):
    from scripts.assemble_release import write_skill_zip

    source = Path("skills/econhdfe")
    out = tmp_path / "skill.zip"
    write_skill_zip(source, out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "econhdfe/SKILL.md" in names
    assert "econhdfe/agents/openai.yaml" in names
    assert "econhdfe/references/empirical-research.md" in names
    assert "econhdfe/scripts/smoke_test.py" in names


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
