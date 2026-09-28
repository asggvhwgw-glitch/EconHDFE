import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "technical_innovation.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("technical_innovation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_technical_innovation_registry_is_complete_and_valid():
    validator = _load_validator()
    assert validator.validate_registry(ROOT) == []
    data = json.loads((ROOT / "docs/technical/innovation-registry.json").read_text())
    assert {item["id"] for item in data["innovations"]} == {
        "HDFE-EXACT-RANK-001",
        "HDFE-NUMCORE-002",
        "HDFE-STRUCTURAL-DESIGN-003",
    }


def test_every_registered_innovation_has_formal_tex_and_pdf():
    data = json.loads((ROOT / "docs/technical/innovation-registry.json").read_text())
    for item in data["innovations"]:
        manuscript = item["manuscript"]
        tex = ROOT / manuscript["tex"]
        pdf = ROOT / manuscript["pdf"]
        assert tex.suffix == ".tex" and tex.stat().st_size > 5_000
        assert pdf.suffix == ".pdf" and pdf.stat().st_size > 50_000


def test_audit_does_not_register_engineering_as_innovation():
    data = json.loads((ROOT / "docs/technical/innovation-registry.json").read_text())
    text = " ".join(item["title"].lower() for item in data["innovations"])
    for engineering_only in (
        "cache", "parallel", "numba", "gpu", "architecture visualization",
        "wild cluster bootstrap", "ppml", "iv-ppml", "split-panel jackknife",
    ):
        assert engineering_only not in text
