from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "docs" / "development" / "economic-module-map.md"


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def test_every_runtime_module_has_an_economics_facing_description():
    text = MAP.read_text()
    assert "Economic / econometric problem" in text
    assert "Computational responsibility" in text
    missing = []
    for path in sorted((ROOT / "econhdfe").rglob("*.py")):
        module = _module_name(path)
        if f"`{module}`" not in text:
            missing.append(module)
    assert not missing, (
        "Every runtime module must state the economic/econometric problem it serves in "
        "docs/development/economic-module-map.md; missing: " + ", ".join(missing)
    )
