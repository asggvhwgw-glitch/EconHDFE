import json
import runpy
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "docs" / "development" / "architecture-map"


def test_architecture_map_generated_outputs_are_current():
    subprocess.run(
        [sys.executable, str(ROOT / "skills" / "econhdfe" / "scripts" / "architecture_map.py"), "--check"],
        cwd=ROOT,
        check=True,
    )


def test_architecture_map_ir_is_closed_and_versioned():
    data = json.loads((MAP_DIR / "architecture.json").read_text())
    assert data["project"] == "econhdfe"
    version = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
    assert data["version"] == version
    node_ids = {node["id"] for node in data["nodes"]}
    assert {"interface", "frontend", "ols", "linear_iv", "ppml", "ppml_iv", "hdfe", "iv", "compute"} <= node_ids
    for edge in data["aggregate_import_edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
        assert edge["source"] != edge["target"]
        assert edge["count"] >= 1
    for flow in data["flows"]:
        assert set(flow["nodes"]) <= node_ids
        for source, target, description in flow["steps"]:
            assert source in node_ids and target in node_ids
            assert description.strip()


def test_architecture_html_is_self_contained_and_markdown_has_mermaid():
    html = (MAP_DIR / "architecture.html").read_text()
    assert "<script src=" not in html
    assert "<link rel=" not in html
    assert "fetch(" not in html
    assert "architecture dependency graph" in html
    markdown = (MAP_DIR / "architecture.md").read_text()
    assert "```mermaid" in markdown
    assert "AST-derived" in markdown or "AST import" in markdown


def test_architecture_node_labels_fit_declared_cards():
    data = json.loads((MAP_DIR / "architecture.json").read_text())
    for node in data["nodes"]:
        lines = node.get("label_lines") or [node["label"]]
        assert 1 <= len(lines) <= 2
        assert all(line.strip() for line in lines)
        assert max(map(len, lines)) <= 25
        assert node["width"] >= 180
        assert node["height"] >= 84
        assert node["x"] >= 0 and node["y"] >= 0
        assert node["x"] + node["width"] <= 1240
        assert node["y"] + node["height"] <= 835


def test_architecture_html_uses_wrapped_svg_titles_and_bounded_detail_text():
    html = (MAP_DIR / "architecture.html").read_text()
    assert "label_lines" in html
    assert "tspan" in html
    assert "overflow-wrap:anywhere" in html
    assert "function makePortPlan(edges)" in html
    assert "function roundedOrthogonal(points" in html
    assert "class:'lane'" in html


def test_architecture_edges_use_port_routing_terminal_gaps_and_open_chevrons():
    html = (MAP_DIR / "architecture.html").read_text()
    assert "function outsidePoint(p,side,gap)" in html
    assert "q.ts,8" in html
    assert "markerUnits=\"userSpaceOnUse\"" in html
    assert "fill=\"none\" stroke=\"var(--line)\"" in html
    assert "class:'edge-halo'" in html
    assert "edge.overview" in html


def test_architecture_repo_area_counts_ignore_release_excluded_transients(tmp_path):
    namespace = runpy.run_path(str(ROOT / "skills" / "econhdfe" / "scripts" / "architecture_map.py"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "kept.md").write_text("# kept\n")
    for name in ("trace.log", "trace.out", "trace.aux", ".coverage"):
        (docs / name).write_text("transient\n")
    areas = {item["name"]: item for item in namespace["_repo_areas"](tmp_path)}
    assert areas["docs"] == {
        "name": "docs",
        "files": 1,
        "python_files": 0,
        "markdown_files": 1,
    }
