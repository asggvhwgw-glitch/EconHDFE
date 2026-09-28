from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_error_report_templates_are_identical_and_privacy_minimized():
    repo = ROOT / "docs/development/ERROR_REPORT_TEMPLATE.md"
    skill = ROOT / "skills/econhdfe/references/error-report-template.md"
    assert repo.read_bytes() == skill.read_bytes()
    text = repo.read_text()
    for heading in (
        "Report identity", "Privacy checklist", "Runtime/environment parameters",
        "Anonymous specification fingerprint", "Sample and dimensional counts",
        "Error / warning fingerprint", "Behavioral stability parameters",
        "Reference-package parity summary", "Numerical/execution diagnostics",
        "Developer-facing summary", "Privacy rule",
    ):
        assert heading in text
    for required in (
        "parameter-only",
        "Do **not** include raw or synthetic observations",
        "Recommended aliases",
        "Stack signature as `module:function` names only",
        "difference magnitude without the underlying estimate",
        "must **not request** raw data",
        "Performance benchmarking is a separate workflow",
    ):
        assert required in text
    for forbidden in (
        "Minimal reproduction",
        "Exact econhdfe call",
        "paste traceback",
        "Attached artifacts",
        "may use the supplied original/raw dataset",
        "may use the supplied original replication scripts",
    ):
        assert forbidden not in text


def test_skill_routes_material_errors_to_parameter_only_template():
    text = (ROOT / "skills/econhdfe/SKILL.md").read_text()
    assert "references/error-report-template.md" in text
    assert "material error or parity mismatch" in text
    assert "parameter-only, privacy-minimized metadata" in text
    assert "Do not request or return raw/sampled/synthetic observations" in text
    assert "Benchmarking remains a separate explicit-consent workflow" in text


def test_planner_report_templates_are_identical_and_privacy_minimized():
    repo = ROOT / "docs/development/PLANNER_REPORT_TEMPLATE.md"
    skill = ROOT / "skills/econhdfe/references/planner-report-template.md"
    assert repo.read_bytes() == skill.read_bytes()
    text = repo.read_text()
    for required in (
        "Automatic thread calibration", "Selected automatic threads", "parameter-only",
        "include no raw or synthetic observations", "Real-data performance benchmarking is a separate explicit-consent workflow",
    ):
        assert required in text
    for forbidden in (
        "Exact econhdfe call", "paste traceback", "Attached artifacts", "hostname / username",
    ):
        assert forbidden not in text


def test_skill_routes_planner_performance_reports_to_privacy_template():
    text = (ROOT / "skills/econhdfe/SKILL.md").read_text()
    assert "references/planner-feedback.md" in text
    assert "references/planner-report-template.md" in text
    assert "planner_report.py" in text
