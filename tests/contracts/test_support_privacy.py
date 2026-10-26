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
