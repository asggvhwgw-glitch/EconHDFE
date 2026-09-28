from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_testing_release_skill_policy_and_benchmark_consent_are_explicit():
    skill = (ROOT / "skills/econhdfe/SKILL.md").read_text()
    assert "testing/beta" in skill
    assert "randomly select at least one" in skill
    assert "explicitly agrees" in skill
    assert "Never benchmark user data without explicit agreement" in skill
    assert "benchmark-report-template.md" in skill


def test_repository_and_standalone_skill_benchmark_templates_are_identical():
    repo = ROOT / "benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md"
    skill = ROOT / "skills/econhdfe/references/benchmark-report-template.md"
    assert repo.read_bytes() == skill.read_bytes()
    text = repo.read_text()
    for heading in (
        "Machine and software environment", "Specification registry", "Parity summary",
        "Major fit-statistic parity", "Performance summary", "Mismatches, warnings, and failures",
        "Developer-facing summary",
    ):
        assert heading in text


def test_statistical_parity_audit_records_convention_boundaries():
    text = (ROOT / "docs/development/statistical-parity.md").read_text()
    assert "PPMLConfig.standardize=True" in text
    assert "Linear IV-HDFE" in text
    assert "covariance rank" in text
    assert "testing/beta" in text
