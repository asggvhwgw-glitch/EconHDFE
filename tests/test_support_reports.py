from __future__ import annotations

from pathlib import Path

import numpy as np

from econhdfe.config import ExecutionConfig, HDFEConfig, InferenceConfig
from econhdfe.errors import NumericalError
from econhdfe.results import RegressionResult
from econhdfe.support_reports import (
    benchmark_report_template,
    build_error_report,
    error_report_template,
    safe_error_payload,
    write_template,
)


ROOT = Path(__file__).resolve().parents[1]


def _result():
    return RegressionResult(
        params=np.array([1234.5]),
        vcov=np.array([[9.0]]),
        stderr=np.array([3.0]),
        residuals=np.array([9876.5, 444.0]),
        fitted=np.array([111.0, 222.0]),
        nobs=100,
        rank=1,
        df_resid=12.0,
        df_absorbed=7,
        converged=True,
        iterations=8,
        dropped_singletons=2,
        names=("highly_sensitive_variable_name",),
        estimator="ols",
        vce="cluster",
        cluster_counts=(13,),
        fe_names=("sensitive_firm_id", "sensitive_year"),
        r2=0.987654321,
    )


def test_safe_payload_never_copies_values_names_paths_or_error_message():
    err = NumericalError(
        "failure in /secret/user/path involving customer_id and value 123456",
        code="numerical.test",
        stage="solver",
        details={
            "iterations": 17,
            "cluster_count": 9,
            "column_name": "customer_id",
            "base_level": 2020,
            "account_number": 123456789,
        },
    )
    payload = safe_error_payload(
        error=err,
        result=_result(),
        hdfe_config=HDFEConfig(solver="map", tolerance=1e-9, max_iter=1000),
        execution_config=ExecutionConfig(threads=4, memory_budget_mb=256),
        inference_config=InferenceConfig(vce="cluster"),
        parity={
            "coefficient_max_abs_diff": 1e-9,
            "unsafe_actual_coefficient": 999.0,
        },
    )
    text = repr(payload)
    for forbidden in (
        "/secret/user/path", "customer_id", "123456789", "1234.5", "9876.5",
        "0.987654321", "sensitive_firm_id", "sensitive_year", "999.0", "2020",
    ):
        assert forbidden not in text
    assert payload["error"]["numeric_details"] == {"iterations": 17, "cluster_count": 9}
    assert payload["result"]["explicit_parameter_count"] == 1
    assert payload["result"]["absorbed_fe_dimensions"] == 2
    assert payload["parity_difference_magnitudes"] == {"coefficient_max_abs_diff": 1e-9}


def test_rendered_error_report_is_parameter_only():
    report = build_error_report(
        result=_result(),
        execution_config=ExecutionConfig(threads=2, memory_budget_mb=128),
        issue_type="reference-package mismatch",
        parity={"stderr_max_rel_diff": 2e-8},
    )
    assert "privacy-minimized" in report
    assert "stderr_max_rel_diff" in report
    assert "sensitive_firm_id" not in report
    assert "highly_sensitive_variable_name" not in report
    assert "1234.5" not in report
    assert "0.987654321" not in report
    assert "Real-data benchmarking is a separate explicit-consent workflow" in report


def test_packaged_templates_match_repository_and_skill(tmp_path):
    # Template APIs return text, so checkout line endings are not content.
    error = error_report_template()
    bench = benchmark_report_template()
    assert error == (ROOT / "docs/development/ERROR_REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    assert error == (ROOT / "skills/econhdfe/references/error-report-template.md").read_text(encoding="utf-8")
    assert bench == (ROOT / "benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    assert bench == (ROOT / "skills/econhdfe/references/benchmark-report-template.md").read_text(encoding="utf-8")
    out = tmp_path / "error.md"
    write_template("error", out)
    assert out.read_text() == error


def test_cli_entrypoint_is_declared():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'econhdfe-report = "econhdfe.support_reports:main"' in pyproject


def test_skill_documents_report_helper_without_weakening_privacy():
    skill = (ROOT / "skills/econhdfe/SKILL.md").read_text()
    dev = (ROOT / "skills/econhdfe/references/developer-guide.md").read_text()
    assert "econhdfe-report error" in skill
    assert "econhdfe-report benchmark" in skill
    assert "explicit allowlist" in skill
    assert "econhdfe.support_reports.write_error_report" in skill
    assert "allowlist based" in dev
    assert "never emit coefficient/fitted/residual arrays" in dev


def test_release_verifier_checks_wheel_templates_and_console_script():
    text = (ROOT / "scripts/verify_release.py").read_text()
    assert "wheel error-report template missing or differs" in text
    assert "wheel benchmark template missing or differs" in text
    assert "econhdfe-report = econhdfe.support_reports:main" in text


def test_template_equality_normalizes_crlf_but_rejects_content_changes(tmp_path, monkeypatch):
    import pytest

    paths = (
        "docs/development/ERROR_REPORT_TEMPLATE.md",
        "skills/econhdfe/references/error-report-template.md",
        "benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md",
        "skills/econhdfe/references/benchmark-report-template.md",
    )
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    test_packaged_templates_match_repository_and_skill(tmp_path)
    (tmp_path / paths[0]).write_text("changed content\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        test_packaged_templates_match_repository_and_skill(tmp_path)
