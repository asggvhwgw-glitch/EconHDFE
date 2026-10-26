from __future__ import annotations
from pathlib import Path
import numpy as np
from econhdfe.config import ExecutionConfig, HDFEConfig, InferenceConfig
from econhdfe.errors import NumericalError
from econhdfe.results import RegressionResult
from econhdfe.support_reports import benchmark_report_template, error_report_template, safe_error_payload, write_template


ROOT = Path(__file__).resolve().parents[2]


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
