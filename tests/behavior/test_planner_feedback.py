from __future__ import annotations

import json
from pathlib import Path

from econhdfe.planner.calibration import (
    ThreadCalibration,
    _select_saturation_thread,
    calibration_candidates,
    clear_in_memory_calibration_cache,
    get_thread_calibration,
)
from econhdfe.planner.report import build_planner_developer_report, write_planner_developer_report


def test_calibration_candidates_measure_all_small_runtime_thread_counts():
    assert calibration_candidates(1) == (1,)
    assert calibration_candidates(4) == (1, 2, 3, 4)
    assert calibration_candidates(8) == tuple(range(1, 9))
    assert calibration_candidates(12) == (1, 2, 4, 8, 12)


def test_saturation_rule_chooses_smallest_near_best_thread_count():
    cands = (1, 2, 3, 4)
    times = (10.0, 5.0, 4.08, 4.0)
    # 3 threads is within 5% of the 4-thread best and is preferred.
    assert _select_saturation_thread(cands, times, 1.05) == 3


def test_calibration_cache_roundtrip_uses_no_user_data(tmp_path):
    clear_in_memory_calibration_cache()
    path = tmp_path / "calibration.json"
    first = get_thread_calibration(max_threads=1, force=True, cache_path=path)
    clear_in_memory_calibration_cache()
    second = get_thread_calibration(max_threads=1, cache_path=path)
    assert first.selected_threads == second.selected_threads == 1
    assert second.source == "cache"
    payload = json.loads(path.read_text())
    text = json.dumps(payload).lower()
    for forbidden in ("hostname", "filepath", "variable", "observation", "command", "traceback"):
        assert forbidden not in text


def _fake_calibration() -> ThreadCalibration:
    return ThreadCalibration(
        schema_version=1,
        calibration_id="abc123",
        runtime_fingerprint="runtimehash",
        max_threads=4,
        candidates=(1, 2, 4),
        median_seconds=(0.40, 0.22, 0.21),
        selected_threads=2,
        near_best_fraction=1.05,
        target_bytes=1_000_000,
        passes=4,
        repeats=3,
        source="cache",
    )


def test_planner_report_is_privacy_minimized_and_writeable(tmp_path):
    report = build_planner_developer_report(
        calibration=_fake_calibration(),
        suspected_issue="auto threads look slower than explicit threads",
        performance_concern="runtime regression",
    )
    text = report.to_markdown()
    for required in (
        "Automatic thread calibration", "Selected automatic threads", "Privacy rule",
        "no raw observations", "no raw or synthetic observations",
    ):
        assert required.lower() in text.lower()
    for forbidden in ("hostname:", "file path:", "exact command:", "raw traceback"):
        assert forbidden not in text.lower()
    path = write_planner_developer_report(
        tmp_path / "planner-report.md", calibration=_fake_calibration(),
    )
    assert path.is_file()
    assert "abc123" in path.read_text()


def test_small_hdfe_auto_threads_skip_calibration_and_stay_single_threaded():
    from econhdfe.hdfe.projection import resolve_threads
    assert resolve_threads("auto", nobs=99_999) == 1
