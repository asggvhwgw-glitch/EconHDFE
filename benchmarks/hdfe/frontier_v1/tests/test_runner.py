"""A previous success must never mask a failed or mismatched worker execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.hdfe.frontier_v1 import run


@pytest.fixture
def request_data():
    return {
        "family": "rigidity_grid", "size": 8, "backend": "scipy_lsmr",
        "rhs": 2, "rhs_kind": "mixed", "seed": 20261029, "repeat": 1,
        "weight_ratio": 1., "threads": 1, "tol": 1e-10,
        "forward_tol": 2e-8, "max_iter": 20, "spectrum_max_categories": 12,
        "timing_trial": 0,
    }


def _fake_children(monkeypatch, behaviors):
    """Model exit codes and output files without native kernels or sleeping."""
    seen_request_paths = []
    actions = iter(behaviors)
    monkeypatch.setattr(run, "environment", lambda: {"test_environment": True})

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.returncode, emit, mutate = next(actions)
            self.pid = 123456789
            request_path = Path(command[command.index("--worker-request") + 1])
            output_path = Path(command[command.index("--worker-output") + 1])
            seen_request_paths.append(request_path)
            if emit:
                request = json.loads(request_path.read_text())
                if mutate is not None:
                    mutate(request)
                run.write_json(output_path, {"request": request, "status": "passed"})

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(run.subprocess, "Popen", FakeProcess)
    return seen_request_paths


def test_old_passed_output_cannot_mask_crashed_child(tmp_path, monkeypatch, request_data):
    output = tmp_path / "pilot.json"
    # This is the pre-fix runner's reused directory and exact output filename.
    old = tmp_path / "pilot_workers" / "000_rigidity_grid_8_scipy_lsmr.json"
    run.write_json(old, {"request": copy.deepcopy(request_data), "status": "passed"})
    _fake_children(monkeypatch, [(9, False, None)])
    report = run.run_requests([request_data], output, timeout_seconds=1)
    assert report["records"][0]["status"] == "process_failed"
    assert report["summary"]["by_status"].get("passed", 0) == 0
    assert report["records"][0]["process_exit_code"] == 9


def test_nonzero_exit_rejects_even_a_newly_written_passed_record(tmp_path, monkeypatch, request_data):
    _fake_children(monkeypatch, [(7, True, None)])
    report = run.run_requests([request_data], tmp_path / "pilot.json", timeout_seconds=1)
    assert report["records"][0]["status"] == "process_failed"
    assert report["records"][0]["process_exit_code"] == 7
    assert report["summary"]["by_status"].get("passed", 0) == 0


def test_worker_request_mismatch_is_rejected(tmp_path, monkeypatch, request_data):
    def different_tolerance(request):
        request["tol"] = request["tol"] * 1000

    _fake_children(monkeypatch, [(0, True, different_tolerance)])
    report = run.run_requests([request_data], tmp_path / "pilot.json", timeout_seconds=1)
    assert report["records"][0]["status"] == "process_failed"
    assert report["summary"]["by_status"].get("passed", 0) == 0


def test_repeated_report_path_has_fresh_worker_directory(tmp_path, monkeypatch, request_data):
    paths = _fake_children(monkeypatch, [(0, True, None), (5, False, None)])
    output = tmp_path / "pilot.json"
    first = run.run_requests([request_data], output, timeout_seconds=1)
    second = run.run_requests([request_data], output, timeout_seconds=1)
    assert first["records"][0]["status"] == "passed"
    assert second["records"][0]["status"] == "process_failed"
    assert paths[0].parent != paths[1].parent
