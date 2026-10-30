"""Native telemetry must preserve measured scopes without certifying a solve."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.hdfe.frontier_v1 import backends
from benchmarks.hdfe.frontier_v1.families import make_case
from benchmarks.hdfe.frontier_v1.validation import certification


_MISSING = object()


def _payload():
    # The shared counters deliberately differ from both replicated iteration
    # entries and cumulative snapshots. These numbers must never be combined.
    return {
        "schema_version": 1,
        "scope": "call",
        "kernel": "fereg",
        "constructor": {"factorizations": 1, "elapsed_s": 2.0},
        "before": {"normal_matvec_batches": 19, "factorizations": 1},
        "after": {"normal_matvec_batches": 28, "factorizations": 2},
        "attempts": [
            {
                "rhs_start": 0, "rhs_width": 2,
                "iteration_scope": "shared_rhs_batch", "iterations": 7,
                "exact": True, "escalated": True, "precond_kinds": ["JointExact"],
                "before": {"normal_matvec_batches": 19, "factorizations": 1},
                "after": {"normal_matvec_batches": 26, "factorizations": 2},
                "delta_diagnostics": {
                    "normal_matvec_batches": 7, "normal_matvec_rhs_equivalents": 14,
                    "factorizations": 1, "deferred_factorization_s": 1.25,
                },
                "kernel_call_s": 2.0,
            },
            {
                "rhs_start": 0, "rhs_width": 2,
                "iteration_scope": "shared_rhs_batch", "iterations": 2,
                "exact": True, "escalated": False, "precond_kinds": ["JointExact"],
                "before": {"normal_matvec_batches": 26, "factorizations": 2},
                "after": {"normal_matvec_batches": 28, "factorizations": 2},
                "delta_diagnostics": {
                    "normal_matvec_batches": 2, "normal_matvec_rhs_equivalents": 4,
                    "factorizations": 0, "deferred_factorization_s": 0.0,
                },
                "kernel_call_s": 0.5,
            },
        ],
    }


def _case():
    return SimpleNamespace(
        groups=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint32),
        y=np.array([[1., 2.], [-1., -2.], [-1., -2.], [1., 2.]]),
        weights=None,
    )


def _install_fake(monkeypatch, name, payload=_MISSING, *, getter_error=None):
    state = SimpleNamespace(clock=0.0, constructors=0, solves=0, getter_delay=0.0)

    class DualAbsorber:
        # Exact legacy signatures catch accidental addition of a required flag.
        def __init__(self, codes, *, weights, backend, threads, scratch_bytes, allow_fallback):
            assert backend == name and allow_fallback is False
            state.constructors += 1
            state.clock += 3.0

        def residualize(self, y, *, tol, max_iter, return_info):
            assert return_info is True
            state.solves += 1
            state.clock += 5.0
            info = {
                "accepted": True, "iterations": [9] * y.shape[1],
                "setup_seconds": 1.25, "max_iter_scope": "per_attempt",
            }
            if payload is not _MISSING:
                info["native_cost_diagnostics"] = payload
            return np.zeros_like(y), info

    class NativeSolver:
        route = "within"
        n_cells = 4
        fallback_count = 0
        schur_condition = None

        def __init__(self, codes, *, weights, strategy, threads, workspace_bytes):
            assert strategy == "auto"
            state.constructors += 1
            state.clock += 3.0

        def solve(self, y, *, tol, max_iter):
            state.solves += 1
            state.clock += 5.0
            return np.zeros_like(y), [True] * y.shape[1], [9] * y.shape[1], [0.] * y.shape[1]

    if payload is not _MISSING or getter_error is not None:
        def read_cost(self):
            state.clock += state.getter_delay
            if getter_error is not None:
                raise getter_error
            return payload if isinstance(payload, str) else json.dumps(payload)

        NativeSolver.cost_diagnostics_json = property(read_cost)

    module = SimpleNamespace(DualAbsorber=DualAbsorber, NativeSolver=NativeSolver)
    monkeypatch.setattr(backends, "_module", lambda _: module)
    monkeypatch.setattr(backends, "_provenance", lambda *_: {"test_backend": True})
    monkeypatch.setattr(backends, "perf_counter", lambda: state.clock)
    return state


@pytest.mark.parametrize("name", ["within", "fereg", "native_auto"])
def test_old_wheel_keeps_legacy_call_and_reports_unknown_cost(monkeypatch, name):
    state = _install_fake(monkeypatch, name)
    for _ in range(2):
        residual, metrics = backends.run_backend(name, _case())
        assert residual.shape == (4, 2) and metrics["converged"]
        assert metrics["native_cost_diagnostics"]["available"] is False
        assert metrics["native_cost_diagnostics"]["reason"] == "not_exposed_by_installed_wheel"
        assert metrics["actual_matvec_counts"] is None
        assert metrics["actual_preconditioner_apply_counts"] is None
        assert metrics["solver_lifecycle"] == "fresh_constructor_first_projection_call"
        json.dumps(metrics, allow_nan=False)
    assert state.constructors == state.solves == 2


@pytest.mark.parametrize("name", ["within", "fereg", "native_auto"])
def test_shared_attempts_and_cumulative_snapshots_are_retained_without_aggregation(monkeypatch, name):
    payload = _payload()
    original = copy.deepcopy(payload)
    _install_fake(monkeypatch, name, payload)
    _, metrics = backends.run_backend(name, _case())
    diagnostic = metrics["native_cost_diagnostics"]
    assert diagnostic["available"] is True
    for key, value in original.items():
        assert diagnostic[key] == value
    assert payload == original
    assert metrics["reported_iterations"] == [9, 9]
    assert metrics["actual_matvec_counts"] is None
    assert metrics["actual_preconditioner_apply_counts"] is None
    assert metrics["count_kind"] == "unavailable_native"
    # Later mutation of a wrapper's stored snapshot cannot rewrite this record.
    payload["after"]["normal_matvec_batches"] = 999
    assert diagnostic["after"]["normal_matvec_batches"] == 28
    json.dumps(metrics, allow_nan=False)


def test_getter_overhead_and_nested_factor_times_are_not_added_to_solver_total(monkeypatch):
    state = _install_fake(monkeypatch, "native_auto", _payload())
    state.getter_delay = 40.0
    _, metrics = backends.run_backend("native_auto", _case())
    assert metrics["setup_s"] == 3.0
    assert metrics["solve_s"] == 5.0
    assert metrics["total_s"] == 8.0
    assert metrics["diagnostics_collection_s"] == 40.0
    assert metrics["native_cost_diagnostics"]["constructor"]["elapsed_s"] == 2.0
    assert metrics["native_cost_diagnostics"]["attempts"][0]["delta_diagnostics"]["deferred_factorization_s"] == 1.25


@pytest.mark.parametrize("problem", ["cumulative_scope", "unknown_version", "nonfinite", "broken_json"])
def test_invalid_new_telemetry_is_not_reported_as_legacy_or_measured_cost(monkeypatch, problem):
    payload = _payload()
    if problem == "cumulative_scope":
        payload["scope"] = "solver_lifetime"
    elif problem == "unknown_version":
        payload["schema_version"] = 2
    elif problem == "nonfinite":
        payload["attempts"][0]["kernel_call_s"] = float("nan")
    else:
        payload = "{incomplete native diagnostic"
    _install_fake(monkeypatch, "native_auto", payload)
    residual, metrics = backends.run_backend("native_auto", _case())
    assert residual.shape == (4, 2) and metrics["converged"]
    diagnostic = metrics["native_cost_diagnostics"]
    assert diagnostic["available"] is False
    assert diagnostic["status"] == "invalid"
    assert diagnostic["reason"] == "invalid_diagnostic_payload"
    assert diagnostic["error"]
    json.dumps(diagnostic, allow_nan=False)


def test_failing_optional_getter_retains_residual_and_exposes_diagnostic_failure(monkeypatch):
    _install_fake(monkeypatch, "native_auto", getter_error=RuntimeError("snapshot unavailable"))
    residual, metrics = backends.run_backend("native_auto", _case())
    assert residual.shape == (4, 2)
    diagnostic = metrics["native_cost_diagnostics"]
    assert diagnostic["available"] is False and diagnostic["status"] == "invalid"
    assert diagnostic["reason"] == "diagnostic_read_failed"
    assert diagnostic["error"]["type"] == "RuntimeError"


def test_observed_cost_and_native_success_do_not_supply_a_forward_certificate(monkeypatch):
    _install_fake(monkeypatch, "fereg", _payload())
    case = _case()
    # These columns already have zero means in each FE dimension, so truth=y.
    # The fake native result is zero: moment-perfect but forward error one.
    case.truth = case.y.copy()
    residual, metrics = backends.run_backend("fereg", case)
    check = certification(case, residual)
    assert metrics["converged"] and metrics["native_cost_diagnostics"]["available"]
    assert check["stationarity_pass"]
    assert check["max_forward_relative_error"] == pytest.approx(1.0)
    assert not check["aggregate_pass"]


@pytest.mark.parametrize("name", ["within", "fereg", "native_auto"])
def test_instrumented_wheel_preserves_small_projection_oracle_when_available(name):
    if not backends.probe_backend(name)["available"]:
        pytest.skip(f"optional backend {name} is not installed")
    case = make_case("rigidity_grid", 4, rhs=2, seed=17, rhs_kind="broadband")
    residual, metrics = backends.run_backend(name, case, tol=1e-12, max_iter=2000, threads=1)
    diagnostic = metrics["native_cost_diagnostics"]
    if diagnostic.get("reason") == "not_exposed_by_installed_wheel":
        pytest.skip(f"installed {name} wheel predates cost diagnostics")
    assert diagnostic["available"] is True, diagnostic
    assert diagnostic["schema_version"] == 1 and diagnostic["scope"] == "call"
    assert isinstance(diagnostic["attempts"], list)
    assert metrics["actual_matvec_counts"] is None
    assert metrics["converged"] and certification(case, residual)["aggregate_pass"]
    json.dumps(metrics, allow_nan=False)
