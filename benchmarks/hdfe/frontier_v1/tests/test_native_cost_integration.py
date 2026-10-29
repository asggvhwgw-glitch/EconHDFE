"""Small real-wheel gates for observational instrumentation and call scopes."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.hdfe.frontier_v1.native_cost import study
from benchmarks.hdfe.frontier_v1.families import make_case
from benchmarks.hdfe.frontier_v1.validation import certification


@pytest.fixture(params=study.NATIVE_BACKENDS)
def native(request):
    name = request.param
    try:
        module = study.backends._module(name)
        supported = getattr(module, "COST_DIAGNOSTICS_SCHEMA", None) == 1
    except study.backends.BackendUnavailable:
        supported = False
    if not supported:
        if os.environ.get("ECONHDFE_REQUIRE_NATIVE_DIAGNOSTICS") == "1":
            pytest.fail(f"research diagnostics schema 1 wheel required: {name}")
        pytest.skip(f"optional instrumented {name} wheel is not installed")
    return name


def _new(name, case, enabled=True):
    solver, y, _, _ = study.create_solver(name, case, diagnostics=enabled, threads=1)
    return solver, y


def _check_deltas(diagnostic):
    def compare(before, after, delta):
        for key, value in delta.items():
            assert not key.startswith("constructor_")
            if isinstance(value, dict):
                compare(before[key], after[key], value)
            else:
                assert value >= 0
                assert value == pytest.approx(after[key] - before[key], rel=1e-12, abs=1e-14)
    for attempt in diagnostic["attempts"]:
        cost = attempt.get("native_cost", attempt)
        compare(cost["before"]["totals"], cost["after"]["totals"], cost["delta"])


def _zero_work(value):
    return all(_zero_work(v) for v in value.values()) if isinstance(value, dict) else value in (None, 0)


def test_toggle_and_row_permutation_preserve_independent_weighted_svd(native):
    rng = np.random.default_rng(91)
    groups = np.column_stack([rng.integers(0, g, 96) for g in (7, 5, 6)])
    y, weights = rng.normal(size=(96, 2)), np.exp(rng.normal(size=96))
    design = np.column_stack([np.eye(g)[groups[:, j]] for j, g in enumerate((7, 5, 6))])
    root_w = np.sqrt(weights)[:, None]
    beta = np.linalg.lstsq(root_w * design, root_w * y, rcond=None)[0]
    expected, observed = y - design @ beta, []
    for enabled, order, permuted in ((True, np.arange(96), False), (False, np.arange(96), False),
                                     (True, rng.permutation(96), True)):
        factors, rhs = ([2, 0, 1], [1, 0]) if permuted else ([0, 1, 2], [0, 1])
        case = SimpleNamespace(groups=groups[order][:, factors], y=y[order][:, rhs], weights=weights[order])
        solver, data = _new(native, case, enabled)
        original = data.copy()
        residual, metrics = study.solve_once(native, solver, data, tol=1e-12)
        np.testing.assert_array_equal(data, original)
        assert metrics["native_converged"]
        diagnostic = metrics["native_cost_diagnostics"]
        assert diagnostic["available"] and diagnostic["enabled"] is enabled
        if not enabled:
            assert _zero_work(diagnostic.get("wrapper_delta", {}))
            assert all(_zero_work(a.get("native_cost", a)["delta"]) for a in diagnostic["attempts"])
        restored = residual[np.argsort(order)][:, np.argsort(rhs)]
        np.testing.assert_allclose(restored, expected, atol=2e-9, rtol=2e-9)
        observed.append(restored)
    np.testing.assert_allclose(observed[0], observed[1], atol=2e-12, rtol=2e-12)


def test_reused_solver_reports_call_deltas_and_reuses_built_state(native):
    case = make_case("rigidity_grid", 8, rhs=2, seed=23, rhs_kind="broadband")
    solver, data = _new(native, case)
    diagnostics = []
    for _ in range(2):
        residual, metrics = study.solve_once(native, solver, data)
        assert metrics["native_converged"] and certification(case, residual)["aggregate_pass"]
        diagnostic = metrics["native_cost_diagnostics"]
        assert diagnostic["available"] and diagnostic["scope"] == "call"
        _check_deltas(diagnostic)
        first = diagnostic["attempts"][0]
        delta = first.get("native_cost", first)["delta"]
        assert (delta["residualize_calls"] if native == "fereg" else delta["solve"]["calls"]) > 0
        diagnostics.append(diagnostic)
    cold, warm = diagnostics
    assert warm["core_before"] == cold["core_after"]
    assert warm["constructor"] == cold["constructor"]
    if native != "native_auto":
        assert len(cold["kernel_builds"]) == 1 and warm["kernel_builds"] == []
    else:
        assert warm["attempts"][0]["delta"]["deferred_build"]["calls"] == 0
    if native == "fereg":
        for attempt in warm["attempts"]:
            report = attempt["core_report"]
            assert report["iteration_scope"] == "one_shared_lockstep_batch"
            assert attempt["iterations"] == [report["shared_iterations"]] * attempt["rhs_width"]
            delta = attempt["native_cost"]["delta"]
            assert delta["matvec_rhs_columns"] == delta["matvec_calls"] * attempt["rhs_width"]


def test_failed_call_does_not_leak_error_or_previous_attempts_into_next_call(native):
    case = make_case("rigidity_grid", 8, rhs=2, seed=19, rhs_kind="broadband")
    solver, data = _new(native, case)
    study.solve_once(native, solver, data)
    bad = data.copy()
    bad[0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        study.solve_once(native, solver, bad)
    failed = study.read_diagnostics(native, solver)
    assert failed["available"]
    outcome = failed if native == "native_auto" else solver.diagnostics
    assert outcome["error"]
    residual, metrics = study.solve_once(native, solver, data)
    recovered = metrics["native_cost_diagnostics"]
    outcome = recovered if native == "native_auto" else solver.diagnostics
    assert not outcome.get("error")
    assert recovered["core_before"] == failed["core_after"]
    _check_deltas(recovered)
    assert metrics["native_converged"] and certification(case, residual)["aggregate_pass"]
