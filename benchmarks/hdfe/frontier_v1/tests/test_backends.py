"""Independent oracles and accounting contracts in the isolated frontier suite."""
import importlib
import json

import numpy as np
import pytest

from benchmarks.hdfe.frontier_v1.backends import (
    BACKEND_NAMES, BackendUnavailable, probe_backend, run_backend,
)


def _case(seed=81):
    rng = np.random.default_rng(seed)
    # Disconnected components and exactly redundant factors deliberately make
    # the dummy design rank deficient. No reference-column deletion is assumed.
    a = rng.integers(0, 7, 96)
    b = rng.integers(0, 5, 96)
    component = np.repeat([0, 1], 48)
    codes = np.column_stack([a + 7 * component, b + 5 * component, a + 7 * component])
    y = rng.normal(size=(96, 3))
    w = np.exp(rng.normal(scale=.4, size=96))
    return {"codes": codes, "rhs": y, "weights": w}


def _dense_reference(case):
    codes, y, w = case["codes"], case["rhs"], case["weights"]
    blocks = []
    for j in range(codes.shape[1]):
        _, inverse = np.unique(codes[:, j], return_inverse=True)
        blocks.append(np.eye(int(inverse.max()) + 1)[inverse])
    d = np.column_stack(blocks)
    beta = np.linalg.lstsq(np.sqrt(w)[:, None] * d, np.sqrt(w)[:, None] * y, rcond=None)[0]
    return y - d @ beta


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_rank_deficient_weighted_projection_matches_independent_svd(name):
    if not probe_backend(name)["available"]:
        pytest.skip(f"optional backend {name} is not installed")
    case = _case()
    expected = _dense_reference(case)
    actual, metrics = run_backend(name, case, tol=1e-12, max_iter=2000, threads=1)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
    assert metrics["converged"] and metrics["output_finite"]
    assert metrics["total_s"] == pytest.approx(metrics["setup_s"] + metrics["solve_s"])
    json.dumps(metrics, allow_nan=False)
    if name == "scipy_lsmr":
        count = metrics["actual_matvec_counts"]
        assert count["total"] == count["matvec"] + count["rmatvec"] > 0
        assert count["total"] == sum(d["matvec"] + d["rmatvec"] for d in metrics["solver_details"])
    else:
        assert metrics["actual_matvec_counts"] is None
        assert metrics["count_kind"] == "unavailable_native"


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_sparse_labels_preserve_rows_and_vector_shape(name):
    if not probe_backend(name)["available"]:
        pytest.skip(f"optional backend {name} is not installed")
    case = _case()
    expected = _dense_reference(case)[:, 0]
    case["codes"] = case["codes"].astype(np.uint32) * 100003 + 17
    case["rhs"] = case["rhs"][:, 0]
    actual, metrics = run_backend(name, case, tol=1e-12, threads=1)
    assert actual.shape == (96,)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)


def test_unavailable_native_is_not_silently_replaced(monkeypatch):
    original = importlib.import_module

    def denied(name, *args, **kwargs):
        if name == "econhdfe_dual":
            raise ImportError("intentional unavailable optional extension")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", denied)
    assert not probe_backend("fereg")["available"]
    with pytest.raises(BackendUnavailable):
        run_backend("fereg", _case())


@pytest.mark.parametrize("name", ["scipy_lsmr", "native_auto"])
def test_iteration_cap_returns_honest_nonconvergence(name):
    if not probe_backend(name)["available"]:
        pytest.skip(f"optional backend {name} is not installed")
    rng = np.random.default_rng(203)
    case = {"codes": rng.integers(0, 100, size=(800, 3)), "rhs": rng.normal(size=(800, 2))}
    output, metrics = run_backend(name, case, tol=1e-13, max_iter=1, threads=1)
    assert output.shape == (800, 2)
    assert not metrics["converged"]
    assert not all(metrics["flags"])
    assert all(i <= 1 for i in metrics["reported_iterations"])


@pytest.mark.parametrize("weight", [0., -1., float("nan"), float("inf")])
def test_common_positive_weight_contract(weight):
    case = _case()
    case["weights"][0] = weight
    with pytest.raises(ValueError, match="strictly positive"):
        run_backend("scipy_lsmr", case)
