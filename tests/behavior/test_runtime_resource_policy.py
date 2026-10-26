"""Regression checks for resource limits at execution boundaries."""
import numpy as np
import pytest

from econhdfe import ExecutionConfig, PPMLConfig, ppmlhdfe
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.group_individual import GroupIndividualAbsorber
from econhdfe.planner.resources import RuntimeResources


def test_ppml_explicit_threads_obey_current_cpu_capacity(monkeypatch):
    from econhdfe.hdfe import projection
    monkeypatch.setattr(projection, "effective_cpu_count", lambda: 1)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(300, 1))
    y = rng.poisson(np.exp(0.1 * x[:, 0]))
    result = ppmlhdfe(
        y=y, x=x, absorb=[np.arange(300) % 15, np.arange(300) % 17],
        config=PPMLConfig(engine="optimized"),
        execution_config=ExecutionConfig(threads=4, memory_budget_mb=32),
    )
    assert result.diagnostics["projection_resources"]["actual_threads"] == 1


@pytest.mark.parametrize("child_budget,expected", [(None, .004), (.002, .002), (128, .004)])
def test_fused_scratch_inherits_or_tightens_parent_budget(child_budget, expected):
    groups = np.arange(1000) % 100
    absorber = HDFEAbsorber(
        [groups], absorb_threads=1, projection_memory_budget_mb=.004,
        fused_rhs_memory_mb=child_budget,
    )
    rhs = np.random.default_rng(42).normal(size=(1000, 20))
    within = absorber.residualize(rhs)
    assert absorber.fused_rhs_memory_mb == expected
    assert absorber._projection_scratch.nbytes <= expected * 1024**2
    # Tiling must preserve the exact one-way projection.
    means = np.stack([rhs[groups == g].mean(axis=0) for g in range(100)])
    np.testing.assert_allclose(within, rhs - means[groups], atol=1e-14)


def test_group_individual_low_budget_selects_matrix_free():
    kwargs = dict(
        standard_groups=[], membership_group=np.repeat(np.arange(100), 2),
        individual_codes=np.arange(200) % 20,
    )
    bounded = GroupIndividualAbsorber(**kwargs, memory_budget_mb=.001)
    roomy = GroupIndividualAbsorber(**kwargs, memory_budget_mb=64)
    assert bounded.incidence_backend == "matrix_free"
    assert roomy.incidence_backend == "csr"
    vector = np.random.default_rng(42).normal(size=20)
    np.testing.assert_allclose(bounded._operator @ vector, roomy._operator @ vector)


def test_calibration_checks_headroom_before_probe_allocation(monkeypatch):
    from econhdfe.planner import calibration
    monkeypatch.setattr(calibration, "_effective_numba_cap", lambda value=None: 2)
    monkeypatch.setattr(calibration, "runtime_thread_fingerprint", lambda **kwargs: "test")
    monkeypatch.setattr(calibration, "runtime_resources", lambda: RuntimeResources(
        2, 64 * 1024**2, 60 * 1024**2,
    ))
    def forbidden(*args, **kwargs):
        pytest.fail("calibration allocated before checking memory headroom")
    monkeypatch.setattr(calibration.np, "empty", forbidden)
    with pytest.raises(MemoryError, match="headroom"):
        calibration.calibrate_threads(max_threads=2)
