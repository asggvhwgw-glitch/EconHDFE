"""Public linear resource policies must reach the actual execution branch."""
import numpy as np
import pandas as pd
import pytest

from econhdfe import (
    ExecutionConfig, HDFEConfig, InferenceConfig, OLSHDFESession,
    IVHDFESession, factor, reg_interaction, olshdfe, ivhdfe,
)
from econhdfe.errors import ConvergenceError
from econhdfe.hdfe.absorber import HDFEConvergenceError
import econhdfe.hdfe.factory as factory
import econhdfe.sessions as sessions


@pytest.fixture
def data():
    rng = np.random.default_rng(12)
    n = 800
    g = np.arange(n) // 80
    df = pd.DataFrame({
        "y": rng.normal(size=n), "z": rng.normal(size=n), "g": g,
        "h": g * 8 + rng.integers(0, 8, n),
        "k": g * 7 + rng.integers(0, 7, n), "id": np.arange(n),
    })
    df["x"] = df.z + rng.normal(size=n)
    return df


def specification(iv, *, block=False):
    role = lambda col: reg_interaction(factor("g", drop_base=False), col) if block else col
    spec = dict(y="y", absorb=["g", "h", "k"] if block else ["g"])
    spec.update(dict(exog=[], endog=[role("x")], instruments=[role("z")])
                if iv else dict(x=[role("x")]))
    return spec


@pytest.mark.parametrize("iv", [False, True])
def test_group_only_forwards_projection_resources(data, monkeypatch, iv):
    original = factory.HDFEAbsorber
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(factory, "HDFEAbsorber", capture)
    fit = ivhdfe if iv else olshdfe
    result = fit(data, **specification(iv), group="id", projection_backend="fused",
                 execution_config=ExecutionConfig(threads=2, memory_budget_mb=32),
                 inference_config=InferenceConfig(iv_advanced_statistics=False))
    assert result.converged
    assert len(seen) == 1
    assert seen[0]["absorb_threads"] == 2
    assert seen[0]["projection_memory_budget_mb"] == 32
    assert seen[0]["projection_backend"] == "fused"


@pytest.mark.parametrize("iv", [False, True])
def test_block_design_honors_iteration_limit(data, iv):
    fit = ivhdfe if iv else olshdfe
    kw = dict(hdfe_config=HDFEConfig(solver="map", max_iter=1, canonicalize=False),
              execution_config=ExecutionConfig(threads=1),
              inference_config=InferenceConfig(iv_advanced_statistics=False))
    message = "heterogeneous-spec IV FE projection" if iv else "heterogeneous-spec FE projection"
    with pytest.raises(HDFEConvergenceError, match=f"^{message} did not converge$") as info:
        fit(data, **specification(iv, block=True), **kw)
    assert isinstance(info.value, ConvergenceError)
    assert info.value.code == "convergence.hdfe" and info.value.stage == "hdfe"
    assert info.value.details["iterations"] >= 1 and "metric" in info.value.details
    result = fit(data, **specification(iv, block=True), allow_nonconverged=True, **kw)
    assert not result.converged
    assert result.absorb_info["iterations"] <= 10


@pytest.mark.parametrize("iv", [False, True])
def test_group_mode_nonconvergence_raises_convergence_error(iv):
    rng = np.random.default_rng(503)
    n_groups, n_ind = 120, 30
    group = np.repeat(np.arange(n_groups), 3)
    individual = np.asarray([(g + s) % n_ind for g in range(n_groups) for s in (0, 7, 13)])
    zg = rng.normal(size=n_groups)
    dg = zg + rng.normal(size=n_groups)
    df = pd.DataFrame({
        "group": group, "ind": individual, "year": (np.arange(n_groups) % 8)[group],
        "y": (0.5 * dg + rng.normal(size=n_groups))[group], "d": dg[group], "z": zg[group],
    })
    spec = dict(exog=[], endog=["d"], instruments=["z"]) if iv else dict(x=["d"])
    fit = ivhdfe if iv else olshdfe
    kw = dict(y="y", absorb=["year", "ind"], group="group", individual="ind", method="lsmr",
              drop_singletons=False, max_iter=1, **spec)
    with pytest.raises(HDFEConvergenceError, match="^group/individual FE absorption did not converge$") as info:
        fit(df, **kw)
    assert isinstance(info.value, ConvergenceError)
    assert info.value.code == "convergence.hdfe" and info.value.details["iterations"] >= 1
    assert not fit(df, **kw, allow_nonconverged=True).converged


@pytest.mark.parametrize("iv", [False, True])
def test_cache_off_discards_between_calls_and_overrides_persistent(data, monkeypatch, tmp_path, iv):
    Session = IVHDFESession if iv else OLSHDFESession
    session = Session(data, execution_config=ExecutionConfig(threads=1, cache="off"),
                      inference_config=InferenceConfig(iv_advanced_statistics=False))
    session.enable_persistent_cache(tmp_path / "cache")
    store = session._persistent_store
    def forbidden(*args, **kwargs):
        pytest.fail("cache=off accessed persistent storage")
    for name in ("load_result", "save_result", "load_within", "save_within"):
        monkeypatch.setattr(type(store), name, forbidden)
    calls = []
    original = sessions.residualize_blocks
    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sessions, "residualize_blocks", capture)
    spec = specification(iv)
    first = session.fit(**spec)
    second = session.fit(**spec)
    assert len(calls) == 2
    np.testing.assert_allclose(first.params, second.params)
    assert session.cache_info()["fe_entries"] == 0
    assert session.cache_info()["within_columns"] == 0
    assert not session._raw_full and not session._component_cache
    assert session._persistent_store is store
    calls.clear()
    results = session.fit_many([spec, spec])
    assert len(results) == 2
    assert len(calls) == 1  # temporary reuse remains useful within the batch
    assert not session._entries and not session._raw_full
    session.prewarm(absorb=["g"], columns=["y", "x"])
    assert not session._entries
    with pytest.raises(ValueError):
        session.fit(**dict(spec, y="missing"))
    assert not session._entries and not session._raw_full
    assert session._persistent_store is store
