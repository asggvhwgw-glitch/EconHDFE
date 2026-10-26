"""Public linear resource policies must reach the actual execution branch."""
import numpy as np
import pandas as pd
import pytest

from econhdfe import (
    ExecutionConfig, HDFEConfig, InferenceConfig, OLSHDFESession,
    IVHDFESession, factor, reg_interaction, olshdfe, ivhdfe,
)
import econhdfe.pipeline as pipeline
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
    original = pipeline.HDFEAbsorber
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "HDFEAbsorber", capture)
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
    with pytest.raises(RuntimeError, match="did not converge"):
        fit(data, **specification(iv, block=True), **kw)
    result = fit(data, **specification(iv, block=True), allow_nonconverged=True, **kw)
    assert not result.converged
    assert result.absorb_info["iterations"] <= 10


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
