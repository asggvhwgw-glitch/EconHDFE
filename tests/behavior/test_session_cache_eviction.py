import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, InferenceConfig, OLSHDFESession, IVHDFESession, olshdfe, ivhdfe
from econhdfe.compute.cache_accounting import retained_array_bytes


@pytest.fixture
def data():
    rng = np.random.default_rng(2026)
    n = 300
    z = rng.normal(size=n)
    return pd.DataFrame(dict(y=rng.normal(size=n), x=z+rng.normal(size=n), z=z,
                             a=np.arange(n)%10, b=np.arange(n)%13, c=np.arange(n)%17))


def spec(iv, fe):
    return dict(y="y", absorb=[fe], **(dict(exog=[], endog=["x"], instruments=["z"])
                                      if iv else dict(x=["x"])))


def session(data, iv=False, cache="auto"):
    return (IVHDFESession if iv else OLSHDFESession)(
        data, execution_config=ExecutionConfig(threads=1, memory_budget_mb=64, cache=cache),
        inference_config=InferenceConfig(iv_advanced_statistics=False))


def test_array_views_count_shared_owner_once_and_exclude_input_storage():
    source = np.arange(100, dtype=float)
    other = np.arange(40, dtype=float)
    cached = {"a": source[::2], "b": source[20:], "owned": other[3:]}
    assert retained_array_bytes(cached, ()) == source.nbytes + other.nbytes
    assert retained_array_bytes(cached, source[1:]) == other.nbytes


@pytest.mark.parametrize("iv", [False, True])
def test_lru_keeps_hot_entry_and_evicts_cold_without_changing_results(data, monkeypatch, iv):
    s = session(data, iv)
    s.fit(**spec(iv, "a")); s.fit(**spec(iv, "b")); s.fit(**spec(iv, "a"))
    hot = s._entries[s._entry_key(("a",))]
    within = hot.within["y"]
    old = s.cache_info()["retained_bytes"]
    monkeypatch.setattr(s, "_cache_limit_bytes", lambda: old - 1)
    s._prepare_resource_admission()
    assert s._entry_key(("b",)) not in s._entries
    assert s._entries[s._entry_key(("a",))] is hot and hot.within["y"] is within
    assert s.cache_info()["evictions"] >= 1
    assert s.cache_info()["retained_bytes"] < old
    calls = hot.residualize_calls
    s.fit(**spec(iv, "a"))
    assert hot.residualize_calls == calls
    assert hot.within["y"] is within
    monkeypatch.setattr(s, "_cache_limit_bytes", lambda: 8 * 1024**2)
    rebuilt = s.fit(**spec(iv, "b"))
    reference = (ivhdfe if iv else olshdfe)(
        data, **spec(iv, "b"), execution_config=ExecutionConfig(threads=1),
        inference_config=InferenceConfig(iv_advanced_statistics=False))
    np.testing.assert_allclose(rebuilt.params, reference.params, atol=1e-10)
    np.testing.assert_allclose(rebuilt.vcov, reference.vcov, atol=1e-10)


def test_fit_many_rebuilds_evicted_prewarmed_entries(data, monkeypatch):
    s = session(data)
    monkeypatch.setattr(s, "_cache_limit_bytes", lambda: 1)
    specs = [spec(False, fe) for fe in ("a", "b", "c", "a")]
    result = s.fit_many(specs)
    reference_session = session(data)
    reference = reference_session.fit_many(specs)
    for got, expected in zip(result, reference, strict=True):
        np.testing.assert_allclose(got.params, expected.params, atol=1e-10)
        np.testing.assert_allclose(got.vcov, expected.vcov, atol=1e-10)
    assert s.cache_info()["fe_entries"] == 0
    assert s.cache_info()["retained_bytes"] == 0
    assert s.cache_info()["evictions"] > 0


def test_pressure_recovery_preserves_inputs_and_persistent_disk(data, monkeypatch, tmp_path):
    original = data.copy(deep=True)
    s = session(data).enable_persistent_cache(tmp_path / "cache")
    first = s.fit(**spec(False, "a"))
    files = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(s, "_cache_limit_bytes", lambda: 0)
    s._prepare_resource_admission()
    assert s.cache_info()["retained_bytes"] == 0
    assert files == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    pd.testing.assert_frame_equal(data, original)
    again = s.fit(**spec(False, "a"))
    np.testing.assert_allclose(first.params, again.params)
    assert s.persistent_cache_info()["result_hits"] == 1


def test_cache_off_still_discards_all_rebuildable_arrays(data):
    s = session(data, cache="off")
    s.fit(**spec(False, "a"))
    assert s.cache_info()["retained_bytes"] == 0
    assert not s._entries and not s._raw_full


def test_auto_to_explicit_same_thread_cap_rebuilds_absorber(data, monkeypatch):
    from econhdfe.planner import scheduler
    from econhdfe.planner.resources import RuntimeResources
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: RuntimeResources(4, 8*1024**3, 1024**3))
    s = session(data)
    s.execution = ExecutionConfig(threads="auto", memory_budget_mb=64)
    s.fit(**spec(False, "a"))
    entry = s._entries[s._entry_key(("a",))]
    absorber = entry.absorber
    within = entry.within["y"]
    assert entry.execution_auto_threads
    assert entry.execution_threads == 4
    s.execution = ExecutionConfig(threads=4, memory_budget_mb=64)
    s.fit(**spec(False, "a"))
    assert not entry.execution_auto_threads
    assert entry.absorber is not absorber
    assert entry.within["y"] is within


def test_pre_admission_reclaims_before_low_headroom_reject(data, monkeypatch):
    import econhdfe.sessions as module
    from econhdfe.planner import scheduler
    from econhdfe.planner.resources import RuntimeResources
    s = session(data)
    s.fit(**spec(False, "a"))
    assert s._entries
    def resources():
        total = 1024**3
        available = 16*1024**2 if s._entries else 256*1024**2
        return RuntimeResources(2, total, total - int(total*.12) - available)
    monkeypatch.setattr(module, "runtime_resources", resources)
    monkeypatch.setattr(scheduler, "runtime_resources", resources)
    result = s.fit(**spec(False, "b"))
    assert result.converged
    assert s.cache_info()["evictions"] >= 1
