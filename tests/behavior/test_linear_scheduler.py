"""Linear estimators share one host-resolved allocation for every stage."""
import importlib
import numpy as np
import pandas as pd
import pytest

from econhdfe import ExecutionConfig, InferenceConfig, OLSHDFESession, IVHDFESession, olshdfe, ivhdfe
from econhdfe.planner import scheduler
from econhdfe.planner.resources import RuntimeResources


@pytest.fixture
def data():
    rng = np.random.default_rng(34)
    n = 200
    z = rng.normal(size=n)
    return pd.DataFrame({"y": rng.normal(size=n), "x": z + rng.normal(size=n),
                         "z": z, "g": np.arange(n) // 10})


def specification(iv):
    base = dict(y="y", absorb=["g"])
    base.update(dict(exog=[], endog=["x"], instruments=["z"]) if iv else dict(x=["x"]))
    return base


@pytest.fixture
def host(monkeypatch):
    state = {"resources": RuntimeResources(4, 16 * 1024**3, 4 * 1024**3)}
    monkeypatch.setattr(scheduler, "runtime_resources", lambda: state["resources"])
    return state


@pytest.mark.parametrize("iv", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_linear_stages_share_auto_or_explicit_allocation(data, host, monkeypatch, iv, explicit):
    module = importlib.import_module("econhdfe.models.linear_iv.api" if iv else "econhdfe.models.ols")
    observed = []
    for name in ("materialize_model_data", "make_standard_absorber",
                 "fit_iv_kclass" if iv else "fit_ols_in_basis"):
        original = getattr(module, name)
        def capture(*args, _original=original, _name=name, **kwargs):
            allocation = scheduler.current_allocation()
            assert allocation is not None, _name
            observed.append((_name, allocation))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, capture)
    config = ExecutionConfig(threads=1, memory_budget_mb=64 if explicit else "auto")
    (ivhdfe if iv else olshdfe)(data, **specification(iv), execution_config=config,
                              inference_config=InferenceConfig(iv_advanced_statistics=False))
    assert len(observed) == 3
    first = observed[0][1]
    assert all(allocation is first for _, allocation in observed)
    if explicit:
        assert first.config.memory_budget_mb == 64
    else:
        assert first.config.memory_budget_mb > 512
    assert first.cpu_threads == 1
    assert scheduler.current_allocation() is None


@pytest.mark.parametrize("iv", [False, True])
def test_legacy_resource_limits_are_preserved(data, host, monkeypatch, iv):
    module = importlib.import_module("econhdfe.models.linear_iv.api" if iv else "econhdfe.models.ols")
    original = module.make_standard_absorber
    seen = []
    def capture(*args, **kwargs):
        seen.append((kwargs["memory_budget_mb"], kwargs["absorb_threads"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "make_standard_absorber", capture)
    (ivhdfe if iv else olshdfe)(data, **specification(iv), memory_budget_mb=48, absorb_threads=1,
                              inference_config=InferenceConfig(iv_advanced_statistics=False))
    assert seen == [(48, 1)]


@pytest.mark.parametrize("iv", [False, True])
def test_session_refreshes_compiled_resources_without_discarding_within(data, host, iv):
    Session = IVHDFESession if iv else OLSHDFESession
    config = ExecutionConfig(threads=1)
    session = Session(data, execution_config=config,
                      inference_config=InferenceConfig(iv_advanced_statistics=False))
    session.fit(**specification(iv))
    entry = session._entries[("g",)]
    old_absorber = entry.absorber
    old_budget = entry.memory_budget_mb
    old_within = dict(entry.within)
    assert old_budget > 512
    assert session.execution is config  # policy stays auto between calls
    host["resources"] = RuntimeResources(4, 16 * 1024**3, 12 * 1024**3)
    session.fit(**specification(iv))
    assert session._entries[("g",)] is entry
    assert entry.memory_budget_mb < old_budget
    assert entry.absorber is not old_absorber
    assert all(entry.within[name] is values for name, values in old_within.items())
    old_absorber = entry.absorber
    session.execution = ExecutionConfig(threads=2, memory_budget_mb=64)
    session.fit(**specification(iv))
    assert entry.memory_budget_mb == 64
    assert entry.execution_threads == 2
    assert entry.absorber is not old_absorber
    assert entry.absorber.projection_memory_budget_mb == 64
    assert scheduler.current_allocation() is None


@pytest.mark.parametrize("iv", [False, True])
def test_session_source_and_solver_run_inside_one_allocation(data, host, monkeypatch, iv):
    from econhdfe.data.source import DataFrameSource
    import econhdfe.sessions as module
    observed = []
    for name in ("prepare_repeated_dataset", "residualize_blocks",
                 "fit_iv_kclass" if iv else "fit_ols_in_basis"):
        original = getattr(module, name)
        def capture(*args, _original=original, _name=name, **kwargs):
            allocation = scheduler.current_allocation()
            assert allocation is not None
            if _name == "prepare_repeated_dataset":
                assert kwargs["memory_budget_mb"] > 512
            observed.append((_name, allocation))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, capture)
    Session = IVHDFESession if iv else OLSHDFESession
    config = ExecutionConfig(threads=1)
    session = Session(DataFrameSource(data), execution_config=config,
                      inference_config=InferenceConfig(iv_advanced_statistics=False))
    assert not observed  # source materialization is deferred to managed fit
    session.fit(**specification(iv))
    assert len(observed) == 3
    assert all(allocation is observed[0][1] for _, allocation in observed)
    assert session.execution is config
    assert scheduler.current_allocation() is None


@pytest.mark.parametrize("iv", [False, True])
def test_persistent_results_survive_auto_headroom_change(data, host, tmp_path, iv):
    from econhdfe.data import CSVSource
    path = tmp_path / "source.csv"
    data.to_csv(path, index=False)
    Session = IVHDFESession if iv else OLSHDFESession
    def create():
        return Session(CSVSource(path), execution_config=ExecutionConfig(threads=1),
                       inference_config=InferenceConfig(iv_advanced_statistics=False)
                       ).enable_persistent_cache(tmp_path / "cache")
    first = create().fit(**specification(iv))
    host["resources"] = RuntimeResources(4, 16 * 1024**3, 12 * 1024**3)
    session = create()
    second = session.fit(**specification(iv))
    assert session.persistent_cache_info()["result_hits"] == 1
    np.testing.assert_allclose(first.params, second.params)


def test_queued_session_reads_auto_policy_after_admission(host, monkeypatch):
    from threading import Event, RLock, Thread, current_thread
    entered, queued, release = Event(), Event(), Event()
    class ObservedLock:
        def __init__(self):
            self.lock = RLock()
        def __enter__(self):
            if current_thread().name == "queued-fit":
                queued.set()
            self.lock.acquire()
            return self
        def __exit__(self, *args):
            self.lock.release()
    monkeypatch.setattr(scheduler, "_ROOT_LOCK", ObservedLock())
    policy = ExecutionConfig(threads=1)
    observed, errors = [], []
    class Session:
        execution = policy
        @scheduler.managed_execution
        def fit(self, first=False):
            observed.append(self.execution.memory_budget_mb)
            if first:
                entered.set()
                assert release.wait(5)
    session = Session()
    host["resources"] = RuntimeResources(4, 16 * 1024**3, 12 * 1024**3)
    def run(first):
        try:
            session.fit(first=first)
        except BaseException as exc:
            errors.append(exc)
    first = Thread(target=run, args=(True,), name="active-fit")
    second = Thread(target=run, args=(False,), name="queued-fit")
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert queued.wait(5)
        host["resources"] = RuntimeResources(4, 16 * 1024**3, 4 * 1024**3)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert len(observed) == 2 and observed[1] > observed[0]
    assert session.execution is policy
