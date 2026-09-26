from __future__ import annotations

from enum import Enum
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import version as versioning


def test_public_version_accepts_patch_and_revision_forms():
    assert versioning._validate_public("0.4.4").release == (0, 4, 4)
    assert versioning._validate_public("0.4.4.1").release == (0, 4, 4, 1)
    assert versioning._validate_public("0.4.4.1rc1").release == (0, 4, 4, 1)
    assert versioning.version_tier("0.4.4") == "patch-or-higher"
    assert versioning.version_tier("0.4.4.1") == "revision"


@pytest.mark.parametrize("bad", ["0.4", "0.4.4.0", "0.4.4.1.1", "0.4.4+local"])
def test_public_version_rejects_noncanonical_forms(bad):
    with pytest.raises(SystemExit):
        versioning._validate_public(bad)


def test_next_version_respects_four_tier_policy():
    assert versioning.next_version("revision", "0.4.4") == "0.4.4.1"
    assert versioning.next_version("revision", "0.4.4.1") == "0.4.4.2"
    assert versioning.next_version("patch", "0.4.4.7") == "0.4.5"
    assert versioning.next_version("minor", "0.4.4.7") == "0.5.0"
    assert versioning.next_version("major", "0.4.4.7") == "1.0.0"


def test_revision_transition_stays_on_base_and_is_sequential():
    versioning._validate_transition(Version("0.4.4"), Version("0.4.4.1"))
    versioning._validate_transition(Version("0.4.4.1"), Version("0.4.4.2"))
    versioning._validate_transition(Version("0.4.4.2"), Version("0.4.5"))
    with pytest.raises(SystemExit, match="sequential"):
        versioning._validate_transition(Version("0.4.4"), Version("0.4.4.2"))
    with pytest.raises(SystemExit, match="current MAJOR.MINOR.PATCH base"):
        versioning._validate_transition(Version("0.4.4"), Version("0.4.5.1"))


def test_revision_gate_rejects_public_contract_changes(monkeypatch):
    monkeypatch.setattr(versioning, "check", lambda **kwargs: "0.4.4.1")
    monkeypatch.setattr(versioning, "check_manifest", lambda **kwargs: None)
    monkeypatch.setattr(
        versioning,
        "check_compatibility",
        lambda **kwargs: [{"id": "x", "path": "exports", "kind": "changed"}],
    )
    monkeypatch.setattr(versioning, "check_skill", lambda: None)
    with pytest.raises(SystemExit, match="REVISION releases may not change the public contract"):
        versioning.gate()


def _compatibility():
    spec = importlib.util.spec_from_file_location("closeout_contract", ROOT / "scripts/compatibility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inherited_enum_signature_is_interpreter_independent(monkeypatch):
    compat = _compatibility()

    class Role(str, Enum):
        OUTCOME = "outcome"

    native = inspect.signature
    for reflected in ("(value, names=None, *, module=None)", "(*values)"):
        monkeypatch.setattr(compat.inspect, "signature", lambda obj: reflected if obj is Role else native(obj))
        assert compat._public_signature(Role) == "(*values)"


def test_custom_signatures_are_not_hidden_by_enum_normalization():
    compat = _compatibility()

    class CustomMeta(type(Enum)):
        def __call__(cls, value, *, option=False):
            return super().__call__(value)

    class CustomRole(str, Enum, metaclass=CustomMeta):
        OUTCOME = "outcome"

    assert "option=False" in compat._public_signature(CustomRole)

    class Role(str, Enum):
        OUTCOME = "outcome"

    Role.__signature__ = inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_ONLY)])
    assert compat._public_signature(Role) == "(value, /)"


def test_real_function_contract_change_still_fails():
    compat = _compatibility()
    old = {"signatures": {"fit": "(x)"}}
    new = {"signatures": {"fit": "(x, *, unsafe=False)"}}
    assert [x["path"] for x in compat.diff_contracts(old, new)] == ["signatures.fit"]


@pytest.mark.parametrize("layer", ["workqueue", "omp", "tbb"])
def test_thread_mask_restored_after_nested_exception(monkeypatch, layer):
    from numba import get_num_threads
    from econhdfe.hdfe import projection

    monkeypatch.setattr(projection, "threading_layer", lambda: layer)
    old = get_num_threads()
    with pytest.raises(RuntimeError, match="sentinel"):
        with projection.numba_thread_limit(1):
            assert get_num_threads() == 1
            with projection.numba_thread_limit(2):
                assert get_num_threads() == min(2, projection.config.NUMBA_NUM_THREADS)
                raise RuntimeError("sentinel")
    assert get_num_threads() == old


def test_workqueue_parallel_bootstrap_in_fresh_process():
    # A regression aborts the child, never the entire test runner. These are the
    # original weighted/unweighted serial-versus-parallel numerical assertions.
    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue", NUMBA_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUTF8="1")
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_cluster_inference_v046.py::test_weighted_wild_bootstrap_parallel_reproducibility_and_distributions",
        "tests/test_core.py::test_parallel_wild_bootstrap_shape_and_reproducibility",
    ]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


@pytest.mark.parametrize("python_version", ["3.10", "3.11", "3.12", "3.13"])
def test_minimum_runtime_constraints_cover_declared_dependencies(python_version):
    import ast
    import re
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = [Requirement(x) for x in ast.literal_eval(re.search(r'^dependencies = (\[.*\])$', text, re.M)[1])]
    env = dict(default_environment(), python_version=python_version)
    selected = {}
    for line in (ROOT / ".github/constraints-min.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate(env):
            assert requirement.name not in selected
            selected[requirement.name] = next(iter(requirement.specifier)).version
    assert set(selected) == {x.name for x in declared}
    for requirement in declared:
        assert selected[requirement.name] in requirement.specifier


def test_release_build_uses_sdist_to_wheel_default():
    text = (ROOT / "scripts/build_release.sh").read_text(encoding="utf-8")
    assert 'python -m build --outdir "$DIST"' in text
    assert 'python -m build --sdist --wheel' not in text
