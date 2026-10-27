"""Tests for local release helpers (network/OS CI success is not mocked here)."""

import importlib.util
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


@pytest.mark.parametrize('mode', [
    'valid',
    'wrong',
])
def test_release_checksum_manifest_coverage(tmp_path,mode):
    m=script('verify_release');a=tmp_path/'a.txt';a.write_text('ok')
    line=f'{m.digest(a)}  a.txt\n'
    if mode=='unlisted':(tmp_path/'other.txt').write_text('not hashed')
    if mode=='duplicate':line*=2
    if mode=='unsafe':line=line.replace('a.txt','../a.txt')
    if mode=='wrong':line='0'*64+'  a.txt\n'
    (tmp_path/'SHA256SUMS.txt').write_text(line)
    if mode=='valid':assert m.verify_checksums(tmp_path)==1
    else:
        with pytest.raises(ValueError):m.verify_checksums(tmp_path)


def test_publication_workflow_uses_trusted_publishing_without_token_fallback():
    text = (ROOT / ".github/workflows/release-v0.6.3.yml").read_text(encoding="utf-8")
    assert "id-token: write" in text
    assert "pypa/gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e" in text
    assert "PYPI_API_TOKEN" not in text
    assert "password:" not in text
    assert "needs.github-release.outputs.wheel_sha256" in text
    assert "needs.github-release.outputs.sdist_sha256" in text


@pytest.mark.parametrize("packaged", [False, True])
def test_profile_hashes_extension_binary_not_python_wrapper(tmp_path, packaged):
    import hashlib
    from importlib.machinery import EXTENSION_SUFFIXES
    from pathlib import Path
    import runpy
    from types import SimpleNamespace
    harness = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                                "benchmarks/hdfe/ppml_end_to_end_profile.py"))
    binary = tmp_path / ("_econhdfe_within" + EXTENSION_SUFFIXES[0])
    binary.write_bytes(b"first native binary")
    module = SimpleNamespace(__file__=str(binary))
    if packaged:
        wrapper = tmp_path / "__init__.py"
        wrapper.write_text("from ._econhdfe_within import *\n")
        module = SimpleNamespace(__file__=str(wrapper), _econhdfe_within=module)
    identity = harness["native_identity"](module)
    assert identity["native_module_file"] == str(binary.resolve())
    assert identity["native_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    first_hash = identity["native_sha256"]
    binary.write_bytes(b"different native binary, same Python wrapper")
    assert harness["native_identity"](module)["native_sha256"] != first_hash
    if packaged:
        assert identity["native_sha256"] != identity["native_import_entrypoint_sha256"]


def test_profile_rejects_unresolved_native_binary_identity(tmp_path):
    from pathlib import Path
    import runpy
    from types import SimpleNamespace
    harness = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                                "benchmarks/hdfe/ppml_end_to_end_profile.py"))
    wrapper = tmp_path / "__init__.py"
    wrapper.write_text("# No verified extension module\n")
    with pytest.raises(RuntimeError, match="extension binary"):
        harness["native_identity"](SimpleNamespace(__file__=str(wrapper)))
