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
