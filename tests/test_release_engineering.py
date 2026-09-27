"""Tests for local release helpers (network/OS CI success is not mocked here)."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_clean_install_wheel_selection_requires_one_file(tmp_path):
    m=script('verify_clean_install')
    with pytest.raises(ValueError):m.select_wheel(None,tmp_path)
    a=tmp_path/'econhdfe-1-py3-none-any.whl';a.write_bytes(b'a')
    assert m.select_wheel(None,tmp_path)==a.resolve()
    (tmp_path/'econhdfe-2-py3-none-any.whl').write_bytes(b'b')
    with pytest.raises(ValueError):m.select_wheel(None,tmp_path)

@pytest.mark.parametrize('mode',['valid','unlisted','duplicate','unsafe','wrong'])
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


def test_closeout_document_is_identical_in_outer_bundle_and_source(tmp_path, monkeypatch):
    import io
    import sys
    import zipfile

    # Synthetic archive fixture tests packaging only, not numerical acceptance.
    m = script('assemble_release')
    root = tmp_path / 'source'
    (root / 'docs').mkdir(parents=True)
    (root / 'skills/econhdfe').mkdir(parents=True)
    payload = '# Release closeout\nCandidate only.\n'
    (root / 'RELEASE_CLOSEOUT.md').write_bytes(payload.encode('utf-8'))
    wheel = tmp_path / 'fixture.whl'
    wheel.write_bytes(b'synthetic packaging fixture')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['assemble_release', '--root', str(root),
                                    '--wheel', str(wheel), '--out-dir', str(out)])
    m.main()
    with zipfile.ZipFile(out / f'econhdfe-v{m.VERSION}-release-bundle.zip') as bundle:
        assert bundle.read('RELEASE_CLOSEOUT.md') == payload.encode()
        raw_source = bundle.read(f'econhdfe-v{m.VERSION}-source.zip')
        with zipfile.ZipFile(io.BytesIO(raw_source)) as source:
            assert source.read(f'econhdfe-{m.VERSION}/RELEASE_CLOSEOUT.md') == payload.encode()
        assert 'RELEASE_CLOSEOUT.md' in bundle.read('SHA256SUMS.txt').decode()


@pytest.mark.parametrize("mode", ["original", "public", "tampered", "missing", "extra", "unsafe", "malformed", "no_fallback"])
def test_benchmark_provenance_preserves_original_and_checks_public_copy(tmp_path, mode):
    import hashlib
    import json

    m = script("verify_release")
    old = b"original private label\n"
    public = b"Project-A\n"
    record = {"source_files": {"record.md": hashlib.sha256(old).hexdigest()}}
    payload = old if mode == "original" else public
    if mode != "original":
        hashes = {"record.md": hashlib.sha256(public).hexdigest()}
        record["public_release_redaction"] = {"public_copy_sha256": hashes}
        if mode == "missing": hashes.clear()
        if mode == "extra": hashes["unlisted.md"] = hashlib.sha256(public).hexdigest()
        if mode == "malformed": hashes["record.md"] = "not-a-hash"
        if mode == "unsafe":
            record["source_files"] = {"../record.md": record["source_files"]["record.md"]}
            record["public_release_redaction"]["public_copy_sha256"] = {"../record.md": hashlib.sha256(public).hexdigest()}
        if mode == "no_fallback": payload = old
        if mode == "tampered": payload += b"changed evidence"
    (tmp_path / "record.md").write_bytes(payload)
    before = json.dumps(record)
    (tmp_path / "provenance.json").write_text(before, encoding="utf-8")
    if mode in {"original", "public"}:
        m.verify_benchmark_provenance(tmp_path)
    else:
        with pytest.raises(ValueError): m.verify_benchmark_provenance(tmp_path)
    assert (tmp_path / "provenance.json").read_text(encoding="utf-8") == before


def test_repository_public_benchmark_hashes_match():
    script("verify_release").verify_benchmark_provenance(ROOT / "benchmarks/real_world/2026-09-12")


def test_publication_workflow_is_manual_and_version_agnostic():
    text = (ROOT / ".github/workflows/release-v0.6.3.yml").read_text(encoding="utf-8")
    assert "name: Publish release" in text
    assert "workflow_dispatch:" in text
    assert "\n  push:" not in text
    assert "2782d93f901ec1eed28ac0f963834701e0a44b09" not in text
    assert "36264493861" not in text
    assert "source_run:" in text and "source_commit:" in text and "version:" in text
    assert "run_conclusion" in text and "run_sha" in text
    assert ".github/workflows/ci.yml" in text
    assert 'gh run download "$SOURCE_RUN"' in text
    assert "sha256sum -c" in text
    assert "Release $TAG already exists; refusing to overwrite an immutable release." in text


def test_publication_workflow_uses_trusted_publishing_without_token_fallback():
    text = (ROOT / ".github/workflows/release-v0.6.3.yml").read_text(encoding="utf-8")
    assert "id-token: write" in text
    assert "pypa/gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e" in text
    assert "PYPI_API_TOKEN" not in text
    assert "password:" not in text
    assert "needs.github-release.outputs.wheel_sha256" in text
    assert "needs.github-release.outputs.sdist_sha256" in text
