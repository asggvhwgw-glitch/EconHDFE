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
    (root / 'RELEASE_CLOSEOUT.md').write_text(payload, encoding='utf-8')
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
