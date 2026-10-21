"""Tests of gate logic using SYNTHETIC evidence; not real release acceptance."""

from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest


ROOT=Path(__file__).resolve().parents[2]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.fixture
def fixture(tmp_path):
    m=script('release_acceptance')
    (tmp_path/'pyproject.toml').write_text('[project]\nversion = "0.6.3"\n')
    (tmp_path/'econhdfe').mkdir();(tmp_path/'econhdfe/__init__.py').write_text('# synthetic runtime\n')
    return m,tmp_path,m.new_manifest(tmp_path)


def passed(fixture):
    m,root,data=fixture
    log=root/'evidence.json';log.write_text('{"synthetic_test_fixture":true}')
    for row in data['checks'].values():
        row.update(execution_status='passed',reason='',environment={'scope':'SYNTHETIC unit fixture'},
                   executed_at='2026-09-14T08:00:00+00:00',code_fingerprint_sha256=data['code_fingerprint_sha256'],
                   evidence=[{'path':log.name,'sha256':m.digest(log)}])
    data['source_commit']='a'*40
    artifacts=root/'out';artifacts.mkdir()
    for name in ['econhdfe-0.6.3-py3-none-any.whl','econhdfe-0.6.3.tar.gz',
                 'econhdfe-v0.6.3-source.zip','econhdfe-v0.6.3-release-bundle.zip']:
        p=artifacts/name;p.write_bytes(b'synthetic artifact content (not a package)')
        data['artifacts'].append({'path':name,'sha256':m.digest(p)})
    return artifacts


def test_only_complete_bound_fixture_passes_formal_logic(fixture):
    m,root,data=fixture;out=passed(fixture)
    assert m.assess(data,root,mode='release',artifact_dir=out)['release_ready']
    data['checks']['support_matrix']['execution_status']='blocked'
    data['checks']['support_matrix']['reason']='Not actually run'
    assert not m.assess(data,root,mode='release',artifact_dir=out)['valid']


def test_malformed_stale_or_unsubstantiated_success_is_rejected(fixture):
    mutation = 'evidence_hash'
    m,root,data=fixture;out=passed(fixture);row=data['checks']['local_regression']
    if mutation=='version':data['release_version']='0.6.2'
    elif mutation=='global_fingerprint':data['code_fingerprint_sha256']='0'*64
    elif mutation=='row_fingerprint':row['code_fingerprint_sha256']='0'*64
    elif mutation=='empty_evidence':row['evidence']=[]
    elif mutation=='evidence_hash':row['evidence'][0]['sha256']='0'*64
    elif mutation=='environment':row['environment']={}
    elif mutation=='time':row['executed_at']='not a time'
    elif mutation=='timezone':row['executed_at']='2026-09-14T08:00:00'
    elif mutation=='unknown_check':data['checks']['pretend']={}
    else:row.update(execution_status='blocked',reason=' ')
    assert not m.assess(data,root,mode='release',artifact_dir=out)['valid']


def test_actual_code_and_evidence_mutations_invalidate_record(fixture):
    m,root,data=fixture;passed(fixture)
    before=m.code_fingerprint(root)
    (root/'econhdfe/__init__.py').write_text('# changed runtime\n')
    assert m.code_fingerprint(root)!=before
    assert not m.assess(data,root)['valid']
    # Resetting the file creates no validation successes.
    reset=m.new_manifest(root)
    assert all(x['execution_status']=='not_run' for x in reset['checks'].values())


def test_detached_final_identity_is_required(fixture):
    issue = 'bad_hash'
    m,root,data=fixture;out=passed(fixture)
    if issue=='commit':data['source_commit']=None
    elif issue=='missing_file':(out/data['artifacts'][0]['path']).unlink()
    elif issue=='bad_hash':data['artifacts'][0]['sha256']='0'*64
    elif issue=='missing_wheel':data['artifacts']=data['artifacts'][1:]
    else:out=None
    assert not m.assess(data,root,mode='release',artifact_dir=out)['valid']


def test_formal_offline_request_fails_before_removing_output(tmp_path):
    # Script contract, including Windows where Bash need not be installed.
    text=(ROOT/'scripts/build_release.sh').read_text()
    assert text.index('Formal release refuses')<text.index('rm -rf')
    assert text.index('check --mode release')<text.index('rm -rf')
    assert 'ECONHDFE_EXECUTION_MANIFEST' in text
