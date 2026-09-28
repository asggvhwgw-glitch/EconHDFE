"""Tests of gate logic using SYNTHETIC evidence; not real release acceptance."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os
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


@pytest.mark.parametrize('status',['not_run','blocked','failed'])
def test_candidate_can_retain_unpassed_but_formal_gate_cannot(fixture,status):
    m,root,data=fixture
    data['checks']['clean_install']['execution_status']=status
    data['checks']['clean_install']['reason']='No dependency access or an observed failure.'
    r=m.assess(data,root);assert r['valid'] and not r['release_ready']
    r=m.assess(data,root,mode='release');assert not r['valid'] and not r['release_ready']


@pytest.mark.parametrize('status',['reviewed','changed','not_applicable',True,[],None])
def test_review_status_never_counts_as_execution(fixture,status):
    m,root,data=fixture;data['checks']['local_regression']['execution_status']=status
    assert not m.assess(data,root)['valid']


def test_only_complete_bound_fixture_passes_formal_logic(fixture):
    m,root,data=fixture;out=passed(fixture)
    assert m.assess(data,root,mode='release',artifact_dir=out)['release_ready']
    data['checks']['support_matrix']['execution_status']='blocked'
    data['checks']['support_matrix']['reason']='Not actually run'
    assert not m.assess(data,root,mode='release',artifact_dir=out)['valid']


@pytest.mark.parametrize('key',['local_regression','isolated_build','support_matrix','reference_claims'])
def test_required_checks_cannot_be_deleted_or_disabled(fixture,key):
    m,root,data=fixture
    del data['checks'][key]
    assert not m.assess(data,root)['valid']
    data=m.new_manifest(root);data['checks'][key]['required']=False
    assert not m.assess(data,root,mode='release')['valid']


@pytest.mark.parametrize('mutation',['version','global_fingerprint','row_fingerprint','empty_evidence','evidence_hash',
                                     'environment','time','timezone','unknown_check','blank_reason'])
def test_malformed_stale_or_unsubstantiated_success_is_rejected(fixture,mutation):
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


@pytest.mark.parametrize('name',['../evidence.json','/etc/passwd','C:/evidence.json','a\\b.json','missing.json'])
def test_evidence_paths_are_confined(fixture,name):
    m,root,data=fixture;passed(fixture)
    data['checks']['local_regression']['evidence'][0]['path']=name
    assert not m.assess(data,root)['valid']


def test_actual_code_and_evidence_mutations_invalidate_record(fixture):
    m,root,data=fixture;passed(fixture)
    before=m.code_fingerprint(root)
    (root/'econhdfe/__init__.py').write_text('# changed runtime\n')
    assert m.code_fingerprint(root)!=before
    assert not m.assess(data,root)['valid']
    # Resetting the file creates no validation successes.
    reset=m.new_manifest(root)
    assert all(x['execution_status']=='not_run' for x in reset['checks'].values())


def test_docs_do_not_masquerade_as_code_hash_and_evidence_hash_is_checked(fixture):
    m,root,data=fixture;passed(fixture)
    (root/'README.md').write_text('Documentation is bound by final artifact hashes.')
    assert m.assess(data,root)['valid']
    (root/'evidence.json').write_text('changed evidence')
    assert not m.assess(data,root)['valid']


@pytest.mark.parametrize('issue',['commit','missing_file','bad_hash','missing_wheel','no_directory'])
def test_detached_final_identity_is_required(fixture,issue):
    m,root,data=fixture;out=passed(fixture)
    if issue=='commit':data['source_commit']=None
    elif issue=='missing_file':(out/data['artifacts'][0]['path']).unlink()
    elif issue=='bad_hash':data['artifacts'][0]['sha256']='0'*64
    elif issue=='missing_wheel':data['artifacts']=data['artifacts'][1:]
    else:out=None
    assert not m.assess(data,root,mode='release',artifact_dir=out)['valid']


def test_cli_missing_manifest_fails_closed(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'scripts/release_acceptance.py'),'check',
                      '--manifest',str(tmp_path/'missing.json')],capture_output=True,text=True)
    assert p.returncode==1 and not json.loads(p.stdout)['release_ready']


def test_formal_offline_request_fails_before_removing_output(tmp_path):
    # Script contract, including Windows where Bash need not be installed.
    text=(ROOT/'scripts/build_release.sh').read_text()
    assert text.index('Formal release refuses')<text.index('rm -rf')
    assert text.index('check --mode release')<text.index('rm -rf')
    assert 'ECONHDFE_EXECUTION_MANIFEST' in text


@pytest.mark.parametrize('cap',['1','2','0','-1','bad'])
def test_full_suite_thread_cap_rejected_without_skipping(cap):
    m=script('run_tests')
    with pytest.raises(ValueError,match='functional|Functional'):
        m.functional_environment({'NUMBA_NUM_THREADS':cap})


def test_functional_thread_defaults_do_not_change_callers_environment():
    m=script('run_tests');env={'OMP_NUM_THREADS':'2'}
    got=m.functional_environment(env)
    assert env=={'OMP_NUM_THREADS':'2'} and got['NUMBA_NUM_THREADS']=='4'
    assert got['OMP_NUM_THREADS']=='2'


@pytest.mark.parametrize('cap,code',[('1',2),('4',0)])
def test_thread_preflight_before_numba_import(cap,code):
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_tests.py'),'--preflight'],
                    env=os.environ|{'NUMBA_NUM_THREADS':cap},capture_output=True,text=True)
    assert p.returncode==code
    if code:
        assert 'no tests were skipped' in p.stderr


def test_installed_extra_test_rejects_escaping_basename(tmp_path):
    wheel=tmp_path/'fake.whl';wheel.write_bytes(b'not installed')
    p=subprocess.run([sys.executable,str(ROOT/'scripts/verify_installed_numerics.py'),
                      '--wheel',str(wheel),'--extra-test','../test_escape.py'],capture_output=True,text=True)
    assert p.returncode==2 and 'test basename' in p.stderr
