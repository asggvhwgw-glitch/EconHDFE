"""Execution-evidence gate, separate from the maintenance REVIEW checklist.

A candidate may honestly retain blocked/not_run checks. A formal release cannot.
This verifies local records/hashes, not their authors' honesty or a signature.
Final artifact hashes belong in a detached execution manifest to avoid a bundle
hashing itself. No command here publishes anything or turns reviewed into passed.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELATIVE_MANIFEST = Path('docs/release/execution.json')
SCHEMA_VERSION = 1
FINGERPRINT_SCOPE = 'runtime-validation-v1'
# Required checks cannot be disabled by editing a "required" field in JSON.
REQUIRED = (
    'local_regression', 'installed_numerics', 'isolated_build', 'clean_install',
    'support_matrix', 'dependency_floor', 'reference_claims', 'final_artifacts',
)
STATUSES = {'not_run', 'blocked', 'failed', 'passed'}
CODE_ROOTS = ('econhdfe', 'scripts', 'tests', 'skills', '.github', 'compatibility')
ROOT_FILES = ('pyproject.toml', 'MANIFEST.in')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_version(root: Path) -> str:
    m = re.search(r'^version\s*=\s*"([^\"]+)"', (root/'pyproject.toml').read_text(encoding='utf-8'), re.M)
    if not m:
        raise ValueError('missing project version')
    return m.group(1)


def code_fingerprint(root: Path) -> str:
    """Portable digest of runtime, validation, CI and packaging inputs.

    Not a git commit or a full-archive hash. Excludes docs/evidence, bytecode,
    JIT caches and build outputs. Those are verified as final artifacts instead.
    """
    files = {root/p for p in ROOT_FILES if (root/p).is_file()}
    for name in CODE_ROOTS:
        base = root/name
        if not base.is_dir():
            continue
        for p in base.rglob('*'):
            rel = p.relative_to(root)
            if any(x in {'__pycache__', '.pytest_cache'} or x.endswith('.egg-info') for x in rel.parts):
                continue
            if p.suffix in {'.pyc','.pyo','.nbc','.nbi','.log'}:
                continue
            if p.is_symlink():
                raise ValueError(f'code fingerprint rejects symlink: {rel}')
            if p.is_file():
                files.add(p)
    h = hashlib.sha256()
    for p in sorted(files, key=lambda p:p.relative_to(root).as_posix()):
        h.update((p.relative_to(root).as_posix()+'\0'+digest(p)+'\n').encode('utf-8'))
    return h.hexdigest()


def new_manifest(root: Path, version: str | None = None) -> dict[str, Any]:
    return {
        'schema_version':SCHEMA_VERSION,
        'release_version':version or project_version(root),
        'fingerprint_scope':FINGERPRINT_SCOPE,
        'code_fingerprint_sha256':code_fingerprint(root),
        'source_commit':None,
        'checks':{name:{'execution_status':'not_run','reason':'Not executed for this release.',
                         'environment':{},'executed_at':None,'code_fingerprint_sha256':None,
                         'evidence':[]} for name in REQUIRED},
        'artifacts':[],
    }


def _safe_path(base: Path, name: Any) -> Path:
    if not isinstance(name,str) or not name or '\\' in name or ':' in name:
        raise ValueError('unsafe evidence/artifact path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name:
        raise ValueError('unsafe evidence/artifact path')
    target=(base/name).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file():
        raise ValueError('missing or escaping evidence/artifact path')
    return target


def _hashed_files(items: Any, base: Path, label: str) -> list[str]:
    errors=[]
    if not isinstance(items,list) or not items:
        return [f'{label}: non-empty hashed files required']
    seen=set()
    for item in items:
        if not isinstance(item,dict):
            errors.append(f'{label}: malformed file record');continue
        name=item.get('path'); sha=item.get('sha256')
        try:
            p=_safe_path(base,name)
            if name in seen:
                raise ValueError('duplicate path')
            seen.add(name)
            if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha) or digest(p)!=sha:
                raise ValueError('SHA256 mismatch')
        except (OSError,ValueError) as exc:
            errors.append(f'{label}: {name!r}: {exc}')
    return errors


def assess(data: Any, root: Path, *, mode: str = 'candidate',
           artifact_dir: Path | None = None) -> dict[str, Any]:
    if mode not in {'candidate','release'}:
        raise ValueError('mode must be candidate or release')
    if not isinstance(data,dict):
        return {'valid':False,'release_ready':False,'mode':mode,'problems':['manifest must be an object'],'unpassed':list(REQUIRED)}
    errors=[]; unpassed=[]
    if data.get('schema_version')!=SCHEMA_VERSION:
        errors.append('unsupported schema_version')
    if data.get('release_version')!=project_version(root):
        errors.append('release_version mismatch')
    current=code_fingerprint(root)
    if data.get('fingerprint_scope')!=FINGERPRINT_SCOPE or data.get('code_fingerprint_sha256')!=current:
        errors.append('stale or wrong code fingerprint: rerun affected validation; do not relabel old evidence')
    checks=data.get('checks')
    if not isinstance(checks,dict):
        checks={};errors.append('checks must be an object')
    if set(checks)!=set(REQUIRED):
        errors.append('required check set mismatch (missing/unknown checks)')
    for name in REQUIRED:
        row=checks.get(name)
        if not isinstance(row,dict):
            errors.append(f'{name}: missing check');unpassed.append(name);continue
        status=row.get('execution_status')
        if not isinstance(status,str) or status not in STATUSES:
            errors.append(f'{name}: invalid execution_status {status!r}')
        if status!='passed':
            unpassed.append(name)
            if not isinstance(row.get('reason'),str) or not row['reason'].strip():
                errors.append(f'{name}: unpassed status needs an explicit reason')
            continue
        if row.get('code_fingerprint_sha256')!=current:
            errors.append(f'{name}: evidence is for another code snapshot')
        if not isinstance(row.get('environment'),dict) or not row['environment']:
            errors.append(f'{name}: passed requires an actual environment')
        try:
            stamp=datetime.fromisoformat(row.get('executed_at',''))
            if stamp.tzinfo is None:
                raise ValueError('timezone missing')
        except (TypeError,ValueError):
            errors.append(f'{name}: passed requires an ISO timestamp with timezone')
        errors.extend(_hashed_files(row.get('evidence'),root,name))
    # A real release must also bind a real commit and the final detached artifacts.
    identity_errors=[]
    if not isinstance(data.get('source_commit'),str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',data['source_commit']):
        identity_errors.append('source_commit missing; a local code digest is not a Git commit')
    if artifact_dir is None:
        identity_errors.append('artifact directory required for final release')
    else:
        identity_errors.extend(_hashed_files(data.get('artifacts'),artifact_dir,'artifacts'))
        names=[x.get('path','') for x in data.get('artifacts',[]) if isinstance(x,dict)] if isinstance(data.get('artifacts'),list) else []
        v=project_version(root)
        if not any(n.endswith('.whl') and Path(n).name.startswith('econhdfe-'+v+'-') for n in names):
            identity_errors.append('current wheel missing from artifact records')
        for required_name in (f'econhdfe-{v}.tar.gz',f'econhdfe-v{v}-source.zip',f'econhdfe-v{v}-release-bundle.zip'):
            if required_name not in {Path(n).name for n in names}:
                identity_errors.append(f'final artifact missing: {required_name}')
    release_ready=not errors and not unpassed and not identity_errors
    if mode=='release':
        errors.extend(f'{name}: execution is not passed' for name in unpassed)
        errors.extend(identity_errors)
    return {'valid':not errors,'release_ready':release_ready,'mode':mode,
            'code_fingerprint_sha256':current,'unpassed':unpassed,
            'identity_pending':identity_errors,'problems':errors}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','reset','fingerprint'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--mode',choices=('candidate','release'),default='candidate')
    p.add_argument('--artifact-dir',type=Path)
    p.add_argument('--json-out',type=Path)
    a=p.parse_args();root=a.root.resolve();path=a.manifest or root/RELATIVE_MANIFEST
    try:
        if a.command=='reset':
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(new_manifest(root),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            print('Execution evidence RESET to not_run; no successful result carried forward.')
            return
        if a.command=='fingerprint':
            print(code_fingerprint(root));return
        data=json.loads(path.read_text(encoding='utf-8'))
        result=assess(data,root,mode=a.mode,artifact_dir=a.artifact_dir)
    except (OSError,ValueError) as exc:
        result={'valid':False,'release_ready':False,'mode':a.mode,'problems':[str(exc)]}
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True);a.json_out.write_text(text+'\n',encoding='utf-8')
    print(text)
    if not result['valid']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
