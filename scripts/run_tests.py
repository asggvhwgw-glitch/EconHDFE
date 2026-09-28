"""Functional-test launcher; benchmark thread caps are NOT suite settings.

Run `python scripts/run_tests.py --suite core -- -q` for public behavior,
`--suite full` for all numerical/tooling gates, or --preflight for an environment
check only. Never skips multithread tests or silently raises an explicit cap.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SUITES = {
    'core': ('tests/contracts', 'tests/behavior'),
    'numerics': ('tests/numerics',),
    'tooling': ('tests/tooling',),
    'full': ('tests',),
}


def functional_environment(env: dict[str,str]) -> dict[str,str]:
    out=dict(env)
    raw=out.get('NUMBA_NUM_THREADS','4')
    try:
        count=int(raw)
    except ValueError as exc:
        raise ValueError('NUMBA_NUM_THREADS must be an integer >=3 for the full functional suite') from exc
    if count<3:
        raise ValueError('Full functional tests require NUMBA_NUM_THREADS>=3; a benchmark cap of 1 is not supported. Unset it or explicitly set 4; no tests were skipped.')
    out['NUMBA_NUM_THREADS']=str(count)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        out.setdefault(key,'1')
    out.setdefault('PYTHONUTF8','1')
    return out


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--suite', choices=tuple(SUITES), help='Explicit test layer; default pytest discovery is the core suite')
    p.add_argument('pytest_args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    try:
        env=functional_environment(dict(os.environ))
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps({k:env[k] for k in ('NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},sort_keys=True),flush=True)
    if not a.preflight:
        args=a.pytest_args
        if args[:1]==['--']:
            args=args[1:]
        raise SystemExit(subprocess.call([sys.executable,'-m','pytest',*(SUITES[a.suite] if a.suite else ()),*(args or ['-q'])],
                                       cwd=Path(__file__).resolve().parents[1],env=env))


if __name__=='__main__':
    main()
