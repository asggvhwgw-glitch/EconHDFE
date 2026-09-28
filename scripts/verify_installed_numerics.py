"""Run independent numerical suites on an installed wheel, outside its source.

This intentionally reuses HOST test/runtime dependencies. It tests packaging
and import provenance, NOT dependency resolution or clean-environment support.
Use verify_clean_install.py separately for that contract.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TEST_FILES = (
    'numerics/test_estimator_oracles.py',
    'numerics/test_inference_boundaries.py',
    'numerics/test_exact_oracles.py',
    'numerics/test_nonlinear_oracles.py',
    'numerics/test_resource_boundaries.py',
    'numerics/test_native_rank_budget.py',
    'numerics/test_stable_linalg.py',
    'contracts/test_public_workflows.py',
    'behavior/test_sample_semantics.py',
)


def resolve_extra_test(root: Path, name: str) -> str:
    """Resolve a unique basename without allowing path traversal or symlinks."""
    if (Path(name).name != name or '/' in name or '\\' in name
            or not name.startswith('test_') or not name.endswith('.py')):
        raise ValueError('--extra-test requires an existing test basename')
    tests = (root/'tests').resolve()
    candidates = [p for p in tests.rglob(name)
                  if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(tests)]
    if len(candidates) != 1:
        raise ValueError('--extra-test requires a unique existing test basename')
    return candidates[0].relative_to(tests).as_posix()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--junitxml',type=Path)
    parser.add_argument('--extra-test', action='append', default=[], help='Additional test basename under repository tests/')
    args=parser.parse_args();wheel=args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix!='.whl':parser.error('a built wheel is required')
    root=Path(__file__).resolve().parents[1]
    try:
        extra_tests = [resolve_extra_test(root, name) for name in args.extra_test]
    except ValueError as exc:
        parser.error(str(exc))
    with tempfile.TemporaryDirectory(prefix='econhdfe-installed-tests-') as td:
        temp=Path(td);site=temp/'site';tests=temp/'tests';tests.mkdir()
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',
                        '--target',str(site),str(wheel)],check=True,cwd=temp)
        for filename in dict.fromkeys(('conftest.py', *TEST_FILES, *extra_tests)):
            target = tests/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/'tests'/filename, target)
        (temp/'pytest.ini').write_text('[pytest]\n',encoding='utf-8')
        env=os.environ.copy();env['PYTHONPATH']=str(site);env['PYTHONNOUSERSITE']='1'
        env['ECONHDFE_AUDIT_MEASUREMENTS']=str(temp/'independent-measurements.json')
        code=("import econhdfe,pyreghdfe,pathlib; "
              f"target=pathlib.Path({str(site)!r}).resolve(); "
              "assert all(target in pathlib.Path(m.__file__).resolve().parents for m in (econhdfe,pyreghdfe)); "
              "print('wheel import provenance:',econhdfe.__file__)")
        subprocess.run([sys.executable,'-c',code],env=env,cwd=temp,check=True)
        command=[sys.executable,'-m','pytest','-q',str(tests)]
        if args.junitxml:
            xml=args.junitxml.resolve();xml.parent.mkdir(parents=True,exist_ok=True)
            command+=['--junitxml',str(xml)]
        subprocess.run(command,env=env,cwd=temp,check=True)
        print('Installed-wheel numerical suites PASS (HOST dependencies, not clean installation)')


if __name__=='__main__':main()
