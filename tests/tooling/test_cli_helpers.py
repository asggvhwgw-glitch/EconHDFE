"""Guards for the follow-up installed-package and frozen-artifact workflow."""
from pathlib import Path
import importlib.util
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def test_both_source_distributions_are_built_after_freezing_repository_state():
    text=(ROOT/'scripts/build_release.sh').read_text()
    freeze=text.index('python scripts/compatibility.py snapshot')
    refresh=text.index('python scripts/generate_architecture_map.py\n')
    assert freeze<refresh<text.index('python -m pip wheel')
    assert refresh<text.index('python -m build')
    assert text.count('python scripts/compatibility.py snapshot')==1


def test_installed_numerics_helper_has_existing_independent_suites():
    path=ROOT/'scripts/verify_installed_numerics.py'
    spec=importlib.util.spec_from_file_location('installed_probe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert len(module.TEST_FILES)==len(set(module.TEST_FILES))
    assert {'numerics/test_estimator_oracles.py', 'numerics/test_nonlinear_oracles.py',
            'contracts/test_public_workflows.py', 'behavior/test_sample_semantics.py'} <= set(module.TEST_FILES)
    assert all((ROOT/'tests'/name).is_file() for name in module.TEST_FILES)
    assert 'HOST' in module.__doc__ and 'NOT' in module.__doc__


def test_installed_numerics_helper_rejects_missing_wheel():
    out=subprocess.run([sys.executable,str(ROOT/'scripts/verify_installed_numerics.py'),
                        '--wheel','/missing/econhdfe.whl'],capture_output=True,text=True)
    assert out.returncode!=0
    assert 'built wheel' in out.stderr


def test_bounded_benchmark_cli_can_show_help_without_importing_package():
    out=subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_release_local.py'),'--help'],capture_output=True,text=True)
    assert out.returncode==0 and '--source' in out.stdout
