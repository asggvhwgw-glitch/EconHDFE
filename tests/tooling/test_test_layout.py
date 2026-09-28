"""Test-selection contracts: the daily subset must never replace release gates."""
import ast
import importlib.util
from pathlib import Path
import pytest
from scripts.run_tests import SUITES
from scripts.verify_installed_numerics import TEST_FILES, resolve_extra_test

ROOT = Path(__file__).resolve().parents[2]


def test_full_suite_contains_all_layers_and_ci_requests_it():
    assert SUITES['core'] == ('tests/contracts', 'tests/behavior')
    assert SUITES['full'] == ('tests',)
    assert all((ROOT/path).is_dir() for paths in SUITES.values() for path in paths)
    assert not list((ROOT/'tests').glob('test_v*.py'))
    ci = (ROOT/'.github/workflows/ci.yml').read_text(encoding='utf-8')
    release = (ROOT/'scripts/build_release.sh').read_text(encoding='utf-8')
    assert 'scripts/run_tests.py --suite full -- -q' in ci
    assert release.count('scripts/run_tests.py --suite full -- -q') == 2
    assert 'branches: [main]' in ci and '  pull_request:' in ci


def test_test_function_names_are_unique_within_each_module():
    for path in (ROOT/'tests').rglob('test_*.py'):
        functions = [n.name for n in ast.parse(path.read_text(encoding='utf-8')).body
                     if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]
        assert len(functions) == len(set(functions)), path


def test_installed_layer_copies_current_contracts_without_source_dependencies():
    assert all((ROOT/'tests'/path).is_file() for path in TEST_FILES)
    assert (ROOT/'tests/conftest.py').is_file()
    assert resolve_extra_test(ROOT, 'test_weighted_reductions.py') == 'numerics/test_weighted_reductions.py'


@pytest.mark.parametrize('name', ['../test_escape.py', 'a\\test_escape.py', '/test_escape.py', 'test_missing.py'])
def test_installed_extra_selection_fails_closed(tmp_path, name):
    (tmp_path/'tests').mkdir()
    with pytest.raises(ValueError, match='test basename'):
        resolve_extra_test(tmp_path, name)


def test_installed_extra_selection_rejects_ambiguous_basename(tmp_path):
    for layer in ('contracts', 'numerics'):
        folder = tmp_path/'tests'/layer
        folder.mkdir(parents=True)
        (folder/'test_same.py').write_text('# fixture\n')
    with pytest.raises(ValueError, match='unique'):
        resolve_extra_test(tmp_path, 'test_same.py')
