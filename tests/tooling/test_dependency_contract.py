"""Architecture means actual dependencies, not words occurring in source comments."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = {
    'data': ('models', 'iv', 'hdfe'),
    'compute': ('hdfe', 'iv', 'models', 'data'),
    'planner': ('data', 'compute', 'hdfe', 'iv', 'models'),
    'hdfe': ('iv', 'models'),
    'iv': ('models', 'hdfe'),
    'resampling': ('models', 'hdfe', 'iv'),
    'models.linear_iv': ('models.ppml', 'models.ppml_iv'),
    'models.ppml': ('models.linear_iv', 'models.ppml_iv'),
}


def imported_names(text, module, is_package=False):
    package = module.split('.') if is_package else module.split('.')[:-1]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            yield from (item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package)-node.level+1] if node.level else []
            if node.module:
                prefix += node.module.split('.')
            yield '.'.join(prefix)
            for alias in node.names:
                yield '.'.join([*prefix, alias.name])


def test_actual_imports_respect_layer_ownership():
    violations = []
    for layer, forbidden in RULES.items():
        for path in (ROOT/'econhdfe'/layer.replace('.', '/')).rglob('*.py'):
            module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
            is_package = path.name == '__init__.py'
            if is_package: module = module.removesuffix('.__init__')
            for imported in imported_names(path.read_text(encoding='utf-8'), module, is_package):
                for target in forbidden:
                    prefix = 'econhdfe.'+target
                    if imported == prefix or imported.startswith(prefix+'.'):
                        violations.append(f'{module} -> {imported}')
    assert not violations, '\n'.join(violations)


def test_dependency_check_resolves_absolute_relative_and_from_imports():
    text = ('# ..models is only a comment\nimport econhdfe.models.ols as m\n'
            'from .. import hdfe\nfrom ..models import ols\nfrom .kernels import dot\n')
    names = set(imported_names(text, 'econhdfe.compute.linalg'))
    assert {'econhdfe.models.ols', 'econhdfe.hdfe', 'econhdfe.compute.kernels.dot'} <= names
