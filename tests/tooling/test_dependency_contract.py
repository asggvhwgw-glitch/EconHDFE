"""Architecture means actual dependencies, not words occurring in source comments."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


# Specification/configuration vocabulary importable from every layer. These
# modules may import only one another.
FOUNDATION = ('errors', 'config', 'factorvars', 'design_specs', 'hdfe.specs')

# Lower layers: the only package units each may import besides itself and
# FOUNDATION.  Everything else in the package is an upper layer for them.
LOWER = {
    'planner': (),
    'compute': ('planner',),
    'resampling': ('planner',),
    'iv': ('compute',),
    'hdfe': ('compute', 'planner'),
    'frontend': (),
    'data': ('frontend', 'compute', 'planner'),
    # Frozen fitted-state schema shared by FE recovery, design and results.
    'prediction': (),
    'effects': ('compute', 'hdfe', 'prediction'),
    'inference': ('compute', 'hdfe', 'planner', 'resampling'),
    'templates': (),
}

# Estimator composition, result/report surfaces and orchestration.  They may
# import any lower layer; sibling estimator families stay independent.
UPPER = (
    'econhdfe', 'api', 'collinearity', 'design', 'design_structure',
    'pipeline', 'postestimation', 'prediction_api', 'reporting', 'results',
    'sessions', 'support_reports',
    'models', 'models.linear_iv', 'models.ppml', 'models.ppml_iv',
)
UPPER_FORBIDDEN = {
    'models.linear_iv': ('models.ppml', 'models.ppml_iv'),
    'models.ppml': ('models.linear_iv', 'models.ppml_iv'),
}

# Registered exceptions: (importing module, imported unit) -> reason.
ALLOWED_EXCEPTIONS = {
    ('econhdfe.data.persistent', 'results'):
        'session result cache encodes/decodes RegressionResult and DofInfo; '
        'moving the codec needs the sessions.py store construction to change',
    ('econhdfe.data.persistent', 'prediction'):
        'session result cache encodes/decodes frozen prediction-state types',
    ('econhdfe.planner.report', 'econhdfe'):
        'lazy __version__ fallback when distribution metadata is unavailable',
}

UNITS = sorted((*FOUNDATION, *LOWER, *UPPER), key=lambda u: -u.count('.'))


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


def unit_of(name):
    """Most specific architectural unit owning an ``econhdfe`` dotted name."""
    if name == 'econhdfe':
        return 'econhdfe'
    if not name.startswith('econhdfe.'):
        return None
    rest = name.removeprefix('econhdfe.')
    for unit in UNITS:
        if unit != 'econhdfe' and (rest == unit or rest.startswith(unit+'.')):
            return unit
    return rest.split('.')[0]


def package_modules():
    for path in sorted((ROOT/'econhdfe').rglob('*.py')):
        module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
        is_package = path.name == '__init__.py'
        if is_package: module = module.removesuffix('.__init__')
        yield module, path, is_package


def test_every_module_belongs_to_a_declared_layer():
    unclassified = [module for module, _, _ in package_modules()
                    if unit_of(module) not in UNITS]
    assert not unclassified, '\n'.join(unclassified)


def test_actual_imports_respect_layer_ownership():
    violations, used_exceptions = [], set()
    for module, path, is_package in package_modules():
        source = unit_of(module)
        if source in FOUNDATION:
            allowed = set(FOUNDATION)
        elif source in LOWER:
            allowed = {source, *FOUNDATION, *LOWER[source]}
        else:
            allowed = None
        forbidden = UPPER_FORBIDDEN.get(source, ())
        for imported in imported_names(path.read_text(encoding='utf-8'), module, is_package):
            target = unit_of(imported)
            if target is None or target == source:
                continue
            if (allowed is None or target in allowed) and target not in forbidden:
                continue
            if (module, target) in ALLOWED_EXCEPTIONS:
                used_exceptions.add((module, target))
                continue
            violations.append(f'{module} -> {imported}')
    assert not violations, '\n'.join(sorted(set(violations)))
    assert used_exceptions == set(ALLOWED_EXCEPTIONS), 'stale exceptions: ' + repr(
        sorted(set(ALLOWED_EXCEPTIONS) - used_exceptions))
