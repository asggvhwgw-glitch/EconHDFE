"""Build, guard, and smoke-check a self-contained linearmodels 7.0 patch.

The installed package is read only. This module also exposes ``build_patch``
for the fresh-process overlay launcher. Only cov_cluster is replaced; every
other upstream function and its source text remain unchanged.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "7.0"
EXPECTED_NORMALIZED_SHA256 = "57e4f7ba7bd5ac3e4da42d7deaac5e543a54b00365cce2299b088a83c338fdcf"
RELATIVE_MODULE = "linearmodels/shared/covariance.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_patch(output_dir=None) -> dict:
    """Return artifact paths and hashes after validating a pinned baseline.

    Outputs are source text, a unified diff, the upstream license and a JSON
    manifest. Nothing is written to the installed linearmodels directory.
    Source hashing normalizes CRLF because wheel installations differ in line
    endings across operating systems; the raw installed-file hash is retained.
    """
    import linearmodels.shared.covariance as upstream
    import numpy as np

    version = importlib.metadata.version("linearmodels")
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected linearmodels {EXPECTED_VERSION}, found {version}")
    baseline_path = Path(upstream.__file__).resolve()
    original_bytes = baseline_path.read_bytes()
    original = original_bytes.decode("utf-8").replace("\r\n", "\n")
    original_hash = _sha(original.encode("utf-8"))
    if original_hash != EXPECTED_NORMALIZED_SHA256:
        raise RuntimeError("Installed covariance.py differs from the pinned linearmodels 7.0 source")

    baseline_tree = ast.parse(original)
    old = next(node for node in baseline_tree.body if isinstance(node, ast.FunctionDef) and node.name == "cov_cluster")
    implementation_path = ROOT / "fastsandwich" / "cluster.py"
    production = ast.parse(implementation_path.read_text(encoding="utf-8"))
    new = copy.deepcopy(next(node for node in production.body if isinstance(node, ast.FunctionDef) and node.name == "cluster_meat"))
    new.name = "cov_cluster"
    new.args = copy.deepcopy(old.args)
    new.returns = copy.deepcopy(old.returns)
    # Preserve upstream's public argument names and signature. All algorithm
    # imports are local, making the substituted function independently usable.
    aliases = ast.parse("x = z\ngroups = clusters\n").body
    # The upstream API has no method parameter. Keep the auto/reduceat branch
    # alone so its patch has neither an undocumented argument nor dead code.
    optimized_body = []
    for statement in new.body[1:]:
        if (isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare)
                and isinstance(statement.test.left, ast.Name)
                and statement.test.left.id == "method"):
            if isinstance(statement.test.ops[0], ast.Eq):
                optimized_body.extend(statement.orelse)
            continue
        optimized_body.append(statement)
    documentation = ast.get_docstring(old) + (
        "\n\nNotes\n-----\n"
        "Optimized grouped reduction with verified singleton and sorted-label shortcuts.\n"
        "Finite real scores and non-missing sortable labels are required; no demeaning\n"
        "is performed. Float64 reduction order may differ from earlier versions.\n"
    )
    new.body = [ast.Expr(value=ast.Constant(value=documentation))] + aliases + optimized_body
    replacement_function = ast.unparse(ast.fix_missing_locations(new)) + "\n"
    lines = original.splitlines(keepends=True)
    replacement = "".join(lines[:old.lineno - 1]) + replacement_function + "".join(lines[old.end_lineno:])
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), replacement.splitlines(keepends=True),
        fromfile="a/" + RELATIVE_MODULE, tofile="b/" + RELATIVE_MODULE,
    ))

    # Confirm AST identity of everything outside the chosen function.
    def unaffected(tree):
        return [ast.dump(node, include_attributes=False) for node in tree.body
                if not (isinstance(node, ast.FunctionDef) and node.name == "cov_cluster")]
    if unaffected(ast.parse(replacement)) != unaffected(baseline_tree):
        raise AssertionError("Patch changes functions outside cov_cluster")
    namespace = {"__name__": "_fastsandwich_patch_validation"}
    exec(compile(replacement, "<generated linearmodels covariance>", "exec"), namespace)
    rng = np.random.default_rng(16954)
    maximum_difference = 0.0
    for n, k in ((1, 1), (29, 4), (67, 8)):
        scores = rng.normal(size=(n, k))
        for labels in (rng.integers(0, 9, n), np.arange(n), np.zeros(n, dtype=int)):
            expected = upstream.cov_cluster(scores, labels)
            actual = namespace["cov_cluster"](scores, labels)
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
            maximum_difference = max(maximum_difference, float(np.max(np.abs(actual - expected))))
    if baseline_path.read_bytes() != original_bytes:
        raise AssertionError("Installed baseline was modified during patch generation")

    output = Path(output_dir).resolve() if output_dir else ROOT / ".cache" / "upstream-patch"
    output.mkdir(parents=True, exist_ok=True)
    patch_path = output / "linearmodels-7.0-cluster.patch"
    replacement_source = output / "linearmodels-7.0-covariance.py"
    patch_path.write_bytes(patch.encode("utf-8"))
    replacement_source.write_bytes(replacement.encode("utf-8"))
    distribution = importlib.metadata.distribution("linearmodels")
    license_entry = next(path for path in distribution.files or [] if str(path).endswith(".dist-info/licenses/LICENSE.md"))
    license_bytes = Path(distribution.locate_file(license_entry)).read_bytes()
    license_path = output / "LICENSE-linearmodels.txt"
    license_path.write_bytes(license_bytes)
    manifest = {
        "linearmodels_version": version,
        "relative_module": RELATIVE_MODULE,
        "baseline_path": str(baseline_path),
        "baseline_sha256": _sha(original_bytes),
        "baseline_normalized_sha256": original_hash,
        "baseline_source_url": "https://raw.githubusercontent.com/bashtage/linearmodels/v7.0/linearmodels/shared/covariance.py",
        "replacement_source": str(replacement_source),
        "replacement_sha256": _sha(replacement.encode("utf-8")),
        "implementation_source": str(implementation_path),
        "implementation_sha256": _sha(implementation_path.read_bytes()),
        "patch_path": str(patch_path),
        "patch_sha256": _sha(patch.encode("utf-8")),
        "upstream_license": "NCSA (University of Illinois/NCSA Open Source License)",
        "license_path": str(license_path),
        "license_sha256": _sha(license_bytes),
        "attribution": "Upstream linearmodels: Copyright (c) 2017 Kevin Sheppard. See LICENSE-linearmodels.txt. Only cov_cluster is replaced by the FastSandwich grouped-reduction implementation.",
        "smoke_comparisons": 9,
        "smoke_max_absolute_difference": maximum_difference,
    }
    (output / "linearmodels-7.0-cluster-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(build_patch(args.output_dir), indent=2))
