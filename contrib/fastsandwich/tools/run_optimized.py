"""Run a script with an isolated, source-guarded linearmodels 7.0 overlay.

Usage: python tools/run_optimized.py examples/panel_policy.py [script arguments]
       python tools/run_optimized.py --prepare-only

The installed package is never edited. The overlay contains the complete
installed linearmodels package with only the validated shared covariance
module replaced by the self-contained implementation from the upstream patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_upstream_patch import build_patch


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_overlay():
    """Build/reuse a workspace-only overlay and return reproducibility metadata.

    Cache identity includes both baseline and replacement hashes. A changed
    implementation gets a distinct directory, avoiding stale bytecode and
    mutation of another concurrently running experiment's source tree.
    """
    manifest = build_patch()
    baseline_path = Path(manifest["baseline_path"]).resolve()
    replacement_path = Path(manifest["replacement_source"]).resolve()
    if file_sha256(baseline_path) != manifest["baseline_sha256"]:
        raise RuntimeError("Installed baseline changed while preparing the overlay")
    if file_sha256(replacement_path) != manifest["replacement_sha256"]:
        raise RuntimeError("Generated replacement does not match its manifest hash")
    package_source = baseline_path.parents[1]
    if package_source.name != "linearmodels" or baseline_path.name != "covariance.py":
        raise RuntimeError("Unexpected baseline package location")

    cache_base = (PROJECT_ROOT / ".cache" / "linearmodels-optimized").resolve()
    cache_key = hashlib.sha256((manifest["baseline_sha256"] +
                               manifest["replacement_sha256"]).encode("ascii")).hexdigest()[:20]
    overlay_root = (cache_base / cache_key).resolve()
    if not overlay_root.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("Overlay must remain inside this project workspace")
    package_target = overlay_root / "linearmodels"
    replacement_target = package_target / "shared" / "covariance.py"
    metadata_path = overlay_root / "overlay.json"
    valid_cache = (metadata_path.is_file() and replacement_target.is_file()
                   and file_sha256(replacement_target) == manifest["replacement_sha256"])
    if not valid_cache:
        overlay_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package_source, package_target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        shutil.copyfile(replacement_path, replacement_target)
        if file_sha256(replacement_target) != manifest["replacement_sha256"]:
            raise RuntimeError("Overlay copy failed source-hash verification")
    if file_sha256(baseline_path) != manifest["baseline_sha256"]:
        raise RuntimeError("Installed baseline changed during overlay preparation")
    # Refresh absolute paths even for a valid cached copy: the project might
    # have been relocated since its overlay was first prepared.
    metadata = dict(manifest, overlay_root=str(overlay_root),
                    package_root=str(package_target), replacement_target=str(replacement_target))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def optimized_environment(overlay_root, environment=None):
    """Return a subprocess environment selecting the overlay before site-packages."""
    environment = dict(os.environ if environment is None else environment)
    search_path = [str(Path(overlay_root).resolve()), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        search_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(search_path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare the isolated package and print its JSON metadata")
    parser.add_argument("script", nargs="?", help="Python script to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.prepare_only and not args.script:
        parser.error("provide a script or --prepare-only")
    if args.prepare_only and args.script:
        parser.error("--prepare-only does not accept a script")
    metadata = prepare_overlay()
    if args.prepare_only:
        print(json.dumps(metadata, indent=2))
        return 0
    script = Path(args.script).resolve()
    if not script.is_file():
        parser.error(f"script does not exist: {script}")
    result = subprocess.run([sys.executable, str(script), *args.script_args],
                            env=optimized_environment(metadata["overlay_root"]), check=False)
    baseline_path = metadata["baseline_path"]
    if file_sha256(baseline_path) != metadata["baseline_sha256"]:
        raise RuntimeError("Installed source changed during script execution")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
