"""Install wheel + dependencies into a NEW venv, then smoke outside the source.

Unlike verify_wheel_install.py, this validates dependency resolution and uses no
host site-packages. Network/index failures are failures, never silently bypassed.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

SMOKES = (
    "scripts/smoke_install.py",
    "skills/econhdfe/scripts/check_environment.py",
    "skills/econhdfe/scripts/smoke_test.py",
    "scripts/smoke_cluster_inference.py",
    "scripts/smoke_support_reports.py",
    "scripts/smoke_fe_recovery.py",
)

def select_wheel(wheel: Path | None, dist: Path | None) -> Path:
    if wheel is not None:
        result = wheel.resolve()
        if not result.is_file() or result.suffix != ".whl":
            raise ValueError(f"missing or invalid wheel: {result}")
        return result
    choices = sorted((dist or Path("dist")).glob("econhdfe-*.whl"))
    if len(choices) != 1:
        raise ValueError(f"expected exactly one econhdfe wheel, found {len(choices)}")
    return choices[0].resolve()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wheel", type=Path)
    group.add_argument("--dist", type=Path)
    parser.add_argument("--find-links", type=Path)
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()
    wheel = select_wheel(args.wheel, args.dist)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="econhdfe-clean-") as temp:
        temp = Path(temp)
        envdir = temp / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(envdir)
        python = envdir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        install = [str(python), "-m", "pip", "install", str(wheel)]
        if args.find_links:
            install.extend(["--find-links", str(args.find_links.resolve())])
        if args.no_index:
            install.append("--no-index")
        subprocess.run(install, cwd=temp, env=env, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], cwd=temp, env=env, check=True)
        code = ("import econhdfe, pathlib, sys; "
                "prefix=pathlib.Path(sys.prefix).resolve(); "
                "assert sys.prefix != sys.base_prefix; "
                "assert prefix in pathlib.Path(econhdfe.__file__).resolve().parents; "
                "print('clean-venv package:', econhdfe.__file__)")
        subprocess.run([str(python), "-c", code], cwd=temp, env=env, check=True)
        for i, rel in enumerate(SMOKES):
            script = temp / f"smoke_{i}.py"
            shutil.copy2(root / rel, script)
            subprocess.run([str(python), str(script)], cwd=temp, env=env, check=True)
        print("clean dependency installation and all six smoke scripts: PASS")

if __name__ == "__main__":
    main()
