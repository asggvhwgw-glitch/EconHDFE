"""Target-directory smoke using HOST dependencies, not a clean virtualenv.

Use verify_clean_install.py for independent dependency-resolution validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("wheel", type=Path)
    args = p.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.exists():
        raise SystemExit(f"missing wheel: {wheel}")
    root = Path(__file__).resolve().parents[1]
    smoke = root / "scripts" / "smoke_install.py"
    skill_env = root / "skills" / "econhdfe" / "scripts" / "check_environment.py"
    skill_smoke = root / "skills" / "econhdfe" / "scripts" / "smoke_test.py"
    cluster_smoke = root / "scripts" / "smoke_cluster_inference.py"
    support_report_smoke = root / "scripts" / "smoke_support_reports.py"
    effects_smoke = root / "scripts" / "smoke_fe_recovery.py"
    with tempfile.TemporaryDirectory(prefix="econhdfe-wheel-") as td:
        td = Path(td)
        site = td / "site"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site), str(wheel)],
            check=True,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(site)
        code = (
            "import econhdfe, pathlib; "
            f"p=pathlib.Path(econhdfe.__file__).resolve(); target=pathlib.Path({str(site)!r}).resolve(); "
            "assert target in p.parents, (p, target); "
            "print('isolated import:', p)"
        )
        subprocess.run([sys.executable, "-c", code], cwd=td, env=env, check=True)
        subprocess.run([sys.executable, str(smoke)], cwd=td, env=env, check=True)
        subprocess.run([sys.executable, str(skill_env)], cwd=td, env=env, check=True)
        subprocess.run([sys.executable, str(skill_smoke)], cwd=td, env=env, check=True)
        subprocess.run([sys.executable, str(cluster_smoke)], cwd=td, env=env, check=True)
        subprocess.run([sys.executable, str(support_report_smoke)], cwd=td, env=env, check=True)
        subprocess.run([sys.executable, str(effects_smoke)], cwd=td, env=env, check=True)


if __name__ == "__main__":
    main()
