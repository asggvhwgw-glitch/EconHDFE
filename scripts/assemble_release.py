"""Assemble the reproducible econhdfe release bundle."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
ROOT_NAME = f"econhdfe-{VERSION}"
TOP_LEVEL_DOCS = ["README.md", "CHANGELOG.md", "NOTICE.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md", "TODO.md", "RELEASE_CLOSEOUT.md"]
BENCHMARK_EVIDENCE = [
    "benchmarks/hdfe/exact_rank.json",
    "benchmarks/hdfe/solver_v044_integration.json",
    "benchmarks/bench_10m_complex_eventstudy_v080.json",
    "benchmarks/ppml/optimized_round_two_way_1m.json",
    "benchmarks/ppml/optimized_round_two_way_5m.json",
    "benchmarks/ppml/optimized_round_two_way_10m.json",
    "benchmarks/ppml/optimized_round_gravity_144k.json",
    "benchmarks/ppml/optimized_round_hierarchy_240k.json",
    "benchmarks/ppml/optimized_round_vce_5m.json",
    "benchmarks/ppml/optimized_round_warm_start_300k.json",
    "benchmarks/ppml/optimized_round_kernel_stability_10m.json",
    "benchmarks/repeated/repeated_spec_v041.json",
    "benchmarks/ppml/v045_execution_fix_synthetic.json",
    "benchmarks/real_world/2026-09-12/README.md",
    "benchmarks/real_world/2026-09-12/benchmark_complete_table_20260912.md",
    "benchmarks/real_world/2026-09-12/mobility_ppml_speedup_diagnosis_20260912.md",
    "benchmarks/real_world/2026-09-12/provenance.json",
    "benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md",
    "benchmarks/planner/post_architecture/v050_vs_v04102_same_host.json",
    "benchmarks/effects/v060_pre_release/summary.json",
    "benchmarks/effects/v060_pre_release/effects_recovery.json",
]
TOP_LEVEL_EXCLUDES = {".git", ".pytest_cache", "build", "dist", "dist_arch", "release", ".venv", "venv"}
ANY_LEVEL_EXCLUDES = {"__pycache__"}


def excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] in TOP_LEVEL_EXCLUDES:
        return True
    if any(part in ANY_LEVEL_EXCLUDES or part.endswith(".egg-info") for part in rel.parts):
        return True
    return path.suffix in {".pyc", ".pyo", ".nbc", ".nbi", ".aux", ".log", ".out"} or path.name == ".coverage"


def write_source_zip(root: Path, out: Path) -> None:
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file() and not excluded(p, root):
                zf.write(p, (Path(ROOT_NAME) / p.relative_to(root)).as_posix())


def write_skill_zip(skill_root: Path, out: Path) -> None:
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(skill_root.rglob("*")):
            if p.is_file() and not excluded(p, skill_root):
                zf.write(p, (Path("econhdfe") / p.relative_to(skill_root)).as_posix())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_relative(root: Path, stage: Path, rel: str) -> None:
    src = root / rel
    if not src.exists():
        return
    dst = stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sdist", type=Path, default=None)
    args = ap.parse_args()
    root, wheel = args.root.resolve(), args.wheel.resolve()
    out = (args.out_dir or root / "dist").resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = out / f"econhdfe-v{VERSION}-source.zip"
    write_source_zip(root, source)
    skill_zip = out / f"econhdfe-skill-v{VERSION}.zip"
    write_skill_zip(root / "skills" / "econhdfe", skill_zip)

    with tempfile.TemporaryDirectory(prefix="econhdfe-release-") as td:
        stage = Path(td) / f"econhdfe-v{VERSION}-release-bundle"
        stage.mkdir()
        for rel in TOP_LEVEL_DOCS:
            copy_relative(root, stage, rel)
        shutil.copytree(root / "docs", stage / "docs")
        for rel in BENCHMARK_EVIDENCE:
            copy_relative(root, stage, rel)
        shutil.copy2(source, stage / source.name)
        shutil.copy2(wheel, stage / wheel.name)
        if args.sdist is not None:
            sdist = args.sdist.resolve()
            if not sdist.is_file():
                raise FileNotFoundError(sdist)
            shutil.copy2(sdist, stage / sdist.name)
        shutil.copy2(skill_zip, stage / skill_zip.name)

        hash_lines = []
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS.txt":
                hash_lines.append(f"{sha256(p)}  {p.relative_to(stage).as_posix()}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n")

        bundle = out / f"econhdfe-v{VERSION}-release-bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(stage).as_posix())
    print(source)
    print(skill_zip)
    print(bundle)


if __name__ == "__main__":
    main()
