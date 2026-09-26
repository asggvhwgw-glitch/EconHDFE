"""Verify an econhdfe release bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
MAINTENANCE = Path("docs/release/maintenance.json")
INNOVATION_REGISTRY = Path("docs/technical/innovation-registry.json")
INNOVATION_AUDIT = Path("docs/technical/innovation-audit.md")
ERROR_REPORT_TEMPLATE = Path("docs/development/ERROR_REPORT_TEMPLATE.md")
PLANNER_REPORT_TEMPLATE = Path("docs/development/PLANNER_REPORT_TEMPLATE.md")
ARCH_MAP = Path("docs/development/architecture-map")
ARCH_FILES = tuple(ARCH_MAP / name for name in ("architecture.json", "architecture.md", "architecture.html"))
REAL_WORLD_BENCHMARK_FILES = (
    Path("benchmarks/real_world/2026-09-12/README.md"),
    Path("benchmarks/real_world/2026-09-12/benchmark_complete_table_20260912.md"),
    Path("benchmarks/real_world/2026-09-12/mobility_ppml_speedup_diagnosis_20260912.md"),
    Path("benchmarks/real_world/2026-09-12/provenance.json"),
    Path("benchmarks/ppml/v045_execution_fix_synthetic.json"),
    Path("benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md"),
    Path("benchmarks/effects/v060_pre_release/summary.json"),
    Path("benchmarks/effects/v060_pre_release/effects_recovery.json"),
)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_checksums(root: Path) -> int:
    listed: set[str] = set()
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        if not line.strip():
            continue
        try:
            expected, name = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError("malformed SHA256 entry") from exc
        name = name.strip()
        rel = Path(name)
        if (rel.is_absolute() or ".." in rel.parts or "\\" in name
                or ":" in name or name in listed or name == "SHA256SUMS.txt"):
            raise ValueError(f"duplicate/unsafe SHA256 member: {name}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"malformed SHA256 digest for {name}")
        listed.add(name)
        target = root / rel
        if not target.is_file() or digest(target) != expected.lower():
            raise ValueError(f"SHA256 mismatch for {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.relative_to(root).as_posix() != "SHA256SUMS.txt"}
    if listed != actual:
        raise ValueError(f"SHA256 coverage mismatch: unlisted={actual-listed}, missing={listed-actual}")
    return len(listed)


def verify_benchmark_provenance(directory: Path) -> None:
    """Verify the declared public copy, or the original when no redaction exists.

    Original hashes stay immutable provenance. A declared redaction must have a
    complete hash map; never fall back to originals or skip a mismatched file.
    """
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    original = provenance.get("source_files")
    if not isinstance(original, dict) or not original:
        raise ValueError("benchmark source hash map missing")
    expected = original
    if "public_release_redaction" in provenance:
        redaction = provenance["public_release_redaction"]
        expected = redaction.get("public_copy_sha256") if isinstance(redaction, dict) else None
    if not isinstance(expected, dict) or set(expected) != set(original):
        raise ValueError("benchmark public hash coverage mismatch")
    for name in original:
        if (not isinstance(name, str) or Path(name).name != name
                or name in {"", ".", ".."} or "\\" in name or ":" in name):
            raise ValueError("unsafe benchmark filename")
        for value in (original[name], expected[name]):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"malformed benchmark hash: {name}")
        archived = directory / name
        if not archived.is_file() or digest(archived) != expected[name]:
            raise ValueError(f"real-world benchmark provenance mismatch: {name}")


def py_digests(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*.py"))}


def tree_digests(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def registered_manuscripts(root: Path) -> tuple[Path, ...]:
    data = json.loads((root / INNOVATION_REGISTRY).read_text())
    out: list[Path] = []
    for item in data.get("innovations", []):
        manuscript = item.get("manuscript", {})
        out.extend((Path(manuscript["tex"]), Path(manuscript["pdf"])))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    with tempfile.TemporaryDirectory(prefix="econhdfe-verify-") as td_raw:
        td = Path(td_raw)
        b = td / "bundle"
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(b)
        skill_name = f"econhdfe-skill-v{VERSION}.zip"
        required = {
            Path("README.md"), Path("RELEASE_CLOSEOUT.md"), Path("TODO.md"), Path("docs/release/execution.json"), Path("CHANGELOG.md"), Path("LICENSE"), Path("NOTICE.md"),
            Path("docs/README.md"), Path("docs/development/architecture.md"), ERROR_REPORT_TEMPLATE, PLANNER_REPORT_TEMPLATE, *ARCH_FILES,
            Path("docs/release/versioning.md"), MAINTENANCE, INNOVATION_REGISTRY, INNOVATION_AUDIT,
            *registered_manuscripts(ROOT), *REAL_WORLD_BENCHMARK_FILES,
            Path("SHA256SUMS.txt"), Path(skill_name),
            Path(f"econhdfe-v{VERSION}-source.zip"), Path(f"econhdfe-{VERSION}-py3-none-any.whl"),
        }
        missing = [rel.as_posix() for rel in required if not (b / rel).is_file()]
        if missing:
            raise SystemExit("bundle missing: " + ", ".join(sorted(missing)))

        verify_checksums(b)

        src_dir, whl_dir, skill_dir = td / "source", td / "wheel", td / "skill"
        with zipfile.ZipFile(b / f"econhdfe-v{VERSION}-source.zip") as zf:
            zf.extractall(src_dir)
        with zipfile.ZipFile(b / f"econhdfe-{VERSION}-py3-none-any.whl") as zf:
            zf.extractall(whl_dir)
        with zipfile.ZipFile(b / skill_name) as zf:
            zf.extractall(skill_dir)
        src = src_dir / f"econhdfe-{VERSION}"
        for rel in (Path("TODO.md"), Path("RELEASE_CLOSEOUT.md"), Path("docs/release/execution.json")):
            if not (src / rel).is_file() or digest(src / rel) != digest(b / rel):
                raise SystemExit(f"bundle/source execution-roadmap mismatch: {rel}")
        # Candidate validity is intentionally weaker than formal authorization.
        # The extracted source must carry verifiable execution records, including
        # any truthful unpassed states. This does not turn them into successes.
        subprocess.run([sys.executable, str(src / "scripts/release_acceptance.py"),
                        "check", "--mode", "candidate"], check=True, cwd=src)

        for pkg in ("econhdfe", "pyreghdfe"):
            if py_digests(src / pkg) != py_digests(whl_dir / pkg):
                raise SystemExit(f"source/wheel mismatch: {pkg}")

        # Technical-innovation registry and manuscripts are source/release audit artifacts, never wheel payload.
        for rel in (INNOVATION_REGISTRY, INNOVATION_AUDIT):
            if not (src / rel).is_file():
                raise SystemExit(f"source missing technical-innovation artifact: {rel.as_posix()}")
            if digest(src / rel) != digest(b / rel):
                raise SystemExit(f"bundle/source technical-innovation artifact mismatch: {rel.as_posix()}")
        for rel in registered_manuscripts(src):
            if not (src / rel).is_file():
                raise SystemExit(f"source missing registered technical manuscript: {rel.as_posix()}")
            if not (b / rel).is_file():
                raise SystemExit(f"bundle missing registered technical manuscript: {rel.as_posix()}")
            if digest(src / rel) != digest(b / rel):
                raise SystemExit(f"bundle/source technical manuscript mismatch: {rel.as_posix()}")
        for rel in ARCH_FILES:
            if not (src / rel).is_file():
                raise SystemExit(f"source missing architecture map: {rel.as_posix()}")
            if digest(src / rel) != digest(b / rel):
                raise SystemExit(f"bundle/source architecture-map mismatch: {rel.as_posix()}")

        for rel in REAL_WORLD_BENCHMARK_FILES:
            if not (src / rel).is_file():
                raise SystemExit(f"source missing benchmark evidence: {rel.as_posix()}")
            if digest(src / rel) != digest(b / rel):
                raise SystemExit(f"bundle/source benchmark evidence mismatch: {rel.as_posix()}")

        verify_benchmark_provenance(src / "benchmarks/real_world/2026-09-12")

        wheel_names = {p.relative_to(whl_dir).as_posix() for p in whl_dir.rglob("*") if p.is_file()}
        if any(name.endswith((".tex", ".pdf")) for name in wheel_names):
            raise SystemExit("wheel must not ship technical manuscript .tex/.pdf files")

        # Installed report helpers must ship byte-identical canonical templates.
        wheel_error_template = whl_dir / "econhdfe" / "templates" / "error-report-template.md"
        wheel_benchmark_template = whl_dir / "econhdfe" / "templates" / "benchmark-report-template.md"
        if not wheel_error_template.is_file() or digest(wheel_error_template) != digest(src / ERROR_REPORT_TEMPLATE):
            raise SystemExit("wheel error-report template missing or differs from repository canonical")
        if not wheel_benchmark_template.is_file() or digest(wheel_benchmark_template) != digest(src / "benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md"):
            raise SystemExit("wheel benchmark template missing or differs from repository canonical")
        entry_points = next(whl_dir.glob("econhdfe-*.dist-info/entry_points.txt"), None)
        if entry_points is None or "econhdfe-report = econhdfe.support_reports:main" not in entry_points.read_text():
            raise SystemExit("wheel missing econhdfe-report console entry point")

        canonical_skill = src / "skills" / "econhdfe"
        packaged_skill = skill_dir / "econhdfe"
        if tree_digests(canonical_skill) != tree_digests(packaged_skill):
            raise SystemExit("standalone skill package differs from canonical source skill")
        repo_benchmark_template = src / "benchmarks" / "real_world" / "BENCHMARK_REPORT_TEMPLATE.md"
        skill_benchmark_template = packaged_skill / "references" / "benchmark-report-template.md"
        if digest(repo_benchmark_template) != digest(skill_benchmark_template):
            raise SystemExit("benchmark report template differs between repository and standalone skill")

        repo_error_template = src / ERROR_REPORT_TEMPLATE
        skill_error_template = packaged_skill / "references" / "error-report-template.md"
        if digest(repo_error_template) != digest(skill_error_template):
            raise SystemExit("error report template differs between repository and standalone skill")
        repo_planner_template = src / PLANNER_REPORT_TEMPLATE
        skill_planner_template = packaged_skill / "references" / "planner-report-template.md"
        if digest(repo_planner_template) != digest(skill_planner_template):
            raise SystemExit("planner report template differs between repository and standalone skill")
        planner_text = repo_planner_template.read_text()
        for required_text in (
            "parameter-only", "Automatic thread calibration", "Selected automatic threads",
            "include no raw or synthetic observations", "Real-data performance benchmarking is a separate explicit-consent workflow",
        ):
            if required_text not in planner_text:
                raise SystemExit(f"privacy-minimized planner report marker missing: {required_text}")
        error_text = repo_error_template.read_text()
        for required_text in (
            "parameter-only",
            "Anonymous specification fingerprint",
            "Recommended aliases",
            "Stack signature as `module:function` names only",
            "must **not request** raw data",
            "Performance benchmarking is a separate workflow",
        ):
            if required_text not in error_text:
                raise SystemExit(f"privacy-minimized error report marker missing: {required_text}")
        for forbidden_text in (
            "## 9. Minimal reproduction",
            "Exact econhdfe call",
            "<paste traceback>",
            "## 12. Attached artifacts",
            "may use the supplied original/raw dataset",
            "may use the supplied original replication scripts",
        ):
            if forbidden_text in error_text:
                raise SystemExit(f"legacy privacy-sensitive error report section present: {forbidden_text}")

        for rel in (
            "SKILL.md", "agents/openai.yaml", "references/installation.md",
            "references/configuration.md", "references/empirical-research.md",
            "references/advanced-validation.md", "references/developer-guide.md",
            "references/architecture-visualization.md", "references/benchmarking.md",
            "references/benchmark-report-template.md", "references/error-report-template.md",
            "references/planner-feedback.md", "references/planner-report-template.md",
            "scripts/check_environment.py", "scripts/smoke_test.py", "scripts/planner_report.py", "scripts/architecture_map.py",
        ):
            if not (packaged_skill / rel).is_file():
                raise SystemExit(f"skill package missing: {rel}")

        bundle_maintenance = json.loads((b / MAINTENANCE).read_text())
        source_maintenance = json.loads((src / MAINTENANCE).read_text())
        if bundle_maintenance != source_maintenance:
            raise SystemExit("release maintenance manifest differs between bundle and source")
        if bundle_maintenance.get("release_version") != VERSION:
            raise SystemExit("release maintenance version mismatch")
        for name, item in bundle_maintenance.get("checks", {}).items():
            if item.get("status") not in {"reviewed", "changed", "not_applicable"}:
                raise SystemExit(f"release maintenance check incomplete: {name}")
            if not str(item.get("evidence", "")).strip():
                raise SystemExit(f"release maintenance evidence missing: {name}")

        baseline = src / "compatibility" / "baselines" / f"econhdfe-{VERSION}.json"
        if not baseline.exists():
            raise SystemExit("current release public-contract snapshot missing from source")
        contract = json.loads(baseline.read_text())
        if contract.get("package_version") != VERSION:
            raise SystemExit("current public-contract snapshot version mismatch")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(src)
        subprocess.run(
            [sys.executable, str(src / "scripts" / "compatibility.py"), "check", "--version", VERSION],
            cwd=src, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, str(src / "scripts" / "skill_validation.py")],
            cwd=src, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, str(src / "scripts" / "technical_innovation.py"), "--root", str(src)],
            cwd=src, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, str(src / "scripts" / "generate_architecture_map.py"), "--check"],
            cwd=src, env=env, check=True,
        )
        fresh_contract = td / "fresh-contract.json"
        subprocess.run(
            [sys.executable, str(src / "scripts" / "compatibility.py"), "snapshot", "--version", VERSION, "--output", str(fresh_contract)],
            cwd=src, env=env, check=True,
        )
        if json.loads(fresh_contract.read_text()) != contract:
            raise SystemExit("frozen public-contract snapshot does not match released source")

        metadata = next(whl_dir.glob("econhdfe-*.dist-info/METADATA")).read_text()
        if f"Version: {VERSION}" not in metadata:
            raise SystemExit("wheel version mismatch")
        init = (src / "econhdfe" / "__init__.py").read_text()
        if not re.search(rf'__version__\s*=\s*["\']{re.escape(VERSION)}["\']', init):
            raise SystemExit("source version mismatch")
    print(f"release verification PASS: {bundle}")


if __name__ == "__main__":
    main()
