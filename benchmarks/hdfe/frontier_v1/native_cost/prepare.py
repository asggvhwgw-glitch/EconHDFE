#!/usr/bin/env python3
"""Prepare the pinned research-only native cost build outside the repository.

Python 3.12+, git, the recorded Rust toolchain and maturin are required. This
script never installs a wheel or alters a production solver. Use a separate
virtual environment for the resulting +cost1 wheels.
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
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCES = {
    "within": {
        "revision": "fc44282cea6124f5f18bb142908605a656d35a7c",
        "url": "https://codeload.github.com/py-econometrics/within/tar.gz/fc44282cea6124f5f18bb142908605a656d35a7c",
        "sha256": "7f8568b97bff327028ae20b00b98e54457fcaed50ebcb6d94c4f7f5e4768c06b",
        "archive": "within.tar.gz",
    },
    "fereg": {
        "revision": "0fa59e3b539addbddaee8bd71acd031c497166a0",
        "url": "https://codeload.github.com/leoliu0/fereg/tar.gz/0fa59e3b539addbddaee8bd71acd031c497166a0",
        "sha256": "0e0987a857d87dccda5b8443cdbaf99e4dd77b08b72edecb0bcfc30ed5cab3fe",
        "archive": "fereg.tar.gz",
    },
    "adapter-source": {
        "revision": "23a0c12f04983514a64e7cc9b10145d1e34c584e",
        "url": "https://codeload.github.com/asggvhwgw-glitch/EconHDFE/tar.gz/23a0c12f04983514a64e7cc9b10145d1e34c584e",
        "sha256": "e7484912c27ad2dd1b0da1c524ae256f8e196ec657ef13b443b1d46e20790525",
        "archive": "adapter-source.tar.gz",
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(work: Path, archives: Path):
    work = work.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("work-dir must be empty; use a new directory to retain earlier builds")
    work.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        archive = archives / spec["archive"]
        if not archive.exists():
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "FE-frontier-cost/1"})
            temporary = archive.with_suffix(archive.suffix + ".partial")
            with urllib.request.urlopen(request, timeout=90) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if digest(temporary) != spec["sha256"]:
                raise ValueError(f"source archive digest mismatch: {name}")
            temporary.replace(archive)
        if digest(archive) != spec["sha256"]:
            raise ValueError(f"source archive digest mismatch: {name}")
        unpack = work / (name + "-unpack")
        unpack.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(unpack, filter="data")
        roots = list(unpack.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError(f"unexpected archive root: {name}")
        roots[0].rename(work / name)
        unpack.rmdir()
    for name, subdir in (("dual", "dual-absorber"), ("native-auto", "within-accel")):
        shutil.copytree(work / "adapter-source" / "native" / subdir, work / name)
    for name in ("within", "fereg", "dual", "native-auto"):
        patch = HERE / "patches" / (name + ".patch")
        # --check first prevents partially applying an unexpected source patch.
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=work / name, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=work / name, check=True)
    for name in ("dual", "native-auto"):
        shutil.copyfile(HERE / "locks" / (name + ".Cargo.lock"), work / name / "Cargo.lock")
    report = {
        "schema_version": 1,
        "purpose": "research cost diagnostics; no numerical policy changes",
        "sources": SOURCES,
        "patches_sha256": {p.name: digest(p) for p in sorted((HERE / "patches").glob("*.patch"))},
        "lockfiles_sha256": {p.name: digest(p) for p in sorted((HERE / "locks").glob("*"))},
        "source_files_sha256": {},
    }
    for directory in ("within", "fereg", "dual", "native-auto"):
        for path in sorted((work / directory).rglob("*")):
            if path.is_file() and (path.suffix in {".rs", ".py", ".toml"} or path.name == "Cargo.lock"):
                report["source_files_sha256"][str(path.relative_to(work))] = digest(path)
    (work / "prepared.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build(work: Path):
    work = work.resolve()
    versions = {}
    for command in ("rustc", "cargo"):
        versions[command] = subprocess.check_output([command, "--version"], text=True).strip()
    versions["maturin"] = subprocess.check_output(
        [sys.executable, "-m", "maturin", "--version"], text=True).strip()
    env = os.environ.copy()
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env.setdefault("CARGO_TARGET_DIR", str(work / "target"))
    wheelhouse = work / "wheelhouse"
    for name in ("native-auto", "dual"):
        subprocess.run([
            sys.executable, "-m", "maturin", "build", "--release", "--locked",
            "--manifest-path", str(work / name / "Cargo.toml"), "--out", str(wheelhouse),
        ], env=env, check=True)
    report = {
        "tools": versions, "python": sys.version,
        "wheels_sha256": {p.name: digest(p) for p in sorted(wheelhouse.glob("*.whl"))},
        "prepared_manifest_sha256": digest(work / "prepared.json"),
    }
    (work / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True, help="verified download cache")
    parser.add_argument("--build", action="store_true", help="also build; never install")
    args = parser.parse_args()
    report = prepare(args.work_dir, args.archives)
    print(json.dumps({"prepared": str(args.work_dir), "source_files": len(report["source_files_sha256"])}))
    if args.build:
        print(json.dumps(build(args.work_dir)))


if __name__ == "__main__":
    main()
