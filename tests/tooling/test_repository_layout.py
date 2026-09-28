from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_root_is_project_entrypoint_not_document_dump():
    legacy_root_docs = {
        "ARCHITECTURE.md", "BENCHMARKS.md", "EXTERNAL_VALIDATION.md", "MIGRATION.md",
        "PERFORMANCE_RELEASE.md", "RELEASE_CHECKLIST.md", "RELEASE_MAINTENANCE.json",
        "RELEASE_MANIFEST.md", "TEST_STATUS.md", "UPSTREAM_REFERENCES.md", "VERSIONING.md",
        "bench_hdfe_exact_rank.json",
    }
    assert not any((ROOT / name).exists() for name in legacy_root_docs)
    for name in ("README.md", "CHANGELOG.md", "LICENSE", "NOTICE.md", "pyproject.toml"):
        assert (ROOT / name).is_file()
